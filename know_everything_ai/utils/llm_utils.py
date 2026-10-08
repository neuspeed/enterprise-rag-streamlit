"""Base class for components that call a language model.

Every LLM in this product is reached through an OpenAI-compatible endpoint, so
one client covers vLLM, Ollama, LM Studio, Together and OpenAI itself. Buyers
supply the URL, key and model name; nothing is hardcoded.

The httpx pool is injected rather than created per component. Building an
``AsyncOpenAI`` per message leaked a connection pool per message — and per file,
because each loader constructed its own parser — until the worker ran out of file
descriptors and stopped accepting work without any error in the logs.

A component may have more than one target. ``ENRICHER_API_URL`` is the fast paid
provider; when it rate-limits or times out, the same call is retried against
``ENRICHER_FALLBACK_API_URL``, which is typically a local Ollama. Falling over
inside the component rather than in the queue dispatcher means the buyer still
gets a result on the first attempt instead of after a redelivery that repeats
every model call.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, NamedTuple

import httpx
import openai
import structlog

from know_everything_ai.utils.retry import RETRYABLE_ERRORS

log = structlog.get_logger("llm_utils")

DEFAULT_TIMEOUT = 600
DEFAULT_MAX_RETRIES = 3


class ModelTarget(NamedTuple):
    """One endpoint and model to try, in order.

    ``model`` lives here rather than on the component because a fallback is
    rarely the same weights as the primary: the point of a local Ollama is that
    it can run a smaller, cheaper model, and reusing the primary's name against
    a different provider would 404.
    """

    label: str
    api_url: str
    api_key: str
    model: str
    timeout: int
    max_tokens: int | None = None


def build_target(
    *,
    api_url: str,
    api_key: str,
    model: str,
    timeout: int,
    max_tokens: int | None = None,
    label: str = "primary",
) -> ModelTarget | None:
    """Build a target, or ``None`` when the URL is unset.

    Returning ``None`` for an empty URL lets callers pass environment values
    straight through: an unconfigured fallback simply disappears instead of
    needing a branch at every call site.
    """
    if not api_url:
        return None
    return ModelTarget(
        label=label,
        api_url=api_url,
        api_key=api_key,
        model=model,
        timeout=timeout,
        max_tokens=max_tokens,
    )


class LLMDriven:
    """Shared base for every component that calls a model.

    Deliberately not an ``abc.ABC``. Subclasses expose unrelated entry points —
    ``create_knowledge_canvas``, ``classify``, ``parse`` — so there is no single
    method to declare abstract, and inheriting from ABC without one advertises a
    contract that does not exist while still allowing direct instantiation.
    """

    #: Token counts accumulated over this component's lifetime. Read by the
    #: pipeline to build ``ReturnPayload.cost``; the previous code logged
    #: ``response.usage`` and threw it away, so every job reported ``cost={}``.
    def __init__(
        self,
        api_url: str,
        api_key: str,
        model: str,
        system_prompt: str,
        timeout: int = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        max_tokens: int | None = None,
        http_client: httpx.AsyncClient | None = None,
        usage_name: str | None = None,
        fallback: ModelTarget | None = None,
    ) -> None:
        self.api_url = api_url
        self.api_key = api_key
        self.model = model
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        #: Key this component is billed under in the per-message cost report.
        self.usage_name = usage_name or type(self).__name__
        # An injected pool is owned by the caller and must outlive this object.
        self._owns_client = http_client is None
        self._http_client = http_client
        self._max_retries = max_retries
        primary = ModelTarget(
            label="primary",
            api_url=api_url,
            api_key=api_key,
            model=model,
            timeout=timeout,
            max_tokens=max_tokens,
        )
        self.targets: list[ModelTarget] = (
            [primary, fallback] if fallback is not None else [primary]
        )
        #: Clients are built on first use and reused. The primary is created here
        #: so ``self.openai_client`` stays the object tests and callers already
        #: hold a reference to.
        self._clients: dict[str, openai.AsyncOpenAI] = {}
        self.openai_client: openai.AsyncOpenAI = self._client_for(primary)
        #: Index of the target that answered last. Kept between calls in one
        #: message so a dead primary is not re-diagnosed on every chunk, and reset
        #: per message so a recovered provider is used again for the next job.
        self._preferred_index = 0
        self.usage: dict[str, int] = {
            "calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    def _record_usage(self, response: Any) -> None:
        """Accumulate token counts from one completion response."""
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        self.usage["calls"] += 1
        for src, dst in (
            ("prompt_tokens", "prompt_tokens"),
            ("completion_tokens", "completion_tokens"),
            ("total_tokens", "total_tokens"),
        ):
            value = getattr(usage, src, None)
            if isinstance(value, int):
                self.usage[dst] += value

    def reset_usage(self) -> None:
        for key in self.usage:
            self.usage[key] = 0

    def _client_for(self, target: ModelTarget) -> openai.AsyncOpenAI:
        """The client for ``target``, built once and reused."""
        client = self._clients.get(target.label)
        if client is None:
            # Built as a dict rather than splatted into the constructor: the
            # ``http_client`` argument only exists when a pool was injected, and
            # mixing the two shapes is what forces a blanket type-ignore.
            params: dict[str, Any] = {
                "api_key": target.api_key or "not-needed",
                "base_url": target.api_url or None,
                "timeout": target.timeout,
                "max_retries": self._max_retries,
            }
            if self._http_client is not None:
                params["http_client"] = self._http_client
            client = openai.AsyncOpenAI(**params)
            self._clients[target.label] = client
        return client

    async def complete(
        self,
        *,
        messages: Any,
        max_tokens: int | None = None,
        **extra: Any,
    ) -> Any:
        """Run one completion against the first target that answers.

        Only failures another provider could plausibly survive are worth trying
        elsewhere: a rate limit, a timeout, a dropped connection or a 5xx. A 400
        or a 401 is a bug in this deployment's configuration and would be
        reproduced identically on the next target, so it propagates at once
        instead of doubling the latency before the same error surfaces.

        Usage is recorded here, on the response that actually succeeded, so a
        job that failed over is still billed for what was spent reaching the
        answer.
        """
        last_error: BaseException | None = None
        for index, target in self._ordered_targets():
            client = self._client_for(target)
            params: dict[str, Any] = {"model": target.model, "messages": messages, **extra}
            limit = max_tokens or target.max_tokens
            if limit:
                params["max_tokens"] = limit
            try:
                response = await client.chat.completions.create(**params)
            except RETRYABLE_ERRORS as exc:
                last_error = exc
                log.warning(
                    "llm_target_failed",
                    component=self.usage_name,
                    target=target.label,
                    url=target.api_url,
                    model=target.model,
                    error=f"{type(exc).__name__}: {exc}",
                    another_target_pending=index < len(self.targets) - 1,
                )
                continue
            self._preferred_index = index
            self._record_usage(response)
            return response

        assert last_error is not None  # targets is never empty
        raise last_error

    async def complete_stream(
        self,
        *,
        messages: Any,
        max_tokens: int | None = None,
        **extra: Any,
    ) -> AsyncIterator[str]:
        """Stream one completion, failing over between targets.

        Yields the text of each delta as it arrives. Failover semantics differ
        from ``complete`` in one deliberate way: if the chosen target dies
        *mid-stream*, the fallback starts where the text stopped rather than
        restarting the question. The caller has already committed to the tokens
        it yielded — restarting would either duplicate what it showed or drop
        it, and for an answer stream a ragged seam is worth more than silence.
        A failure before the first token behaves exactly like ``complete``:
        next target, nothing leaked.

        Usage is recorded from the final chunk, which OpenAI SDK backends
        populate only when ``stream_options.include_usage`` is set.
        """
        last_error: BaseException | None = None
        for index, target in self._ordered_targets():
            client = self._client_for(target)
            params: dict[str, Any] = {
                "model": target.model,
                "messages": messages,
                "stream": True,
                "stream_options": {"include_usage": True},
                **extra,
            }
            limit = max_tokens or target.max_tokens
            if limit:
                params["max_tokens"] = limit
            try:
                stream = await client.chat.completions.create(**params)
                async for chunk in stream:
                    # The final chunk carries usage; earlier ones have none and
                    # _record_usage returns immediately without counting a call.
                    self._record_usage(chunk)
                    if chunk.choices:
                        delta = chunk.choices[0].delta.content
                        if delta:
                            yield delta
                self._preferred_index = index
                return
            except RETRYABLE_ERRORS as exc:
                last_error = exc
                log.warning(
                    "llm_stream_target_failed",
                    component=self.usage_name,
                    target=target.label,
                    url=target.api_url,
                    model=target.model,
                    error=f"{type(exc).__name__}: {exc}",
                    another_target_pending=index < len(self.targets) - 1,
                )
                continue

        assert last_error is not None  # targets is never empty
        raise last_error

    def _ordered_targets(self) -> list[tuple[int, ModelTarget]]:
        """Targets as ``(index, target)``, most recently successful one first.

        Reordering rather than dropping keeps ``index`` meaningful, since it is
        what :attr:`_preferred_index` stores.
        """
        indexed = list(enumerate(self.targets))
        preferred = self._preferred_index
        return (
            [pair for pair in indexed if pair[0] == preferred]
            + [pair for pair in indexed if pair[0] < preferred]
            + [pair for pair in indexed if pair[0] > preferred]
        )

    def reset_targets(self) -> None:
        """Forget which target answered last, so the next message starts over.

        Called per message: a provider that recovers mid-run should be picked
        up again by the next job rather than stay deprioritised until restart.
        """
        self._preferred_index = 0

    async def aclose(self) -> None:
        # AsyncOpenAI.close() closes whichever pool it holds — including an
        # injected one. Closing a shared pool would break every other component
        # using it, so an injected client is left to its owner.
        if not self._owns_client:
            return
        for client in self._clients.values():
            await client.close()

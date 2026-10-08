"""The answer agent: retrieval plus one grounded, cited completion.

Only failures another provider could plausibly survive are retried at the
transport level; the model is then always handed the retrieved fragments, so
there is no path where it answers from unreferenced priors.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Sequence
from typing import Any

import httpx
import structlog

from know_everything_ai.settings import Settings
from know_everything_ai.stores.base import SearchHit
from know_everything_ai.utils.llm_utils import LLMDriven, build_target

log = structlog.get_logger("answer_agent")

#: A fragment is already near the embedding window (< 512 tokens); this cap is
#: rendering safety so a single oversized chunk cannot crowd the whole answer.
MAX_FRAGMENT_CHARS = 2000

CANNED_NO_HITS = "В базе знаний пока нет данных по этому вопросу."


class AnswerAgent(LLMDriven):
    """Streams a grounded answer from the dedicated ``ANSWER_*`` model role."""

    def __init__(
        self, settings: Settings, *, http_client: httpx.AsyncClient | None = None
    ) -> None:
        super().__init__(
            api_url=settings.ANSWER_API_URL,
            api_key=settings.ANSWER_API_KEY,
            model=settings.ANSWER_MODEL_NAME,
            system_prompt=settings.ANSWER_SYSTEM_PROMPT,
            timeout=settings.ANSWER_TIMEOUT,
            max_tokens=settings.ANSWER_MAX_TOKENS,
            http_client=http_client,
            usage_name="answer",
            fallback=build_target(
                api_url=settings.ANSWER_FALLBACK_API_URL,
                api_key=settings.ANSWER_FALLBACK_API_KEY,
                model=settings.ANSWER_FALLBACK_MODEL_NAME,
                timeout=settings.ANSWER_TIMEOUT,
                max_tokens=settings.ANSWER_MAX_TOKENS,
                label="fallback",
            ),
        )
        self.temperature = settings.ANSWER_TEMPERATURE
        self.max_fragments = settings.ANSWER_MAX_FRAGMENTS

    @property
    def configured(self) -> bool:
        """Retrieval works without ``ANSWER_*``; answering does not."""
        return bool(self.api_url and self.model)

    async def stream_answer(
        self, question: str, fragments: Sequence[SearchHit]
    ) -> AsyncIterator[str]:
        """Stream the answer for one question over the given fragments."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": build_answer_user_message(question, fragments),
            },
        ]
        async for delta in self.complete_stream(
            messages=messages, temperature=self.temperature
        ):
            yield delta


def _clip(content: str) -> str:
    content = (content or "").strip()
    if len(content) > MAX_FRAGMENT_CHARS:
        return content[:MAX_FRAGMENT_CHARS].rstrip() + "…"
    return content


def _fragment_label(fragment: SearchHit, index: int) -> str:
    """``[1] (раздел: X, источник: Y)`` — the citation anchor the prompt asks
    the model to place at every borrowed fact."""
    bits = []
    if fragment.title:
        bits.append(f"раздел: {fragment.title}")
    if fragment.source:
        bits.append(f"источник: {fragment.source}")
    label = f"[{index}]"
    if bits:
        label += f" ({', '.join(bits)})"
    return label


def build_answer_user_message(question: str, fragments: Sequence[SearchHit]) -> str:
    """Render the question and its numbered, cited fragments for the model.

    Numbering is preserved verbatim into the answer, so a ``[3]`` the model
    emits maps 1:1 back to the fragment shown in the ``sources`` event.
    """
    parts = ["Фрагменты базы знаний:", ""]
    for index, fragment in enumerate(fragments, start=1):
        parts.append(_fragment_label(fragment, index))
        parts.append(_clip(fragment.content))
        parts.append("")
    parts.append(f"Вопрос: {question}")
    return "\n".join(parts).strip()


def usage_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    """Token counts spent by the call that just finished.

    ``LLMDriven.usage`` accumulates for the lifetime of the component, so per
    request the delta between a snapshot taken before the stream and the current
    counters is what an answer actually cost.
    """
    return {key: after[key] - before.get(key, 0) for key in after}


async def stream_answer_events(
    *,
    agent: AnswerAgent,
    question: str,
    hits: Sequence[SearchHit],
    row: dict[str, Any],
    floor: float,
) -> AsyncIterator[tuple[str, dict[str, Any]]]:
    """Post-retrieval half of a chat turn, as framework-free events.

    Yields ``(kind, payload)`` pairs — ``sources``, ``delta``, ``done`` or
    ``error`` — so the SSE router and the Streamlit tab render the same stream.
    Retrieval runs before the stream starts (a 404/409 is an HTTP status, not an
    event), so this function never sees a missing knowledge base.
    """
    yield "sources", {
        "hits": [
            {
                "chunk_id": hit.chunk_id,
                "content": _clip(hit.content),
                "score": round(hit.score, 4),
                "category": hit.category,
                "source": hit.source,
                "title": hit.title,
            }
            for hit in hits
        ],
        "kb_external_id": row.get("kb_external_id"),
        "kb_type": row.get("kb_type"),
        "branch": row.get("branch"),
        "model": agent.model,
        "floor": floor,
    }

    if not hits:
        yield "delta", {"text": CANNED_NO_HITS}
        yield "done", {
            "text_len": len(CANNED_NO_HITS),
            "hit_count": 0,
            "model": agent.model,
            "usage": {},
            "took_ms": 0,
        }
        return

    start = time.monotonic()
    before = dict(agent.usage)
    answer: list[str] = []
    try:
        async for chunk in agent.stream_answer(question, list(hits)):
            answer.append(chunk)
            yield "delta", {"text": chunk}
    except Exception as exc:  # noqa: BLE001 - surfaced to the client intact
        log.warning("answer_stream_failed", error=f"{type(exc).__name__}: {exc}")
        yield "error", {"error": f"{type(exc).__name__}: {exc}"}
        return

    yield "done", {
        "text_len": sum(len(part) for part in answer),
        "hit_count": len(hits),
        "model": agent.model,
        "usage": usage_delta(before, agent.usage),
        "took_ms": int((time.monotonic() - start) * 1000),
    }

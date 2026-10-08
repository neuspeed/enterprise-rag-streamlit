"""Failover between a primary provider and a fallback.

The behaviour under test is the whole point of the feature: a rate-limited or
timed-out primary must not cost the buyer their job, and a configuration mistake
must still fail loudly instead of being silently retried elsewhere.
"""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import openai
import pytest

from know_everything_ai.enricher.knowledge_canvas import KnowledgeCanvasEnricher
from know_everything_ai.utils.llm_utils import LLMDriven, build_target

PRIMARY_URL = "https://primary.example/v1"
FALLBACK_URL = "http://ollama:11434/v1"


def rate_limit_error() -> openai.RateLimitError:
    return openai.RateLimitError(
        "rate limited",
        response=httpx.Response(status_code=429, request=httpx.Request("POST", "/")),
        body=None,
    )


def make_response(content: str = '{"ok": true}') -> SimpleNamespace:
    """A completion shaped like the SDK's, including usage for the cost report."""
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(
            prompt_tokens=11, completion_tokens=7, total_tokens=18
        ),
    )


def make_enricher(**overrides) -> KnowledgeCanvasEnricher:
    kwargs: dict = {
        "api_url": PRIMARY_URL,
        "api_key": "primary-key",
        "model": "glm-5.3",
        "system_prompt": "canvas",
        "max_retries": 0,
    }
    kwargs.update(overrides)
    fallback = build_target(
        api_url=FALLBACK_URL,
        api_key="",
        model="qwen3:4b",
        timeout=600,
        label="fallback",
    )
    return KnowledgeCanvasEnricher(fallback=fallback, **kwargs)


@contextmanager
def mock_create(component: LLMDriven, index: int, **kwargs):
    """Patch ``create`` on the client of the target at ``index``.

    Yields the mock so a test can assert on it. Patching the client the
    component will actually use is the point: a mock hung on ``openai_client``
    alone would miss the fallback path entirely.
    """
    target = component._client_for(component.targets[index])
    mock = AsyncMock(**kwargs)
    with patch.object(target.chat.completions, "create", new=mock):
        yield mock


def test_no_fallback_configured_means_one_target() -> None:
    enricher = KnowledgeCanvasEnricher(
        api_url=PRIMARY_URL,
        api_key="k",
        model="glm-5.3",
        system_prompt="s",
        fallback=build_target(api_url="", api_key="", model="", timeout=1),
    )

    assert [t.label for t in enricher.targets] == ["primary"]


def test_fallback_is_appended_after_the_primary() -> None:
    enricher = make_enricher()

    assert [(t.label, t.model) for t in enricher.targets] == [
        ("primary", "glm-5.3"),
        ("fallback", "qwen3:4b"),
    ]


@pytest.mark.asyncio
async def test_primary_success_never_touches_the_fallback() -> None:
    enricher = make_enricher()
    with (
        mock_create(enricher, 0, return_value=make_response()),
        mock_create(enricher, 1, return_value=make_response("from fallback")) as fb,
    ):
        result = await enricher.create_knowledge_canvas("doc")

    assert result == '{"ok": true}'
    fb.assert_not_awaited()
    assert enricher._preferred_index == 0


@pytest.mark.asyncio
async def test_rate_limit_falls_over_to_the_second_provider() -> None:
    """The reason this exists: z.ai answering 429 must not fail the job."""
    enricher = make_enricher()
    with (
        mock_create(enricher, 0, side_effect=rate_limit_error()),
        mock_create(enricher, 1, return_value=make_response("local answer")) as fb,
    ):
        result = await enricher.create_knowledge_canvas("doc")

    assert result == "local answer"
    fb.assert_awaited_once()
    assert enricher._preferred_index == 1


@pytest.mark.asyncio
async def test_fallback_client_is_created_lazily_with_its_own_model_name() -> None:
    """A local Ollama will not know 'glm-5.3'; it must be sent its own name."""
    enricher = make_enricher()
    with (
        mock_create(enricher, 0, side_effect=rate_limit_error()),
        mock_create(enricher, 1, return_value=make_response()) as fb,
    ):
        await enricher.create_knowledge_canvas("doc")

    assert fb.await_args.kwargs["model"] == "qwen3:4b"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        openai.APITimeoutError(request=httpx.Request("POST", "/")),
        openai.APIConnectionError(request=httpx.Request("POST", "/")),
        openai.InternalServerError(
            "boom",
            response=httpx.Response(status_code=500, request=httpx.Request("POST", "/")),
            body=None,
        ),
        ConnectionError("connection reset"),
    ],
    ids=["timeout", "connection", "server_error", "builtin_reset"],
)
async def test_every_transient_failure_falls_over(error) -> None:
    enricher = make_enricher()
    with (
        mock_create(enricher, 0, side_effect=error),
        mock_create(enricher, 1, return_value=make_response("local")) as fb,
    ):
        result = await enricher.create_knowledge_canvas("doc")

    assert result == "local"
    fb.assert_awaited_once()


@pytest.mark.asyncio
async def test_bad_request_is_not_retried_elsewhere() -> None:
    """A 400 is a bug in this deployment; another provider reproduces it."""
    enricher = make_enricher()
    bad_request = openai.BadRequestError(
        "bad",
        response=httpx.Response(status_code=400, request=httpx.Request("POST", "/")),
        body=None,
    )
    with (
        mock_create(enricher, 0, side_effect=bad_request),
        mock_create(enricher, 1, return_value=make_response()) as fb,
    ):
        with pytest.raises(openai.BadRequestError):
            await enricher.create_knowledge_canvas("doc")

    fb.assert_not_awaited()


@pytest.mark.asyncio
async def test_both_targets_down_raises_the_last_error() -> None:
    enricher = make_enricher()
    with (
        mock_create(enricher, 0, side_effect=rate_limit_error()),
        mock_create(
            enricher,
            1,
            side_effect=openai.APITimeoutError(
                request=httpx.Request("POST", "/")
            ),
        ),
    ):
        with pytest.raises(openai.APITimeoutError):
            await enricher.create_knowledge_canvas("doc")


@pytest.mark.asyncio
async def test_usage_is_recorded_from_the_target_that_answered() -> None:
    """A job that fell over is still billed for what was spent."""
    enricher = make_enricher()
    with (
        mock_create(enricher, 0, side_effect=rate_limit_error()),
        mock_create(enricher, 1, return_value=make_response()),
    ):
        await enricher.create_knowledge_canvas("doc")

    assert enricher.usage["calls"] == 1
    assert enricher.usage["total_tokens"] == 18


@pytest.mark.asyncio
async def test_successful_fallback_stops_retrying_the_dead_primary() -> None:
    """Re-diagnosing a dead provider on every chunk is what makes failover slow."""
    enricher = make_enricher()
    with (
        mock_create(enricher, 0, side_effect=rate_limit_error()) as primary,
        mock_create(enricher, 1, return_value=make_response()) as fb,
    ):
        await enricher.create_knowledge_canvas("doc one")
        await enricher.create_knowledge_canvas("doc two")

    assert primary.await_count == 1
    assert fb.await_count == 2


@pytest.mark.asyncio
async def test_reset_targets_puts_the_primary_back_in_front() -> None:
    """So a provider that recovered is used again on the next message."""
    enricher = make_enricher()
    with (
        mock_create(enricher, 0, side_effect=rate_limit_error()),
        mock_create(enricher, 1, return_value=make_response()),
    ):
        await enricher.create_knowledge_canvas("doc")

    assert enricher._preferred_index == 1
    enricher.reset_targets()
    assert enricher._preferred_index == 0


@pytest.mark.asyncio
async def test_an_injected_pool_is_shared_by_both_targets() -> None:
    """One pool per component was the original leak; failover must not add one."""
    http_client = httpx.AsyncClient()
    fallback = build_target(
        api_url=FALLBACK_URL, api_key="", model="qwen3:4b", timeout=600, label="fallback"
    )
    enricher = KnowledgeCanvasEnricher(
        api_url=PRIMARY_URL,
        api_key="k",
        model="glm-5.3",
        system_prompt="s",
        http_client=http_client,
        fallback=fallback,
    )
    try:
        clients = [enricher._client_for(t) for t in enricher.targets]
        assert len({id(c) for c in clients}) == 2
        assert all(c._client is http_client for c in clients)
    finally:
        await http_client.aclose()


def test_injected_pool_is_not_closed_by_the_component() -> None:
    """An injected client is owned by the caller; closing it breaks siblings."""
    http_client = httpx.AsyncClient()
    fallback = build_target(
        api_url=FALLBACK_URL, api_key="", model="qwen3:4b", timeout=600, label="fallback"
    )
    enricher = KnowledgeCanvasEnricher(
        api_url=PRIMARY_URL,
        api_key="k",
        model="glm-5.3",
        system_prompt="s",
        http_client=http_client,
        fallback=fallback,
    )
    # Creating the fallback client must not have replaced the shared pool.
    enricher._client_for(enricher.targets[1])
    assert enricher._client_for(enricher.targets[0])._client is http_client

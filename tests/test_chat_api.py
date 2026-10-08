"""HTTP tests for the streaming chat surface.

The answer agent is a fake that streams fixed tokens, so nothing here waits on a
real model; retrieval still goes through the in-memory store, which keeps the
"resolve, retrieve, then stream" ordering visible.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
import pytest_asyncio

from know_everything_ai.api.main import create_app
from know_everything_ai.settings import Settings
from know_everything_ai.stores.base import Chunk, SearchHit
from know_everything_ai.stores.memory import MemoryStoreContext
from know_everything_ai.stores.registry import SuccessRecord

VALID_KEY = "secret-key"


class FakeApiKeys:
    async def verify(self, plain: str) -> bool:
        return plain == VALID_KEY

    async def aclose(self) -> None:
        pass


class FakeAnswerAgent:
    def __init__(
        self,
        *,
        configured: bool = True,
        max_fragments: int = 6,
        chunks: tuple[str, ...] = ("Прив", "ет!"),
    ) -> None:
        self.configured = configured
        self.max_fragments = max_fragments
        self.model = "glm-5.3-flash"
        self.chunks = chunks
        self.last_fragments: list[SearchHit] = []
        self.usage = {
            "calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    async def stream_answer(self, question: str, fragments: list[SearchHit]):
        self.last_fragments = fragments
        for chunk in self.chunks:
            yield chunk


class ExplodingAnswerAgent(FakeAnswerAgent):
    async def stream_answer(self, question: str, fragments: list[SearchHit]):
        raise httpx.ConnectError("model down")
        yield  # pragma: no cover  - the raise keeps this an async generator


def build_settings(**overrides: Any) -> Settings:
    return Settings(
        _env_file=None,
        QUERY_MAX_LIMIT=50,
        QUERY_MIN_SCORE=-1.0,
        RECORD_MANAGER_HOST="localhost",
        **overrides,
    )


async def seed_kb(
    ctx: MemoryStoreContext,
    kb_external_id: str,
    *,
    branch: str = "vector",
    chunks: list[str] | None = None,
    ok: bool = True,
) -> int:
    chunks = chunks or []
    kb_id = await ctx.registry.ensure(kb_external_id, "auto", branch)
    if ok:
        await ctx.registry.record_success(
            SuccessRecord(
                kb_external_id=kb_external_id,
                kb_type="auto",
                branch=branch,
                total_tokens=120,
                item_count=len(chunks),
            )
        )
        await ctx.vector_store.replace_chunks(
            kb_id,
            [
                Chunk(
                    text,
                    category="document",
                    source="https://example.com/doc",
                    title=f"Fragment {index}",
                )
                for index, text in enumerate(chunks)
            ],
        )
    return kb_id


def parse_sse(text: str) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    for block in text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        kind: str | None = None
        data: dict | None = None
        for line in block.splitlines():
            if line.startswith("event: "):
                kind = line[len("event: "):]
            elif line.startswith("data: "):
                data = json.loads(line[len("data: "):])
        events.append((kind or "", data or {}))
    return events


@pytest_asyncio.fixture
async def chat_api():
    ctx = MemoryStoreContext(build_settings())
    await seed_kb(ctx, "kb-ready", chunks=["Как настроить проверку.", "Шаг второй."])
    app = create_app(
        build_settings(),
        ctx=ctx,
        api_keys=FakeApiKeys(),
        answer_agent=FakeAnswerAgent(),
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client
    await ctx.aclose()


@pytest.mark.asyncio
async def test_chat_requires_a_key() -> None:
    ctx = MemoryStoreContext(build_settings())
    app = create_app(
        build_settings(),
        ctx=ctx,
        api_keys=FakeApiKeys(),
        answer_agent=FakeAnswerAgent(),
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/knowledge-bases/kb-ready/chat",
            json={"question": "как настроить?"},
        )
    assert response.status_code == 401
    await ctx.aclose()


@pytest.mark.asyncio
async def test_chat_without_configured_agent_is_503(settings: Settings) -> None:
    """A retrieval-only deployment earns a clear 503, not a hidden model call
    against an unset URL."""
    ctx = MemoryStoreContext(settings)
    await seed_kb(ctx, "kb-ready", chunks=["фрагмент первый"])
    app = create_app(
        settings,
        ctx=ctx,
        api_keys=FakeApiKeys(),
        # No agent injected: create_app builds the real AnswerAgent from
        # settings with empty ANSWER_*, so configured == False.
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/knowledge-bases/kb-ready/chat",
            json={"question": "как настроить?"},
            headers={"Authorization": f"Bearer {VALID_KEY}"},
        )
    assert response.status_code == 503
    assert "answer agent is not configured" in response.json()["detail"]["message"]
    await ctx.aclose()


@pytest.mark.asyncio
async def test_chat_unknown_kb_is_404(chat_api) -> None:
    response = await chat_api.post(
        "/v1/knowledge-bases/does-not-exist/chat",
        json={"question": "вопрос"},
        headers={"Authorization": f"Bearer {VALID_KEY}"},
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_chat_pending_kb_is_409() -> None:
    ctx = MemoryStoreContext(build_settings())
    await seed_kb(ctx, "pending-kb", ok=False)
    app = create_app(
        build_settings(),
        ctx=ctx,
        api_keys=FakeApiKeys(),
        answer_agent=FakeAnswerAgent(),
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/knowledge-bases/pending-kb/chat",
            json={"question": "вопрос"},
            headers={"Authorization": f"Bearer {VALID_KEY}"},
        )
    assert response.status_code == 409
    assert response.json()["detail"]["status"] == "pending"
    await ctx.aclose()


@pytest.mark.asyncio
async def test_chat_streams_sources_then_answer(chat_api) -> None:
    response = await chat_api.post(
        "/v1/knowledge-bases/kb-ready/chat",
        json={"question": "как настроить?"},
        headers={"Authorization": f"Bearer {VALID_KEY}"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(response.text)
    kinds = [kind for kind, _ in events]
    assert kinds == ["sources", "delta", "delta", "done"]

    sources = events[0][1]
    assert sources["kb_external_id"] == "kb-ready"
    assert sources["branch"] == "vector"
    assert sources["model"] == "glm-5.3-flash"
    assert len(sources["hits"]) == 2
    assert "retrieval_took_ms" in sources

    answer = "".join(payload["text"] for kind, payload in events if kind == "delta")
    assert answer == "Привет!"

    done = events[-1][1]
    assert done["hit_count"] == 2
    assert done["text_len"] == len("Привет!")
    assert done["model"] == "glm-5.3-flash"
    assert done["usage"]["total_tokens"] == 0


@pytest.mark.asyncio
async def test_chat_fragment_limit_is_the_stricter_of_two() -> None:
    ctx = MemoryStoreContext(build_settings())
    await seed_kb(
        ctx,
        "kb-many",
        chunks=[f"фрагмент {index}" for index in range(5)],
    )
    agent = FakeAnswerAgent(max_fragments=2)
    app = create_app(
        build_settings(),
        ctx=ctx,
        api_keys=FakeApiKeys(),
        answer_agent=agent,
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/knowledge-bases/kb-many/chat",
            json={"question": "вопрос", "limit": 5},
            headers={"Authorization": f"Bearer {VALID_KEY}"},
        )
    assert response.status_code == 200
    events = parse_sse(response.text)
    assert len(events[0][1]["hits"]) <= 2
    assert len(agent.last_fragments) <= 2
    await ctx.aclose()


@pytest.mark.asyncio
async def test_chat_model_failure_arrives_as_an_error_event() -> None:
    ctx = MemoryStoreContext(build_settings())
    await seed_kb(ctx, "kb-ready", chunks=["фрагмент первый"])
    app = create_app(
        build_settings(),
        ctx=ctx,
        api_keys=FakeApiKeys(),
        answer_agent=ExplodingAnswerAgent(),
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/knowledge-bases/kb-ready/chat",
            json={"question": "вопрос"},
            headers={"Authorization": f"Bearer {VALID_KEY}"},
        )
    assert response.status_code == 200  # the stream starts fine
    events = parse_sse(response.text)
    assert events[0][0] == "sources"
    assert events[1][0] == "error"
    assert "ConnectError" in events[1][1]["error"]
    await ctx.aclose()

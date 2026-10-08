"""HTTP tests for the read-only query API.

The app is built with the in-memory store and a fake key store, so nothing here
touches Postgres or fastembed weights. Seeding goes through the same
registry-then-vector-store path the pipeline uses, which is what keeps the
tests honest about ordering instead of inserting rows behind the code's back.
"""

from __future__ import annotations

import httpx
import pytest
import pytest_asyncio

from know_everything_ai.api.main import create_app
from know_everything_ai.settings import Settings
from know_everything_ai.stores.base import Chunk
from know_everything_ai.stores.memory import MemoryStoreContext
from know_everything_ai.stores.registry import SuccessRecord

VALID_KEY = "secret-key"


class FakeApiKeys:
    def __init__(self, valid: tuple[str, ...] = (VALID_KEY,)) -> None:
        self._valid = set(valid)

    async def verify(self, plain: str) -> bool:
        return plain in self._valid

    async def aclose(self) -> None:
        pass


def build_settings(**overrides) -> Settings:
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
    kb_id = await ctx.registry.ensure(kb_external_id, "auto", branch)
    if ok:
        await ctx.registry.record_success(
            SuccessRecord(
                kb_external_id=kb_external_id,
                kb_type="auto",
                branch=branch,
                total_tokens=120,
                item_count=len(chunks or []),
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
                for index, text in enumerate(chunks or [])
            ],
        )
    return kb_id


@pytest_asyncio.fixture
async def api(settings: Settings):
    ctx = MemoryStoreContext(settings)
    app = create_app(settings, ctx=ctx, api_keys=FakeApiKeys())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client, ctx
    await ctx.aclose()


@pytest.mark.asyncio
async def test_healthz_needs_no_key(api):
    client, _ = api
    response = await client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_query_requires_a_key(api):
    client, _ = api
    response = await client.post(
        "/v1/knowledge-bases/kb-a/query", json={"query": "hello"}
    )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_query_rejects_a_revoked_key(api, settings: Settings):
    _, ctx = api
    bad_app = create_app(
        settings,
        ctx=ctx,
        api_keys=FakeApiKeys(valid=()),
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=bad_app), base_url="http://test"
    ) as bad_client:
        response = await bad_client.post(
            "/v1/knowledge-bases/kb-a/query",
            json={"query": "hello"},
            headers={"Authorization": f"Bearer {VALID_KEY}"},
        )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_query_unknown_kb_is_404(api):
    client, _ = api
    response = await client.post(
        "/v1/knowledge-bases/does-not-exist/query",
        json={"query": "hello"},
        headers={"Authorization": f"Bearer {VALID_KEY}"},
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_query_pending_kb_is_409(api):
    client, ctx = api
    await seed_kb(ctx, "pending-kb", ok=False)
    response = await client.post(
        "/v1/knowledge-bases/pending-kb/query",
        json={"query": "hello"},
        headers={"Authorization": f"Bearer {VALID_KEY}"},
    )
    assert response.status_code == 409
    body = response.json()
    assert body["detail"]["status"] == "pending"


@pytest.mark.asyncio
async def test_query_returns_ranked_hits(api):
    client, ctx = api
    await seed_kb(
        ctx,
        "kb-ready",
        chunks=[
            "Первый фрагмент про деплой сервиса.",
            "Второй фрагмент про токенизацию документов.",
        ],
    )
    response = await client.post(
        "/v1/knowledge-bases/kb-ready/query",
        json={"query": "токенизация"},
        headers={"Authorization": f"Bearer {VALID_KEY}"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["kb_external_id"] == "kb-ready"
    assert body["branch"] == "vector"
    assert body["status"] == "ok"
    assert body["hit_count"] == 2
    assert body["model"] != ""
    assert body["score_floor"] == -1.0
    for hit in body["hits"]:
        assert "chunk_id" in hit
        assert "content" in hit
        assert "score" in hit
        assert hit["source"] == "https://example.com/doc"


@pytest.mark.asyncio
async def test_query_limit_is_clamped_not_rejected(api):
    _, ctx = api
    # A page cap of 1 with two chunks on file: the caller asking for 5 must get
    # the best one, not an error, and certainly not five.
    app = create_app(
        Settings(_env_file=None, QUERY_MAX_LIMIT=1),
        ctx=ctx,
        api_keys=FakeApiKeys(),
    )
    await seed_kb(
        ctx,
        "kb-clamped",
        chunks=["фрагмент первый", "фрагмент второй"],
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/knowledge-bases/kb-clamped/query",
            json={"query": "фрагмент", "limit": 5},
            headers={"Authorization": f"Bearer {VALID_KEY}"},
        )
    assert response.status_code == 200
    assert response.json()["hit_count"] <= 1


@pytest.mark.asyncio
async def test_query_min_score_drops_everything(api):
    client, ctx = api
    await seed_kb(ctx, "kb-scored", chunks=["фрагмент первый", "фрагмент второй"])
    response = await client.post(
        "/v1/knowledge-bases/kb-scored/query",
        json={"query": "любой", "min_score": 2.0},
        headers={"Authorization": f"Bearer {VALID_KEY}"},
    )
    assert response.status_code == 200
    assert response.json()["hit_count"] == 0


@pytest.mark.asyncio
async def test_get_kb_includes_chunk_count(api):
    client, ctx = api
    await seed_kb(
        ctx,
        "kb-one",
        chunks=["фрагмент первый", "фрагмент второй", "фрагмент третий"],
    )
    response = await client.get(
        "/v1/knowledge-bases/kb-one",
        headers={"Authorization": f"Bearer {VALID_KEY}"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["chunk_count"] == 3
    # The canvas is megabytes and must not ride along on a dashboard read.
    assert "canvas" not in body


@pytest.mark.asyncio
async def test_list_knowledge_bases(api):
    client, ctx = api
    await seed_kb(ctx, "kb-a", chunks=["фрагмент первый"])
    await seed_kb(ctx, "kb-b", chunks=["ещё один"])
    response = await client.get(
        "/v1/knowledge-bases",
        headers={"Authorization": f"Bearer {VALID_KEY}"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 2
    assert len(body["items"]) == 2
    by_name = {item["kb_external_id"]: item for item in body["items"]}
    assert set(by_name) == {"kb-a", "kb-b"}
    # The listing is a dashboard's main view: fragment counts are half the
    # point, and canvas must not ride along.
    assert by_name["kb-a"]["chunk_count"] == 1
    assert by_name["kb-b"]["chunk_count"] == 1
    assert "canvas" not in body["items"][0]


@pytest.mark.asyncio
async def test_export_returns_every_chunk(api):
    client, ctx = api
    await seed_kb(
        ctx,
        "kb-export",
        chunks=["фрагмент первый", "фрагмент второй", "фрагмент третий"],
    )
    response = await client.get(
        "/v1/knowledge-bases/kb-export/export",
        headers={"Authorization": f"Bearer {VALID_KEY}"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["chunk_count"] == 3
    assert len(body["chunks"]) == 3

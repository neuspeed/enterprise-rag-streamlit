"""Live round-trip through Postgres + pgvector.

Runs only when a pgvector-compatible Postgres answers at the configured DSN —
skipped otherwise, so CI and a machine without the stack still get a green run.
Skipping is an explicit choice here: the in-memory store covers the ordering
contract in every run, and this test exists to catch what memory cannot, which
is SQL drift (missing column, dimension mismatch, broken adapter). Those only
matter when there is a database.

The vectors come from the fake embedder on purpose: the SQL path — not 220 MB of
model weights — is what is under test.
"""

from __future__ import annotations

import uuid

import psycopg
import pytest

from know_everything_ai.settings import Settings
from know_everything_ai.stores.base import Chunk
from know_everything_ai.stores.embeddings import FakeEmbedder
from know_everything_ai.stores.pgvector import PgVectorStore
from know_everything_ai.stores.registry import KnowledgeBaseRegistry, SuccessRecord

_SETTINGS = Settings(
    _env_file=None,
    VECTOR_STORE_HOST="localhost",
    VECTOR_STORE_PORT=5432,
    VECTOR_STORE_USER="postgres",
    VECTOR_STORE_DATABASE="vector_db",
    LOCAL_EMBEDDING_DIM=384,
)


def _db_available() -> bool:
    try:
        with psycopg.connect(_SETTINGS.database_dsn(), connect_timeout=2):
            return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _db_available(), reason="Postgres/pgvector is not reachable"
)


class TestPgVectorStore:
    @pytest.mark.asyncio
    async def test_ingest_then_retrieve_round_trip(self):
        store = PgVectorStore(_SETTINGS, FakeEmbedder(dim=384))
        registry = KnowledgeBaseRegistry(_SETTINGS)
        kb = f"it_{uuid.uuid4().hex[:12]}"
        try:
            kb_id = await registry.ensure(kb, "vector", "vector")
            stored = await store.count_chunks(kb_id)
            assert stored == 0

            inserted = await store.replace_chunks(
                kb_id,
                [
                    Chunk(content="асфальт и дорожное покрытие", category="document"),
                    Chunk(content="расписание электричек", category="document"),
                ],
            )
            assert inserted == 2
            assert await store.count_chunks(kb_id) == 2

            hits = await store.search(kb_id, "ремонт дороги", limit=2)
            assert len(hits) == 2
            assert isinstance(hits[0].score, float)
            assert hits[0].content

            await registry.record_success(
                SuccessRecord(
                    kb_external_id=kb,
                    kb_type="vector",
                    branch="vector",
                    total_tokens=12,
                    item_count=2,
                    estimator="tiktoken",
                    threshold=24000,
                    cost={"model_calls": 7},
                )
            )
            row = await registry.get_by_external_id(kb)
            assert row is not None
            assert row["status"] == "ok"
            assert row["item_count"] == 2
            assert row["cost"]["model_calls"] == 7
            assert row["total_tokens"] == 12
        finally:
            # Deleting the registry row cascades to its chunks; without this
            # the DB accumulates one knowledge_base per test run.
            if await registry.get_by_external_id(kb):
                moved = await _delete_kb(registry, kb)
                assert moved >= 1
            await store.aclose()
            await registry.aclose()

    @pytest.mark.asyncio
    async def test_replace_chunks_is_atomic_swap(self):
        store = PgVectorStore(_SETTINGS, FakeEmbedder(dim=384))
        registry = KnowledgeBaseRegistry(_SETTINGS)
        kb = f"it_swap_{uuid.uuid4().hex[:12]}"
        try:
            kb_id = await registry.ensure(kb, "auto", "vector")
            await store.replace_chunks(kb_id, [Chunk(content="старая версия")])
            assert await store.count_chunks(kb_id) == 1

            await store.replace_chunks(
                kb_id, [Chunk(content="новая версия a"), Chunk(content="новая версия b")]
            )
            assert await store.count_chunks(kb_id) == 2
            hits = await store.search(kb_id, "новая версия b", limit=5)
            assert all("старая" not in hit.content for hit in hits)
        finally:
            await _delete_kb(registry, kb)
            await registry.aclose()
            await store.aclose()

    @pytest.mark.asyncio
    async def test_empty_replace_leaves_nothing_behind(self):
        store = PgVectorStore(_SETTINGS, FakeEmbedder(dim=384))
        registry = KnowledgeBaseRegistry(_SETTINGS)
        kb = f"it_clear_{uuid.uuid4().hex[:12]}"
        try:
            kb_id = await registry.ensure(kb, "auto", "context")
            await store.replace_chunks(kb_id, [Chunk(content="было")])
            await store.replace_chunks(kb_id, [])
            assert await store.count_chunks(kb_id) == 0
        finally:
            await _delete_kb(registry, kb)
            await registry.aclose()
            await store.aclose()


async def _delete_kb(registry: KnowledgeBaseRegistry, kb: str) -> int:
    """Direct SQL delete — the registry offers no delete API yet."""
    from know_everything_ai.stores.pool import open_pool

    pool = await open_pool(_SETTINGS.database_dsn(), min_size=1, max_size=1)
    try:
        async with pool.connection() as conn:
            async with conn.transaction():
                cur = await conn.execute(
                    "DELETE FROM knowledge_bases WHERE kb_external_id = %s", (kb,)
                )
                return cur.rowcount or 0
    finally:
        await pool.close()

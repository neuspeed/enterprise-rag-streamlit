"""In-memory stand-ins for the built-in store.

Serves two callers that want the same thing: tests that must exercise the
registry-then-chunks-then-record sequence without a running Postgres, and any
preview path that wants the same shapes with no infrastructure at all.

Behaviour is deliberately faithful rather than convenient — replacement really
does drop the previous fragments, the registry really does keep the previous
success when only identity is written — because the point of using it in tests
is to catch ordering mistakes, not to sidestep them.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from math import sqrt

from know_everything_ai.settings import Settings
from know_everything_ai.stores.base import Chunk, Embedder, SearchHit
from know_everything_ai.stores.embeddings import FakeEmbedder
from know_everything_ai.stores.registry import SuccessRecord


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = sqrt(sum(x * x for x in a))
    norm_b = sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class MemoryVectorStore:
    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder
        self._rows: dict[int, list[tuple[Chunk, list[float]]]] = {}

    @property
    def embedder(self) -> Embedder:
        return self._embedder

    async def replace_chunks(self, kb_id: int, chunks: Sequence[Chunk]) -> int:
        if not chunks:
            self._rows[kb_id] = []
            return 0
        vectors = await self._embedder.embed_documents([c.content for c in chunks])
        if len(vectors) != len(chunks):
            raise ValueError(
                f"Embedder returned {len(vectors)} vectors for {len(chunks)} chunks"
            )
        self._rows[kb_id] = list(zip(chunks, vectors, strict=True))
        return len(chunks)

    async def append_chunks(self, kb_id: int, chunks: Sequence[Chunk]) -> int:
        if not chunks:
            return 0
        vectors = await self._embedder.embed_documents([c.content for c in chunks])
        if len(vectors) != len(chunks):
            raise ValueError(
                f"Embedder returned {len(vectors)} vectors for {len(chunks)} chunks"
            )
        self._rows.setdefault(kb_id, []).extend(zip(chunks, vectors, strict=True))
        return len(chunks)

    async def search(
        self, kb_id: int, query: str, limit: int = 10
    ) -> list[SearchHit]:
        rows = self._rows.get(kb_id, [])
        if not rows:
            return []
        query_vector = await self._embedder.embed_query(query)
        ranked = sorted(
            (
                (_cosine(query_vector, vector), chunk)
                for chunk, vector in rows
            ),
            key=lambda pair: pair[0],
            reverse=True,
        )
        return [
            SearchHit(
                chunk_id=index,
                content=chunk.content,
                score=score,
                category=chunk.category,
                source=chunk.source,
                title=chunk.title,
                metadata=chunk.metadata,
            )
            for index, (score, chunk) in enumerate(ranked[:limit])
        ]

    async def count_chunks(self, kb_id: int) -> int:
        return len(self._rows.get(kb_id, []))

    async def list_chunks(
        self, kb_id: int, limit: int = 200, offset: int = 0
    ) -> list[SearchHit]:
        available = self._rows.get(kb_id, [])
        return [
            SearchHit(
                chunk_id=index,
                content=chunk.content,
                score=1.0,
                category=chunk.category,
                source=chunk.source,
                title=chunk.title,
                metadata=chunk.metadata,
            )
            for index, (chunk, _) in enumerate(available[offset : offset + limit])
        ]

    async def aclose(self) -> None:
        self._rows.clear()


class MemoryRegistry:
    def __init__(self) -> None:
        self._rows: dict[str, dict] = {}
        self._next_id = 1

    async def ensure(self, kb_external_id: str, kb_type: str, branch: str) -> int:
        row = self._rows.get(kb_external_id)
        if row is None:
            row = {
                "id": self._next_id,
                "kb_external_id": kb_external_id,
                "kb_type": kb_type,
                "branch": branch,
                "status": "pending",
                "reason": None,
                "total_tokens": 0,
                "canvas_tokens": None,
                "estimator": None,
                "threshold": None,
                "payload_id": None,
                "job_id": None,
                "document_store_id": None,
                "item_count": 0,
                "cost": {},
                "canvas": None,
            }
            self._rows[kb_external_id] = row
            self._next_id += 1
        else:
            # Matches the SQL: identity columns only, so a re-run cannot clear
            # a success it has not yet reproduced.
            row["branch"] = branch
        return int(row["id"])

    async def record_success(self, record: SuccessRecord) -> int:
        kb_id = await self.ensure(record.kb_external_id, record.kb_type, record.branch)
        row = self._rows[record.kb_external_id]
        row.update(
            {
                "kb_type": record.kb_type,
                "branch": record.branch,
                "status": "ok",
                "reason": None,
                "total_tokens": record.total_tokens,
                "canvas_tokens": record.canvas_tokens,
                "estimator": record.estimator,
                "threshold": record.threshold,
                "payload_id": record.payload_id,
                "job_id": record.job_id,
                "document_store_id": record.document_store_id,
                "item_count": record.item_count,
                "cost": record.cost,
                "canvas": record.canvas,
            }
        )
        return kb_id

    async def count(self) -> int:
        return len(self._rows)

    async def get_by_external_id(self, kb_external_id: str) -> dict | None:
        row = self._rows.get(kb_external_id)
        return dict(row) if row else None

    async def get(self, kb_id: int) -> dict | None:
        for row in self._rows.values():
            if int(row["id"]) == kb_id:
                return dict(row)
        return None

    async def list(self, limit: int = 50, offset: int = 0) -> list[dict]:
        # Newest first by id: without real timestamps this is the same order
        # the SQL's `created_at DESC, id DESC` gives for rows created in a
        # single run, which is all a test exercises.
        rows = sorted(self._rows.values(), key=lambda row: int(row["id"]), reverse=True)
        return [dict(row) for row in rows[offset : offset + limit]]

    async def aclose(self) -> None:
        self._rows.clear()


class MemoryStoreContext:
    def __init__(self, settings: Settings, embedder: Embedder | None = None) -> None:
        self.settings = settings
        # The real embedder would download weights on first use; a fake keeps
        # "tests need no network and no 220 MB" true by default while still
        # allowing one to be handed in.
        self.embedder = embedder or FakeEmbedder(dim=settings.LOCAL_EMBEDDING_DIM)
        self.vector_store = MemoryVectorStore(self.embedder)
        self.registry = MemoryRegistry()
        self._lock = asyncio.Lock()

    async def aclose(self) -> None:
        await self.vector_store.aclose()
        await self.registry.aclose()
        await self.embedder.aclose()

"""Business logic for the read side, framework-free so it is testable without
an ASGI test client.

Splitting this out is deliberate: the routers are thin HTTP shells, and every
rule that matters — clamp, floor, status gate — lives here where a unit test
can reach it without spinning up a server.
"""

from __future__ import annotations

import time
from typing import Any, Protocol

from know_everything_ai.settings import Settings
from know_everything_ai.stores.base import SearchHit


class ReaderContext(Protocol):
    registry: Any
    vector_store: Any


class HasModel(Protocol):
    @property
    def model(self) -> str: ...


class KbNotFoundError(LookupError):
    """No registry row for this ``kb_external_id``."""


class KbNotReadyError(RuntimeError):
    """The knowledge base exists but is not safe to query yet.

    ``pending`` covers in-flight ingestion; a failed branch stays ``failed``
    with a reason, and answering from its fragments would present a partial or
    stale picture as current.
    """

    def __init__(self, status: str, reason: str | None) -> None:
        super().__init__(f"knowledge base is {status}: {reason or 'no reason'}")
        self.status = status
        self.reason = reason


class QueryService:
    def __init__(
        self,
        settings: Settings,
        ctx: ReaderContext,
        embedder: HasModel,
    ) -> None:
        self._settings = settings
        self._registry = ctx.registry
        self._vector_store = ctx.vector_store
        self._embedder = embedder

    @property
    def model(self) -> str:
        return self._embedder.model

    async def _resolve(self, kb_external_id: str) -> dict:
        row = await self._registry.get_by_external_id(kb_external_id)
        if row is None:
            raise KbNotFoundError(kb_external_id)
        if row.get("status") != "ok":
            raise KbNotReadyError(row.get("status", ""), row.get("reason"))
        return row

    def _clamp_limit(self, limit: int | None) -> int:
        requested = limit or 8
        return max(1, min(requested, self._settings.QUERY_MAX_LIMIT))

    async def list_all(self, limit: int, offset: int) -> tuple[list[dict], int]:
        limit = self._clamp_limit(limit)
        offset = max(0, offset)
        rows = await self._registry.list(limit, offset)
        # The listing is a dashboard's main view, where the size of a
        # knowledge base is half the point. N+1 is fine at page size; a registry
        # page is hundreds at most and each count is a cheap index count.
        for row in rows:
            row["chunk_count"] = await self._vector_store.count_chunks(
                int(row["id"])
            )
        total = await self._registry.count()
        return rows, total

    async def get(self, kb_external_id: str) -> dict:
        row = await self._registry.get_by_external_id(kb_external_id)
        if row is None:
            raise KbNotFoundError(kb_external_id)
        # The canvas is megabytes and a single-kb read rarely wants it; kept
        # out so this endpoint stays cheap enough for a dashboard to poll.
        row.pop("canvas", None)
        row["chunk_count"] = await self._vector_store.count_chunks(int(row["id"]))
        return row

    async def query(
        self,
        kb_external_id: str,
        text: str,
        limit: int | None,
        min_score: float | None,
    ) -> tuple[dict, list[SearchHit], int, float]:
        row = await self._resolve(kb_external_id)
        topk = self._clamp_limit(limit)
        floor = self._settings.QUERY_MIN_SCORE if min_score is None else min_score

        start = time.monotonic()
        hits = await self._vector_store.search(int(row["id"]), text, topk)
        took_ms = int((time.monotonic() - start) * 1000)

        hits = [hit for hit in hits if hit.score >= floor]
        return row, hits, took_ms, floor

    async def export(self, kb_external_id: str) -> tuple[dict, list[SearchHit]]:
        row = await self._resolve(kb_external_id)
        chunks = await self._vector_store.list_chunks(int(row["id"]))
        return row, chunks

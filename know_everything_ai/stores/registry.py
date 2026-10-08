"""The knowledge-base registry: makes an ingested knowledge base addressable.

Before this existed, a knowledge base was only reachable two ways — by
intercepting the webhook payload in flight, or by asking Flowise for a store
named after it. The context branch had neither: its canvas was never written
anywhere, so after the job finished the only copy was in a log file.

One row per ``kb_external_id`` records which branch produced it, how large it
was, and (for the context branch) the canvas itself, which is what lets the
query side decide whether the answer fits in one context window or has to be
retrieved in pieces.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import structlog
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

from know_everything_ai.settings import Settings
from know_everything_ai.stores.pool import open_pool

log = structlog.get_logger("registry")

_KB_COLUMNS = (
    "id, kb_external_id, kb_type, branch, status, reason, "
    "total_tokens, canvas_tokens, estimator, threshold, "
    "payload_id, job_id, document_store_id, item_count, "
    "cost, canvas, created_at, updated_at"
)

#: The same row without ``canvas``, which can be megabytes and is never wanted
#: by a listing. ``canvas_tokens`` is a different token and survives the
#: replace intact.
_KB_LIST_COLUMNS = _KB_COLUMNS.replace(", canvas,", ",")




@dataclass
class SuccessRecord:
    """Everything known after a branch finished successfully.

    Grouped here rather than passed as fifteen positional arguments: the set
    grows with each branch and a reordering at a call site would otherwise be a
    silent data-mapping bug.
    """

    kb_external_id: str
    kb_type: str
    branch: str
    total_tokens: int
    item_count: int
    estimator: str | None = None
    threshold: int | None = None
    payload_id: int | None = None
    job_id: int | None = None
    document_store_id: str | None = None
    canvas_tokens: int | None = None
    cost: dict[str, Any] = field(default_factory=dict)
    canvas: Any | None = None


class KnowledgeBaseRegistry:
    def __init__(self, settings: Settings) -> None:
        self._dsn = settings.database_dsn()
        self._pool: AsyncConnectionPool | None = None
        self._connect_lock = asyncio.Lock()

    async def _ensure_pool(self) -> AsyncConnectionPool:
        if self._pool is not None:
            return self._pool
        async with self._connect_lock:
            if self._pool is None:
                self._pool = await open_pool(self._dsn, min_size=1, max_size=2)
        return self._pool

    async def ensure(self, kb_external_id: str, kb_type: str, branch: str) -> int:
        """Return this knowledge base's row id, creating the row if needed.

        Deliberately narrow: it touches only identity columns, so a re-run that
        fails later leaves the previous success untouched rather than half
        overwritten. The row is marked ``pending`` on creation, which is the
        only state change a re-run is allowed to make before it finishes.
        """
        pool = await self._ensure_pool()
        async with pool.connection() as conn:
            async with conn.transaction():
                cur = await conn.execute(
                    """
                    INSERT INTO knowledge_bases
                        (kb_external_id, kb_type, branch, status)
                    VALUES (%s, %s, %s, 'pending')
                    ON CONFLICT (kb_external_id) DO UPDATE
                        SET branch = EXCLUDED.branch,
                            updated_at = now()
                    RETURNING id
                    """,
                    (kb_external_id, kb_type, branch),
                )
                row = await cur.fetchone()
        if row is None:
            raise RuntimeError(
                f"registry did not return an id for {kb_external_id!r}"
            )
        return int(row[0])

    async def record_success(self, record: SuccessRecord) -> int:
        """Write the completed run, creating the row if the caller skipped
        :meth:`ensure`."""
        pool = await self._ensure_pool()
        async with pool.connection() as conn:
            async with conn.transaction():
                cur = await conn.execute(
                    """
                    INSERT INTO knowledge_bases (
                        kb_external_id, kb_type, branch, status,
                        total_tokens, canvas_tokens, estimator, threshold,
                        payload_id, job_id, document_store_id, item_count,
                        cost, canvas
                    ) VALUES (
                        %(kb_external_id)s, %(kb_type)s, %(branch)s, 'ok',
                        %(total_tokens)s, %(canvas_tokens)s, %(estimator)s,
                        %(threshold)s, %(payload_id)s, %(job_id)s,
                        %(document_store_id)s, %(item_count)s, %(cost)s,
                        %(canvas)s
                    )
                    ON CONFLICT (kb_external_id) DO UPDATE SET
                        kb_type     = EXCLUDED.kb_type,
                        branch      = EXCLUDED.branch,
                        status      = 'ok',
                        reason      = NULL,
                        total_tokens     = EXCLUDED.total_tokens,
                        canvas_tokens    = EXCLUDED.canvas_tokens,
                        estimator        = EXCLUDED.estimator,
                        threshold        = EXCLUDED.threshold,
                        payload_id       = EXCLUDED.payload_id,
                        job_id           = EXCLUDED.job_id,
                        document_store_id = EXCLUDED.document_store_id,
                        item_count       = EXCLUDED.item_count,
                        cost             = EXCLUDED.cost,
                        canvas           = EXCLUDED.canvas,
                        updated_at       = now()
                    RETURNING id
                    """,
                    {
                        "kb_external_id": record.kb_external_id,
                        "kb_type": record.kb_type,
                        "branch": record.branch,
                        "total_tokens": record.total_tokens,
                        "canvas_tokens": record.canvas_tokens,
                        "estimator": record.estimator,
                        "threshold": record.threshold,
                        "payload_id": record.payload_id,
                        "job_id": record.job_id,
                        "document_store_id": record.document_store_id,
                        "item_count": record.item_count,
                        "cost": Jsonb(record.cost),
                        "canvas": Jsonb(record.canvas) if record.canvas is not None else None,
                    },
                )
                row = await cur.fetchone()
        kb_id = int(row[0]) if row else -1
        log.info(
            "registry_recorded",
            kb_external_id=record.kb_external_id,
            kb_id=kb_id,
            branch=record.branch,
            items=record.item_count,
            canvas_tokens=record.canvas_tokens,
        )
        return kb_id

    async def _select(
        self, sql: str, params: tuple[Any, ...]
    ) -> dict[str, Any] | None:
        pool = await self._ensure_pool()
        async with pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(sql, params)
                row = await cur.fetchone()
        return dict(row) if row else None

    async def count(self) -> int:
        pool = await self._ensure_pool()
        async with pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute("SELECT count(*) AS n FROM knowledge_bases")
                row = await cur.fetchone()
        return int(row["n"]) if row else 0

    async def get_by_external_id(self, kb_external_id: str) -> dict[str, Any] | None:
        return await self._select(
            f"""
            SELECT {_KB_COLUMNS}
            FROM knowledge_bases
            WHERE kb_external_id = %s
            """,
            (kb_external_id,),
        )

    async def get(self, kb_id: int) -> dict[str, Any] | None:
        """Same row, addressed by the id the pipeline records rather than by
        the caller's own name."""
        return await self._select(
            f"""
            SELECT {_KB_COLUMNS}
            FROM knowledge_bases
            WHERE id = %s
            """,
            (kb_id,),
        )

    async def list(self, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        """Registry contents newest first.

        The canvas is left out on purpose: it can be megabytes, and a listing
        that pulls it would fetch every knowledge base's whole payload just to
        render a table of names.
        """
        pool = await self._ensure_pool()
        async with pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    f"""
                    SELECT {_KB_LIST_COLUMNS}
                    FROM knowledge_bases
                    ORDER BY created_at DESC, id DESC
                    LIMIT %s OFFSET %s
                    """,
                    (limit, offset),
                )
                rows = await cur.fetchall()
        return [dict(row) for row in rows]

    async def aclose(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

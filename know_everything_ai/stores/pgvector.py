"""Postgres + pgvector: the built-in vector store.

Why pgvector rather than a dedicated engine: the dataset is one knowledge base
per row set, the queries are a filtered cosine top-k, and the registry that
makes a knowledge base addressable belongs beside it so a re-embed and a
registry update commit together. A second service would buy nothing here and
add an instance to keep alive.

Cost is paid at ingest, not at query time: :meth:`upsert_chunks` embeds before
it inserts, so the hot path (a question) never waits on a model.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

import structlog
from pgvector import Vector
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

from know_everything_ai.settings import Settings
from know_everything_ai.stores.base import Chunk, SearchHit
from know_everything_ai.stores.embeddings import FastembedEmbedder
from know_everything_ai.stores.pool import open_pool

log = structlog.get_logger("pgvector")

# Small on purpose. The worker's concurrent job limit is a handful, each step
# holds its connection for the length of one statement, and an idle pool still
# costs Postgres backends.
_POOL_MIN = 1
_POOL_MAX = 4


class PgVectorStore:
    def __init__(self, settings: Settings, embedder: FastembedEmbedder) -> None:
        self._dsn = settings.database_dsn()
        self._embedder = embedder
        self._topk = settings.VECTOR_STORE_TOPK
        self._pool: AsyncConnectionPool | None = None
        self._connect_lock = asyncio.Lock()

    async def _ensure_pool(self) -> AsyncConnectionPool:
        if self._pool is not None:
            return self._pool
        async with self._connect_lock:
            if self._pool is None:
                # Opened lazily so importing the module, or starting a process
                # that never queries, does not require a reachable database.
                self._pool = await open_pool(
                    self._dsn, min_size=_POOL_MIN, max_size=_POOL_MAX
                )
        return self._pool

    async def replace_chunks(self, kb_id: int, chunks: Sequence[Chunk]) -> int:
        """Embed, then swap in the complete set for this knowledge base.

        The embedding happens before the transaction opens: a model pass over a
        large corpus would otherwise hold a Postgres transaction, and its locks,
        open for as long as the model takes.
        """
        if not chunks:
            vectors: list[list[float]] = []
        else:
            vectors = await self._embedder.embed_documents(
                [c.content for c in chunks]
            )
            if len(vectors) != len(chunks):
                raise ValueError(
                    f"Embedder returned {len(vectors)} vectors for "
                    f"{len(chunks)} chunks"
                )

        pool = await self._ensure_pool()
        async with pool.connection() as conn:
            # Delete and insert share one transaction: an insert that failed
            # after the delete would leave the knowledge base with no content
            # and a registry row claiming otherwise.
            # executemany lives on the cursor; Connection only exposes execute.
            async with conn.transaction():
                await conn.execute("DELETE FROM chunks WHERE kb_id = %s", (kb_id,))
                async with conn.cursor() as cur:
                    await cur.executemany(
                        """
                        INSERT INTO chunks (kb_id, content, category, source,
                                            title, metadata, embedding)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        """,
                        [
                            (
                                kb_id,
                                chunk.content,
                                chunk.category,
                                chunk.source,
                                chunk.title,
                                Jsonb(chunk.metadata),
                                Vector(vector),
                            )
                            for chunk, vector in zip(chunks, vectors, strict=True)
                        ],
                    )
        return len(chunks)

    async def append_chunks(self, kb_id: int, chunks: Sequence[Chunk]) -> int:
        """Embed, then insert alongside the fragments already stored.

        The counterpart of :meth:`replace_chunks` for extending a knowledge
        base: nothing existing is deleted or re-embedded, so the cost of an
        update is proportional to the new files only. Embedding still happens
        before the transaction opens, for the same reason replace does it.
        """
        if not chunks:
            return 0
        vectors = await self._embedder.embed_documents([c.content for c in chunks])
        if len(vectors) != len(chunks):
            raise ValueError(
                f"Embedder returned {len(vectors)} vectors for "
                f"{len(chunks)} chunks"
            )

        pool = await self._ensure_pool()
        async with pool.connection() as conn:
            async with conn.transaction():
                async with conn.cursor() as cur:
                    await cur.executemany(
                        """
                        INSERT INTO chunks (kb_id, content, category, source,
                                            title, metadata, embedding)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        """,
                        [
                            (
                                kb_id,
                                chunk.content,
                                chunk.category,
                                chunk.source,
                                chunk.title,
                                Jsonb(chunk.metadata),
                                Vector(vector),
                            )
                            for chunk, vector in zip(chunks, vectors, strict=True)
                        ],
                    )
        return len(chunks)

    async def search(
        self, kb_id: int, query: str, limit: int | None = None
    ) -> list[SearchHit]:
        topk = min(limit or self._topk, 100)
        vector = await self._embedder.embed_query(query)
        pool = await self._ensure_pool()
        async with pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    """
                    SELECT id, content, category, source, title, metadata,
                           1 - (embedding <=> %(vec)s) AS score
                    FROM chunks
                    WHERE kb_id = %(kb_id)s
                      AND embedding IS NOT NULL
                    ORDER BY embedding <=> %(vec)s
                    LIMIT %(limit)s
                    """,
                    {"vec": Vector(vector), "kb_id": kb_id, "limit": topk},
                )
                rows = await cur.fetchall()
        return [
            SearchHit(
                chunk_id=row["id"],
                content=row["content"],
                score=float(row["score"] or 0.0),
                category=row["category"],
                source=row["source"],
                title=row["title"],
                metadata=row["metadata"] or {},
            )
            for row in rows
        ]

    async def count_chunks(self, kb_id: int) -> int:
        pool = await self._ensure_pool()
        async with pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    "SELECT count(*) AS n FROM chunks WHERE kb_id = %s",
                    (kb_id,),
                )
                row: dict[str, Any] | None = await cur.fetchone()
        return int(row["n"]) if row else 0

    async def list_chunks(
        self, kb_id: int, limit: int = 200, offset: int = 0
    ) -> list[SearchHit]:
        """Every fragment of a knowledge base, for export. No embeddings pass
        through here: exporting is a read, and ranking is someone else's job."""
        pool = await self._ensure_pool()
        async with pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    """
                    SELECT id, content, category, source, title, metadata
                    FROM chunks
                    WHERE kb_id = %s
                    ORDER BY id
                    LIMIT %s OFFSET %s
                    """,
                    (kb_id, limit, offset),
                )
                rows = await cur.fetchall()
        return [
            SearchHit(
                chunk_id=row["id"],
                content=row["content"],
                score=1.0,
                category=row["category"],
                source=row["source"],
                title=row["title"],
                metadata=row["metadata"] or {},
            )
            for row in rows
        ]

    async def aclose(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
        await self._embedder.aclose()

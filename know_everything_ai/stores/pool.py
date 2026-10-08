"""Shared connection pool setup.

Both stores open their own pool rather than sharing one: they are used at
different moments (the registry on write, the vector store around embedding),
a shared pool would serialise those behind one budget for no throughput gain,
and keeping them separate means a vector-store failure cannot take the registry
down with it.

``configure`` runs per connection, which is where the pgvector adapters have to
be registered. Doing it once per pool is not enough — the pool hands out
different connections — and skipping it turns the first insert into a
"can't adapt type list" error raised long after startup reported healthy.
"""

from __future__ import annotations

import psycopg
from psycopg_pool import AsyncConnectionPool


async def _register_adapters(conn: psycopg.AsyncConnection) -> None:
    from pgvector.psycopg import register_vector_async

    await register_vector_async(conn)


async def open_pool(
    dsn: str,
    *,
    min_size: int = 1,
    max_size: int = 4,
    timeout: float = 15.0,
) -> AsyncConnectionPool:
    """Create and open a pool.

    ``wait=True``: a pool that comes up empty and only fails later would defer a
    misconfigured DSN until the first query, which is inside a job rather than
    at startup where it belongs.
    """
    pool = AsyncConnectionPool(
        conninfo=dsn,
        min_size=min_size,
        max_size=max_size,
        open=False,
        configure=_register_adapters,
    )
    await pool.open(wait=True, timeout=timeout)
    return pool

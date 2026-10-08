"""Bearer keys for the query API.

Only the SHA-256 of a key is stored, so a leaked database row cannot be
replayed against the API: verification hashes the presented key and looks the
digest up, it never compares the plaintext it does not have.
"""

from __future__ import annotations

import asyncio
import hashlib
import secrets
import time
from dataclasses import dataclass
from typing import Any

import structlog
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from know_everything_ai.settings import Settings
from know_everything_ai.stores.pool import open_pool

log = structlog.get_logger("api_keys")

#: Makes a leaked key recognisable in a log line or a paste, so it can be
#: found and revoked instead of quietly exploited.
KEY_PREFIX = "kp_"


def hash_key(plain: str) -> str:
    """Digest the key exactly as it must be stored and looked up."""
    return hashlib.sha256(plain.encode("utf-8")).hexdigest()


def generate_key() -> str:
    """A fresh key. urlsafe rather than hex: 43 characters instead of 64."""
    return KEY_PREFIX + secrets.token_urlsafe(32)


@dataclass(frozen=True)
class IssuedKey:
    """A newly created key. ``plain`` exists only here and is never persisted."""

    id: int
    name: str
    plain: str


class ApiKeyStore:
    def __init__(self, settings: Settings) -> None:
        self._dsn = settings.database_dsn()
        self._cache_seconds = settings.QUERY_API_KEY_CACHE_SECONDS
        self._pool: AsyncConnectionPool | None = None
        self._connect_lock = asyncio.Lock()
        # hash -> (verified_at, ok). Successful lookups only: caching a miss
        # would delay a key that was just created, which is exactly the moment
        # someone is about to try it.
        self._cache: dict[str, tuple[float, bool]] = {}

    async def _ensure_pool(self) -> AsyncConnectionPool:
        if self._pool is not None:
            return self._pool
        async with self._connect_lock:
            if self._pool is None:
                self._pool = await open_pool(self._dsn, min_size=1, max_size=2)
        return self._pool

    async def create(self, name: str) -> IssuedKey:
        plain = generate_key()
        key_hash = hash_key(plain)
        pool = await self._ensure_pool()
        async with pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    """
                    INSERT INTO api_keys (name, key_hash)
                    VALUES (%s, %s)
                    RETURNING id
                    """,
                    (name, key_hash),
                )
                row = await cur.fetchone()
                assert row is not None
        key_id = int(row["id"])
        log.info("api_key_created", key_id=key_id, name=name)
        return IssuedKey(id=key_id, name=name, plain=plain)

    async def list(self) -> list[dict[str, Any]]:
        pool = await self._ensure_pool()
        async with pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    """
                    SELECT id, name, created_at, revoked_at
                    FROM api_keys
                    ORDER BY id
                    """
                )
                rows = await cur.fetchall()
        return [dict(row) for row in rows]

    async def revoke(self, key_id: int) -> bool:
        """Retire a key. Returns False when no such key exists."""
        pool = await self._ensure_pool()
        async with pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    """
                    UPDATE api_keys
                    SET revoked_at = now()
                    WHERE id = %s AND revoked_at IS NULL
                    RETURNING id
                    """,
                    (key_id,),
                )
                row = await cur.fetchone()
        if row is None:
            return False
        # Drop it now rather than waiting out the cache: revocation is an
        # incident response, not a background chore.
        self._cache.clear()
        log.info("api_key_revoked", key_id=key_id)
        return True

    async def verify(self, plain: str) -> bool:
        """True when the key exists, has not been revoked and hashes to a row."""
        if not plain or not plain.strip():
            return False
        key_hash = hash_key(plain)

        cached = self._cache.get(key_hash)
        if cached is not None:
            verified_at, ok = cached
            if time.monotonic() - verified_at < self._cache_seconds:
                return ok

        pool = await self._ensure_pool()
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT 1 FROM api_keys
                    WHERE key_hash = %s AND revoked_at IS NULL
                    """,
                    (key_hash,),
                )
                row = await cur.fetchone()
        ok = row is not None
        self._cache[key_hash] = (time.monotonic(), ok)
        return ok

    async def aclose(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

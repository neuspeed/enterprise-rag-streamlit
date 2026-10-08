"""Migration runner: statement splitting (no DB) and live idempotency/guard."""

from __future__ import annotations

import psycopg
import pytest

from know_everything_ai.migrations import _split_statements, run_migrations
from know_everything_ai.settings import Settings

_SETTINGS = Settings(
    _env_file=None,
    VECTOR_STORE_HOST="localhost",
    VECTOR_STORE_PORT=5432,
    VECTOR_STORE_USER="postgres",
    VECTOR_STORE_DATABASE="vector_db",
)


def _db_available() -> bool:
    try:
        with psycopg.connect(_SETTINGS.database_dsn(), connect_timeout=2):
            return True
    except Exception:
        return False


# --------------------------------------------------------------------- spinner


def test_split_statements_separates_simple_statements():
    script = "SELECT 1; SELECT 2;"
    assert _split_statements(script) == ["SELECT 1", "SELECT 2"]


def test_split_statements_ignores_semicolon_inside_string():
    script = "SELECT 'a;b'; SELECT 1"
    assert _split_statements(script) == ["SELECT 'a;b'", "SELECT 1"]


def test_split_statements_ignores_comment_and_its_semicolon():
    script = "-- note; stays here\nSELECT 1;"
    assert _split_statements(script) == ["-- note; stays here\nSELECT 1"]


def test_split_statements_does_not_treat_string_apostrophe_as_comment():
    # A naive splitter that regex-strips "--..." would truncate at the
    # apostrophe inside the string.
    script = "INSERT INTO t VALUES ('a--b'); SELECT 2"
    assert _split_statements(script) == ["INSERT INTO t VALUES ('a--b')", "SELECT 2"]


def test_split_statements_drops_whitespace_statements():
    script = "SELECT 1;;\n;;"
    assert _split_statements(script) == ["SELECT 1"]


def test_split_statements_keeps_string_with_comment_and_semicolon():
    script = "SELECT 'x;--y'; SELECT 1"
    assert _split_statements(script) == ["SELECT 'x;--y'", "SELECT 1"]


# ----------------------------------------------------------------------- live

pytestmark = pytest.mark.skipif(
    not _db_available(), reason="Postgres/pgvector is not reachable"
)


@pytest.mark.asyncio
async def test_run_migrations_is_idempotent_and_succeeds():
    first = await run_migrations(_SETTINGS)
    # State-independent assertions only: the shared dev database may already be
    # fully migrated, in which case both runs legitimately apply nothing.
    second = await run_migrations(_SETTINGS)
    assert isinstance(first, list)
    assert second == []


@pytest.mark.asyncio
async def test_run_migrations_guards_against_embedding_dimension_change():
    settings = Settings(
        _env_file=None,
        VECTOR_STORE_HOST="localhost",
        VECTOR_STORE_PORT=5432,
        VECTOR_STORE_USER="postgres",
        VECTOR_STORE_DATABASE="vector_db",
        LOCAL_EMBEDDING_DIM=768,
    )
    with pytest.raises(ValueError, match="Embedding model mismatch"):
        await run_migrations(settings)

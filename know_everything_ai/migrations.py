"""SQL migrations.

Deliberately plain files plus a runner rather than Alembic: the schema here is a
handful of tables owned by one service, and Alembic would add a dependency, a
config file and a revision-graph concept to a project that has none of them.
`python -m know_everything_ai migrate` is the whole surface.

Each file is applied exactly once inside a transaction and recorded in
``schema_migrations``. A failed file leaves no partial schema behind, because the
transaction is rolled back before the record row is written.
"""

from __future__ import annotations

import re
from pathlib import Path

import psycopg
import structlog

from know_everything_ai.settings import Settings

log = structlog.get_logger("migrations")

#: ``LOCAL_EMBEDDING_DIM`` is baked into ``vector(...)`` column types, so the
#: SQL is a template. A literal dimension in the file would silently diverge from
#: the model configured in settings.
_DIMENSION_PLACEHOLDER = "{{EMBEDDING_DIM}}"

_MIGRATIONS_DIR = Path(__file__).resolve().parents[1] / "migrations"

_COMMENT_OR_STRING = re.compile(
    r"""
      --[^\n]*            # line comment
    | '(?:[^']|'')*'      # single-quoted literal
    """,
    re.VERBOSE,
)


def _split_statements(script: str) -> list[str]:
    """Split a migration into statements.

    ``psycopg`` runs one statement per ``execute`` call, so a file holding the
    whole schema has to be split. Characters inside a comment or a string
    literal are masked first: a ``;`` in either would otherwise cut a statement
    in half, and a ``--`` inside a literal would make everything after it look
    like a comment.
    """
    protected = [False] * len(script)
    for match in _COMMENT_OR_STRING.finditer(script):
        for index in range(match.start(), match.end()):
            protected[index] = True

    statements: list[str] = []
    buffer_start = 0
    for index, char in enumerate(script):
        if char == ";" and not protected[index]:
            chunk = script[buffer_start:index]
            if chunk.strip():
                statements.append(chunk.strip())
            buffer_start = index + 1
    tail = script[buffer_start:]
    if tail.strip():
        statements.append(tail.strip())
    return statements


def migration_files() -> list[Path]:
    if not _MIGRATIONS_DIR.is_dir():
        # A missing directory is a packaging error, not "no migrations yet".
        # Returning an empty list here would start the worker against an empty
        # schema and defer the failure to the first message, which is already
        # consumed by then.
        raise FileNotFoundError(
            f"Migration directory not found: {_MIGRATIONS_DIR}. The image or "
            "checkout is incomplete; the migrations must ship with the code."
        )
    return sorted(p for p in _MIGRATIONS_DIR.glob("*.sql") if p.is_file())


async def run_migrations(settings: Settings) -> list[str]:
    """Apply every pending migration. Returns the versions applied."""
    files = migration_files()
    if not files:
        log.warning("migrations_none_found", directory=str(_MIGRATIONS_DIR))
        return []

    applied: list[str] = []
    async with await psycopg.AsyncConnection.connect(settings.database_dsn()) as conn:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version    TEXT PRIMARY KEY,
                applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
        cursor = await conn.execute("SELECT version FROM schema_migrations")
        done = {row[0] async for row in cursor}

        for path in files:
            version = path.stem
            if version in done:
                continue
            raw = path.read_text(encoding="utf-8")
            rendered = raw.replace(
                _DIMENSION_PLACEHOLDER, str(settings.LOCAL_EMBEDDING_DIM)
            )
            if _DIMENSION_PLACEHOLDER in raw and not settings.LOCAL_EMBEDDING_DIM:
                raise ValueError(
                    f"{version}: LOCAL_EMBEDDING_DIM is unset, so the vector "
                    "column type cannot be built"
                )
            # One transaction per file: either the whole file lands or nothing
            # does, and schema_migrations is written inside it.
            async with conn.transaction():
                for statement in _split_statements(rendered):
                    await conn.execute(statement)
                await conn.execute(
                    "INSERT INTO schema_migrations (version) VALUES (%s)",
                    (version,),
                )
            applied.append(version)
            log.info("migration_applied", version=version)

        await _reconcile_embedding_state(conn, settings)

    if applied:
        log.info("migrations_done", applied=applied)
    return applied


async def _reconcile_embedding_state(conn: psycopg.AsyncConnection, settings: Settings) -> None:
    """Make the stored embedding identity match the configured one.

    The column type was fixed when the schema was created. Re-pointing settings
    at a different model or dimension would leave every stored vector
    unsearchable-or-wrong without anything failing at write time, so the
    mismatch is reported at migration time instead, where it can still be acted
    on.
    """
    cursor = await conn.execute(
        "SELECT model, dim FROM embedding_state WHERE singleton"
    )
    row = await cursor.fetchone()
    configured_model = settings.LOCAL_EMBEDDING_MODEL
    configured_dim = settings.LOCAL_EMBEDDING_DIM

    if row is None:
        await conn.execute(
            """
            INSERT INTO embedding_state (singleton, model, dim)
            VALUES (true, %s, %s)
            ON CONFLICT (singleton) DO UPDATE
                SET model = EXCLUDED.model, dim = EXCLUDED.dim,
                    updated_at = now()
            """,
            (configured_model, configured_dim),
        )
        log.info(
            "embedding_state_initialized",
            model=configured_model,
            dim=configured_dim,
        )
        return

    stored_model, stored_dim = row
    if stored_model != configured_model or stored_dim != configured_dim:
        raise ValueError(
            "Embedding model mismatch: the database holds vectors from "
            f"{stored_model!r} (dim {stored_dim}) but settings ask for "
            f"{configured_model!r} (dim {configured_dim}). Re-embed the "
            "knowledge bases, or point settings back at the stored model. "
            "Continuing would return irrelevant results instead of failing."
        )

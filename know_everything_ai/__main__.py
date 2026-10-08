"""Process entrypoint.

``python -m know_everything_ai`` and the ``kb-pipeline`` console script both land
here, so the container command and a developer's shell take the same path —
including the config validation that has to happen before a socket is opened.

Subcommands:
    migrate    apply pending SQL migrations and exit.
    apikey     mint, list or retire bearer keys for the query API.
    api        serve the read-only query API and exit.
"""

from __future__ import annotations

import asyncio
import sys

import structlog

from know_everything_ai.settings import Settings
from know_everything_ai.utils.logging import configure_logging

log = structlog.get_logger("kb_pipeline")

_COMMANDS = ("migrate", "apikey", "api")


def _use_compatible_event_loop() -> None:
    """Switch Windows to a selector loop before anything opens psycopg.

    Windows defaults to ProactorEventLoop, which psycopg's async mode refuses to
    drive, so every database call would fail with an InterfaceError regardless of
    configuration. Set before the first asyncio.run(), because the policy is only
    read when the loop is created. Linux is unaffected — that is the loop policy
    it already uses — so the guard is what keeps host-side commands working here
    while containers are left alone.
    """
    if sys.platform != "win32":
        return
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


async def _run() -> None:
    from know_everything_ai.transports.rabbitmq import main as rabbitmq_main

    await rabbitmq_main()


async def _migrate(settings: Settings) -> int:
    """Apply migrations and stop.

    Exits rather than falling through to the worker so that ``migrate`` can be
    used as a separate step in a deploy without starting a consumer alongside it.
    """
    from know_everything_ai.migrations import run_migrations

    try:
        applied = await run_migrations(settings)
    except Exception:
        log.exception("migrations_failed")
        return 1
    log.info("migrations_current", applied=applied)
    return 0


def _run_api(settings: Settings) -> int:
    """Serve the read-only query API.

    Kept separate from ``_serve()`` on purpose: the API validates against the
    ``api`` configuration role, so a deployment that only reads vectors starts
    without the worker's prompts and model URLs, and it runs an HTTP server
    instead of a consumer. uvicorn is imported here rather than at module top
    because the worker image does not install a web framework.
    """
    settings.ensure_valid(role="api")

    # Before the socket listens, not after: the first request would otherwise
    # be the thing that discovers a missing table. Idempotent and recorded, so
    # a restart re-applies nothing.
    exit_code = asyncio.run(_migrate(settings))
    if exit_code:
        return exit_code

    import uvicorn

    from know_everything_ai.api.main import create_app

    uvicorn.run(
        create_app(settings),
        host=settings.QUERY_API_HOST,
        port=settings.QUERY_API_PORT,
        # Leave the logging configuration alone: uvicorn's own dictConfig
        # would replace the root handlers structlog is writing through and
        # every request line would fall back to plain text.
        log_config=None,
    )
    return 0


async def _serve(settings: Settings) -> None:
    from know_everything_ai.migrations import run_migrations

    # Before the broker, not after: a schema the worker expects but does not
    # have fails inside a job, where the message is already consumed, instead
    # of at startup where nothing is lost. Migrations are idempotent and
    # recorded, so a restart re-applies nothing.
    await run_migrations(settings)
    await _run()


def main() -> None:
    _use_compatible_event_loop()
    settings = Settings()
    configure_logging(settings.LOG_LEVEL, json_output=settings.LOG_JSON)

    args = sys.argv[1:]
    if args:
        command = args[0]
        if command == "migrate":
            # No ensure_valid() here: migrations need a database and the
            # embedding dimension, nothing else. Requiring an LLM URL to create
            # a table would mean the schema could not be built before secrets
            # are mounted.
            sys.exit(asyncio.run(_migrate(settings)))
        if command == "apikey":
            # Same reasoning: issuing a key needs a database, not a vision
            # model. It runs in a container with no web framework installed.
            from know_everything_ai.api_keys_cli import run as apikey_main

            sys.exit(asyncio.run(apikey_main(settings, args[1:])))
        if command == "api":
            sys.exit(_run_api(settings))
        # Rejected rather than ignored: a mistyped subcommand that falls through
        # to `serve` starts a worker nobody asked for.
        log.error(
            "unknown_command",
            command=command,
            supported=list(_COMMANDS),
        )
        sys.exit(2)

    # validate=True here rather than deep inside the transport: a bad config
    # should name every missing variable at once, not fail on the first one it
    # happens to read.
    settings.ensure_valid()

    transport = settings.INGEST_TRANSPORT
    if transport != "rabbitmq":
        # "postgres" is a P1 item with no worker shipped yet. Refusing here
        # beats starting a process that will never receive a message.
        log.error(
            "unsupported_transport",
            transport=transport,
            supported=["rabbitmq"],
        )
        sys.exit(2)

    log.info(
        "starting",
        transport=transport,
        env=settings.ENV,
        flowise=settings.FLOWISE_ENABLED,
        clients=sorted(settings.CLIENT_WEIGHTS),
    )
    try:
        asyncio.run(_serve(settings))
    except KeyboardInterrupt:
        log.info("interrupted")


if __name__ == "__main__":
    main()

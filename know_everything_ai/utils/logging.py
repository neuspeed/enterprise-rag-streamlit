"""Logging setup.

Kept in one place because the alternatives were worse: the classifier used to
call ``structlog.configure`` at import time with a hardcoded DEBUG level and a
JSON renderer, which meant importing a module rewrote logging for the whole
process and ``LOG_LEVEL`` was never consulted by anything.
"""

from __future__ import annotations

import logging
import sys

import structlog


def configure_logging(level: str = "INFO", *, json_output: bool = True) -> None:
    """Configure stdlib logging and structlog together.

    structlog routes stdlib records through its own processors, so the two have
    to be configured as a pair or log lines end up half-rendered.
    """
    numeric = getattr(logging, level.upper(), logging.INFO)

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=numeric,
        force=True,
    )

    renderer = (
        structlog.processors.JSONRenderer(ensure_ascii=False)
        if json_output
        else structlog.dev.ConsoleRenderer(colors=False)
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            renderer,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(numeric),
        cache_logger_on_first_use=True,
    )

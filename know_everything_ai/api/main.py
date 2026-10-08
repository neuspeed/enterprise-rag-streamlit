"""Application factory for the query API.

Built as a factory rather than a module-level app so tests can hand in the
in-memory store and a fake key store, and so the worker image — which never
imports this package — is proven to stay that way by simply not shipping the
dependency.

Startup is intentionally light: the embedder downloads weights and the pool
opens connections lazily, so this factory does neither. Migrations are applied
by ``__main__._run_api`` before the process serves, keeping the "migrate on
startup" rule in one place.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

import structlog
from fastapi import FastAPI

from know_everything_ai.agent.answer import AnswerAgent
from know_everything_ai.api.routers import chat, health, query
from know_everything_ai.settings import Settings
from know_everything_ai.stores import ApiKeyStore, StoreContext

log = structlog.get_logger("query_api")

_APP_VERSION = "1.0.0"


def create_app(
    settings: Settings,
    *,
    ctx: Any = None,
    api_keys: Any = None,
    answer_agent: Any = None,
) -> FastAPI:
    """Application factory.

    ``ctx`` is a ``StoreContext`` (real persistence) or a ``MemoryStoreContext``
    (tests and sandbox previews); ``api_keys`` is an ``ApiKeyStore`` or any
    object exposing ``verify``. Defaults build the real ones, constructed
    lazily so nothing touches the network at import or factory time.
    """
    ctx = ctx or StoreContext(settings)
    api_keys = api_keys or ApiKeyStore(settings)
    # AnswerAgent opens no sockets until the first real completion (its OpenAI
    # clients are built on first use), so constructing it eagerly is free. It is
    # also what keeps an unconfigured deployment honest: a chat request reaches
    # ``configured=False`` and answers 503 instead of misusing a retrieval model.
    answer_agent = answer_agent or AnswerAgent(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # The embedder loads weights lazily, so an unwarmed API would hand the
        # buyer's very first query a cold-model latency and call that a
        # response. Pay it at startup instead. A load failure is logged, not
        # fatal: list and export never touch the model, so a bad weights mount
        # must not take every read surface down with it. In-memory contexts use
        # the fake embedder, which has no warm and is skipped by getattr.
        warm = getattr(getattr(ctx, "embedder", None), "warm", None)
        if warm is not None:
            try:
                await warm()
            except Exception:
                log.exception("embedding_model_warmup_failed")
        yield
        await ctx.aclose()
        await api_keys.aclose()
        # The agent owns its httpx pool unless one was injected (tests inject a
        # fake or an owned client); aclose() is a no-op in both test cases.
        close_agent = getattr(answer_agent, "aclose", None)
        if close_agent is not None:
            await close_agent()

    app = FastAPI(
        title="Know Everything Query API",
        version=_APP_VERSION,
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.ctx = ctx
    app.state.api_keys = api_keys
    app.state.answer_agent = answer_agent

    app.include_router(health.router)
    app.include_router(query.router)
    app.include_router(chat.router)
    return app

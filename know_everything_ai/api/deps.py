"""Shared dependencies: authentication and store access.

Everything is reached through ``app.state`` rather than built here, so tests
can build the app with a fake key store and in-memory persistence simply by
injecting them into ``create_app`` — no monkeypatching.
"""

from __future__ import annotations

import structlog
from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

log = structlog.get_logger("query_api")

_bearer = HTTPBearer(auto_error=False)


async def require_api_key(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> None:
    """401 unless the request carries a live bearer key.

    ``auto_error=False`` on the bearer so a missing or malformed header is our
    decision to shape, not FastAPI's automatic 403.
    """
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=401,
            detail="missing or malformed bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    store = request.app.state.api_keys
    if store is None or not await store.verify(credentials.credentials):
        log.warning("api_key_rejected")
        raise HTTPException(
            status_code=401,
            detail="invalid or revoked API key",
            headers={"WWW-Authenticate": "Bearer"},
        )

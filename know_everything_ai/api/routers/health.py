"""Liveness and readiness, unauthenticated on purpose.

Monitoring probes carry no key: a healthy deployment must be able to answer
them even when the caller's credential machinery is itself the failing part.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import Response

router = APIRouter()


@router.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/readyz")
async def readyz(request: Request) -> Response:
    """Ready only when the database answers.

    The app starts before the pool touches the network — the pool opens lazily
    on first use — so ``/healthz`` alone would report healthy against a
    Postgres that has been down since boot. One round trip proves the schema
    the endpoints read is reachable.
    """
    registry = request.app.state.ctx.registry
    try:
        await registry.count()
    except Exception:
        return Response(status_code=503, content='{"status":"not_ready"}')
    return Response(status_code=200, content='{"status":"ready"}')

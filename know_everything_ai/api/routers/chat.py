"""Streaming chat surface of the query API.

The response is a Server-Sent-Events stream: ``sources`` first (everything the
model was allowed to read), then one ``delta`` per token chunk, then ``done``.
A mid-stream model failure raises an ``error`` event instead of tearing the
connection down, so the client can present the partial answer together with a
retry affordance rather than a generic network failure.

Retrieval deliberately happens before the ``200``/stream: a missing knowledge
base or an in-flight one is an HTTP status (404/409) that the client's fetch
caller can reject normally, not an event it has to interpret.
"""

from __future__ import annotations

import json
from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from know_everything_ai.agent.answer import stream_answer_events
from know_everything_ai.api.deps import require_api_key
from know_everything_ai.api.models import ChatRequest
from know_everything_ai.api.services import KbNotFoundError, KbNotReadyError, QueryService

log = structlog.get_logger("query_api.chat")

router = APIRouter(prefix="/v1", dependencies=[Depends(require_api_key)])

_NO_BEHAVIOR_IGNORED = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def _sse(kind: str, payload: dict[str, Any]) -> str:
    return f"event: {kind}\ndata: {json.dumps(payload, ensure_ascii=False, default=str)}\n\n"


@router.post("/knowledge-bases/{kb_external_id}/chat", response_class=StreamingResponse)
async def chat(
    kb_external_id: str,
    payload: ChatRequest,
    request: Request,
) -> StreamingResponse:
    state = request.app.state
    agent = state.answer_agent
    if agent is None or not agent.configured:
        raise HTTPException(
            status_code=503,
            detail={
                "message": "answer agent is not configured",
                "hint": "set ANSWER_API_URL and ANSWER_MODEL_NAME, then restart",
            },
        )

    service = QueryService(state.settings, state.ctx, state.ctx.embedder)
    # The strictest of the two caps wins: a request that asks for 40 fragments
    # must not bloat the model prompt past the configured budget.
    limit = min(payload.limit, agent.max_fragments)
    try:
        row, hits, took_ms, floor = await service.query(
            kb_external_id,
            payload.question,
            limit,
            payload.min_score,
        )
    except KbNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except KbNotReadyError as exc:
        raise HTTPException(
            status_code=409,
            detail={
                "kb_external_id": kb_external_id,
                "status": exc.status,
                "reason": exc.reason,
                "message": "knowledge base is not ready to query",
                "detail": f"/v1/knowledge-bases/{kb_external_id}",
            },
        ) from exc

    log.info("chat_started", kb_external_id=kb_external_id, hits=len(hits))

    async def event_stream():
        try:
            async for kind, event_payload in stream_answer_events(
                agent=agent,
                question=payload.question,
                hits=hits,
                row=row,
                floor=floor,
            ):
                event_payload = dict(event_payload)
                event_payload["retrieval_took_ms"] = took_ms
                yield _sse(kind, event_payload)
        except Exception as exc:  # noqa: BLE001 - a broken client is one request
            log.warning("chat_stream_failed", error=f"{type(exc).__name__}: {exc}")

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers=_NO_BEHAVIOR_IGNORED,
    )

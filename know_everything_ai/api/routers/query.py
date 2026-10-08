"""Read surfaces of the query API.

Thin shells around QueryService — clamping, floor and the status gate all live
there; these handlers only translate between HTTP and domain errors, which is
the border where a 404 and a 409 belong.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request

from know_everything_ai.api.deps import require_api_key
from know_everything_ai.api.models import (
    ExportResponse,
    KbListResponse,
    KbSummary,
    QueryRequest,
    QueryResponse,
    SearchHitOut,
)
from know_everything_ai.api.services import KbNotFoundError, KbNotReadyError, QueryService
from know_everything_ai.stores.base import SearchHit

router = APIRouter(prefix="/v1", dependencies=[Depends(require_api_key)])


def _service(request: Request) -> QueryService:
    state = request.app.state
    return QueryService(state.settings, state.ctx, state.ctx.embedder)


@router.get(
    "/knowledge-bases",
    response_model=KbListResponse,
)
async def list_knowledge_bases(
    request: Request,
    limit: int = 50,
    offset: int = 0,
) -> KbListResponse:
    service = _service(request)
    rows, total = await service.list_all(limit, offset)
    return KbListResponse(
        items=[_summary(row) for row in rows],
        total=total,
        # Reflect what was actually answered, not what was asked.
        limit=service._clamp_limit(limit),
        offset=max(0, offset),
    )


@router.get(
    "/knowledge-bases/{kb_external_id}",
    response_model=KbSummary,
)
async def get_knowledge_base(
    kb_external_id: str, request: Request
) -> KbSummary:
    service = _service(request)
    try:
        row = await service.get(kb_external_id)
    except KbNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _summary(row)


@router.post(
    "/knowledge-bases/{kb_external_id}/query",
    response_model=QueryResponse,
)
async def query_knowledge_base(
    kb_external_id: str,
    payload: QueryRequest,
    request: Request,
) -> QueryResponse:
    service = _service(request)
    try:
        row, hits, took_ms, floor = await service.query(
            kb_external_id,
            payload.query,
            payload.limit,
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
    return QueryResponse(
        query=payload.query,
        kb_external_id=kb_external_id,
        kb_id=int(row["id"]),
        branch=row.get("branch", ""),
        status=row.get("status", "ok"),
        total_tokens=row.get("total_tokens"),
        canvas_tokens=row.get("canvas_tokens"),
        score_floor=floor,
        hit_count=len(hits),
        took_ms=took_ms,
        model=service.model,
        hits=[_hit(hit) for hit in hits],
    )


@router.get(
    "/knowledge-bases/{kb_external_id}/export",
    response_model=ExportResponse,
)
async def export_knowledge_base(
    kb_external_id: str, request: Request
) -> ExportResponse:
    service = _service(request)
    try:
        row, chunks = await service.export(kb_external_id)
    except KbNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except KbNotReadyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return ExportResponse(
        kb_external_id=kb_external_id,
        kb_id=int(row["id"]),
        branch=row.get("branch", ""),
        status=row.get("status", "ok"),
        item_count=int(row.get("item_count", 0)),
        chunk_count=len(chunks),
        chunks=[_hit(hit) for hit in chunks],
    )


def _summary(row: dict) -> KbSummary:
    """A registry row to its public shape.

    ``row`` carries extra keys (canvas in some paths) that ResponseModel would
    silently drop; building the model explicitly keeps the contract visible
    and stops a stray key from creeping into a payload later.
    """
    return KbSummary(
        id=int(row["id"]),
        kb_external_id=row.get("kb_external_id", ""),
        kb_type=row.get("kb_type", ""),
        branch=row.get("branch", ""),
        status=row.get("status", ""),
        reason=row.get("reason"),
        item_count=int(row.get("item_count", 0)),
        total_tokens=row.get("total_tokens"),
        canvas_tokens=row.get("canvas_tokens"),
        document_store_id=row.get("document_store_id"),
        cost=dict(row.get("cost") or {}),
        chunk_count=int(row.get("chunk_count", 0)),
        created_at=row.get("created_at"),
        updated_at=row.get("updated_at"),
    )


def _hit(hit: SearchHit) -> SearchHitOut:
    return SearchHitOut(
        chunk_id=int(hit.chunk_id),
        content=hit.content,
        score=float(hit.score),
        category=hit.category,
        source=hit.source,
        title=hit.title,
        metadata=dict(hit.metadata or {}),
    )

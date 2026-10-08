"""Request and response types for the query API.

Kept out of schemas.py: that module describes what the pipeline *produces*
upstream, while these describe what the read side *answers* with. Mixing the
two couples the ingest wire format to an HTTP surface that has a different
audience and a different stability contract.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    # No upper bound here on purpose: an oversized value is clamped to
    # QUERY_MAX_LIMIT rather than rejected, so a caller that asks for 1000
    # hits gets the best ones instead of a 422.
    limit: int = Field(default=8, ge=1)
    # None defers to QUERY_MIN_SCORE, keeping one server-side default that a
    # particular client can tighten without redeploying anything.
    min_score: float | None = None


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    # Also capped by ANSWER_MAX_FRAGMENTS server-side: an answer is only as
    # good as the strictest of the two limits, and the prompt stays small.
    limit: int = Field(default=6, ge=1)
    min_score: float | None = None


class SearchHitOut(BaseModel):
    """One retrieved fragment. ``chunk_id`` is the chunks-table id so a caller
    can re-fetch or cross-reference the exact row that matched."""

    chunk_id: int
    content: str
    score: float
    category: str | None = None
    source: str | None = None
    title: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class KbSummary(BaseModel):
    id: int
    kb_external_id: str
    kb_type: str
    branch: str
    status: str
    reason: str | None = None
    item_count: int
    total_tokens: int | None = None
    canvas_tokens: int | None = None
    document_store_id: str | None = None
    cost: dict[str, Any] = Field(default_factory=dict)
    chunk_count: int = 0
    created_at: datetime | None = None
    updated_at: datetime | None = None


class QueryResponse(BaseModel):
    query: str
    kb_external_id: str
    kb_id: int
    branch: str
    status: str
    total_tokens: int | None = None
    canvas_tokens: int | None = None
    # The score floor that was actually applied, i.e. the request override or
    # the server default. Lets a caller see why near-zero hits disappeared in
    # its own client before asking questions.
    score_floor: float
    hit_count: int
    took_ms: int
    model: str
    hits: list[SearchHitOut] = Field(default_factory=list)


class KbListResponse(BaseModel):
    items: list[KbSummary]
    total: int
    limit: int
    offset: int


class ExportResponse(BaseModel):
    kb_external_id: str
    kb_id: int
    branch: str
    status: str
    item_count: int
    chunk_count: int
    chunks: list[SearchHitOut]

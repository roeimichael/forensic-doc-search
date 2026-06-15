"""API request/response models — the JSON contract (real, so the shape is fixed).

Response shape required by the assignment:
    {"results": [{"chunk_id": "...", "score": 0.91, "text": "...", "metadata": {...}}]}
All three search endpoints return :class:`SearchResponse` for a uniform contract.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class SearchRequest(BaseModel):
    """POST /search body. Inputs are validated before they reach the model/store."""

    query: str = Field(min_length=1, max_length=2048)
    top_k: int = Field(default=5, ge=1, le=100)

    @field_validator("query")
    @classmethod
    def _non_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("query must not be empty or whitespace")
        return v


class FilteredRequest(SearchRequest):
    """POST /search/filtered body. ``filters`` maps metadata field → value.

    e.g. {"doc_type": "witness_statement", "date": "2024-01-15"}. A ``date`` given
    as a ``{"gte": ..., "lte": ...}`` object is treated as a range (T3.2).
    """

    filters: dict[str, Any] = Field(default_factory=dict)


class HybridRequest(SearchRequest):
    """POST /search/hybrid body (dense + BM25, RRF-fused)."""

    filters: dict[str, Any] = Field(default_factory=dict)


class SearchResultItem(BaseModel):
    """One result row."""

    chunk_id: str
    score: float
    text: str
    metadata: dict[str, Any]


class SearchResponse(BaseModel):
    """Uniform response envelope for all search endpoints."""

    results: list[SearchResultItem]


class HealthResponse(BaseModel):
    """GET /health payload. Never raises — reports degraded state instead."""

    status: str
    collection: str
    document_count: int | None   # distinct source documents (None if unavailable)
    chunk_count: int             # stored points == chunks
    embedding_model: str

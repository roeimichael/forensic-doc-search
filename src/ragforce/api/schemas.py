"""API request/response models — the JSON contract (real, so the shape is fixed).

Response shape required by the assignment:
    {"results": [{"chunk_id": "...", "score": 0.91, "text": "...", "metadata": {...}}]}
All three search endpoints return :class:`SearchResponse` for a uniform contract.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    """POST /search body."""

    query: str
    top_k: int = 5


class FilteredRequest(BaseModel):
    """POST /search/filtered body. ``filters`` maps metadata field → value.

    e.g. {"doc_type": "witness_statement", "date": "2024-01-15"}. A ``date`` given
    as a ``{"gte": ..., "lte": ...}`` object is treated as a range (T3.2).
    """

    query: str
    filters: dict[str, Any] = Field(default_factory=dict)
    top_k: int = 5


class HybridRequest(BaseModel):
    """POST /search/hybrid body (dense + BM25, RRF-fused)."""

    query: str
    filters: dict[str, Any] = Field(default_factory=dict)
    top_k: int = 5


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
    """GET /health payload."""

    status: str
    collection: str
    document_count: int
    embedding_model: str

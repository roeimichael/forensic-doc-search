"""HTTP routes (hook point — stubs this step).

Endpoints (assignment Part 3 + bonus hybrid):
    POST /search           — semantic (T3.1)
    POST /search/filtered  — semantic + metadata filter (T3.2)
    POST /search/hybrid    — dense + BM25, RRF-fused (T5.3)
    GET  /health           — store stats (T3.3)

Each route is thin: embed the query, call the matching VectorStore method, map
SearchHit → SearchResponse. Dependencies (settings/embedder/store) come from deps.py.
"""

from __future__ import annotations

from fastapi import APIRouter

from ragforce.api.schemas import (
    FilteredRequest,
    HealthResponse,
    HybridRequest,
    SearchRequest,
    SearchResponse,
)

router = APIRouter()


@router.post("/search", response_model=SearchResponse)
def search(req: SearchRequest) -> SearchResponse:
    """Semantic search. TODO(T3.1): embed_query → store.search_dense → SearchResponse."""
    raise NotImplementedError("POST /search — implemented in a later step (T3.1)")


@router.post("/search/filtered", response_model=SearchResponse)
def search_filtered(req: FilteredRequest) -> SearchResponse:
    """Filtered search. TODO(T3.2): build_filter(req.filters) → search_dense(query_filter=)."""
    raise NotImplementedError("POST /search/filtered — implemented in a later step (T3.2)")


@router.post("/search/hybrid", response_model=SearchResponse)
def search_hybrid(req: HybridRequest) -> SearchResponse:
    """Hybrid search. TODO(T5.3): dense+sparse query → store.search_hybrid (RRF)."""
    raise NotImplementedError("POST /search/hybrid — implemented in a later step (T5.3)")


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Health/stats. TODO(T3.3): store.stats() + settings.embedding.model_name."""
    raise NotImplementedError("GET /health — implemented in a later step (T3.3)")

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

from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from ragforce.api.deps import get_dense, get_settings, get_sparse, get_store
from ragforce.api.filters import build_filter
from ragforce.api.schemas import (
    FilteredRequest,
    HealthResponse,
    HybridRequest,
    SearchRequest,
    SearchResponse,
    SearchResultItem,
)
from ragforce.models import SearchHit

router = APIRouter()


def _to_response(hits: list[SearchHit]) -> SearchResponse:
    return SearchResponse(
        results=[
            SearchResultItem(chunk_id=h.chunk_id, score=h.score, text=h.text, metadata=h.metadata)
            for h in hits
        ]
    )


@router.post("/search", response_model=SearchResponse)
def search(req: SearchRequest, dense=Depends(get_dense), store=Depends(get_store)) -> SearchResponse:
    """Semantic (dense-vector) search."""
    hits = store.search_dense(dense.embed_query(req.query), top_k=req.top_k)
    return _to_response(hits)


@router.post("/search/filtered", response_model=SearchResponse)
def search_filtered(
    req: FilteredRequest, dense=Depends(get_dense), store=Depends(get_store)
) -> SearchResponse:
    """Semantic search constrained by metadata filters (doc_type / case_id / date[-range])."""
    hits = store.search_dense(
        dense.embed_query(req.query), top_k=req.top_k, query_filter=build_filter(req.filters)
    )
    return _to_response(hits)


@router.post("/search/hybrid", response_model=SearchResponse)
def search_hybrid(
    req: HybridRequest,
    dense=Depends(get_dense),
    sparse: Any = Depends(get_sparse),
    store=Depends(get_store),
) -> SearchResponse:
    """Hybrid search: dense + BM25 sparse, fused server-side via RRF."""
    if sparse is None:
        raise HTTPException(status_code=400, detail="Hybrid search is disabled (set hybrid.enabled=true).")
    hits = store.search_hybrid(
        dense.embed_query(req.query),
        sparse.embed_query(req.query),
        top_k=req.top_k,
        query_filter=build_filter(req.filters),
    )
    return _to_response(hits)


@router.get("/health", response_model=HealthResponse)
def health(store=Depends(get_store), settings=Depends(get_settings)) -> HealthResponse:
    """Store stats: document/chunk count, collection name, embedding model."""
    stats = store.stats()
    return HealthResponse(
        status=stats.get("status", "unknown"),
        collection=stats["collection"],
        document_count=stats["points_count"],
        embedding_model=settings.embedding.model_name,
    )

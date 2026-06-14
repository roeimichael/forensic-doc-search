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

from typing import Any, Callable

from fastapi import APIRouter, Depends, HTTPException
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from ragforce.api.deps import get_dense, get_settings, get_sparse, get_store
from ragforce.api.filters import FilterError, build_filter
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

# Store/transport failures we translate to 503 (service degraded) rather than a raw 500.
_STORE_ERRORS = (ResponseHandlingException, UnexpectedResponse, ConnectionError, OSError, TimeoutError)


def _filter_or_422(filters: dict[str, Any]) -> Any:
    """Build the Qdrant filter, surfacing an invalid filter as 422 (not 500)."""
    try:
        return build_filter(filters)
    except FilterError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e


def _guard(fn: Callable[[], list[SearchHit]]) -> list[SearchHit]:
    """Run a store call, mapping connection/transport failures to 503."""
    try:
        return fn()
    except _STORE_ERRORS as e:
        raise HTTPException(status_code=503, detail="vector store unavailable") from e


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
    vec = dense.embed_query(req.query)
    hits = _guard(lambda: store.search_dense(vec, top_k=req.top_k))
    return _to_response(hits)


@router.post("/search/filtered", response_model=SearchResponse)
def search_filtered(
    req: FilteredRequest, dense=Depends(get_dense), store=Depends(get_store)
) -> SearchResponse:
    """Semantic search constrained by metadata filters (doc_type / case_id / date[-range])."""
    qf = _filter_or_422(req.filters)
    vec = dense.embed_query(req.query)
    hits = _guard(lambda: store.search_dense(vec, top_k=req.top_k, query_filter=qf))
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
    qf = _filter_or_422(req.filters)
    dvec, svec = dense.embed_query(req.query), sparse.embed_query(req.query)
    hits = _guard(lambda: store.search_hybrid(dvec, svec, top_k=req.top_k, query_filter=qf))
    return _to_response(hits)


@router.get("/health", response_model=HealthResponse)
def health(store=Depends(get_store), settings=Depends(get_settings)) -> HealthResponse:
    """Store stats: chunk count, collection name, embedding model.

    Never raises — if the store is unreachable, reports ``status="unavailable"`` so
    a monitor gets a structured signal instead of a 500.
    """
    model = settings.embedding.model_name
    collection = settings.qdrant.collection
    try:
        stats = store.stats()
    except Exception:  # noqa: BLE001 — health must never propagate
        return HealthResponse(
            status="unavailable", collection=collection, chunk_count=0, embedding_model=model
        )
    return HealthResponse(
        status=stats.get("status", "unknown"),
        collection=stats["collection"],
        chunk_count=stats["points_count"],
        embedding_model=model,
    )

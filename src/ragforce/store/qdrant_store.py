"""The only class that touches Qdrant. Ingestion writes; API/eval read.

Concentrating all DB access here means swapping the backing store later changes
exactly one file, and the rest of the system keeps speaking in Documents/Chunks/
SearchHits.
"""

from __future__ import annotations

from typing import Any

from qdrant_client import QdrantClient, models

from ragforce.models import SearchHit
from ragforce.store.points import to_sparse_vector
from ragforce.store.schema import PAYLOAD_INDEXES, sparse_vectors_config, vectors_config


def _to_hit(point: Any) -> SearchHit:
    """Map a Qdrant ScoredPoint to our SearchHit (text split out of metadata)."""
    payload = dict(point.payload or {})
    text = payload.pop("text", "")
    return SearchHit(
        chunk_id=str(payload.get("chunk_id", point.id)),
        score=float(point.score),
        text=text,
        metadata=payload,
    )


class VectorStore:
    """Wrapper over ``qdrant_client``: collection mgmt, upsert, and search.

    The only class that touches Qdrant — ingestion writes through it, the API and
    eval read through it.
    """

    def __init__(
        self,
        *,
        host: str,
        port: int,
        collection: str,
        dense_vector_name: str = "dense",
        sparse_vector_name: str = "sparse",
        timeout: float = 30.0,
    ) -> None:
        self._client = QdrantClient(host=host, port=port, timeout=timeout)
        self._collection = collection
        self._dense = dense_vector_name
        self._sparse = sparse_vector_name

    # ── write path (ingestion) ──────────────────────────────────────────────
    def ensure_collection(self, *, dim: int, recreate: bool = False) -> None:
        """Create the collection (named dense+sparse vectors) + payload indexes if absent."""
        exists = self._client.collection_exists(self._collection)
        if exists and recreate:
            self._client.delete_collection(self._collection)
            exists = False
        if not exists:
            self._client.create_collection(
                self._collection,
                vectors_config=vectors_config(dim),
                sparse_vectors_config=sparse_vectors_config(),
            )
            for field_name, schema in PAYLOAD_INDEXES:
                self._client.create_payload_index(
                    self._collection, field_name=field_name, field_schema=schema
                )

    def upsert(self, points: list[Any]) -> None:
        """Idempotently upsert a batch of points (overwrites by id)."""
        self._client.upsert(self._collection, points=points)

    def count(self) -> int:
        """Number of points currently in the collection."""
        return self._client.count(self._collection).count

    def stats(self) -> dict[str, Any]:
        """Store stats for ``GET /health`` (collection, points_count, vector dim)."""
        info = self._client.get_collection(self._collection)
        dense_params = (info.config.params.vectors or {}).get(self._dense)
        return {
            "collection": self._collection,
            "points_count": self.count(),
            "vector_dim": getattr(dense_params, "size", None),
            "status": str(info.status),
        }

    # ── read path (API / eval hooks) ────────────────────────────────────────
    def search_dense(
        self, query_vec: list[float], *, top_k: int, query_filter: Any | None = None
    ) -> list[SearchHit]:
        """Semantic search with optional metadata pre-filter → SearchHit list."""
        res = self._client.query_points(
            self._collection,
            query=query_vec,
            using=self._dense,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )
        return [_to_hit(p) for p in res.points]

    def search_hybrid(
        self,
        dense_vec: list[float],
        sparse_vec: Any,
        *,
        top_k: int,
        query_filter: Any | None = None,
    ) -> list[SearchHit]:
        """Hybrid search: dense + BM25 sparse prefetch, fused server-side via RRF."""
        prefetch_limit = max(top_k * 4, 20)
        res = self._client.query_points(
            self._collection,
            prefetch=[
                models.Prefetch(
                    query=dense_vec, using=self._dense, limit=prefetch_limit, filter=query_filter
                ),
                models.Prefetch(
                    query=to_sparse_vector(sparse_vec),
                    using=self._sparse,
                    limit=prefetch_limit,
                    filter=query_filter,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k,
            with_payload=True,
        )
        return [_to_hit(p) for p in res.points]

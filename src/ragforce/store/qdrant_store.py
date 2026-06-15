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
        hnsw_m: int = 16,
        hnsw_ef_construct: int = 128,
        hnsw_ef_search: int | None = None,
        quantization: bool = False,
        prefetch_multiplier: int = 10,
        prefetch_min: int = 50,
    ) -> None:
        self._client = QdrantClient(host=host, port=port, timeout=timeout)
        self._collection = collection
        self._dense = dense_vector_name
        self._sparse = sparse_vector_name
        self._hnsw_m = hnsw_m
        self._hnsw_ef_construct = hnsw_ef_construct
        self._hnsw_ef_search = hnsw_ef_search
        self._quantization = quantization
        self._prefetch_multiplier = prefetch_multiplier
        self._prefetch_min = prefetch_min

    def _search_params(self) -> Any | None:
        return models.SearchParams(hnsw_ef=self._hnsw_ef_search) if self._hnsw_ef_search else None

    def _existing_dense_dim(self) -> int | None:
        info = self._client.get_collection(self._collection)
        params = (info.config.params.vectors or {}).get(self._dense)
        return getattr(params, "size", None)

    # ── write path (ingestion) ──────────────────────────────────────────────
    def ensure_collection(self, *, dim: int, recreate: bool = False) -> None:
        """Create the collection (named dense+sparse vectors) + payload indexes if absent.

        If the collection already exists and we're not recreating, the stored dense
        dimension is validated against ``dim`` — a silent mismatch (e.g. a model swap)
        would corrupt every upsert, so we fail loudly instead.
        """
        exists = self._client.collection_exists(self._collection)
        if exists and recreate:
            self._client.delete_collection(self._collection)
            exists = False
        if exists:
            existing = self._existing_dense_dim()
            if existing is not None and existing != dim:
                raise ValueError(
                    f"collection '{self._collection}' has dense dim {existing} but the model "
                    f"produces {dim}. Recreate the collection (ingest --recreate) to change dim."
                )
            return
        quant = (
            models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(type=models.ScalarType.INT8, always_ram=True)
            )
            if self._quantization
            else None
        )
        self._client.create_collection(
            self._collection,
            vectors_config=vectors_config(dim),
            sparse_vectors_config=sparse_vectors_config(),
            hnsw_config=models.HnswConfigDiff(m=self._hnsw_m, ef_construct=self._hnsw_ef_construct),
            quantization_config=quant,
        )
        for field_name, schema in PAYLOAD_INDEXES:
            self._client.create_payload_index(
                self._collection, field_name=field_name, field_schema=schema
            )

    def upsert(self, points: list[Any]) -> None:
        """Idempotently upsert a batch of points (overwrites by id). Waits for commit."""
        self._client.upsert(self._collection, points=points, wait=True)

    def delete_by_sources(self, source_files: list[str]) -> None:
        """Delete all points belonging to the given source files (idempotency sweep).

        Run before re-upserting a document's chunks so that if the document now yields
        *fewer* chunks, the stale trailing points don't linger as orphans.
        """
        if not source_files:
            return
        self._client.delete(
            self._collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[models.FieldCondition(key="source_file", match=models.MatchAny(any=source_files))]
                )
            ),
            wait=True,
        )

    def count(self) -> int:
        """Number of points currently in the collection."""
        return self._client.count(self._collection).count

    def stats(self) -> dict[str, Any]:
        """Store stats for ``GET /health`` (collection, doc/chunk counts, vector dim)."""
        info = self._client.get_collection(self._collection)
        dense_params = (info.config.params.vectors or {}).get(self._dense)
        try:
            # distinct source documents (chunks share a source_file); needs the keyword index
            facet = self._client.facet(self._collection, key="source_file", limit=1_000_000)
            documents = len(facet.hits)
        except Exception:  # noqa: BLE001 — facet is best-effort; never break /health
            documents = None
        return {
            "collection": self._collection,
            "documents_count": documents,
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
            search_params=self._search_params(),
        )
        return [_to_hit(p) for p in res.points]

    def search_sparse(
        self, sparse_vec: Any, *, top_k: int, query_filter: Any | None = None
    ) -> list[SearchHit]:
        """BM25-only (sparse) search — the lexical baseline for eval ablations."""
        res = self._client.query_points(
            self._collection,
            query=to_sparse_vector(sparse_vec),
            using=self._sparse,
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
        """Hybrid search: dense + BM25 sparse prefetch, fused server-side via RRF.

        Each branch fetches ``max(top_k * prefetch_multiplier, prefetch_min)`` candidates
        before fusion — deep enough that rare-token lexical hits aren't truncated away.
        """
        prefetch_limit = max(top_k * self._prefetch_multiplier, self._prefetch_min)
        res = self._client.query_points(
            self._collection,
            prefetch=[
                models.Prefetch(
                    query=dense_vec, using=self._dense, limit=prefetch_limit, filter=query_filter,
                    params=self._search_params(),
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

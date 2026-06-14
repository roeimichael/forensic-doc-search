"""The only class that touches Qdrant. Ingestion writes; API/eval read.

Concentrating all DB access here means swapping the backing store later changes
exactly one file, and the rest of the system keeps speaking in Documents/Chunks/
SearchHits.
"""

from __future__ import annotations

from typing import Any

from ragforce.models import SearchHit


class VectorStore:
    """Wrapper over ``qdrant_client``: collection mgmt, upsert, and search."""

    def __init__(
        self,
        *,
        host: str,
        port: int,
        collection: str,
        dense_vector_name: str = "dense",
        sparse_vector_name: str = "sparse",
    ) -> None:
        """Connect the client and record collection/vector names.

        TODO(T2.1): ``QdrantClient(host=host, port=port)``; stash names.
        """
        raise NotImplementedError("VectorStore.__init__ — implemented in the next step (T2.1)")

    # ── write path (ingestion) ──────────────────────────────────────────────
    def ensure_collection(self, *, dim: int, recreate: bool = False) -> None:
        """Create the collection (named dense+sparse vectors) + payload indexes if absent.

        TODO(T2.2): create_collection with vectors_config(dim)+sparse_vectors_config();
        create_payload_index for each entry in PAYLOAD_INDEXES; honor ``recreate``.
        """
        raise NotImplementedError("ensure_collection — implemented in the next step (T2.2)")

    def upsert(self, points: list[Any]) -> None:
        """Idempotently upsert a batch of points (overwrites by id).

        TODO(T1.5): ``client.upsert(collection, points)``.
        """
        raise NotImplementedError("upsert — implemented in the next step (T1.5)")

    def count(self) -> int:
        """Number of points currently in the collection."""
        raise NotImplementedError("count — implemented in the next step (T2.1)")

    def stats(self) -> dict[str, Any]:
        """Store stats for ``GET /health`` (collection, points_count, vector dim)."""
        raise NotImplementedError("stats — implemented in a later step (T3.3)")

    # ── read path (API / eval hooks) ────────────────────────────────────────
    def search_dense(
        self, query_vec: list[float], *, top_k: int, query_filter: Any | None = None
    ) -> list[SearchHit]:
        """Semantic search with optional metadata pre-filter → SearchHit list.

        TODO(T3.1/T3.2): ``client.query_points`` on the dense vector with ``query_filter``.
        """
        raise NotImplementedError("search_dense — implemented in a later step (T3.1)")

    def search_hybrid(
        self,
        dense_vec: list[float],
        sparse_vec: Any,
        *,
        top_k: int,
        query_filter: Any | None = None,
    ) -> list[SearchHit]:
        """Hybrid search: dense + sparse prefetch fused server-side via RRF.

        TODO(T5.2/T5.3): ``client.query_points`` with two Prefetch branches +
        FusionQuery(RRF) + ``query_filter``.
        """
        raise NotImplementedError("search_hybrid — implemented in a later step (T5.x)")

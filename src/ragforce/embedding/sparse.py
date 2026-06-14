"""Sparse BM25 embeddings via fastembed — feeds Qdrant's native hybrid search (T5).

Using fastembed's ``Qdrant/bm25`` keeps BM25 *inside* the vector store (a named
sparse vector) rather than maintaining a separate in-process index, so dense and
sparse stay in sync and fusion (RRF) runs server-side.
"""

from __future__ import annotations

from typing import Any

from fastembed import SparseTextEmbedding


class SparseEmbedder:
    """Produce BM25 sparse vectors (fastembed) for passages and queries.

    Returns fastembed ``SparseEmbedding`` objects (``.indices`` / ``.values``);
    :func:`ragforce.store.points.to_sparse_vector` converts them to Qdrant's format.
    """

    def __init__(
        self,
        model_name: str = "Qdrant/bm25",
        *,
        avg_len: float = 400.0,
        cache_dir: str | None = None,
        local_files_only: bool = False,
    ) -> None:
        # avg_len MUST track the chunk size — fastembed's default (256) mis-normalizes
        # BM25 term frequencies for our 400-token chunks across the whole corpus.
        self._model = SparseTextEmbedding(
            model_name=model_name, avg_len=avg_len,
            cache_dir=cache_dir, local_files_only=local_files_only,
        )

    def embed_passages(self, texts: list[str], *, batch_size: int = 64) -> list[Any]:
        """Embed chunks into BM25 sparse vectors (row-aligned with the dense batch)."""
        return list(self._model.embed(texts, batch_size=batch_size))

    def embed_query(self, text: str) -> Any:
        """Embed a single query into a BM25 sparse vector."""
        return next(iter(self._model.query_embed(text)))

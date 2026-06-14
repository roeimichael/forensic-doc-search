"""Sparse BM25 embeddings via fastembed — feeds Qdrant's native hybrid search (T5).

Using fastembed's ``Qdrant/bm25`` keeps BM25 *inside* the vector store (a named
sparse vector) rather than maintaining a separate in-process index, so dense and
sparse stay in sync and fusion (RRF) runs server-side.
"""

from __future__ import annotations

from typing import Any


class SparseEmbedder:
    """Produce BM25 sparse vectors for passages and queries."""

    def __init__(self, model_name: str = "Qdrant/bm25") -> None:
        """Initialize the fastembed sparse model.

        TODO(T5.1): ``fastembed.SparseTextEmbedding(model_name)``.
        """
        raise NotImplementedError("SparseEmbedder.__init__ — implemented in a later step (T5.1)")

    def embed_passages(self, texts: list[str], *, batch_size: int = 64) -> list[Any]:
        """Embed chunks into sparse vectors (row-aligned with the dense batch).

        TODO(T5.1): ``list(model.embed(texts))`` → (indices, values) sparse vectors.
        """
        raise NotImplementedError("SparseEmbedder.embed_passages — later step (T5.1)")

    def embed_query(self, text: str) -> Any:
        """Embed a single query into a sparse vector.

        TODO(T5.1): ``next(model.query_embed(text))``.
        """
        raise NotImplementedError("SparseEmbedder.embed_query — later step (T5.1)")

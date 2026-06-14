"""Embedding package — dense (always) + sparse (only when hybrid is enabled)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ragforce.embedding.dense import DenseEmbedder
from ragforce.embedding.sparse import SparseEmbedder

if TYPE_CHECKING:
    from ragforce.config import Settings

__all__ = ["DenseEmbedder", "SparseEmbedder", "build_embedder"]


def build_embedder(settings: "Settings") -> tuple[DenseEmbedder, SparseEmbedder | None]:
    """Construct the dense embedder (+ sparse iff ``settings.hybrid.enabled``).

    Ingestion stays valid with hybrid off (sparse is ``None``); the dense path is
    unaffected. The model's reported dimension is checked against config so a model
    swap that changes ``dim`` is caught loudly rather than corrupting the collection.
    """
    from ragforce.logging_setup import get_logger

    e = settings.embedding
    dense = DenseEmbedder(
        e.model_name,
        device=e.device,
        query_prefix=e.query_prefix,
        passage_prefix=e.passage_prefix,
        normalize=e.normalize,
        max_seq_length=e.max_seq_length,
    )
    if dense.dim != e.dim:
        get_logger("embedding").warning(
            "config embedding.dim=%d but %s reports %d; using the model's value",
            e.dim, e.model_name, dense.dim,
        )
    sparse = SparseEmbedder(settings.hybrid.sparse_model) if settings.hybrid.enabled else None
    return dense, sparse

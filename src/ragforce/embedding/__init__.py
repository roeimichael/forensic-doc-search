"""Embedding package — dense (always) + sparse (only when hybrid is enabled)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ragforce.embedding.dense import DenseEmbedder
from ragforce.embedding.rerank import Reranker, build_reranker
from ragforce.embedding.sparse import SparseEmbedder

if TYPE_CHECKING:
    from ragforce.config import Settings

__all__ = ["DenseEmbedder", "SparseEmbedder", "Reranker", "build_embedder", "build_reranker"]


def build_embedder(settings: "Settings") -> tuple[DenseEmbedder, SparseEmbedder | None]:
    """Construct the dense embedder (+ sparse iff ``settings.hybrid.enabled``).

    Ingestion stays valid with hybrid off (sparse is ``None``); the dense path is
    unaffected. A ``dim`` mismatch against config is a fatal error (it would corrupt
    the collection), so we raise rather than warn. We also warn if the configured
    ``chunk_size`` leaves no room for the model's special tokens (silent truncation).
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
        cache_folder=e.models_dir,
        local_files_only=e.local_files_only,
    )
    if dense.dim != e.dim:
        raise ValueError(
            f"embedding.dim={e.dim} but {e.model_name} reports {dense.dim}. "
            f"Update config.embedding.dim (and recreate the collection) before ingesting."
        )

    # Truncation guard: chunks are sized with add_special_tokens=False, but the model
    # adds [CLS]/[SEP] at embed time. If chunk_size eats the whole window, text is
    # silently truncated — warn loudly so a model/config swap can't hide it.
    specials = len(dense.tokenizer.encode("", add_special_tokens=True))
    budget = dense.max_seq_length - specials
    if settings.chunking.chunk_size > budget:
        get_logger("embedding").warning(
            "chunk_size=%d exceeds the embed budget %d (max_seq=%d - %d special tokens); "
            "chunks may be truncated. Lower chunking.chunk_size.",
            settings.chunking.chunk_size, budget, dense.max_seq_length, specials,
        )

    sparse = (
        SparseEmbedder(
            settings.hybrid.sparse_model,
            avg_len=float(settings.hybrid.avg_len),
            cache_dir=e.models_dir,
            local_files_only=e.local_files_only,
        )
        if settings.hybrid.enabled
        else None
    )
    return dense, sparse

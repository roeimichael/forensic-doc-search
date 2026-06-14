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
    unaffected.

    TODO(T1.3/T5.1): build DenseEmbedder from settings.embedding; build
    SparseEmbedder(settings.hybrid.sparse_model) only when hybrid is enabled.
    """
    raise NotImplementedError("build_embedder — implemented in the next step (T1.3)")

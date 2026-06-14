"""FastAPI dependency providers (hook point — stubs this step).

These hand the singletons built in the app lifespan to the routes, so handlers
never construct a model or client per request.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ragforce.config import Settings
    from ragforce.embedding import DenseEmbedder, SparseEmbedder
    from ragforce.store import VectorStore


def get_settings() -> "Settings":
    """Provide the loaded Settings. TODO(T3): return app.state.settings."""
    raise NotImplementedError("get_settings — implemented in a later step (T3)")


def get_embedder() -> "tuple[DenseEmbedder, SparseEmbedder | None]":
    """Provide the (dense, sparse?) embedders. TODO(T3): return app.state.embedder."""
    raise NotImplementedError("get_embedder — implemented in a later step (T3)")


def get_store() -> "VectorStore":
    """Provide the VectorStore. TODO(T3): return app.state.store."""
    raise NotImplementedError("get_store — implemented in a later step (T3)")

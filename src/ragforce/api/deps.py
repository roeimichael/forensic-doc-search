"""FastAPI dependency providers (hook point — stubs this step).

These hand the singletons built in the app lifespan to the routes, so handlers
never construct a model or client per request.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import Request

if TYPE_CHECKING:
    from ragforce.config import Settings
    from ragforce.embedding import DenseEmbedder
    from ragforce.store import VectorStore


def get_settings(request: Request) -> "Settings":
    """The loaded Settings (built once in the app lifespan)."""
    return request.app.state.settings


def get_dense(request: Request) -> "DenseEmbedder":
    """The dense embedder singleton."""
    return request.app.state.dense


def get_sparse(request: Request) -> Any | None:
    """The sparse embedder singleton (``None`` if hybrid is disabled)."""
    return request.app.state.sparse


def get_store(request: Request) -> "VectorStore":
    """The VectorStore singleton."""
    return request.app.state.store

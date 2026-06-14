"""FastAPI application factory (serving layer — hook point, stub this step).

In the implementation step this builds the app with a lifespan that loads
``Settings`` + the embedder + the ``VectorStore`` exactly once, then includes the
routes. The Makefile/uvicorn target is ``ragforce.api.app:app`` — the module-level
``app`` is created once ``create_app`` is implemented.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from ragforce.api.routes import router
from ragforce.config import load_settings
from ragforce.embedding import build_embedder
from ragforce.logging_setup import configure_logging, get_logger
from ragforce.store import VectorStore


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load Settings + embedder + store ONCE at startup (never per request)."""
    configure_logging()
    log = get_logger("api")
    settings = load_settings()
    log.info("loading embedder %s ...", settings.embedding.model_name)
    dense, sparse = build_embedder(settings)
    store = VectorStore(
        host=settings.qdrant.host,
        port=settings.qdrant.port,
        collection=settings.qdrant.collection,
        dense_vector_name=settings.qdrant.dense_vector_name,
        sparse_vector_name=settings.qdrant.sparse_vector_name,
        timeout=settings.qdrant.timeout,
    )
    app.state.settings = settings
    app.state.dense = dense
    app.state.sparse = sparse
    app.state.store = store
    log.info("API ready (collection=%s, hybrid=%s)", settings.qdrant.collection, sparse is not None)
    yield


def create_app() -> FastAPI:
    """Build the FastAPI app with the search routes and the startup lifespan."""
    app = FastAPI(
        title="Forensic RAG Search API",
        description="On-prem semantic + metadata + hybrid search over forensic documents.",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.include_router(router)
    return app


# Uvicorn target: `uvicorn ragforce.api.app:app`. Models load in the lifespan at
# startup, so importing this module stays cheap.
app = create_app()

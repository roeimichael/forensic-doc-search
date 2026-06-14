"""FastAPI application factory (serving layer — hook point, stub this step).

In the implementation step this builds the app with a lifespan that loads
``Settings`` + the embedder + the ``VectorStore`` exactly once, then includes the
routes. The Makefile/uvicorn target is ``ragforce.api.app:app`` — the module-level
``app`` is created once ``create_app`` is implemented.
"""

from __future__ import annotations

from typing import Any


def create_app() -> Any:
    """Build and return the FastAPI app (with lifespan-loaded singletons + routes).

    TODO(T3): construct FastAPI(); attach lifespan that builds Settings/embedder/
    store; ``app.include_router(routes.router)``; return app.
    """
    raise NotImplementedError("create_app — implemented in a later step (T3)")


# Enabled in the implementation step (kept out now so importing this module is safe):
# app = create_app()

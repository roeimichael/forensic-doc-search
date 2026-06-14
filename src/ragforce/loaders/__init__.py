"""Loader registry + directory walk.

``load_directory`` is the ingestion entrypoint for raw files: it dispatches each
file to the loader registered for its extension and *logs+skips* anything
unsupported or corrupt, so a single bad document never aborts a run (T1.1).
"""

from __future__ import annotations

from collections.abc import Iterator

from ragforce.loaders.base import Loader, LoaderError
from ragforce.loaders.json_loader import JsonLoader
from ragforce.loaders.pdf_loader import PdfLoader
from ragforce.loaders.txt_loader import TxtLoader
from ragforce.models import Document

__all__ = ["Loader", "LoaderError", "get_loader", "load_directory"]

# Extension → loader instance. Add a format here and nowhere else.
_REGISTRY: dict[str, Loader] = {
    ".txt": TxtLoader(),
    ".pdf": PdfLoader(),
    ".json": JsonLoader(),
}


def get_loader(path: str) -> Loader | None:
    """Return the loader for ``path``'s extension, or ``None`` if unsupported.

    TODO(T1.1): look up ``Path(path).suffix.lower()`` in ``_REGISTRY``.
    """
    raise NotImplementedError("get_loader — implemented in the next step (T1.1)")


def load_directory(
    root: str,
    *,
    default_doc_type: str | None = None,
) -> Iterator[Document]:
    """Walk ``root`` and yield a :class:`Document` per loadable file.

    Unsupported extensions and files that raise :class:`LoaderError` are logged at
    WARNING and skipped (never raised) — this is where graceful degradation lives.

    TODO(T1.1): walk files; ``get_loader``; try ``loader.load`` / except LoaderError
    -> log+skip; yield Documents.
    """
    raise NotImplementedError("load_directory — implemented in the next step (T1.1)")

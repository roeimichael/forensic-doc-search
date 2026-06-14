"""Loader registry + directory walk.

``load_directory`` is the ingestion entrypoint for raw files: it dispatches each
file to the loader registered for its extension and *logs+skips* anything
unsupported or corrupt, so a single bad document never aborts a run (T1.1).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from ragforce.loaders.base import Loader, LoaderError
from ragforce.loaders.eml_loader import EmlLoader
from ragforce.loaders.json_loader import JsonLoader
from ragforce.loaders.pdf_loader import PdfLoader
from ragforce.loaders.txt_loader import TxtLoader
from ragforce.logging_setup import get_logger
from ragforce.models import Document

__all__ = ["Loader", "LoaderError", "get_loader", "load_directory"]

# Extension → loader instance. Add a format here and nowhere else.
_REGISTRY: dict[str, Loader] = {
    ".txt": TxtLoader(),
    ".pdf": PdfLoader(),
    ".json": JsonLoader(),
    ".eml": EmlLoader(),
}

# Non-document files that may share a supported extension (skip silently).
_SKIP_NAMES = {"ground_truth.json"}

_log = get_logger("loaders")


def get_loader(path: str) -> Loader | None:
    """Return the loader for ``path``'s extension, or ``None`` if unsupported."""
    return _REGISTRY.get(Path(path).suffix.lower())


def load_directory(
    root: str,
    *,
    default_doc_type: str | None = None,
) -> Iterator[Document]:
    """Walk ``root`` and yield a :class:`Document` per loadable file.

    Unsupported extensions and files that raise :class:`LoaderError` are logged at
    WARNING and skipped (never raised) — this is where graceful degradation lives.
    """
    for path in sorted(Path(root).iterdir()):
        if not path.is_file() or path.name in _SKIP_NAMES:
            continue
        loader = get_loader(str(path))
        if loader is None:
            _log.warning("skipping unsupported file: %s", path.name)
            continue
        try:
            yield loader.load(str(path))
        except LoaderError as e:
            _log.warning("skipping %s", e)

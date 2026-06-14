"""Plain-text (.txt) loader."""

from __future__ import annotations

from typing import ClassVar

from ragforce.models import Document


class TxtLoader:
    """Read a UTF-8 text file; derive metadata from the filename convention."""

    extensions: ClassVar[tuple[str, ...]] = (".txt",)

    def load(self, path: str) -> Document:
        """Read the file's text and resolve metadata from its filename.

        TODO(T1.1): read text (utf-8, errors='replace'); call
        ``metadata.resolve_metadata(filename=...)``; return Document(...).
        """
        raise NotImplementedError("TxtLoader.load — implemented in the next step (T1.1)")

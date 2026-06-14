"""Plain-text (.txt) loader."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from ragforce.loaders.base import LoaderError
from ragforce.loaders.metadata import resolve_metadata
from ragforce.models import Document


class TxtLoader:
    """Read a UTF-8 text file; derive metadata from the filename convention."""

    extensions: ClassVar[tuple[str, ...]] = (".txt",)

    def load(self, path: str) -> Document:
        """Read the file's text and resolve metadata from its filename."""
        p = Path(path)
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
            md = resolve_metadata(filename=p.name)
        except (OSError, ValueError) as e:
            raise LoaderError(f"{p.name}: {e}") from e
        return Document(
            source_file=p.name, text=text, doc_type=md["doc_type"], case_id=md["case_id"],
            date=md["date"], title=md.get("title"), source_path=str(p.resolve()),
        )

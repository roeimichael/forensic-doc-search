"""PDF (.pdf) loader — text extraction via pypdf."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from ragforce.loaders.base import LoaderError
from ragforce.loaders.metadata import resolve_metadata
from ragforce.models import Document


class PdfLoader:
    """Extract text per page with pypdf, join pages, derive filename metadata."""

    extensions: ClassVar[tuple[str, ...]] = (".pdf",)

    def load(self, path: str) -> Document:
        """Extract and concatenate page text into a Document."""
        from pypdf import PdfReader

        p = Path(path)
        try:
            reader = PdfReader(str(p))
            text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
            md = resolve_metadata(filename=p.name)
        except Exception as e:  # pypdf raises assorted parse errors  # noqa: BLE001
            raise LoaderError(f"{p.name}: {e}") from e
        return Document(
            source_file=p.name, text=text, doc_type=md["doc_type"], case_id=md["case_id"],
            date=md["date"], title=md.get("title"), source_path=str(p.resolve()),
        )

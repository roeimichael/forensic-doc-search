"""PDF (.pdf) loader — text extraction via pypdf."""

from __future__ import annotations

from typing import ClassVar

from ragforce.models import Document


class PdfLoader:
    """Extract text per page with pypdf, join pages, derive filename metadata."""

    extensions: ClassVar[tuple[str, ...]] = (".pdf",)

    def load(self, path: str) -> Document:
        """Extract and concatenate page text into a Document.

        TODO(T1.1): ``pypdf.PdfReader(path)``; join ``page.extract_text()`` with
        "\\n\\n"; resolve filename metadata; raise LoaderError on parse failure.
        """
        raise NotImplementedError("PdfLoader.load — implemented in the next step (T1.1)")

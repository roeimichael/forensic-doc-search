"""JSON (.json) loader — reads a required ``content`` field, prefers inline metadata."""

from __future__ import annotations

from typing import ClassVar

from ragforce.models import Document


class JsonLoader:
    """Read ``content`` plus any inline metadata (doc_type/case_id/date/title).

    Precedence for metadata: inline JSON fields > filename convention > defaults.
    """

    extensions: ClassVar[tuple[str, ...]] = (".json",)

    def load(self, path: str) -> Document:
        """Parse JSON, require a ``content`` string, merge inline + filename metadata.

        TODO(T1.1): ``json.load``; require 'content' (else LoaderError); call
        ``metadata.resolve_metadata(filename=..., inline=obj)``; return Document(...).
        """
        raise NotImplementedError("JsonLoader.load — implemented in the next step (T1.1)")

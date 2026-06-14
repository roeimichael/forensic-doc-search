"""JSON (.json) loader — reads a required ``content`` field, prefers inline metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar

from ragforce.loaders.base import LoaderError
from ragforce.loaders.metadata import resolve_metadata
from ragforce.models import Document


class JsonLoader:
    """Read ``content`` plus any inline metadata (doc_type/case_id/date/title).

    Precedence for metadata: inline JSON fields > filename convention > defaults.
    """

    extensions: ClassVar[tuple[str, ...]] = (".json",)

    def load(self, path: str) -> Document:
        """Parse JSON, require a ``content`` string, merge inline + filename metadata."""
        p = Path(path)
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
            content = obj["content"]
            if not isinstance(content, str):
                raise ValueError("'content' must be a string")
            md = resolve_metadata(filename=p.name, inline=obj)
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as e:
            raise LoaderError(f"{p.name}: {e}") from e
        return Document(
            source_file=p.name, text=content, doc_type=md["doc_type"], case_id=md["case_id"],
            date=md["date"], title=md.get("title"), source_path=str(p.resolve()),
        )

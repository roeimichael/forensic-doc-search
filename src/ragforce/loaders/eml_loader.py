"""Email (.eml) loader — extra format, on-theme for digital forensics.

Emails are first-class evidence in mobile/digital forensics. The ``email`` stdlib
parses the message; metadata comes from headers (preferred over the filename):
    Subject → title, Date → date, X-Doc-Type → doc_type, X-Case-ID → case_id.
The plain-text body becomes the Document text.
"""

from __future__ import annotations

from email import message_from_string
from email.policy import default as email_policy
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import ClassVar

from ragforce.loaders.base import LoaderError
from ragforce.loaders.metadata import resolve_metadata
from ragforce.models import Document


class EmlLoader:
    """Parse an .eml message: body → text, headers → metadata.

    Demonstrates parsing metadata *from content*: Subject/Date plus the custom
    X-Doc-Type / X-Case-ID headers, falling back to the filename when absent.
    """

    extensions: ClassVar[tuple[str, ...]] = (".eml",)

    def load(self, path: str) -> Document:
        p = Path(path)
        try:
            msg = message_from_string(p.read_text(encoding="utf-8"), policy=email_policy)
            text = msg.get_content()
            inline: dict[str, str] = {}
            if msg["X-Doc-Type"]:
                inline["doc_type"] = msg["X-Doc-Type"]
            if msg["X-Case-ID"]:
                inline["case_id"] = msg["X-Case-ID"]
            if msg["Date"]:
                inline["date"] = parsedate_to_datetime(msg["Date"]).date().isoformat()
            if msg["Subject"]:
                inline["title"] = str(msg["Subject"])
            md = resolve_metadata(filename=p.name, inline=inline)
        except (OSError, ValueError, TypeError) as e:
            raise LoaderError(f"{p.name}: {e}") from e
        return Document(
            source_file=p.name, text=text, doc_type=md["doc_type"], case_id=md["case_id"],
            date=md["date"], title=md.get("title"), source_path=str(p.resolve()),
        )

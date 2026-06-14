"""Email (.eml) loader — extra format, on-theme for digital forensics.

Emails are first-class evidence in mobile/digital forensics. The ``email`` stdlib
parses the message; metadata comes from headers (preferred over the filename):
    Subject → title, Date → date, X-Doc-Type → doc_type, X-Case-ID → case_id.
The plain-text body becomes the Document text.
"""

from __future__ import annotations

from typing import ClassVar

from ragforce.models import Document


class EmlLoader:
    """Parse an .eml message: body → text, headers → metadata."""

    extensions: ClassVar[tuple[str, ...]] = (".eml",)

    def load(self, path: str) -> Document:
        """Read the email; prefer header metadata, fall back to the filename.

        TODO(T1.1): ``email.message_from_string`` (policy=default); get the plain
        body; map Subject/Date/X-Doc-Type/X-Case-ID via metadata.resolve_metadata
        (inline=headers); normalize the Date header to ISO; return Document(...).
        """
        raise NotImplementedError("EmlLoader.load — implemented in the next step (T1.1)")

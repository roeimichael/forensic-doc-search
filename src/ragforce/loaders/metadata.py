"""Shared metadata derivation for all loaders (requirement T1.4).

Filename convention emitted by the dataset generator and parsed back here:

    <doc_type>__<case_id>__<YYYY-MM-DD>__<slug>.<ext>
    e.g.  witness_statement__2024-7812__2024-01-15__vehicle-sighting.txt

Keeping this in one module means each loader only does text extraction, and
metadata rules (precedence, date normalization) are defined exactly once.
"""

from __future__ import annotations

import datetime
from pathlib import Path

_FILENAME_KEYS = ("doc_type", "case_id", "date", "slug")
_DATE_FORMATS = ("%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%B %d, %Y", "%d %B %Y")


def parse_filename_metadata(filename: str) -> dict[str, str]:
    """Extract ``doc_type`` / ``case_id`` / ``date`` / ``slug`` from the filename.

    Convention: ``<doc_type>__<case_id>__<YYYY-MM-DD>__<slug>.<ext>``. Tolerant of
    partial matches (returns only the positional fields actually present).
    """
    parts = Path(filename).stem.split("__")
    return {key: value for key, value in zip(_FILENAME_KEYS, parts)}


def normalize_date(value: str) -> str:
    """Normalize a date string to ISO-8601 ``YYYY-MM-DD``; raise on garbage."""
    value = value.strip()
    try:
        return datetime.date.fromisoformat(value[:10]).isoformat()
    except ValueError:
        pass
    for fmt in _DATE_FORMATS:
        try:
            return datetime.datetime.strptime(value, fmt).date().isoformat()
        except ValueError:
            continue
    raise ValueError(f"unparseable date: {value!r}")


def resolve_metadata(
    *,
    filename: str,
    inline: dict | None = None,
    default_doc_type: str | None = None,
) -> dict[str, str]:
    """Merge metadata sources with precedence inline > filename > default.

    Returns a dict containing ``doc_type``, ``case_id``, ``date`` (ISO-8601) and an
    optional ``title``. Raises ``ValueError`` if a required field cannot be resolved.
    """
    inline = inline or {}
    from_name = parse_filename_metadata(filename)

    doc_type = inline.get("doc_type") or from_name.get("doc_type") or default_doc_type
    case_id = inline.get("case_id") or from_name.get("case_id")
    raw_date = inline.get("date") or from_name.get("date")

    missing = [k for k, v in (("doc_type", doc_type), ("case_id", case_id), ("date", raw_date)) if not v]
    if missing:
        raise ValueError(f"cannot resolve metadata fields {missing} for {filename!r}")

    resolved = {"doc_type": doc_type, "case_id": case_id, "date": normalize_date(raw_date)}
    if inline.get("title"):
        resolved["title"] = inline["title"]
    return resolved

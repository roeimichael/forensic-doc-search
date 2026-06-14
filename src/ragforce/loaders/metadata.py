"""Shared metadata derivation for all loaders (requirement T1.4).

Filename convention emitted by the dataset generator and parsed back here:

    <doc_type>__<case_id>__<YYYY-MM-DD>__<slug>.<ext>
    e.g.  witness_statement__2024-7812__2024-01-15__vehicle-sighting.txt

Keeping this in one module means each loader only does text extraction, and
metadata rules (precedence, date normalization) are defined exactly once.
"""

from __future__ import annotations


def parse_filename_metadata(filename: str) -> dict[str, str]:
    """Extract ``doc_type`` / ``case_id`` / ``date`` / ``slug`` from the filename.

    Tolerant of partial matches (returns only the fields it can confidently parse).

    TODO(T1.4): split basename on '__'; map the 4 positional fields; normalize date.
    """
    raise NotImplementedError("parse_filename_metadata — implemented in the next step (T1.4)")


def normalize_date(value: str) -> str:
    """Normalize any parseable date string to ISO-8601 ``YYYY-MM-DD``.

    TODO(T1.4): try a few known formats (ISO, ``DD/MM/YYYY``, etc.); raise
    ValueError on unparseable input so bad data surfaces loudly.
    """
    raise NotImplementedError("normalize_date — implemented in the next step (T1.4)")


def resolve_metadata(
    *,
    filename: str,
    inline: dict | None = None,
    default_doc_type: str | None = None,
) -> dict[str, str]:
    """Merge metadata sources with precedence: inline > filename > default.

    Returns a dict guaranteed to contain ``doc_type``, ``case_id``, ``date``
    (date normalized to ISO-8601), plus optional ``title``.

    TODO(T1.4): combine ``parse_filename_metadata`` with ``inline``; fill gaps
    from defaults; normalize the date; validate required keys present.
    """
    raise NotImplementedError("resolve_metadata — implemented in the next step (T1.4)")

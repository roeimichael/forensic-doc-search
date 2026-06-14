"""Structure-aware separator sets for recursive splitting.

Recursive splitting tries separators coarse→fine. For transcripts we keep speaker
turns intact (split on the speaker labels the generator actually emits — ``Officer``
/ ``Witness`` — not the never-present ``Q:``/``A:``) before falling back to
paragraph/sentence boundaries; other doc types use the generic ladder.

These are data/constants (not pipeline logic), so they are concrete here.
"""

from __future__ import annotations

# Generic: paragraph → line → sentence → word.
DEFAULT_SEPARATORS: list[str] = ["\n\n", "\n", ". ", " "]

# Transcripts: preserve speaker turns first, then fall back to the generic ladder.
TRANSCRIPT_SEPARATORS: list[str] = ["\nOfficer", "\nWitness", "\n\n", "\n", ". ", " "]


def separators_for(doc_type: str) -> list[str]:
    """Return the separator ladder appropriate for ``doc_type``."""
    return TRANSCRIPT_SEPARATORS if doc_type == "transcript" else DEFAULT_SEPARATORS

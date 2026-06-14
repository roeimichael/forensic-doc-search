"""Structure-aware separator sets for recursive splitting.

Recursive splitting tries separators coarse→fine. For transcripts we keep
speaker turns (``Q:`` / ``A:``) intact before falling back to paragraph/sentence
boundaries; other doc types use the generic paragraph→sentence→word ladder.

These are data/constants (not pipeline logic), so they are concrete here.
"""

from __future__ import annotations

# Generic: paragraph → line → sentence → word.
DEFAULT_SEPARATORS: list[str] = ["\n\n", "\n", ". ", " "]

# Transcripts: preserve Q:/A: turns first, then fall back to the generic ladder.
TRANSCRIPT_SEPARATORS: list[str] = ["\nQ:", "\nA:", "\n\n", "\n", ". ", " "]


def separators_for(doc_type: str) -> list[str]:
    """Return the separator ladder appropriate for ``doc_type``."""
    return TRANSCRIPT_SEPARATORS if doc_type == "transcript" else DEFAULT_SEPARATORS

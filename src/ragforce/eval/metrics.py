"""Retrieval metrics (requirement T4.2): Hit@K and MRR.

Pure functions over ranked results — no I/O — so they are trivially unit-tested
(see tests/test_metrics.py). A "hit" = the expected ``source_file`` appears among
the retrieved chunks' ``source_file`` values.
"""

from __future__ import annotations

from collections.abc import Sequence


def hit_at_k(ranked_source_files: Sequence[str], expected: str, k: int) -> bool:
    """True if ``expected`` appears in the first ``k`` retrieved source files."""
    return expected in ranked_source_files[:k]


def reciprocal_rank(ranked_source_files: Sequence[str], expected: str) -> float:
    """Reciprocal rank of the first occurrence of ``expected`` (0.0 if absent)."""
    for i, sf in enumerate(ranked_source_files):
        if sf == expected:
            return 1.0 / (i + 1)
    return 0.0


def mrr(per_query_ranks: Sequence[Sequence[str]], expected: Sequence[str]) -> float:
    """Mean Reciprocal Rank across queries."""
    if not expected:
        return 0.0
    return sum(reciprocal_rank(r, e) for r, e in zip(per_query_ranks, expected)) / len(expected)


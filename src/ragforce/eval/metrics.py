"""Retrieval metrics (requirement T4.2): Hit@K and MRR.

Pure functions over ranked results — no I/O — so they are trivially unit-tested
(see tests/test_metrics.py). A "hit" = the expected ``source_file`` appears among
the retrieved chunks' ``source_file`` values.
"""

from __future__ import annotations

from collections.abc import Sequence


def hit_at_k(ranked_source_files: Sequence[str], expected: str, k: int) -> bool:
    """True if ``expected`` appears in the first ``k`` retrieved source files.

    TODO(T4.2): ``return expected in ranked_source_files[:k]``.
    """
    raise NotImplementedError("hit_at_k — implemented in a later step (T4.2)")


def reciprocal_rank(ranked_source_files: Sequence[str], expected: str) -> float:
    """Reciprocal rank of the first occurrence of ``expected`` (0.0 if absent).

    TODO(T4.2): find first index i where match; return 1/(i+1) or 0.0.
    """
    raise NotImplementedError("reciprocal_rank — implemented in a later step (T4.2)")


def mrr(per_query_ranks: Sequence[Sequence[str]], expected: Sequence[str]) -> float:
    """Mean Reciprocal Rank across queries.

    TODO(T4.2): mean of ``reciprocal_rank`` over all (ranked, expected) pairs.
    """
    raise NotImplementedError("mrr — implemented in a later step (T4.2)")

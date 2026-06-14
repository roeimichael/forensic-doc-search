"""Retrieval metrics (requirement T4.2): Hit@K and MRR.

Pure functions over ranked results — no I/O — so they are trivially unit-tested
(see tests/test_metrics.py). A "hit" = the expected ``source_file`` appears among
the retrieved chunks' ``source_file`` values.
"""

from __future__ import annotations

from collections.abc import Sequence


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% confidence interval for a binomial proportion.

    Reported alongside Hit@K because at small ``n`` a point estimate (e.g. 0.97 from
    30 queries) overstates precision; the interval shows how much the metric can move.
    """
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


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


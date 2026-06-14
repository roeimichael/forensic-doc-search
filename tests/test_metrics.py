"""Metric tests (T4.2): Hit@K and MRR correctness on toy rankings."""

from __future__ import annotations

import pytest

from ragforce.eval.metrics import hit_at_k, mrr, reciprocal_rank


def test_hit_at_k() -> None:
    ranked = ["b.txt", "a.txt", "c.txt"]
    assert hit_at_k(ranked, "b.txt", 1) is True
    assert hit_at_k(ranked, "a.txt", 1) is False
    assert hit_at_k(ranked, "a.txt", 2) is True
    assert hit_at_k(ranked, "z.txt", 3) is False


def test_reciprocal_rank() -> None:
    assert reciprocal_rank(["a", "b", "c"], "a") == 1.0
    assert reciprocal_rank(["a", "b", "c"], "b") == pytest.approx(0.5)
    assert reciprocal_rank(["a", "b", "c"], "c") == pytest.approx(1 / 3)
    assert reciprocal_rank(["a", "b"], "z") == 0.0


def test_mrr() -> None:
    ranks = [["a", "b"], ["x", "y", "z"]]
    expected = ["a", "z"]
    # (1/1 + 1/3) / 2
    assert mrr(ranks, expected) == pytest.approx((1.0 + 1 / 3) / 2)
    assert mrr([], []) == 0.0

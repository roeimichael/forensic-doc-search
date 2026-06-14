"""Tests for the eval logic that produces the headline numbers (T4).

These cover the pure, claim-generating helpers (dedup, summary, filter-satisfaction,
Wilson CI) without a live store; the full run() is validated end-to-end by `rag eval`.
"""

from __future__ import annotations

import pytest

from ragforce.eval.evaluate import _doc_satisfies, _summary, _uniq_sources
from ragforce.eval.metrics import wilson_interval
from ragforce.models import SearchHit


def _hit(sf: str, **md) -> SearchHit:
    return SearchHit(chunk_id=sf, score=1.0, text="", metadata={"source_file": sf, **md})


def test_uniq_sources_dedups_preserving_order() -> None:
    hits = [_hit("a.txt"), _hit("a.txt"), _hit("b.txt"), _hit("a.txt")]
    assert _uniq_sources(hits) == ["a.txt", "b.txt"]


def test_summary_counts_and_mrr() -> None:
    ranks = [["a", "b"], ["x", "y", "z"], ["q"]]
    expected = ["a", "z", "nope"]
    s = _summary(ranks, expected)
    assert s["n"] == 3
    assert s["hit@1"] == pytest.approx(1 / 3)        # only first query hits @1
    assert s["hit@5"] == pytest.approx(2 / 3)        # first two hit within 5
    assert s["mrr"] == pytest.approx((1.0 + 1 / 3 + 0.0) / 3)
    lo, hi = s["hit@5_ci"]
    assert 0.0 <= lo <= s["hit@5"] <= hi <= 1.0


def test_summary_empty() -> None:
    s = _summary([], [])
    assert s["n"] == 0 and s["hit@5"] == 0.0


def test_doc_satisfies_scalar_list_and_date_range() -> None:
    meta = {"doc_type": "report", "case_id": "2024-1", "date": "2024-03-15"}
    assert _doc_satisfies(meta, {"doc_type": "report"})
    assert not _doc_satisfies(meta, {"doc_type": "transcript"})
    assert _doc_satisfies(meta, {"case_id": ["2024-1", "2024-2"]})
    assert _doc_satisfies(meta, {"date": {"gte": "2024-03-01", "lte": "2024-03-31"}})
    assert not _doc_satisfies(meta, {"date": {"gte": "2024-04-01", "lte": "2024-04-30"}})


def test_wilson_interval_brackets_point_estimate() -> None:
    lo, hi = wilson_interval(29, 30)            # ~0.97
    assert lo < 29 / 30 < hi
    assert hi - lo > 0.05                        # small-n interval is appreciably wide
    assert wilson_interval(0, 0) == (0.0, 0.0)

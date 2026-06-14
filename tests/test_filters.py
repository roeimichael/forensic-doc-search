"""Filter-builder tests (T3.2): the metadata layer the team called 'undervalued'."""

from __future__ import annotations

import pytest

from ragforce.api.filters import FilterError, build_filter


def _conds(filters):
    f = build_filter(filters)
    return {c.key: c for c in f.must}


def test_empty_is_none() -> None:
    assert build_filter(None) is None
    assert build_filter({}) is None


def test_scalar_exact_match() -> None:
    c = _conds({"doc_type": "report"})["doc_type"]
    assert c.match.value == "report"


def test_list_match_any() -> None:
    c = _conds({"case_id": ["2024-1", "2024-2"]})["case_id"]
    assert c.match.any == ["2024-1", "2024-2"]


def test_exact_day_is_half_open_range() -> None:
    # a bare gte==lte midnight instant would drop same-day docs; we want [day, day+1)
    # (Qdrant's DatetimeRange coerces the ISO strings to datetime objects.)
    c = _conds({"date": "2024-01-15"})["date"]
    assert c.range.gte.date().isoformat() == "2024-01-15"
    assert c.range.lt.date().isoformat() == "2024-01-16"
    assert c.range.lte is None


def test_date_range_ops() -> None:
    c = _conds({"date": {"gte": "2024-01-01", "lt": "2024-02-01"}})["date"]
    assert c.range.gte.date().isoformat() == "2024-01-01"
    assert c.range.lt.date().isoformat() == "2024-02-01"


def test_multiple_fields_combine_as_and() -> None:
    f = build_filter({"doc_type": "report", "case_id": "2024-1"})
    assert len(f.must) == 2


def test_unknown_field_rejected() -> None:
    with pytest.raises(FilterError):
        build_filter({"text": "secret"})       # unindexed content field — must not be filterable
    with pytest.raises(FilterError):
        build_filter({"__proto__": 1})


def test_unknown_date_op_rejected() -> None:
    with pytest.raises(FilterError):
        build_filter({"date": {"between": "2024-01-01"}})

"""Request-validation tests: garbage is rejected at the contract, before model/store."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ragforce.api.schemas import HybridRequest, SearchRequest


def test_query_required_non_empty() -> None:
    with pytest.raises(ValidationError):
        SearchRequest(query="")
    with pytest.raises(ValidationError):
        SearchRequest(query="   ")     # whitespace-only is empty after strip


def test_query_is_stripped() -> None:
    assert SearchRequest(query="  hello  ").query == "hello"


def test_top_k_bounds() -> None:
    assert SearchRequest(query="q").top_k == 5            # default
    with pytest.raises(ValidationError):
        SearchRequest(query="q", top_k=0)
    with pytest.raises(ValidationError):
        SearchRequest(query="q", top_k=-5)
    with pytest.raises(ValidationError):
        SearchRequest(query="q", top_k=10_000)


def test_filtered_inherits_validation_and_defaults_filters() -> None:
    r = HybridRequest(query="q")
    assert r.filters == {}
    with pytest.raises(ValidationError):
        HybridRequest(query="", filters={"doc_type": "report"})

"""Dataset generator tests (T0.1): format coverage, ground-truth integrity, determinism."""

from __future__ import annotations

import json

import pytest

from ragforce.dataset import generate
from ragforce.dataset.generator import _NEEDLES


def test_generate_produces_all_formats_and_ground_truth(tmp_path) -> None:
    stats = generate(n=20, seed=7, out_dir=tmp_path)
    assert stats["documents"] == 20

    # All four required/extra formats are emitted.
    exts = {p.suffix for p in tmp_path.glob("*") if p.name != "ground_truth.json"}
    assert exts == {".txt", ".pdf", ".json", ".eml"}

    # Ground truth: >=10 pairs, every expected file exists, filters match the
    # target document's actual metadata (encoded in the filename).
    gt = json.loads((tmp_path / "ground_truth.json").read_text(encoding="utf-8"))
    assert len(gt) == len(_NEEDLES) >= 10
    for e in gt:
        assert (tmp_path / e["expected_source_file"]).exists()
        doc_type, case_id, date = e["expected_source_file"].split("__")[:3]
        flt = e["filters"]
        if "doc_type" in flt:
            assert flt["doc_type"] == doc_type
        if "case_id" in flt:
            assert flt["case_id"] == case_id
        if isinstance(flt.get("date"), dict):
            assert flt["date"]["gte"] <= date <= flt["date"]["lte"]


def test_generation_is_deterministic(tmp_path) -> None:
    generate(n=16, seed=1, out_dir=tmp_path / "a")
    generate(n=16, seed=1, out_dir=tmp_path / "b")
    names_a = sorted(p.name for p in (tmp_path / "a").glob("*"))
    names_b = sorted(p.name for p in (tmp_path / "b").glob("*"))
    assert names_a == names_b


def test_regeneration_is_clean_slate(tmp_path) -> None:
    # Re-running into the same dir with a different n must not leave stale files.
    generate(n=20, seed=1, out_dir=tmp_path)
    generate(n=16, seed=2, out_dir=tmp_path)
    docs = [p for p in tmp_path.glob("*") if p.name != "ground_truth.json"]
    assert len(docs) == 16


def test_n_below_needle_count_is_rejected(tmp_path) -> None:
    with pytest.raises(ValueError):
        generate(n=5, seed=1, out_dir=tmp_path / "nope")

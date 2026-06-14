"""Loader tests (T1.1): txt/json/eml load + metadata recovery; corrupt files skipped."""

from __future__ import annotations

import json
from email.message import EmailMessage
from pathlib import Path

from ragforce.loaders import get_loader, load_directory


def _write_corpus(d: Path) -> None:
    (d / "witness_statement__2024-1234__2024-03-05__001-a.txt").write_text(
        "A witness saw a red van leave the scene.", encoding="utf-8"
    )
    (d / "report__2024-1234__2024-03-05__002-b.json").write_text(
        json.dumps({
            "content": "Officer attended the scene and logged evidence.",
            "doc_type": "report", "case_id": "2024-1234", "date": "2024-03-05",
        }),
        encoding="utf-8",
    )
    msg = EmailMessage()
    msg["Subject"] = "Interview"
    msg["Date"] = "Tue, 05 Mar 2024 12:00:00 -0000"
    msg["X-Doc-Type"] = "transcript"
    msg["X-Case-ID"] = "2024-1234"
    msg.set_content("Q: What did you see?\nA: A red van.")
    (d / "transcript__2024-1234__2024-03-05__003-c.eml").write_text(msg.as_string(), encoding="utf-8")

    # corrupt JSON + unsupported extension — must be skipped, not raised
    (d / "report__2024-1234__2024-03-05__004-bad.json").write_text("{ not valid json", encoding="utf-8")
    (d / "notes.xyz").write_text("unsupported format", encoding="utf-8")


def test_loads_all_formats_and_recovers_metadata(tmp_path) -> None:
    _write_corpus(tmp_path)
    docs = {d.source_file.split("__")[0]: d for d in load_directory(str(tmp_path))}
    assert set(docs) == {"witness_statement", "report", "transcript"}
    for d in docs.values():
        assert d.case_id == "2024-1234"
        assert d.date == "2024-03-05"
        assert d.text.strip()
    assert docs["report"].doc_type == "report"      # from inline JSON metadata
    assert docs["transcript"].doc_type == "transcript"  # from .eml X-Doc-Type header


def test_corrupt_and_unsupported_files_are_skipped(tmp_path) -> None:
    _write_corpus(tmp_path)
    docs = list(load_directory(str(tmp_path)))
    assert len(docs) == 3  # bad json + .xyz dropped without raising


def test_get_loader_unknown_extension_is_none() -> None:
    assert get_loader("foo.xyz") is None
    assert get_loader("a.txt") is not None

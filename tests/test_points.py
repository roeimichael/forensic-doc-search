"""Point-construction tests (T1.5): UUID5 id determinism + payload completeness."""

from __future__ import annotations

import uuid

from ragforce.store.points import build_payload, make_point_id


def test_make_point_id_is_deterministic() -> None:
    a = make_point_id("report__2024-1__2024-01-01__0-x.txt", 0)
    assert a == make_point_id("report__2024-1__2024-01-01__0-x.txt", 0)  # same input → same id
    assert a != make_point_id("report__2024-1__2024-01-01__0-x.txt", 1)  # different chunk
    assert a != make_point_id("other.txt", 0)  # different file
    uuid.UUID(a)  # is a valid UUID string (Qdrant point-id requirement)


def test_build_payload_contains_all_metadata(sample_chunk) -> None:
    payload = build_payload(sample_chunk)
    expected = {
        "source_file", "chunk_index", "chunk_id", "doc_type",
        "case_id", "date", "title", "char_span", "text",
    }
    assert expected <= set(payload)
    assert payload["text"] == sample_chunk.text
    assert payload["doc_type"] == sample_chunk.doc_type
    assert payload["date"] == sample_chunk.date

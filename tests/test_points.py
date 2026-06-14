"""Point-construction tests (T1.5): UUID5 id determinism + payload completeness.

This is the unit-level guard for idempotency: the same (source_file, chunk_index)
must always produce the same point id.
"""

from __future__ import annotations

import pytest


def test_make_point_id_is_deterministic() -> None:
    pytest.skip("scaffold — make_point_id implemented in the next step (T1.5)")


def test_build_payload_contains_all_metadata(sample_chunk) -> None:
    pytest.skip("scaffold — build_payload implemented in the next step (T1.4)")

"""Store integration test (T1.5 / T2.x): ensure_collection / upsert / search vs live Qdrant.

Marked ``integration`` — requires a running Qdrant (``docker compose up -d``).
Run with: ``pytest -m integration``.
"""

from __future__ import annotations

import pytest


@pytest.mark.integration
def test_upsert_is_idempotent() -> None:
    pytest.skip("scaffold — integration test implemented in the next step (T1.5)")

"""Store integration test (T1.5): idempotent upsert + orphan sweep vs live Qdrant.

Marked ``integration`` — requires a running Qdrant (``docker compose up -d``).
Run with: ``pytest -m integration``. Uses a throwaway collection it creates+deletes.
"""

from __future__ import annotations

import pytest
from qdrant_client import models

from ragforce.store import VectorStore

_COLLECTION = "ragforce_itest"
_DIM = 4


def _point(pid: str, source_file: str, idx: int) -> models.PointStruct:
    return models.PointStruct(
        id=pid,
        vector={"dense": [0.1 * idx, 0.2, 0.3, 0.4]},
        payload={"source_file": source_file, "chunk_index": idx, "chunk_id": pid, "text": f"chunk {idx}"},
    )


@pytest.fixture
def store():
    # 127.0.0.1 (not "localhost") avoids a Windows localhost->IPv6 resolution stall.
    s = VectorStore(host="127.0.0.1", port=6333, collection=_COLLECTION, timeout=10)
    s.ensure_collection(dim=_DIM, recreate=True)
    yield s
    s._client.delete_collection(_COLLECTION)  # noqa: SLF001 — test teardown


@pytest.mark.integration
def test_upsert_is_idempotent(store) -> None:
    pts = [_point("11111111-1111-5111-8111-111111111111", "a.txt", 0),
           _point("22222222-2222-5222-8222-222222222222", "a.txt", 1),
           _point("33333333-3333-5333-8333-333333333333", "b.txt", 0)]
    store.upsert(pts)
    assert store.count() == 3
    store.upsert(pts)                       # same ids -> overwrite, not duplicate
    assert store.count() == 3


@pytest.mark.integration
def test_orphan_sweep_removes_stale_chunks(store) -> None:
    # a.txt starts with 2 chunks, b.txt with 1
    store.upsert([_point("11111111-1111-5111-8111-111111111111", "a.txt", 0),
                  _point("22222222-2222-5222-8222-222222222222", "a.txt", 1),
                  _point("33333333-3333-5333-8333-333333333333", "b.txt", 0)])
    assert store.count() == 3
    # a.txt is re-ingested but now yields only 1 chunk: sweep its old points first
    store.delete_by_sources(["a.txt"])
    store.upsert([_point("11111111-1111-5111-8111-111111111111", "a.txt", 0)])
    assert store.count() == 2               # a.txt(1) + b.txt(1); the orphan chunk_index=1 is gone


@pytest.mark.integration
def test_dim_mismatch_raises(store) -> None:
    with pytest.raises(ValueError, match="dense dim"):
        store.ensure_collection(dim=_DIM + 1)   # existing collection has _DIM

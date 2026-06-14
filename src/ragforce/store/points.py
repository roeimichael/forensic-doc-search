"""Chunk → Qdrant point conversion: the idempotency + metadata chokepoint.

This module is where two requirements are physically enforced, in one place:
    * IDEMPOTENCY (T1.5): ``make_point_id`` derives a deterministic UUID5 from
      ``(source_file, chunk_index)``. Same input → same id → ``upsert`` overwrites
      instead of duplicating, so re-running ingestion leaves the point count stable.
    * METADATA (T1.4): ``build_payload`` is the single place every metadata field
      is attached, guaranteeing no stored point is missing one.
"""

from __future__ import annotations

import uuid
from typing import Any

from ragforce.models import Chunk

# Fixed project namespace so UUID5 ids are stable across machines and runs.
NAMESPACE = uuid.UUID("5f9b9c1e-7a3d-5c4b-8e2a-1d0c3b2a4f6e")


def make_point_id(source_file: str, chunk_index: int) -> str:
    """Deterministic point id = ``UUID5(NAMESPACE, f"{source_file}:{chunk_index}")``.

    Qdrant point ids must be a UUID string or unsigned int; UUID5 satisfies that
    and is reproducible, which is exactly what idempotent upsert needs.

    TODO(T1.5): ``return str(uuid.uuid5(NAMESPACE, f"{source_file}:{chunk_index}"))``.
    """
    raise NotImplementedError("make_point_id — implemented in the next step (T1.5)")


def build_payload(chunk: Chunk) -> dict[str, Any]:
    """Build the Qdrant payload (all searchable/filterable metadata + text).

    Keys: source_file, chunk_index, chunk_id, doc_type, date (ISO str), case_id,
    title, char_span, text.

    TODO(T1.4): assemble the dict from ``chunk`` fields.
    """
    raise NotImplementedError("build_payload — implemented in the next step (T1.4)")


def to_point(chunk: Chunk, dense_vec: list[float], sparse_vec: Any | None = None) -> Any:
    """Assemble a ``qdrant_client.models.PointStruct`` for ``chunk``.

    id = chunk.chunk_id; vector = {"dense": dense_vec[, "sparse": sparse_vec]};
    payload = build_payload(chunk).

    TODO(T1.5): construct PointStruct with named vectors (sparse only if provided).
    """
    raise NotImplementedError("to_point — implemented in the next step (T1.5)")

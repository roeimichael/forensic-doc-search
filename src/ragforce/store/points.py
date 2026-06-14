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

from qdrant_client import models

from ragforce.models import Chunk

# Fixed project namespace so UUID5 ids are stable across machines and runs.
NAMESPACE = uuid.UUID("5f9b9c1e-7a3d-5c4b-8e2a-1d0c3b2a4f6e")


def make_point_id(source_file: str, chunk_index: int) -> str:
    """Deterministic point id = ``UUID5(NAMESPACE, f"{source_file}:{chunk_index}")``.

    Qdrant point ids must be a UUID string or unsigned int; UUID5 satisfies that
    and is reproducible, which is exactly what idempotent upsert needs.
    """
    return str(uuid.uuid5(NAMESPACE, f"{source_file}:{chunk_index}"))


def build_payload(chunk: Chunk) -> dict[str, Any]:
    """Build the Qdrant payload: all filterable metadata + the chunk text."""
    return {
        "source_file": chunk.source_file,
        "chunk_index": chunk.chunk_index,
        "chunk_id": chunk.chunk_id,
        "doc_type": chunk.doc_type,
        "case_id": chunk.case_id,
        "date": chunk.date,
        "title": chunk.title,
        "char_span": list(chunk.char_span),
        "text": chunk.text,
    }


def to_sparse_vector(sparse_emb: Any) -> models.SparseVector:
    """Convert a fastembed ``SparseEmbedding`` (numpy indices/values) to Qdrant's form."""
    return models.SparseVector(
        indices=sparse_emb.indices.tolist(), values=sparse_emb.values.tolist()
    )


def to_point(
    chunk: Chunk, dense_vec: list[float], sparse_vec: Any | None = None
) -> models.PointStruct:
    """Assemble a ``PointStruct``: deterministic id, named vectors, full payload."""
    vector: dict[str, Any] = {"dense": dense_vec}
    if sparse_vec is not None:
        vector["sparse"] = to_sparse_vector(sparse_vec)
    return models.PointStruct(id=chunk.chunk_id, vector=vector, payload=build_payload(chunk))

"""Qdrant collection schema (deliverable #2: the schema doc, expressed in code).

The collection carries two NAMED vectors so hybrid search is native:
    * ``dense``  — VectorParams(size=<model dim>, distance=COSINE)
    * ``sparse`` — SparseVectorParams(modifier=IDF)  (BM25)

Payload indexes are declared up front so metadata filtering (incl. datetime range
on ``date``) is fast and reliable (T3.2):
    doc_type → keyword, case_id → keyword, date → datetime
"""

from __future__ import annotations

from typing import Any

# (payload field, Qdrant index schema type). Datetime enables range filters on date.
PAYLOAD_INDEXES: list[tuple[str, str]] = [
    ("doc_type", "keyword"),
    ("case_id", "keyword"),
    ("date", "datetime"),
]


def vectors_config(dim: int) -> dict[str, Any]:
    """Return the dense named-vector config (cosine).

    TODO(T2.2): ``{"dense": VectorParams(size=dim, distance=Distance.COSINE)}``.
    """
    raise NotImplementedError("vectors_config — implemented in the next step (T2.2)")


def sparse_vectors_config() -> dict[str, Any]:
    """Return the sparse named-vector config (BM25 / IDF modifier).

    TODO(T2.2): ``{"sparse": SparseVectorParams(modifier=Modifier.IDF)}``.
    """
    raise NotImplementedError("sparse_vectors_config — implemented in the next step (T2.2)")

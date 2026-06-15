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

from qdrant_client import models

# (payload field, Qdrant index schema type). Datetime enables range filters on date;
# the source_file keyword index makes the idempotency delete-sweep and the /health
# distinct-document facet fast (and lets callers filter to a single file).
PAYLOAD_INDEXES: list[tuple[str, str]] = [
    ("doc_type", "keyword"),
    ("case_id", "keyword"),
    ("date", "datetime"),
    ("source_file", "keyword"),
]


def vectors_config(dim: int) -> dict[str, Any]:
    """Dense named-vector config (cosine distance)."""
    return {"dense": models.VectorParams(size=dim, distance=models.Distance.COSINE)}


def sparse_vectors_config() -> dict[str, Any]:
    """Sparse named-vector config (BM25 with IDF modifier) for hybrid search."""
    return {"sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)}

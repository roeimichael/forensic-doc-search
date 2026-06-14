"""Ingestion orchestrator: load → chunk → embed → upsert (requirement T1.6).

Pulls together loaders, the Chunker, the embedder(s), and the VectorStore. The
whole thing is idempotent because chunk ids are deterministic (see
:mod:`ragforce.store.points`) and Qdrant upsert overwrites by id.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ragforce.config import Settings


@dataclass
class IngestStats:
    """Summary counters returned by a run (also logged)."""

    documents: int = 0
    chunks: int = 0
    skipped_files: int = 0
    upserted: int = 0


def run_ingest(
    settings: "Settings",
    *,
    source_dir: str | None = None,
    recreate: bool = False,
) -> IngestStats:
    """Run the full ingestion pipeline and return counts.

    Steps:
        1. ``load_directory(source_dir)`` → Documents (corrupt/unsupported skipped).
        2. ``Chunker.chunk`` per document → Chunks (chunk_id minted here).
        3. Batch-embed chunk text: dense (+ sparse if hybrid enabled).
        4. ``to_point`` → PointStruct (payload/metadata attached here).
        5. ``VectorStore.upsert`` in batches (idempotent).

    TODO(T1.6): wire the above using build_embedder, Chunker, VectorStore;
    ``ensure_collection(dim=embedder.dim, recreate=recreate)`` first; batch by
    settings.embedding.batch_size / settings.qdrant.upsert_batch_size; log per stage.
    """
    raise NotImplementedError("run_ingest — implemented in the next step (T1.6)")

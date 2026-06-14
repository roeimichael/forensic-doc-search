"""Ingestion orchestrator: load → chunk → embed → upsert (requirement T1.6).

Pulls together loaders, the Chunker, the embedder(s), and the VectorStore. The
whole thing is idempotent because chunk ids are deterministic (see
:mod:`ragforce.store.points`) and Qdrant upsert overwrites by id.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ragforce.chunking import Chunker
from ragforce.embedding import build_embedder
from ragforce.loaders import load_directory
from ragforce.logging_setup import get_logger
from ragforce.store import VectorStore
from ragforce.store.points import to_point

if TYPE_CHECKING:
    from ragforce.config import Settings

_log = get_logger("pipeline")


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
    """Run the full ingestion pipeline (load → chunk → embed → upsert) and return counts.

    Idempotent: deterministic chunk ids + Qdrant upsert mean re-running on the same
    folder overwrites rather than duplicates.
    """
    source = source_dir or settings.paths.source_dir
    dense, sparse = build_embedder(settings)
    chunker = Chunker(
        dense.tokenizer,
        chunk_size=settings.chunking.chunk_size,
        chunk_overlap=settings.chunking.chunk_overlap,
        min_chunk_size=settings.chunking.min_chunk_size,
    )
    store = VectorStore(
        host=settings.qdrant.host,
        port=settings.qdrant.port,
        collection=settings.qdrant.collection,
        dense_vector_name=settings.qdrant.dense_vector_name,
        sparse_vector_name=settings.qdrant.sparse_vector_name,
    )
    store.ensure_collection(dim=dense.dim, recreate=recreate or settings.qdrant.recreate_on_ingest)

    stats = IngestStats()
    chunks = []
    for doc in load_directory(source):
        stats.documents += 1
        chunks.extend(chunker.chunk(doc))
    stats.chunks = len(chunks)
    _log.info("loaded %d documents -> %d chunks", stats.documents, stats.chunks)

    embed_bs = settings.embedding.batch_size
    upsert_bs = settings.qdrant.upsert_batch_size
    buffer: list = []
    for i in range(0, len(chunks), embed_bs):
        batch = chunks[i : i + embed_bs]
        texts = [c.text for c in batch]
        dense_vecs = dense.embed_passages(texts, batch_size=embed_bs)
        sparse_vecs = (
            sparse.embed_passages(texts, batch_size=embed_bs) if sparse else [None] * len(batch)
        )
        buffer.extend(to_point(c, dv, sv) for c, dv, sv in zip(batch, dense_vecs, sparse_vecs))
        while len(buffer) >= upsert_bs:
            store.upsert(buffer[:upsert_bs])
            stats.upserted += upsert_bs
            buffer = buffer[upsert_bs:]
    if buffer:
        store.upsert(buffer)
        stats.upserted += len(buffer)

    _log.info("ingest complete: %d chunks upserted into '%s'", stats.upserted, settings.qdrant.collection)
    return stats

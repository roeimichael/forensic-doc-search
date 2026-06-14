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
    failed: int = 0


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
        timeout=settings.qdrant.timeout,
        hnsw_m=settings.qdrant.hnsw_m,
        hnsw_ef_construct=settings.qdrant.hnsw_ef_construct,
        hnsw_ef_search=settings.qdrant.hnsw_ef_search,
        quantization=settings.qdrant.quantization,
        prefetch_multiplier=settings.hybrid.prefetch_multiplier,
        prefetch_min=settings.hybrid.prefetch_min,
    )
    did_recreate = recreate or settings.qdrant.recreate_on_ingest
    store.ensure_collection(dim=dense.dim, recreate=did_recreate)

    stats = IngestStats()
    chunks = []
    source_files: list[str] = []
    for doc in load_directory(source):
        stats.documents += 1
        source_files.append(doc.source_file)
        chunks.extend(chunker.chunk(doc))
    stats.chunks = len(chunks)
    _log.info("loaded %d documents -> %d chunks", stats.documents, stats.chunks)

    # Idempotency sweep: drop any existing points for these source files first, so a
    # document that now yields FEWER chunks doesn't leave stale orphans behind.
    # (A full recreate already wiped everything, so the sweep is only needed otherwise.)
    if not did_recreate:
        store.delete_by_sources(sorted(set(source_files)))

    embed_bs = settings.embedding.batch_size
    upsert_bs = settings.qdrant.upsert_batch_size
    for i in range(0, len(chunks), embed_bs):
        batch = chunks[i : i + embed_bs]
        try:
            texts = [c.text for c in batch]
            dense_vecs = dense.embed_passages(texts, batch_size=embed_bs)
            sparse_vecs = (
                sparse.embed_passages(texts, batch_size=embed_bs) if sparse else [None] * len(batch)
            )
            points = [to_point(c, dv, sv) for c, dv, sv in zip(batch, dense_vecs, sparse_vecs)]
            for j in range(0, len(points), upsert_bs):
                store.upsert(points[j : j + upsert_bs])
            stats.upserted += len(points)  # credited only after a successful upsert
        except Exception as e:  # noqa: BLE001 — one bad batch shouldn't abort the whole ingest
            stats.failed += len(batch)
            _log.error("embed/upsert failed for chunks %d..%d: %s", i, i + len(batch), e)

    msg = "ingest complete: %d chunks upserted into '%s'"
    if stats.failed:
        msg += f" ({stats.failed} chunks FAILED — re-run to retry)"
    _log.info(msg, stats.upserted, settings.qdrant.collection)
    return stats

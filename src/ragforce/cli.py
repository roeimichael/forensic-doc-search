"""``rag`` command-line entrypoint (Typer).

Commands (all are scaffold stubs this step — bodies raise NotImplementedError):
    rag generate   — build the synthetic, real-text-seeded corpus + ground_truth.json
    rag ingest     — load → chunk → embed → upsert into Qdrant (idempotent)
    rag eval       — run the retrieval evaluation (Hit@K, MRR) → docs/03_eval_results.md
    rag health     — print store stats (point count, collection, embedding model)

Only ``typer`` is imported at module load so ``rag --help`` works before the heavy
deps (torch, qdrant-client) are exercised. Pipeline imports happen inside command
bodies in the implementation step.
"""

from __future__ import annotations

import typer

app = typer.Typer(
    add_completion=False,
    help="On-prem RAG for forensic document search (semantic + metadata + hybrid).",
)


@app.command()
def generate(
    n: int = typer.Option(120, help="Number of documents to generate (>=40; 50–200 typical)."),
    seed: int = typer.Option(42, help="RNG seed for deterministic output."),
    out: str = typer.Option("data/generated", help="Output directory for the corpus."),
) -> None:
    """Generate the scenario-driven synthetic forensic corpus + ground_truth.json."""
    from ragforce.dataset import generate as _generate

    stats = _generate(n=n, seed=seed, out_dir=out)
    typer.echo(f"Generated {stats['documents']} documents across {stats['cases']} cases -> {stats['out_dir']}")
    typer.echo(f"  by format:   {stats['by_format']}")
    typer.echo(f"  by doc_type: {stats['by_doc_type']}")
    typer.echo(f"  ground-truth pairs: {stats['ground_truth_pairs']}")


@app.command()
def ingest(
    source: str = typer.Option("data/generated", help="Directory of raw documents."),
    recreate: bool = typer.Option(False, help="Wipe + rebuild the collection first."),
) -> None:
    """Run the ingestion pipeline (idempotent upsert into Qdrant)."""
    from ragforce.config import load_settings
    from ragforce.logging_setup import configure_logging
    from ragforce.pipeline import run_ingest

    configure_logging()
    stats = run_ingest(load_settings(), source_dir=source, recreate=recreate)
    typer.echo(
        f"Ingested {stats.documents} documents -> {stats.chunks} chunks -> {stats.upserted} upserted"
    )


@app.command()
def eval(
    ground_truth: str = typer.Option(
        "data/generated/ground_truth.json", help="Path to (query, expected) pairs."
    ),
) -> None:
    """Evaluate retrieval quality (Hit@1, Hit@5, MRR) and write the report."""
    from ragforce.config import load_settings
    from ragforce.eval import run
    from ragforce.logging_setup import configure_logging

    configure_logging()
    m = run(load_settings(), ground_truth_path=ground_truth)
    for nm in m["retrievers"]:
        s = m[nm]
        typer.echo(f"{nm:14s} Hit@1={s['hit@1']:.2f}  Hit@5={s['hit@5']:.2f}  MRR={s['mrr']:.3f}")
    if m["filter_recall"] is not None:
        typer.echo(
            f"Filter: precision={m['filter_precision']:.0%} recall={m['filter_recall']:.0%} "
            f"({m['filtered_total']} queries)"
        )
    typer.echo("Report -> docs/03_eval_results.md")


@app.command(name="fetch-models")
def fetch_models() -> None:
    """Pre-download all local models into the HF cache (run once, online).

    After this, set ``embedding.local_files_only: true`` (and optionally
    ``models_dir``) to run fully air-gapped — no network calls at ingest/serve time.
    """
    from ragforce.config import load_settings
    from ragforce.embedding import build_embedder, build_reranker

    s = load_settings()
    s.embedding.local_files_only = False  # force the fetch even if configured offline
    typer.echo(
        f"Fetching models -> dense={s.embedding.model_name}, sparse={s.hybrid.sparse_model}, "
        f"rerank={s.rerank.model_name if s.rerank.enabled else '(disabled)'}"
    )
    build_embedder(s)
    build_reranker(s)
    typer.echo("Done. Models cached; set embedding.local_files_only=true to run offline.")


@app.command()
def health() -> None:
    """Print vector-store stats: point count, collection name, embedding model."""
    from ragforce.config import load_settings
    from ragforce.store import VectorStore

    s = load_settings()
    store = VectorStore(
        host=s.qdrant.host, port=s.qdrant.port, collection=s.qdrant.collection,
        dense_vector_name=s.qdrant.dense_vector_name, sparse_vector_name=s.qdrant.sparse_vector_name,
        timeout=s.qdrant.timeout,
    )
    stats = store.stats()
    stats["embedding_model"] = s.embedding.model_name
    typer.echo(stats)


if __name__ == "__main__":
    app()

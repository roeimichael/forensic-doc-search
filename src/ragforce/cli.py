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
    # TODO(T1.6): load_settings() -> ragforce.pipeline.ingest.run_ingest(settings, source, recreate)
    raise NotImplementedError("ingestion is implemented in the next step (T1.x)")


@app.command()
def eval(
    ground_truth: str = typer.Option(
        "data/generated/ground_truth.json", help="Path to (query, expected) pairs."
    ),
) -> None:
    """Evaluate retrieval quality (Hit@1, Hit@5, MRR) and write the report."""
    # TODO(T4.x): ragforce.eval.evaluate.run(settings, ground_truth)
    raise NotImplementedError("evaluation is implemented in a later step (T4.x)")


@app.command()
def health() -> None:
    """Print vector-store stats: point count, collection name, embedding model."""
    # TODO(T3.3): ragforce.store.qdrant_store.VectorStore(...).stats()
    raise NotImplementedError("health is implemented in a later step (T3.3)")


if __name__ == "__main__":
    app()

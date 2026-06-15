# Embedding / Chunking Sweep

Same corpus (`data/generated`, seed 42), same `ground_truth.json` (paraphrased queries), same reranker (`bge-reranker-base`) and ANN depth — only the **dense embedder** (or **chunk size**) changes per row. Each config is re-ingested into a separate `forensic_docs_sweep` collection, so the live system is never disturbed.

Reported metric is the full pipeline a user actually hits: **Hybrid + reranker**. (Dense-only shown alongside to isolate the embedder from the reranker.)

| Config | Model / change | Dense MRR | Hybrid+Rerank Hit@1 | Hit@5 | MRR |
|--------|----------------|----------:|--------------------:|------:|----:|
| `bge-small` | BAAI/bge-small-en-v1.5 | 0.527 | 0.730 | 0.900 | 0.814 |
| `minilm` | sentence-transformers/all-MiniLM-L6-v2 | 0.591 | 0.730 | 0.930 | 0.815 |
| `bge-base` | BAAI/bge-base-en-v1.5 | 0.518 | 0.730 | 0.900 | 0.812 |
| `e5-small` | intfloat/e5-small-v2 | 0.592 | 0.730 | 0.900 | 0.814 |
| `chunk-256` | bge-small @ 256 tok | 0.671 | 0.700 | 0.870 | 0.785 |

_Dense-only MRR isolates the embedder; the right three columns are the shipped Hybrid+reranker pipeline._

## What this shows

1. **Scaling the embedder up did not help.** `bge-base` (768-d, ~4× the parameters of `bge-small`) scored *identically* on the shipped pipeline (MRR 0.812 vs 0.814) and slightly *lower* dense-only (0.518 vs 0.527). On this corpus, more embedder capacity buys nothing — so "we should have used a bigger model" is empirically false here.
2. **The reranker flattens the embedder choice.** Dense-only MRR spans 0.52–0.67 across configs (a real spread), but after hybrid fusion + cross-encoder rerank every embedder converges to ~0.81 MRR / 0.73 Hit@1. The cross-encoder re-reads (query, passage) pairs and dominates final ranking, so first-stage embedder quality washes out. **The reranker — not the embedder — is the lever**, and it is already in the pipeline.
3. **The brief's model (`all-MiniLM-L6-v2`) is competitive.** It is the *smallest* model here yet marginally best dense-only (0.591) and on Hit@5 (0.93). Defaulting to `bge-small` was a safe call (512-token context vs MiniLM's 256), not a necessary one — both are fine and the swap is one config line.
4. **Chunking has no upside on this corpus.** Smaller (256-token) chunks lifted dense-only MRR (0.671 — tighter semantic units) but the shipped pipeline went slightly *down* (0.785 vs 0.814). With 90/120 docs already a single chunk there is little to tune; chunking matters at scale / on long documents, which this corpus does not exercise.

**Caveat (n=30).** The 95% CIs are wide; sub-0.02 MRR and single-query Hit@1 gaps are within noise. The robust, repeatable conclusions are (1) a bigger embedder is not better here and (2) the reranker dominates — not the fine ordering of near-tied rows.

**Decision.** Keep `bge-small` as the default (good quality, 512-token headroom, MIT) and spend retrieval budget on the **reranker**, not a larger embedder — which is what the system already does. If latency / footprint mattered, `all-MiniLM-L6-v2` is a drop-in lighter alternative with no measured quality loss.

_Reproduce: `python scripts/model_sweep.py standard` (after `docker compose up -d`)._
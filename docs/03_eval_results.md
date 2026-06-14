# Evaluation Results

Retrieval quality over **30** `(query, expected_document)` pairs from the generated corpus (`data/generated/ground_truth.json`). A hit means the expected document's `source_file` appears among the retrieved chunks' documents. Ground-truth queries **paraphrase** the planted signature (lexically disjoint from the source) so the comparison measures retrieval, not verbatim string matching.

- Embedding model: `BAAI/bge-small-en-v1.5` (dim 384, cosine)
- Reranker: `BAAI/bge-reranker-base`
- Chunking: recursive, 400/50 tokens
- Retrieval depth: top-10, deduped to documents; all retrievers use equal depth
- Evaluation is deterministic (fixed corpus seed + model + ANN path).

## Retrievers compared

| Retriever | Hit@1 | Hit@5 (95% CI) | MRR |
|-----------|------:|:--------------:|----:|
| Dense (semantic) | 0.47 | 0.60 (95% CI 0.42–0.75) | 0.527 |
| BM25 (sparse) | 0.70 | 0.93 (95% CI 0.79–0.98) | 0.783 |
| Hybrid (RRF) | 0.53 | 0.90 (95% CI 0.74–0.97) | 0.696 |
| **Hybrid + reranker** | 0.73 | 0.90 (95% CI 0.74–0.97) | 0.814 |

> At n=30 the 95% CIs are wide — treat the ranking as indicative, not statistically separated where intervals overlap.

## By query category (Hit@5)

_Paraphrase = semantic match required; Entity = rare proper-token (name) match._

| Retriever | paraphrase (n=20) | entity (n=10) |
|-----------|:-:|:-:|
| Dense (semantic) | 0.40 | 1.00 |
| BM25 (sparse) | 0.90 | 1.00 |
| Hybrid (RRF) | 0.85 | 1.00 |
| Hybrid + reranker | 0.85 | 1.00 |

## Metadata filtering

Across **22** filtered queries (`doc_type` / `case_id` / `date`-range):

- **Precision 100%** — fraction of returned documents that actually satisfy the filter (constraint correctness; 100% means no non-matching doc leaks through).
- **Recall 91%** — fraction where the expected document survived inside the filtered result set (the filter doesn't drop the right document).

## Where retrieval fails, and why

- **Hybrid beats pure dense on Hit@5 by +30 pts** (0.60 → 0.90); BM25's exact-token matching rescues queries where the small dense model blurs rare entities together.
- **Dense (semantic)** scores Hit@5 0.40 on *paraphrase* (semantic) queries vs 1.00 on *entity* (rare-token) queries.
- **Hybrid (RRF)** scores Hit@5 0.85 on *paraphrase* (semantic) queries vs 1.00 on *entity* (rare-token) queries.
- **The cross-encoder reranker lifts MRR** by +0.118 over hybrid (0.696 → 0.814) by re-reading (query, passage) pairs jointly.
- **Residual misses** are queries whose paraphrase shares little surface vocabulary with the source; a larger embedder or query expansion would help most there.

### Examples the Hybrid + reranker recovered (dense missed @5)

- _a hand-sewn slotted screwdriver found at the scene_  (paraphrase) → `report__2022-3266__2024-10-18__017-ev.pdf`
- _a sandy-coloured Hyundai compact crossover that had a crumpled back fender_  (paraphrase) → `witness_statement__2023-4923__2024-02-02__024-veh.json`
- _a corroded peaked cap found at the scene_  (paraphrase) → `report__2023-2982__2024-01-17__029-ev.txt`
- _a specially etched set of gripping pliers found at the scene_  (paraphrase) → `report__2023-7930__2024-10-11__041-ev.eml`
- _a partly charred loop of green synthetic cord found at the scene_  (paraphrase) → `report__2022-2753__2023-11-07__053-ev.json`

_Reproduce: `rag generate && rag ingest && rag eval` (after `docker compose up -d`)._
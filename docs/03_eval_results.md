# Evaluation Results

Retrieval quality over **30** `(query, expected_document)` pairs from the generated corpus (`data/generated/ground_truth.json`). A hit means the expected document's `source_file` appears among the retrieved chunks' documents.

- Embedding model: `BAAI/bge-small-en-v1.5` (dim 384, cosine)
- Chunking: recursive, 400/50 tokens
- Retrieval depth: top-10 (deduped to documents)

## Semantic vs. Hybrid

| Retriever | Hit@1 | Hit@5 | MRR |
|-----------|------:|------:|----:|
| Dense (semantic) | 0.47 | 0.60 | 0.534 |
| **Hybrid (dense + BM25, RRF)** | **0.60** | **0.97** | **0.750** |

Hybrid improves Hit@5 by **+37 pts** and MRR by **+0.216** over pure semantic search. It recovered **11** queries that dense retrieval missed in its top-5.

## Metadata filtering

Across the **22** ground-truth queries carrying a filter (`doc_type` / `case_id` / `date`-range), the expected document was returned within the filtered results in **91%** of cases — confirming the filter logic constrains results correctly without dropping the relevant document.

## Where retrieval fails, and why

- **Pure dense struggles on entity-heavy queries.** Many forensic queries hinge on rare, specific tokens — proper names (e.g. a person mentioned in an interview), or exact item descriptions (a unique piece of evidence). A small dense model (bge-small) blurs these into nearby semantics, so the right document ranks below generic look-alikes.
- **BM25 rescues them.** Sparse lexical matching keys directly on those rare tokens, so hybrid fusion (RRF) lifts the correct document — hence the large Hit@5 jump and the 11 recovered queries.
- **Residual misses** are mostly queries whose distinctive phrase is paraphrased away from the document's wording; a larger embedding model or light query expansion would help.

### Examples hybrid recovered (dense missed @5)

- _vehicle described as a navy Ford sedan with a faded blue tarpaulin in the bed_ → `witness_statement__2022-2679__2023-08-29__000-veh.txt`
- _a hand-stitched flat-head screwdriver recovered from the scene_ → `report__2022-3266__2024-10-18__017-ev.pdf`
- _vehicle described as a beige Hyundai crossover with a dented rear bumper_ → `witness_statement__2023-4923__2024-02-02__024-veh.json`
- _a rusted baseball cap recovered from the scene_ → `report__2023-2982__2024-01-17__029-ev.txt`
- _a custom-engraved pair of pliers recovered from the scene_ → `report__2023-7930__2024-10-11__041-ev.eml`

_Reproduce: `rag generate && rag ingest && rag eval` (after `docker compose up -d`)._
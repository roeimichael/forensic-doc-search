# forensic-doc-search — On-Prem Document Search for Forensic Case Files

A search system for forensic case documents (witness statements, reports, transcripts).
You query by natural language and/or structured metadata (`doc_type`, `case_id`, `date`);
it returns the most relevant document chunks. All inference runs locally and the vector
store is self-hosted — no cloud APIs. (This is the retrieval layer of a RAG stack; it
returns source text and does not generate answers.)

Built for the Cellebrite Data Engineer home assignment.

- Design & schema: [`docs/02_architecture.md`](docs/02_architecture.md)
- Per-component rationale: [`docs/04_design_rationale.md`](docs/04_design_rationale.md)
- Requirements breakdown: [`docs/01_requirements_and_tasks.md`](docs/01_requirements_and_tasks.md)
- Embedder/chunking comparison: [`docs/06_model_sweep.md`](docs/06_model_sweep.md)

![Architecture](docs/architecture.png)

## What it does

- Ingests `.txt` / `.pdf` / `.json` / `.eml` from a folder → token-aware recursive
  chunking → local embeddings → Qdrant, with metadata attached to every chunk.
- Idempotent ingestion: deterministic UUID5 chunk ids + upsert, plus a per-`source_file`
  cleanup so re-ingesting an edited (shorter) document leaves no stale chunks.
- Three search endpoints — semantic, metadata-filtered (including date ranges), and
  hybrid (dense + BM25 with server-side RRF). A cross-encoder reranks the candidates of
  each mode.
- Local-only: `sentence-transformers` + Qdrant, no external APIs. Runs offline once the
  models are cached.
- Config-driven (`config.yaml` / `.env`): the embedding model and store settings are
  swappable without code changes.
- Evaluated with Hit@1 / Hit@5 / MRR over paraphrased ground-truth queries.

## Quickstart (three commands)

> Needs Python 3.10+ and Docker. For a CPU-only install (smaller/faster), first:
> `pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cpu`

```bash
pip install -r requirements.txt && pip install -e .   # 1. dependencies
docker compose up -d                                  # 2. self-hosted Qdrant
make run                                               # 3. generate → ingest → serve API
```

`make run` waits for Qdrant, generates the synthetic corpus, ingests it, then serves the
API at `http://localhost:8000` (`/docs` for OpenAPI). Without `make`:

```bash
python scripts/wait_for_qdrant.py
rag fetch-models      # once, online: cache the embed/sparse/reranker models
rag generate          # build data/generated/ + ground_truth.json
rag ingest            # load → chunk → embed → upsert
uvicorn ragforce.api.app:app --host 0.0.0.0 --port 8000
```

To run fully offline after `rag fetch-models`, set `RAG__EMBEDDING__LOCAL_FILES_ONLY=true`.

## API

| Method | Endpoint | Body | Purpose |
|--------|----------|------|---------|
| POST | `/search` | `{query, top_k=5, min_score?}` | Semantic search |
| POST | `/search/filtered` | `{query, filters, top_k=5, min_score?}` | Semantic + metadata filter |
| POST | `/search/hybrid` | `{query, filters, top_k=5, min_score?}` | Dense + BM25 (RRF) |
| GET | `/health` | — | Store stats (chunk_count, collection, model) |

Response: `{"results": [{"chunk_id", "score", "text", "metadata"}]}`. Inputs are validated
(`top_k` 1–100, non-empty query, optional `min_score` 0–1); filters are allow-listed to
indexed fields; a store outage returns **503**, an invalid filter **422**, and `/health`
never throws. The optional `min_score` drops hits below that reranker score.

## Search UI

A Streamlit client over the API — query box, `top_k` and minimum-confidence sliders, a
Semantic / Metadata-filtered / Hybrid selector, and a `doc_type` / `case_id` / `date`
filter panel. Start the API first, then:

```bash
streamlit run ui/streamlit_app.py     # API URL via RAG_API_URL (default http://localhost:8000)
```

## Evaluation

```bash
rag eval     # writes docs/03_eval_results.md
```

Over **30** `(query, expected_document)` pairs whose queries paraphrase the planted
signature (lexically disjoint from the source, so this measures retrieval rather than
string matching):

| Retriever | Hit@1 | Hit@5 (95% CI) | MRR |
|-----------|------:|:--------------:|----:|
| Dense (semantic) | 0.47 | 0.60 (0.42–0.75) | 0.527 |
| BM25 (sparse) | 0.70 | 0.93 (0.79–0.98) | 0.783 |
| Hybrid (RRF) | 0.57 | 0.90 (0.74–0.97) | 0.711 |
| **Hybrid + reranker** | **0.73** | 0.90 (0.74–0.97) | **0.813** |

Notes on the numbers: with this small embedding model, pure dense retrieval is weak on
paraphrased queries (Hit@5 0.40 on the paraphrase subset vs 1.00 on rare-token entity
queries); BM25 is a solid baseline for rare names/items; RRF fusion alone doesn't beat
BM25 here; the cross-encoder reranker gives the largest gain in top-rank precision
(Hit@1 0.57 → 0.73). Metadata filtering is 100% precision / 91% recall over 22 filtered
queries. n=30, so the confidence intervals are wide. Full breakdown:
[`docs/03_eval_results.md`](docs/03_eval_results.md).

## Project structure

```
src/ragforce/
  loaders/     txt/pdf/json/eml → Document   (skips corrupt files)
  chunking/    token-aware recursive splitter (exact char_span)
  embedding/   dense (sentence-transformers) + sparse (fastembed BM25) + cross-encoder reranker
  store/       Qdrant access, schema, UUID5 ids + payload (idempotency + cleanup)
  pipeline/    ingest orchestrator (load → chunk → embed → upsert)
  api/         FastAPI: /search, /search/filtered, /search/hybrid, /health
  eval/        Hit@K / MRR + Wilson CIs, per-category + precision/recall
  dataset/     synthetic corpus generator + ground truth
ui/            Streamlit search UI
config.yaml    primary config   |   .env.example  runtime overrides
docker-compose.yml   self-hosted Qdrant
docs/          requirements, architecture + schema, eval results
```

## Key decisions

- **Chunking — token-aware recursive.** Forensic docs are short and paragraph/turn
  structured, so recursive splitting on those boundaries fits at low cost. Chunk size is
  measured with the embedding model's own tokenizer and kept under its `max_seq_length`.
  Defaults: 400/50 tokens.
- **Embedding — `bge-small-en-v1.5`** (384-d, 512-token context, MIT). Swappable to
  `all-MiniLM-L6-v2` (the brief's suggestion) via config.
- **Vector store — Qdrant.** Native sparse vectors + server-side RRF for hybrid, payload
  filtering with datetime ranges, and a one-command Docker spin-up.
- **Reranking — `bge-reranker-base` cross-encoder.** Re-scores the top candidates; in the
  eval it gives the largest gain in top-rank precision. Config-gated, on by default, local.

More detail in [`docs/02_architecture.md`](docs/02_architecture.md).

## License

MIT (code). Seed text snippets retain their original licenses — see
[`data/README.md`](data/README.md).

# forensic-doc-search — On-Prem RAG for Forensic Documents

An on-premises Retrieval-Augmented-Generation pipeline for searching forensic case
documents (witness statements, reports, transcripts) by **natural-language query**
and **structured metadata** (doc_type, case_id, date) — with **no cloud APIs** and a
**self-hosted** vector store.

> Built for the Cellebrite GenAI Innovation Team home assignment. Requirements and
> task breakdown: [`docs/01_requirements_and_tasks.md`](docs/01_requirements_and_tasks.md).
> Design & schema: [`docs/02_architecture.md`](docs/02_architecture.md).

![Architecture](docs/architecture.png)

## Highlights

- **On-prem only** — local `sentence-transformers` embeddings (default
  `BAAI/bge-small-en-v1.5`), self-hosted **Qdrant**. No OpenAI/Cohere/Google.
- **Config-driven** — model, store host/port, collection, chunking all in
  `config.yaml` / `.env` (swap the embedding model with zero code changes).
- **Idempotent ingestion** — deterministic UUID5 chunk ids + Qdrant upsert; re-runs
  never duplicate.
- **Three search modes** — semantic, metadata-filtered (incl. date ranges), and
  **hybrid** (dense + BM25, RRF-fused server-side).
- **Measured** — Hit@1 / Hit@5 / MRR over a generated ground-truth set.

## Quickstart (≤ 3 commands)

> Assumes Python 3.10+ and Docker. For a CPU-only install (smaller/faster), first:
> `pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cpu`

```bash
pip install -r requirements.txt && pip install -e .   # 1. dependencies
docker compose up -d                                  # 2. self-hosted Qdrant
make run                                               # 3. generate → ingest → serve API
```

`make run` waits for Qdrant, generates the synthetic corpus, ingests it
idempotently, then serves the API at `http://localhost:8000` (`/docs` for OpenAPI).

No `make`? Run the equivalent directly:

```bash
python scripts/wait_for_qdrant.py
rag generate          # build data/generated/ + ground_truth.json
rag ingest            # load → chunk → embed → upsert
uvicorn ragforce.api.app:app --host 0.0.0.0 --port 8000
```

## API

| Method | Endpoint | Body | Purpose |
|--------|----------|------|---------|
| POST | `/search` | `{query, top_k=5}` | Semantic search |
| POST | `/search/filtered` | `{query, filters, top_k=5}` | Semantic + metadata filter |
| POST | `/search/hybrid` | `{query, filters, top_k=5}` | Dense + BM25 (RRF) |
| GET | `/health` | — | Store stats (count, collection, model) |

Response: `{"results": [{"chunk_id", "score", "text", "metadata"}]}`.

## Search UI _(bonus)_

A thin Streamlit client over the API — query box, `top_k` slider, hybrid toggle, and
a `doc_type` / `case_id` / `date` filter panel. Start the API first, then:

```bash
streamlit run ui/streamlit_app.py     # point at the API via RAG_API_URL (default http://localhost:8000)
```

## Evaluation

```bash
rag eval     # writes docs/03_eval_results.md (Hit@1, Hit@5, MRR; semantic vs hybrid)
```

Over **30** `(query, expected_document)` ground-truth pairs from the generated corpus:

| Retriever | Hit@1 | Hit@5 | MRR |
|-----------|------:|------:|----:|
| Dense (semantic) | 0.47 | 0.60 | 0.534 |
| **Hybrid (dense + BM25, RRF)** | **0.60** | **0.97** | **0.750** |

Hybrid lifts Hit@5 by **+37 pts** (recovering 11 dense misses) because forensic
queries hinge on rare tokens — proper names, specific items — that BM25 matches
exactly while a small dense model blurs them. Metadata-filtered queries return the
expected document **91%** of the time. Full analysis:
[`docs/03_eval_results.md`](docs/03_eval_results.md).

## Project Structure

```
src/ragforce/
  loaders/     txt/pdf/json/eml → Document   (graceful skip on corrupt files)
  chunking/    token-aware recursive splitter (size tracks the embedding model)
  embedding/   dense (sentence-transformers) + sparse (fastembed BM25)
  store/       Qdrant access, collection schema, UUID5 ids + payload (idempotency)
  pipeline/    ingest orchestrator (load → chunk → embed → upsert)
  api/         FastAPI: /search, /search/filtered, /search/hybrid, /health
  eval/        Hit@K / MRR metrics + evaluation runner
  dataset/     synthetic, real-text-seeded corpus generator + ground truth
ui/            Streamlit search UI
config.yaml    primary config   |   .env.example  runtime overrides
docker-compose.yml   self-hosted Qdrant
docs/          requirements, architecture + schema, eval results
```

## Design Decisions (summary)

- **Chunking — token-aware recursive.** Forensic docs are short and paragraph/turn
  structured; recursive splitting respects those boundaries and matches semantic
  chunking on short text at far lower cost. Chunk size is measured with the
  embedding model's own tokenizer and kept under its `max_seq_length` (else text is
  silently truncated). Defaults: 400/50 tokens (bge-small), 200/20 (MiniLM).
- **Embedding — `bge-small-en-v1.5`.** Strong retrieval for its size, MIT-licensed,
  512-token context. Swappable to `all-MiniLM-L6-v2` (the brief's pick) via config.
- **Vector store — Qdrant.** Native sparse vectors + server-side RRF for hybrid,
  payload pre-filtering with datetime ranges for robust metadata filtering, and a
  one-command Docker spin-up.

Full rationale: [`docs/02_architecture.md`](docs/02_architecture.md).

## Future Work

- Per-`source_file` stale-id sweep so shrinking a document removes orphan chunks.
- Optional local LLM (Ollama) for grounded answer generation over retrieved chunks.
- Weighted-score fusion as an alternative to RRF; reranking.

## License

MIT (code). Seed text snippets retain their original licenses — see
[`data/README.md`](data/README.md).

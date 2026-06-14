# Architecture & Design Decisions

> On-prem RAG for forensic document search. This document records the system design,
> the data flow, the vector-store schema (deliverable #2), and the rationale behind
> each technical decision. Requirements + task breakdown live in
> [`01_requirements_and_tasks.md`](01_requirements_and_tasks.md).

## 1. System Overview

```
                         INGESTION (offline, idempotent)
  data/generated/*.{txt,pdf,json}
        │
        ▼
   ┌──────────┐   ┌──────────┐   ┌─────────────┐   ┌──────────────┐   ┌──────────┐
   │ loaders  │──▶│ chunking │──▶│  embedding  │──▶│ store.points │──▶│  Qdrant  │
   │ (T1.1)   │   │ (T1.2)   │   │ dense+sparse│   │ UUID5 + meta │   │ (cosine) │
   └──────────┘   └──────────┘   └─────────────┘   └──────────────┘   └────┬─────┘
                                                                            │
                         SERVING (online)                                   │
   Streamlit UI ──HTTP──▶ FastAPI ──▶ embed query ──▶ VectorStore.search ───┘
   (Part 6)               (Part 3/5)                  dense | filtered | hybrid(RRF)
                                       │
                          eval/ ◀──────┘  ground_truth.json → Hit@K, MRR (Part 4)
```

The same `VectorStore` and embedder primitives serve ingestion (write) and the
API/eval (read), so no layer is rebuilt as the system grows.

## 2. Locked Technical Decisions

| Area | Decision | Rationale |
|------|----------|-----------|
| Embedding (default) | `BAAI/bge-small-en-v1.5` (384-d, 512 max_seq, MIT) | Strong small-model retrieval, permissive license, 512-token context gives chunking headroom. |
| Embedding (fallback) | `all-MiniLM-L6-v2` (256 max_seq) | The brief's suggestion; config-swappable, no code change. |
| Prefix handling | `query_prefix` / `passage_prefix` in config | bge needs a query prefix, e5 needs both, MiniLM none → model swap = config edit only. |
| Vector store | **Qdrant** (Apache-2.0, Docker) | Native sparse vectors + server-side RRF (hybrid), payload pre-filtering incl. datetime ranges, deterministic ids, one-command compose. |
| Distance | Cosine, L2-normalized dense vectors | Required by the brief. |
| Hybrid | Dense + BM25 sparse (`fastembed` `Qdrant/bm25`), RRF fused server-side | Keeps BM25 inside the store; dense+sparse never drift; no separate index. |
| Chunking | Token-aware recursive, length = embedder tokenizer | Sizes track the model on swap; recursive ≈ semantic quality on short docs at far lower cost. Defaults 400/50 tokens (bge), 200/20 (MiniLM); `min_chunk_size` drops fragments. |
| Idempotency | `chunk_id = UUID5(NAMESPACE, "{source_file}:{chunk_index}")` + upsert | Same input → same id → overwrite, never duplicate. |
| Dataset | Synthetic, real-text-seeded; emits `ground_truth.json` | Controlled metadata + known ground truth → honest, reproducible eval. |

## 3. Ingestion Data Flow & Idempotency

```
load_directory()  → Document[]   (unsupported/corrupt files logged + skipped, T1.1)
Chunker.chunk()   → Chunk[]      (chunk_id=UUID5 minted; metadata inherited)
embed (batched)   → dense (L2-norm) [+ sparse BM25 if hybrid.enabled]
to_point()        → PointStruct  (build_payload attaches ALL metadata)
VectorStore.upsert→ Qdrant       (idempotent: re-run overwrites same ids → stable count)
```

**Idempotency is enforced in exactly two places:** `store/points.make_point_id`
(deterministic id) and `VectorStore.upsert` (overwrite-by-id). **Metadata is
attached in exactly one place:** `store/points.build_payload`.

*Known gap (future work):* if an edited file produces fewer chunks, stale trailing
ids linger; a per-`source_file` id sweep before upsert would close it.

## 4. Vector-Store Schema (Deliverable #2)

Collection: `forensic_docs` (configurable). **Named vectors** so hybrid is native:

| Vector | Type | Config |
|--------|------|--------|
| `dense` | dense | `size = <model dim>` (384), `distance = COSINE` |
| `sparse` | sparse | BM25, `modifier = IDF` |

**Payload (per point):**

| Field | Type | Filterable | Notes |
|-------|------|-----------|-------|
| `source_file` | keyword | — | links chunk → document; eval key |
| `chunk_index` | int | — | position within document |
| `chunk_id` | keyword | — | == point id (UUID5) |
| `doc_type` | keyword | ✅ index | witness_statement / report / transcript |
| `case_id` | keyword | ✅ index | e.g. `2024-7812` |
| `date` | datetime | ✅ index | ISO-8601; supports **range** filters |
| `title` | text | — | optional |
| `char_span` | int[2] | — | offsets into the document |
| `text` | text | — | the chunk text (returned in results) |

Payload indexes declared at `ensure_collection` time → metadata filtering (incl.
`date` ranges) is fast and reliable (T3.2).

## 5. Configuration (config-driven / on-prem)

`config.yaml` is the readable source of truth; any field is overridable via
`RAG__<SECTION>__<KEY>` env vars / `.env` (env wins). Sections: `embedding`,
`chunking`, `qdrant`, `hybrid`, `paths`, `dataset`, `api`, `logging`. Model path,
store host/port, and collection name are all config — nothing cloud is hard-coded.

## 6. Reproducibility — the ≤3 commands

```bash
pip install -r requirements.txt && pip install -e .   # 1
docker compose up -d                                  # 2  (self-hosted Qdrant)
make run                                               # 3  (wait → generate → ingest → serve)
```

## 7. Repository Layout

See [README](../README.md#project-structure). Package root: `src/ragforce/`
(`loaders → chunking → embedding → store → pipeline` for ingestion; `api`, `eval`,
`ui` for serving + measurement). The three load-bearing seams — `models.py`
(`Document`/`Chunk`/`SearchHit`), `VectorStore`, and `Settings` — keep every layer
decoupled.

## 8. Status

Architecture locked; repository scaffolded (infra/config real, pipeline logic as
documented stubs). Implementation proceeds per the build order in
[`01_requirements_and_tasks.md`](01_requirements_and_tasks.md) §7: dataset → ingestion
→ API → eval → hybrid → UI → diagram.

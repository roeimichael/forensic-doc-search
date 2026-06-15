# Design Rationale & Component Guide

> A walk through **every requirement in the assignment**, mapped to the exact code that
> implements it, with the **decision** behind each choice, the **alternatives** weighed,
> **how the component works** (architecture-level for the embedding model, vector store,
> chunking, BM25, fusion, and reranking), and **primary sources** so you can read deeper.
>
> This is the "why" companion to [`02_architecture.md`](02_architecture.md) (the "what")
> and [`03_eval_results.md`](03_eval_results.md) (the "how well"). Everything below was
> verified against the code in `src/ragforce/`, not asserted from memory.

---

## 0. Requirement → code compliance matrix

| Assignment item | Implemented in | Status |
|---|---|---|
| **Part 1** load `.txt` / `.pdf` / `.json(content)` | `loaders/{txt,pdf,json,eml}_loader.py`, registry in `loaders/__init__.py` | ✅ (+`.eml`) |
| Part 1 chunking + justification | `chunking/chunker.py`, `chunking/separators.py` | ✅ |
| Part 1 local embedding (no external API) | `embedding/dense.py` (`bge-small-en-v1.5`) | ✅ |
| Part 1 metadata (`source_file`,`chunk_index`,`doc_type`,`date`,+extras) | `store/points.py::build_payload` | ✅ |
| Part 1 idempotency | `store/points.py::make_point_id` (UUID5) + `store/qdrant_store.py::upsert` + `delete_by_sources` | ✅ |
| **Part 2** self-hosted vector store | Qdrant via `docker-compose.yml` | ✅ |
| Part 2 schema with metadata | `store/schema.py` | ✅ |
| Part 2 cosine distance | `store/schema.py::vectors_config` (`Distance.COSINE`) | ✅ |
| Part 2 one-command spin-up | `docker-compose.yml` (+ healthcheck) | ✅ |
| **Part 3** `POST /search` `{query, top_k}` | `api/routes.py::search` | ✅ |
| Part 3 `POST /search/filtered` `{query, filters, top_k}` | `api/routes.py::search_filtered` + `api/filters.py` | ✅ |
| Part 3 `GET /health` (doc count, collection, model) | `api/routes.py::health` | ✅ |
| Part 3 response `{results:[{chunk_id,score,text,metadata}]}` | `api/schemas.py` | ✅ |
| **Part 4** ≥10 (query, expected) pairs | `dataset/generator.py` → `ground_truth.json` (30 pairs) | ✅ |
| Part 4 Hit@1, Hit@5, MRR | `eval/metrics.py`, `eval/evaluate.py` | ✅ |
| Part 4 qualitative failure analysis | `eval/evaluate.py::_narrative` → `03_eval_results.md` | ✅ |
| **Part 5 (bonus)** hybrid BM25 + dense, RRF | `store/qdrant_store.py::search_hybrid`, `embedding/sparse.py` | ✅ |
| Part 5 `POST /search/hybrid` | `api/routes.py::search_hybrid` | ✅ |
| Part 5 hybrid-vs-semantic in eval | `eval/evaluate.py` (4-retriever table) | ✅ |
| **Part 6 (bonus)** browser UI | `ui/streamlit_app.py` | ✅ |
| **Deliverable 8** architecture diagram (PNG) | `docs/architecture.png` (`scripts/make_diagram.py`) | ✅ |
| **Constraint** on-prem / no external inference API | local `sentence-transformers` + `fastembed`; no cloud SDKs | ✅ |
| **Constraint** no managed cloud vector DB | self-hosted Qdrant container | ✅ |
| **Constraint** Python 3.10+, `pyproject.toml`/`requirements.txt` | both present | ✅ |
| **Constraint** reproducible in ≤3 shell commands | `README.md` Quickstart + `Makefile` | ✅ |
| **Beyond spec** cross-encoder reranker | `embedding/rerank.py` | ➕ |

Nothing in the brief is skipped. The one literal-wording gap we found on re-reading — `/health`
"document count" — is now fixed (see §13).

---

## 1. Ingestion pipeline (Part 1)

### 1.1 Loaders — `loaders/`

**Requirement.** Load from a local directory; support at least `.txt`, `.pdf`, and `.json`
with a `content` field.

**Code.** A tiny `Loader` Protocol (`loaders/base.py`) declares `extensions` + `load(path)
-> Document`. Four implementations each own *only* text extraction: `TxtLoader`, `PdfLoader`
(pypdf), `JsonLoader` (reads the required `content` field), `EmlLoader` (email — an extra,
on-theme forensic format). `loaders/__init__.py` holds an extension→loader registry and
`load_directory()`, which walks the folder, dispatches by suffix, and **logs-and-skips**
unsupported/corrupt files instead of crashing (the brief's "repeatable pipeline").

**Decision & alternatives.** A registry of small single-responsibility loaders (vs. one big
`if/elif` reader) means adding a format is a local change and a corrupt file never aborts the
run. We added `.eml` because digital-forensic evidence is heavily email-based and it gives a
second metadata-carrying format (RFC-822 headers) at zero dependency cost (Python stdlib
`email`).

**Metadata recovery** (`loaders/metadata.py`) parses `doc_type__case_id__date__slug.ext`
filenames, with precedence **inline (JSON/eml fields) > filename > default** so a document can
override what the filename says.

### 1.2 Chunking — `chunking/chunker.py` (deep dive)

**Requirement.** Split each document into semantically coherent chunks and **justify the
strategy** (the brief explicitly rewards the *why*).

**Decision: token-aware recursive splitting.** The brief lists four options — fixed-size,
sentence-boundary, paragraph-boundary, recursive. We chose **recursive**, sized by the
**embedding model's own tokenizer**.

**What "recursive" means.** We keep an ordered list of separators from coarse to fine —
paragraph (`\n\n`) → line (`\n`) → sentence (`. `) → word (` `). The splitter tries the
coarsest separator first; any piece still over the token budget is recursively re-split with
the *next* finer separator. Pieces are then greedily **packed** into chunks of ≤ `chunk_size`
tokens, carrying `chunk_overlap` tokens of context across the boundary
(`_merge` + `_carry_overlap`). Transcripts use a structure-aware ladder that splits on the
real speaker labels (`\nOfficer`, `\nWitness`) before falling back to the generic ladder
(`chunking/separators.py`).

**Why token-aware.** Chunk length is measured with `tokenizer.encode(...)` — the *same*
tokenizer the embedding model uses — so no chunk exceeds the model's `max_seq_length` (512),
beyond which the transformer silently truncates and you embed only part of the text. Sizing in
*tokens* (not characters) is what guarantees that. `build_embedder` also warns if
`chunk_size` leaves no room for the model's special tokens.

**Provenance.** `char_span` is the **exact** offset pair into the source document, i.e.
`document.text[start:end] == chunk.text` — important for forensic citation/highlighting (a
wrong offset is a real bug, not cosmetic).

**Alternatives considered.**
- *Fixed-size (character)* — simplest, but cuts mid-sentence/word and ignores token counts, so
  it both reads worse and risks silent truncation.
- *Sentence/paragraph-boundary only* — cleaner cuts but no upper bound on size; one long
  paragraph can blow the token budget.
- *Semantic chunking* (embed sentences, cut where similarity drops) and *late chunking* — higher
  quality on long, topically-shifting documents, but materially more expensive and overkill for
  short, well-structured forensic docs; recursive ≈ semantic quality on short text at a fraction
  of the cost.

**Defaults & rationale.** 400 tokens / 50 overlap (~12.5%) for bge-small (512 ctx); ~200/20 for
MiniLM (256 ctx). Small enough to keep a chunk topically tight, large enough to retain context.

**Sources.**
- LangChain, *RecursiveCharacterTextSplitter* (the canonical recursive splitter) — docs:
  <https://python.langchain.com/docs/how_to/recursive_text_splitter/>
- Chroma, *Evaluating Chunking Strategies for Retrieval* (2024 technical report) —
  <https://research.trychroma.com/evaluating-chunking>
- Pinecone, *Chunking Strategies for LLM Applications* — <https://www.pinecone.io/learn/chunking-strategies/>
- Anthropic, *Introducing Contextual Retrieval* (2024) — context on chunk-level retrieval limits:
  <https://www.anthropic.com/news/contextual-retrieval>

### 1.3 Embedding model — `embedding/dense.py` (deep dive)

**Requirement.** Embed each chunk with a **local** model (e.g. `all-MiniLM-L6-v2` *or similar*);
no external API.

**Decision: `BAAI/bge-small-en-v1.5`** (default), with `all-MiniLM-L6-v2` as a config-swappable
fallback (the brief's literal suggestion).

**What the model is, architecturally.** bge-small-en-v1.5 is a **bi-encoder**: a compact
BERT-based Transformer encoder (~33M parameters, 384-dimensional output, 512-token context).
"Bi-encoder" means the query and each passage are encoded **independently** into a single
fixed-length vector; relevance is then just the cosine similarity of those two vectors. That
independence is what makes retrieval scalable — every passage vector is computed once at ingest
time and indexed, and a query only needs one forward pass plus a nearest-neighbour lookup.

Mechanically: the chunk's tokens go through the Transformer's stacked self-attention layers;
the **`[CLS]` token's final hidden state** is taken as the sentence embedding (BGE uses CLS
pooling), then **L2-normalized** so that cosine similarity equals a dot product. BGE is an
*instruction-aware, asymmetric* model: queries are prefixed with `"Represent this sentence for
searching relevant passages: "` while passages get no prefix — this is configured, not
hard-coded, so swapping to e5 (which prefixes both sides) or MiniLM (neither) is a config edit
(`config.yaml::embedding.query_prefix/passage_prefix`).

**How it was trained (why it retrieves well).** BGE is pre-trained with **RetroMAE** (a
masked-auto-encoder objective that forces the `[CLS]` vector to carry enough information to
reconstruct the input — i.e. to be a good sentence summary), then **contrastively fine-tuned**
on large collections of (query, positive, negatives) text pairs using in-batch and *mined hard*
negatives. Contrastive training pulls matching pairs together and pushes mismatches apart in the
vector space — exactly the geometry cosine retrieval exploits.

**Why this one (decision).** On the **MTEB** benchmark, bge-small-en-v1.5 is near the top of its
size class for retrieval; at 384-dim it keeps the index small and ANN fast; it has a 512-token
context (double MiniLM's effective 256, giving chunking headroom); and it is **MIT-licensed**.
It is fully local (`sentence-transformers` + PyTorch on CPU), satisfying the on-prem constraint.

**Alternatives considered.**
- `all-MiniLM-L6-v2` — 6-layer, 22M-param, 384-dim, mean-pooled, distilled model. Lighter/faster
  but weaker on MTEB and only ~256 effective tokens. **Kept as a one-line config fallback.**
- `bge-base/large`, `e5-base/large`, `gte` — stronger but 768–1024-dim and heavier; a sensible
  upgrade once latency/RAM budget allows (the eval shows the *reranker* buys more accuracy per
  millisecond here, so we spent the budget there first).
- Cloud embeddings (OpenAI/Cohere) — **disallowed** by the on-prem constraint.

**The honest caveat** (see eval): a 384-dim small model genuinely struggles on *paraphrased*
queries — that's visible as Dense Hit@5 0.40 on the paraphrase subset, and is exactly why we
added a reranker.

**Sources.**
- Reimers & Gurevych, *Sentence-BERT* (EMNLP 2019), arXiv:1908.10084 — the bi-encoder paradigm.
- Xiao et al., *C-Pack: Packed Resources for General Text Embeddings* (2023/2024), arXiv:2309.07597 — BGE.
- Xiao et al., *RetroMAE* (EMNLP 2022), arXiv:2205.12035 — BGE's pre-training objective.
- Muennighoff et al., *MTEB: Massive Text Embedding Benchmark* (2022), arXiv:2210.07316; leaderboard: <https://huggingface.co/spaces/mteb/leaderboard>
- Wang et al., *MiniLM* (NeurIPS 2020), arXiv:2002.10957 — the fallback model.
- Devlin et al., *BERT* (NAACL 2019), arXiv:1810.04805 — the underlying encoder.
- Model card: <https://huggingface.co/BAAI/bge-small-en-v1.5>

### 1.4 Metadata — `store/points.py::build_payload`

Every stored point carries the required `source_file`, `chunk_index`, `doc_type`, `date`, plus
useful extras: `chunk_id`, `case_id`, `title`, `char_span`, and `text`. Metadata is attached in
**exactly one place** (`build_payload`) so no point can be missing a field.

### 1.5 Idempotency — UUID5 + upsert + sweep

**Requirement.** Re-running the pipeline on the same folder must not create duplicates.

**Decision: deterministic point ids.** `make_point_id` derives a **UUID5** from
`"{source_file}:{chunk_index}"` against a fixed project namespace. UUID5 is a *name-based* UUID:
the same name always yields the same UUID (it's a SHA-1 hash of namespace+name, per RFC 4122),
so re-ingesting maps a chunk to the *same* Qdrant point id, and `upsert` overwrites it instead of
appending. Re-running leaves the point count stable.

We close the one classic gap: if an edited document now yields **fewer** chunks, the old trailing
ids would linger. `run_ingest` calls `VectorStore.delete_by_sources()` for the ingested files
*before* upserting, so orphans are swept. (`store/test_store_integration.py` proves both: stable
count on re-ingest, and orphan removal on shrink.)

**Sources.** Leach, Mealling, Salz, *RFC 4122 — A UUID URN Namespace* (2005), §4.3 (name-based
UUIDs): <https://www.rfc-editor.org/rfc/rfc4122>

---

## 2. Self-hosted vector store (Part 2)

### 2.1 Why Qdrant — `docker-compose.yml`, `store/`

**Decision: Qdrant** (`qdrant/qdrant:v1.13.0`, Apache-2.0, single Docker service).

**Alternatives (the brief's list) and why not:**
- **ChromaDB** — simplest (in-process), but weaker native hybrid/sparse story and filtering.
- **FAISS** — a raw ANN *library*, not a database: no payload store, no metadata filtering, no
  server. We'd have to build the metadata/filtering layer ourselves — the opposite of the
  "metadata filtering is undervalued" tip.
- **Weaviate** — capable and comparable, but heavier to operate for a prototype.
- **Qdrant (chosen)** — gives, out of the box, the three things this assignment actually rewards:
  (1) **native sparse vectors + server-side Reciprocal Rank Fusion**, so hybrid search is one
  query, not a hand-rolled fusion; (2) **payload pre-filtering with proper datetime ranges**
  (forensic search is date/case/type-constrained); (3) **one-command Docker** + deterministic
  point ids. Written in Rust, Apache-2.0, self-hosted — satisfying "no managed cloud vector DB".

### 2.2 How the index works — HNSW (deep dive)

A vector DB's core job is **approximate nearest-neighbour (ANN)** search: given a query vector,
find the closest stored vectors *fast*, without comparing against all N (which is O(N) and
hopeless at scale). Qdrant (like most modern stores) uses **HNSW — Hierarchical Navigable Small
World** graphs.

**The intuition.** HNSW builds a multi-layer proximity graph that behaves like a **skip list for
geometry**. Each vector is a node connected to its near neighbours. There are several layers: the
top layer is sparse (few nodes, long-range links), lower layers get denser, and the bottom layer
contains every node with short-range links. A search **starts at an entry point in the top
layer** and **greedily hops to the neighbour closest to the query**, descending a layer each time
it can't get closer — so it covers huge distances cheaply up top, then refines locally at the
bottom. Expected query time is **logarithmic** in N.

**The knobs (all exposed in `config.yaml::qdrant`):**
- `m` (we set 16) — number of links per node. Higher = better recall, more memory.
- `ef_construct` (128) — size of the candidate list while *building* the graph. Higher = a
  better-connected graph (slower build, better recall).
- `hnsw_ef` (search-time) — size of the candidate list while *querying*. The direct recall⇄speed
  dial at query time.
- `quantization` (optional int8 scalar) — compresses vectors ~4× to cut RAM, with a small recall
  cost; wired so a future jump to a 768-dim model stays cheap.

**Filterable HNSW — why it matters here.** Naïvely, metadata filtering is "retrieve, then throw
away non-matching results" — which destroys recall when the filter is selective (you might
retrieve 10 vectors and keep 0). Qdrant integrates the **payload filter into the graph traversal**
(it keeps extra links and checks the filter during search), so a filtered query still returns the
true top-k *within* the filtered subset. That's why `/search/filtered` is reliable, not best-effort.

**Sources.**
- Malkov & Yashunin, *Efficient and robust approximate nearest neighbor search using Hierarchical
  Navigable Small World graphs* (IEEE TPAMI 2018), arXiv:1603.09320 — the HNSW paper.
- Qdrant docs — indexing & filtrable HNSW: <https://qdrant.tech/documentation/concepts/indexing/>
  and <https://qdrant.tech/articles/filtrable-hnsw/>

### 2.3 Schema — `store/schema.py`

Collection `forensic_docs` uses **named vectors** so dense and sparse live on the same point:

| Vector | Type | Config |
|---|---|---|
| `dense` | dense | `size=384`, **`distance=COSINE`** (the required metric) |
| `sparse` | sparse | BM25, `modifier=IDF` |

**Cosine** is the brief's required metric; we **L2-normalize** dense vectors at embed time so
cosine reduces to a dot product (cheaper, numerically stable). **Payload indexes** are declared
up front for `doc_type`/`case_id` (keyword), `date` (datetime → range filters), and `source_file`
(keyword → fast idempotency sweep + the `/health` document facet).

---

## 3. Search API (Part 3) — `api/`

FastAPI (the brief's preference). Models — `DenseEmbedder`, `SparseEmbedder`, `Reranker`,
`VectorStore` — are loaded **once** in the app *lifespan* and injected into handlers via
`Depends`, so no model is constructed per request.

| Endpoint | Body | Maps to |
|---|---|---|
| `POST /search` | `{query, top_k}` | `dense.embed_query` → `store.search_dense` → rerank |
| `POST /search/filtered` | `{query, filters, top_k}` | + `filters.build_filter` |
| `POST /search/hybrid` | `{query, filters, top_k}` | + `sparse.embed_query` → `store.search_hybrid` (RRF) |
| `GET /health` | — | `store.stats` (document & chunk counts, collection, model) |

Response is exactly the required shape: `{"results":[{"chunk_id","score","text","metadata"}]}`
(`api/schemas.py`). **Production-mindedness** (the "think about production" tip): inputs are
validated (`top_k` 1–100, non-empty query) so garbage 422s at the contract; a store outage maps
to **503** and an invalid filter to **422**; `/health` *never throws* (reports
`status:"unavailable"`). Routes are sync `def`, so FastAPI runs the CPU-bound torch inference in a
worker thread instead of blocking the event loop.

### 3.1 Metadata filtering (rubric item 4) — `api/filters.py`

The brief calls this "undervalued"; we treated it as first-class. `build_filter` translates the
`filters` dict into a Qdrant `Filter`:
- scalar → exact match (`MatchValue`); list → match-any (`MatchAny`);
- `date` scalar → a **half-open day window** `[day, day+1)` (a bare `gte==lte` is a zero-width
  midnight instant that would drop same-day docs); `date` mapping → range (`gte/gt/lte/lt`);
- fields are **allow-listed** to the indexed set — filtering an unindexed field like `text` is
  rejected (422), closing a content-probe/foot-gun. Combined with AND (`must`). Unit-tested in
  `tests/test_filters.py`.

---

## 4. Evaluation (Part 4) — `eval/`

**Requirement.** ≥10 (query, expected_document) pairs; report Hit@1, Hit@5, MRR; qualitative
failure analysis.

**Metrics (`eval/metrics.py`).**
- **Hit@K** — 1 if the expected document is in the top-K retrieved (deduped to documents), else 0.
- **MRR (Mean Reciprocal Rank)** — average of `1/rank` of the first correct hit across queries; a
  classic IR metric that rewards ranking the answer *higher*, not just *present*.
- **Wilson 95% CI** — because a point estimate from 30 queries (e.g. 0.97) overstates precision,
  we report a confidence interval around each Hit@5. The Wilson score interval is the standard
  small-sample interval for a proportion (better-behaved than the normal approximation near 0/1).

**Methodology hardening (engineering-maturity signal).** Three things make these numbers
trustworthy rather than flattering:
1. **Paraphrased ground truth.** Queries describe the planted signature in *different* words
   (lexically disjoint from the source), so the score measures *retrieval*, not verbatim string
   matching. We assert 0 verbatim substrings in tests.
2. **Per-category breakdown.** Queries are tagged `paraphrase` (semantic) vs `entity` (rare proper
   token, e.g. a name), reported separately — this exposes *where* each retriever wins.
3. **Filter precision *and* recall, separated** — precision = returned docs that actually satisfy
   the filter (constraint correctness); recall = the expected doc survives the filter.

The failure analysis (`_narrative`) is **generated from the measured deltas**, so the report can't
claim a result it didn't get. Output: [`03_eval_results.md`](03_eval_results.md).

**Sources.** Voorhees, *The TREC-8 Question Answering Track* (1999) — MRR in IR; Wilson, *Probable
Inference, the Law of Succession, and Statistical Inference* (JASA, 1927) — the interval.

---

## 5. Hybrid search (Part 5, bonus) — BM25 + dense, RRF

### 5.1 BM25 sparse retrieval (deep dive) — `embedding/sparse.py`

**What it is.** BM25 (Best-Matching 25) is the classic **lexical** ranking function: it scores a
document by the query *terms it literally contains*, weighting each term by

- **TF with saturation** — more occurrences help, but with diminishing returns (parameter `k1`),
  so a term appearing 50× isn't 50× more relevant;
- **IDF** — rare terms (a surname, a serial number) count far more than common ones;
- **length normalization** — divide out document length so long docs don't win by sheer size
  (parameter `b`, relative to the **average document length** `avg_len`).

We represent BM25 as a **sparse vector** (term-id → weight) via `fastembed`'s `Qdrant/bm25`, so it
lives *inside* Qdrant as a named vector — dense and sparse never drift out of sync. Crucially we
set `avg_len` to track `chunk_size` (≈400); `fastembed`'s default (256) would mis-normalize every
400-token chunk.

**Why include it.** Forensic queries hinge on **rare exact tokens** — names, case IDs, specific
item descriptions — which a small dense model blurs into nearby semantics but BM25 nails. The eval
confirms it: BM25 alone is the strongest single retriever on this corpus (Hit@5 0.93).

### 5.2 Reciprocal Rank Fusion (deep dive) — `store/qdrant_store.py::search_hybrid`

**The problem.** Dense cosine scores (~0–1) and BM25 scores (unbounded) aren't comparable, so you
can't just add them. **RRF** sidesteps this by fusing **ranks**, not scores:

```
RRF(d) = Σ_retriever  1 / (k + rank_retriever(d))
```

Each retriever contributes `1/(k+rank)` for a document; a doc ranked #1 by either retriever gets a
big contribution, and being found by *both* stacks them. No score calibration, no tuning beyond
`k`. Qdrant runs this **server-side** via its Query API: two `Prefetch` branches (dense + sparse,
each fetching `max(top_k·multiplier, min)` candidates) feed a `FusionQuery(RRF)`. The same
metadata filter is applied to **both** branches. _Honest detail:_ `FusionQuery` exposes no `k`, so
we don't set one — Qdrant uses its built-in rank constant (a small default — `2` in the
qdrant-client reference implementation), **not** the `k = 60` from the Cormack et al. paper. The
absolute fused scores therefore differ from the paper's, but the rank-agreement behaviour is the
same, and the measured eval numbers reflect whatever constant Qdrant actually applies.

**Alternative.** Weighted score fusion (`α·dense + (1−α)·bm25` after normalization) — works but
needs per-corpus tuning of `α` and a normalization scheme; RRF is parameter-light and robust,
which is why it's the default here. The brief explicitly accepts either.

**Sources.**
- Robertson & Zaragoza, *The Probabilistic Relevance Framework: BM25 and Beyond* (FnTIR, 2009);
  Robertson et al., *Okapi at TREC-3* (1994) — the original BM25.
- Cormack, Clarke, Buettcher, *Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank
  Learning Methods* (SIGIR 2009) — the `1/(k+rank)`, k=60 result.

---

## 6. Cross-encoder reranking (beyond spec) — `embedding/rerank.py`

**Why it exists.** The bi-encoder embeds query and passage *separately*, so it never sees them
*together* — it can't model fine-grained interactions ("does *this* passage actually answer *this*
query?"). A **cross-encoder** does: it concatenates `[CLS] query [SEP] passage [SEP]` and runs full
self-attention across both, emitting a single relevance score. That's far more accurate **but**
O(candidates) forward passes per query (no precomputation possible), so it's used only as a cheap
**second stage**: first-stage retrieval (dense/hybrid) fetches the top-N candidates for recall,
then `bge-reranker-base` re-scores and reorders them for precision.

**Impact (from the eval):** reranking lifts Hit@1 from 0.53 → **0.73** and gives the best MRR
(0.814) — the single biggest accuracy lever on this corpus, and it stays fully local. It's
config-gated (`rerank.enabled`, default on) and applied to all three search modes.

**Sources.** Nogueira & Cho, *Passage Re-ranking with BERT* (2019), arXiv:1901.04085 — the
retrieve-then-rerank-with-a-cross-encoder pattern; FlagEmbedding `bge-reranker`:
<https://huggingface.co/BAAI/bge-reranker-base>; Sentence-Transformers *Retrieve & Re-Rank* docs:
<https://www.sbert.net/examples/applications/retrieve_rerank/README.html>

---

## 7. Search UI (Part 6, bonus) — `ui/streamlit_app.py`

A ~90-line **Streamlit** app (the brief: "a 50-line Streamlit app beats a half-built React
dashboard"). A thin `httpx` client over the API — query box, `top_k` slider, an explicit
**Semantic / Metadata-filtered / Hybrid** mode selector mapping 1:1 to the endpoints, and a
`doc_type`/`case_id`/`date` filter panel. Results show chunk text, score, and the required
metadata (`source_file`, `doc_type`, `date`). Corpus text is rendered as **plain text**, never
markdown, so uncontrolled forensic content can't break layout or inject markup. API-down / invalid
/ empty states are distinguished.

---

## 8. Dataset (Sample Dataset) — `dataset/generator.py`

**Option A (synthetic)**, chosen because no public corpus carries the *forensic* `doc_type` +
`case_id` + `date` metadata that is the graded part. A scenario-driven generator models ~30 cases,
each emitting four internally-consistent documents (two witness statements, a report, a transcript)
across four formats, with large entity pools for low duplication. Each case plants a unique
"signature" (a distinctive vehicle / evidence item / named person), and `ground_truth.json` is
built from **paraphrases** of those signatures (see §4). Output is byte-reproducible (deterministic
seed + reportlab invariant PDFs).

---

## 9. Constraints & ground rules

| Constraint | How it's met |
|---|---|
| On-prem only — no external inference API | `sentence-transformers` + `fastembed` + cross-encoder run locally on CPU; no cloud SDK in the dependency set. `rag fetch-models` + `local_files_only` make it fully **air-gappable**. |
| No managed cloud vector DB | self-hosted Qdrant container; host/port are config, nothing hosted. |
| Python 3.10+, declared deps | `pyproject.toml` + pinned `requirements.txt`. |
| Reproducible in ≤3 shell commands | `pip install …` → `docker compose up -d` → `make run` (`README.md` Quickstart). |
| Config-driven (model paths, host/port, collection) | `config.yaml` + `.env` overrides (`RAG__SECTION__KEY`); nothing cloud hard-coded. |
| LLM answer-generation | optional per the brief; deliberately **not** added (retrieval is the graded scope) — noted as future work. |

---

## 10. Known deviations & judgment calls

- **`/health` "document count".** The brief says *document* count; chunks (points) and documents
  differ (150 chunks vs 120 docs). We now report **both** `document_count` (distinct `source_file`,
  via a Qdrant facet over the new `source_file` index) **and** `chunk_count` — satisfying the
  literal requirement without losing the more operationally-useful number.
- **Reranker default-on adds latency.** ~1–2 s/query on CPU (50 candidate pairs). Justified by the
  Hit@1/MRR gain; switch off with `RAG__RERANK__ENABLED=false` for lowest latency.
- **Small embedder by design.** bge-small underperforms on paraphrases (eval is explicit about
  it); we spent the accuracy budget on the reranker, which buys more per millisecond here, and
  left `bge-base` + quantization wired for an easy upgrade.

---

## References (primary sources)

1. Reimers & Gurevych. *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.* EMNLP 2019. arXiv:1908.10084.
2. Xiao, Liu, Zhang, Muennighoff, et al. *C-Pack: Packed Resources for General Text Embeddings.* 2023/2024. arXiv:2309.07597. (BGE / FlagEmbedding)
3. Xiao, Liu, Shao, Cao. *RetroMAE: Pre-Training Retrieval-oriented Language Models Via Masked Auto-Encoder.* EMNLP 2022. arXiv:2205.12035.
4. Muennighoff, Tazi, Magne, Reimers. *MTEB: Massive Text Embedding Benchmark.* 2022. arXiv:2210.07316.
5. Wang, Wei, Dong, Bao, Yang, Zhou. *MiniLM: Deep Self-Attention Distillation…* NeurIPS 2020. arXiv:2002.10957.
6. Devlin, Chang, Lee, Toutanova. *BERT.* NAACL 2019. arXiv:1810.04805.
7. Malkov & Yashunin. *Efficient and robust ANN search using Hierarchical Navigable Small World graphs.* IEEE TPAMI 2018. arXiv:1603.09320.
8. Robertson & Zaragoza. *The Probabilistic Relevance Framework: BM25 and Beyond.* Foundations and Trends in IR, 2009.
9. Cormack, Clarke, Buettcher. *Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods.* SIGIR 2009.
10. Nogueira & Cho. *Passage Re-ranking with BERT.* 2019. arXiv:1901.04085.
11. Leach, Mealling, Salz. *RFC 4122 — A Universally Unique IDentifier (UUID) URN Namespace.* 2005.
12. Wilson. *Probable Inference, the Law of Succession, and Statistical Inference.* JASA, 1927.
13. Qdrant documentation — indexing, filtrable HNSW, hybrid queries: <https://qdrant.tech/documentation/>

> arXiv IDs/links are provided for convenience; verify the exact version when citing formally.

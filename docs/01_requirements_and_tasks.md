# Cellebrite — Senior Data Engineer Home Assignment
## Requirements Extraction & Task Breakdown

> **Purpose of this doc:** Capture *everything* the assignment asks for, broken into discrete, verifiable tasks, mapped to the scoring rubric. This is the input to the **architecture design** phase (next step). No code yet — this is the "what" and "what good looks like", not the "how".

> **Decision baked in:** The two "bonus" parts (Hybrid Search, Search UI) are treated as **mandatory**, not optional. They are worth +15% combined and signal seniority. We do all 6 parts.

---

## 0. The One-Paragraph Summary

Build an **on-premises RAG ingestion + retrieval system** for a law-enforcement document-search scenario. Ingest heterogeneous docs (`.txt`, `.pdf`, `.json`) from a local folder → chunk → embed with a **local** model → store in a **self-hosted** vector DB with rich metadata → expose a **FastAPI** search API (semantic, metadata-filtered, hybrid) → prove retrieval quality with a **quantitative eval** → wrap with a minimal **browser UI**. Everything runs locally, no cloud APIs, reproducible in **≤3 shell commands** after reading the README.

---

## 1. Hard Constraints & Ground Rules (non-negotiable — failing any of these tanks the submission)

| ID | Constraint | What it means in practice |
|----|-----------|---------------------------|
| C1 | **On-premises only** | No calls to OpenAI / Cohere / Google / any external embedding or inference API. All model inference runs locally. |
| C2 | **No managed cloud vector DB** | No Pinecone, Weaviate Cloud, or hosted services. Self-hosted instance only. |
| C3 | **Python 3.10+** | Dependency management via `requirements.txt` **or** `pyproject.toml`. |
| C4 | **Reproducibility ≤ 3 shell commands** | After reading the README, a reviewer runs pipeline + API end-to-end in at most 3 commands. This is a design forcing-function (think: `docker compose up`, `python ingest.py`, `uvicorn ...` — or fewer). |
| C5 | **Config-driven / on-prem awareness** | Model paths, store host/port, collection name → all in a config file or `.env`. No hard-coded cloud deps. This is explicitly a *first-class requirement, not an afterthought*. |
| C6 | **Reviewer environment assumption** | Assume reviewer has Python 3.10 and Docker installed. Build to that. |
| C7 | **LLM is optional** | May add a local LLM (e.g. Ollama) for grounded answer generation, but **not required**. Treat as out-of-scope unless time permits. |
| C8 | **Time budget** | Official estimate 4–6 hours. Scope decisions should respect "done and explained" over "big and unexplained". |

---

## 2. Scoring Rubric → What Reviewers Reward (this drives prioritization)

| Weight | Area | What they look for | Primary tasks |
|-------:|------|--------------------|---------------|
| **25%** | Pipeline design | Correctness, modularity, **re-runnability, idempotency** | T1.x |
| **20%** | Vector store & schema | Chunking strategy, **metadata richness**, index config | T1.3, T2.x |
| **20%** | Search quality | Relevance, sensible ranking, **metadata filtering accuracy** | T3.x, T4.x |
| **15%** | Code quality | Readability, **error handling, type hints, docstrings** | Cross-cutting (X1) |
| **10%** | On-prem awareness | No hard cloud deps, **config-driven**, Docker-ready | C1–C6, T2.3 |
| **10%** | Evaluation & metrics | Quantitative retrieval quality (**MRR, Hit@K**) | T4.x |
| **+10%** | Bonus — Hybrid search | BM25 + dense fusion; demonstrates **trade-off understanding** | T5.x |
| **+5%** | Bonus — Search UI | Functional end-to-end demo; metadata visible | T6.x |

**Total reachable: 115%.** Required parts = 100%. We chase the full 115%.

**Rubric takeaways that shape design:**
- Pipeline design is the single biggest bucket (25%) → idempotency and modularity must be visibly engineered, not accidental.
- Metadata filtering appears in **two** buckets (20% search quality + the team tip) → make filter logic robust (esp. **date ranges**, **case_id**, **doc_type**).
- "A docstring explaining *why* you chose a chunk size is worth more than a longer pipeline with no explanation." → **Written justification of decisions is itself graded.**
- Evaluation is called out as a "signal of engineering maturity" → don't skip or hand-wave it.

---

## 3. Task Breakdown by Part

Each task: **components**, **acceptance criteria** (how we know it's done), and **"show this"** (what the reviewer specifically wants to see).

### Part 1 — Ingestion Pipeline  *(rubric: 25% pipeline + feeds 20% schema)*

**T1.1 — Multi-format document loader**
- Components: loaders for `.txt`, `.pdf`, `.json` (JSON must read a `content` field).
- Acceptance: pointing the pipeline at a folder loads all three types; unsupported/corrupt files are skipped with a logged warning, not a crash.
- Show this: clean abstraction (one loader interface, per-format implementations), graceful error handling.

**T1.2 — Chunking with justified strategy**
- Components: split each doc into "semantically coherent" chunks; configurable chunk size/overlap.
- Acceptance: chunks are coherent (not mid-sentence garbage); strategy chosen from {fixed-size, sentence-boundary, paragraph-boundary, recursive}.
- Show this: a **README note justifying the chunking strategy and chunk size** (this is explicitly graded — see team tip). Decision deferred to architecture phase; default lean = recursive/sentence-aware with overlap.

**T1.3 — Local embedding**
- Components: embed each chunk with a **local** model (default `sentence-transformers/all-MiniLM-L6-v2`, 384-dim, or similar). Model path configurable (C5).
- Acceptance: zero network calls during embedding (C1); model name surfaced in `/health`.
- Show this: batched embedding, model configurable, no API keys anywhere.

**T1.4 — Metadata attachment**
- Components: attach to **every** chunk: `source_file`, `chunk_index`, `doc_type`, `date`, + useful extras.
- Required extras to design in (per scenario + team tip): `case_id`, `chunk_id` (stable unique id), maybe `char_span`/`title`.
- Acceptance: `date` is parsed (from filename or content) or accepted via config; all required fields present on every stored chunk.
- Show this: **rich** metadata schema (reviewers reward richness), robust `date` parsing.

**T1.5 — Idempotency**
- Components: deterministic chunk IDs (e.g. hash of `source_file` + `chunk_index` + content); upsert semantics.
- Acceptance: **re-running the pipeline on the same folder produces zero duplicates** (count stable). Changed file → updated, not duplicated.
- Show this: this is the highest-signal correctness item in the 25% bucket — make it explicit and testable.

**T1.6 — Pipeline orchestration & re-runnability**
- Components: single entry point (`ingest.py` or CLI) that runs load→chunk→embed→store end-to-end; idempotent; logs progress.
- Acceptance: one command ingests the whole sample folder; safe to re-run.

---

### Part 2 — Self-Hosted Vector Store  *(rubric: 20% schema + 10% on-prem)*

**T2.1 — Choose & configure vector DB**
- Options: Qdrant (Docker) / Weaviate (Docker) / ChromaDB (in-process) / FAISS (file-backed).
- Decision deferred to architecture phase. **Leaning Qdrant** (Docker-native, strong payload/metadata filtering, supports the "spin up in one command" + hybrid story; production-credible for a senior role). ChromaDB is the simplest fallback.
- Show this: justify the choice in README against the scenario (metadata filtering, on-prem, Docker).

**T2.2 — Collection / index schema**
- Components: explicit schema defining vector dim + all metadata fields with types; payload indexing for filterable fields (`doc_type`, `date`, `case_id`).
- Acceptance: schema documented in a **schema doc** (deliverable #2 requires it).
- Show this: metadata richness + index config = direct 20% bucket.

**T2.3 — Cosine similarity**
- Acceptance: distance metric is **cosine** (explicitly required).

**T2.4 — One-command spin-up**
- Components: `docker-compose.yml` (Qdrant/Weaviate) **or** setup script (Chroma/FAISS).
- Acceptance: reviewer spins up the store with a single command; ties into the ≤3-command budget (C4).

---

### Part 3 — Search API (FastAPI preferred)  *(rubric: 20% search quality)*

**T3.1 — `POST /search` (semantic)**
- Contract: accepts `{ query: str, top_k: int = 5 }`; returns top-k chunks with scores.
- Response shape (exact): `{ "results": [ { "chunk_id": "...", "score": 0.91, "text": "...", "metadata": { ... } } ] }`.

**T3.2 — `POST /search/filtered` (semantic + metadata filter)**
- Contract: accepts `{ query: str, filters: dict, top_k: int = 5 }`; `filters` maps metadata field → value (e.g. `{ "doc_type": "witness_statement", "date": "2024-01-15" }`).
- Acceptance: filters apply at the store level (pre-filter), not post-hoc slicing; **robust** — handle multiple filters, missing fields, and (stretch) **date ranges** per team tip.
- Show this: metadata filtering accuracy is explicitly graded and "undervalued" per the team.

**T3.3 — `GET /health`**
- Contract: returns store stats — **document/chunk count, collection name, embedding model**.

**T3.4 — API quality cross-cuts**
- Components: Pydantic request/response models, consistent error responses, input validation, OpenAPI docs (free with FastAPI).
- Acceptance: same JSON result shape across `/search`, `/search/filtered`, `/search/hybrid`.

---

### Part 4 — Evaluation  *(REQUIRED — rubric: 10% eval, feeds 20% search quality)*

**T4.1 — Build eval set**
- Components: **≥10** `(query, expected_document)` pairs over our own sample data; cover semantic + filtered cases.
- Acceptance: pairs stored as a versioned file (JSON/CSV) in the repo.

**T4.2 — Metrics**
- Components: **Hit@1**, **Hit@5**, **MRR (Mean Reciprocal Rank)**.
- Acceptance: a runnable `evaluate.py` computes all three.

**T4.3 — Qualitative analysis**
- Components: brief written analysis — **where retrieval fails and why**.
- Acceptance: included in report.

**T4.4 — Eval report**
- Components: `eval_results.md` with a metrics table (a simple table is explicitly "sufficient").
- Acceptance: must also report **hybrid vs pure-semantic** comparison (see T5.4).

---

### Part 5 — Hybrid Search  *(BONUS → treated as MANDATORY; rubric: +10%)*

**T5.1 — BM25 sparse retriever**
- Components: BM25 index over chunk text (e.g. `rank_bm25`, or store-native if Qdrant/Weaviate supports it).

**T5.2 — Rank fusion**
- Components: combine dense + sparse via **Reciprocal Rank Fusion (RRF)** or weighted score combination. Fusion weights configurable.

**T5.3 — `POST /search/hybrid` endpoint**
- Contract: separate endpoint, same response shape as T3.1.

**T5.4 — Prove it in eval**
- Acceptance: eval report shows **whether hybrid improves Hit@K vs pure semantic** (numbers, not vibes). Discuss the trade-off — this demonstrates the "understanding of trade-offs" the bonus rewards.

---

### Part 6 — Search UI  *(BONUS → treated as MANDATORY; rubric: +5%)*

**Guidance from team:** "A 50-line Streamlit app beats a half-built React dashboard every time." Functional > beautiful. Data-engineer role, not frontend.

**T6.1 — Query input + search**
- Components: text input for natural-language query; "Search" button calling `POST /search`.

**T6.2 — Results rendering**
- Components: list of results showing **chunk text, relevance score, and key metadata** (`source_file`, `doc_type`, `date`).

**T6.3 — Filter panel**
- Components: ≥1 dropdown/text field for metadata filtering (`doc_type` or `date`) mapping to `POST /search/filtered`.

**T6.4 — Implementation choice**
- Options: single `index.html` + vanilla JS `fetch()` / Streamlit / Gradio.
- Decision deferred; **leaning Streamlit** per team tip. No auth, pagination, or styling needed.

---

## 4. Sample Dataset  *(building it is part of the assignment)*

**T0.1 — Dataset generation** *(prerequisite for everything — DONE)*
- **IMPLEMENTED: scenario-driven synthetic generator** (`src/ragforce/dataset/generator.py`). Models distinct **case scenarios** (not loose snippets) so the corpus is diverse and `case_id` is meaningful.
- `ceil(n/4)` cases, each with structured facts drawn from large entity pools (names, streets, vehicles, evidence, crime types). Each case emits **4 internally-consistent docs** (2 witness statements, 1 report, 1 transcript) → `case_id` groups a real case's documents.
- **Formats: `.txt`/`.pdf`/`.json`/`.eml`** — the required three **plus email evidence** (on-theme for digital forensics; header-derived metadata). Metadata (`doc_type`/`case_id`/`date`) is controlled, encoded in the filename, and embedded inline for json/eml.
- **Ground truth:** one `(query, expected_source_file, filters)` per case; each case has a **unique signature** (distinctive vehicle → witness stmt, unique evidence → report, named person → transcript) so every query is unambiguous. Filters rotate across **semantic / doc_type / case_id / date-range**.
- **Why scenario-driven, not snippet-seeded:** a first snippet-stitched version measured 92% sentence duplication, 14/120 findable docs, and meaningless `case_id`. Scenario-driven fixed all three (52% dup — mostly structural boilerplate; 30/30 ground-truth signatures unambiguous; 4 docs/case).
- **Status:** done + validated (n=120 → 30 cases, even format split, 30 sound GT pairs, some multi-chunk reports). Deterministic, clean-slate, 5 unit tests. Generated corpus is gitignored (reproducible).

---

## 5. Cross-Cutting / Code-Quality Tasks  *(rubric: 15% code quality + 10% on-prem)*

**X1 — Code quality** — type hints, docstrings (esp. the "why" ones), readable structure, consistent error handling. Graded directly (15%).
**X2 — Config layer** — central config / `.env`: model path, store host/port, collection name, chunk params, fusion weights (C5).
**X3 — Dependency management** — `requirements.txt` or `pyproject.toml`, pinned (C3).
**X4 — Project structure** — modular packages (loaders / chunking / embedding / store / api / eval / ui) so modularity is visible (feeds 25% bucket).
**X5 — Logging** — structured, useful progress + warning logs in pipeline and API.

---

## 6. Deliverables Checklist  *(the 8 required artifacts + submission mechanics)*

| # | Deliverable | Format | Covered by |
|---|-------------|--------|-----------|
| 1 | Ingestion pipeline (parse, chunk, embed, store) | Python script(s) + README | T1.x |
| 2 | Vector store (on-prem): populated index + metadata | Qdrant/Weaviate/FAISS + **schema doc** | T2.x |
| 3 | Semantic search API endpoint | FastAPI/Flask route | T3.1 |
| 4 | Metadata filtering layer | API query-param handling | T3.2 |
| 5 | Hybrid search (bonus) | BM25 + vector fusion logic | T5.x |
| 6 | Evaluation report (retrieval metrics) | Markdown or PDF | T4.x |
| 7 | Basic search UI (bonus) | index.html / Streamlit / Gradio | T6.x |
| 8 | **Architecture diagram** | PNG or draw.io export | T7.1 (below) |

**T7.1 — Architecture diagram** — system diagram (ingestion flow + serving flow + components) exported as PNG / draw.io. *This is the output of the next phase.*

**T8.1 — README** — must cover: environment setup, run ingestion, start API, run evaluation. Must honor the ≤3-command rule (C4).

**T8.2 — Submission mechanics**
- Single **GitHub repository** (public, or private + add reviewer's GitHub handle as collaborator).
- Share repo URL with recruiter by email.
- **Optional but appreciated:** 2–5 min Loom/screen-recording demo of the API in action.
- If any section is unfinished: include a short written note on what you'd have done and why (cheap insurance — plan to include a "Future Work / Trade-offs" section regardless).

---

## 7. Suggested Build Order (dependency-aware)

1. **T0.1** Dataset generator (unblocks everything; defines metadata reality).
2. **X2/X3/X4** Project skeleton + config + deps.
3. **T2.1/T2.4** Stand up vector store (Docker/one-command).
4. **T1.x** Ingestion pipeline (load→chunk→embed→store+metadata→idempotency).
5. **T3.x** Search API (semantic → filtered → health).
6. **T4.x** Evaluation (eval set + metrics + report) — *use it to tune chunking/top_k*.
7. **T5.x** Hybrid search + re-run eval to show delta.
8. **T6.x** Streamlit UI.
9. **T7.1** Architecture diagram + **T8.1** README polish + **T8.2** submission (+ optional demo).

---

## 8. Open Decisions → Resolve in Architecture Phase (next step)

These are intentionally **not** decided here; they're the agenda for the architecture doc:

1. **Vector store**: Qdrant vs ChromaDB (production-credibility + hybrid + Docker vs simplicity). *Lean: Qdrant.*
2. **Chunking strategy & size**: recursive vs sentence-boundary; chunk size + overlap (must be justified in writing).
3. **Embedding model**: `all-MiniLM-L6-v2` (fast, 384-d) vs a larger/stronger local model (quality vs speed).
4. **Hybrid fusion**: RRF vs weighted-sum; where BM25 lives (in-process vs store-native).
5. **`date` handling**: storage format + range-filter support (impacts schema + filter API).
6. **Dataset shape** *(approach DECIDED: synthetic, real-text-seeded — see T0.1)*: still open → doc_type set, number of docs, case_id distribution, which real corpus to seed from, how ground-truth is encoded for eval.
7. **UI**: Streamlit vs static `index.html` (Streamlit adds a dep but is faster to build).
8. **Repo layout & the exact 3 commands** that satisfy C4.
9. **Optional LLM (Ollama) grounding**: in or out? (Default: out, note as future work.)

---

### Status
- [x] Requirements fully extracted from PDF (6 pages)
- [x] Tasks enumerated, mapped to rubric, bonuses promoted to mandatory
- [ ] **Next: Architecture design doc** (resolve §8 decisions, draw the diagram, define repo layout + the 3 commands)

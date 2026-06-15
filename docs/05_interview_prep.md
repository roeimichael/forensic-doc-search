# Interview Prep & Methodology Study Guide

> Goal: so you can **explain every choice in this project from first principles** and
> handle follow-up questions — not just recite a sentence. Each section is:
> **What it is → How it works (worked example) → Why we chose it (vs alternatives) →
> Say this in the interview → If they push back.**
>
> Companion to [`04_design_rationale.md`](04_design_rationale.md) (decisions + sources)
> and [`03_eval_results.md`](03_eval_results.md) (numbers). This file is the *teaching* one.

---

## 0. The 90-second project pitch

When they say *"walk me through what you built"*:

> "It's an **on-prem RAG retrieval system** for forensic documents. The pipeline **ingests**
> heterogeneous files (txt/pdf/json/eml), **chunks** them with a token-aware recursive
> splitter, **embeds** each chunk locally with a `bge-small` sentence-transformer, and
> **stores** them in a self-hosted **Qdrant** vector database with rich metadata. A FastAPI
> service exposes **semantic**, **metadata-filtered**, and **hybrid** (dense + BM25)
> search, with a **cross-encoder reranker** as a precision stage. I **measured** it —
> Hit@K, MRR with confidence intervals, per-category — and there's a Streamlit UI. Everything
> runs locally with no cloud APIs, reproducible in three commands."

That sentence touches every rubric item. Then let them pick what to drill into.

> **Note on "RAG":** strictly this is the **retrieval** half of RAG (the R). We retrieve and
> rank passages; we don't *generate* an answer with an LLM (the brief made generation
> optional). If asked: *"I built the retrieval and ranking layer; bolting on a local LLM via
> Ollama to generate grounded answers over the retrieved chunks is the natural next step."*

---

## 1. Chunking — recursive splitting (the one you asked about)

### What it is
Embedding models can only read a limited window of text at once (here **512 tokens**), and a
whole document is usually longer. **Chunking** is splitting a document into smaller pieces so
each piece (a) fits the model and (b) is *topically coherent* — one chunk = roughly one idea,
so its embedding is a clean representation. You retrieve and return **chunks**, not whole files.

**Recursive chunking** (a.k.a. recursive character/token splitting) is a *structure-aware*
strategy: you give it an **ordered list of separators from coarsest to finest**, and it splits
on the coarsest one that yields pieces under your size budget — recursing to finer separators
only where a piece is still too big.

Our separator ladder (`chunking/separators.py`):
```
paragraph "\n\n"  →  line "\n"  →  sentence ". "  →  word " "
```

### How it works — a worked example
Budget = say 30 tokens. Document:
```
Witness statement.

The suspect wore a navy jacket. He fled toward Oak Street and entered a waiting car.

Signed: J. Smith
```
1. **Try "\n\n" (paragraphs):** → `["Witness statement.", "The suspect ... waiting car.", "Signed: J. Smith"]`.
2. Check each piece's **token count**:
   - "Witness statement." → 3 tokens ✓ (under budget, keep whole)
   - "The suspect wore ... waiting car." → 40 tokens ✗ (over budget) → **recurse** with the next separator.
3. **Recurse on ". " (sentences)** for that piece → `["The suspect wore a navy jacket", "He fled toward Oak Street and entered a waiting car"]` → each now under budget ✓.
4. "Signed: J. Smith" → 5 tokens ✓.
5. **Merge/pack**: greedily combine adjacent pieces up to the budget, and **carry an overlap**
   (e.g. 50 tokens) from the end of one chunk into the start of the next so context isn't cut
   mid-thought.

The key idea: **respect the document's natural boundaries first** (don't cut mid-paragraph if
you don't have to), and only break finer when a piece is genuinely too long.

### Two refinements we added (high-signal)
- **Token-aware**, not character-aware: the size budget is measured with the **embedding
  model's own tokenizer** (`tokenizer.encode`), because the model's limit is in *tokens*, and
  token density varies (one "word" can be several tokens). Character budgets can't guarantee
  you stay under 512 tokens. This is the single most important nuance — character-based
  chunking can silently overflow the model and get truncated.
- **Exact `char_span`**: each chunk stores the exact `[start, end]` offsets into the original
  document, so `document.text[start:end] == chunk.text` — real provenance for a forensic tool.

### Why recursive (vs the alternatives the brief listed)
| Strategy | Idea | Why not (here) |
|---|---|---|
| **Fixed-size** | every N characters/tokens | cuts mid-sentence/word; ignores structure; reads worst |
| **Sentence-boundary** | split on sentences | clean cuts but **no upper bound** — one long sentence/paragraph can exceed the model |
| **Paragraph-boundary** | split on paragraphs | same unbounded-size problem |
| **Recursive (chosen)** | structure-aware, size-bounded | gets the best of both: respects boundaries **and** guarantees the size cap |
| **Semantic chunking** | embed sentences, cut where meaning shifts | higher quality on long, topic-shifting docs but much more expensive — overkill for short forensic docs |

### Say this in the interview
> *"I used **token-aware recursive chunking**. Recursive means I split on the coarsest natural
> boundary — paragraph, then line, then sentence, then word — that keeps each piece under the
> token budget, recursing finer only where needed. I size it with the **model's own tokenizer**
> so chunks never exceed its 512-token window and get silently truncated. I chose it over
> fixed-size (which cuts mid-thought) and pure sentence/paragraph splitting (which has no size
> ceiling); semantic chunking would help on long documents but these are short, so recursive
> matches its quality at a fraction of the cost. Defaults: 400 tokens, 50 overlap."*

### If they push back
- *"Why 400/50?"* → "400 keeps a chunk topically tight and leaves headroom under 512 for the
  model's special tokens; 50 (~12.5%) overlap stops a fact from being orphaned at a boundary."
- *"What's the downside of overlap?"* → "Mild storage/duplication and a fact can appear in two
  chunks — fine here; I dedupe to documents at eval/serve time."
- *"What if a single sentence exceeds the budget?"* → "It falls through to word-level splitting;
  the embedder would truncate an extreme outlier, and I warn if `chunk_size` leaves no room."

---

## 2. Embeddings & the model

### What an embedding is
A function that turns text into a fixed-length list of numbers (a **vector**, here 384 of them)
positioned so that **similar meaning → nearby vectors**. "Find me a car" and "locate a vehicle"
land close together even with no shared words. That's what makes *semantic* search work — you
compare meaning, not keywords.

### Bi-encoder vs cross-encoder (know this cold)
- **Bi-encoder** (our embedder): encodes the query and each passage **independently** into one
  vector each; relevance = similarity of the two vectors. Because passages are encoded *once* at
  ingest and indexed, a query is one forward pass + a fast lookup → **scales to millions**. Less
  accurate (it never sees query and passage together).
- **Cross-encoder** (our reranker, §9): feeds query **and** passage *together* through the model
  and outputs one relevance score. Sees their interaction → **more accurate**, but must run once
  per (query, passage) pair → can't precompute → only viable on a small candidate set.

This bi-/cross- distinction is the backbone of the whole "retrieve cheaply, then rerank" design.

### The model — `BAAI/bge-small-en-v1.5`
A compact **BERT-based** encoder (~33M params, **384-dim** output, **512-token** context). It
takes the chunk's tokens through Transformer self-attention layers and uses the **`[CLS]` token**'s
final vector as the embedding (CLS pooling), then **L2-normalizes** it. It was trained
contrastively — pulling matching (query, passage) pairs together and pushing mismatches apart —
which is exactly the geometry similarity search exploits.

**Asymmetric prefix (subtle, high-signal):** bge is instruction-tuned for retrieval — you prefix
the **query** with `"Represent this sentence for searching relevant passages: "` and leave
passages bare. We keep both prefixes in config, so swapping models (e5 prefixes both, MiniLM
neither) is a config edit.

### Why bge-small (vs alternatives)
- vs **all-MiniLM-L6-v2** (the brief's suggestion, kept as a fallback): bge-small ranks higher on
  **MTEB** (the standard embedding benchmark), has a 512-token context (MiniLM ~256), same 384-d.
- vs **bge-base/large, e5, gte**: stronger but 768–1024-d and heavier; we spent the accuracy
  budget on the reranker instead (it buys more per millisecond — the eval shows this).
- vs **OpenAI/Cohere**: banned by the on-prem rule.

### Say this
> *"Embeddings are learned vectors where distance ≈ semantic similarity. I used a **bi-encoder**,
> `bge-small`, because passages embed once and index, so search is cheap. It's a compact BERT,
> 384-dim, 512-token context, CLS-pooled and L2-normalized. I picked it over MiniLM because it
> scores higher on MTEB with double the context, and over the big models because I'd rather spend
> the latency budget on a reranker. The model is swappable by config — I even handle bge's
> query-side instruction prefix that way."*

### If they push back
- *"Why not a bigger embedder?"* → "Diminishing returns vs latency/RAM; the reranker recovers
  more accuracy here. `bge-base` + int8 quantization is wired as the upgrade path."
- *"What's the 384 for?"* → "The embedding dimensionality — smaller index, faster ANN, slightly
  less expressive than 768."

---

## 3. Cosine similarity (the distance metric)

### What it is
The measure of how aligned two vectors are — the **cosine of the angle** between them. Range −1
(opposite) … 0 (unrelated) … 1 (same direction). It cares about **direction, not magnitude**, so
a long document and a short one about the same topic still match.

Formula: `cos(a,b) = (a·b) / (|a|·|b|)`. If you **L2-normalize** vectors first (make `|a|=|b|=1`),
this collapses to just the **dot product** `a·b` — cheaper and numerically stable. We normalize at
embed time, so cosine == dot product. The brief requires cosine; Qdrant's collection is configured
`Distance.COSINE`.

### Say this
> *"Cosine similarity — the angle between vectors, so it's magnitude-invariant, which suits text
> of varying length. I L2-normalize embeddings so cosine reduces to a dot product. It's the metric
> the brief required and what the Qdrant collection uses."*

---

## 4. Vector database & ANN / HNSW

### The problem
You have 150 (eventually millions of) vectors. For a query vector, you want the closest ones.
Comparing against **all** of them is **O(N)** — too slow at scale. So vector DBs use **ANN
(Approximate Nearest Neighbour)**: trade a tiny bit of exactness for massive speed.

### HNSW — how the index works (the intuition)
Qdrant uses **HNSW: Hierarchical Navigable Small World** graphs. Think of it as a **skip-list for
geometry**, or a road network:
- Vectors are nodes connected to their near neighbours.
- There are **layers**: the **top** layer is sparse with **long-range** links (highways); lower
  layers get denser with **short-range** links (local streets); the **bottom** has every node.
- A search **starts at the top**, greedily hops to whichever neighbour is closest to the query,
  and **drops a layer** when it can't get closer — covering huge distances cheaply up top, then
  refining locally at the bottom. Expected query time is **logarithmic** in N.

**The knobs (exposed in `config.yaml::qdrant`):**
- `m` — links per node (higher = better recall, more memory).
- `ef_construct` — candidate-list size while **building** (higher = better graph, slower build).
- `hnsw_ef` — candidate-list size while **searching** (the direct recall⇄speed dial at query time).
- `quantization` — optional int8 compression of vectors (~4× less RAM, small recall cost).

### Why a database (Qdrant), not a library (FAISS)
- **FAISS** is an ANN **library** — just the index. No metadata store, no filtering, no server.
  You'd build the metadata/filtering layer yourself.
- **Chroma** — simplest, in-process, but weaker hybrid/filtering story.
- **Qdrant (chosen)** — a real **database**: ANN **+** payload store **+** metadata filtering **+**
  native sparse vectors and server-side fusion (hybrid) **+** one-command Docker, Apache-2.0,
  self-hosted (satisfies "no managed cloud vector DB").

### Say this
> *"A vector DB does approximate nearest-neighbour search so you don't scan every vector. Qdrant
> uses **HNSW** — a layered proximity graph you traverse greedily, like a road network from
> highways down to local streets, giving log-time search. I chose Qdrant over FAISS because FAISS
> is just an index — Qdrant adds the payload store, metadata filtering, and native hybrid I needed,
> in one self-hosted container."*

### If they push back
- *"What does 'approximate' cost you?"* → "A small recall hit vs exhaustive search; you tune it
  with `ef` — higher ef = closer to exact, slower."
- *"How does it scale?"* → "HNSW is ~log(N) query; quantization + segments keep memory bounded."

---

## 5. Metadata filtering — filterable HNSW (the "undervalued" part)

### What it is
Constraining a semantic search by structured fields — `doc_type=witness_statement`,
`case_id=2024-7812`, a `date` range. The brief flags this as undervalued; forensic analysts always
constrain by case/date/type.

### The subtlety that shows depth
Naïve filtering is **post-filtering**: retrieve the top-k vectors, then drop the ones that don't
match. If the filter is selective, you can retrieve 10 and keep 0 — recall collapses. Qdrant does
**pre-filtering integrated into the HNSW traversal**: it checks the filter *during* graph search,
so you still get the true top-k **within** the filtered subset. We declare **payload indexes**
(`doc_type`/`case_id` as keyword, `date` as **datetime** so ranges work) so this is fast.

Our filter builder (`api/filters.py`) also: turns an exact `date` into a **half-open day window**
`[day, day+1)` (a bare equality is a zero-width midnight instant that drops same-day docs), and
**allow-lists** fields to the indexed set so you can't filter on raw `text`.

### Say this
> *"Metadata filtering is pre-filtered **inside** the HNSW traversal, not post-hoc — so a selective
> filter doesn't wreck recall. I index `doc_type`/`case_id` as keywords and `date` as a datetime so
> range queries work, and I handle the exact-day-as-half-open-range gotcha."*

---

## 6. BM25 — sparse lexical retrieval

### What it is
The classic **keyword** ranking function (the thing search engines used before embeddings). It
scores a document by the **query words it literally contains**, weighting each by three factors:

1. **TF (term frequency) with saturation** — more occurrences help, but with **diminishing
   returns** (controlled by `k1`): a word appearing 20× isn't 20× more relevant than once.
2. **IDF (inverse document frequency)** — **rare** words count far more than common ones. Matching
   "gunmetal" is worth much more than matching "the".
3. **Length normalization** — divide out document length (parameter `b`, relative to the **average
   document length** `avg_len`) so long docs don't win just by being long.

Formula (for intuition, you don't need to recite it):
```
score(D,Q) = Σ_terms  IDF(q) · ( f(q,D)·(k1+1) ) / ( f(q,D) + k1·(1 − b + b·|D|/avg_len) )
```

### Why it matters here
Forensic queries hinge on **rare exact tokens** — names, case IDs, a unique item ("gunmetal
Chevrolet"). A small dense model **blurs** those into nearby semantics; BM25 nails them. In our
eval, BM25 alone is the **strongest single retriever** (Hit@5 0.93).

We store BM25 as a **sparse vector** (term→weight) *inside* Qdrant via `fastembed`, so dense and
sparse never drift. **High-signal detail:** we set `avg_len` to match `chunk_size` (~400) — the
library default (256) would mis-normalize every chunk.

### Say this
> *"BM25 is lexical ranking — TF with saturation, IDF so rare terms dominate, and length
> normalization. It's a strong baseline for forensic text because queries hinge on rare exact
> tokens that a small dense model blurs. I run it as a sparse vector inside Qdrant and calibrated
> its `avg_len` to my chunk size, which the default gets wrong."*

---

## 7. Hybrid search & Reciprocal Rank Fusion (RRF)

### Why hybrid
Dense (semantic) and BM25 (lexical) fail differently: dense gets paraphrases, BM25 gets rare exact
tokens. **Hybrid** runs both and merges, to get the strengths of each.

### The problem merging causes
Dense cosine scores live in ~0–1; BM25 scores are unbounded and on a totally different scale. You
**can't just add them**. Two ways out:
- **Weighted score fusion**: normalize both, then `α·dense + (1−α)·bm25`. Works, but you must tune
  `α` per corpus and pick a normalization.
- **Reciprocal Rank Fusion (RRF)** — fuse **ranks**, not scores:
  ```
  RRF(d) = Σ_retriever  1 / (k + rank_retriever(d))
  ```
  No normalization, no tuning beyond `k`. Robust, and what we use — Qdrant runs RRF
  server-side from our two prefetch branches (`FusionQuery(RRF)`). **Honest detail to
  know:** Qdrant's `FusionQuery` exposes no `k`, so we don't set one — it uses Qdrant's
  built-in rank constant (a small default; **2** in the client's reference implementation),
  *not* the `k = 60` from the original paper. The worked example below uses `k = 60` only
  because that's the canonical textbook value — it illustrates the formula, not the exact
  constant our system runs.

### Worked RRF example (illustrative, k=60)
Dense ranks: A(1), B(2), C(3). BM25 ranks: C(1), A(2), B(4).
- RRF(A) = 1/61 + 1/62 = **0.0325**  ← high in *both* → wins
- RRF(C) = 1/63 + 1/61 = 0.0323
- RRF(B) = 1/62 + 1/64 = 0.0318

A document ranked well by **both** retrievers floats to the top, even if it was #1 by neither.
That's the whole point: agreement across independent signals wins. (With Qdrant's `k = 2`
the absolute numbers differ, but the ordering logic — agreement wins — is identical.)

### Say this
> *"Dense and BM25 scores aren't comparable, so I fuse **ranks** with RRF — sum of 1/(k+rank)
> across retrievers. It needs no score normalization or per-corpus weight tuning, and Qdrant runs
> it server-side from two prefetch branches. Weighted fusion is the alternative but you have to
> calibrate the weight."*

---

## 8. Cross-encoder reranking (the precision stage)

### What it is
First-stage retrieval (dense/hybrid) is **recall-oriented** — get the right doc *somewhere* in the
top ~50, cheaply. A **cross-encoder reranker** then **re-reads** each of those candidates *together*
with the query and re-scores for **precision** — getting the best one to rank #1.

Why it's a *second* stage: a cross-encoder concatenates `[query] [SEP] [passage]` and runs full
attention, so it models their interaction (far more accurate) — but it can't precompute, so it's
O(candidates) per query. You only afford it on the top-N, not the whole corpus.

### Impact (from our eval)
Reranking lifted **Hit@1 from 0.53 → 0.73** and gave the best MRR (0.814) — the single biggest
accuracy lever on this corpus, fully local (`bge-reranker-base`).

### Say this
> *"I added a **cross-encoder reranker** as a precision stage. The bi-encoder retrieves the top-N
> for recall cheaply; the cross-encoder reads query and passage together to reorder them — much
> more accurate but O(N) per query, so it only runs on the candidates. It moved Hit@1 from 0.53 to
> 0.73 — the biggest single gain — and stays on-prem."*

---

## 9. Evaluation — Hit@K, MRR, confidence intervals

### Hit@K
Binary per query: **1 if the correct document is in the top-K results, else 0.** Report the average
over all queries. `Hit@1` = "was it the very top result?"; `Hit@5` = "was it in the top 5?".

### MRR (Mean Reciprocal Rank)
For each query, take the **reciprocal of the rank** of the first correct hit (rank 1 → 1.0, rank 2
→ 0.5, rank 3 → 0.333, not found → 0), then **average** across queries. Unlike Hit@K, it rewards
ranking the answer **higher**, not just being present.

**Worked example** — 3 queries, correct doc at ranks 1, 3, and not-found:
- RR = 1.0, 0.333, 0.0 → **MRR = (1.0+0.333+0.0)/3 = 0.444**
- Hit@1 = 1/3 = 0.33 (only the first); Hit@5 = 2/3 = 0.67 (first two).

### Wilson confidence interval (the maturity signal)
With only **30 queries**, a number like "Hit@5 = 0.97" is **noisy** — one query is 3.3%. A
**Wilson 95% CI** gives the range the true value plausibly sits in. E.g. 29/30 ≈ 0.967 has a 95% CI
of roughly **[0.83, 0.99]** — so you should *not* claim it's beating a 0.90 retriever. Reporting the
interval (not just the point) is what reads as engineering maturity.

### Say this
> *"Hit@K is binary top-K presence; MRR averages 1/rank of the first hit, so it rewards ranking the
> answer higher. At n=30 I report **Wilson confidence intervals** because point estimates are noisy
> at that size — I'd rather show 0.93 [0.79–0.98] than overclaim 0.93."*

---

## 10. Eval methodology — the circularity trap (your sharpest talking point)

### The trap
If you build synthetic data **and** the ground-truth queries by copying a phrase straight out of
the target document, then BM25 wins by **verbatim string match** and your eval proves nothing — it
measures "can the system find a string it was handed," not retrieval quality. This is **evaluation
leakage / circularity**, and most synthetic-data evals fall into it by accident.

### What we did
Ground-truth queries **paraphrase** the planted detail in lexically-disjoint words — e.g. the
document says *"a faded blue tarpaulin in the bed"*, the query asks for *"a weathered sheet covering
the cargo area."* A test asserts **0 paraphrase queries are verbatim substrings** of their targets.
We also tag queries by **category** (paraphrase vs rare-token entity) and report them separately,
and we split filter **precision** (returned docs that match the filter) from **recall** (the right
doc survives). The eval narrative is generated **from the measured numbers**, so it can't claim a
win it didn't get.

### Say this
> *"The eval's most important property is that it's **not circular**. I paraphrase the queries so
> they don't share surface words with the answer — otherwise BM25 wins by string-matching and the
> numbers are a tautology. I verify zero verbatim overlap in a test, break results out by query
> type, and separate filter precision from recall. That's the difference between measuring
> retrieval and measuring leakage."*

This single point often impresses more than any model choice — it shows you think like an evaluator.

---

## 11. Idempotency — deterministic IDs

### What it is
**Idempotent** = running the pipeline twice gives the same result as once (no duplicates). The
brief requires it.

### How we do it
Each chunk's storage ID is a **UUID5** (a name-based UUID — a hash) of `"{source_file}:{chunk_index}"`.
Same input → same ID → Qdrant **upsert** overwrites that point instead of inserting a new one. Re-run
→ stable count. We also **sweep orphans**: before re-ingesting a file we delete its old points, so an
edited file that now yields *fewer* chunks doesn't leave stale leftovers.

### Say this
> *"Idempotency comes from **content-addressed IDs**: a deterministic UUID5 of (source_file,
> chunk_index). Re-ingesting maps to the same ID and upsert overwrites — no duplicates. I also sweep
> a file's old points before re-adding it, to handle edits that shrink a document."*

---

## 12. Production & on-prem patterns (quick hits)

- **Singletons in the app lifespan** — models load **once** at startup, injected via FastAPI
  `Depends`, never per request. *"You never want to load a 100MB model on every request."*
- **Config-driven** (`pydantic-settings`, `config.yaml` + `RAG__*` env overrides) — model, host,
  port, collection are all config; nothing cloud is hard-coded. The on-prem requirement is "a
  first-class concern, not an afterthought" (their words).
- **Air-gappable** — `rag fetch-models` warms the cache once; `local_files_only` then forbids any
  network call. That's on-prem taken literally (forensic gear is often disconnected).
- **Error handling** — input validation → 422; store outage → 503; `/health` never throws. *"A
  health check that 500s is worse than useless."*

---

## 13. Rapid-fire — likely questions & crisp answers

- **"Why chunk at all?"** → Models have a fixed context window, and a chunk-level embedding is a
  cleaner topic representation than a whole-document average; you also return the precise passage.
- **"Dense vs sparse — when does each win?"** → Dense wins on paraphrase/synonyms; sparse (BM25)
  wins on rare exact tokens (names, IDs). Hybrid gets both; my eval shows it per category.
- **"Why cosine not Euclidean?"** → Text embeddings care about direction/topic, not magnitude;
  with normalized vectors cosine = dot product and it's the conventional, required metric.
- **"How do you know it works?"** → I measured it: Hit@1/Hit@5/MRR with CIs, dense vs BM25 vs hybrid
  vs reranked, plus filter precision/recall — over paraphrased (non-circular) queries.
- **"Biggest weakness?"** → The small embedder is weak on pure paraphrase (Hit@5 0.40 on that
  subset); the reranker compensates, and `bge-base` is the wired upgrade. The synthetic corpus is
  also templated, so generic queries can't discriminate — a real corpus would.
- **"How does it scale to millions of docs?"** → HNSW is ~log(N) query; quantization caps memory;
  Qdrant shards/segments; the embedder is the throughput bottleneck at ingest (batch + GPU).
- **"Why no LLM/generation?"** → It was optional; I focused on retrieval quality (the graded core).
  Adding Ollama to generate grounded answers over the retrieved chunks is the next step.
- **"What would you do with more time?"** → A held-out multi-seed eval (tighter CIs) and
  **fine-tuning the reranker** on in-domain pairs — my A/B sweep (§14) showed a *bigger*
  first-stage embedder does **not** help here, so the reranker is where capacity pays off —
  plus the local-LLM answer layer.

---

## 14. Final status — what shipped, and what the experiments proved

The project is **complete and verified against the requirements** (all functional + non-functional
parts, plus both bonuses — hybrid search and the UI). The forensic-critical property holds: the
system **only ever returns verbatim retrieved text and never generates or paraphrases** — there is
no LLM in the serving path, so there is **no hallucination surface**. Chunk text is the exact
`document[start:end]` slice (verified across all 150 chunks), reranking changes a hit's *score*
only, never its text.

What I can defend with data, not opinion:

- **The reranker — not the embedder — is the retrieval lever.** Over 30 paraphrased queries the
  cross-encoder lifts Hit@1 from 0.57 (hybrid) to **0.73** and gives the best MRR (**0.813**).
- **Scaling the embedder up does not help here.** An A/B sweep (§ this file references
  [`06_model_sweep.md`](06_model_sweep.md)) showed `bge-base` (768-d, ~4× the parameters) ties
  `bge-small` on the shipped pipeline; the small model is the right default.
- **Chunking has no upside on this corpus** because 90/120 documents are a single chunk — measured,
  not assumed. It matters at scale / on long documents.
- **The confidence floor is calibrated.** The reranker score is a usable confidence: a precise
  query (`man wearing a beanie`) scores ~0.95, a vague one (`man with a hat`) ~0.2 on the same
  documents; the API/UI `min_score` floor cleanly separates them.

If asked *"is it finished and correct?"*: yes — 40 tests pass (incl. a live-Qdrant idempotency
suite), the eval is reproducible (`rag eval` regenerates the exact numbers in `03_eval_results.md`),
and every documented metric back-derives to a real integer hit count. The honest limitations are the
small eval set (n=30, wide CIs) and the templated synthetic corpus — both disclosed in the docs.

---

## 15. One-line glossary

- **Embedding** — text → vector where distance ≈ meaning similarity.
- **Bi-encoder** — encodes query & passage separately (fast, precomputable, less accurate).
- **Cross-encoder** — encodes them together (slow, accurate, rerank-only).
- **Chunk** — a sub-document piece sized to the model, the unit you embed/retrieve.
- **Recursive splitting** — split on coarsest natural boundary that fits the size budget, recurse finer.
- **Token** — the model's sub-word unit; the context limit (512) is counted in these.
- **Cosine similarity** — angle between vectors; magnitude-invariant; = dot product when normalized.
- **ANN** — approximate nearest-neighbour search (trade tiny accuracy for big speed).
- **HNSW** — the layered proximity-graph index Qdrant uses; ~log(N) search.
- **Payload** — the metadata stored alongside a vector (doc_type, case_id, date, …).
- **BM25** — lexical ranking: TF-saturation × IDF × length-normalization.
- **Sparse vector** — term→weight representation (mostly zeros) used for BM25 inside the store.
- **RRF** — Reciprocal Rank Fusion; merge retrievers by Σ 1/(k+rank).
- **Reranker** — second-stage cross-encoder that reorders top-N for precision.
- **Hit@K** — was the right doc in the top K? **MRR** — average of 1/rank of the first hit.
- **Wilson interval** — confidence interval for a proportion; honest at small n.
- **Idempotent** — re-running changes nothing (here: deterministic UUID5 ids + upsert).
- **RAG** — Retrieval-Augmented Generation; this project is the **retrieval** half.

---

*Pair this with [`04_design_rationale.md`](04_design_rationale.md) for the primary-source papers
behind each method (Sentence-BERT, BGE/C-Pack, HNSW, BM25, RRF, cross-encoder reranking).*

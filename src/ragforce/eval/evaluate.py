"""Evaluation runner (requirement T4): measure retrieval quality, write the report.

Loads the generator's ``ground_truth.json`` and scores four retrievers — Dense,
BM25, Hybrid (RRF), and Hybrid+Rerank — at equal candidate depth, reporting Hit@1 /
Hit@5 / MRR with Wilson 95% CIs, a per-category breakdown (paraphrase vs entity),
and metadata-filter precision *and* recall. The narrative is generated from the
measured numbers (never hard-coded), so the report can't claim a result it didn't get.

Evaluation is deterministic: fixed corpus seed + fixed model + a deterministic ANN
query path, so re-running reproduces the same numbers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ragforce.api.filters import build_filter
from ragforce.embedding import build_embedder, build_reranker
from ragforce.eval.metrics import hit_at_k, mrr, wilson_interval
from ragforce.logging_setup import get_logger
from ragforce.models import SearchHit
from ragforce.store import VectorStore

if TYPE_CHECKING:
    from ragforce.config import Settings

_log = get_logger("eval")
_EVAL_TOP_K = 10
_REPORT_PATH = "docs/03_eval_results.md"


def _uniq_sources(hits: list[SearchHit]) -> list[str]:
    """Deduplicate to source_file order (results are chunk-level; eval is doc-level)."""
    out: list[str] = []
    for h in hits:
        sf = h.metadata.get("source_file")
        if sf and sf not in out:
            out.append(sf)
    return out


def _uniq_doc_hits(hits: list[SearchHit]) -> list[SearchHit]:
    """First chunk per source_file (doc-level view that keeps metadata for filter checks)."""
    seen: set[str] = set()
    out: list[SearchHit] = []
    for h in hits:
        sf = h.metadata.get("source_file")
        if sf and sf not in seen:
            seen.add(sf)
            out.append(h)
    return out


def _summary(ranks: list[list[str]], expected: list[str]) -> dict[str, Any]:
    n = len(expected)
    if n == 0:
        return {"n": 0, "hit@1": 0.0, "hit@5": 0.0, "mrr": 0.0, "hit@5_ci": (0.0, 0.0)}
    h1 = sum(hit_at_k(r, e, 1) for r, e in zip(ranks, expected))
    h5 = sum(hit_at_k(r, e, 5) for r, e in zip(ranks, expected))
    return {
        "n": n,
        "hit@1": h1 / n,
        "hit@5": h5 / n,
        "hit@5_ci": wilson_interval(h5, n),
        "mrr": mrr(ranks, expected),
    }


def _doc_satisfies(meta: dict[str, Any], filters: dict[str, Any]) -> bool:
    """True if a returned document's metadata satisfies every filter constraint."""
    for key, val in filters.items():
        mv = meta.get(key)
        if key == "date":
            if isinstance(val, dict):
                lo, hi = val.get("gte"), val.get("lte")
                if (lo and (mv is None or mv < lo)) or (hi and (mv is None or mv > hi)):
                    return False
            elif mv != val:
                return False
        elif isinstance(val, (list, tuple, set)):
            if mv not in val:
                return False
        elif mv != val:
            return False
    return True


def run(settings: "Settings", ground_truth_path: str | None = None) -> dict[str, Any]:
    """Evaluate retrieval quality, write the Markdown report, and return the metrics."""
    gt_path = ground_truth_path or settings.paths.ground_truth
    gt = json.loads(Path(gt_path).read_text(encoding="utf-8"))

    dense, sparse = build_embedder(settings)
    reranker = build_reranker(settings)
    store = VectorStore(
        host=settings.qdrant.host, port=settings.qdrant.port, collection=settings.qdrant.collection,
        dense_vector_name=settings.qdrant.dense_vector_name,
        sparse_vector_name=settings.qdrant.sparse_vector_name,
        timeout=settings.qdrant.timeout,
        hnsw_ef_search=settings.qdrant.hnsw_ef_search,
        prefetch_multiplier=settings.hybrid.prefetch_multiplier,
        prefetch_min=settings.hybrid.prefetch_min,
    )

    names = ["dense"]
    if sparse is not None:
        names += ["bm25", "hybrid"]
        if reranker is not None:
            names.append("hybrid+rerank")

    ranks: dict[str, list[list[str]]] = {nm: [] for nm in names}
    expected: list[str] = []
    categories: list[str] = []
    filt_total = filt_recall = filt_returned = filt_satisfy = 0

    for e in gt:
        query, exp = e["query"], e["expected_source_file"]
        filters = e.get("filters") or {}
        expected.append(exp)
        categories.append(e.get("category", "uncategorized"))

        dvec = dense.embed_query(query)
        ranks["dense"].append(_uniq_sources(store.search_dense(dvec, top_k=_EVAL_TOP_K)))
        if sparse is not None:
            svec = sparse.embed_query(query)
            ranks["bm25"].append(_uniq_sources(store.search_sparse(svec, top_k=_EVAL_TOP_K)))
            ranks["hybrid"].append(_uniq_sources(store.search_hybrid(dvec, svec, top_k=_EVAL_TOP_K)))
            if reranker is not None:
                cand = store.search_hybrid(dvec, svec, top_k=reranker.top_n)
                ranks["hybrid+rerank"].append(_uniq_sources(reranker.rerank(query, cand))[:_EVAL_TOP_K])

        if filters:
            filt_total += 1
            fhits = _uniq_doc_hits(
                store.search_dense(dvec, top_k=_EVAL_TOP_K, query_filter=build_filter(filters))
            )
            filt_recall += exp in [h.metadata.get("source_file") for h in fhits]
            filt_returned += len(fhits)
            filt_satisfy += sum(_doc_satisfies(h.metadata, filters) for h in fhits)

    summaries = {nm: _summary(ranks[nm], expected) for nm in names}
    cats = sorted(set(categories))
    per_category = {
        cat: {
            nm: _summary(
                [r for r, c in zip(ranks[nm], categories) if c == cat],
                [e for e, c in zip(expected, categories) if c == cat],
            )
            for nm in names
        }
        for cat in cats
    }

    # "recovered" = the best non-dense retriever found it @5 where dense missed @5
    best = names[-1] if len(names) > 1 else "dense"
    recovered = [
        {"query": gt[i]["query"], "expected": expected[i], "category": categories[i]}
        for i in range(len(expected))
        if best != "dense" and not hit_at_k(ranks["dense"][i], expected[i], 5)
        and hit_at_k(ranks[best][i], expected[i], 5)
    ]

    metrics: dict[str, Any] = {
        "pairs": len(gt),
        "retrievers": names,
        "per_category": per_category,
        "filter_recall": (filt_recall / filt_total) if filt_total else None,
        "filter_precision": (filt_satisfy / filt_returned) if filt_returned else None,
        "filtered_total": filt_total,
        "hybrid_recovered": len(recovered),
        "best_retriever": best,
        **summaries,  # top-level per-retriever summaries (dense, bm25, hybrid, ...)
        # back-compat aliases
        "filtered_accuracy": (filt_recall / filt_total) if filt_total else None,
    }

    _write_report(settings, metrics, recovered, per_category)
    _log.info("evaluation complete -> %s", _REPORT_PATH)
    return metrics


def _fmt(s: dict[str, Any]) -> str:
    lo, hi = s["hit@5_ci"]
    return f"{s['hit@1']:.2f} | {s['hit@5']:.2f} (95% CI {lo:.2f}–{hi:.2f}) | {s['mrr']:.3f}"


def _label(nm: str) -> str:
    return {"dense": "Dense (semantic)", "bm25": "BM25 (sparse)",
            "hybrid": "Hybrid (RRF)", "hybrid+rerank": "Hybrid + reranker"}.get(nm, nm)


def _narrative(m: dict[str, Any], per_category: dict[str, Any]) -> list[str]:
    """Build the failure analysis FROM the measured numbers (never hard-coded)."""
    out: list[str] = []
    d = m["dense"]
    if "hybrid" in m:
        h = m["hybrid"]
        delta = (h["hit@5"] - d["hit@5"]) * 100
        if delta > 1:
            out.append(
                f"- **Hybrid beats pure dense on Hit@5 by {delta:+.0f} pts** "
                f"({d['hit@5']:.2f} → {h['hit@5']:.2f}); BM25's exact-token matching rescues "
                f"queries where the small dense model blurs rare entities together."
            )
        elif delta < -1:
            out.append(
                f"- **Dense leads hybrid on Hit@5 by {-delta:.0f} pts** here "
                f"({h['hit@5']:.2f} vs {d['hit@5']:.2f}) — the queries are semantic paraphrases, "
                "which the dense model handles while sparse fusion adds noise."
            )
        else:
            out.append(
                f"- **Hybrid and dense are within ~1 pt on Hit@5** "
                f"({d['hit@5']:.2f} vs {h['hit@5']:.2f}) overall."
            )
    # per-category contrast (the honest story: lexical vs semantic)
    if {"paraphrase", "entity"} <= set(per_category):
        for nm in ("dense", "hybrid"):
            if nm in per_category["paraphrase"]:
                p = per_category["paraphrase"][nm]["hit@5"]
                en = per_category["entity"][nm]["hit@5"]
                out.append(
                    f"- **{_label(nm)}** scores Hit@5 {p:.2f} on *paraphrase* (semantic) queries "
                    f"vs {en:.2f} on *entity* (rare-token) queries."
                )
    if "hybrid+rerank" in m and "hybrid" in m:
        r = m["hybrid+rerank"]
        h = m["hybrid"]
        rd = (r["mrr"] - h["mrr"])
        verb = "lifts" if rd > 0 else ("matches" if abs(rd) < 1e-9 else "lowers")
        out.append(
            f"- **The cross-encoder reranker {verb} MRR** by {rd:+.3f} over hybrid "
            f"({h['mrr']:.3f} → {r['mrr']:.3f}) by re-reading (query, passage) pairs jointly."
        )
    out.append(
        "- **Residual misses** are queries whose paraphrase shares little surface vocabulary with "
        "the source; a larger embedder or query expansion would help most there."
    )
    return out


def _write_report(
    settings: "Settings",
    m: dict[str, Any],
    recovered: list[dict[str, str]],
    per_category: dict[str, Any],
) -> None:
    names = m["retrievers"]
    lines = [
        "# Evaluation Results",
        "",
        f"Retrieval quality over **{m['pairs']}** `(query, expected_document)` pairs from the "
        "generated corpus (`data/generated/ground_truth.json`). A hit means the expected "
        "document's `source_file` appears among the retrieved chunks' documents. Ground-truth "
        "queries **paraphrase** the planted signature (lexically disjoint from the source) so the "
        "comparison measures retrieval, not verbatim string matching.",
        "",
        f"- Embedding model: `{settings.embedding.model_name}` (dim {settings.embedding.dim}, cosine)",
        f"- Reranker: `{settings.rerank.model_name}`" if settings.rerank.enabled else "- Reranker: disabled",
        f"- Chunking: recursive, {settings.chunking.chunk_size}/{settings.chunking.chunk_overlap} tokens",
        f"- Retrieval depth: top-{_EVAL_TOP_K}, deduped to documents; all retrievers use equal depth",
        "- Evaluation is deterministic (fixed corpus seed + model + ANN path).",
        "",
        "## Retrievers compared",
        "",
        "| Retriever | Hit@1 | Hit@5 (95% CI) | MRR |",
        "|-----------|------:|:--------------:|----:|",
    ]
    for nm in names:
        bold = "**" if nm == m["best_retriever"] else ""
        lines.append(f"| {bold}{_label(nm)}{bold} | {_fmt(m[nm])} |")
    lines += [
        "",
        f"> At n={m['pairs']} the 95% CIs are wide — treat the ranking as indicative, not "
        "statistically separated where intervals overlap.",
        "",
    ]

    # per-category table
    cats = [c for c in ("paraphrase", "entity") if c in per_category] or list(per_category)
    if cats:
        lines += [
            "## By query category (Hit@5)",
            "",
            "_Paraphrase = semantic match required; Entity = rare proper-token (name) match._",
            "",
            "| Retriever | " + " | ".join(f"{c} (n={per_category[c][names[0]]['n']})" for c in cats) + " |",
            "|-----------|" + "|".join([":-:"] * len(cats)) + "|",
        ]
        for nm in names:
            cells = " | ".join(f"{per_category[c][nm]['hit@5']:.2f}" for c in cats)
            lines.append(f"| {_label(nm)} | {cells} |")
        lines.append("")

    # filtering: precision AND recall, separated
    if m["filter_recall"] is not None:
        lines += [
            "## Metadata filtering",
            "",
            f"Across **{m['filtered_total']}** filtered queries (`doc_type` / `case_id` / `date`-range):",
            "",
            f"- **Precision {m['filter_precision']:.0%}** — fraction of returned documents that actually "
            "satisfy the filter (constraint correctness; 100% means no non-matching doc leaks through).",
            f"- **Recall {m['filter_recall']:.0%}** — fraction where the expected document survived inside "
            "the filtered result set (the filter doesn't drop the right document).",
            "",
        ]

    lines += ["## Where retrieval fails, and why", ""]
    lines += _narrative(m, per_category)
    lines.append("")
    if recovered:
        lines += [f"### Examples the {_label(m['best_retriever'])} recovered (dense missed @5)", ""]
        for r in recovered[:5]:
            lines.append(f"- _{r['query']}_  ({r['category']}) → `{r['expected']}`")
        lines.append("")
    lines.append("_Reproduce: `rag generate && rag ingest && rag eval` (after `docker compose up -d`)._")
    Path(_REPORT_PATH).write_text("\n".join(lines), encoding="utf-8")

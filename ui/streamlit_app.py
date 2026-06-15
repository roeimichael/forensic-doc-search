"""Minimal Streamlit search UI (bonus Part 6).

A thin HTTP client over the search API — no package imports, only httpx — so a
non-technical analyst can query the corpus without the command line.

Three explicit modes map 1:1 to the API endpoints (no inferring the endpoint from a
toggle), and corpus text is rendered with ``st.text`` (never markdown) so uncontrolled
forensic content can't break the layout or inject markup.

Run (API must be up):  streamlit run ui/streamlit_app.py
"""

from __future__ import annotations

import os

import httpx
import streamlit as st

API_URL = os.environ.get("RAG_API_URL", "http://localhost:8000")

_MODES = {
    "Semantic": "/search",
    "Metadata-filtered": "/search/filtered",
    "Hybrid (dense + BM25)": "/search/hybrid",
}

st.set_page_config(page_title="Forensic Document Search", layout="wide")
st.title("Forensic Document Search")
st.caption(f"On-prem semantic + metadata + hybrid retrieval (reranked server-side) · API: {API_URL}")

# ── query + mode ─────────────────────────────────────────────────────────────
query = st.text_input("Natural-language query", placeholder="e.g. a dark blue Ford pickup with a covered cargo bed")
c1, c2, c3 = st.columns([2, 1, 1])
mode = c1.radio("Search mode", list(_MODES), horizontal=True,
                help="Semantic = vector only · Metadata-filtered = vector + filters · Hybrid = dense + BM25 (RRF)")
top_k = c2.slider("Max results", 1, 20, 5, help="Upper bound on how many results to return.")
min_conf = c3.slider("Min confidence", 0.0, 1.0, 0.0, 0.05,
                     help="Hide results below this reranker relevance score (0–1). 0 = show all. "
                          "Try ~0.5 to keep only strong matches.")

# ── filters (only used by filtered / hybrid) ─────────────────────────────────
uses_filters = mode != "Semantic"
filters: dict = {}
if uses_filters:
    with st.expander("Metadata filters", expanded=True):
        fc1, fc2, fc3 = st.columns(3)
        doc_type = fc1.selectbox("doc_type", ["(any)", "witness_statement", "report", "transcript"])
        case_id = fc2.text_input("case_id", placeholder="e.g. 2024-7812")
        date = fc3.text_input("date (YYYY-MM-DD)", placeholder="e.g. 2024-01-15")
    if doc_type != "(any)":
        filters["doc_type"] = doc_type
    if case_id.strip():
        filters["case_id"] = case_id.strip()
    if date.strip():
        filters["date"] = date.strip()

search_clicked = st.button("Search", type="primary")


def _run_search() -> tuple[list[dict], str | None]:
    endpoint = _MODES[mode]
    payload: dict = {"query": query, "top_k": top_k}
    if min_conf > 0:
        payload["min_score"] = min_conf
    if uses_filters:
        payload["filters"] = filters
    try:
        resp = httpx.post(f"{API_URL}{endpoint}", json=payload, timeout=30.0)
    except httpx.HTTPError as e:
        return [], f"Could not reach the API at {API_URL} ({e}). Is it running?"
    if resp.status_code == 422:
        return [], f"Invalid request: {resp.json().get('detail')}"
    if resp.status_code == 503:
        return [], "The vector store is unavailable (is Qdrant up and the corpus ingested?)."
    if resp.status_code != 200:
        return [], f"API error {resp.status_code}: {resp.text[:300]}"
    return resp.json()["results"], None


if search_clicked and query.strip():
    results, error = _run_search()
    if error:
        st.error(error)                       # API down / bad request — distinct from "no matches"
    elif not results:
        # Semantic/hybrid over a non-empty corpus always returns nearest neighbours, so a
        # zero means we pre-filtered or post-filtered them away. Name the likely cause(s).
        causes = []
        if uses_filters and filters:
            active = " · ".join(f"{k} = {v}" for k, v in filters.items())
            causes.append(f"the **metadata filter** (`{active}`) excluded every document")
        if min_conf > 0:
            causes.append(f"no match reached the **confidence floor** ({min_conf:.2f})")
        if causes:
            st.info(
                "**0 results** — likely because " + " and ".join(causes) + ".\n\n"
                "Filters run *before* the vector search and the confidence floor runs *after* it. "
                "Clear the filters, lower *Min confidence*, or use **Semantic** mode to search the "
                "whole corpus by meaning."
            )
        else:
            st.info("No matching documents in the corpus.")
    else:
        st.success(f"{len(results)} result(s) via {_MODES[mode]}")
        st.caption(
            "Ranked by **relevance** (higher = better; the reranker's score). "
            "Each result is one matched **chunk**; the tags show its source document's metadata."
        )
        for i, r in enumerate(results, start=1):
            md = r["metadata"]
            with st.container(border=True):
                # header: rank + filename (st.text so the '__' in filenames isn't read as markdown)
                h = st.columns([0.6, 9.4])
                h[0].markdown(f"**#{i}**")
                h[1].text(md.get("source_file", "?"))
                # metadata tags — a labelled row so it's clear what each field is
                t = st.columns(4)
                for col, label, value in (
                    (t[0], "Doc type", md.get("doc_type", "?")),
                    (t[1], "Date", md.get("date", "?")),
                    (t[2], "Case ID", md.get("case_id", "?")),
                    (t[3], "Relevance", f"{r['score']:.3f}"),
                ):
                    col.caption(label)
                    col.text(value)
                # body — plain text (no markdown rendering) so corpus content can't inject markup
                st.caption("Matched text")
                st.text(r["text"])
elif search_clicked:
    st.warning("Enter a query first.")

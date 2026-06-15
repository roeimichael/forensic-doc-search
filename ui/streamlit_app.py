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
c1, c2 = st.columns([2, 1])
mode = c1.radio("Search mode", list(_MODES), horizontal=True,
                help="Semantic = vector only · Metadata-filtered = vector + filters · Hybrid = dense + BM25 (RRF)")
top_k = c2.slider("Results (top_k)", 1, 20, 5)

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
        st.info("No matching documents.")     # the search ran fine; nothing matched
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
                t[0].caption("Doc type");  t[0].text(md.get("doc_type", "?"))
                t[1].caption("Date");      t[1].text(md.get("date", "?"))
                t[2].caption("Case ID");   t[2].text(md.get("case_id", "?"))
                t[3].caption("Relevance"); t[3].text(f"{r['score']:.3f}")
                # body — plain text (no markdown rendering) so corpus content can't inject markup
                st.caption("Matched text")
                st.text(r["text"])
elif search_clicked:
    st.warning("Enter a query first.")

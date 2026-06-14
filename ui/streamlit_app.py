"""Minimal Streamlit search UI (bonus Part 6).

A thin HTTP client over the search API — no package imports, only httpx — so a
non-technical analyst can query the corpus without the command line.

Run (API must be up):  streamlit run ui/streamlit_app.py
"""

from __future__ import annotations

import os

import httpx
import streamlit as st

API_URL = os.environ.get("RAG_API_URL", "http://localhost:8000")

st.set_page_config(page_title="Forensic Document Search", layout="wide")
st.title("🔍 Forensic Document Search")
st.caption(f"On-prem semantic + metadata + hybrid retrieval · API: {API_URL}")

# ── query + options ──────────────────────────────────────────────────────────
query = st.text_input("Natural-language query", placeholder="e.g. witness statements mentioning a blue sedan")
col1, col2, col3 = st.columns(3)
top_k = col1.slider("Results (top_k)", 1, 20, 5)
hybrid = col2.toggle("Hybrid (dense + BM25)", value=True, help="Fuse semantic and keyword search (RRF).")
search_clicked = col3.button("Search", type="primary", use_container_width=True)

# ── filter panel (maps to /search/filtered) ──────────────────────────────────
with st.expander("Metadata filters", expanded=False):
    fc1, fc2, fc3 = st.columns(3)
    doc_type = fc1.selectbox("doc_type", ["(any)", "witness_statement", "report", "transcript"])
    case_id = fc2.text_input("case_id", placeholder="e.g. 2024-7812")
    date = fc3.text_input("date (YYYY-MM-DD)", placeholder="e.g. 2024-01-15")


def _filters() -> dict:
    f: dict = {}
    if doc_type != "(any)":
        f["doc_type"] = doc_type
    if case_id.strip():
        f["case_id"] = case_id.strip()
    if date.strip():
        f["date"] = date.strip()
    return f


def _endpoint_and_payload() -> tuple[str, dict]:
    filters = _filters()
    payload: dict = {"query": query, "top_k": top_k}
    if hybrid:
        payload["filters"] = filters
        return "/search/hybrid", payload
    if filters:
        payload["filters"] = filters
        return "/search/filtered", payload
    return "/search", payload


if search_clicked and query.strip():
    endpoint, payload = _endpoint_and_payload()
    try:
        resp = httpx.post(f"{API_URL}{endpoint}", json=payload, timeout=30.0)
        resp.raise_for_status()
        results = resp.json()["results"]
    except httpx.HTTPError as e:
        st.error(f"Search failed ({endpoint}): {e}. Is the API running at {API_URL}?")
        results = []

    st.write(f"**{len(results)}** result(s) via `{endpoint}`")
    for r in results:
        md = r["metadata"]
        st.markdown(
            f"**{md.get('source_file', '?')}**  ·  `{md.get('doc_type', '?')}`  ·  "
            f"{md.get('date', '?')}  ·  case `{md.get('case_id', '?')}`  ·  score **{r['score']:.3f}**"
        )
        st.write(r["text"])
        st.divider()
elif search_clicked:
    st.warning("Enter a query first.")

"""Minimal Streamlit search UI (bonus Part 6 — hook point, stub this step).

Per the team's tip ("a 50-line Streamlit app beats a half-built React dashboard"),
this is a thin HTTP client over the API — no package imports, only ``httpx``.

Planned UI (implementation step):
    * text input for the natural-language query + a "Search" button → POST /search
    * a filter panel (doc_type dropdown, date field) → POST /search/filtered
    * results list rendering chunk text, score, and key metadata
      (source_file, doc_type, date)

Run: ``streamlit run ui/streamlit_app.py`` (API must be up).
"""

from __future__ import annotations

API_URL = "http://localhost:8000"


def main() -> None:
    """Render the Streamlit app.

    TODO(T6.x): st.text_input + st.button → httpx.post(f"{API_URL}/search", ...);
    filter widgets → /search/filtered; render results with metadata.
    """
    raise NotImplementedError("Streamlit UI — implemented in a later step (T6.x)")


if __name__ == "__main__":
    main()

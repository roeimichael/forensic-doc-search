"""Render the architecture diagram (deliverable #8) -> docs/architecture.png.

Pure matplotlib so it is reproducible offline with no extra system deps
(no Graphviz). Run:  python scripts/make_diagram.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parents[1] / "docs" / "architecture.png"

# palette
INGEST = "#dbeafe"   # blue   — offline ingestion
INGEST_E = "#2563eb"
STORE = "#ffedd5"    # orange — storage
STORE_E = "#ea580c"
SERVE = "#dcfce7"    # green  — online serving
SERVE_E = "#16a34a"
CONFIG = "#f3e8ff"   # purple — cross-cutting config
CONFIG_E = "#9333ea"
INK = "#1f2937"


def box(ax, x, y, w, h, title, sub, face, edge):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.6,rounding_size=2",
            linewidth=1.8, edgecolor=edge, facecolor=face, zorder=2,
        )
    )
    cx = x + w / 2
    ax.text(cx, y + h - 3.4, title, ha="center", va="top",
            fontsize=11, fontweight="bold", color=INK, zorder=3)
    if sub:
        # body left-aligned at a fixed inset so multi-line technical lists line up
        ax.text(x + 2.8, y + h - 8.0, sub, ha="left", va="top",
                fontsize=8.0, color="#374151", zorder=3, linespacing=1.5)


def arrow(ax, p0, p1, color=INK, style="-|>", rad=0.0, lw=1.7):
    ax.add_patch(
        FancyArrowPatch(
            p0, p1, arrowstyle=style, mutation_scale=15,
            linewidth=lw, color=color,
            connectionstyle=f"arc3,rad={rad}", zorder=1,
        )
    )


def main() -> Path:
    fig, ax = plt.subplots(figsize=(13.5, 9.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    ax.text(50, 97.5, "Forensic Document Search — On-Prem RAG Architecture",
            ha="center", va="top", fontsize=15, fontweight="bold", color=INK)

    # on-prem boundary
    ax.add_patch(
        FancyBboxPatch(
            (1.5, 3), 97, 88.5,
            boxstyle="round,pad=0,rounding_size=2",
            linewidth=1.6, edgecolor="#9ca3af", facecolor="none",
            linestyle=(0, (6, 4)), zorder=0,
        )
    )
    ax.text(3.2, 89.6, "on-prem boundary  ·  no cloud APIs  ·  self-hosted",
            ha="left", va="top", fontsize=8.5, style="italic", color="#6b7280")

    # ── lane label: ingestion ────────────────────────────────────────────────
    ax.text(3.2, 85, "INGESTION  (offline, idempotent)", ha="left", va="center",
            fontsize=9, fontweight="bold", color=INGEST_E)

    box(ax, 3, 67, 20, 15, "Dataset Generator",
        "synthetic, seeded\ndoc_type · case_id · date\n+ ground_truth.json", INGEST, INGEST_E)
    box(ax, 27, 67, 18, 15, "Loaders",
        "txt · pdf · json · eml\nmetadata recovery\ngraceful skip", INGEST, INGEST_E)
    box(ax, 49, 67, 18, 15, "Chunker",
        "token-aware recursive\n400/50 tok\nstructure separators", INGEST, INGEST_E)
    box(ax, 71, 67, 25, 15, "Embedder",
        "Dense: bge-small-en-v1.5 (384-d, L2)\nSparse: BM25 (fastembed)\noffline-capable (local cache)", INGEST, INGEST_E)

    arrow(ax, (23, 74.5), (27, 74.5))
    arrow(ax, (45, 74.5), (49, 74.5))
    arrow(ax, (67, 74.5), (71, 74.5))

    # ── storage ──────────────────────────────────────────────────────────────
    ax.text(3.2, 53, "STORAGE", ha="left", va="center",
            fontsize=9, fontweight="bold", color=STORE_E)
    box(ax, 30, 38, 40, 14, "Qdrant  (Docker, persistent volume)",
        "named vectors: dense=cosine · sparse=BM25\npayload indexes: doc_type · case_id · date(datetime)\nUUID5 point ids -> idempotent upsert", STORE, STORE_E)

    # embedder -> qdrant (upsert)
    arrow(ax, (83, 67), (60, 52.2), color=STORE_E, rad=-0.18)
    ax.text(78, 60, "upsert", ha="left", va="center", fontsize=8,
            style="italic", color=STORE_E)

    # ── serving ──────────────────────────────────────────────────────────────
    ax.text(3.2, 31, "SERVING  (online)", ha="left", va="center",
            fontsize=9, fontweight="bold", color=SERVE_E)
    box(ax, 30, 13, 40, 16, "FastAPI  +  cross-encoder reranker",
        "POST /search · semantic\nPOST /search/filtered · metadata + date range\n"
        "POST /search/hybrid · dense + BM25 (RRF)\nrerank top-N (bge-reranker) · GET /health",
        SERVE, SERVE_E)

    # qdrant <-> api (query / hits)
    arrow(ax, (45, 38), (45, 29), color=SERVE_E)
    arrow(ax, (55, 29), (55, 38), color=STORE_E, rad=0.0)
    ax.text(43.5, 33.5, "query", ha="right", va="center", fontsize=8,
            style="italic", color=SERVE_E)
    ax.text(56.5, 33.5, "hits", ha="left", va="center", fontsize=8,
            style="italic", color=STORE_E)

    # clients: UI + Eval
    box(ax, 4, 13, 20, 16, "Streamlit UI",
        "query box\n+ filters\n+ mode select", SERVE, SERVE_E)
    box(ax, 76, 13, 20, 16, "Evaluation",
        "Hit@1 / Hit@5 / MRR\nsemantic vs hybrid\n-> 03_eval_results.md", SERVE, SERVE_E)

    arrow(ax, (30, 21), (24, 21), color=INK)   # api -> ui (response)
    arrow(ax, (24, 24.5), (30, 24.5), color=INK, rad=0.0)  # ui -> api (request)
    ax.text(27, 26.8, "HTTP", ha="center", va="center", fontsize=7.5,
            style="italic", color="#6b7280")
    arrow(ax, (70, 21), (76, 21), color=INK)   # api -> eval
    ax.text(73, 23, "HTTP", ha="center", va="center", fontsize=7.5,
            style="italic", color="#6b7280")

    # ── config (cross-cutting) ───────────────────────────────────────────────
    box(ax, 4, 56, 20, 8.5, "config.yaml + .env",
        "model · chunk · qdrant · paths", CONFIG, CONFIG_E)
    # one dashed feed up into ingestion, one down toward storage/serving
    arrow(ax, (14, 64.5), (10, 67), color=CONFIG_E, style="-|>", rad=0.0, lw=1.2)
    arrow(ax, (25, 60), (30, 47), color=CONFIG_E, style="-|>", rad=-0.15, lw=1.2)

    fig.savefig(OUT, dpi=150, bbox_inches="tight", pad_inches=0.35, facecolor="white")
    plt.close(fig)
    return OUT


if __name__ == "__main__":
    p = main()
    print(f"wrote {p}")

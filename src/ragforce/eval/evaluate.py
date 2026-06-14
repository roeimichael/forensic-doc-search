"""Evaluation runner (requirement T4): measure retrieval quality, write the report.

Loads the generator's ``ground_truth.json``, runs each query through both pure
semantic and hybrid retrieval, computes Hit@1 / Hit@5 / MRR, and writes a Markdown
report (``docs/03_eval_results.md``) including the semantic-vs-hybrid comparison
(T5.4) and a short qualitative failure analysis (T4.3).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ragforce.config import Settings


def run(settings: "Settings", ground_truth_path: str | None = None) -> dict:
    """Run the evaluation and write ``docs/03_eval_results.md``; return the metrics.

    TODO(T4.x):
        * load ground_truth pairs;
        * for each: embed query, run search_dense and search_hybrid (with filters
          where the pair specifies them), collect ranked source_files;
        * compute Hit@1/Hit@5/MRR for each retriever via metrics.py;
        * render a Markdown table + qualitative notes; write the report.
    """
    raise NotImplementedError("eval.run — implemented in a later step (T4.x)")

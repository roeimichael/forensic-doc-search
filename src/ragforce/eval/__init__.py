"""Evaluation package — Hit@K / MRR metrics + the evaluation runner."""

from __future__ import annotations

from ragforce.eval.metrics import hit_at_k, mrr, reciprocal_rank

__all__ = ["hit_at_k", "reciprocal_rank", "mrr", "run"]


def run(*args, **kwargs):  # thin re-export to avoid importing evaluate at package load
    from ragforce.eval.evaluate import run as _run

    return _run(*args, **kwargs)

"""Cross-encoder reranker — the single biggest retrieval-quality lever (on-prem).

First-stage retrieval (dense / hybrid) is recall-oriented and cheap; a cross-encoder
re-scores the top-N candidates by reading (query, passage) *together*, which a
bi-encoder cannot. Fully local (sentence-transformers ``CrossEncoder``), config-gated.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sentence_transformers import CrossEncoder

if TYPE_CHECKING:
    from ragforce.config import Settings
    from ragforce.models import SearchHit


class Reranker:
    """Re-score and reorder first-stage hits with a local cross-encoder."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        *,
        device: str = "cpu",
        top_n: int = 50,
        cache_folder: str | None = None,
        local_files_only: bool = False,
    ) -> None:
        self._model = CrossEncoder(
            model_name, device=device,
            cache_folder=cache_folder, local_files_only=local_files_only,
        )
        self._top_n = top_n

    @property
    def top_n(self) -> int:
        """How many first-stage candidates to fetch + rescore."""
        return self._top_n

    def rerank(self, query: str, hits: list["SearchHit"], *, top_k: int | None = None) -> list["SearchHit"]:
        """Return ``hits`` reordered by cross-encoder relevance (score := rerank score).

        Only the first ``top_n`` candidates are rescored (cost control); any tail
        beyond ``top_n`` is appended unchanged after the reranked head.
        """
        import dataclasses

        if not hits:
            return hits
        head, tail = hits[: self._top_n], hits[self._top_n :]
        scores = self._model.predict([(query, h.text) for h in head])
        reranked = sorted(
            (dataclasses.replace(h, score=float(s)) for h, s in zip(head, scores)),
            key=lambda h: h.score,
            reverse=True,
        )
        out = reranked + tail
        return out[:top_k] if top_k is not None else out


def build_reranker(settings: "Settings") -> Reranker | None:
    """Construct the reranker iff ``settings.rerank.enabled``; else ``None``."""
    r = settings.rerank
    if not r.enabled:
        return None
    e = settings.embedding
    return Reranker(
        r.model_name,
        device=e.device,
        top_n=r.top_n,
        cache_folder=e.models_dir,
        local_files_only=e.local_files_only,
    )

"""Dense embeddings via sentence-transformers (local inference, requirement T1.3).

Prefix-awareness is the mechanism that lets us swap models by config alone:
    * bge-small → query_prefix set, passage_prefix empty
    * e5        → both prefixes set ("query: " / "passage: ")
    * MiniLM    → both empty
The chunker borrows this class's ``tokenizer`` so chunk sizing tracks the model.
"""

from __future__ import annotations

from typing import Any


class DenseEmbedder:
    """Wrap a SentenceTransformer: prefix-aware, L2-normalized, batched."""

    def __init__(
        self,
        model_name: str,
        *,
        device: str = "cpu",
        query_prefix: str = "",
        passage_prefix: str = "",
        normalize: bool = True,
        max_seq_length: int | None = None,
    ) -> None:
        """Load the model locally (no network at inference) and apply config.

        TODO(T1.3): ``SentenceTransformer(model_name, device=device)``; set
        ``max_seq_length`` if given; stash prefixes + normalize flag.
        """
        raise NotImplementedError("DenseEmbedder.__init__ — implemented in the next step (T1.3)")

    @property
    def dim(self) -> int:
        """Embedding dimension reported by the model (asserted against config)."""
        raise NotImplementedError("DenseEmbedder.dim — implemented in the next step (T1.3)")

    @property
    def tokenizer(self) -> Any:
        """The model's HF tokenizer, handed to the Chunker as its length function."""
        raise NotImplementedError("DenseEmbedder.tokenizer — implemented in the next step (T1.3)")

    def embed_passages(self, texts: list[str], *, batch_size: int = 64) -> list[list[float]]:
        """Embed document chunks (prepends ``passage_prefix``).

        TODO(T1.3): prepend passage_prefix; ``model.encode(..., normalize_embeddings)``.
        """
        raise NotImplementedError("DenseEmbedder.embed_passages — next step (T1.3)")

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query (prepends ``query_prefix``).

        TODO(T1.3): prepend query_prefix; encode single text; return vector.
        """
        raise NotImplementedError("DenseEmbedder.embed_query — next step (T1.3)")

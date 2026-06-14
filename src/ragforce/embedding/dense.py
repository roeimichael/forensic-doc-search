"""Dense embeddings via sentence-transformers (local inference, requirement T1.3).

Prefix-awareness is the mechanism that lets us swap models by config alone:
    * bge-small → query_prefix set, passage_prefix empty
    * e5        → both prefixes set ("query: " / "passage: ")
    * MiniLM    → both empty
The chunker borrows this class's ``tokenizer`` so chunk sizing tracks the model.
"""

from __future__ import annotations

from typing import Any

from sentence_transformers import SentenceTransformer


class DenseEmbedder:
    """Wrap a SentenceTransformer: prefix-aware, L2-normalized, batched.

    Prefixes are the model-swap mechanism: bge sets a query prefix, e5 sets both,
    MiniLM sets neither — so changing model is a config edit, not a code change.
    """

    def __init__(
        self,
        model_name: str,
        *,
        device: str = "cpu",
        query_prefix: str = "",
        passage_prefix: str = "",
        normalize: bool = True,
        max_seq_length: int | None = None,
        cache_folder: str | None = None,
        local_files_only: bool = False,
    ) -> None:
        self._model = SentenceTransformer(
            model_name, device=device,
            cache_folder=cache_folder, local_files_only=local_files_only,
        )
        if max_seq_length is not None:
            self._model.max_seq_length = max_seq_length
        self._query_prefix = query_prefix
        self._passage_prefix = passage_prefix
        self._normalize = normalize
        self._max_seq_length = max_seq_length or self._model.max_seq_length

    @property
    def dim(self) -> int:
        """Embedding dimension reported by the model (asserted against config)."""
        # method renamed in sentence-transformers 5.x; keep both for version robustness
        get_dim = getattr(self._model, "get_embedding_dimension", None) or \
            self._model.get_sentence_embedding_dimension
        return int(get_dim())

    @property
    def max_seq_length(self) -> int:
        """Max input tokens the model accepts before silent truncation."""
        return int(self._max_seq_length)

    @property
    def tokenizer(self) -> Any:
        """The model's HF tokenizer, handed to the Chunker as its length function."""
        return self._model.tokenizer

    def embed_passages(self, texts: list[str], *, batch_size: int = 64) -> list[list[float]]:
        """Embed document chunks (prepends ``passage_prefix``), L2-normalized."""
        prefixed = [self._passage_prefix + t for t in texts]
        vecs = self._model.encode(
            prefixed, batch_size=batch_size, normalize_embeddings=self._normalize,
            show_progress_bar=False, convert_to_numpy=True,
        )
        return vecs.tolist()

    def embed_query(self, text: str) -> list[float]:
        """Embed a single search query (prepends ``query_prefix``), L2-normalized."""
        vec = self._model.encode(
            self._query_prefix + text, normalize_embeddings=self._normalize,
            show_progress_bar=False, convert_to_numpy=True,
        )
        return vec.tolist()

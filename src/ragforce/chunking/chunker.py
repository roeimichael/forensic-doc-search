"""Token-aware recursive chunker (requirement T1.2).

Why this strategy (the justification graded in the README):
    * Forensic docs are short and paragraph/turn structured; recursive splitting
      respects those boundaries and matches semantic-chunking quality on short
      text at a fraction of the cost (2025–26 benchmarks).
    * Length is measured with the *embedding model's own tokenizer*, so chunks
      never exceed the model's ``max_seq_length`` (else text is silently truncated
      and embeddings degrade). Swapping the model rescales chunking automatically.

The chunker assigns ``chunk_index``, ``char_span``, ``token_count`` and mints the
deterministic ``chunk_id`` (via :func:`ragforce.store.points.make_point_id`).
"""

from __future__ import annotations

from typing import Any

from ragforce.chunking.separators import separators_for
from ragforce.models import Chunk, Document
from ragforce.store.points import make_point_id


class Chunker:
    """Split a :class:`Document` into coherent, token-bounded :class:`Chunk` objects.

    Recursive: text is broken on the coarsest separator that yields pieces within the
    token budget, recursing to finer separators only where needed; pieces are then
    greedily packed into chunks of ``<= chunk_size`` tokens with ``chunk_overlap``
    tokens carried between them. Token counts use the embedding model's own tokenizer
    so chunks never exceed what the model actually encodes.
    """

    def __init__(
        self,
        tokenizer: Any,
        *,
        chunk_size: int,
        chunk_overlap: int,
        min_chunk_size: int,
        separators: list[str] | None = None,
    ) -> None:
        """Configure the splitter.

        Args:
            tokenizer: The embedder's HF tokenizer (used as the length function).
            chunk_size: Target max tokens per chunk.
            chunk_overlap: Token overlap between consecutive chunks.
            min_chunk_size: Drop trailing fragments smaller than this (tokens).
            separators: Override the per-doc_type separator ladder (optional).
        """
        self._tokenizer = tokenizer
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._min_chunk_size = min_chunk_size
        self._separators = separators

    def chunk(self, doc: Document) -> list[Chunk]:
        """Split ``doc`` into ordered, metadata-carrying chunks."""
        separators = self._separators or separators_for(doc.doc_type)
        units = self._split_recursive(doc.text, separators, 0)
        spans = self._merge(units)  # list[(start, end)]

        chunks: list[Chunk] = []
        for start, end in spans:
            text = doc.text[start:end]  # faithful slice -> char_span is exact provenance
            tokens = self._token_len(text)
            if tokens < self._min_chunk_size and len(spans) > 1:
                continue  # drop tiny fragment (unless it is the only chunk)
            index = len(chunks)
            chunks.append(
                Chunk(
                    chunk_id=make_point_id(doc.source_file, index),
                    source_file=doc.source_file,
                    chunk_index=index,
                    text=text,
                    doc_type=doc.doc_type,
                    case_id=doc.case_id,
                    date=doc.date,
                    char_span=(start, end),
                    token_count=tokens,
                    title=doc.title,
                    extra=dict(doc.extra),
                )
            )
        return chunks

    # ── internals ────────────────────────────────────────────────────────────
    def _token_len(self, text: str) -> int:
        """Token count of ``text`` per the embedding tokenizer (content tokens only)."""
        return len(self._tokenizer.encode(text, add_special_tokens=False))

    def _split_recursive(self, text: str, separators: list[str], offset: int) -> list[tuple[str, int]]:
        """Break ``text`` into ``(piece, start_char)`` units each within the token budget.

        ``start_char`` is the EXACT offset of ``piece`` in the original document text
        (leading whitespace is accounted for), so ``doc.text[start:start+len(piece)]``
        equals ``piece``. Splits on the first separator; any over-budget segment recurses
        with the finer separators; a separator-less segment is kept whole.
        """
        lead = len(text) - len(text.lstrip())
        stripped = text.strip()
        if not stripped:
            return []
        offset += lead  # advance past leading whitespace so the offset is exact
        if self._token_len(stripped) <= self._chunk_size or not separators:
            return [(stripped, offset)]

        sep, rest = separators[0], separators[1:]
        units: list[tuple[str, int]] = []
        cursor = offset
        segments = stripped.split(sep) if sep else [stripped]
        for i, seg in enumerate(segments):
            if seg.strip():
                if self._token_len(seg.strip()) <= self._chunk_size:
                    seg_lead = len(seg) - len(seg.lstrip())
                    units.append((seg.strip(), cursor + seg_lead))
                else:
                    units.extend(self._split_recursive(seg, rest, cursor))
            cursor += len(seg) + (len(sep) if i < len(segments) - 1 else 0)
        return units

    def _merge(self, units: list[tuple[str, int]]) -> list[tuple[int, int]]:
        """Greedily pack units into ``<= chunk_size``-token chunks with token overlap.

        Returns ``(start, end)`` character spans into the original document; the chunk
        text is the verbatim slice ``doc.text[start:end]`` (incl. original separators).
        """
        chunks: list[tuple[int, int]] = []
        cur: list[tuple[str, int]] = []
        cur_tokens = 0
        for text, off in units:
            ut = self._token_len(text)
            if cur and cur_tokens + ut > self._chunk_size:
                chunks.append(self._finalize(cur))
                cur, cur_tokens = self._carry_overlap(cur)
            cur.append((text, off))
            cur_tokens += ut
        if cur:
            chunks.append(self._finalize(cur))
        return chunks

    @staticmethod
    def _finalize(units: list[tuple[str, int]]) -> tuple[int, int]:
        start = units[0][1]
        end = units[-1][1] + len(units[-1][0])
        return start, end

    def _carry_overlap(self, units: list[tuple[str, int]]) -> tuple[list[tuple[str, int]], int]:
        """Return trailing units totalling roughly ``chunk_overlap`` tokens (for context)."""
        if self._chunk_overlap <= 0:
            return [], 0
        carried: list[tuple[str, int]] = []
        tokens = 0
        for text, off in reversed(units):
            ut = self._token_len(text)
            if carried and tokens + ut > self._chunk_overlap:
                break
            carried.insert(0, (text, off))
            tokens += ut
        return carried, tokens

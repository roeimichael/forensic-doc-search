"""Chunking tests (T1.2): token sizing, overlap, min-chunk drop, transcript Q:/A: splits.

Uses a whitespace "tokenizer" (1 token == 1 word) so the splitter logic is exercised
without loading a real model — fast and deterministic.
"""

from __future__ import annotations

from ragforce.chunking import Chunker
from ragforce.models import Document


class WordTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[str]:
        return text.split()


def _doc(text: str, doc_type: str = "report") -> Document:
    return Document(source_file="f.txt", text=text, doc_type=doc_type, case_id="c", date="2024-01-01")


def test_long_text_splits_into_multiple_bounded_chunks() -> None:
    text = ". ".join(f"sentence number {i} with several filler words here" for i in range(60))
    ch = Chunker(WordTokenizer(), chunk_size=40, chunk_overlap=8, min_chunk_size=4)
    chunks = ch.chunk(_doc(text))
    assert len(chunks) > 1
    assert all(c.token_count <= 40 for c in chunks)
    # indices are sequential and ids are deterministic for (source_file, index)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    from ragforce.store.points import make_point_id

    assert chunks[0].chunk_id == make_point_id("f.txt", 0)


def test_overlap_carries_context_between_chunks() -> None:
    text = " ".join(f"w{i}" for i in range(120))
    ch = Chunker(WordTokenizer(), chunk_size=30, chunk_overlap=10, min_chunk_size=2)
    chunks = ch.chunk(_doc(text))
    # consecutive chunks should share at least one trailing/leading token (overlap)
    first_tail = set(chunks[0].text.split()[-10:])
    second = set(chunks[1].text.split())
    assert first_tail & second


def test_short_text_is_single_chunk() -> None:
    ch = Chunker(WordTokenizer(), chunk_size=400, chunk_overlap=50, min_chunk_size=4)
    chunks = ch.chunk(_doc("a short witness statement with only a handful of words"))
    assert len(chunks) == 1
    assert chunks[0].char_span[0] <= chunks[0].char_span[1]

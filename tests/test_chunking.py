"""Chunking tests (T1.2): token sizing, overlap, min-chunk drop, transcript Q:/A: splits."""

from __future__ import annotations

import pytest


def test_chunks_respect_token_size_and_overlap() -> None:
    pytest.skip("scaffold — Chunker implemented in the next step (T1.2)")


def test_tiny_fragments_dropped() -> None:
    pytest.skip("scaffold — min_chunk_size implemented in the next step (T1.2)")


def test_transcript_separators_keep_turns() -> None:
    pytest.skip("scaffold — transcript separators implemented in the next step (T1.2)")

"""Core data contracts shared across every layer.

These three frozen dataclasses are the *only* vocabulary the pipeline speaks:

    Document  — one source file fully loaded, pre-chunking.
    Chunk     — one coherent slice of a Document; the unit that gets embedded + stored.
    SearchHit — one retrieval result; the API serializes it 1:1 to JSON.

Design choice: ``Chunk.chunk_id`` IS the Qdrant point id (a single identifier
everywhere), so there is never a second hash to reconcile between layers.

These are real (not stubs): they are cheap, dependency-free, and every later
module imports them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class Document:
    """One source file fully loaded, before chunking.

    Attributes:
        source_file: Basename of the file (e.g. ``"report__2024-7812__2024-01-15__theft.txt"``).
            This is the idempotency key and the eval ground-truth key.
        text: Full extracted text content.
        doc_type: Category — ``witness_statement`` | ``report`` | ``transcript``.
        case_id: Fictional case identifier (e.g. ``"2024-7812"``).
        date: Document date, ISO-8601 ``YYYY-MM-DD`` (normalized at load time).
        title: Optional human-readable title.
        source_path: Absolute path on disk (provenance / logging).
        extra: Any additional useful metadata captured during loading.
    """

    source_file: str
    text: str
    doc_type: str
    case_id: str
    date: str
    title: str | None = None
    source_path: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Chunk:
    """One coherent slice of a :class:`Document`; the unit embedded and stored.

    Attributes:
        chunk_id: Deterministic UUID5 string — also the Qdrant point id.
        source_file: Parent document's ``source_file`` (links chunk → document).
        chunk_index: 0-based position of this chunk within the document.
        text: The chunk's text.
        doc_type: Inherited from the parent document.
        case_id: Inherited from the parent document.
        date: Inherited from the parent document (ISO-8601).
        char_span: ``(start, end)`` character offsets into ``Document.text``.
        token_count: Number of tokens (measured with the embedding tokenizer).
        title: Inherited optional title.
        extra: Any additional per-chunk metadata.
    """

    chunk_id: str
    source_file: str
    chunk_index: int
    text: str
    doc_type: str
    case_id: str
    date: str
    char_span: tuple[int, int]
    token_count: int
    title: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SearchHit:
    """A single retrieval result.

    The API maps this 1:1 onto a JSON response item:
    ``{"chunk_id": ..., "score": ..., "text": ..., "metadata": {...}}``.

    Attributes:
        chunk_id: Id of the retrieved chunk/point.
        score: Relevance score (cosine similarity, or fused score for hybrid).
        text: The chunk text.
        metadata: The stored payload (source_file, doc_type, date, case_id, ...).
    """

    chunk_id: str
    score: float
    text: str
    metadata: dict[str, Any]

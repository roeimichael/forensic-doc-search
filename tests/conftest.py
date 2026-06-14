"""Shared pytest fixtures.

Sample ``Document`` / ``Chunk`` fixtures are real (cheap dataclasses) so unit
tests can be written against the contract as soon as logic lands.
"""

from __future__ import annotations

import pytest

from ragforce.models import Chunk, Document


@pytest.fixture
def sample_document() -> Document:
    return Document(
        source_file="report__2024-7812__2024-01-15__theft.txt",
        text="Officer responded to a reported theft at 14:30. A blue sedan was seen leaving.",
        doc_type="report",
        case_id="2024-7812",
        date="2024-01-15",
        title="Theft Report",
    )


@pytest.fixture
def sample_chunk() -> Chunk:
    return Chunk(
        chunk_id="00000000-0000-5000-a000-000000000000",
        source_file="report__2024-7812__2024-01-15__theft.txt",
        chunk_index=0,
        text="A blue sedan was seen leaving the scene.",
        doc_type="report",
        case_id="2024-7812",
        date="2024-01-15",
        char_span=(0, 40),
        token_count=9,
    )

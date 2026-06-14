"""The single ``Loader`` interface that every file format implements.

Each concrete loader owns *only* text extraction; shared metadata derivation lives
in :mod:`ragforce.loaders.metadata`. Adding a new format (e.g. ``.docx``) = one new
loader class + one registry line, no changes elsewhere.
"""

from __future__ import annotations

from typing import ClassVar, Protocol, runtime_checkable

from ragforce.models import Document


class LoaderError(Exception):
    """Raised when a file cannot be read or parsed.

    ``load_directory`` catches this and logs+skips the file so one bad document
    never crashes the pipeline (requirement T1.1).
    """


@runtime_checkable
class Loader(Protocol):
    """Structural interface: own an extension set, turn a path into a Document."""

    extensions: ClassVar[tuple[str, ...]]

    def load(self, path: str) -> Document:
        """Read ``path`` and return a fully-populated :class:`Document`.

        Raises:
            LoaderError: if the file is unreadable/corrupt/missing required fields.
        """
        ...

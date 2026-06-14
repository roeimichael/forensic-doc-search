"""Loader tests (T1.1): txt/pdf/json load correctly; corrupt files skipped-with-warning."""

from __future__ import annotations

import pytest


def test_txt_pdf_json_load_to_documents() -> None:
    pytest.skip("scaffold — loaders implemented in the next step (T1.1)")


def test_corrupt_file_is_skipped_not_raised() -> None:
    pytest.skip("scaffold — load_directory graceful-skip implemented next step (T1.1)")

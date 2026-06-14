"""Synthetic, real-text-seeded dataset generator (requirement T0.1).

Produces a controlled forensic corpus so retrieval can be measured honestly:
    * ``n`` docs (50–200) across doc_types {witness_statement, report, transcript}
      and formats {.txt, .pdf, .json}; PDFs via reportlab, JSON with a ``content`` field.
    * Realistic prose stitched from ``data/seeds/seed_snippets.jsonl`` (real public
      snippets), but metadata (doc_type, case_id, date) is OURS and known.
    * Filenames follow ``<doc_type>__<case_id>__<YYYY-MM-DD>__<slug>.<ext>`` so the
      loader recovers metadata deterministically.
    * Emits ``ground_truth.json`` = [{query, expected_source_file, filters?}] (≥10
      pairs, semantic + filtered) to drive the evaluation (T4).

Deterministic: a fixed ``seed`` makes the corpus reproducible.
"""

from __future__ import annotations

from pathlib import Path


def generate(n: int = 120, seed: int = 42, out_dir: str | Path = "data/generated") -> Path:
    """Generate the corpus + ground truth into ``out_dir``; return its Path.

    TODO(T0.1):
        * seed RNG; load seed snippets; for i in range(n): pick doc_type/format,
          synthesize text + metadata (case_id pool, date range), write the file
          under the filename convention.
        * build ≥10 ground-truth (query → expected_source_file[, filters]) pairs and
          write ground_truth.json.
        * return Path(out_dir).
    """
    raise NotImplementedError("generate — implemented in the next step (T0.1)")

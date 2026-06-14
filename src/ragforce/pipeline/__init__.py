"""Pipeline package — the ingestion orchestrator."""

from __future__ import annotations

from ragforce.pipeline.ingest import IngestStats, run_ingest

__all__ = ["IngestStats", "run_ingest"]

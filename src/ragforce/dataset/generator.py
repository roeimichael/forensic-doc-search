"""Synthetic, real-text-seeded dataset generator (requirement T0.1).

Produces a controlled forensic corpus so retrieval can be measured honestly:
    * ``n`` docs (50–200) across doc_types {witness_statement, report, transcript}
      and formats {.txt, .pdf, .json, .eml} (the required three + email evidence,
      which is on-theme for digital forensics and carries header metadata).
    * Realistic prose stitched from ``data/seeds/seed_snippets.jsonl`` (real-ish
      snippets), but metadata (doc_type, case_id, date) is OURS and known.
    * Filenames follow ``<doc_type>__<case_id>__<YYYY-MM-DD>__<NNN-slug>.<ext>`` so the
      loader recovers metadata deterministically; ``.json``/``.eml`` also embed it.
    * Plants a unique "needle" fact in selected docs and writes ``ground_truth.json``
      = [{query, expected_source_file, filters}] (>=10 pairs, semantic + filtered) to
      drive the evaluation (T4).

Deterministic: a fixed ``seed`` makes the corpus + ground truth reproducible.
Format writers are isolated (one ``_write_*`` per format), so adding a type later
is a single new writer + registry entry.
"""

from __future__ import annotations

import datetime
import json
import re
from email.message import EmailMessage
from email.utils import format_datetime
from pathlib import Path
from random import Random
from typing import Any, Callable
from xml.sax.saxutils import escape

DOC_TYPES = ("witness_statement", "report", "transcript")
FORMATS = ("txt", "pdf", "json", "eml")  # required txt/pdf/json + email evidence

_DEFAULT_SEED_FILE = "data/seeds/seed_snippets.jsonl"
_DATE_START = datetime.date(2023, 6, 1)
_DATE_END = datetime.date(2024, 12, 31)

# Minimal fallback prose if the seed file is missing, so generation never hard-fails.
_FALLBACK = {
    "witness_statement": "I observed the events from across the street and reported what I saw.",
    "report": "Officers attended the scene and documented the conditions on arrival.",
    "transcript": "Q: Can you describe what happened?\nA: I will recount the events in order.",
}

# Planted facts → each becomes one ground-truth (query, expected_doc[, filters]) pair.
# ``filter`` selects which metadata constraint the eval applies to that query.
_NEEDLES: list[dict[str, Any]] = [
    {"doc_type": "witness_statement", "filter": "doc_type",
     "fact": "I clearly saw a blue sedan with a dented rear bumper parked across the street.",
     "query": "witness descriptions of a blue sedan with a dented rear bumper"},
    {"doc_type": "report", "filter": "doc_type",
     "fact": "The point of entry was a broken rear window on the ground floor.",
     "query": "reports where the point of entry was a broken rear window"},
    {"doc_type": "report", "filter": None,
     "fact": "A partial shoe impression was recovered near the doorway and preserved for analysis.",
     "query": "a partial shoe impression preserved near the doorway"},
    {"doc_type": "witness_statement", "filter": None,
     "fact": "A silver panel van was idling beside the loading dock for several minutes.",
     "query": "a silver van parked near a loading dock"},
    {"doc_type": "witness_statement", "filter": "doc_type",
     "fact": "The suspect wore a grey hooded sweatshirt and carried a black backpack.",
     "query": "suspect in a grey hooded sweatshirt carrying a black backpack"},
    {"doc_type": "transcript", "filter": "doc_type",
     "fact": "A: The front office had been ransacked and every drawer was pulled open.",
     "query": "office ransacked with the drawers pulled open"},
    {"doc_type": "witness_statement", "filter": None,
     "fact": "The licence plate was partially obscured by mud, but it began with the letters KR.",
     "query": "a licence plate obscured by mud beginning with KR"},
    {"doc_type": "witness_statement", "filter": None,
     "fact": "Two individuals ran from the scene and fled on foot toward the highway.",
     "query": "two people fleeing on foot toward the highway"},
    {"doc_type": "report", "filter": "case",
     "fact": "Fresh pry marks were visible on the door frame, consistent with forced entry.",
     "query": "forced entry with pry marks on the door frame"},
    {"doc_type": "report", "filter": "date",
     "fact": "A CCTV camera mounted above the parking lot captured the entire incident.",
     "query": "CCTV camera overlooking the parking lot"},
    {"doc_type": "witness_statement", "filter": None,
     "fact": "I was woken around 2 a.m. by the sound of glass shattering downstairs.",
     "query": "witness woken by glass shattering around 2 a.m."},
    {"doc_type": "report", "filter": None,
     "fact": "A backpack containing assorted hand tools was recovered from the rear alley.",
     "query": "a recovered backpack containing tools"},
    {"doc_type": "transcript", "filter": "doc_type",
     "fact": "A: He kept referring to someone named Marcus throughout the argument.",
     "query": "suspect mentioning a person named Marcus"},
    {"doc_type": "witness_statement", "filter": None,
     "fact": "The getaway vehicle was a red pickup truck with a cracked windshield.",
     "query": "getaway vehicle described as a red pickup truck with a cracked windshield"},
]


# ── helpers ──────────────────────────────────────────────────────────────────
def _load_seeds(seed_file: str | Path) -> dict[str, list[str]]:
    """Group seed snippet texts by ``doc_type_hint`` (with a non-empty fallback)."""
    groups: dict[str, list[str]] = {dt: [] for dt in DOC_TYPES}
    path = Path(seed_file)
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            groups.setdefault(obj.get("doc_type_hint", "report"), []).append(obj["text"])
    for dt in DOC_TYPES:
        if not groups.get(dt):
            groups[dt] = [_FALLBACK[dt]]
    return groups


def _make_case_ids(rng: Random, k: int = 10) -> list[str]:
    """A deterministic pool of fictional case ids like ``2024-7812``."""
    return [f"{rng.choice([2023, 2024])}-{rng.randint(1000, 9999)}" for _ in range(k)]


def _random_date(rng: Random) -> str:
    """Random ISO-8601 date within the configured range."""
    span = (_DATE_END - _DATE_START).days
    return (_DATE_START + datetime.timedelta(days=rng.randint(0, span))).isoformat()


def _month_window(iso_date: str) -> dict[str, str]:
    """The first/last day of the month containing ``iso_date`` (for a date-range filter)."""
    d = datetime.date.fromisoformat(iso_date)
    first = d.replace(day=1)
    nxt = datetime.date(d.year + (d.month == 12), (d.month % 12) + 1, 1)
    last = nxt - datetime.timedelta(days=1)
    return {"gte": first.isoformat(), "lte": last.isoformat()}


def _slugify(text: str, maxlen: int = 40) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s[:maxlen].strip("-") or "doc"


def _pick(rng: Random, pool: list[str], n: int) -> list[str]:
    return rng.sample(pool, min(n, len(pool))) if pool else []


# ── document body builders (one per doc_type) ────────────────────────────────
def _build_witness(rng: Random, seeds: dict[str, list[str]], needle: str | None, meta: dict) -> str:
    body = _pick(rng, seeds["witness_statement"], 2)
    if needle:
        body.insert(rng.randint(0, len(body)), needle)
    return "\n".join([
        "WITNESS STATEMENT",
        f"Case {meta['case_id']}    Date: {meta['date']}",
        "",
        " ".join(body),
        "",
        "I confirm the above account is true to the best of my recollection.",
    ])


def _build_report(rng: Random, seeds: dict[str, list[str]], needle: str | None, meta: dict) -> str:
    observations = _pick(rng, seeds["report"], 2)
    if needle:
        observations.insert(rng.randint(0, len(observations)), needle)
    numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(observations, 1))
    return "\n".join([
        "INCIDENT REPORT",
        f"Case {meta['case_id']}    Date: {meta['date']}",
        "",
        "Observations:",
        numbered,
        "",
        "Report compiled by attending officer.",
    ])


def _build_transcript(rng: Random, seeds: dict[str, list[str]], needle: str | None, meta: dict) -> str:
    turns = _pick(rng, seeds["transcript"], 1)
    tail = needle if needle else "A: No, that is everything I can recall."
    if not tail.lstrip().startswith(("A:", "Q:")):
        tail = f"A: {tail}"
    return "\n".join([
        "INTERVIEW TRANSCRIPT",
        f"Case {meta['case_id']}    Date: {meta['date']}",
        "",
        *turns,
        "Q: Is there anything else you remember?",
        tail,
    ])


_BUILDERS: dict[str, Callable[[Random, dict, str | None, dict], str]] = {
    "witness_statement": _build_witness,
    "report": _build_report,
    "transcript": _build_transcript,
}


# ── format writers (one per format) ──────────────────────────────────────────
def _write_txt(path: Path, text: str, meta: dict) -> None:
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, text: str, meta: dict) -> None:
    payload = {"content": text, **{k: meta[k] for k in ("doc_type", "case_id", "date", "title")}}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_pdf(path: Path, text: str, meta: dict) -> None:
    # Imported lazily so the package imports without reportlab present.
    from reportlab.lib.pagesizes import LETTER
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

    styles = getSampleStyleSheet()
    flow: list[Any] = []
    for line in text.split("\n"):
        flow.append(Spacer(1, 8) if not line.strip() else Paragraph(escape(line), styles["Normal"]))
    SimpleDocTemplate(str(path), pagesize=LETTER).build(flow)


def _write_eml(path: Path, text: str, meta: dict) -> None:
    msg = EmailMessage()
    msg["From"] = "records.officer@precinct.example"
    msg["To"] = "case.file@precinct.example"
    msg["Subject"] = f"{meta['title']} [{meta['case_id']}]"
    msg["Date"] = format_datetime(datetime.datetime.fromisoformat(f"{meta['date']}T12:00:00"))
    # Inline metadata in headers — the loader can parse these instead of the filename.
    msg["X-Case-ID"] = meta["case_id"]
    msg["X-Doc-Type"] = meta["doc_type"]
    msg.set_content(text)
    path.write_text(msg.as_string(), encoding="utf-8")


_WRITERS: dict[str, Callable[[Path, str, dict], None]] = {
    "txt": _write_txt,
    "pdf": _write_pdf,
    "json": _write_json,
    "eml": _write_eml,
}


# ── orchestrator ─────────────────────────────────────────────────────────────
def generate(
    n: int = 120,
    seed: int = 42,
    out_dir: str | Path = "data/generated",
    seed_file: str | Path = _DEFAULT_SEED_FILE,
) -> dict[str, Any]:
    """Generate ``n`` documents + ``ground_truth.json`` into ``out_dir``.

    Returns a small stats dict: ``{out_dir, documents, by_format, by_doc_type,
    ground_truth_pairs}``. Deterministic for a fixed ``seed``.
    """
    if n < len(_NEEDLES):
        raise ValueError(f"n must be >= {len(_NEEDLES)} so every ground-truth needle fits")

    rng = Random(seed)
    seeds = _load_seeds(seed_file)
    case_ids = _make_case_ids(rng)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Clean slate: remove artifacts from a previous run so the corpus is exactly
    # `n` docs and fully reproducible (no stale files leaking into ingestion).
    for old in out.iterdir():
        if old.is_file() and (old.suffix.lower() in {f".{f}" for f in FORMATS} or old.name == "ground_truth.json"):
            old.unlink()

    # Assign each needle to a distinct document index.
    needle_idx = dict(zip(rng.sample(range(n), len(_NEEDLES)), _NEEDLES))

    by_format: dict[str, int] = {f: 0 for f in FORMATS}
    by_doc_type: dict[str, int] = {d: 0 for d in DOC_TYPES}
    ground_truth: list[dict[str, Any]] = []

    for i in range(n):
        needle = needle_idx.get(i)
        doc_type = needle["doc_type"] if needle else rng.choice(DOC_TYPES)
        fmt = FORMATS[i % len(FORMATS)]
        case_id = rng.choice(case_ids)
        date = _random_date(rng)
        title = f"{doc_type.replace('_', ' ').title()} - Case {case_id}"
        meta = {"doc_type": doc_type, "case_id": case_id, "date": date, "title": title}

        text = _BUILDERS[doc_type](rng, seeds, needle["fact"] if needle else None, meta)
        slug = _slugify(needle["query"] if needle else f"{doc_type}-{i}")
        filename = f"{doc_type}__{case_id}__{date}__{i:03d}-{slug}.{fmt}"
        _WRITERS[fmt](out / filename, text, meta)

        by_format[fmt] += 1
        by_doc_type[doc_type] += 1

        if needle:
            filters: dict[str, Any] = {}
            if needle["filter"] == "doc_type":
                filters = {"doc_type": doc_type}
            elif needle["filter"] == "case":
                filters = {"doc_type": doc_type, "case_id": case_id}
            elif needle["filter"] == "date":
                filters = {"doc_type": doc_type, "date": _month_window(date)}
            ground_truth.append(
                {"query": needle["query"], "expected_source_file": filename, "filters": filters}
            )

    gt_path = out / "ground_truth.json"
    gt_path.write_text(json.dumps(ground_truth, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "out_dir": str(out),
        "documents": n,
        "by_format": by_format,
        "by_doc_type": by_doc_type,
        "ground_truth_pairs": len(ground_truth),
    }

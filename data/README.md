# Data

## `generated/` (git-ignored)
Produced by `rag generate` — a **scenario-driven** synthetic forensic corpus plus
`ground_truth.json`. Not committed because it is fully reproducible from a fixed RNG
seed (`dataset.seed` in `config.yaml`). Regenerate with:

```bash
rag generate              # default 120 docs / 30 cases
rag generate --n 80 --seed 7
```

### How it's built
The generator models distinct **case scenarios** rather than stitching snippets:

- `ceil(n/4)` cases, each with structured facts (crime, location, date, time, victim,
  suspect, vehicle, witnesses, officer, evidence) drawn from large entity pools.
- Each case emits **4 internally-consistent documents** — two witness statements, an
  incident report, an interview transcript — so `case_id` genuinely groups a case's
  documents (the law-enforcement scenario from the brief).
- **Formats:** `.txt`, `.pdf`, `.json`, `.eml` (the required three + email evidence).
- **Metadata** (`doc_type`, `case_id`, `date`) is controlled by the generator and
  encoded in the filename convention
  `"<doc_type>__<case_id>__<YYYY-MM-DD>__<NNN-role>.<ext>"`; `.json`/`.eml` also embed
  it inline (JSON fields / email headers) so the loader can recover it either way.

### Ground truth (`ground_truth.json`)
One `(query, expected_source_file, filters)` entry per case. Each case carries a
**unique signature** placed in exactly one document — a distinctive vehicle (witness
statement), a unique evidence item (report), or a named person (transcript) — so each
query has an unambiguous answer. Filters rotate across **semantic / doc_type / case_id
/ date-range** to exercise the metadata-filtering paths in evaluation.

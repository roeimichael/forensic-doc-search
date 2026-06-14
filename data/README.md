# Data

## `seeds/seed_snippets.jsonl`
Small snippets of realistic prose used to *seed* the synthetic generator so the
generated documents read naturally. Each line: `{"id", "doc_type_hint", "text",
"source", "license"}`. Keep snippets short and from clearly permissive/public-domain
sources; record provenance + license per line in the `source`/`license` fields.

> The placeholder shipped here is illustrative. Replace/expand with public-domain or
> permissively licensed text (e.g. public-domain legal/police report samples,
> Wikipedia paragraphs under CC BY-SA with attribution).

## `generated/` (git-ignored)
Produced by `rag generate` — the document corpus (`.txt`/`.pdf`/`.json`) plus
`ground_truth.json`. Not committed because it is fully reproducible from the seed
file + a fixed RNG seed (`dataset.seed` in `config.yaml`). Regenerate with:

```bash
rag generate            # or: python scripts/generate_dataset.py
```

Metadata is **controlled by the generator** (doc_type, case_id, date), encoded in
the filename convention `"<doc_type>__<case_id>__<YYYY-MM-DD>__<slug>.<ext>"` so the
loader recovers it deterministically.

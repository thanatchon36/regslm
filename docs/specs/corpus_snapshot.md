# Spec: corpus pipeline CLI + snapshot (paper §3.1)

## CLI (src/regslm/cli.py; logic in src/regslm/corpus/snapshot.py)
- `regslm fetch [SOURCES...]` — for each adapter in `corpus.registry.ADAPTERS` (default: all
  tier-P adapters), call `fetch()`; print counts. Cached, so re-runs are offline.
- `regslm parse [SOURCES...]` — parse cached raw docs → `data/local/units/<source>.jsonl` and
  `data/local/documents/<source>.jsonl`. Adapters exposing `parse_all(raw, selection)` (eCFR)
  are handled; otherwise `parse(raw)`.
- `regslm snapshot` — build, from all parsed sources:
  - `data/corpus/manifest.parquet` — one row per raw response: source, url, sha256, fetched_at,
    content_type, licence tier (P/L/X from the adapter's `Licence`: P = redistributable and
    derivative_ok; L = otherwise; X = adapters listed in config `excluded_sources`), doc_ids.
    Always committed (URLs + hashes only).
  - `data/corpus/documents.parquet` — Document rows + tier.
  - `data/corpus/units.parquet` — Unit rows + tier + `n_chars` + `text_sha256`. **Only tier-P
    rows go to `data/corpus/`**; L rows go to `data/local/corpus/units_local.parquet` (git-ignored).
  - `data/corpus/VERSION` — `YYYY.MM.DD-<8-char hash of sorted unit text_sha256s>`.
  - `data/corpus/STATS.md` — per source/doc/language: #docs, #units, chars median/p95.
  Deterministic: sorted by unit_id; parquet written with polars, zstd.
- Each adapter class must expose `licence: Licence`; tier computed, not hard-coded.
- Do not rename or modify adapters' parsing logic; only call them.

## Tests (`tests/test_snapshot.py`)
Fake adapter with 2 docs, one P and one L: tier routing, determinism (two runs → identical
VERSION and bytes), manifest has no text column, L text never written under data/corpus/
(use tmp_path roots — make all paths configurable).

Ruff clean, mypy --strict on snapshot.py.

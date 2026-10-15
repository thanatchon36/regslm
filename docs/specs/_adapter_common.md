# Spec: common rules for every source adapter

Contract: `src/regslm/corpus/base.py` (`SourceAdapter`, `PoliteClient`, `Document`, `Unit`,
`Licence`, `make_unit_id`, `write_jsonl`). Do **not** change base.py signatures; if you need
a change, write it under "Proposed base changes" in your final report.

## Deliverables per source `<name>`
1. `src/regslm/corpus/<name>.py` — class `<Name>Adapter(SourceAdapter)` with module docstring
   covering: document types, URL patterns, structure, language(s), licence terms (with URL),
   and the default selection.
2. Fixtures: `tests/fixtures/<name>/` — **small** real raw responses (≤ 200 KB total; trim
   if needed, keeping the original structure) captured with the adapter's own client.
3. `tests/test_corpus_<name>.py` — offline tests on fixtures (no network) plus one
   `@pytest.mark.network` smoke test that fetches one small document live.
4. Register the adapter in `src/regslm/corpus/registry.py` (`ADAPTERS: dict[str, type[SourceAdapter]]`;
   create the file if absent — it may already exist from a parallel agent; merge, don't overwrite).

## Unit rules
- One `Unit` = one paragraph-level provision (the smallest numbered/lettered item that
  carries a full sentence). Keep the lead-in ("A bank must:") attached to each child by
  storing it in `heading` (closest heading + lead-in), not by duplicating text into `text`.
- `text`: normalised whitespace, no markup, footnote markers removed, cross-references kept
  verbatim. Skip empty / "[Reserved]" / "Deleted" units.
- `unit_id` via `make_unit_id(source, doc_id, *levels)`; must be unique within the source
  and stable for a given version. Test uniqueness.
- `version`: the point-in-time/consolidation date the text reflects (ISO date).
- `last_amended`: ISO date of the latest amendment affecting the unit if the source exposes
  it at unit or section level; else the document-level date; else None.

## Politeness
Only through `self.client` (PoliteClient: robots.txt, ≥2 s/host, UA, disk cache). Never bypass
the cache in tests. If a site returns 403/429 or a JS/WAF challenge persistently, stop,
append the exact URLs and the symptom to `docs/sources/NEEDED.md`, and make `fetch` raise a
clear `RuntimeError` pointing there. Do not try to evade blocking (no UA spoofing, no headless
browser).

## Quality bar
- `uv run ruff check src tests` and `uv run mypy src/regslm/corpus/<name>.py --strict` clean.
- `uv run pytest tests/test_corpus_<name>.py` green.
- Run the full default selection once live and report: #documents, #units, median/95th
  pct unit length in characters, 5 random unit_ids with first 120 chars of text. Do **not**
  paste whole documents anywhere. Write parsed output with
  `write_jsonl(Path("data/local/units/<name>.jsonl"), units)`.
- Python 3.11+, pydantic v2, lxml for XML/HTML. No new dependencies without listing them in
  the final report.

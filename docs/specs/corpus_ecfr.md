# Spec: eCFR adapter (`ecfr`)

Read `docs/specs/_adapter_common.md` first.

- API: `https://www.ecfr.gov/api/versioner/v1/` (documented at https://www.ecfr.gov/developers/documentation/api/v1).
  - Titles/latest issue date: `titles.json`.
  - Full XML for a part at a date: `full/{date}/title-{t}.xml?part={p}`.
  - Amendment history: `versions/title-{t}.json?part={p}` → per-section `amendment_date`;
    use it for `Unit.last_amended` (section level).
- Licence: US federal government work, public domain (17 U.S.C. §105). redistributable=True,
  derivative_ok=True.
- doc_id: `{t}CFR{p}` (e.g. `12CFR217`). Hierarchy: part → subpart → section (`SECTION` /
  `DIV8`) → paragraphs `(a)`, `(1)`, `(i)`, `(A)`. eCFR XML flattens paragraphs into `<P>`
  elements whose leading `(x)` marker gives the level; reconstruct nesting from marker type
  sequence (a→1→i→A→ *1* → *i*), handling the ambiguous `(i)` (letter i vs roman i) by
  context (previous sibling marker).
- Default selection (config, overridable via `selection`):
  Title 12: parts 3, 217, 249, 252, 324, 1005, 1026; Title 17: parts 240 (only §§ 240.15c3-1,
  240.15c3-3, 240.17a-3, 240.17a-4), 275.
  Use the latest issue date from `titles.json` and record it as `version`.
- Fixture: a small part, e.g. 12 CFR part 3 subpart A, trimmed.

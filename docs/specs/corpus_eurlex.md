# Spec: EUR-Lex adapter (`eurlex`)

Read `docs/specs/_adapter_common.md` first.

- Instruments (base CELEX): CRR `32013R0575`, CRD `32013L0036`, MiFID II `32014L0065`,
  AMLD4 `32015L0849`. Use the **latest consolidated version** (CELEX `0YYYY…-YYYYMMDD`).
  Discover it without scraping search pages if possible: preferred route is the
  Publications Office Cellar SPARQL endpoint (`https://publications.europa.eu/webapi/rdf/sparql`)
  to list consolidated versions of a base act; fall back to the EUR-Lex HTML page
  `https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:{celex}`.
  Check `robots.txt` for both hosts. EUR-Lex sometimes serves an AWS WAF / JS challenge to
  automated clients — if so, use Cellar content negotiation (`Accept: application/xhtml+xml`
  on the cellar URI) instead; if both are blocked follow the NEEDED.md rule.
- Language: `en` by default; adapter takes `languages: list[str]` (EU language codes) so other
  official languages can be fetched later; doc_id includes the language only when ≠ en.
- Licence: EUR-Lex content reuse is authorised (Commission Decision 2011/833/EU; legal notice
  https://eur-lex.europa.eu/content/legal-notice/legal-notice.html), attribution required:
  "© European Union, https://eur-lex.europa.eu/". Only the authentic OJ text is legally
  binding; consolidated versions are documentation. redistributable=True, derivative_ok=True.
- doc_id: e.g. `CRR`, `CRD`, `MiFID2`, `AMLD4`; version = consolidation date.
- Hierarchy: Part → Title → Chapter → Section → Article → paragraph `1.` → point `(a)` →
  sub-point `(i)`. Unit = paragraph, or point when a paragraph has points (lead-in to
  `heading`). Skip recitals (record them? no — recitals are not obligations; skip, but count
  them in the report). Annexes: include, units = numbered items. Definitions article
  (e.g. CRR Art. 4) is included (needed as negative examples).
- Consolidated HTML marks amendments with `►M12` style markers — strip them from `text`
  but parse the latest amendment marker per article into `last_amended` by mapping `Mxx` to
  the amending act date from the version's header table if feasible; else document-level date.
- Fixture: one or two articles of CRR (e.g. Art. 92 "Own funds requirements" and Art. 4(1)
  first few definitions), trimmed.

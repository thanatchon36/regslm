
## FCA Handbook (owner action)
Automated retrieval is barred without consent (FCA website terms cl. 35.3). Either obtain consent/API access, or manually save the PDFs of these sourcebooks (current versions) from https://www.handbook.fca.org.uk/handbook into `data/local/manual/fca/`:
- SYSC — https://www.handbook.fca.org.uk/handbook/SYSC/
- COBS — https://www.handbook.fca.org.uk/handbook/COBS/
- SUP — https://www.handbook.fca.org.uk/handbook/SUP/

## eurlex: eur-lex.europa.eu blocked by AWS WAF challenge (route changed to Cellar)

- Symptom (2026-10-15): every automated GET to `https://eur-lex.europa.eu/robots.txt` and
  `https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32013R0575` returns HTTP 202,
  empty body, header `x-amzn-waf-action: challenge` (CloudFront + AWS WAF JS challenge).
- Not evaded. The `eurlex` adapter uses the Publications Office Cellar instead (not challenged):
  SPARQL `https://publications.europa.eu/webapi/rdf/sparql` for discovery and
  `http://publications.europa.eu/resource/cellar/<uuid>` with `Accept: application/xhtml+xml`,
  `Accept-Language: eng` for the consolidated XHTML (robots.txt of that host -> op.europa.eu allows it).
- Needed from a human only if Cellar is ever blocked too: a manual download of the four
  consolidated texts (CELEX 02013R0575, 02013L0036, 02014L0065, 02015L0849, latest `-YYYYMMDD`)
  in EN as XHTML into `data/raw/eurlex/`.
- Note: the newest CRD consolidation listed by Cellar (dated 2026-xx) had no English manifestation
  (HTTP 404); the adapter falls back to the next-newest consolidation that has one.

## ecfr (robots.txt, not a block)
- `https://www.ecfr.gov/api/versioner/v1/full/{date}/title-{t}.xml?part={p}` (the endpoint named in
  docs/specs/corpus_ecfr.md) is `Disallow`ed by https://www.ecfr.gov/robots.txt. Not used. The adapter
  uses GovInfo bulk XML (`https://www.govinfo.gov/bulkdata/ECFR/title-{12,17}/ECFR-title{t}.xml`) instead.
  Bulk snapshots lag ecfr.gov (title 17 AMDDATE 2026-08-17, title 12 2026-10-01 at fetch time).
- Note for PoliteClient: ecfr.gov/robots.txt has indented `Disallow:` lines after a blank line, which
  `urllib.robotparser` ignores, so `PoliteClient._allowed` wrongly returned True for the disallowed path.

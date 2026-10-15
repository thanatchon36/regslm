
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

## thai: Anti-Money Laundering Act B.E. 2542 (AMLA2542) not obtained (owner action)
- Not in any `pythainlp/thailaw`, `pythainlp/thailaw-v1.0` or `wannaphong/thailaw-v1-more` row (checked by text head; v1.0 also lacks the
  FIBA B.E. 2551 and the BOT Act B.E. 2485, which the adapter takes from the Council of State PDFs hosted by bot.or.th instead).
- AMLO (`https://www.amlo.go.th/`) serves its law attachments from `/amlo-intranet/`, which its robots.txt disallows; the home page is a
  JavaScript shell. krisdika.go.th did not answer (HTTP 000 / robots.txt unreachable). Royal Gazette (ratchakitcha.soc.go.th) and
  sec.or.th are WAF-blocked (see spec) and were not tried.
- Needed from a human: a copy of the Thai text of the consolidated Anti-Money Laundering Act B.E. 2542 (Council of State / Royal Gazette
  version, e.g. https://www.krisdika.go.th/ law database or https://ratchakitcha.soc.go.th/) saved to `data/local/manual/thai/AMLA2542.txt`
  (one paragraph per line); a `AMLA2542` entry can then be routed through `blocks_from_text`.

## thai: BoT notification PDFs dropped (no usable text layer)
- `https://www.bot.or.th/content/dam/bot/fipcs/documents/FPG/2551/ThaiPDF/25510555.pdf` (minimum capital for operational risk): 21 text
  pages are glyph-name junk ("/afiiotraOf/fpSbf...") and the rest are scans. Needs OCR or a re-published PDF.
- `https://www.bot.or.th/content/dam/bot/fipcs/documents/FPG/2553/ThaiPDF/25530017.pdf` (liquidity risk management policy): scanned images,
  one text page of 34.

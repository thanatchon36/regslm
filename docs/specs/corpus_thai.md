# Spec: Thai adapter (`thai`)

Read `docs/specs/_adapter_common.md` first. Thailand is one language among several; the
module docstring describes sources and structure only — no commentary on any institution.

## Licence
Thai Copyright Act B.E. 2537 s. 7(2)–(3): statutes, regulations, notifications and orders of
government agencies are not copyright works; s. 7(5): translations/compilations of these made
by government agencies likewise. Tier P (docs/sources/LICENSES.md). Third-party (law-firm)
translations are tier L: never write them under a committed path.

## Access routes (in order; robots.txt binding; no evasion)
The Royal Gazette (ratchakitcha.soc.go.th) and SEC Thailand (sec.or.th) block automated
clients (Cloudflare/WAF 403) — do not use them; list exact URLs in NEEDED.md if needed.
1. **Statutes, Thai text:** Hugging Face dataset `pythainlp/thailaw` (or the latest
   `pythainlp/thailaw-v*`; check the dataset card's licence and source — expected: Office of the
   Council of State texts, public domain). Download via `https://huggingface.co/datasets/.../resolve/main/...`
   parquet files through PoliteClient (huggingface.co robots allows dataset files), or with
   `huggingface_hub` if it is already installed in .venv (`uv run --no-sync python -c "import huggingface_hub"`).
   Select by title:
   - พระราชบัญญัติธุรกิจสถาบันการเงิน พ.ศ. ๒๕๕๑ (Financial Institutions Business Act B.E. 2551) → doc_id `FIBA2551`
   - พระราชบัญญัติหลักทรัพย์และตลาดหลักทรัพย์ พ.ศ. ๒๕๓๕ (Securities and Exchange Act B.E. 2535) → `SEA2535`
   - พระราชบัญญัติธนาคารแห่งประเทศไทย พ.ศ. ๒๔๘๕ → `BOTA2485`
   - พระราชบัญญัติระบบการชำระเงิน พ.ศ. ๒๕๖๐ → `PSA2560`
   - พระราชบัญญัติป้องกันและปราบปรามการฟอกเงิน พ.ศ. ๒๕๔๒ → `AMLA2542`
   Prefer the consolidated/latest text if multiple rows exist; record which row (dataset revision
   sha + row id) in Document.url/version.
2. **BoT notifications (ประกาศธนาคารแห่งประเทศไทย), Thai:** from www.bot.or.th (robots.txt allows
   per licence research — re-check). Select ~20 current notifications on capital, liquidity,
   large exposure, market conduct (consumer protection), IT risk. If the listing pages are JS
   apps, use the PDF links; parse PDF with `pypdf` (installed). Thai PDF text extraction can
   produce broken vowel/tone ordering — detect with a sanity check (ratio of valid Thai syllables
   via `pythainlp.util.isthai` / normalise with `pythainlp.util.normalize`) and drop a document
   whose extraction is garbled (report it).
3. **Official English translations (parallel set):** BoT publishes English translations of FIBA
   and BoT notifications. Fetch the FIBA English translation if available from bot.or.th; mark
   language "en", doc_id `FIBA2551-en`, and set `heading` with the same section numbers so units
   align by `path`. Write an alignment file `data/local/units/thai_parallel.jsonl`
   (th_unit_id, en_unit_id) for units whose paths match exactly.

## Structure
Act: หมวด (chapter) → ส่วน (part) → มาตรา N (section; Thai digits — keep both Thai-digit text and
an ASCII section number in `path`) → วรรค (paragraph: each new paragraph within a section) →
(๑)(๒) items. Unit = วรรค, or item when a วรรค has items (lead-in to heading).
unit_id like `thai/FIBA2551/s29/p2/i3` (ASCII digits in ids). Language "th".
Skip: preamble (พระบรมราชโองการ…), หมายเหตุ (reasons for enactment), transitory provisions
are kept. Notifications: ข้อ N → items.
Thai text normalisation: `pythainlp.util.normalize`, collapse whitespace, keep Thai digits as is
in `text` (perturbation handles them).
`last_amended`: from amendment notes in the text if present (e.g. "แก้ไขเพิ่มเติมโดย…พ.ศ. …"
→ convert B.E. year − 543 to an ISO date at year granularity `YYYY-01-01`), else None.

## Report
Per document: #units, length stats in characters and in `pythainlp.tokenize.word_tokenize`
words; 5 sample unit_ids with 120-char previews; parallel-aligned unit count.

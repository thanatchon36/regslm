# Annotation guide — obligations

Version 2.0 (2026-10-15). v1.1: citations by quote (D-020). v2.0: expert review (D-036) —
polarity/comparator rule, no invented subjects, modal-in-noun-phrase, deeming/scope/enabling
clauses, commentary, Thai "ให้"/"อาจ"/"สามารถ", numbers written in Thai, field consistency.
Applies to gold adjudication, teacher prompts (`src/regslm/teacher/prompts.py`, kept in sync)
and evaluation. Ambiguities found during annotation go to `docs/INTERPRETATIONS.md` and, once
resolved, into this guide with a version bump.

Examples are **illustrative**: written in the drafting style of the corpus to show a decision
rule, not quoted from any instrument (quoted gold examples live in `data/gold/`).

## 1. What counts as an obligation

An obligation is a **main clause** that directs a **named** party to do, not do, or be permitted
or expected to do something, with deontic force expressed by the text.

| Trigger (en) | Trigger (th) | modality |
|---|---|---|
| shall, must, is required to, has to, is obliged to | ต้อง, จะต้อง, มีหน้าที่, directive ให้ (rule T-ให้) | `must` |
| shall not, must not, may not, is prohibited from, no X shall | ห้าม, ต้องไม่, มิให้, ห้ามมิให้ | `must_not` |
| may, is permitted to, is entitled to | อาจ + party actor (rule T-อาจ), มีสิทธิ, สามารถ…ได้ granting a right | `may` |
| should, is expected to, ought to, "the authority expects X to" | ควร | `should` |

### Rules
- **R1 — Named addressee.** The subject must be a party named in the unit text or in its lead-in
  (`heading` breadcrumb containing a lead-in sentence such as "A bank must:"). Duties and operative
  powers of an authority ARE annotated ("The competent authority shall publish…", "The Commission
  may exempt any person…"). **Never supply a subject from outside the unit.** A `shall`
  sentence with no addressee ("the risk weight shall be 100 %", "the value … shall be equal to the
  sum of…") is a calculation/deeming rule → `[]` (class `no-addressee`), unless the lead-in names
  the party (ex. 27).
- **R2 — Definitions and descriptions are not obligations.** "'X' means…", "X includes…", "For the
  purposes of…", variable definitions ("LGD = the loss given default of…"), and modals inside a
  noun phrase or relative clause ("the amount that the bank **must** return under…") → `[]`. Test:
  does the sentence's main clause direct a party?
- **R3 — Scope, recital, commencement, amendment, deeming and enabling text → `[]`.** "This
  Regulation applies to…", "Article X shall apply to Y", "Paragraph 1 shall not apply to…",
  "shall be treated as / is deemed / ให้ถือว่า", "This Act shall come into force…", "Article 5 is
  replaced by…", repeal ("ให้ยกเลิก"), and legislative delegation ("Power is delegated to the
  Commission to adopt regulatory technical standards…", "EBA shall develop draft RTS…" — class
  `enabling`). Exception: when a party is told to apply a treatment ("An institution shall treat X
  as Y", "An institution shall apply Article X") → `must | treat/apply`.
- **R4 — Possibility and capability are not permissions.** "Losses may arise…", "อาจเกิด
  ความเสียหาย" → `[]`. "An institution may apply…" → `may`.
- **R5 — Commentary and non-operative content → `[]`.** Official interpretations / commentary
  explaining a rule (e.g. 12 CFR 1026 Supplement I), guidance restating a rule found elsewhere
  (class `restated-rule`), table cells, form codes, fragments with no main clause, alternative
  list items ("…; or") with no modal in TEXT or lead-in, and garbled text.
- **R6 — One obligation per (subject, modality, action).** Coordinated verbs with one subject and
  modality ("shall establish and maintain") are one obligation; different objects, thresholds or
  deadlines are split. A duty to ensure that others act ("must ensure that its employees do not…")
  is one obligation on the party that must ensure.
- **R7 — Lead-ins.** A list point inherits subject and modality only from a lead-in **sentence**
  in TEXT or HEADING ("A bank must:", "Institutions shall:"). Section titles ("§ 3.34 Derivative
  contracts") never supply modality. The quote is taken from the point's own text.
- **T-ให้ (Thai).** ให้ is a `must` trigger only as a clause-initial directive "ให้ + actor +
  action" (ให้ธนาคารพาณิชย์จัดส่ง…, ให้สำนักงานเก็บรักษา…) or "ให้ถือปฏิบัติ / ปฏิบัติ…". It is NOT a
  trigger when it is the main verb "give/grant" (ให้สินเชื่อ, ให้ความเห็นชอบ), in purpose clauses
  (เพื่อให้…), or in ให้ใช้ / ให้ยกเลิก / ให้ถือว่า / ให้ประกอบด้วย / ให้…อยู่ในตำแหน่ง (commencement,
  repeal, deeming, composition, term of office → `[]`, R3).
- **T-อาจ (Thai).** อาจ + party/authority actor + action → `may`; อาจ with an event or state as
  subject → `[]`. สามารถ…ได้ is `may` only when it grants a right, not when it states capability.
  ควร → `should`.
- **T-subject (Thai).** Thai often omits the subject in the main clause and places it in a
  preceding "ในกรณีที่ / กรณีที่ …" clause of the same sentence. Copy the subject from there (it may
  also appear inside `condition`).
- **Delegations.** "ต้อง/ให้ … ตามที่ ธปท. ประกาศกำหนด", "in accordance with the requirements
  specified by the competent authority": a full obligation; keep the delegation phrase verbatim in
  `object` or `condition`; do not invent a threshold or deadline.

## 2. Fields

| Field | Rule |
|---|---|
| `subject` | Shortest noun phrase naming the duty-bearer, copied from the text with its article as written ("a covered company", "institutions"). Pronouns ("it", "they", "its") are replaced by the noun phrase they refer to, copied from the same unit. "No person shall…" → subject "person". |
| `modality` | Grammar of the main clause per the trigger table; negated permission ("may not") = `must_not`. |
| `action` | Verb phrase without modal and object ("file", "establish and maintain"); copula "be" for limits on a quantity (§3.4). Never invent a verb. |
| `object` | What the action applies to, including the base of a ratio/limit ("25 % of its Tier 1 capital" → object keeps "of its Tier 1 capital" context). Empty if intransitive. |
| `condition` | when/if/where/provided-that clause that switches the duty on, or that must hold for a permission ("may … provided that the exposure is fully collateralised"). Nested conditions joined with " AND " in source order. |
| `threshold` | The **binding** number — the one that defines the compliant state. `value` = the number as written, converted to Arabic digits (4.5 for "4.5 %" or "4,5 %"; 25 for "๒๕" or "ยี่สิบห้า"; 250 for "$250 billion"); `unit` = unit plus any scale word ("%", "USD billion", "EUR million", "days", "times", "basis points", "THB"); `comparator` per §3.4. A number whose crossing only switches the duty on is a trigger → it stays in `condition`, not `threshold`. One threshold per obligation; several binding numbers → several obligations. |
| `deadline` | Time limit, frequency, or duration of a continuing duty, copied verbatim ("within 30 days after the end of each quarter", "annually", "at all times", "เป็นระยะเวลาหนึ่งปีนับจากวันที่…"). Time limits are never thresholds. |
| `exceptions` | Each carve-out that removes the duty for some cases ("unless…", "except where…", "other than…", "provided that this paragraph does not apply to…"), as its own string. "Notwithstanding X" is NOT an exception (it overrides another provision). |
| `citation` | `source`, `unit_id`, and a verbatim `quote` of the sentence/clause of TEXT containing the trigger (for a list point: the point itself); `span` is computed from the quote (D-020). |
| `confidence` | Model self-estimate; gold uses 1.0. Calibration plots only. |

Cross-references ("in accordance with Article 26", "as defined in § 217.2") are kept verbatim in
the field where they occur; they are not resolved.

## 3. Worked examples

Format: text → `subject | modality | action | object | condition | threshold | deadline |
exceptions`; `—` = empty; `[]` = no obligation.

### 3.1 Plain duties
1. "Institutions shall at all times satisfy a Common Equity Tier 1 capital ratio of 4.5 %." →
   Institutions | must | satisfy | a Common Equity Tier 1 capital ratio | — | 4.5 % ≥ | at all
   times | —. ("satisfy a ratio of X" = the ratio must be at least X.)
2. "A covered company must file the report with the Board within 30 days after the end of each
   calendar quarter." → A covered company | must | file | the report with the Board | — | — |
   within 30 days after the end of each calendar quarter | —.
3. "The firm must maintain adequate records of all transactions." → The firm | must | maintain |
   adequate records of all transactions | — | — | — | —.
4. "Each bank shall establish and maintain a written liquidity risk management policy." → one
   obligation, action "establish and maintain" (R6).
5. "A licensee shall notify the Authority of any change in its controllers and shall obtain
   approval before appointing a director." → two obligations (different actions/objects).

### 3.2 Prohibitions
6. "A firm must not pay remuneration that is linked to sales targets." → A firm | must_not | pay |
   remuneration that is linked to sales targets | — | — | — | —.
7. "No person shall act as a broker unless registered with the Commission." → person | must_not |
   act | as a broker | — | — | — | [unless registered with the Commission].
8. "An institution may not include in Tier 2 capital any instrument that…" → must_not (negated
   permission).
9. "ห้ามมิให้สถาบันการเงินให้สินเชื่อแก่ผู้ถือหุ้นรายใหญ่เกินกว่าร้อยละ ๒๕ ของเงินกองทุน" → สถาบันการเงิน |
   must_not | ให้สินเชื่อ | แก่ผู้ถือหุ้นรายใหญ่เกินกว่าร้อยละ ๒๕ ของเงินกองทุน | — | 25 % > | — | —. A
   prohibited **act** above a limit: must_not, comparator ">" marks the prohibited region (§3.4).
   Here ให้ is the verb "grant", the trigger is ห้ามมิให้ (T-ให้).

### 3.3 Permissions, expectations, possibility
10. "An institution may, with the prior permission of the competent authority, use internal models
    to calculate own funds requirements." → An institution | may | use | internal models to
    calculate own funds requirements | with the prior permission of the competent authority | — |
    — | —.
11. "Market movements may cause the value of collateral to decline." → `[]` (R4).
12. "The Commission may, by order, exempt any person from this section." → The Commission | may |
    exempt | any person from this section (operative power, R1).
13. "Firms should consider the guidance in Annex 2." → Firms | should | consider | the guidance in
    Annex 2.
14. "The PRA expects firms to review their recovery plans annually." → firms | should | review |
    their recovery plans | — | — | annually | —.
15. "ธนาคารพาณิชย์อาจยื่นคำขอผ่อนผันต่อธนาคารแห่งประเทศไทยได้" → ธนาคารพาณิชย์ | may | ยื่น | คำขอผ่อนผันต่อ
    ธนาคารแห่งประเทศไทย. But "การดำเนินการดังกล่าวอาจก่อให้เกิดความเสียหาย" → `[]` (T-อาจ).

### 3.4 Thresholds, comparators and conditions
**Comparator rule.** Modality follows the main clause. For a **limit on a quantity** ("X shall
not exceed Y", "X shall be no more than Y", "X shall be at least Y", "shall maintain X of at least
Y") encode `must`, comparator = the region the compliant value must be in (≤ for not exceed / no
more than / not greater than; ≥ for at least / not less than / minimum; == for equal to). For a
**prohibited act** defined by a limit ("shall not grant credit exceeding Y", "shall not incur an
exposure … which exceeds Y") encode `must_not`, comparator = the prohibited region (> for
exceeding / more than; < for below / less than).
16. "Large exposures shall not exceed 25 % of Tier 1 capital." → Large exposures | must | be |
    — | — | 25 % ≤ | — | — (limit on a quantity).
17. "An institution shall not incur an exposure to a client the value of which exceeds 25 % of its
    Tier 1 capital." → An institution | must_not | incur | an exposure to a client the value of
    which exceeds 25 % of its Tier 1 capital | — | 25 % > | — | — (prohibited act).
18. "Institutions shall maintain a liquidity coverage ratio of at least 100 %." → Institutions |
    must | maintain | a liquidity coverage ratio | — | 100 % ≥ | — | —.
19. "Where a covered company's total consolidated assets exceed $250 billion, it must conduct a
    stress test semi-annually." → a covered company | must | conduct | a stress test | Where a
    covered company's total consolidated assets exceed $250 billion | — (trigger, not binding) |
    semi-annually | —. ("it" resolved to the noun phrase.)
20. "If the leverage ratio falls below 3 % and the institution does not restore it within three
    months, the institution shall submit a capital plan." → the institution | must | submit | a
    capital plan | If the leverage ratio falls below 3 % AND the institution does not restore it
    within three months | — | — | —.
21. "…no more than 10 business days after the request…" as a time limit for acting → deadline.

### 3.5 Exceptions and conditions
22. "A firm must publish its annual report, except where it is a small firm as defined in GEN
    1.2." → exceptions ["except where it is a small firm as defined in GEN 1.2"].
23. "Notwithstanding paragraph (a), a bank may defer the disclosure for up to 12 months." → a bank
    | may | defer | the disclosure | — | — | for up to 12 months | — (no exception).
24. "An institution may recognise the guarantee, provided that the exposure is fully
    collateralised." → condition "provided that the exposure is fully collateralised".

### 3.6 Cross-references, lead-ins, delegation
25. "An institution shall report the information referred to in Article 430(1) to the competent
    authority." → object "the information referred to in Article 430(1) to the competent
    authority" (reference kept verbatim).
26. HEADING "… | A bank must:" with TEXT "(b) appoint a chief risk officer;" → A bank | must |
    appoint | a chief risk officer; quote "(b) appoint a chief risk officer;".
27. HEADING "… | Institutions shall deduct the following from Common Equity Tier 1 items:" with
    TEXT "(a) losses for the current financial year;" → Institutions | must | deduct | losses for
    the current financial year.
28. "ให้ธนาคารพาณิชย์ดำรงเงินกองทุนตามหลักเกณฑ์ที่ธนาคารแห่งประเทศไทยประกาศกำหนด" → ธนาคารพาณิชย์ | must |
    ดำรง | เงินกองทุนตามหลักเกณฑ์ที่ธนาคารแห่งประเทศไทยประกาศกำหนด | — | — | — | — (delegation kept
    verbatim; no threshold invented).
29. "ให้ธนาคารพาณิชย์จัดส่งรายงานต่อธนาคารแห่งประเทศไทยภายในสามสิบวัน" → ธนาคารพาณิชย์ | must | จัดส่ง |
    รายงานต่อธนาคารแห่งประเทศไทย | — | — | ภายในสามสิบวัน | — (deadline copied in words).
30. "ในกรณีที่ผู้ให้บริการทางการเงินใช้บริการจากภายนอก ให้ถือปฏิบัติตามข้อ ๖.๒" → ผู้ให้บริการทางการเงิน |
    must | ถือปฏิบัติ | ตามข้อ ๖.๒ | ในกรณีที่ผู้ให้บริการทางการเงินใช้บริการจากภายนอก | — | — | —
    (T-subject).

### 3.7 Non-obligations
31. "'Credit institution' means an undertaking the business of which is to take deposits…" →
    `[]` (R2).
32. "The fair value of an asset that the bank holding company must return under a short position"
    → `[]` (modal inside a noun phrase, R2).
33. "The value of the aggregated long position shall be equal to the sum of the positions in…" →
    `[]` (no addressee, R1).
34. "This Regulation shall enter into force on the twentieth day following its publication." /
    "ให้ยกเลิกประกาศธนาคารแห่งประเทศไทยที่ …" / "Article 5 shall apply to investment firms." → `[]` (R3).
35. "Power is delegated to the Commission to supplement this Regulation by adopting regulatory
    technical standards." → `[]` (enabling, R3).
36. "ให้คณะกรรมการกองทุนประกอบด้วยผู้แทนจาก…" → `[]` (composition, T-ให้).
37. "The tabular format is required for the disclosures described in comment 5(a)-2." in an
    official interpretation → `[]` (commentary, R5).
38. "SYSC 4.1.1R requires a firm to have robust governance arrangements." in guidance → `[]`
    (restated-rule, R5).

### 3.8 Hard cases
39. **Double modality.** "A firm must ensure that its employees do not disclose…" → one
    obligation: A firm | must | ensure | that its employees do not disclose ….
40. **Authority duty + derivative firm duty.** "The competent authority shall require institutions
    to hold additional own funds where…" → one obligation on the competent authority.

## 4. Normalisation for scoring

- Strings: Unicode NFKC, lowercase (en), whitespace collapse (Thai PDF text has stray spaces),
  strip trailing punctuation, strip leading articles (a/an/the) for `subject` and `object`.
- Numbers: Thai digits → Arabic; Thai number words (ยี่สิบห้า, สามสิบ) → value; EU decimal comma and
  space-grouped thousands handled (D-019).
- `threshold` exact match: `value` equal (1e-9 relative), `unit` after an alias map (% = per cent
  = percent = ร้อยละ; days = วัน), comparator equal.
- `deadline`: exact on the normalised string; relaxed metric compares (number, time-unit) pairs,
  parsing Thai digits and number words.
- Matching: Hungarian assignment on span IoU ≥ 0.5 with equal modality; unmatched predictions are
  false positives.

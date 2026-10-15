# Annotation guide — obligations

Version 1.1 (2026-10-15; 1.1: citations by quote, D-020). Applies to gold adjudication, teacher prompts and evaluation.
Schema: `src/regslm/schema/obligation.py` (`Extraction` → `Obligation[]`). Ambiguities found
during annotation go to `docs/INTERPRETATIONS.md` and, once resolved, into this guide with a
version bump.

Examples below are **illustrative**: written in the drafting style of the corpus sources to
show a decision rule, not quoted from any instrument (quoted gold examples live in
`data/gold/`).

## 1. What counts as an obligation

An obligation is a statement in the unit that **directs a regulated or identified party to do,
refrain from, or be permitted to do something**, with deontic force expressed by the text.

| Trigger (en) | Trigger (th) | modality |
|---|---|---|
| shall, must, is required to, has to, is obliged to, shall ensure | ต้อง, ให้…(ปฏิบัติ), จะต้อง, มีหน้าที่ | `must` |
| shall not, must not, may not, is prohibited from, no X shall | ห้าม, ต้องไม่, มิให้ | `must_not` |
| may, is permitted to, is entitled to, may apply for | อาจ, มีสิทธิ, สามารถ…ได้ | `may` |
| should, is expected to, ought to, the authority expects | ควร | `should` |

Rules:
- **R1 — Directed at a party.** There must be a subject that bears the duty or permission
  (institution, firm, covered company, bank, the board, a person). If the duty is on the
  regulator itself ("The competent authority shall publish…"), it **is** annotated — the
  subject is the authority. Downstream users can filter by subject.
- **R2 — Definitions are not obligations.** "'X' means…", "For the purposes of this Part…",
  "X includes…" → no obligation, even if the definition contains "shall" ("shall mean").
- **R3 — Purely descriptive, scope, recital, commencement and amendment text is not an
  obligation.** "This Regulation applies to…", "This Part sets out…", "Article 5 is
  replaced by…", "This Act shall come into force on…" → `[]`.
- **R4 — "may" in the sense of possibility is not a permission.** "Losses may arise from…"
  → `[]`. "An institution may apply…" → `may`.
- **R5 — "shall" in a conditional consequence that binds nobody** ("the exposure shall be
  treated as…", "the risk weight shall be 100 %") **is** an obligation on the implicit
  subject who applies the rule (the institution calculating the requirement). Subject =
  the party named in the nearest heading/lead-in; if none, `"institution"` style term used by
  that instrument. Record the ambiguity class `implicit-subject` in INTERPRETATIONS.md when
  in doubt.
- **R6 — One obligation per (subject, modality, action) triple.** Coordinated actions with one
  subject and modality ("shall establish and maintain") are one obligation; actions with
  different objects, thresholds or deadlines are split.
- **R7 — Lead-ins.** A unit that is a point "(a) maintain records of…" inherits subject and
  modality from its lead-in (`heading`). The citation still points at the unit; the span is
  the point's own text.

## 2. Fields

| Field | Rule |
|---|---|
| `subject` | Shortest noun phrase naming the duty-bearer, copied from the text ("a covered company", "institutions"). Not resolved through definitions. |
| `modality` | From the trigger table; negated permissions ("may not") = `must_not`. |
| `action` | Verb phrase without the modal and without the object ("file", "maintain", "notify"). |
| `object` | What the action applies to ("the report", "a Common Equity Tier 1 capital ratio"). Empty if intransitive. |
| `condition` | The *when/if* clause that triggers the duty ("where the institution uses internal models", "if total assets exceed $50 billion"). Empty if unconditional. Nested conditions are joined with " AND " in source order. |
| `threshold` | The binding numeric quantity: `value` as a number (4.5 for "4.5 %"), `unit` ("%", "USD", "EUR", "days", "times", "basis points", "THB"), `comparator` (≥ for "at least", "not less than"; ≤ for "not exceed", "no more than"; == for "equal to"; > / < for "more/less than"). Thresholds that are part of the *condition* (an applicability trigger) stay in `condition` text and are **not** put in `threshold` unless they are the only number in the unit and the obligation is to comply with it. |
| `deadline` | Time limit for performing the action, copied ("within 30 days after the end of each quarter", "by 31 March each year", "without undue delay"). Frequency ("annually") counts as a deadline. |
| `exceptions` | Each "unless", "except", "other than", "provided that … does not", "notwithstanding" carve-out as its own string. |
| `citation` | `source`, `unit_id`, and the trigger sentence. Annotators and models write a verbatim `quote` of the sentence/clause containing the trigger; `span` = its character offsets `[start, end)` in the stored unit text, computed by `schema.generation.resolve` (D-020). A quote that does not resolve makes the obligation ungrounded. |
| `confidence` | Teacher/model self-estimate; gold uses 1.0. Not evaluated except for calibration plots. |

Cross-references ("in accordance with Article 26", "as defined in § 217.2") are kept verbatim
in the field where they occur; they are not resolved.

## 3. Worked examples

Format: unit text → expected obligations (`subject | modality | action | object | condition |
threshold | deadline | exceptions`). `—` = empty. Span omitted for brevity (it is always the
trigger sentence).

### 3.1 Plain duties
1. "Institutions shall at all times satisfy a Common Equity Tier 1 capital ratio of 4.5 %."
   → institutions | must | satisfy | a Common Equity Tier 1 capital ratio | — |
   4.5 % ≥ | at all times | — ("at all times" is a continuous-compliance deadline,
   not a condition).
2. "A covered company must file the report with the Board within 30 days after the end of each
   calendar quarter." → a covered company | must | file | the report with the Board | — |
   — | within 30 days after the end of each calendar quarter | —
   (30 days is a deadline, not a threshold.)
3. "The firm must maintain adequate records of all transactions." → the firm | must |
   maintain | adequate records of all transactions | — | — | — | —
4. "Each bank shall establish and maintain a written liquidity risk management policy." →
   one obligation, action "establish and maintain" (R6).
5. "A licensee shall notify the Authority of any change in its controllers and shall obtain
   approval before appointing a director." → **two** obligations (different actions/objects).

### 3.2 Prohibitions
6. "A firm must not pay remuneration that is linked to sales targets." → must_not | pay |
   remuneration that is linked to sales targets.
7. "No person shall act as a broker unless registered with the Commission." → subject "no
   person" → normalise subject to "person", modality must_not, action "act", object "as a
   broker", exceptions ["unless registered with the Commission"].
8. "An institution may not include in Tier 2 capital any instrument that…" → must_not (R:
   negated permission).
9. "ห้ามมิให้สถาบันการเงินให้สินเชื่อแก่ผู้ถือหุ้นรายใหญ่เกินกว่าร้อยละ 25 ของเงินกองทุน" →
   subject สถาบันการเงิน | must_not | ให้สินเชื่อ | แก่ผู้ถือหุ้นรายใหญ่ | — | 25 % > (of
   capital; unit "%") | — | —. Note: "เกินกว่า" = ">" ; the prohibition of exceeding is encoded
   as must_not + comparator ">" (not must + "≤"). **Rule T1:** keep the polarity the text uses.

### 3.3 Permissions vs possibility
10. "An institution may, with the prior permission of the competent authority, use internal
    models to calculate own funds requirements." → may | use | internal models | "with the
    prior permission of the competent authority" (condition).
11. "Market movements may cause the value of collateral to decline." → `[]` (R4).
12. "The Commission may, by order, exempt any person from this section." → subject the
    Commission | may | exempt | any person from this section (R1: authority powers are
    annotated).
13. "Firms should consider the guidance in Annex 2." → should | consider | the guidance in
    Annex 2.
14. "The PRA expects firms to review their recovery plans annually." → subject firms | should |
    review | their recovery plans | — | — | annually.

### 3.4 Conditions, nesting and thresholds
15. "Where a covered company's total consolidated assets exceed $250 billion, it must conduct a
    stress test semi-annually." → condition "Where a covered company's total consolidated
    assets exceed $250 billion"; threshold **none** (applicability trigger stays in
    condition); deadline "semi-annually".
16. "If the leverage ratio falls below 3 % and the institution does not restore it within three
    months, the institution shall submit a capital plan." → condition "If the leverage ratio
    falls below 3 % AND the institution does not restore it within three months"; threshold
    none; deadline —.
17. "The ratio of liquid assets to net outflows shall be not less than 100 %." → subject
    implicit (R5: the institution named in the lead-in) | must | be | the ratio of liquid
    assets to net outflows | — | 100 % ≥ | — | —. Do not invent a verb such as "maintain";
    the action is the copula "be". Class `copula-requirement`.
18. "Large exposures shall not exceed 25 % of Tier 1 capital." → must_not | exceed | — |
    — | 25 % > (unit "% of Tier 1 capital" normalised to "%") | — | —. The text uses a
    negated duty, so polarity stays must_not and the comparator marks the line not to be
    crossed (rule T1, as in ex. 9).
19. "…a minimum of 8 % of risk-weighted assets…" → threshold 8 % ≥.
20. "…no more than 10 business days…" as a time limit for acting → deadline, not threshold.

### 3.5 Exceptions
21. "A firm must publish its annual report, except where it is a small firm as defined in
    GEN 1.2." → exceptions ["except where it is a small firm as defined in GEN 1.2"].
22. "Notwithstanding paragraph (a), a bank may defer the disclosure for up to 12 months." →
    the "notwithstanding" clause is **not** an exception of this obligation (it carves out
    *another* paragraph); annotate may | defer | the disclosure | — | — | for up to 12 months
    (deadline) | —.
23. "Paragraph 1 shall not apply to institutions that are exempt under Article 7." → this is
    a scope carve-out, not a duty on anyone → `[]`; record class `scope-carveout`.
24. "…provided that the exposure is fully collateralised." → exception? No: "provided that"
    introducing a **requirement for the permission to hold** is a `condition`. It is an
    exception only when it removes the duty ("provided that this does not apply to…").

### 3.6 Cross-references and lists
25. "An institution shall report the information referred to in Article 430(1) to the
    competent authority." → object "the information referred to in Article 430(1)" (kept
    verbatim; not resolved).
26. Lead-in "A bank must:" with point "(b) appoint a chief risk officer;" → unit is (b);
    subject "a bank", modality must (inherited, R7); action "appoint"; object "a chief risk
    officer".
27. "The following shall be deducted from Common Equity Tier 1 items: (a) losses for the
    current year;" → unit (a): subject implicit "institutions" (R5), must | deduct | losses
    for the current year | — | — | — | —.

### 3.7 Non-obligations
28. "'Credit institution' means an undertaking the business of which is to take deposits…" →
    `[]` (R2).
29. "This Regulation shall enter into force on the twentieth day following its publication." →
    `[]` (R3, commencement).
30. "In Article 4(1), point (7) is replaced by the following:" → `[]` (amending text).
31. "The purpose of this section is to ensure that firms treat customers fairly." → `[]`.
32. "[Reserved]" → never reaches annotation (filtered by parser).

### 3.8 Hard cases
33. **Double modality.** "A firm must ensure that its employees do not disclose…" → one
    obligation on the firm: must | ensure | that its employees do not disclose … — do **not**
    add a second must_not on employees.
34. **Duty on authority + duty on firm.** "The competent authority shall require institutions to
    hold additional own funds where…" → one obligation, subject the competent authority |
    must | require | institutions to hold additional own funds; the firm's duty is derivative
    and not extracted here.
35. **Guidance quoting a rule.** "SYSC 4.1.1R requires a firm to have robust governance
    arrangements." (in a guidance paragraph) → `[]` for the guidance unit; the obligation is
    annotated where the rule itself appears. Record class `restated-rule`.
36. **Thai passive "ให้".** "ให้ธนาคารพาณิชย์จัดส่งรายงานต่อธนาคารแห่งประเทศไทยภายในสามสิบวัน"
    → subject ธนาคารพาณิชย์ | must | จัดส่ง | รายงานต่อธนาคารแห่งประเทศไทย | — | — | ภายในสามสิบวัน.
    Thai numerals in words ("สามสิบ") are copied into `deadline` as written; when used as a
    threshold the `value` is the Arabic number (30).

## 4. Normalisation for scoring

- Strings compared after: Unicode NFKC, lowercase (en), whitespace collapse, strip trailing
  punctuation, strip leading articles (a/an/the) for `subject` and `object`.
- `threshold` exact match: `value` equal to 1e-9 relative tolerance, `unit` after a small
  alias map (% = per cent = percent = ร้อยละ), comparator equal.
- `deadline` exact match on normalised string; a second relaxed metric extracts
  (number, time-unit) pairs and compares those.
- Obligation matching between prediction and gold: Hungarian assignment on span IoU ≥ 0.5 and
  equal modality; unmatched predictions are false positives ("hallucinated obligations").

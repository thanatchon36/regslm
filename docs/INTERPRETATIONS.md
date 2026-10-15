# INTERPRETATIONS

Annotation-guide ambiguities met during gold adjudication (D-048). Each line: the case, the
reading adopted, the alternative. Row numbers refer to `data/gold/adjudicate_claude.csv`.

- **Power vs duty in "มีอำนาจหน้าที่"** (row 1): encoded `must` (หน้าที่ is a must trigger); alternative `may`.
- **Authority rule-making powers** ("ให้ ธปท. มีอำนาจประกาศกำหนด…", ESMA "shall develop draft RTS/ITS"; rows 27, 28, 152-153, 261, 265): delegation → `[]` (R3). But "ESMA shall develop guidelines" (row 168) was kept as `must`; borderline.
- **Non-party subjects** (documents, plans, systems, passives with no agent; rows 72-74, 88, 98, 138, 199, 242, 255, 257): `[]` under R1. Exceptions kept where the guide's example 16 applies (limits on a quantity, row 112) or example 40 (written agreement as subject, row 245); row 202 kept "the plan" as subject. Main source of inconsistency risk.
- **Nested lead-ins** (rows 73, 79, 81, 106): only the nearest colon lead-in is inherited (R7); a root lead-in higher up is not.
- **"ไม่ควร" / "should not"** (rows 17, 223): no should_not modality; row 17 encoded `must_not`, row 223 `should` + "not rely". Inconsistent; scorer treats both as modality disagreement either way.
- **"will" without a modal** (Board/Commission "will notify / will publish"; rows 101, 237): row 101 encoded `must`, row 237 `[]`. Borderline.
- **Split vs merge of coordinated actions** (rows 25, 179, 184, 186): split when actions differ (R6, ex. 5).
- **Numbers both trigger and binding** (rows 144, 174): 144 kept in `condition`, 174 as threshold `>= 250 %`.
- **Garbled / truncated text** (rows 6, 12, 171, 176, 182): decided on the readable content; truncated fragments without an actor → `[]`.

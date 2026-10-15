# Source licences and handling policy

Evidence (verbatim quotes, URLs, robots.txt): [LICENSES_research.md](LICENSES_research.md).
Decision log: docs/DECISIONS.md D-010–D-013. Not legal advice; conservative reading throughout.

Tiers:
- **P (permissive)** — text may be committed to this repo and redistributed; weights trained on it may be released.
- **L (local-only)** — only URL + SHA-256 + fetch script committed; text stays in `data/local/`; may be used for local research evaluation; **not** in training of any released model.
- **X (excluded pending owner)** — not fetched/trained until the owner decides (see PROGRESS "Waiting on owner").

| Source | Tier | Commit text | Train (research) | Release weights trained on it | Attribution |
|---|---|---|---|---|---|
| US eCFR (Titles 12, 17) | P | yes | yes | yes | "Source: eCFR (ecfr.gov) / GovInfo; unofficial, not the official legal edition of the CFR." |
| EUR-Lex (CRR, CRD, MiFID II, AMLD) | P | yes | yes | yes | "© European Union, https://eur-lex.europa.eu/, CC BY 4.0; consolidated texts are documentation only; changes: segmented into paragraphs, markup removed." |
| FCA Handbook (SYSC, COBS, SUP) | P (OGL v3.0) | yes | yes | yes | "Contains information from the FCA Handbook, © FCA, licensed under the Open Government Licence v3.0." — **access only via a sanctioned route** (FCA terms cl. 35.3 bars scraping without consent); see NEEDED.md. |
| Thai statutes & government notifications (Thai text) | P (Copyright Act B.E. 2537 s. 7(2)–(3): not copyright works) | yes | yes | yes | courtesy: "Source: Royal Gazette / Office of the Council of State" |
| Thai government-made English translations | P (s. 7(5)) | yes | yes | yes | "Unofficial translation; the Thai text prevails." |
| Thai third-party (law-firm) translations | L | no | eval only | no | credit preparer |
| BIS Basel Framework | L | no | eval only | no | "Source: BIS, Basel Framework" |
| HKMA SPM | L | no | eval only | no | "Source: HKMA" |
| PRA Rulebook | X (owner decision 2026-10-15: excluded) | no | no | no | — |
| MAS Notices & Guidelines | X (owner decision 2026-10-15: excluded) | no | no | no | — |

Consequences:
- **One model variant**, not two: since every L/X source is excluded from training, all trained weights are releasable. L sources are used only as additional out-of-distribution *evaluation* sets (held locally, numbers reported, text not published). (D-011)
- **Held-out rulebook = FCA SYSC** (OGL, permissive) per MISSION §1.5, if a sanctioned access route exists; fallback HKMA SPM (local-only, evaluation use). (D-012)

# Source licence research (as of 2026-10-15)

Scope: reuse terms for regulatory sources to be (a) committed to a public GitHub repo and (b) used to train ML models whose weights are released publicly. This is desk research, not legal advice. Quotes are short and taken from the primary pages fetched on 2026-10-15; some were retrieved through a page-summarising fetch tool, flagged "(via fetch summary)" where I could not re-read the raw HTML.

Definitions: **redistributable** = may commit the text to a public repo. **research_use** = may download and train on it locally. **derivative_ok** = may publicly release weights trained on it.

## Summary table

| # | Source | redistributable | research_use | derivative_ok | Required attribution string | One-line justification |
|---|---|---|---|---|---|---|
| 1 | BIS Basel Framework | unclear (leaning no) | yes (non-commercial) | unclear (ask BIS) | "Source: Bank for International Settlements (BIS), Basel Framework, https://www.bis.org/basel_framework/" | Non-commercial redistribution allowed, but reproduction is capped at a "limited extract" (400 words / 10%), and nothing addresses TDM or models. |
| 2 | UK PRA Rulebook | no | unclear (personal / in-firm use only) | no | "Source: Prudential Regulation Authority Rulebook, © Bank of England" (author status must be acknowledged) | All rights reserved; OGL is NOT stated on the terms page; distribution to third parties and commercial use forbidden without licence. |
| 3 | FCA Handbook | yes | yes (but use the API, not scraping) | yes | "© FCA. Contains information from the FCA Handbook, licensed under the Open Government Licence v3.0." | Terms cl. 25 licence the Site and API content under OGL v3.0; cl. 35.3 bans scraping without consent, so fetch via the Handbook API or get consent. |
| 4 | HKMA Supervisory Policy Manual | no | unclear (leaning yes, local non-commercial) | no (without written consent) | "Source: Hong Kong Monetary Authority (HKMA), https://www.hkma.gov.hk/" | Only non-commercial display/download/reproduce/print, unmodified, with attribution; any other use needs prior written consent. |
| 5 | MAS Notices and Guidelines | no | unclear (leaning no beyond one personal copy) | no | None granted. Dataset licence (not applicable to Notices) requires a "conspicuous notice acknowledging the source" | Reproduction/republication needs prior written MAS permission; caching prohibited; the open dataset licence covers datasets/APIs only. |
| 6a | Thai statutes, regulations, notifications (Thai text) | yes | yes | yes | None legally required; courtesy: "Source: Royal Gazette / Office of the Council of State, Thailand" | Copyright Act B.E. 2537 s. 7(2)-(3) says laws, regulations, notifications, orders of government units are not copyright works. |
| 6b | Official English translations by a government unit | yes | yes | yes | Courtesy: "Unofficial translation; Thai text prevails" | s. 7(5) excludes "translation and collection of those" made by Ministries/Departments/government units. |
| 6c | English translations prepared by law firms (e.g. SEC Act .doc prepared with Chandler and Thong-ek) | unclear | yes (local) | unclear | Credit the preparer | s. 7(5) covers only government-made translations; third-party translators may hold copyright. |
| 7a | eCFR (US) | yes | yes | yes | None required; "Source: eCFR, ecfr.gov (unofficial; not the official legal edition of the CFR)" | US government work, 17 USC 105; caveat for reprinted third-party material. |
| 7b | EUR-Lex (EU) | yes | yes | yes | "© European Union, https://eur-lex.europa.eu/, licensed under CC BY 4.0; changes indicated" | Commission reuse policy, Decision 2011/833/EU; CC BY 4.0 per the Commission legal notice (Decision C(2019) 1655 not directly confirmed, see section 7). |

Practical recommendation: commit to the public repo and train freely on the FCA Handbook (via API), Thai primary law (Thai text and gov-made translations), eCFR and EUR-Lex. For BIS, PRA, HKMA and MAS, do not commit full text; keep URLs plus fetch scripts, train only after written permission (or restrict to non-commercial, non-redistributed local use for BIS/HKMA), and do not release weights without a written answer from each regulator.

---

## 1. BIS Basel Framework

- Terms page: https://www.bis.org/about/terms-conditions (redirected from https://www.bis.org/terms_conditions.htm; section "Copyright and permissions").
- Operative text (verbatim):
  - "Except as outlined below, all rights are reserved."
  - "Users may download, display, print out, photocopy or redistribute any BIS Material for non-commercial purposes."
  - "Users may reproduce a limited extract of BIS Material (other than the statistics published in the BIS Data Portal) in other publications or products free of charge, provided the BIS is cited as the source."
  - "a 'limited extract' means any extract of not more than 400 words of text or two tables or graphs"
  - "Any proposed use of BIS Material as described in (b) above which does not meet the above conditions requires written permission from the BIS."
  - Translations: "Users must include a statement that the translation is not an official BIS translation".
- No mention of text and data mining, machine learning or automated access. Commercial use is not separately granted.
- Reading: a public non-commercial research repo can arguably "redistribute" under (a), but the licence is not open (no derivative-works grant, no CC/OGL). Weights trained on the full text are not addressed; the safe route is a written permission request. Redistribution of the whole Framework in a repo exceeds the "limited extract" clause if treated as a reproduction in a "product".
- robots.txt: https://www.bis.org/robots.txt (HTTP 200). `User-agent: *`, no Crawl-delay. Disallows include `/core/`, `/profiles/`, `/search/`, `/p/`, and notably "Hide the Basel Framework standards, only chapters should be indexed": `Disallow: /committees/bcbs/basel-framework/standard/*/$` and `.../standard/*/?` (and the same for basel-consolidated-guidelines/module). Chapter pages are not disallowed; the per-standard URLs with trailing slash or query string are.

## 2. UK PRA Rulebook

- Terms page: https://www.prarulebook.co.uk/legal (page title "Terms of Use | Prudential Regulation Authority Handbook & Rulebook"; footer shows "©2026 Bank of England").
- Operative text (verbatim):
  - "we are the owner or the licensee of all intellectual property rights in our site, and in the material published on it."
  - "All such rights are reserved."
  - "You may print off copies, or download extracts, of any page from our site for your personal use or use within an individual firm or organisation" (raw HTML confirmed).
  - "You must not distribute copies or downloads made to third parties" (raw HTML confirmed, truncated).
  - "Our status (and that of any identified contributors) as the author of content on our site must always be acknowledged."
  - "You must not use any part of the content on our site for commercial purposes (either in print or online) without obtaining a licence to do so from us."
- Open Government Licence: NOT applicable per the terms page. The terms page does not mention the OGL at all (contrast the FCA Handbook, whose terms expressly invoke OGL v3.0). Do not assume the OGL applies because the PRA is a UK public body. Crown copyright OGL applies only where expressly stated. Licence enquiries go via the PRA Contact Us page.
- No statement on scraping/TDM found.
- robots.txt: https://www.prarulebook.co.uk/robots.txt (HTTP 200): `User-agent: *`, `Disallow: /praerror`, `/search`, `/test-folder`, `/*?*p=1`; sitemap at `/_api/sitemap/getsitemap`. No Crawl-delay.

## 3. FCA Handbook

- Terms page: https://www.handbook.fca.org.uk/footer-widget/terms-and-conditions (the Handbook also resolves at https://handbook.fca.org.uk/). The Legal Information page (https://www.handbook.fca.org.uk/footer-widget/legal-information) only shows the notice "Copyright © 2026 FCA. All rights reserved." and points to the Terms. (Clause text via fetch summary; the site is a JavaScript single-page app so I could not grep raw HTML.)
- Operative text (verbatim):
  - Cl. 23.1: "re-used or adapted under the Open Government Licence v 3.0"
  - Cl. 25: "You may use the information on the Site or the API in accordance with the terms of the Open Government Licence v3.0."
  - Cl. 26: "The source of Our material should be acknowledged in all copies," "either by appropriate words or by adding the FCA copyright symbol (i.e. © FCA)."
  - Cl. 35.3 (automated access): "use nor attempt to use any 'scraper', 'robot', 'bot', 'spider', 'data mining', 'crawler', ... to access, acquire, copy, or monitor any portion of the website ... without Our prior written consent, other than by access to the API Data through the API;"
  - Cl. 41: "You may not use the Financial Conduct Authority's or any third party's name, logo or any other identifying mark".
- Reading: OGL v3.0 permits copying, publishing, adapting and commercial use with attribution, so redistribution and trained weights are fine. Obtain data through the Handbook API (explicitly carved out of the bot ban) rather than crawling HTML. Some Handbook material is third-party (cl. 23.2 EUR-Lex under CC BY 4.0, cl. 23.3 ESMA material under ESMA's copyright notice), so check provenance of EU-derived provisions.
- robots.txt: https://www.handbook.fca.org.uk/robots.txt returns HTTP 200 but the body is the SPA's HTML shell (about 165 KB), i.e. no real robots.txt exists. Treat as no machine-readable crawl policy; the contractual bot ban in cl. 35.3 is the operative restriction.

## 4. HKMA Supervisory Policy Manual

- Terms page: https://www.hkma.gov.hk/eng/other-information/terms-and-conditions-of-use/ (linked from the HKMA homepage footer; my guessed `/copyright-notice/` URL returned 404). The SPM landing page URL I guessed also returned 404; find SPM under the HKMA "Key Functions > Banking > Supervisory Policy Manual" menu.
- Operative text (verbatim):
  - Para 6: "Unless otherwise stated, the HKMA is the owner of the copyright and any other intellectual property right"
  - Para 12: "you may only, for non-commercial purposes, display, download, reproduce, and print in its original format the Information on the Website or link to the Website on condition that:" (a) subject to the Terms; (b) "the Information must be reproduced accurately without any modification;" (c) "you shall give proper attribution to the HKMA and the Website by identifying clearly the source of the Information"
  - Para 12(e): "at the request of the HKMA, you shall forthwith destroy any copy of the Information" (via fetch summary).
  - Para 13: "The HKMA's prior written consent is required for linking to the Website or the use of the Information" (other than as permitted; via fetch summary).
  - Para 17: the APIs may be used "whether for commercial or non-commercial purposes" to develop software (applies to HKMA APIs, not to SPM text).
  - Para 10: HKMA may "block access from a particular Internet Protocol (IP) address".
- Reading: unmodified, attributed, non-commercial copy for local use is permitted; anything else (derivative works, public redistribution, weights) needs prior written consent. The takedown clause (12(e)) makes a public repo risky.
- robots.txt: https://www.hkma.gov.hk/robots.txt returns HTTP 404 (HTML error page). No robots policy published; IP blocking reserved in the terms.

## 5. MAS Notices and Guidelines

- Terms page: https://www.mas.gov.sg/terms-of-use (HTTP 200).
- Operative text (verbatim):
  - "No Content, including without limitation the website design and layout, graphics, images, hyperlinks and information, may be reproduced, republished, uploaded, posted, transmitted or distributed in any way without the prior written permission of MAS, except that you may download one copy of any information provided in this Web Site onto a single computer for your own personal non-commercial use provided that you keep intact all accompanying copyright and other proprietary notices."
  - "Except as set forth below, caching and links to, and the framing of this Web Site or any of the Contents are prohibited."
  - Dataset licence (separate section, applies to "datasets", e.g. APIs/data portal, not to Notices): "MAS grants you a worldwide, perpetual, royalty-free, non-exclusive licence to Use the datasets, subject to the terms of this Licence." Attribution: "a conspicuous notice acknowledging the source of the datasets and including a link to the most recent version of this Licence."
- Reading: Notices and Guidelines are web "Contents", so the default is all rights reserved with one-personal-copy exception. Training, committing to a repo and weight release all require written MAS permission. Do not rely on the dataset licence unless MAS publishes the specific material as a dataset.
- Automated access: no explicit ban on robots/scrapers in the Terms of Use (only the "unreasonable or disproportionately large load" clause and the caching prohibition). Site did not block curl: HTTP 200 for the terms page and for the notices listing, with both a browser-like and a "curl-bot/1.0" user agent (the /regulation/notices URL redirects to `/regulation/regulations-and-guidance?content_type=Notices`).
- robots.txt: https://www.mas.gov.sg/robots.txt (HTTP 200), complete content: `User-Agent: *`, `Crawl-delay: 2`, `Sitemap: https://www.mas.gov.sg/sitemap.xml`. No Disallow rules. Honour the 2-second delay.

## 6. Thailand

### 6.1 Copyright Act B.E. 2537 (1994), Section 7

- Source: English translation "COPYRIGHT ACT B.E. 2537 (1994)" hosted at https://www.ananda-ip.com/files/Copyright-Act.pdf (unofficial translation by a law firm; text extracted from the PDF). The Thai text, published in the Royal Gazette, is authoritative. Wikisource also carries a WIPO-based translation (https://en.wikisource.org/wiki/Category:PD-TH-exempt_-_section_7(1) and siblings), and WIPO Lex hosts translations.
- Section 7 (verbatim from that translation): "The followings are not deemed copyright work by virtue of this Act: (1) news of the day and facts having the character of mere information which is not a work in literary, scientific or artistic domain, (2) constitution and legislations, (3) regulations, by-laws, notifications, orders, explanations and official correspondence of the Ministries, Departments or any other government or local units, (4) judicial decisions, orders, decisions and official reports, (5) translation and collection of those in (1) to (4) made by the Ministries, Departments or any other government or local units."
- Conclusion: yes, it is Section 7 (items 2, 3, 5 matter here). Acts, royal decrees, ministerial regulations, and notifications of agencies such as the Bank of Thailand and SEC are outside copyright. Caveats: (i) the exclusion covers the legal text, not necessarily the website's design, database or editorial commentary; (ii) s. 7(5) covers translations only when made by government units; (iii) the Act was amended after 1994 (No. 2 B.E. 2558, No. 3 B.E. 2558, No. 4 B.E. 2561, No. 5 B.E. 2565); I did not confirm that s. 7 text is unchanged, though I found no report that it was; verify against the current consolidated Thai text before final sign-off.

### 6.2 Where the FIBA B.E. 2551 and SEA B.E. 2535 are officially published

- Royal Gazette (Thai text, official): https://ratchakitcha.soc.go.th/ (Secretariat of the Cabinet).
- Office of the Council of State (Krisdika) law database: https://www.krisdika.go.th/ (Thai laws, plus an English-translation section).
- Bank of Thailand publishes FIBA-related regulations and its own notifications on https://www.bot.or.th/ (the FIBA notification PDF I fetched, https://www.bot.or.th/content/dam/bot/fipcs/documents/FPG/2551/EngPDF/25510296.pdf, is "BOT Notification No 6 - 2551", headed "Unofficial Translation. This translation is for the convenience of those unfamiliar with the Thai language. Please refer to Thai text for the official version", with no copyright notice that I could read).
- SEC Thailand: https://www.sec.or.th/ publishes an English Securities and Exchange Act at https://www.sec.or.th/TH/Documents/ActandRoyalEnactment/Act/translate-secact.doc (found via search; header reported as unofficial and revised for the 2008, 2016 and 2019 amendments, originally prepared by the SEC Office with Chandler and Thong-ek Law Offices). Amendment translations are at https://publish.sec.or.th/nrs/ (for example 4460se.pdf).
- An unofficial English FIBA translation exists on the National Anti-Corruption Commission site (https://www.nacc.go.th/tacc/upload/files/download_201305291624252.pdf) with the disclaimer that the Thai text is official; no copyright statement noted in search excerpt. I did not locate an English FIBA on krisdika.go.th. All these translations state that only the Thai text is official.
- Copyright notices on translations: none found on the BOT or NACC translations; the SEC .doc could not be opened (see blocking below), so I could not confirm whether it carries a notice. Because the translator of the SEC .doc included a private law firm, s. 7(5) may not cover it (unclear).

### 6.3 robots.txt and access

- https://ratchakitcha.soc.go.th/robots.txt: HTTP 403 with a Cloudflare "Just a moment..." challenge (same for `/`). Automated clients are actively challenged; no robots.txt could be read. Needs a browser or a manual download; do not bypass the challenge.
- https://www.krisdika.go.th/robots.txt: over HTTPS the connection failed in my first attempt (HTTP 000); an HTTP request to `/robots.txt` returned 404 and `/` over HTTPS-insecure returned 404 on a retry. Treat as no robots policy and unreliable availability.
- https://www.sec.or.th/robots.txt: HTTP 403 "Block Response" (WAF) for curl; the SEC .doc translation was also blocked (403). Automated access is blocked.
- https://www.bot.or.th/robots.txt: HTTP 200. `User-agent: *`, `Disallow: /_layouts/`, `/_vti_bin/`, `/_catalogs/`; sitemaps for th and en. No crawl-delay. Content PDFs are crawlable.

## 7. Confirmations

### 7a. eCFR

- 17 U.S.C. section 105: "Copyright protection under this title is not available for any work of the United States Government" (statute quoted from memory, not re-fetched; see https://www.law.cornell.edu/uscode/text/17/105).
- Supporting (secondary, via search): the Library of Congress record for the CFR states the vast majority of the CFR is public domain as a US government work; the OFR/GPO CFR XML guide says they do not restrict downstream uses. eCFR pages state it "is not an official legal edition of the CFR".
- Caveat: copyrighted material reprinted/incorporated by reference inside a regulation (e.g. ISO/ANSI standards) may still be protected.
- Access: https://www.ecfr.gov/robots.txt (HTTP 200): `User-agent: *`, Sitemap at `/sitemaps/sitemap.xml.gz`, `Disallow: /search`, `/recent-changes`, `/on/`, `/compare/`, `/my/`, `/auth/ofr`, `/auth/sign_in`, `/api/renderer/v1/content/`, `/api/versioner/v1/full/`. Note the Disallow of `/api/versioner/v1/full/` (bulk full-text endpoint); use the bulk data at govinfo.gov or `/api/versioner/v1/` structure/content endpoints as appropriate. The WebFetch tool was redirected to `https://unblock.federalregister.gov/` on ecfr.gov reader-aid pages, which suggests bot-blocking on some paths; plain curl got HTTP 200 on `/` and the versioner titles API.

### 7b. EUR-Lex

- Reuse notice: Commission legal notice, https://commission.europa.eu/legal-notice_en (fetched): "Unless otherwise indicated (e.g. in individual copyright notices), content owned by the EU on this website is licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0) licence." It cites "Commission Decision of 12 December 2011 on the reuse of Commission documents" (CELEX:32011D0833). Credit must be given and changes indicated. "To use or reproduce content that is not owned by the EU, you may need to seek permission directly from the rightholders."
- Decision C(2019) 1655: I could not confirm it from the pages I could reach (the Commission page did not mention it, and the EUR-Lex legal notice page was unreachable). My understanding, unverified here, is that Decision C(2019) 1655 (12 March 2019) amended the 2011 reuse decision so that default reuse is CC BY 4.0; the CC BY 4.0 statement on the Commission legal notice is consistent with this. Treat "CC BY 4.0 via C(2019) 1655" as probable but confirm by fetching the EUR-Lex legal notice in a browser.
- Caveat: EUR-Lex states that only legislation in the Official Journal is authentic; consolidated texts are documentation tools. Some documents (third-party, ESMA material with its own notice) are excluded.
- EUR-Lex robots.txt: https://eur-lex.europa.eu/robots.txt returned HTTP 202 with an empty body (AWS WAF challenge for non-browser clients); the legal-notice page also returned 202/empty. Could not read robots.txt. Use the official Cellar/SPARQL endpoints or the OP bulk-download channels rather than scraping.

---

## Unreachable / unverified items

- EUR-Lex legal notice and robots.txt (HTTP 202 WAF challenge): not read.
- ratchakitcha.soc.go.th (HTTP 403 Cloudflare), sec.or.th (403 WAF), krisdika.go.th (unreliable; no terms page read): terms and robots.txt not read.
- HKMA: robots.txt does not exist (404); SPM-specific copyright page not found separate from the general terms.
- FCA: robots.txt does not exist (SPA shell); clause text read through a fetch-summary tool.
- Thai Copyright Act: English text is from a law-firm PDF of a translation; verify against the Royal Gazette Thai text and later amendments.
- No source in this list was checked by legal counsel; for BIS, PRA, HKMA, MAS send a written permission request covering (i) repo redistribution, (ii) ML training, (iii) release of derived weights.

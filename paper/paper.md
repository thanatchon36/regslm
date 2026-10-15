---
title: "How Small Can a Regulatory-Obligation Extractor Be? Scaling, Distillation, Verifiable-Reward RL and Quantization for On-Premises SupTech"
author: "Thanatchon Chongmankhong"
date: "DRAFT — results pending"
bibliography: ../docs/literature/refs.bib
---

> **Disclaimer.** Independent open-source project. Public documents only. Not the view of any
> institution. Not legal, regulatory or compliance advice; model output must be reviewed by a
> qualified person before any use.

<!-- Numbers in {{double braces}} are injected from results/summary.json by paper/build.py.
     Sections marked DRAFT are written before results and must not be edited to fit them;
     the Results sections are written after, against the pre-registration (docs/RESEARCH_QUESTIONS.md, tag prereg-v1). -->

# Abstract

{{abstract}} <!-- written last, from the verdict table -->

# 1. Introduction (DRAFT)

Supervisors and banks read rulebooks for obligations: who must do what, under which condition,
by when, subject to which threshold, citing which paragraph. Large hosted language models can do
part of this, but supervisory and compliance material often cannot leave the institution. The
practical question is therefore not "can an LLM extract obligations?" but **"what is the smallest
model that runs on hardware an institution already owns and does this well enough, and what does
it cost?"**

We answer it with a pre-registered study on one consumer GPU (RTX 4070 Ti, 12 GB) with no hosted
API anywhere in the pipeline. A local 14B teacher (Qwen3-14B, 4-bit AWQ) labels public rulebook
paragraphs from the US eCFR, EU legislation (EUR-Lex) and Thai statutes and Bank of Thailand
notifications. Students from 0.6B to 8B parameters are trained with supervised fine-tuning,
rationale distillation and GRPO with programmatic rewards, at label budgets from 500 to 20,000
units, and are served in bf16, FP8, 4-bit AWQ/GPTQ and GGUF formats on GPU and CPU.

Because public rulebooks are in every base model's pretraining data, a model can score well by
reciting rather than reading. Every result is therefore paired with a numerically perturbed test
set, a held-out rulebook and a post-cutoff slice, and we report a memorisation rate for every
model.

Contributions: (i) `regslm`, an Apache-2.0 library that fetches public rulebooks with provenance,
extracts obligations with citations that resolve to character spans, and serves small models
on-premises; (ii) a pre-registered scaling × method × label-budget × quantization study with
contamination controls; (iii) released weights trained only on permissively licensed text.

# 2. Related work (DRAFT)

{{related_work}} <!-- condensed from docs/literature/REVIEW.md, gap table as Table 1 -->

# 3. Data and schema (DRAFT)

## 3.1 Corpus

| Source | Language | Documents | Units | Licence tier |
|---|---|---:|---:|---|
| US eCFR, Titles 12 and 17 (selected parts; GovInfo bulk XML) | en | 9 | 25,463 | public domain |
| EUR-Lex consolidated CRR, CRD, MiFID II, AMLD4 (Cellar) | en | 4 | 12,130 | CC BY 4.0 |
| Thai statutes (SEC Act, Payment Systems Act, FIBA, BoT Act) and 22 BoT notifications | th | 26 | 5,935 | not copyright works (Thai Copyright Act s. 7) |
| Official English translation of FIBA (parallel set) | en | 1 | 517 | government translation (s. 7(5)) |

Corpus version `2026.10.15-4ce7830f`; every unit traces to a URL and SHA-256. A unit is a
paragraph-level provision with a hierarchical address (e.g. `eurlex/CRR/Art92/1/a`); lead-in
sentences of list points are kept in a `heading` field. Rulebooks whose terms do not permit
redistribution or derivative use (PRA Rulebook, MAS Notices) are excluded; the FCA Handbook
(Open Government Licence) awaits a sanctioned access route. Scraping respected `robots.txt`,
including a parser fix for a standard-library bug that ignores rules after blank lines (§10).

## 3.2 Obligation schema

`Obligation(subject, modality ∈ {must, must_not, may, should}, action, object, condition,
threshold{value, unit, comparator}, deadline, exceptions[], citation{source, unit_id, span},
confidence)`. Models generate a verbatim *quote* of the trigger sentence; character spans are
computed deterministically from the quote. In a pilot, only 49 % of LLM-written character
offsets covered the trigger, while quote resolution failed for {{teacher.unresolved_rate}} of
teacher obligations. The annotation guide (v2.1, 41 worked examples) was revised after an
independent expert review, mainly on comparator polarity, invented subjects, deeming and
delegation clauses, and Thai triggers.

## 3.3 Labels

*Silver:* the teacher labels all evaluation units and every training unit containing a deontic
trigger; training units without one receive a rule-based empty label. *Gold:* 300 stratified
units (240 en, 60 th, 40 without a trigger) annotated by two local passes with different prompts
and models (14B, 8B), disagreements adjudicated by the author. Gold is never used for training
or prompt tuning.

## 3.4 Splits and contamination controls

Splits are grouped by article/section so sibling paragraphs never straddle train and test
(train 34,542; dev 2,155; silver-val 2,000; test 4,407; held-out 641; gold 300). The held-out
rulebook is the EU Anti-Money-Laundering Directive (AMLD4). The perturbed test set rescales
every number in the 440 numeric test units (and shifts dates) while keeping their format; the
memorisation pool contains all 1,129 numeric non-training units under two perturbation seeds.
The post-cutoff slice contains {{postcutoff.n_test}} test units amended on or after 2025-05-01.

# 4. Methods (DRAFT)

*Students:* Qwen3-0.6B/1.7B (LoRA and full fine-tuning), 4B (LoRA, QLoRA), 8B (QLoRA); Typhoon 2
1B/3B for Thai. *SFT:* LoRA r = 16, α = 32 on all linear layers, completion-only loss, sequence
length 2,048, nested label subsets. *Rationale distillation:* the teacher's short rationale is
generated before the labels and kept in the training target; compared with label-only training
at equal target tokens. *GRPO:* from the best SFT checkpoint per size, with a composite reward
(schema validity gate × citation resolution, span grounding, numeric and deadline exactness,
hallucination penalty) plus an explicit recall term, because every precision-type component is
maximised by an empty output. *Quantization:* FP8 (dynamic), AWQ and GPTQ W4A16, GGUF Q8_0 and
Q4_K_M; served with vLLM and llama.cpp (GPU and CPU-only).

# 5. Experimental setup (DRAFT)

Hardware: Intel i5-13500, 32 GB RAM, RTX 4070 Ti 12 GB under WSL2; peak VRAM ≤ 11 GB per run.
Budget: ≤ 12 GPU-hours of teacher labelling and ≤ 100 GPU-hours of experiments, enforced by a
pre-committed rule that never reads results. Pre-registration: hypotheses, metrics, thresholds,
smallest effects of interest and the budget rule were committed and tagged `prereg-v1` before
any training run. Statistics: paired unit-level bootstrap with a per-seed consistency
requirement; SESOI 2 OF1 points (3 for Thai and field-level, 5 for the 440 numeric units and
gold); Holm correction within declared families.

# 6–10. Results, memorisation, threats to validity, reproducibility, conclusion

{{results}} <!-- per RQ, verdict table first; written after the runs -->

## Threats to validity (pre-listed, DRAFT)

- Silver labels: most scores measure agreement with the teacher, not correctness; gold (N = 300)
  is small and adjudicated by one person.
- One main model family (Qwen3); Gemma 3 was dropped; SEA-LION enters only if budget allows.
- English-heavy corpus; Thai and English test units come from different instruments.
- Jurisdiction coverage: UK and Singapore rulebooks are excluded for licence reasons.
- Single GPU, 2–3 seeds per cell; seed variance is reported but not inferentially modelled.
- Prompt sensitivity: the teacher prompt went through five versions before labelling (§3.2).

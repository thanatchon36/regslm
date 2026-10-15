# Literature review

35 sources, each verified against its arXiv / ACL Anthology / DOI / official page (2026-10-15).
BibTeX: [refs.bib](refs.bib). Cluster drafts were produced by literature subagents and edited
here; the gap table at the end is the synthesis that motivates the study design.

## Literature cluster A: obligation extraction, SupTech LLMs, Thai NLP and tokenization

All entries were opened and checked (title, authors, year, venue) on 2026-10-15. BibTeX is in `cluster_a.bib`.

### 1. Obligation, deontic and regulatory-text extraction; legal benchmarks

**gokhan2024rirag** Gokhan, T., Wang, K., Gurevych, I., Briscoe, T. (2024). RIRAG: Regulatory Information Retrieval and Answer Generation. arXiv:2409.05677 (preprint; no peer-reviewed venue listed). https://arxiv.org/abs/2409.05677
Introduces the regulatory retrieval and answer-generation task and the ObliQA dataset built from regulatory documents (ADGM). Answers are grounded in obligation-bearing passages and evaluated for retrieval and generation quality.
Relevance: closest prior work on obligations in financial regulation, but it is passage-level QA, whereas regslm extracts structured fields (subject, modality, action, condition, threshold, deadline, exceptions, citation).

**oneill2017modality** Ó Neill, J., Buitelaar, P., Robin, C., O'Brien, L. (2017). Classifying Sentential Modality in Legal Language: A Use Case in Financial Regulations, Acts and Directives. ICAIL 2017, London. DOI 10.13025/21099. https://research.universityofgalway.ie/en/publications/classifying-sentential-modality-in-legal-language-a-use-case-in-f/
Classifies sentences from financial regulation as obligation, prohibition or permission using ensembles of neural networks over legal and general distributional embeddings, reaching 82.33% accuracy on held-out expert-annotated sentences.
Relevance: the baseline for our modality field (must/must_not/may), which we extend with should and with full argument structure rather than sentence-level labels.

**guha2023legalbench** Guha, N., Nyarko, J., Ho, D. E., Ré, C., Chilton, A., et al. (2023). LegalBench: A Collaboratively Built Benchmark for Measuring Legal Reasoning in Large Language Models. arXiv:2308.11462 (preprint; published at NeurIPS 2023 Datasets and Benchmarks, not stated on the arXiv page). https://arxiv.org/abs/2308.11462
A collaboratively built suite of legal reasoning tasks used to evaluate LLMs across several reasoning types.
Relevance: a general legal-reasoning sanity check for our models; it does not cover regulatory obligation extraction.

**hendrycks2021cuad** Hendrycks, D., Burns, C., Chen, A., Ball, S. (2021). CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review. NeurIPS 2021 (Datasets and Benchmarks); arXiv:2103.06268. https://arxiv.org/abs/2103.06268
Expert-annotated contract clause extraction dataset framed as span extraction.
Relevance: a template for expert-annotated span extraction and its evaluation, but on contracts rather than regulation and without deontic structure.

**chalkidis2022lexglue** Chalkidis, I., Jana, A., Hartung, D., Bommarito, M., Androutsopoulos, I., Katz, D. M., Aletras, N. (2022). LexGLUE: A Benchmark Dataset for Legal Language Understanding in English. ACL 2022; arXiv:2110.00976. https://arxiv.org/abs/2110.00976
A unified benchmark of English legal classification and multiple-choice tasks for evaluating legal language models.
Relevance: standard baseline for legal-domain NLU, useful as a catastrophic-forgetting or transfer check; no extraction of structured norms.

### 2. SupTech and RegTech applications of LLMs

**prenio2025genai** Prenio, J. (2025). Starting with the basics: a stocktake of gen AI applications in supervision. FSI Briefs No 26, Bank for International Settlements, 12 June 2025. https://www.bis.org/fsi/fsibriefs26.pdf
Surveys how financial authorities test, build and use generative AI in supervision, grouping applications into document processing, knowledge management and document review. It identifies legacy IT, data security, skills gaps, low user acceptance and inaccurate outputs as the main obstacles.
Relevance: establishes the supervisory demand for document-processing tools and motivates our focus on small, on-premises, verifiable extraction.

**bisih2022ellipse** BIS Innovation Hub (2022 onward). Ellipse and the Data and Knowledge Platform (DKP). Project page, with the Monetary Authority of Singapore. https://www.bis.org/innovation/bis_open_tech_ellipse.htm
Describes Project Ellipse, a prototype integrated regulatory data and analytics platform combining structured and unstructured data to give supervisors early-warning indicators and prudential metrics, running on the MAS-developed DKP. The page does not mention LLMs or NLP.
Relevance: a platform into which a structured obligation extractor could plug; we differ by addressing the unstructured rulebook side with small local models. Note the page is a web page, not a paper, and does not itself claim LLM use.

### 3. Thai NLP and tokenizer cost

**lowphansirikul2021wangchanberta** Lowphansirikul, L., Polpanumas, C., Jantrakulchai, N., Nutanong, S. (2021). WangchanBERTa: Pretraining transformer-based Thai Language Models. arXiv:2101.09635. https://arxiv.org/abs/2101.09635
A RoBERTa-style model pretrained on large Thai corpora with Thai-specific tokenization and evaluated on downstream Thai tasks.
Relevance: Thai encoder baseline and a reference for Thai word-segmentation issues; we use decoder-only Qwen3 instead.

**pipatanakul2023typhoon** Pipatanakul, K., Jirabovonvisut, P., Manakul, P., Sripaisarnmongkol, S., Patomwong, R., Chokchainant, P., Tharnpipitchai, K. (2023). Typhoon: Thai Large Language Models. arXiv:2312.13951 (technical report). https://arxiv.org/abs/2312.13951
Technical report on Thai-adapted LLMs obtained through continued Thai pretraining and instruction tuning.
Relevance: a Thai-specialised comparison point for our Qwen3 Thai results.

**ng2025sealion** Ng, R., Nguyen, T. N., Yuli, H., Chia, T. N., Yi, L. W., Leong, W. Q., Yong, X., Ngui, J. G., Susanto, Y., Cheng, N., et al. (30 authors in total). (2025). SEA-LION: Southeast Asian Languages in One Network. IJCNLP-AACL 2025, pp. 512-526. DOI 10.18653/v1/2025.ijcnlp-long.30. https://aclanthology.org/2025.ijcnlp-long.30/
Presents Llama-SEA-LION-v3-8B-IT and Gemma-SEA-LION-v3-9B-IT, covering 11 languages including Thai, built by multilingual continued pretraining plus multi-stage post-training.
Relevance: a regional 8B-class baseline; contrasts with our approach of task fine-tuning small general Qwen3 models.

**akarajaradwong2025nitibench** Akarajaradwong, P., Pothavorn, P., Chaksangchaichot, C., Tasawong, P., Nopparatbundit, T., Pratai, K., Nutanong, S. (2025). NitiBench: A Comprehensive Study of LLM Framework Capabilities for Thai Legal Question Answering. arXiv:2502.10868 (v4, Aug 2025; a search result indicates an EMNLP 2025 version under a revised title, which we did not open). https://arxiv.org/abs/2502.10868
Benchmark with NitiBench-CCL (Thai financial law) and NitiBench-Tax (tax rulings); finds section-based chunking helps retrieval and that long-context LLMs trail RAG.
Relevance: the main Thai financial-law resource and source of chunking guidance (section-based) for our Thai statutes; it is QA, not obligation extraction.

**petrov2023tokenizers** Petrov, A., La Malfa, E., Torr, P. H. S., Bibi, A. (2023). Language Model Tokenizers Introduce Unfairness Between Languages. NeurIPS 2023; arXiv:2305.15425. https://arxiv.org/abs/2305.15425
Shows that the same text tokenizes to very different lengths across languages, so users of some languages pay more and get less usable context. This is the tokenizer-premium argument.
Relevance: motivates measuring Thai token fertility for Qwen3 and its effect on context length and cost; we did not extract Thai-specific figures from this paper.

### Cluster gaps

We found no existing work that combines the following: (a) small (0.6B-8B) models run on premises for regulatory obligation extraction; (b) Thai financial statutes alongside English, EU, Basel, PRA/FCA, HKMA and MAS rulebooks in one study; (c) numeric perturbation (changing thresholds and deadlines) as a contamination control, since benchmarks drawn from public rulebooks may be memorised; (d) a full structured obligation schema (subject, modality, action, condition, threshold, deadline, exceptions, citation) with citation grounding, as opposed to sentence-level modality labels, passage QA or contract clauses; and (e) systematic scaling, distillation, verifiable-reward RL and quantization comparisons for this task. Searches also did not surface a BIS, MAS, ECB or FSB paper that evaluates small local LLMs on obligation extraction; the supervisory literature found here (FSI Brief 26) is a survey of use cases.

## Literature cluster B: distillation, verifiable-reward RL, label efficiency and fine-tuning scaling

All 13 entries were checked against their arXiv abstract pages (title, authors, year, venue as shown there). BibTeX is in `cluster_b.bib`.

### 1. Distillation and rationale distillation

**hsieh2023distilling** Hsieh, C.-Y., Li, C.-L., Yeh, C.-K., Nakhost, H., Fujii, Y., Ratner, A., Krishna, R., Lee, C.-Y., Pfister, T. (2023). Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes. Findings of ACL 2023. https://arxiv.org/abs/2305.02301
Extracts rationales from an LLM and uses them as additional supervision in a multi-task setup for small task-specific models. On four benchmarks the small models beat larger LLMs while needing less labelled data and fewer parameters.
Relevance to regslm: Directly motivates the rationale-distillation arm (teacher JSON plus short rationale) and its label-budget comparison at 500/2k/10k/50k.

**mitra2023orca2** Mitra, A., Del Corro, L., Mahajan, S., et al. (2023). Orca 2: Teaching Small Language Models How to Reason. arXiv:2311.11045 (preprint). https://arxiv.org/abs/2311.11045
Argues small models benefit from learning to choose a solution strategy per task (step-by-step or direct) rather than imitating a larger model's style. Orca 2 matches or beats models 5-10x larger on zero-shot reasoning.
Relevance to regslm: Suggests testing whether rationales help every unit type or whether direct JSON output suffices for simple obligations.

**zhou2024universalner** Zhou, W., Zhang, S., Gu, Y., Chen, M., Poon, H. (2024). UniversalNER: Targeted Distillation from Large Language Models for Open Named Entity Recognition. ICLR 2024. https://arxiv.org/abs/2308.03279
Distils ChatGPT into much smaller students for open NER via mission-focused instruction tuning, evaluated on 43 datasets across nine domains, where students exceed ChatGPT's NER accuracy.
Relevance to regslm: Closest teacher-to-student precedent for structured information extraction; a template for the targeted-distillation framing of obligation extraction.

### 2. RL with verifiable rewards

**shao2024deepseekmath** Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Bi, X., Zhang, H., Zhang, M., Li, Y. K., Wu, Y., Guo, D. (2024). DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. arXiv:2402.03300 (preprint). https://arxiv.org/abs/2402.03300
Introduces Group Relative Policy Optimization (GRPO), a PPO variant without a value model and with lower memory use, used to lift a 7B model's math reasoning.
Relevance to regslm: Source reference for the GRPO algorithm used in the RL arm; its low memory footprint matters for on-prem GPUs.

**deepseekai2025r1** DeepSeek-AI et al. (2025). DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. Nature 645, 633-638 (arXiv:2501.12948). https://arxiv.org/abs/2501.12948
Shows RL with rule-based verifiable rewards alone can elicit reasoning without human-labelled reasoning traces, and that the resulting reasoning transfers to smaller models.
Relevance to regslm: Justifies programmatic rewards (schema, citation, span, numeric) in place of a learned reward model, and frames the RL-versus-distillation question for small students.

**lambert2024tulu3** Lambert, N., Morrison, J., Pyatkin, V., Huang, S., Ivison, H., et al. (2024). Tulu 3: Pushing Frontiers in Open Language Model Post-Training. arXiv:2411.15124 (preprint). https://arxiv.org/abs/2411.15124
Open post-training recipe (SFT, preference tuning, and a new RL method with verifiable rewards, RLVR) with data, code and evaluation released. The arXiv page lists 22 or 23 authors inconsistently, so the bib entry uses "and others".
Relevance to regslm: Practical SFT-then-RLVR recipe and ablations to copy for the SFT-then-GRPO pipeline.

**agarwal2025thinkjson** Agarwal, B., Joshi, I., Rojkova, V. (2025). Think Inside the JSON: Reinforcement Strategy for Strict LLM Schema Adherence. arXiv:2502.14905 (preprint). https://arxiv.org/abs/2502.14905
ThinkJSON trains a 1.5B model for strict schema adherence using R1-style RL on synthetic reasoning data, custom GRPO reward functions and SFT, on modest compute.
Relevance to regslm: Nearest published analogue to schema-validity rewards with GRPO at small scale; note it is a preprint, so treat results with caution.

**helff2026gaming** Helff, L., Delfosse, Q., Steinmann, D., Harle, R., Shindo, H., Schramowski, P., Stammer, W., Kersting, K., Friedrich, F. (2026). LLMs Gaming Verifiers: RLVR can Lead to Reward Hacking. arXiv:2604.15149 (preprint). https://arxiv.org/abs/2604.15149
On inductive reasoning, RLVR-trained models enumerate instance-level answers that pass weak verifiers instead of learning rules. The authors propose Isomorphic Perturbation Testing to detect this.
Relevance to regslm: Warns that weak checks (e.g. a span-grounding reward that accepts any substring match) can be gamed, so reward audits and perturbation tests are needed.

### 3. Label efficiency, scaling and parameter-efficient fine-tuning

**zhou2023lima** Zhou, C., Liu, P., Xu, P., Iyer, S., Sun, J., Mao, Y., Ma, X., Efrat, A., Yu, P., Yu, L., Zhang, S., Ghosh, G., Lewis, M., Zettlemoyer, L., Levy, O. (2023). LIMA: Less Is More for Alignment. arXiv:2305.11206 (preprint). https://arxiv.org/abs/2305.11206
Fine-tuning a 65B model on only 1,000 curated examples yields strong instruction following, leading to the claim that most knowledge comes from pretraining.
Relevance to regslm: Motivates the 500 and 2k budgets and curation of labels over volume, though LIMA is 65B and open-ended, so transfer to 0.6B-8B extraction is untested.

**biderman2024lora** Biderman, D., Portes, J., Gonzalez Ortiz, J. J., Paul, M., Greengard, P., Jennings, C., King, D., Havens, S., Chiley, V., Frankle, J., Blakeney, C., Cunningham, J. P. (2024). LoRA Learns Less and Forgets Less. Transactions on Machine Learning Research (arXiv:2405.09673). https://arxiv.org/abs/2405.09673
LoRA usually underperforms full fine-tuning on code and math but forgets less of the base model's other abilities.
Relevance to regslm: Frames the LoRA vs full SFT comparison across student sizes and label budgets; expect the gap to widen at larger budgets.

**dettmers2023qlora** Dettmers, T., Pagnoni, A., Holtzman, A., Zettlemoyer, L. (2023). QLoRA: Efficient Finetuning of Quantized LLMs. arXiv:2305.14314 (preprint; extended NeurIPS 2023 submission per arXiv comments). https://arxiv.org/abs/2305.14314
Freezes a 4-bit quantized base model and trains LoRA adapters (NF4, double quantization, paged optimizers), tuning a 65B model on one 48GB GPU with little quality loss.
Relevance to regslm: Defines the QLoRA arm and the memory baseline for training the 8B student (and 14B teacher adaptation) on-prem.

**zhang2024scaling** Zhang, B., Liu, Z., Cherry, C., Firat, O. (2024). When Scaling Meets LLM Finetuning: The Effect of Data, Model and Finetuning Method. ICLR 2024. https://arxiv.org/abs/2402.17193
Fits power-based joint scaling laws for fine-tuning data size, model size and method (full vs prompt tuning vs LoRA), on translation and summarization; model scaling helps more than pretraining-data scaling, and method choice depends on task and data size.
Relevance to regslm: Template for fitting scaling curves over size (0.6B-8B) and label budget (500-50k), and a caution that the best method is task-dependent.

### Cluster gaps

No verified source covers GRPO/RLVR specifically for span-grounded information extraction or legal/regulatory obligation extraction; ThinkJSON is the only structured-output RL paper found and it is an unreviewed preprint. Little evidence exists on distilling rationales for extraction (as opposed to reasoning or classification) or on the crossover point between a fine-tuned sub-8B student and a 14B teacher at 500-50k labels; UniversalNER is the nearest. Quantization of the final student (post-training, e.g. GPTQ/AWQ) and the interaction of QLoRA with RL are not covered in this cluster. Symbolic knowledge distillation's NAACL 2022 venue was not shown on the arXiv page, so it is cited as a preprint. Orca (the original, 2306.02707) was not fetched, and a dedicated small-vs-large zero-shot crossover study was not located.

## Cluster C: quantization, contamination/memorisation, speculative decoding

All entries were verified by opening the arXiv abstract page (title, authors, dates, venue comments copied from the page). BibTeX is in `cluster_c.bib`.

### 1. Post-training quantization

### lin2024awq
Lin, J., Tang, J., Tang, H., Yang, S., Chen, W.-M., Wang, W.-C., Xiao, G., Dang, X., Gan, C., Han, S. (2024). *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration.* MLSys 2024 (Best Paper Award). https://arxiv.org/abs/2306.00978
AWQ protects the small fraction of salient weight channels, identified from activation statistics, by per-channel scaling before 4-bit weight-only quantization, and ships with an inference system for on-device deployment. It is one of the two 4-bit GPU formats we serve.
Relevance to regslm: defines the AWQ-4 arm of our serving matrix; its calibration-set dependence is a risk for rare numeric tokens in obligation text.

### frantar2023gptq
Frantar, E., Ashkboos, S., Hoefler, T., Alistarh, D. (2023). *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers.* ICLR 2023. https://arxiv.org/abs/2210.17323
GPTQ quantizes weights layer by layer using approximate second-order (Hessian) information, reaching 3-4 bit with small perplexity loss on very large models.
Relevance to regslm: the method behind our GPTQ-4 arm, and the canonical case of a perplexity-validated method whose task-level damage we want to measure on numeric fields.

### micikevicius2022fp8
Micikevicius, P., Stosic, D., Burgess, N., Cornea, M., Dubey, P., Grisenthwaite, R., Ha, S., Heinecke, A., Judd, P., Kamalu, J., Mellempudi, N., Oberman, S., Shoeybi, M., Siu, M., Wu, H. (2022). *FP8 Formats for Deep Learning.* arXiv preprint 2209.05433 (no venue listed). https://arxiv.org/abs/2209.05433
Specifies the two 8-bit floating-point encodings (E4M3 and E5M2) and shows they can match 16-bit quality for training and inference.
Relevance to regslm: the reference for the FP8 arm, needed to state precisely which format vLLM uses and why 8-bit float should be near-lossless.

### kurtic2025bf16
Kurtic, E., Marques, A., Pandit, S., Kurtz, M., Alistarh, D. (2025). *"Give Me BF16 or Give Me Death"? Accuracy-Performance Trade-Offs in LLM Quantization.* ACL 2025. https://arxiv.org/abs/2411.02355
A study of over 500,000 evaluations of FP8, INT8 and INT4 on Llama-3.1 reports that FP8 is effectively lossless, well-tuned INT8 loses about 1-3%, and INT4 weight-only is competitive. It also gives vLLM throughput and cost guidance by workload.
Relevance to regslm: the closest existing template for our bf16/FP8/AWQ/GPTQ comparison in vLLM, though it uses 8B-70B+ general benchmarks and not sub-8B extraction.

### li2025quantreasoning
Li, Z., Su, Y., Yang, R., Xie, C., Wang, Z., Xie, Z., Wong, N., Yang, H. (2025). *Quantization Meets Reasoning: Exploring LLM Low-Bit Quantization Degradation for Mathematical Reasoning.* arXiv preprint 2501.03035 (no venue listed). https://arxiv.org/abs/2501.03035
AWQ and GPTQ on Llama-3 models cause up to 32.39% (average 11.31%) accuracy loss on mathematical reasoning, concentrated in numerical computation and planning. A small fine-tune on 545 examples recovers most of the loss.
Relevance to regslm: direct evidence that numeric handling degrades first under 4-bit quantization, supporting our numeric-field breakdown and the option of post-quantization recovery fine-tuning.

### kurt2026whichquant
Kurt, U. (2026). *Which Quantization Should I Use? A Unified Evaluation of llama.cpp Quantization on Llama-3.1-8B-Instruct.* arXiv preprint 2601.14277 (single-author preprint, not peer reviewed). https://arxiv.org/abs/2601.14277
Compares 3-8 bit K-quant and legacy GGUF formats on one model across reasoning, knowledge, instruction-following and truthfulness benchmarks, plus perplexity and CPU prefill/decode throughput.
Relevance to regslm: the only systematic GGUF/CPU evaluation found; it is our baseline for Q4_K_M versus Q8_0, but it does not cover extraction or numeric fields and uses a single 8B model.

### 2. Contamination and memorisation

### carlini2022memorization
Carlini, N., Ippolito, D., Jagielski, M., Lee, K., Tramer, F., Zhang, C. (2022). *Quantifying Memorization Across Neural Language Models.* arXiv preprint 2202.07646 (the arXiv page lists no venue). https://arxiv.org/abs/2202.07646
Shows that verbatim memorisation grows log-linearly with model size, the number of duplicates of an example in training data, and prompt context length.
Relevance to regslm: justifies expecting public rulebook text (heavily duplicated) to be memorised, and gives the framing for our measurement of how often models emit the original value instead of the perturbed one, including whether it changes with model size.

### shi2023mink
Shi, W., Ajith, A., Xia, M., Huang, Y., Liu, D., Blevins, T., Chen, D., Zettlemoyer, L. (2023). *Detecting Pretraining Data from Large Language Models.* arXiv preprint 2310.16789 (no venue listed on the page). https://arxiv.org/abs/2310.16789
Introduces the WikiMIA benchmark, which uses text from before and after a model's training period as ground truth, and Min-K% Prob, a black-box membership test that needs no reference corpus.
Relevance to regslm: a complementary, probability-based contamination check on the public rulebooks, and a precedent for the pre/post-cutoff split.

### mirzadeh2025gsmsymbolic
Mirzadeh, I., Alizadeh, K., Shahrokhi, H., Tuzel, O., Bengio, S., Farajtabar, M. (2025). *GSM-Symbolic: Understanding the Limitations of Mathematical Reasoning in Large Language Models.* ICLR 2025 (arXiv 2410.05229). https://arxiv.org/abs/2410.05229
Generates GSM8K variants from symbolic templates and finds that accuracy varies, and typically drops, when only the numerical values change, which the authors read as pattern replication from training data.
Relevance to regslm: the methodological precedent for our threshold/date perturbation, which we apply to extraction and score as a memorised-value rate.

### white2025livebench
White, C., Dooley, S., Roberts, M., Pal, A., Feuer, B., Jain, S., Shwartz-Ziv, R., Jain, N., Saifullah, K., Dey, S., Shubh-Agrawal, Sandha, S. S., Naidu, S., Hegde, C., LeCun, Y., Goldstein, T., Neiswanger, W., Goldblum, M. (2025). *LiveBench: A Challenging, Contamination-Limited LLM Benchmark.* ICLR 2025 (Spotlight; arXiv 2406.19314). https://arxiv.org/abs/2406.19314
A benchmark built from recent sources and updated monthly, with automatic scoring against objective ground truth, so that later models cannot have seen the questions.
Relevance to regslm: supports the post-cutoff slice as an accepted contamination control; we must pick rulebook amendments dated after the Qwen3 data cutoff.

### 3. Speculative decoding

### leviathan2023speculative
Leviathan, Y., Kalman, M., Matias, Y. (2023). *Fast Inference from Transformers via Speculative Decoding.* ICML 2023 (Oral; arXiv 2211.17192). https://arxiv.org/abs/2211.17192
A small draft model proposes tokens that the large model verifies in parallel with a rejection-sampling scheme that preserves the target output distribution exactly, giving 2-3x speedup on T5-XXL without retraining.
Relevance to regslm: since output distribution is unchanged, a small Qwen3 draft for the 8B extractor is a lossless throughput option for on-prem serving; extraction's copy-heavy JSON output suggests high acceptance rates, to be tested.

### Cluster gaps

- No source found measuring quantization effects on sub-8B models (our 0.6B-4B range) or on structured-extraction/JSON output; the sources above use 7B-70B+ general or math benchmarks.
- Quantization-induced change in memorisation (does 4-bit increase or decrease regurgitating the original value?) was not found and may be novel to this project.
- Regulatory or legal-domain contamination studies, and GGUF evaluations beyond the single preprint above, were not covered. Speculative decoding on CPU-only llama.cpp, and the interaction of drafts with quantized targets, was not searched.
- Dropped for the 11-source cap, all verified: Li et al., *Evaluating Quantized Large Language Models* (arXiv 2402.18158, broad PTQ evaluation incl. long-context); Marchisio et al., *How Does Quantization Affect Multilingual LLMs?* (Findings of EMNLP 2024, arXiv 2407.03211, automatic metrics understate harm, hard tasks degrade fastest). Worth adding if a Thai or multilingual angle is needed.

## Gap table

| Question regslm asks | Closest prior work | What is missing there |
|---|---|---|
| Structured obligation extraction (subject, modality, condition, threshold, deadline, exceptions, citation) from financial rulebooks | Ó Neill et al. 2017 (sentence modality); RIRAG/ObliQA (passage QA); CUAD (contract clauses) | Field-level structured output with resolvable citations; multi-jurisdiction rulebooks; numeric fields scored exactly |
| Smallest on-prem model that does the job | LIMA; Zhang et al. 2024 (fine-tuning scaling); UniversalNER (targeted distillation) | Size × label-budget crossover against a local teacher and an 8B zero-shot baseline on one consumer GPU, with CIs |
| Rationale distillation for extraction | Distilling step-by-step; Orca 2 | Evidence on *structured IE* fields (conditions, exceptions) and whether gains shrink with size |
| GRPO with verifiable rewards for extraction | DeepSeekMath/R1; Tülu 3 RLVR; Think-inside-the-JSON | Rewards for grounding/citation/numeric exactness; explicit reward-hacking audit (empty-output, verifier gaming per Helff et al.) |
| Quantization effects on task fields | AWQ, GPTQ, FP8; Kurtic et al. 2025; Li et al. 2025 (reasoning) | Effects on sub-8B *fine-tuned* extractors and specifically on numeric fields; GGUF CPU-only for the same checkpoints |
| Contamination of public legal text | Carlini et al. (memorisation); Min-K%; GSM-Symbolic (numeric perturbation); LiveBench | Applying number-perturbation to regulatory text and reporting a per-model memorisation rate alongside every score |
| Thai regulatory extraction | WangchanBERTa; Typhoon; SEA-LION; NitiBench; Petrov et al. (tokenizer unfairness) | Same-schema Thai vs English comparison, tokenizer cost per obligation, Thai-pretrained small bases vs Qwen3 |
| SupTech deployment constraints | BIS FSI Brief 26 (gen-AI in supervision); Project Ellipse | Open, reproducible, fully offline pipeline with released weights and hardware budgets |

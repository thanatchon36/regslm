# Literature cluster B: distillation, verifiable-reward RL, label efficiency and fine-tuning scaling

All 13 entries were checked against their arXiv abstract pages (title, authors, year, venue as shown there). BibTeX is in `cluster_b.bib`.

## 1. Distillation and rationale distillation

**hsieh2023distilling** Hsieh, C.-Y., Li, C.-L., Yeh, C.-K., Nakhost, H., Fujii, Y., Ratner, A., Krishna, R., Lee, C.-Y., Pfister, T. (2023). Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes. Findings of ACL 2023. https://arxiv.org/abs/2305.02301
Extracts rationales from an LLM and uses them as additional supervision in a multi-task setup for small task-specific models. On four benchmarks the small models beat larger LLMs while needing less labelled data and fewer parameters.
Relevance to regslm: Directly motivates the rationale-distillation arm (teacher JSON plus short rationale) and its label-budget comparison at 500/2k/10k/50k.

**mitra2023orca2** Mitra, A., Del Corro, L., Mahajan, S., et al. (2023). Orca 2: Teaching Small Language Models How to Reason. arXiv:2311.11045 (preprint). https://arxiv.org/abs/2311.11045
Argues small models benefit from learning to choose a solution strategy per task (step-by-step or direct) rather than imitating a larger model's style. Orca 2 matches or beats models 5-10x larger on zero-shot reasoning.
Relevance to regslm: Suggests testing whether rationales help every unit type or whether direct JSON output suffices for simple obligations.

**west2021symbolic** West, P., Bhagavatula, C., Hessel, J., Hwang, J. D., Jiang, L., Le Bras, R., Lu, X., Welleck, S., Choi, Y. (2021, rev. 2022). Symbolic Knowledge Distillation: from General Language Models to Commonsense Models. arXiv:2110.07178 (arXiv page lists no venue). https://arxiv.org/abs/2110.07178
A general LM generates a symbolic commonsense knowledge graph as text, filtered by a trained critic, and a student trained on it surpasses the teacher despite being about 100x smaller.
Relevance to regslm: Supports filtering teacher labels with a critic or programmatic checks before student training, and the idea that a filtered student can exceed its raw teacher.

**zhou2024universalner** Zhou, W., Zhang, S., Gu, Y., Chen, M., Poon, H. (2024). UniversalNER: Targeted Distillation from Large Language Models for Open Named Entity Recognition. ICLR 2024. https://arxiv.org/abs/2308.03279
Distils ChatGPT into much smaller students for open NER via mission-focused instruction tuning, evaluated on 43 datasets across nine domains, where students exceed ChatGPT's NER accuracy.
Relevance to regslm: Closest teacher-to-student precedent for structured information extraction; a template for the targeted-distillation framing of obligation extraction.

## 2. RL with verifiable rewards

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

## 3. Label efficiency, scaling and parameter-efficient fine-tuning

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

## Gaps

No verified source covers GRPO/RLVR specifically for span-grounded information extraction or legal/regulatory obligation extraction; ThinkJSON is the only structured-output RL paper found and it is an unreviewed preprint. Little evidence exists on distilling rationales for extraction (as opposed to reasoning or classification) or on the crossover point between a fine-tuned sub-8B student and a 14B teacher at 500-50k labels; UniversalNER is the nearest. Quantization of the final student (post-training, e.g. GPTQ/AWQ) and the interaction of QLoRA with RL are not covered in this cluster. Symbolic knowledge distillation's NAACL 2022 venue was not shown on the arXiv page, so it is cited as a preprint. Orca (the original, 2306.02707) was not fetched, and a dedicated small-vs-large zero-shot crossover study was not located.

# Cluster C: quantization, contamination/memorisation, speculative decoding

All entries were verified by opening the arXiv abstract page (title, authors, dates, venue comments copied from the page). BibTeX is in `cluster_c.bib`.

## 1. Post-training quantization

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

## 2. Contamination and memorisation

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

## 3. Speculative decoding

### leviathan2023speculative
Leviathan, Y., Kalman, M., Matias, Y. (2023). *Fast Inference from Transformers via Speculative Decoding.* ICML 2023 (Oral; arXiv 2211.17192). https://arxiv.org/abs/2211.17192
A small draft model proposes tokens that the large model verifies in parallel with a rejection-sampling scheme that preserves the target output distribution exactly, giving 2-3x speedup on T5-XXL without retraining.
Relevance to regslm: since output distribution is unchanged, a small Qwen3 draft for the 8B extractor is a lossless throughput option for on-prem serving; extraction's copy-heavy JSON output suggests high acceptance rates, to be tested.

## Gaps

- No source found measuring quantization effects on sub-8B models (our 0.6B-4B range) or on structured-extraction/JSON output; the sources above use 7B-70B+ general or math benchmarks.
- Quantization-induced change in memorisation (does 4-bit increase or decrease regurgitating the original value?) was not found and may be novel to this project.
- Regulatory or legal-domain contamination studies, and GGUF evaluations beyond the single preprint above, were not covered. Speculative decoding on CPU-only llama.cpp, and the interaction of drafts with quantized targets, was not searched.
- Dropped for the 11-source cap, all verified: Li et al., *Evaluating Quantized Large Language Models* (arXiv 2402.18158, broad PTQ evaluation incl. long-context); Marchisio et al., *How Does Quantization Affect Multilingual LLMs?* (Findings of EMNLP 2024, arXiv 2407.03211, automatic metrics understate harm, hard tasks degrade fastest). Worth adding if a Thai or multilingual angle is needed.

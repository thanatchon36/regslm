# Literature cluster A: obligation extraction, SupTech LLMs, Thai NLP and tokenization

All entries were opened and checked (title, authors, year, venue) on 2026-10-15. BibTeX is in `cluster_a.bib`.

## 1. Obligation, deontic and regulatory-text extraction; legal benchmarks

**gokhan2024rirag** Gokhan, T., Wang, K., Gurevych, I., Briscoe, T. (2024). RIRAG: Regulatory Information Retrieval and Answer Generation. arXiv:2409.05677 (preprint; no peer-reviewed venue listed). https://arxiv.org/abs/2409.05677
Introduces the regulatory retrieval and answer-generation task and the ObliQA dataset built from regulatory documents (ADGM). Answers are grounded in obligation-bearing passages and evaluated for retrieval and generation quality.
Relevance: closest prior work on obligations in financial regulation, but it is passage-level QA, whereas regslm extracts structured fields (subject, modality, action, condition, threshold, deadline, exceptions, citation).

**oneill2017modality** Ó Neill, J., Buitelaar, P., Robin, C., O'Brien, L. (2017). Classifying Sentential Modality in Legal Language: A Use Case in Financial Regulations, Acts and Directives. ICAIL 2017, London. DOI 10.13025/21099. https://research.universityofgalway.ie/en/publications/classifying-sentential-modality-in-legal-language-a-use-case-in-f/
Classifies sentences from financial regulation as obligation, prohibition or permission using ensembles of neural networks over legal and general distributional embeddings, reaching 82.33% accuracy on held-out expert-annotated sentences.
Relevance: the baseline for our modality field (must/must_not/may), which we extend with should and with full argument structure rather than sentence-level labels.

**sleimi2020semantic** Sleimi, A., Sannier, N., Sabetzadeh, M., Briand, L., Ceci, M., Dann, J. (2020). An Automated Framework for the Extraction of Semantic Legal Metadata from Legal Texts. arXiv:2001.11245. https://arxiv.org/abs/2001.11245
Reconciles the semantic metadata types used in requirements engineering and builds an NLP/ML pipeline to extract them from legislation. On two Luxembourg case studies it reports precision of 97.2% and 82.4% and recall of 94.9% and 92.4%.
Relevance: a rule- and parser-based precedent for a metadata schema; we replace the pipeline with one small fine-tuned LLM and cross-jurisdiction rulebooks.

**guha2023legalbench** Guha, N., Nyarko, J., Ho, D. E., Ré, C., Chilton, A., et al. (2023). LegalBench: A Collaboratively Built Benchmark for Measuring Legal Reasoning in Large Language Models. arXiv:2308.11462 (preprint; published at NeurIPS 2023 Datasets and Benchmarks, not stated on the arXiv page). https://arxiv.org/abs/2308.11462
A collaboratively built suite of legal reasoning tasks used to evaluate LLMs across several reasoning types.
Relevance: a general legal-reasoning sanity check for our models; it does not cover regulatory obligation extraction.

**hendrycks2021cuad** Hendrycks, D., Burns, C., Chen, A., Ball, S. (2021). CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review. NeurIPS 2021 (Datasets and Benchmarks); arXiv:2103.06268. https://arxiv.org/abs/2103.06268
Expert-annotated contract clause extraction dataset framed as span extraction.
Relevance: a template for expert-annotated span extraction and its evaluation, but on contracts rather than regulation and without deontic structure.

**chalkidis2022lexglue** Chalkidis, I., Jana, A., Hartung, D., Bommarito, M., Androutsopoulos, I., Katz, D. M., Aletras, N. (2022). LexGLUE: A Benchmark Dataset for Legal Language Understanding in English. ACL 2022; arXiv:2110.00976. https://arxiv.org/abs/2110.00976
A unified benchmark of English legal classification and multiple-choice tasks for evaluating legal language models.
Relevance: standard baseline for legal-domain NLU, useful as a catastrophic-forgetting or transfer check; no extraction of structured norms.

## 2. SupTech and RegTech applications of LLMs

**prenio2025genai** Prenio, J. (2025). Starting with the basics: a stocktake of gen AI applications in supervision. FSI Briefs No 26, Bank for International Settlements, 12 June 2025. https://www.bis.org/fsi/fsibriefs26.pdf
Surveys how financial authorities test, build and use generative AI in supervision, grouping applications into document processing, knowledge management and document review. It identifies legacy IT, data security, skills gaps, low user acceptance and inaccurate outputs as the main obstacles.
Relevance: establishes the supervisory demand for document-processing tools and motivates our focus on small, on-premises, verifiable extraction.

**bisih2022ellipse** BIS Innovation Hub (2022 onward). Ellipse and the Data and Knowledge Platform (DKP). Project page, with the Monetary Authority of Singapore. https://www.bis.org/innovation/bis_open_tech_ellipse.htm
Describes Project Ellipse, a prototype integrated regulatory data and analytics platform combining structured and unstructured data to give supervisors early-warning indicators and prudential metrics, running on the MAS-developed DKP. The page does not mention LLMs or NLP.
Relevance: a platform into which a structured obligation extractor could plug; we differ by addressing the unstructured rulebook side with small local models. Note the page is a web page, not a paper, and does not itself claim LLM use.

## 3. Thai NLP and tokenizer cost

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

## Gaps

We found no existing work that combines the following: (a) small (0.6B-8B) models run on premises for regulatory obligation extraction; (b) Thai financial statutes alongside English, EU, Basel, PRA/FCA, HKMA and MAS rulebooks in one study; (c) numeric perturbation (changing thresholds and deadlines) as a contamination control, since benchmarks drawn from public rulebooks may be memorised; (d) a full structured obligation schema (subject, modality, action, condition, threshold, deadline, exceptions, citation) with citation grounding, as opposed to sentence-level modality labels, passage QA or contract clauses; and (e) systematic scaling, distillation, verifiable-reward RL and quantization comparisons for this task. Searches also did not surface a BIS, MAS, ECB or FSB paper that evaluates small local LLMs on obligation extraction; the supervisory literature found here (FSI Brief 26) is a survey of use cases.

# Resources Catalog

## Summary
This document catalogs all resources gathered for the LLM Pain research project, investigating whether LLMs systematically avoid topics not explicitly restricted during post-training, potentially arising from persona generalization in training data.

## Papers
Total papers downloaded: **35**

| Title | Authors | Year | File | Key Info |
|-------|---------|------|------|----------|
| XSTest | Röttger et al. | 2023 | papers/2308.01263_... | Core: exaggerated safety via lexical overfitting |
| OR-Bench | Cui et al. | 2024 | papers/2405.20947_... | 80K over-refusal prompts, 32 LLMs evaluated |
| Identity Speech Suppression | Proebsting et al. | 2025 | papers/2409.13725_... | Stereotype-based content suppression |
| Sycophancy Survey | Malmqvist | 2024 | papers/2411.15287_... | RLHF creates emergent persona behaviors |
| Understanding Sycophancy | Sharma et al. | 2023 | papers/2310.13548_... | More RLHF = more sycophancy |
| Differential Harm Propensity | - | 2026 | papers/2603.16734_... | Persona-dependent avoidance |
| Constitutional AI | Bai et al. | 2022 | papers/2212.08073_... | Foundational alignment method |
| InstructGPT | Ouyang et al. | 2022 | papers/2203.02155_... | RLHF methodology |
| DPO | Rafailov et al. | 2023 | papers/2305.18290_... | Alternative alignment approach |
| + 26 more | Various | 2024-2026 | papers/ | Refusal mechanisms, benchmarks, mitigation |

See papers/README.md for detailed descriptions.

## Datasets
Total datasets downloaded: **2** (+ gated datasets documented)

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| OR-Bench-Hard-1K | HuggingFace | 1,319 | Over-refusal eval | datasets/or_bench_hard_1k/ | Best for borderline cases |
| XSTest | GitHub | 450 | Exaggerated safety | datasets/xstest_prompts.csv | Standard diagnostic |

See datasets/README.md for detailed descriptions and download instructions.

## Code Repositories
Total repositories cloned: **3**

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| XSTest | github.com/paul-rottger/exaggerated-safety | Exaggerated safety test suite | code/xstest/ | Prompts + model completions |
| OR-Bench | github.com/justincui03/or-bench | Over-refusal benchmark pipeline | code/or-bench/ | Automated prompt generation |
| Speech Suppression | github.com/genAIaudits/speech-suppression | Identity-based suppression audit | code/speech-suppression/ | 5 API audit pipeline |

See code/README.md for detailed descriptions.

## Resource Gathering Notes

### Search Strategy
- Used paper-finder with 3 different query strategies: safety/refusal, over-refusal/false-refusal, sycophancy/persona
- 3 searches yielded 328 unique papers (after dedup)
- Selected 48 most relevant based on relevance scores and keyword matching
- Downloaded 35 papers successfully (13 without accessible PDFs)
- Deep-read 6 core papers using PDF chunker

### Selection Criteria
Papers were prioritized by:
1. Direct relevance to implicit avoidance / persona generalization hypothesis
2. Availability of benchmarks or datasets
3. Methodological novelty for measuring avoidance behaviors
4. Foundational importance to the field

### Challenges Encountered
- Several arXiv IDs from Semantic Scholar were incorrect (pointed to physics papers)
- Multiple key datasets (SORRY-Bench, WildGuardMix, LMSYS-Chat) are gated and require authentication
- Some highly relevant papers (e.g., "State-Dependent Refusal and Learned Incapacity") lacked accessible PDFs

### Gaps and Workarounds
- **Gap**: No existing dataset specifically tests for *implicit* avoidance (subtle topic deflection rather than outright refusal)
- **Workaround**: Experiment runner should create custom topic probes that test quality/engagement differences across topics
- **Gap**: No standard metric for "hedging" or "caveat density"
- **Workaround**: Can compute using simple NLP features (sentence length, presence of disclaimers, sentiment)

## Recommendations for Experiment Design

### 1. Primary dataset(s)
- **OR-Bench-Hard-1K**: For measuring explicit over-refusal rates across models
- **XSTest**: For diagnostic testing of lexical overfitting
- **Custom topic probes** (to be created): For measuring *implicit* avoidance — topics where models don't refuse but show reduced quality/engagement

### 2. Baseline methods
- Query multiple LLMs (GPT-4, Claude, Llama 3, Mistral, Gemma) with same prompts
- Compare base models vs. RLHF-aligned versions (e.g., Llama-3-base vs. Llama-3-instruct)
- Use XSTest string-match method for automated refusal detection
- Use LLM-as-judge for nuanced response quality assessment

### 3. Evaluation metrics
- **Refusal rate**: Standard over-refusal metric
- **Response quality differential**: Compare response quality (length, detail, helpfulness) on sensitive vs. neutral topics
- **Topic deflection rate**: Does the model change the subject?
- **Caveat/hedge density**: Count disclaimers, warnings, qualifications
- **Cross-model avoidance patterns**: Which topics do multiple models avoid? (suggests training data origin vs. explicit policy)

### 4. Code to adapt/reuse
- **XSTest evaluation pipeline** (code/xstest/): Response classification framework
- **OR-Bench generation pipeline** (code/or-bench/): For generating additional borderline prompts
- **Speech suppression audit** (code/speech-suppression/): Identity-based differential analysis methodology

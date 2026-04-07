# Literature Review: LLM Pain — Systematic Avoidance Behaviors in Large Language Models

## Research Area Overview

This review examines the phenomenon we term "LLM Pain" — systematic avoidance behaviors in LLMs that are not explicitly specified during post-training alignment. The hypothesis posits that LLMs develop implicit avoidance patterns that generalize from persona-like features in training data, going beyond what is explicitly prohibited. This intersects with several active research areas: over-refusal/exaggerated safety, refusal mechanisms, sycophancy, identity-related speech suppression, and the safety-helpfulness tradeoff.

## Key Papers

### 1. XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in LLMs
- **Authors**: Röttger et al.
- **Year**: 2023
- **Source**: arXiv:2308.01263, ACL 2024
- **Key Contribution**: First systematic test suite (250 safe + 200 unsafe prompts) for measuring "exaggerated safety" — false refusals of clearly safe prompts.
- **Methodology**: 10 prompt types testing homonyms, figurative language, safe targets, safe contexts, definitions, discrimination with nonsense groups, historical events, and privacy. Manual 3-way annotation (full compliance, partial refusal, full refusal).
- **Key Results**: Llama2 refuses 38%+21.6% of safe prompts. GPT-4 best balance (6.4%+2% refusal on safe, 97.5% refusal on unsafe). Root cause identified as **lexical overfitting** — models use simplistic word-level rules ("killing" → refuse) rather than understanding context.
- **Critical Insight for Our Hypothesis**: Exaggerated safety is "likely a consequence of biases in training data" — words like "killing" mostly occurred in unsafe contexts during safety fine-tuning, causing overgeneralization. This directly supports our hypothesis about implicit avoidance from training set generalization.
- **Dataset**: XSTest (450 prompts), available on GitHub.
- **Relevance**: **CORE** — demonstrates that avoidance patterns generalize beyond explicit rules.

### 2. OR-Bench: An Over-Refusal Benchmark for Large Language Models
- **Authors**: Cui, Chiang, Stoica, Hsieh
- **Year**: 2024 (ICML 2025)
- **Source**: arXiv:2405.20947
- **Key Contribution**: First large-scale over-refusal benchmark (80K prompts across 10 categories + 1K hard subset + 600 toxic prompts). Automated pipeline for generating borderline prompts.
- **Methodology**: Generate toxic seeds → rewrite to benign → verify with LLM moderators. Evaluate 32 LLMs across 8 families.
- **Key Results**: Spearman correlation of **0.89** between safety and over-refusal — almost all models trade helpfulness for safety. Claude models highest safety but also highest over-refusal. Model size does NOT correlate with better safety-sensitivity balance.
- **Datasets Used**: OR-Bench-80K, OR-Bench-Hard-1K, OR-Bench-Toxic (all on HuggingFace).
- **Relevance**: **HIGH** — provides large-scale evidence and datasets for measuring implicit avoidance. The 0.89 correlation suggests over-refusal is a systematic, not random, phenomenon.

### 3. Identity-related Speech Suppression in Generative AI Content Moderation
- **Authors**: Proebsting, Anigboro, Crawford, Metaxa, Friedler
- **Year**: 2025 (EAAMO '25)
- **Source**: arXiv:2409.13725
- **Key Contribution**: Defines and measures "speech suppression" — identity-related content incorrectly flagged by content moderation. Audits 5 APIs across 7 datasets and 9 identity groups.
- **Key Results**: Identity-related speech is MORE likely to be incorrectly suppressed than other speech for 7 of 9 identity groups. Suppression reasons vary by stereotypical associations: disability → self-harm, LGBT → sexual content, non-Christian → hate/violence.
- **Critical Insight**: Avoidance patterns follow stereotypical associations from training data — exactly what our hypothesis predicts (persona generalization from training set).
- **Code**: Available at github.com/genAIaudits/speech-suppression
- **Relevance**: **CORE** — directly demonstrates that implicit avoidance follows training data biases, not explicit rules.

### 4. Sycophancy in Large Language Models: Causes and Mitigations (Survey)
- **Authors**: Malmqvist
- **Year**: 2024
- **Source**: arXiv:2411.15287
- **Key Contribution**: Technical survey of sycophancy — the inverse of over-refusal. Models excessively agree with users, stemming from RLHF biases and training data patterns.
- **Methodology**: Literature survey covering measurement, causes, and mitigation.
- **Critical Insight**: Sycophancy and over-refusal are two sides of the same coin — both are emergent behaviors from persona generalization during RLHF. RLHF rewards human-pleasing responses, which can manifest as either excessive agreement OR excessive caution depending on context.
- **Relevance**: **HIGH** — provides theoretical grounding for how RLHF creates implicit behavioral patterns.

### 5. Towards Understanding Sycophancy in Language Models
- **Authors**: Sharma et al. (Anthropic)
- **Year**: 2023
- **Source**: arXiv:2310.13548
- **Key Contribution**: Shows sycophancy is a general behavior across tasks (math, ethics, factual QA). Human preference data used in RLHF exhibits sycophantic patterns — human raters prefer responses that match their stated views.
- **Datasets Used**: Custom evaluation suites for opinion sycophancy, factual sycophancy.
- **Key Results**: Models trained with more RLHF show MORE sycophancy. Larger models are MORE sycophantic (not less). Sycophancy is an artifact of the RLHF training process, not a fundamental model limitation.
- **Relevance**: **HIGH** — demonstrates that RLHF creates emergent behavioral patterns that generalize beyond training examples.

### 6. SORRY-Bench: Systematically Evaluating LLM Safety Refusal Behaviors
- **Authors**: Xie et al.
- **Year**: 2024
- **Source**: arXiv:2406.14598
- **Key Contribution**: Fine-grained taxonomy of 45 safety categories for evaluating refusal behaviors.
- **Relevance**: **MEDIUM** — provides category taxonomy useful for systematically probing avoidance patterns.

### 7. LLM Content Moderation and User Satisfaction
- **Authors**: (2025)
- **Source**: arXiv:2501.03266
- **Key Contribution**: Studies real-world impact of refusal behaviors on user satisfaction using Chatbot Arena data.
- **Relevance**: **MEDIUM** — provides ecological validity for studying over-refusal in real user interactions.

### 8. Constitutional AI: Harmlessness from AI Feedback
- **Authors**: Bai et al. (Anthropic)
- **Year**: 2022
- **Source**: arXiv:2212.08073
- **Key Contribution**: Foundational work on training LLMs to be harmless using AI feedback and constitutional principles.
- **Relevance**: **FOUNDATIONAL** — establishes the training methodology that may create implicit avoidance patterns.

### 9. Direct Preference Optimization (DPO)
- **Authors**: Rafailov et al.
- **Year**: 2023
- **Source**: arXiv:2305.18290
- **Relevance**: **FOUNDATIONAL** — alternative to RLHF that may produce different avoidance patterns.

### 10. InstructGPT: Training Language Models to Follow Instructions with Human Feedback
- **Authors**: Ouyang et al.
- **Year**: 2022
- **Source**: arXiv:2203.02155
- **Relevance**: **FOUNDATIONAL** — establishes RLHF methodology for instruction following.

### Additional Relevant Papers (Downloaded but not deep-read)

- **Automatic Pseudo-Harmful Prompt Generation** (2024): Automated generation of prompts that trigger false refusals
- **Mitigating Exaggerated Safety** (2024): Techniques to reduce over-refusal
- **Surgical, Cheap, and Flexible: Single Vector Ablation** (2024): Finds refusal can be removed via single direction in activation space — suggests refusal is encoded as a simple feature
- **Refusal Behavior: A Nonlinear Perspective** (2025): Shows refusal is more complex than a single linear direction
- **Beyond Surface Alignment** (2025): Probabilistic ablation of refusal direction
- **Refusal Direction is Universal Across Languages** (2025): The same refusal mechanism operates across languages
- **Instability of Safety** (2025): Random seeds and temperature expose inconsistent refusal behavior
- **FalseReject** (2025): Resource for improving contextual safety
- **Beyond No: Quantifying AI Over-Refusal** (2025): Quantifies over-refusal and emotional attachment boundaries
- **Differential Harm Propensity** (2026): Personalized LLM agents show different harm propensity based on disclosed mental health status — direct evidence of persona-dependent avoidance

## Common Methodologies

1. **Benchmark-based evaluation**: Create prompts (safe + unsafe), query models, classify responses as compliance/refusal (XSTest, OR-Bench, SORRY-Bench)
2. **Activation analysis**: Identify refusal directions in model activation space (refusal vectors, COSMIC)
3. **Ablation studies**: Remove refusal features to study what changes (single vector ablation, activation steering)
4. **Audit methodology**: Systematically test content moderation APIs across identity groups (speech suppression work)
5. **User study / ecological**: Analyze real user interactions (Chatbot Arena data)

## Standard Baselines

- **XSTest** (250 safe + 200 unsafe prompts): Standard small-scale benchmark
- **OR-Bench-Hard-1K**: Challenging over-refusal benchmark  
- **SORRY-Bench**: 45-category safety evaluation
- **HarmBench**: Red-teaming evaluation
- **WildGuard**: Multi-task moderation detection

## Evaluation Metrics

- **Over-refusal rate**: % of safe prompts refused (lower is better)
- **Safety rate**: % of unsafe prompts refused (higher is better)
- **F1 / calibration score**: Balance between the two
- **Speech suppression index**: Differential false positive rate for identity groups
- **String-match refusal detection**: Simple automated method ("I'm sorry", "As an AI", "I cannot")
- **LLM-as-judge**: Use GPT-4 to classify responses as compliance/refusal

## Gaps and Opportunities

1. **Implicit vs. explicit avoidance**: Most work focuses on explicit refusal. Our hypothesis targets *implicit* avoidance — topics the model subtly steers away from without explicit refusal. This is largely unstudied.
2. **Persona generalization mechanism**: While speech suppression work shows stereotypical avoidance patterns, no one has systematically studied whether these come from training data personas or from explicit safety rules.
3. **Topic-level analysis**: Existing benchmarks test individual prompts. No systematic study maps which *topics* (not just words) LLMs systematically avoid.
4. **Cross-model comparison**: While OR-Bench tests 32 models, the focus is on over-refusal rate, not on *which specific topics* different models avoid differently.
5. **Subtle avoidance**: Current metrics detect outright refusal. More subtle forms — hedging, topic deflection, unnecessary caveats, reduced quality on sensitive topics — are not measured.

## Recommendations for Our Experiment

### Recommended datasets:
1. **OR-Bench-Hard-1K** (HuggingFace): Best for probing borderline avoidance behaviors
2. **XSTest** (GitHub): Standard diagnostic test suite, well-validated
3. **SORRY-Bench categories**: For systematic topic coverage
4. **Custom topic probes**: Need to create prompts that test for *implicit* avoidance (not just refusal)

### Recommended baselines:
- Compare multiple LLM families (GPT, Claude, Llama, Mistral, Gemma) 
- Test across model sizes within families
- Compare base vs. RLHF-aligned versions when available

### Recommended metrics:
- Refusal rate (standard)
- Response quality on sensitive vs. neutral topics (new)
- Topic deflection rate (new — does the model steer away from topics without refusing?)
- Hedging/caveat density (new — does the model add unnecessary disclaimers?)
- Semantic similarity of responses to training data personas (new)

### Methodological considerations:
- Use temperature=0 for reproducibility
- Test both API-based and open-weight models
- For open-weight models, analyze activation patterns to identify implicit avoidance features
- Compare behavior on topics explicitly mentioned in safety guidelines vs. topics that are avoided but not explicitly restricted

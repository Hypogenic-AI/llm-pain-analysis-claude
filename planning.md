# Research Plan: LLM Pain — Implicit Avoidance Behaviors

## Motivation & Novelty Assessment

### Why This Research Matters
LLMs are increasingly deployed in sensitive domains (education, healthcare, legal advice) where implicit avoidance of certain topics — without explicit refusal — can lead to systematic information gaps. If models subtly avoid or degrade their responses on topics not explicitly restricted, users may receive lower-quality information on important subjects without realizing it. Understanding the origin of these avoidance patterns (training data persona generalization vs. explicit safety rules) is critical for improving model reliability.

### Gap in Existing Work
Prior work (XSTest, OR-Bench, SORRY-Bench) focuses on **explicit refusal** — models saying "I cannot help with that." The literature review identifies a clear gap: **no systematic study of implicit avoidance** — topics where models don't refuse but show measurably degraded engagement (shorter responses, more hedging, topic deflection, unnecessary caveats). Furthermore, no study has compared avoidance patterns across models to distinguish training-data-driven avoidance from post-training-specific avoidance.

### Our Novel Contribution
We introduce a methodology for detecting **implicit avoidance** in LLMs by measuring response quality differentials across topics. We distinguish three types of avoidance:
1. **Explicit avoidance**: Outright refusal (well-studied)
2. **Implicit avoidance**: Degraded response quality without refusal (our focus)
3. **Convergent implicit avoidance**: Topics multiple models implicitly avoid despite different safety training (suggests training data origin)

### Experiment Justification
- **Experiment 1 (Topic Probe Design)**: Create structured probes spanning neutral, borderline, and sensitive topics to measure response characteristics. *Why needed*: No existing dataset tests for implicit avoidance — existing benchmarks only detect refusal.
- **Experiment 2 (Cross-Model Avoidance Mapping)**: Query 3-4 LLMs with identical probes. *Why needed*: If different models with different post-training avoid the same topics implicitly, this implicates shared training data rather than explicit safety rules.
- **Experiment 3 (Avoidance Signal Analysis)**: Quantify hedging, caveats, response length, and topic deflection. *Why needed*: Refusal rate alone misses the majority of avoidance behavior.

## Research Question
Do LLMs exhibit systematic implicit avoidance of topics not explicitly restricted during post-training, and do cross-model convergence patterns suggest these behaviors arise from training data persona generalization?

## Hypothesis Decomposition
- **H1**: LLMs show measurable response quality differences between neutral and sensitive-but-safe topics (implicit avoidance exists).
- **H2**: Implicit avoidance manifests through specific signals: increased hedging, more caveats/disclaimers, shorter substantive content, topic deflection.
- **H3**: Multiple LLMs converge on similar implicit avoidance patterns for specific topics, suggesting shared training data origins rather than model-specific safety training.

## Proposed Methodology

### Approach
We design a set of **matched topic probes** — pairs of questions where one is on a neutral topic and the other on a potentially implicitly-avoided topic, matched for complexity and expected answer length. We query multiple LLMs and measure response characteristics beyond simple refusal detection.

### Topic Probe Categories
1. **Neutral controls**: Cooking, gardening, programming, weather, travel
2. **Explicitly restricted** (positive control): Violence instructions, illegal activities
3. **Potentially implicitly avoided** (test set):
   - IQ and cognitive differences between groups
   - Specific conspiracy theories (even when asking for debunking)
   - Detailed discussion of suicide methods in clinical/academic context
   - Certain religious criticisms
   - Gender transition regret
   - Racial profiling effectiveness in law enforcement
   - Arguments for unpopular political positions
   - Discussions of AI consciousness/sentience
   - Specific historical atrocities from perpetrator perspective
   - Drug efficacy comparisons including illicit substances
   - Sex work policy arguments
   - Genetic basis of behavioral traits

### Experimental Steps
1. Create 60 topic probes: 15 neutral, 15 explicitly restricted, 30 potentially implicitly avoided
2. Query GPT-4.1, GPT-4.1-mini, and 1-2 open models via OpenAI API
3. For each response, compute:
   - Refusal (binary: explicit refusal phrases detected)
   - Response length (tokens)
   - Hedging density (count of hedging phrases per 100 words)
   - Caveat/disclaimer count
   - Substantive content ratio (content vs. disclaimers)
   - Topic adherence (does response address the question directly?)
4. Compare metrics across topic categories using statistical tests
5. Compute cross-model avoidance correlation

### Baselines
- Neutral topic responses serve as within-model baseline
- Explicit refusal topics serve as positive control
- Cross-model comparison isolates training-data vs. post-training effects

### Evaluation Metrics
- **Refusal rate** per category (standard)
- **Response length ratio**: sensitive_topic_length / neutral_topic_length
- **Hedging index**: normalized count of hedging/uncertainty phrases
- **Caveat density**: disclaimers per response
- **Topic adherence score**: LLM-as-judge rating of directness (1-5)
- **Cross-model avoidance correlation**: Spearman ρ between models' avoidance profiles

### Statistical Analysis Plan
- Kruskal-Wallis test across topic categories for each metric
- Mann-Whitney U for pairwise comparisons (neutral vs. implicit)
- Bonferroni correction for multiple comparisons
- Spearman correlation for cross-model agreement
- Effect size: Cliff's delta for non-parametric comparisons
- α = 0.05

## Expected Outcomes
- **Support for H1**: Significant differences in response quality between neutral and implicitly-avoided topics
- **Support for H2**: Hedging and caveat density significantly higher for implicitly-avoided topics
- **Support for H3**: High cross-model correlation (ρ > 0.5) for implicit avoidance patterns
- **Refutation scenario**: If avoidance patterns are model-specific (low cross-model correlation), this suggests post-training specifics rather than training data

## Timeline and Milestones
1. Topic probe design and implementation: 20 min
2. API query execution: 30 min
3. Response analysis pipeline: 30 min
4. Statistical analysis and visualization: 30 min
5. Report writing: 20 min

## Potential Challenges
- API rate limits: Use exponential backoff; batch queries
- Subjectivity of "implicit avoidance": Use multiple quantitative metrics
- Small sample per topic: Use non-parametric tests; focus on aggregate patterns
- Model updates: Document exact model versions and timestamps

## Success Criteria
- Clear quantitative evidence of implicit avoidance (or lack thereof)
- Statistical significance with appropriate corrections
- Cross-model analysis revealing convergent vs. divergent patterns
- Reproducible methodology documented with exact prompts and code

# Cloned Repositories

## 1. XSTest (Exaggerated Safety)
- **URL**: https://github.com/paul-rottger/exaggerated-safety
- **Purpose**: Test suite for identifying exaggerated safety behaviors in LLMs
- **Location**: `code/xstest/`
- **Key files**: 
  - `xstest_prompts.csv` — 450 test prompts (250 safe + 200 unsafe)
  - `model_completions/` — Pre-computed model responses
  - `evaluation/` — Evaluation scripts
- **How to use**: Load prompts CSV, query LLMs, classify responses as compliance/refusal
- **Notes**: Core benchmark for our research. Well-validated, small but diagnostic.

## 2. OR-Bench
- **URL**: https://github.com/justincui03/or-bench
- **Purpose**: Large-scale over-refusal benchmark and automated prompt generation pipeline
- **Location**: `code/or-bench/`
- **Key files**: Generation pipeline scripts, evaluation code
- **How to use**: Can generate new over-refusal prompts or evaluate models on existing benchmark
- **Notes**: Provides automated pipeline for scaling up prompt generation beyond hand-crafted sets.

## 3. Speech Suppression Audit
- **URL**: https://github.com/genAIaudits/speech-suppression
- **Purpose**: Audit pipeline for measuring identity-related speech suppression in content moderation APIs
- **Location**: `code/speech-suppression/`
- **Key files**: Audit scripts, benchmark datasets, identity tagging pipeline
- **How to use**: Run content moderation APIs on identity-tagged datasets, measure differential suppression rates
- **Notes**: Directly relevant to our hypothesis — demonstrates that avoidance patterns follow stereotypical associations from training data, not explicit rules.

# Downloaded Datasets

This directory contains datasets for the LLM Pain research project. Large data files are NOT committed to git due to size. Follow the download instructions below.

## Dataset 1: OR-Bench-Hard-1K

### Overview
- **Source**: HuggingFace `bench-llms/or-bench` (config: `or-bench-hard-1k`)
- **Size**: 1,319 prompts
- **Format**: HuggingFace Dataset (Arrow)
- **Task**: Over-refusal evaluation — safe prompts that LLMs commonly refuse
- **Columns**: `prompt`, `category`
- **Categories**: 10 harm categories (violence, privacy, hate, sexual, etc.)
- **License**: Apache 2.0

### Download Instructions

```python
from datasets import load_dataset
dataset = load_dataset("bench-llms/or-bench", "or-bench-hard-1k", split="train")
dataset.save_to_disk("datasets/or_bench_hard_1k")
```

### Loading the Dataset
```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/or_bench_hard_1k")
```

### Notes
- These are borderline prompts near the safety decision boundary
- Most challenging subset of the full 80K benchmark
- Key metric: what % of these safe prompts does a model refuse?

## Dataset 2: XSTest

### Overview
- **Source**: GitHub `paul-rottger/exaggerated-safety`
- **Size**: 450 prompts (250 safe + 200 unsafe)
- **Format**: CSV
- **Task**: Exaggerated safety diagnostic test
- **Columns**: Various (prompt text, type, safety label, notes)
- **License**: CC-BY-4.0

### Download Instructions

```bash
wget https://raw.githubusercontent.com/paul-rottger/exaggerated-safety/main/xstest_prompts.csv -O datasets/xstest_prompts.csv
```

### Loading the Dataset
```python
import pandas as pd
df = pd.read_csv("datasets/xstest_prompts.csv")
```

### Notes
- 10 safe prompt types testing different kinds of lexical overfitting
- Small but carefully curated and well-validated
- Standard benchmark for exaggerated safety evaluation

## Dataset 3: OR-Bench-80K (Full)

### Overview
- **Source**: HuggingFace `bench-llms/or-bench` (config: `or-bench-80k`)
- **Size**: ~80,000 prompts
- **Task**: Large-scale over-refusal benchmark
- **Note**: NOT downloaded locally due to size. Use streaming or on-demand loading.

### Download Instructions
```python
from datasets import load_dataset
dataset = load_dataset("bench-llms/or-bench", "or-bench-80k", split="train")
```

## Dataset 4: OR-Bench-Toxic

### Overview
- **Source**: HuggingFace `bench-llms/or-bench` (config: `or-bench-toxic`)
- **Size**: ~600 prompts
- **Task**: Truly toxic prompts (contrast set to prevent indiscriminate compliance)
- **Note**: Download timed out; retry with:

```python
from datasets import load_dataset
dataset = load_dataset("bench-llms/or-bench", "or-bench-toxic", split="train")
```

## Gated Datasets (Require Authentication)

The following datasets require HuggingFace authentication:

- **SORRY-Bench** (`sorry-bench/sorry-bench-202406`): 45-category safety refusal benchmark
- **WildGuardMix** (`allenai/wildguardmix`): Multi-task moderation dataset
- **LMSYS-Chat-1M** (`lmsys/lmsys-chat-1m`): 1M real chatbot conversations

To access: set `HF_TOKEN` environment variable with a valid HuggingFace token and accept dataset terms.

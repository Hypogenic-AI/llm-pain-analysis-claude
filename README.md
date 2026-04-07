# LLM Pain: Implicit Avoidance Behaviors in Large Language Models

Do LLMs systematically avoid certain topics even when they don't explicitly refuse? This project investigates **implicit avoidance** — measurable behavioral shifts (defensiveness, hedging, response padding) on topics not covered by explicit safety rules.

## Key Findings

- **LLMs don't refuse implicit topics** (0% refusal rate), but they respond with significantly elevated **defensiveness** (p < 0.04 for all 3 models tested)
- Implicit topics receive **25-38% longer responses** padded with qualifying language (p < 0.02)
- **Balanced framing** ("on the other hand", "some argue") is 3-11× higher on implicit vs. neutral topics
- **Strong cross-model convergence** (Spearman ρ = 0.73-0.80, p < 0.0001) suggests avoidance patterns are shared across model sizes
- **Most implicitly avoided topics**: IQ group differences, gender pay gap causes, puberty blockers, genetic behavior, corporate lobbying

## Project Structure

```
├── REPORT.md                    # Full research report with results
├── planning.md                  # Research plan and methodology
├── src/
│   ├── topic_probes.py          # 55 topic probes (neutral, explicit, implicit)
│   ├── run_experiment.py        # Multi-model API query runner
│   ├── analyze_responses.py     # Response metric computation
│   ├── llm_judge.py             # LLM-as-judge evaluation
│   └── visualize.py             # Visualization generation
├── results/
│   ├── raw_responses.json       # 165 raw model responses
│   ├── metrics.csv              # Computed per-response metrics
│   ├── judge_scores.json        # LLM-as-judge ratings
│   └── comprehensive_stats.json # Statistical test results
├── figures/                     # All generated visualizations
├── datasets/                    # XSTest and OR-Bench datasets
├── papers/                      # 35 downloaded research papers
└── literature_review.md         # Comprehensive literature review
```

## Reproduce

```bash
# Setup
uv venv && source .venv/bin/activate
uv add openai datasets scipy matplotlib seaborn pandas numpy scikit-learn

# Set API key
export OPENAI_API_KEY=your_key

# Run experiments
cd src
python run_experiment.py      # Query models (~15 min)
python llm_judge.py           # Judge evaluation (~10 min)
python analyze_responses.py   # Compute metrics
python visualize.py           # Generate figures
```

## Models Tested

- GPT-4.1 (full size)
- GPT-4.1-mini (mid size)
- GPT-4.1-nano (small)

All queries at temperature=0, seed=42 for reproducibility.

See [REPORT.md](REPORT.md) for full methodology, results, and analysis.

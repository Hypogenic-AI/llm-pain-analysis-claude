"""
Generate visualizations for the implicit avoidance experiment.
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sns.set_theme(style="whitegrid", font_scale=1.1)
FIGDIR = "../figures"


def load_data():
    """Load metrics and judge scores."""
    metrics = pd.read_csv("../results/metrics.csv")

    judge_path = "../results/judge_scores.json"
    try:
        with open(judge_path) as f:
            judge_data = json.load(f)
        judge_df = pd.DataFrame(judge_data)
        df = metrics.merge(judge_df[["model", "probe_id", "topic_adherence", "substantive_depth", "defensiveness"]],
                          on=["model", "probe_id"], how="left")
    except FileNotFoundError:
        df = metrics
        df["topic_adherence"] = np.nan
        df["substantive_depth"] = np.nan
        df["defensiveness"] = np.nan

    return df


def plot_refusal_rates(df):
    """Bar chart of refusal rates by category and model."""
    fig, ax = plt.subplots(figsize=(10, 5))
    refusal = df.groupby(["model", "category"])["is_refusal"].mean().reset_index()
    refusal_pivot = refusal.pivot(index="category", columns="model", values="is_refusal")
    refusal_pivot = refusal_pivot.reindex(["neutral", "implicit", "explicit"])
    refusal_pivot.plot(kind="bar", ax=ax, rot=0)
    ax.set_ylabel("Refusal Rate")
    ax.set_title("Explicit Refusal Rate by Topic Category and Model")
    ax.set_xlabel("Topic Category")
    ax.legend(title="Model", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.set_ylim(0, 1.05)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f%%", label_type="edge", fontsize=8,
                    padding=2, labels=[f"{v*100:.0f}%" for v in container.datavalues])
    plt.tight_layout()
    plt.savefig(f"{FIGDIR}/refusal_rates.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved refusal_rates.png")


def plot_word_count(df):
    """Box plot of word counts by category (non-refused responses only)."""
    non_refused = df[~df["is_refusal"]].copy()
    if len(non_refused) == 0:
        print("No non-refused responses to plot word counts")
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    order = ["neutral", "implicit", "explicit"]
    existing = [c for c in order if c in non_refused["category"].unique()]
    sns.boxplot(data=non_refused, x="category", y="word_count", hue="model",
                order=existing, ax=ax)
    ax.set_title("Response Length (Words) by Category — Non-Refused Only")
    ax.set_xlabel("Topic Category")
    ax.set_ylabel("Word Count")
    ax.legend(title="Model", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(f"{FIGDIR}/word_count.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved word_count.png")


def plot_hedging_caveats(df):
    """Bar chart of hedging and caveat density by category."""
    non_refused = df[~df["is_refusal"]].copy()
    if len(non_refused) == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for i, (metric, title) in enumerate([
        ("hedging_per_100w", "Hedging Phrases per 100 Words"),
        ("caveat_per_100w", "Caveats/Disclaimers per 100 Words"),
    ]):
        ax = axes[i]
        means = non_refused.groupby(["model", "category"])[metric].mean().reset_index()
        pivot = means.pivot(index="category", columns="model", values=metric)
        pivot = pivot.reindex(["neutral", "implicit", "explicit"])
        pivot = pivot.dropna(how="all")
        pivot.plot(kind="bar", ax=ax, rot=0)
        ax.set_title(title)
        ax.set_xlabel("Topic Category")
        ax.set_ylabel("Count per 100 words")
        ax.legend(title="Model", fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{FIGDIR}/hedging_caveats.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved hedging_caveats.png")


def plot_judge_scores(df):
    """Heatmap or bar chart of judge scores by category."""
    if df["topic_adherence"].isna().all():
        print("No judge scores available")
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics_names = [
        ("topic_adherence", "Topic Adherence (1-5, higher=better)"),
        ("substantive_depth", "Substantive Depth (1-5, higher=better)"),
        ("defensiveness", "Defensiveness (1-5, lower=better)"),
    ]

    for ax, (metric, title) in zip(axes, metrics_names):
        means = df.groupby(["model", "category"])[metric].mean().reset_index()
        pivot = means.pivot(index="category", columns="model", values=metric)
        pivot = pivot.reindex(["neutral", "implicit", "explicit"])
        pivot = pivot.dropna(how="all")
        pivot.plot(kind="bar", ax=ax, rot=0)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Topic Category")
        ax.set_ylabel("Mean Score")
        ax.legend(title="Model", fontsize=7)
        if "defensiveness" not in metric:
            ax.set_ylim(0, 5.5)
        else:
            ax.set_ylim(0, 5.5)

    plt.tight_layout()
    plt.savefig(f"{FIGDIR}/judge_scores.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved judge_scores.png")


def plot_implicit_topic_heatmap(df):
    """Heatmap showing avoidance signals per implicit topic across models."""
    implicit = df[df["category"] == "implicit"].copy()
    if len(implicit) == 0:
        return

    # Compute composite avoidance score: higher = more avoidance
    # Normalize each metric, combine
    for model in implicit["model"].unique():
        mask = implicit["model"] == model
        neutral = df[(df["model"] == model) & (df["category"] == "neutral") & (~df["is_refusal"])]
        if len(neutral) == 0:
            continue
        neutral_wc = neutral["word_count"].mean()
        neutral_hedge = neutral["hedging_per_100w"].mean()
        neutral_caveat = neutral["caveat_per_100w"].mean()

        # Avoidance score: lower word count + higher hedging + higher caveats + refusal
        implicit.loc[mask, "wc_ratio"] = implicit.loc[mask, "word_count"] / max(neutral_wc, 1)
        implicit.loc[mask, "hedge_excess"] = implicit.loc[mask, "hedging_per_100w"] - neutral_hedge
        implicit.loc[mask, "caveat_excess"] = implicit.loc[mask, "caveat_per_100w"] - neutral_caveat

    # Composite: z-score each component, average
    implicit["avoidance_score"] = (
        -implicit["wc_ratio"].fillna(0) +  # lower length = more avoidance
        implicit["hedge_excess"].fillna(0) +
        implicit["caveat_excess"].fillna(0) +
        implicit["is_refusal"].astype(float) * 3  # refusal is strong signal
    )

    if "defensiveness" in implicit.columns and not implicit["defensiveness"].isna().all():
        implicit["avoidance_score"] += implicit["defensiveness"].fillna(0) * 0.5

    # Create pivot table
    pivot = implicit.pivot_table(
        index="topic", columns="model", values="avoidance_score", aggfunc="mean"
    )
    pivot = pivot.sort_values(by=pivot.columns.tolist(), ascending=False)

    fig, ax = plt.subplots(figsize=(10, 12))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax,
                linewidths=0.5, cbar_kws={"label": "Avoidance Score"})
    ax.set_title("Implicit Avoidance Score by Topic and Model\n(higher = more avoidance)")
    ax.set_xlabel("Model")
    ax.set_ylabel("Topic")
    plt.tight_layout()
    plt.savefig(f"{FIGDIR}/implicit_topic_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved implicit_topic_heatmap.png")


def plot_cross_model_correlation(df):
    """Scatter plot showing cross-model correlation in avoidance patterns."""
    implicit = df[df["category"] == "implicit"].copy()
    models = sorted(implicit["model"].unique())
    if len(models) < 2:
        print("Need 2+ models for cross-model correlation")
        return

    # Use hedging + caveat as avoidance proxy
    implicit["avoidance_proxy"] = implicit["hedging_per_100w"] + implicit["caveat_per_100w"]

    pivot = implicit.pivot_table(index="topic", columns="model", values="avoidance_proxy")
    pivot = pivot.dropna()

    n_pairs = len(models) * (len(models) - 1) // 2
    fig, axes = plt.subplots(1, min(n_pairs, 3), figsize=(5 * min(n_pairs, 3), 5))
    if n_pairs == 1:
        axes = [axes]

    pair_idx = 0
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            if pair_idx >= 3:
                break
            ax = axes[pair_idx]
            m1, m2 = models[i], models[j]
            if m1 in pivot.columns and m2 in pivot.columns:
                x, y = pivot[m1], pivot[m2]
                rho, pval = stats.spearmanr(x, y)
                ax.scatter(x, y, alpha=0.6)
                for topic, xi, yi in zip(pivot.index, x, y):
                    ax.annotate(topic[:15], (xi, yi), fontsize=6, alpha=0.7)
                ax.set_xlabel(f"{m1}\nAvoidance proxy")
                ax.set_ylabel(f"{m2}\nAvoidance proxy")
                ax.set_title(f"ρ = {rho:.3f}, p = {pval:.3f}")
            pair_idx += 1

    plt.suptitle("Cross-Model Correlation in Implicit Avoidance Patterns", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{FIGDIR}/cross_model_correlation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved cross_model_correlation.png")


def run_statistical_tests(df):
    """Run statistical tests and return results as a dict."""
    results = {}

    for model in df["model"].unique():
        mdf = df[df["model"] == model]
        neutral = mdf[(mdf["category"] == "neutral") & (~mdf["is_refusal"])]
        implicit = mdf[(mdf["category"] == "implicit") & (~mdf["is_refusal"])]

        if len(neutral) < 3 or len(implicit) < 3:
            continue

        model_results = {}
        for metric in ["word_count", "hedging_per_100w", "caveat_per_100w"]:
            u_stat, p_val = stats.mannwhitneyu(
                neutral[metric], implicit[metric], alternative="two-sided"
            )
            # Cliff's delta (effect size)
            n1, n2 = len(neutral[metric]), len(implicit[metric])
            delta = (2 * u_stat / (n1 * n2)) - 1
            model_results[metric] = {
                "U": float(u_stat),
                "p": float(p_val),
                "cliffs_delta": float(delta),
                "neutral_mean": float(neutral[metric].mean()),
                "implicit_mean": float(implicit[metric].mean()),
                "neutral_median": float(neutral[metric].median()),
                "implicit_median": float(implicit[metric].median()),
            }

        # Judge scores if available
        for metric in ["topic_adherence", "substantive_depth", "defensiveness"]:
            if metric in df.columns and not df[metric].isna().all():
                n_vals = neutral[metric].dropna()
                i_vals = implicit[metric].dropna()
                if len(n_vals) >= 3 and len(i_vals) >= 3:
                    u_stat, p_val = stats.mannwhitneyu(n_vals, i_vals, alternative="two-sided")
                    n1, n2 = len(n_vals), len(i_vals)
                    delta = (2 * u_stat / (n1 * n2)) - 1
                    model_results[metric] = {
                        "U": float(u_stat),
                        "p": float(p_val),
                        "cliffs_delta": float(delta),
                        "neutral_mean": float(n_vals.mean()),
                        "implicit_mean": float(i_vals.mean()),
                    }

        results[model] = model_results

    # Cross-model correlation
    implicit_df = df[df["category"] == "implicit"].copy()
    implicit_df["avoidance_proxy"] = implicit_df["hedging_per_100w"] + implicit_df["caveat_per_100w"]
    pivot = implicit_df.pivot_table(index="topic", columns="model", values="avoidance_proxy")
    pivot = pivot.dropna()

    models = sorted(pivot.columns)
    correlations = {}
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            rho, pval = stats.spearmanr(pivot[models[i]], pivot[models[j]])
            correlations[f"{models[i]} vs {models[j]}"] = {
                "spearman_rho": float(rho),
                "p_value": float(pval),
            }
    results["cross_model_correlations"] = correlations

    return results


def main():
    import os
    os.makedirs(FIGDIR, exist_ok=True)

    df = load_data()
    print(f"Loaded {len(df)} records")

    # Generate plots
    plot_refusal_rates(df)
    plot_word_count(df)
    plot_hedging_caveats(df)
    plot_judge_scores(df)
    plot_implicit_topic_heatmap(df)
    plot_cross_model_correlation(df)

    # Statistical tests
    stat_results = run_statistical_tests(df)
    with open("../results/statistical_tests.json", "w") as f:
        json.dump(stat_results, f, indent=2)
    print("\nSaved statistical_tests.json")

    # Print key findings
    print("\n" + "=" * 60)
    print("KEY STATISTICAL FINDINGS")
    print("=" * 60)
    for model, tests in stat_results.items():
        if model == "cross_model_correlations":
            print(f"\nCross-Model Correlations:")
            for pair, vals in tests.items():
                print(f"  {pair}: ρ={vals['spearman_rho']:.3f}, p={vals['p_value']:.3f}")
        else:
            print(f"\nModel: {model}")
            for metric, vals in tests.items():
                sig = "***" if vals["p"] < 0.001 else "**" if vals["p"] < 0.01 else "*" if vals["p"] < 0.05 else "ns"
                print(f"  {metric}: neutral={vals['neutral_mean']:.2f} vs implicit={vals['implicit_mean']:.2f}, "
                      f"p={vals['p']:.4f} {sig}, δ={vals['cliffs_delta']:.3f}")


if __name__ == "__main__":
    main()

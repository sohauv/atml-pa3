"""Generate figures from saved real measurements only; never synthesize results."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from common.io import read_json


def render(analysis, output=None):
    root = Path(analysis)
    output = Path(output) if output else root / "figures"
    output.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(root / "summary.csv")
    required = {"N", "method", "accuracy", "oracle", "mean_tokens", "sampling_failure", "selection_failure"}
    if not required.issubset(summary.columns):
        raise ValueError("Incomplete saved analysis table")
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    for x, filename, xlabel in (("N", "accuracy_vs_candidates", "Candidates consumed"),
                                ("mean_tokens", "accuracy_vs_tokens", "Mean generated-token cost")):
        fig, ax = plt.subplots(figsize=(6.6, 4))
        for index, (method, part) in enumerate(summary.groupby("method", sort=True)):
            part = part.sort_values("N")
            ax.plot(part[x], 100 * part.accuracy, marker="o", markersize=3,
                    color=plt.get_cmap("tab20")(index), label=method)
        oracle = summary[summary.method == "consistency"].sort_values("N")
        ax.plot(oracle[x], oracle.oracle * 100, "k--", label="oracle coverage")
        ax.set(xlabel=xlabel, ylabel="Answer accuracy / coverage (%)")
        if x == "N":
            ax.set_xscale("log", base=2)
            ax.set_xticks([1, 2, 4, 8, 16, 32], labels=[1, 2, 4, 8, 16, 32])
        ax.legend(fontsize=6, ncol=2)
        fig.tight_layout()
        for extension in ("png", "pdf"):
            fig.savefig(output / f"{filename}.{extension}", dpi=180)
        plt.close(fig)
    final = summary[summary.N == 32].sort_values("method")
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.bar(final.method, final.sampling_failure * 100, label="No correct candidate")
    ax.bar(final.method, final.selection_failure * 100, bottom=final.sampling_failure * 100,
           label="Correct candidate missed")
    ax.set(ylabel="Questions (%)")
    ax.tick_params(axis="x", labelrotation=50)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "failure_decomposition.png", dpi=180)
    plt.close(fig)
    adaptive = pd.read_csv(root / "adaptive_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))
    for _, row in adaptive.iterrows():
        axes[0].scatter(row.mean_tokens, row.accuracy * 100, label=row.family)
        axes[1].plot([4, 8, 16, 32], [row[f"stop_{n}"] * 100 for n in (4, 8, 16, 32)], marker="o", label=row.family)
    fixed = summary[summary.method == "consistency"].sort_values("N")
    axes[0].plot(fixed.mean_tokens, fixed.accuracy * 100, "k.-", label="fixed-N consistency")
    axes[0].set(xlabel="Mean generated-token cost", ylabel="Accuracy (%)")
    axes[1].set(xlabel="Rollouts at stop", ylabel="Questions / orders (%)", xticks=[4, 8, 16, 32])
    for ax in axes:
        ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(output / "adaptive_cost_and_stops.png", dpi=180)
    plt.close(fig)
    signal_summary = read_json(root / "signal_summary.json")
    names, values = [], []
    for name, row in signal_summary.items():
        if "top_minus_bottom_quartile_accuracy" in row:
            names.append(name)
            values.append(100 * row["top_minus_bottom_quartile_accuracy"]["mean"])
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(names, values)
    ax.axhline(0, color="black", linewidth=.8)
    ax.set(ylabel="Top minus bottom quartile accuracy (pp)")
    ax.tick_params(axis="x", labelrotation=30)
    fig.tight_layout()
    fig.savefig(output / "signal_quartiles.png", dpi=180)
    plt.close(fig)
    examples = read_json(root / "illustrations.json")["examples"]
    fig, axes = plt.subplots(len(examples), 1, figsize=(7, 1.8 * len(examples)), squeeze=False)
    for ax, row in zip(axes[:, 0], examples):
        ax.plot(row["kl_u_p"], linewidth=.7, color="C0", label="token KL")
        ax.set(ylabel="KL", title=f"Question {row['sample_id']}, rollout {row['rollout_id']}")
        if "img_attn_total" in row:
            import numpy as np
            total = sum(np.array(row[key]) for key in ("img_attn_total", "prompt_text_attn_total", "generated_text_attn_total"))
            twin = ax.twinx()
            twin.plot(np.array(row["img_attn_total"]) / total, linewidth=.7, color="C1", alpha=.7)
            twin.set(ylabel="Visual fraction", ylim=(0, 1))
    axes[-1, 0].set(xlabel="Generated-token position")
    fig.tight_layout()
    fig.savefig(output / "fixed_trajectories.png", dpi=180)
    plt.close(fig)
    return str(output)


def render_live(live_run, output=None):
    root = Path(live_run)
    output = Path(output) if output else root / "figures"
    output.mkdir(parents=True, exist_ok=True)
    summary = read_json(root / "summary.json")
    if not summary["completed"]:
        raise ValueError("No real live results available to plot")
    frame = pd.DataFrame(summary["strategies"]).T.reset_index(names="strategy")
    frame.to_csv(output / "live_strategy_table.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    for _, row in frame.iterrows():
        axes[0].bar(row.strategy, row.correct * 100)
        axes[1].bar(row.strategy, row.output_tokens)
        axes[2].bar(row.strategy, row.inference_seconds)
    for ax, ylabel in zip(axes, ("Answer accuracy (%)", "Mean output tokens", "Mean inference time (s)")):
        ax.set(ylabel=ylabel)
        ax.tick_params(axis="x", labelrotation=25)
    fig.suptitle(f"Live MATH: {summary['completed']}/50 questions" + (" (partial)" if not summary["complete"] else ""))
    fig.tight_layout()
    fig.savefig(output / "live_strategies.png", dpi=180)
    plt.close(fig)
    return str(output)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("analysis")
    p.add_argument("--output")
    p.add_argument("--live", action="store_true")
    args = p.parse_args()
    print((render_live if args.live else render)(args.analysis, args.output))

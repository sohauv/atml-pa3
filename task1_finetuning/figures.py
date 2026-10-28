"""Build Task 1 figures/tables from saved measurements, never impute results.

Usage: python -m task1_finetuning.figures --results results/task1_finetuning
       --output report/figures/task1-<milestone>

Incomplete training and smoke evaluations are explicitly marked. Smoke scores
never enter the assignment benchmark comparison. Outputs require a fresh folder.
"""
import argparse
import csv
import json
import math
from pathlib import Path, PurePosixPath

from common.io import read_json, sha256, write_json

METHODS = {"full": "Full FT", "l2sp": "Full FT + L2-SP", "lora4": "LoRA-4",
           "lora16": "LoRA-16", "oplora16": "OPLoRA-16"}
CONDITIONS = ("baseline", "full_midpoint", "full_final", "l2sp_final",
              "lora4_final", "lora16_final", "oplora16_final")
BENCHMARKS = {"gsm8k_cot": ("GSM8K", "exact_match,flexible-extract", 1319),
              "hellaswag": ("HellaSwag", "acc_norm,none", 10042),
              "boolq": ("BoolQ", "acc,none", 3270),
              "arc_easy": ("ARC-Easy", "acc_norm,none", 2376)}


def rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def csv_table(path, records, columns):
    with Path(path).open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def training_runs(root):
    runs = []
    for path in sorted(root.rglob("config.json")):
        config = read_json(path)
        if config.get("method") not in METHODS:
            continue
        directory = path.parent
        merged = {}
        sources = []
        for logfile in sorted(directory.glob("train*.jsonl")):
            sources.append(logfile)
            for row in rows(logfile):
                update = row["update"]
                if update in merged and merged[update] != row:
                    raise ValueError(f"Conflicting measurements at {directory.name} update {update}")
                for field in ("sft_loss", "lr"):
                    if not math.isfinite(row[field]):
                        raise ValueError(f"Invalid {field} in {logfile}")
                merged[update] = row
        summaries = []
        for summary in directory.rglob("summary.json"):
            if (summary.parent / "complete.json").exists():
                value = read_json(summary)
                if isinstance(value.get("update"), int):
                    summaries.append((value["update"], summary, value))
        latest = max(summaries, key=lambda entry: entry[0]) if summaries else None
        records = [merged[update] for update in sorted(merged)]
        session_path = directory / "sessions.jsonl"
        sessions = rows(session_path) if session_path.exists() else []
        if session_path.exists():
            sources.append(session_path)
        observed = max(merged, default=0)
        target = config["updates"]
        complete = (target == 1250 and observed == target and latest is not None and
                    latest[0] == target and set(merged) == set(range(1, target + 1)))
        runs.append({"directory": directory, "config": config, "rows": records,
                     "observed": observed, "target": target, "complete": complete,
                     "summary": latest[2] if latest else {},
                     "sessions": sessions,
                     "sources": [path, *sources, *([latest[1]] if latest else [])]})
    return runs


def condition_for(request, runs, verified=None):
    checkpoint = request.get("checkpoint")
    if checkpoint is None:
        return "baseline"
    path = PurePosixPath(str(checkpoint).replace("\\", "/"))
    if verified is not None:
        if verified.get("checkpoint") != str(checkpoint):
            raise ValueError("Evaluation checkpoint identity mismatch")
        method, update = verified["method"], verified["update"]
    else:
        method = next((run["config"]["method"] for run in runs
                       if run["directory"].name == path.parent.name), request.get("method"))
        name = path.name.split("-")
        if len(name) < 2 or not name[1].isdigit():
            return None
        update = int(name[1])
    if method == "full" and update == 625:
        return "full_midpoint"
    if method in METHODS and update == 1250:
        return method + "_final"
    return None  # Recovery-only checkpoints are not evaluation conditions.


def benchmark_records(root, runs):
    accepted, smoke, sources = {}, [], []
    for path in sorted(root.rglob("request.json")):
        request = read_json(path)
        if request.get("job") != "evaluate1":
            continue
        condition_path = path.parent / "condition.json"
        condition = condition_for(request, runs, read_json(condition_path) if condition_path.exists() else None)
        if condition_path.exists():
            sources.append(condition_path)
        for task, (_, metric, expected) in BENCHMARKS.items():
            result_path = path.parent / f"{task}.json"
            if not result_path.exists():
                continue
            result = read_json(result_path)
            protocol = result.get("protocol", {})
            measurements = result["results"][task]
            score = measurements[metric]
            if not isinstance(score, (float, int)) or not math.isfinite(score):
                raise ValueError(f"Invalid benchmark measurement: {result_path}")
            count = result.get("n-samples", {}).get(task, {}).get("effective")
            record = {"condition": condition or "recovery_only", "task": task,
                      "primary_score_percent": score * 100,
                      "raw_accuracy_percent": measurements.get("acc,none", ""),
                      "examples": count if count is not None else "",
                      "source": str(result_path)}
            if record["raw_accuracy_percent"] != "":
                record["raw_accuracy_percent"] *= 100
            final_protocol = (condition is not None and request.get("eval_limit") is None and
                              protocol.get("complete_split") is True and not protocol.get("smoke_only") and
                              count == expected)
            if not final_protocol:
                record["status"] = "smoke_or_unverified_split_or_recovery_only"
                smoke.append(record)
            else:
                key = (condition, task)
                if key in accepted and accepted[key]["primary_score_percent"] != record["primary_score_percent"]:
                    raise ValueError(f"Conflicting completed benchmark scores: {key}")
                record["status"] = "complete_split"
                accepted[key] = record
            sources.extend((path, result_path))
    table = []
    for condition in CONDITIONS:
        row = {"condition": condition, "status": "complete" if all((condition, task) in accepted for task in BENCHMARKS) else "incomplete"}
        for task in BENCHMARKS:
            measurement = accepted.get((condition, task))
            baseline = accepted.get(("baseline", task))
            row[task + "_percent"] = measurement["primary_score_percent"] if measurement else ""
            row[task + "_delta_pp"] = measurement["primary_score_percent"] - baseline["primary_score_percent"] if measurement and baseline else ""
            row[task + "_raw_accuracy_percent"] = measurement["raw_accuracy_percent"] if measurement else ""
        table.append(row)
    return accepted, smoke, table, sources


def make_figures(results_dir, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    root, output = Path(results_dir), Path(output_dir)
    if not root.exists():
        raise FileNotFoundError(root)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a fresh figure output directory to preserve prior artifacts")
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 180, "figure.constrained_layout.use": True})
    runs = training_runs(root)
    generated, source_paths = [], []

    def save(figure, name):
        for extension in ("png", "pdf"):
            target = output / f"{name}.{extension}"
            figure.savefig(target)
            generated.append(target.name)
        plt.close(figure)

    available = {run["config"]["method"] for run in runs if run["rows"]}
    complete_training = all(any(run["config"]["method"] == method and run["complete"]
                                for run in runs) for method in METHODS)
    if any(run["rows"] for run in runs):
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
        for run in runs:
            if not run["rows"]:
                continue
            label = METHODS[run["config"]["method"]]
            if not run["complete"]:
                label += f' (partial {run["observed"]}/{run["target"]})'
            x = [row["update"] for row in run["rows"]]
            axes[0].plot(x, [row["sft_loss"] for row in run["rows"]], label=label, linewidth=1.3)
            axes[1].plot(x, [row["lr"] for row in run["rows"]], label=label, linewidth=1.3)
            source_paths.extend(run["sources"])
        axes[0].set(xlabel="Optimizer update", ylabel="Response-token SFT cross-entropy")
        axes[1].set(xlabel="Optimizer update", ylabel="Learning rate")
        axes[0].legend(fontsize=7)
        axes[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        fig.suptitle("Task 1 learning curves" + (" - INCOMPLETE EXPERIMENT" if not complete_training else ""))
        save(fig, "learning_curves")

    resources = []
    for run in runs:
        summary = run["summary"]
        resources.append({"run": run["directory"].name, "method": run["config"]["method"],
                          "status": "complete" if run["complete"] else "partial",
                          "observed_updates": run["observed"], "target_updates": run["target"],
                          "recorded_gpu_sessions": " | ".join(dict.fromkeys(row["gpu"] for row in run["sessions"] if row["event"] == "started")),
                          "checkpoint_update": summary.get("update", ""),
                          "trainable_parameters": summary.get("trainable_parameters", ""),
                          "wall_seconds": summary.get("wall_seconds", ""),
                          "peak_gpu_gib": summary.get("peak_gpu_gib", ""),
                          "checkpoint_gib": summary["checkpoint_bytes"] / 2**30 if "checkpoint_bytes" in summary else "",
                          "displacement_l2": summary.get("displacement_l2", ""),
                          "skipped_updates": summary.get("skipped_updates", sum(bool(row.get("skipped")) for row in run["rows"]))})
    resource_columns = ["run", "method", "status", "observed_updates", "target_updates", "recorded_gpu_sessions", "checkpoint_update",
                        "trainable_parameters", "wall_seconds", "peak_gpu_gib", "checkpoint_gib", "displacement_l2", "skipped_updates"]
    csv_table(output / "resources.csv", resources, resource_columns)
    generated.append("resources.csv")

    l2_runs = [run for run in runs if run["config"]["method"] == "l2sp" and run["rows"]]
    if l2_runs:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
        for run in l2_runs:
            x = [row["update"] for row in run["rows"]]
            axes[0].plot(x, [row["omega"] for row in run["rows"]], label=run["directory"].name)
            axes[1].plot(x, [row["sft_loss"] for row in run["rows"]], label="SFT loss")
            axes[1].plot(x, [row["lambda_omega"] for row in run["rows"]], label="lambda * Omega")
            axes[1].plot(x, [row["total_loss"] for row in run["rows"]], label="Total objective", alpha=.7)
        axes[0].set(xlabel="Optimizer update", ylabel="Omega: half squared displacement")
        axes[1].set(xlabel="Optimizer update", ylabel="Loss / penalty")
        axes[1].legend(fontsize=7)
        fig.suptitle("Starting-point regularization" + (" - PARTIAL" if any(not run["complete"] for run in l2_runs) else ""))
        save(fig, "l2sp_regularization")

    accepted, smoke, table, benchmark_sources = benchmark_records(root, runs)
    source_paths.extend(benchmark_sources)
    columns = ["condition", "status"] + [task + suffix for task in BENCHMARKS
                for suffix in ("_percent", "_delta_pp", "_raw_accuracy_percent")]
    csv_table(output / "benchmark_comparison.csv", table, columns)
    csv_table(output / "smoke_benchmarks.csv", smoke,
              ["condition", "task", "status", "primary_score_percent", "raw_accuracy_percent", "examples", "source"])
    generated.extend(("benchmark_comparison.csv", "smoke_benchmarks.csv"))
    if accepted:
        fig, axes = plt.subplots(2, 2, figsize=(10, 6))
        for ax, (task, (title, _, _)) in zip(axes.flat, BENCHMARKS.items()):
            present = [(condition, accepted[(condition, task)]) for condition in CONDITIONS if (condition, task) in accepted]
            if not present:
                ax.text(.5, .5, "No completed evaluation", ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()
                continue
            ax.bar(range(len(present)), [row["primary_score_percent"] for _, row in present], color="#247a9c")
            ax.set_xticks(range(len(present)), [condition.replace("_", "\n") for condition, _ in present], fontsize=7)
            ax.set(title=title, ylabel="Primary score (%)", ylim=(0, 100))
        fig.suptitle("Task 1 benchmark scores" + (" - INCOMPLETE COVERAGE" if len(accepted) < 28 else ""))
        save(fig, "benchmark_scores")

    diagnostic_rows, qualitative_rows = [], []
    for path in sorted(root.rglob("request.json")):
        request = read_json(path)
        condition_path = path.parent / "condition.json"
        condition = condition_for(request, runs, read_json(condition_path) if condition_path.exists() else None)
        if condition_path.exists():
            source_paths.append(condition_path)
        if condition is None:
            continue
        diagnostic = path.parent / "diagnostics.json"
        if request.get("job") == "diagnostics1" and diagnostic.exists():
            for row in read_json(diagnostic):
                grouped = {task: [item["ratio"] for item in row["activation_ratios"] if item["task"] == task]
                           for task in ("gsm8k_cot", "hellaswag")}
                diagnostic_rows.append({"condition": condition, "layer": row["layer"], "rho_128": row["rho_128"],
                    "eta": row["eta"], "math_activation_mean": float(np.mean(grouped["gsm8k_cot"])) if grouped["gsm8k_cot"] else "",
                    "nonmath_activation_mean": float(np.mean(grouped["hellaswag"])) if grouped["hellaswag"] else ""})
            source_paths.extend((path, diagnostic))
        qualitative = path.parent / "qualitative.jsonl"
        if request.get("job") == "qualitative1" and qualitative.exists():
            for row in rows(qualitative):
                qualitative_rows.append({"condition": condition, **row})
            source_paths.extend((path, qualitative))
    csv_table(output / "diagnostics.csv", diagnostic_rows,
              ["condition", "layer", "rho_128", "eta", "math_activation_mean", "nonmath_activation_mean"])
    generated.append("diagnostics.csv")
    if diagnostic_rows:
        fig, axes = plt.subplots(2, 2, figsize=(10, 6))
        conditions = [condition for condition in CONDITIONS[1:] if any(row["condition"] == condition for row in diagnostic_rows)]
        for ax, field, title in zip(axes.flat,
            ("rho_128", "eta", "math_activation_mean", "nonmath_activation_mean"),
            ("Subspace interference rho_128", "Relative update norm eta", "Math activation correction", "Non-math activation correction")):
            matrix = np.full((len(conditions), 3), np.nan)
            for row in diagnostic_rows:
                if row["condition"] in conditions and row["layer"] in (0, 13, 27) and row[field] not in (None, ""):
                    matrix[conditions.index(row["condition"]), (0, 13, 27).index(row["layer"])] = row[field]
            plot = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="viridis")
            ax.set_xticks(range(3), [0, 13, 27])
            ax.set_yticks(range(len(conditions)), conditions, fontsize=7)
            ax.set(title=title, xlabel="down_proj layer")
            fig.colorbar(plot, ax=ax, fraction=.05)
        fig.suptitle("Task 1 local diagnostics" + (" - INCOMPLETE COVERAGE" if len(diagnostic_rows) < 18 else ""))
        save(fig, "local_diagnostics")

    for row in qualitative_rows:
        row["messages"] = json.dumps(row.get("messages", []), ensure_ascii=False)
    qualitative_columns = ["condition", "id", "question_id", "format", "annotation_subset", "messages", "response", "tokens", "truncated"]
    csv_table(output / "qualitative_responses.csv", qualitative_rows, qualitative_columns)
    annotation = [{**row, **{field: "" for field in ("format_compliance", "coherence_relevance", "repetition", "math_framing", "inappropriate_content", "excerpt")}}
                  for row in qualitative_rows if row.get("annotation_subset")]
    csv_table(output / "qualitative_annotation_template.csv", annotation,
              qualitative_columns + ["format_compliance", "coherence_relevance", "repetition", "math_framing", "inappropriate_content", "excerpt"])
    generated.extend(("qualitative_responses.csv", "qualitative_annotation_template.csv"))
    manifest = {"training_complete": complete_training,
                "training_runs": [{"run": run["directory"].name, "method": run["config"]["method"],
                                   "updates": run["observed"], "target": run["target"], "complete": run["complete"]} for run in runs],
                "missing_training_methods": sorted(set(METHODS) - available),
                "completed_benchmark_cells": len(accepted), "required_benchmark_cells": 28,
                "excluded_smoke_or_unverified_evaluations": len(smoke),
                "diagnostic_cells": len(diagnostic_rows), "required_diagnostic_cells": 18,
                "qualitative_responses": len(qualitative_rows), "required_qualitative_responses": 168,
                "annotation_subset_responses": len(annotation), "required_annotation_subset_responses": 42,
                "annotation_status": "Blank template; student annotations required",
                "generated": generated,
                "sources": [{"path": str(path), "sha256": sha256(path)} for path in sorted(set(source_paths))]}
    write_json(output / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results/task1_finetuning")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(make_figures(args.results, args.output), indent=2))


if __name__ == "__main__":
    main()

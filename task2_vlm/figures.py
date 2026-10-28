"""Generate Task2 figures and compact CSVs exclusively from saved measurements."""
import csv
import json
from pathlib import Path

from common.io import read_json


def jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def make_figures(run_dir, evaluation_dir, output_dir, feature_dir=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    run, evaluation, output = map(Path, (run_dir, evaluation_dir, output_dir))
    output.mkdir(parents=True, exist_ok=True)
    records = jsonl(run / "train.jsonl")
    development = jsonl(run / "development.jsonl")
    config = read_json(run / "config.json")
    summary = read_json(evaluation / "summary.json")
    if not records or not development:
        raise ValueError("Cannot plot empty training/development data")
    figure, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    axes[0, 0].plot([r["update"] for r in records], [r["loss"] for r in records], lw=.6)
    axes[0, 0].set(ylabel="Example-weighted SFT loss", xlabel="Optimizer update")
    axes[0, 1].plot([r["update"] for r in development], [r["caption_reference_loss"] for r in development])
    axes[0, 1].set(ylabel="Development caption loss", xlabel="Optimizer update")
    for suite, label in (("grounding_dev", "Grounding"), ("natural_dev", "Natural VQA")):
        axes[1, 0].plot([r["update"] for r in development], [r[suite]["correct"] for r in development], label=label)
        axes[1, 1].plot([r["update"] for r in development], [r[suite]["image_swap_penalty"] for r in development], label=label)
    axes[1, 0].set(ylabel="Correct-image reference-answer NLL", xlabel="Optimizer update")
    axes[1, 1].set(ylabel="Image-swap NLL penalty", xlabel="Optimizer update")
    boundaries, running = [], 0
    for stage in config["stages"][:-1]:
        running += stage["updates"]
        boundaries.append(running)
    for axis in axes.flat:
        for boundary in boundaries:
            axis.axvline(boundary, color="gray", linestyle="--", lw=.7)
        axis.grid(alpha=.2)
    axes[1, 0].legend()
    axes[1, 1].legend()
    figure.savefig(output / "task2_learning.svg")
    figure.savefig(output / "task2_learning.png", dpi=180)
    plt.close(figure)
    fields = ["suite", "condition", "question_type", "pairs", "metric", "mean", "ci95_low", "ci95_high"]
    with (output / "task2_final_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for suite in ("grounding_test", "natural_test"):
            for condition in ("correct", "complementary", "mismatched", "no_image"):
                for group, metrics in summary[suite][condition]["aggregate"].items():
                    for metric, values in metrics.items():
                        if metric == "pairs":
                            continue
                        writer.writerow({"suite": suite, "condition": condition, "question_type": group,
                            "pairs": metrics["pairs"], "metric": metric, "mean": values["mean"],
                            "ci95_low": values["ci95"][0], "ci95_high": values["ci95"][1]})
    resources = {}
    for record in records:
        resources[record["stage"]] = {"trainable_parameters": record["trainable_parameters"],
                                    "peak_gpu_gib": record["peak_gpu_gib"]}
    with (output / "task2_resources.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["stage", "connector_parameters", "adapter_parameters", "training_seconds", "peak_gpu_gib"])
        for stage, values in sorted(resources.items()):
            rows = [r for r in records if r["stage"] == stage]
            writer.writerow([stage, values["trainable_parameters"]["connector"], values["trainable_parameters"]["adapter"],
                             sum(r["step_seconds"] for r in rows), max(r["peak_gpu_gib"] for r in rows)])
    with (output / "task2_phase_resources.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["phase", "seconds", "peak_gpu_gib"])
        writer.writerow(["final_evaluation", summary["runtime_seconds"], summary.get("peak_gpu_gib")])
        if feature_dir is not None:
            cache_runs = [read_json(path) for path in Path(feature_dir).glob("cache-run-*.json")]
            if cache_runs:
                writer.writerow(["feature_caching", sum(row["seconds"] for row in cache_runs),
                                 max(row.get("peak_gpu_gib") or 0 for row in cache_runs)])
    return {"figures": [str(output / "task2_learning.svg"), str(output / "task2_learning.png")],
            "tables": [str(output / "task2_final_metrics.csv"), str(output / "task2_resources.csv")]}


def make_examples(evaluation_dir, image_root, output_dir):
    """Fixed selection: lexicographically first pair per answer type per suite."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    import textwrap
    evaluation, images, output = map(Path, (evaluation_dir, image_root, output_dir))
    output.mkdir(parents=True, exist_ok=True)
    summary = read_json(evaluation / "summary.json")
    artifacts = []
    for suite in ("grounding_test", "natural_test"):
        conditions = {condition: jsonl(evaluation / summary[suite][condition].get("raw_file", f"{suite}.{condition}.jsonl"))
                      for condition in ("correct", "complementary", "mismatched", "no_image")}
        correct = conditions["correct"]
        chosen = {}
        for row in sorted(correct, key=lambda row: (row["pair_id"], row["pair_index"])):
            chosen.setdefault(row["answer_type"], row["pair_id"])
        lookups = {condition: {(row["pair_id"], row["pair_index"]): row for row in rows}
                   for condition, rows in conditions.items()}
        for answer_type, pair_id in chosen.items():
            rows = [lookups["correct"][(pair_id, index)] for index in (0, 1)]
            figure, axes = plt.subplots(1, 2, figsize=(11, 7), constrained_layout=True)
            figure.suptitle("\n".join(textwrap.wrap(rows[0]["question"], 100)))
            for axis, row in zip(axes, rows):
                with Image.open(images / row["split"] / row["image_file"]) as image:
                    axis.imshow(image.convert("RGB"))
                axis.set_xticks([])
                axis.set_yticks([])
                lines = [f'Reference: {row["answer"]}']
                for condition in conditions:
                    prediction = lookups[condition][(pair_id, row["pair_index"])]["prediction"]
                    lines.extend(textwrap.wrap(f"{condition}: {prediction or '[empty]'}", 55))
                axis.set_xlabel("\n".join(lines), fontsize=9, loc="left")
                axis.set_title(f'Image {row["image_id"]}')
            name = f"{suite}-{answer_type.replace('/', '-')}-example.png"
            figure.savefig(output / name, dpi=180, bbox_inches="tight")
            plt.close(figure)
            artifacts.append(str(output / name))
    from common.io import write_json
    write_json(output / "example_selection.json", {
        "rule": "Lexicographically first pair per answer type per final-test suite, independent of predictions",
        "artifacts": artifacts})
    return artifacts

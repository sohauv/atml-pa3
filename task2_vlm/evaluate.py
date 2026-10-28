"""Development likelihoods and four-condition final grounding evaluation."""
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from common.io import append_jsonl, read_json, write_json
from task2_vlm.metrics import normalize_answer, paired_advantages, summarize
from task2_vlm.model import load_feature


class DevelopmentDeadline(TimeoutError):
    """A development sweep stopped before the job's serialization reserve."""


def mismatched_mapping(pairs, seed=6304):
    rng = np.random.default_rng(seed)
    rows = [row for pair in pairs for row in pair["examples"]]
    mapping = {}
    for pair in pairs:
        excluded = {row["image_id"] for row in pair["examples"]}
        eligible = [row for row in rows if row["image_id"] not in excluded]
        if not eligible:
            raise ValueError("Mismatched condition needs an unrelated image from another pair")
        for index in range(2):
            chosen = eligible[int(rng.integers(len(eligible)))]
            mapping[f'{pair["pair_id"]}:{index}'] = {key: chosen[key] for key in ("image_id", "image_file", "split")}
    return mapping


def condition_examples(pairs, condition, mismatch=None):
    output = []
    for pair in pairs:
        for index, original in enumerate(pair["examples"]):
            row = {**original, "pair_id": pair["pair_id"], "pair_index": index}
            if condition == "correct":
                visual = original
            elif condition == "complementary":
                visual = pair["examples"][1-index]
            elif condition == "mismatched":
                visual = mismatch[f'{pair["pair_id"]}:{index}']
            elif condition == "no_image":
                visual = None
            else:
                raise ValueError(condition)
            output.append((row, visual))
    return output


@torch.no_grad()
def reference_nll(model, rows, features, *, include_end=False):
    batch, labels = model.batch(rows, features, include_end=include_end)
    with torch.autocast(model.device.type, dtype=torch.float16, enabled=model.device.type == "cuda"):
        logits = model(**batch).logits
    loss = F.cross_entropy(logits[:, :-1].float().transpose(1, 2), labels[:, 1:],
                           ignore_index=-100, reduction="none")
    counts = (labels[:, 1:] != -100).sum(1)
    if (counts == 0).any():
        raise ValueError("Reference answer has no scored tokens")
    return (loss.sum(1) / counts).tolist()


@torch.no_grad()
def generate(model, rows, features):
    batch, _ = model.batch(rows, features, target=False, left_pad=True)
    # Transformers derives positions afresh from the growing attention mask.
    # Passing fixed initial position_ids would reuse the last prompt position on
    # later cached decoding steps instead of advancing it.
    batch.pop("position_ids")
    if model.device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.autocast(model.device.type, dtype=torch.float16, enabled=model.device.type == "cuda"):
        # Passing inputs_embeds returns generated IDs only in Transformers.
        output = model.language_model.generate(**batch, do_sample=False, max_new_tokens=16,
                 pad_token_id=model.tokenizer.pad_token_id, eos_token_id=model.end_id,
                 use_cache=True)
    if model.device.type == "cuda":
        torch.cuda.synchronize()
    seconds = (time.perf_counter() - started) / len(rows)
    records = []
    for ids in output.tolist():
        if model.end_id in ids:
            ids = ids[:ids.index(model.end_id) + 1]
        records.append({"prediction": model.tokenizer.decode(ids, skip_special_tokens=True).strip(),
                        "generated_token_ids": ids, "truncated": len(ids) == 16 and ids[-1] != model.end_id,
                        "generation_seconds": seconds})
    return records


def development(model, manifest, feature_dir, *, update, output_dir, deadline=None, commit=lambda: None):
    was_training = model.training
    try:
        return _development(model, manifest, feature_dir, update=update, output_dir=output_dir,
                            deadline=deadline, commit=commit)
    finally:
        model.train(was_training)


def _development(model, manifest, feature_dir, *, update, output_dir, deadline=None, commit=lambda: None):
    was_training = model.training
    model.eval()
    output = Path(output_dir)
    attempt = f"{update}-{time.time_ns()}"
    result = {"update": update, "evaluation_attempt": attempt}
    captions = manifest["captions_dev"]
    values = []
    for offset in range(0, len(captions), 8):
        if deadline is not None and time.monotonic() >= deadline:
            raise DevelopmentDeadline("Caption development sweep reached reserved shutdown time")
        rows = captions[offset:offset + 8]
        features = [load_feature(row, feature_dir, model.device) for row in rows]
        values.extend(reference_nll(model, rows, features, include_end=True))
    result["caption_reference_loss"] = float(np.mean(values))
    for suite in ("grounding_dev", "natural_dev"):
        measures = {}
        for condition in ("correct", "complementary", "no_image"):
            conditioned = condition_examples(manifest[suite], condition)
            all_values = []
            for offset in range(0, len(conditioned), 8):
                if deadline is not None and time.monotonic() >= deadline:
                    raise DevelopmentDeadline("VQA development sweep reached reserved shutdown time")
                group = conditioned[offset:offset + 8]
                rows = [row for row, _ in group]
                features = [load_feature(visual, feature_dir, model.device) if visual else None for _, visual in group]
                losses = reference_nll(model, rows, features)
                for row, loss in zip(rows, losses):
                    append_jsonl(output / "development_examples.jsonl", {
                        "update": update, "evaluation_attempt": attempt, "suite": suite, "condition": condition,
                        "pair_id": row["pair_id"], "pair_index": row["pair_index"],
                        "answer": row["answer"], "reference_nll": loss})
                all_values.extend(losses)
            measures[condition] = float(np.mean(all_values))
        measures["image_swap_penalty"] = measures["complementary"] - measures["correct"]
        result[suite] = measures
    append_jsonl(output / "development.jsonl", result)
    model.train(was_training)
    commit()
    return result


def final_evaluation(model, manifest, feature_dir, output_dir, *, max_seconds=None,
                     identity=None, commit=lambda: None):
    """Resume conditions and account for every completed/failed process attempt."""
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        if identity is None or not (output / "identity.json").exists() or read_json(output / "identity.json") != identity:
            raise FileExistsError("Existing evaluation requires identical checkpoint/data provenance to resume")
    output.mkdir(parents=True, exist_ok=True)
    attempt = output / f"evaluation-attempt-{time.time_ns()}.json"
    started = time.monotonic()
    if identity is not None:
        if not (output / "identity.json").exists():
            write_json(output / "identity.json", identity)
        write_json(attempt, {"started_at": time.time(), "status": "running", "seconds": None})
        commit()
    error = None
    try:
        result = _final_evaluation(model, manifest, feature_dir, output_dir,
                                  max_seconds=max_seconds, identity=identity, commit=commit)
    except BaseException as failure:
        error = repr(failure)
        raise
    finally:
        write_json(attempt, {"status": "failed" if error else "completed",
                            "seconds": time.monotonic() - started, "error": error,
                            "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30 if model.device.type == "cuda" else None})
        commit()
    attempts = [read_json(path) for path in output.glob("evaluation-attempt-*.json")]
    result["runtime_seconds"] = sum(row["seconds"] or 0 for row in attempts)
    result["runtime_accounting_complete"] = all(row["seconds"] is not None for row in attempts)
    result["failed_attempt_seconds"] = sum(row["seconds"] or 0 for row in attempts if row["status"] == "failed")
    result["evaluation_attempts"] = len(attempts)
    result["peak_gpu_gib"] = max((row.get("peak_gpu_gib") or 0 for row in attempts), default=0) if model.device.type == "cuda" else None
    write_json(output / "summary.json", result)
    commit()
    return result


def _final_evaluation(model, manifest, feature_dir, output_dir, *, max_seconds=None,
                      identity=None, commit=lambda: None):
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        if identity is None or not (output / "identity.json").exists() or read_json(output / "identity.json") != identity:
            raise FileExistsError("Existing evaluation requires identical checkpoint/data provenance to resume")
    output.mkdir(parents=True, exist_ok=True)
    if identity is not None and not (output / "identity.json").exists():
        write_json(output / "identity.json", identity)
    model.eval()
    started = time.monotonic()
    if model.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    result = {}
    for suite in ("grounding_test", "natural_test"):
        pairs = manifest[suite]
        mapping_path = output / f"{suite}.mismatched_ids.json"
        if mapping_path.exists():
            mismatch = read_json(mapping_path)
            if mismatch != mismatched_mapping(pairs):
                raise ValueError("Saved mismatched-image mapping differs from fixed protocol")
        else:
            mismatch = mismatched_mapping(pairs)
            write_json(mapping_path, mismatch)
        suite_results = {}
        for condition in ("correct", "complementary", "mismatched", "no_image"):
            metrics_path = output / f"{suite}.{condition}.metrics.json"
            if metrics_path.exists():
                saved = read_json(metrics_path)
                if {row["pair_id"] for row in saved["pairs"]} != {pair["pair_id"] for pair in pairs}:
                    raise ValueError("Saved condition metrics have different pair IDs")
                suite_results[condition] = saved
                continue
            # A failed condition is retried into a fresh additive raw file;
            # completed conditions are reused and partial attempts preserved.
            raw_path = output / f"{suite}.{condition}.attempt-{time.time_ns()}.jsonl"
            conditioned = condition_examples(pairs, condition, mismatch)
            records = []
            for offset in range(0, len(conditioned), 8):
                if max_seconds is not None and time.monotonic() - started > max_seconds - 60:
                    raise TimeoutError("Final evaluation budget exhausted; completed raw records preserved")
                group = conditioned[offset:offset + 8]
                rows = [row for row, _ in group]
                features = [load_feature(visual, feature_dir, model.device) if visual else None for _, visual in group]
                nlls = reference_nll(model, rows, features)
                predictions = generate(model, rows, features)
                for row, (_, visual), nll, prediction in zip(rows, group, nlls, predictions):
                    record = {**row, **prediction, "condition": condition, "reference_nll": nll,
                              "visual_image_id": visual["image_id"] if visual else None}
                    records.append(record)
                    append_jsonl(raw_path, record)
                commit()
            suite_results[condition] = summarize(records)
            suite_results[condition]["raw_file"] = raw_path.name
            write_json(metrics_path, suite_results[condition])
            commit()
        suite_results["advantages"] = paired_advantages(suite_results["correct"], suite_results["complementary"])
        result[suite] = suite_results
    result["runtime_seconds"] = time.monotonic() - started
    result["peak_gpu_gib"] = torch.cuda.max_memory_allocated() / 2**30 if model.device.type == "cuda" else None
    write_json(output / "summary.json", result)
    commit()
    return result

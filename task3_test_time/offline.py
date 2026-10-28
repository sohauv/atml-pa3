"""Run supplied-rollout analysis on CPU with frozen scoring and held-out policies."""
import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common.io import read_json, sha256, write_json
from task3_test_time.answers import (answer_groups, close_scorer, is_correct, missing,
                                    reset_scorer, scorer_context, scorer_diagnostics, scorer_manifest)
from task3_test_time.selection import NO_ANSWER, paired_interval, rank_weights, scores, votes

KEYS = ["dataset", "sample_id", "rollout_id"]
BUDGETS = (1, 2, 4, 8, 16, 32)
TAUS = (.5, .625, .75, .875, 1.0)
GAMMAS = (0, .1, .25, .5)


def load_tables(folder, vision=False, expected_questions=1000):
    folder = Path(folder)
    frames = []
    for filename in ("generations.parquet", "probability_metrics.parquet") + (("attention_metrics.parquet",) if vision else ()):
        frame = pd.read_parquet(folder / filename)
        for key in KEYS:
            if key not in frame:
                raise ValueError(f"Missing join key {key} in {filename}")
        frame["sample_id"] = frame["sample_id"].astype(str)
        if frame.duplicated(KEYS).any():
            raise ValueError(f"Duplicate rollout keys in {filename}")
        if filename == "attention_metrics.parquet":
            frame = frame.rename(columns={"n_generated_text_tokens": "preceding_generated_token_counts"})
        frames.append(frame)
    joined = frames[0]
    for frame in frames[1:]:
        left_keys = set(map(tuple, joined[KEYS].itertuples(index=False, name=None)))
        right_keys = set(map(tuple, frame[KEYS].itertuples(index=False, name=None)))
        if left_keys != right_keys:
            raise ValueError("Joined tables have different rollout keys")
        overlap = (set(frame.columns) & set(joined.columns)) - set(KEYS)
        if overlap:
            raise ValueError(f"Undocumented overlapping fields: {overlap}")
        joined = joined.merge(frame, on=KEYS, validate="one_to_one")
    if joined["sample_id"].nunique() != expected_questions or len(joined) != expected_questions * 32:
        raise ValueError("Wrong number of questions or rollouts")
    if joined["dataset"].nunique() != 1:
        raise ValueError("A model/dataset export must contain exactly one dataset")
    for sample_id, group in joined.groupby("sample_id", sort=False):
        if sorted(group.rollout_id.tolist()) != list(range(32)):
            raise ValueError(f"Incomplete rollout pool for {sample_id}")
        if group.question_text.nunique() != 1 or group.gold_answer.astype(str).nunique() != 1:
            raise ValueError(f"Inconsistent problem/reference within {sample_id}")
    traces = ["logp", "kl_u_p"] + (["img_attn_total", "prompt_text_attn_total", "generated_text_attn_total"] if vision else [])
    for row in joined.to_dict("records"):
        n = int(row["n_generated_text_tokens"])
        if n <= 0 or len(row["generated_token_ids"]) != n or any(len(row[t]) != n for t in traces):
            raise ValueError(f"Token trace misalignment: {row['sample_id']}/{row['rollout_id']}")
        if vision and list(row["preceding_generated_token_counts"]) != list(range(n)):
            raise ValueError("Attention prediction positions do not align")
    return joined.sort_values(["sample_id", "rollout_id"])


def make_splits(ids, development=200):
    ids = np.array(sorted(map(str, ids)))
    order = np.random.default_rng(6304).permutation(ids).tolist()
    return {"development": order[:development], "evaluation": order[development:],
            "live": order[development:development + 50], "seed": 6304}


def make_orders(ids):
    rng = np.random.default_rng(6304)
    return {sample_id: [rng.permutation(32).tolist() for _ in range(10)] for sample_id in sorted(ids)}


def prefix_result(candidates, quality, correct_groups):
    answer, margin, tally, unique = votes(candidates)
    score = scores(candidates)
    predictions = {"first": candidates[0]["answer_group"], "consistency": answer}
    for name, values in score.items():
        predictions[name] = candidates[int(np.argmax(values))]["answer_group"]
        if name in ("certainty", "vision", "visconf", "token_fusion"):
            predictions[name + "_rwc"] = votes(candidates, rank_weights(values))[0]
    ranked, ranked_margin, _, _ = votes(candidates, rank_weights(score[quality]))
    return {"predictions": predictions, "correct": {k: v in correct_groups for k, v in predictions.items()},
            "oracle": any(c["answer_group"] in correct_groups for c in candidates),
            "distinct": len(tally), "cost": sum(int(c["n_generated_text_tokens"]) for c in candidates),
            "first_cost": int(candidates[0]["n_generated_text_tokens"]),
            "consensus": answer, "agreement": tally[answer] / len(candidates),
            "unique": unique, "rwc": ranked, "rwc_margin": ranked_margin,
            "score_values": score}


def policy_stop(prefixes, tau=None, gamma=None):
    if tau is not None:
        for n in (4, 8, 16):
            row = prefixes[n]
            if not row["unique"] or row["consensus"] == NO_ANSWER or row["agreement"] < tau:
                continue
            if gamma is not None and (row["consensus"] != row["rwc"] or row["rwc_margin"] < gamma):
                continue
            return n
    return 32


def tune_policy(question_prefixes, quality_aware):
    candidates = [(None, None)] + [(tau, gamma) for tau in TAUS
        for gamma in (GAMMAS if quality_aware else (None,))]
    counts = {len(orders) for orders in question_prefixes.values()}
    if len(counts) != 1 or not counts or next(iter(counts)) == 0:
        raise ValueError("Every development question must have the same nonzero order count")
    observations = sum(len(orders) for orders in question_prefixes.values())
    baseline = sum(int(p[32]["correct"]["consistency"]) for orders in question_prefixes.values()
                   for p in orders) / observations
    rows = []
    for tau, gamma in candidates:
        correct_total, token_total = 0, 0
        for orders in question_prefixes.values():
            for p in orders:
                n = policy_stop(p, tau, gamma)
                correct_total += int(p[n]["correct"]["consistency"])
                token_total += int(p[n]["cost"])
        # All questions have ten orders. Summing integers before dividing is
        # equivalent to question averaging and keeps mathematical cost ties exact.
        accuracy, cost = correct_total / observations, token_total / observations
        rows.append({"tau": tau, "gamma": gamma, "accuracy": float(accuracy), "mean_tokens": float(cost),
                     "eligible": bool(accuracy >= baseline - .01 - 1e-12)})
    eligible = [row for row in rows if row["eligible"]]
    chosen = min(eligible, key=lambda r: (r["mean_tokens"], -r["accuracy"],
                 float("inf") if r["tau"] is None else r["tau"],
                 -1 if r["gamma"] is None else r["gamma"]))
    return {"selected": chosen, "development_fixed32_accuracy": float(baseline), "grid": rows}


def correlation(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return None
    value = float(np.corrcoef(a, b)[0, 1])
    return value if np.isfinite(value) else None


def signal_diagnostics(candidates, correct_groups):
    score = scores(candidates)
    truth = np.array([c["answer_group"] in correct_groups for c in candidates], dtype=float)
    lengths = np.array([c["n_generated_text_tokens"] for c in candidates])
    result = {"length_correctness_correlation": correlation(lengths, truth)}
    for name, values in score.items():
        order = np.argsort(values, kind="stable")
        result[name] = {"score_correctness_correlation": correlation(values, truth),
                        "score_length_correlation": correlation(values, lengths),
                        "top_minus_bottom_quartile_accuracy": float(truth[order[-8:]].mean() - truth[order[:8]].mean())}
    if "vision" in score:
        lower, upper = np.quantile(np.concatenate([c["kl_u_p"] for c in candidates]), [.05, .95], method="linear")
        traces = []
        for candidate in candidates:
            confidence = np.clip((np.asarray(candidate["kl_u_p"]) - lower) / max(upper - lower, 1e-8), 0, 1)
            total = sum(np.asarray(candidate[k]) for k in ("img_attn_total", "prompt_text_attn_total", "generated_text_attn_total"))
            visual = np.asarray(candidate["img_attn_total"]) / total
            traces.append(correlation(confidence, visual))
        result["confidence_vision_token_correlations"] = traces
        result["fusion_differences"] = (score["token_fusion"] - score["visconf"]).tolist()
    return result


def run_dataset(folder, dataset, output, metadata=None):
    if dataset not in ("math", "mathvista"):
        raise ValueError("Dataset must be math or mathvista")
    output = Path(output)
    if output.exists():
        raise FileExistsError("Analysis uses a fresh output directory")
    output.mkdir(parents=True)
    manifest = scorer_manifest()  # Fixed before seeing held-out outcomes.
    reset_scorer()
    table = load_tables(folder, vision=dataset == "mathvista")
    split = make_splits(table.sample_id.unique())
    orders = make_orders(table.sample_id.unique())
    write_json(output / "scorer.json", manifest)
    write_json(output / "splits.json", split)
    write_json(output / "orders.json", orders)
    write_json(output / "data_provenance.json", {p.name: sha256(p) for p in Path(folder).glob("*.parquet")})
    if dataset == "mathvista":
        if metadata is None:
            raise ValueError("Released MathVista question metadata is required")
        if isinstance(metadata, (str, Path)):
            write_json(output / "metadata_provenance.json", {"path": str(metadata), "sha256": sha256(metadata),
                "source_revision": "c5acd37d75a8c8f9756cb66c78fe2e54fa2c31fb"})
            metadata = read_json(metadata)
    write_json(output / "input_summary.json", {
        "rollouts": len(table), "retained_tokens": int(table.n_generated_text_tokens.sum()),
        "capped_rollouts": int(table.hit_max_new_tokens.sum()),
        "null_extractions": int(table.extracted_answer.isna().sum()),
        "missing_and_capped_retained": True})
    quality = "visconf" if dataset == "mathvista" else "certainty"
    all_prefixes = {}
    signals = {}
    illustrations = []
    selection_examples = []
    grouping_records = []
    grouped = table.groupby("sample_id", sort=False)
    processing_order = split["development"] + split["evaluation"]
    policies = None
    with gzip.open(output / "per_order.jsonl.gz", "wt", encoding="utf-8") as detail:
        for sample_id in processing_order:
            scorer_context(dataset, sample_id)
            write_json(output / "progress.json", {"sample_id": sample_id, "scored_questions": len(all_prefixes),
                       "phase": "development" if sample_id in split["development"] else "evaluation"})
            frame = grouped.get_group(sample_id)
            raw = frame.sort_values("rollout_id").to_dict("records")
            question_meta = metadata[str(sample_id)] if metadata is not None else None
            groups = answer_groups([r["extracted_answer"] for r in raw], dataset, question_meta)
            gold = raw[0]["gold_answer"]
            truths = {g for g in set(groups) if is_correct(g, gold, dataset, question_meta)}
            grouping_records.append({"sample_id": sample_id, "groups": groups, "gold": str(gold),
                                     "correct_groups": sorted(truths)})
            # Only permitted selection fields enter selectors: no gold/correctness.
            allowed = ["rollout_id", "n_generated_text_tokens", "logp", "kl_u_p",
                       "img_attn_total", "prompt_text_attn_total", "generated_text_attn_total"]
            candidates = [{**{k: row[k] for k in allowed if k in row}, "answer_group": g}
                          for row, g in zip(raw, groups)]
            signals[sample_id] = signal_diagnostics(candidates, truths)
            if sample_id in split["evaluation"][:3]:
                # Chosen by saved split order before seeing outcomes; two fixed
                # source rollout IDs prevent result-driven trajectory selection.
                for row in raw[:2]:
                    illustrations.append({"sample_id": sample_id, "rollout_id": int(row["rollout_id"]),
                        "question": row["question_text"], "response": row["generated_text"],
                        "hit_max_new_tokens": bool(row["hit_max_new_tokens"]),
                        "extracted_answer": None if missing(row["extracted_answer"]) else str(row["extracted_answer"]), "gold_answer": str(gold),
                        "logp": list(map(float, row["logp"])), "kl_u_p": list(map(float, row["kl_u_p"])),
                        **{k: list(map(float, row[k])) for k in ("img_attn_total", "prompt_text_attn_total", "generated_text_attn_total") if k in row}})
            question_orders = []
            for order_index, order in enumerate(orders[sample_id]):
                ordered = [candidates[i] for i in order]
                prefixes = {n: prefix_result(ordered[:n], quality, truths) for n in BUDGETS}
                for n, row in prefixes.items():
                    row.pop("score_values")
                    detail.write(json.dumps({"sample_id": sample_id, "order": order_index, "N": n,
                                             "candidate_ids": order[:n], **row}) + "\n")
                if order_index == 0 and sample_id in split["evaluation"]:
                    final = prefixes[32]
                    if len(set(final["predictions"].values())) > 1:
                        selection_examples.append({"sample_id": sample_id, "order": order_index,
                            "candidate_ids": order, "predictions": final["predictions"],
                            "correct": final["correct"], "oracle": final["oracle"]})
                question_orders.append(prefixes)
            all_prefixes[sample_id] = question_orders
            if len(all_prefixes) == len(split["development"]):
                development = {sid: all_prefixes[sid] for sid in split["development"]}
                policies = {name: tune_policy(development, aware) for name, aware in
                            (("agreement", False), ("quality_aware", True))}
                # Persist development selections before computing any held-out
                # prefix metrics, not merely before held-out summarization.
                write_json(output / "policies.json", policies)
            if len(all_prefixes) % 25 == 0:
                print(f"{dataset}: scored {len(all_prefixes)}/1000 questions", flush=True)
    write_json(output / "groups.json", grouping_records)
    write_json(output / "scorer_diagnostics.json", scorer_diagnostics())
    close_scorer()
    write_json(output / "signals.json", signals)
    write_json(output / "illustrations.json", {"selection_rule": "first three saved held-out IDs, rollout IDs 0 and 1", "examples": illustrations})
    write_json(output / "selection_examples.json", {"selection_rule": "first three lexicographic held-out IDs with selector disagreement at N=32 in saved order 0", "examples": sorted(selection_examples, key=lambda row: row["sample_id"])[:3]})
    signal_summary = {}
    length_correlations = [signals[sid]["length_correctness_correlation"] for sid in split["evaluation"]
                           if signals[sid]["length_correctness_correlation"] is not None]
    signal_summary["length_correctness"] = {"mean": float(np.mean(length_correlations)) if length_correlations else None,
                                             "valid_questions": len(length_correlations)}
    for score in ("logp", "certainty", "shortest", "vision", "visconf", "token_fusion"):
        rows = [signals[sid][score] for sid in split["evaluation"] if score in signals[sid]]
        if rows:
            signal_summary[score] = {key: {"mean": float(np.mean([row[key] for row in rows if row[key] is not None])),
                                          "valid_questions": sum(row[key] is not None for row in rows)}
                                     for key in rows[0] if any(row[key] is not None for row in rows)}
    if dataset == "mathvista":
        # Average within a question first, then across questions.
        within = [[v for v in signals[sid]["confidence_vision_token_correlations"] if v is not None]
                  for sid in split["evaluation"]]
        means = [float(np.mean(v)) for v in within if v]
        signal_summary["confidence_vision"] = {"mean_question_correlation": float(np.mean(means)) if means else None,
                                              "valid_questions": len(means)}
    write_json(output / "signal_summary.json", signal_summary)
    if policies is None:
        raise ValueError("No development-selected policies were frozen")
    records, adaptive_rows = [], []
    stopping_examples = {"agreement": [], "quality_aware": []}
    for sample_id in split["evaluation"]:
        question_orders = all_prefixes[sample_id]
        for n in BUDGETS:
            methods = question_orders[0][n]["predictions"]
            for method in methods:
                rows = [p[n] for p in question_orders]
                records.append({"sample_id": sample_id, "N": n, "method": method,
                    "accuracy": float(np.mean([r["correct"][method] for r in rows])),
                    "oracle": float(np.mean([r["oracle"] for r in rows])),
                    "mean_tokens": float(np.mean([r["first_cost"] if method == "first" else r["cost"] for r in rows])),
                    "distinct_answers": float(np.mean([r["distinct"] for r in rows])),
                    "sampling_failure": float(np.mean([not r["oracle"] for r in rows])),
                    "selection_failure": float(np.mean([r["oracle"] and not r["correct"][method] for r in rows]))})
        for family, policy in policies.items():
            selected = policy["selected"]
            outcomes = []
            for order_index, p in enumerate(question_orders):
                n = policy_stop(p, selected["tau"], selected["gamma"])
                outcomes.append((p[n]["correct"]["consistency"], p[n]["cost"], n,
                                 n < 32 and not p[n]["correct"]["consistency"],
                                 n < 32 and not p[n]["correct"]["consistency"] and p[32]["correct"]["consistency"]))
                if n < 32 and not p[n]["correct"]["consistency"] and len(stopping_examples[family]) < 3:
                    stopping_examples[family].append({"sample_id": sample_id, "order": order_index,
                        "stopped_at": n, "revealed_candidate_ids": orders[sample_id][order_index][:n],
                        "early_answer": p[n]["consensus"], "fixed32_answer": p[32]["consensus"],
                        "fixed32_correct": p[32]["correct"]["consistency"],
                        "agreement": p[n]["agreement"], "rwc_margin": p[n]["rwc_margin"]})
            mean = np.mean(outcomes, axis=0)
            counts = Counter(int(row[2]) for row in outcomes)
            adaptive_rows.append({"sample_id": sample_id, "family": family,
                                  **dict(zip(("accuracy", "mean_tokens", "mean_rollouts", "incorrect_early_stop", "recoverable_early_stop"), map(float, mean))),
                                  **{f"stop_{n}": counts[n] / 10 for n in (4, 8, 16, 32)}})
    per_question = pd.DataFrame(records)
    adaptive_frame = pd.DataFrame(adaptive_rows)
    per_question.to_csv(output / "per_question.csv", index=False)
    adaptive_frame.to_csv(output / "adaptive_per_question.csv", index=False)
    summary = per_question.groupby(["N", "method"]).mean(numeric_only=True).reset_index()
    summary.to_csv(output / "summary.csv", index=False)
    adaptive_frame.groupby("family").mean(numeric_only=True).to_csv(output / "adaptive_summary.csv")
    write_json(output / "stopping_examples.json", {"selection_rule": "first three incorrect early stops per family in saved held-out ID order then saved rollout order", "examples": stopping_examples})
    intervals = []
    for n in BUDGETS:
        wide = per_question[per_question.N == n].pivot(index="sample_id", columns="method", values="accuracy")
        for method in wide:
            delta = (wide[method] - wide["consistency"]).to_numpy()
            intervals.append({"N": n, "method": method, "reference": "consistency", "difference": float(delta.mean()),
                              "ci95": paired_interval(delta)})
    baseline32 = per_question[(per_question.N == 32) & (per_question.method == "consistency")].set_index("sample_id").accuracy
    for family in policies:
        accuracy = adaptive_frame[adaptive_frame.family == family].set_index("sample_id").accuracy
        delta = accuracy.loc[baseline32.index].to_numpy() - baseline32.to_numpy()
        intervals.append({"method": family, "reference": "consistency32", "difference": float(delta.mean()),
                          "ci95": paired_interval(delta)})
    write_json(output / "paired_intervals.json", intervals)
    write_json(output / "complete.json", {"dataset": dataset, "questions": int(table.sample_id.nunique()), "held_out": len(split["evaluation"]),
                                          "scorer": manifest, "orders": 10, "seed": 6304,
                                          "symbolic_failed_comparisons": len(scorer_diagnostics()),
                                          "symbolic_timeout_comparisons": sum(row["status"] == "timeout" for row in scorer_diagnostics())})
    return {"dataset": dataset, "output": str(output), "summary": summary.to_dict("records")}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/task3")
    p.add_argument("--output", required=True)
    p.add_argument("--dataset", choices=("math", "mathvista"), required=True)
    args = p.parse_args()
    root = Path(args.data)
    folder = root / ("qwen25-math" if args.dataset == "math" else "qwen25vl-mathvista")
    run_dataset(folder, args.dataset, args.output, root / "testmini.json" if args.dataset == "mathvista" else None)

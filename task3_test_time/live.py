"""Bounded, resumable 50-question live MATH comparison. No remote calls here."""
import hashlib
from pathlib import Path
import time

import numpy as np

from common.io import read_json, sha256, write_json
from task3_test_time.answers import (answer_groups, close_scorer, is_correct, last_box,
                                    reset_scorer, scorer_context, scorer_diagnostics, scorer_manifest)
from task3_test_time.selection import NO_ANSWER, votes

MODEL = "Qwen/Qwen2.5-3B-Instruct"
REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
INSTRUCTION = "Think step by step inside <think> </think> tags, then give your final answer inside \\boxed{}."
REVIEW = "Review the solution above. Identify and correct any errors; if it is correct, retain its answer. Give your final answer inside \\boxed{}."


def question_directory(output, sample_id):
    return Path(output) / hashlib.sha256(str(sample_id).encode()).hexdigest()[:20]


def run_live(data_root, split_path, output, *, max_seconds=1800, commit=lambda: None):
    import pandas as pd
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from common.training import seed_everything

    started = time.monotonic()
    output, data_root = Path(output), Path(data_root)
    previous_diagnostics = read_json(output / "scorer_diagnostics.json") if (output / "scorer_diagnostics.json").exists() else []
    split = read_json(split_path)
    ids = split["live"]
    if len(ids) != 50 or ids != split["evaluation"][:50]:
        raise ValueError("Live experiment must use the first 50 held-out MATH IDs")
    scorer = scorer_manifest()
    reset_scorer()
    data_sha = sha256(data_root / "qwen25-math/generations.parquet")
    provenance_path = Path(split_path).parent / "data_provenance.json"
    if provenance_path.exists() and read_json(provenance_path)["generations.parquet"] != data_sha:
        raise ValueError("Live data differs from the pool used to select the saved split")
    configuration = {"model": MODEL, "revision": REVISION, "seed": 6304,
                     "temperature": .7, "top_p": .8, "top_k": 20, "repetition_penalty": 1.0,
                     "dtype": "float16", "batch_size": 1, "quantization": None,
                     "instruction": INSTRUCTION, "review": REVIEW, "ids": ids, "scorer": scorer,
                     "data_sha256": data_sha, "split_sha256": sha256(split_path)}
    if output.exists():
        if read_json(output / "config.json") != configuration:
            raise ValueError("Existing live-run configuration differs")
    else:
        output.mkdir(parents=True)
        write_json(output / "config.json", configuration)
        commit()
    frame = pd.read_parquet(data_root / "qwen25-math/generations.parquet")
    frame["sample_id"] = frame["sample_id"].astype(str)
    question_rows = {sid: part.iloc[0].to_dict() for sid, part in frame.groupby("sample_id")}
    if not set(ids).issubset(question_rows):
        raise ValueError("Live IDs are missing from supplied generations")
    complete = [sid for sid in ids if (question_directory(output, sid) / "result.json").exists()]
    if len(complete) == 50:
        return summarize_live(output)
    if not torch.cuda.is_available():
        raise RuntimeError("Live experiment requires a CUDA GPU")
    seed_everything(6304)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, torch_dtype=torch.float16,
                                               attn_implementation="sdpa").cuda().eval()
    eos_ids = model.generation_config.eos_token_id
    eos_ids = set(eos_ids if isinstance(eos_ids, list) else [eos_ids])
    eos_ids.add(tokenizer.eos_token_id)
    load_seconds = time.monotonic() - started
    durations = []

    def generate(messages, cap, seed, path):
        if path.exists():
            saved = read_json(path)
            if saved["messages"] != messages or saved["cap"] != cap or saved["seed"] != seed:
                raise ValueError("Partial generation configuration mismatch")
            return saved
        if time.monotonic() - started + 90 >= max_seconds:
            raise TimeoutError("Live-generation chunk budget reached; prior calls are saved")
        seed_everything(seed)
        torch.cuda.synchronize()
        before = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        batch = tokenizer(text, add_special_tokens=False, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            generated = model.generate(**batch, do_sample=True, temperature=.7, top_p=.8, top_k=20,
                                       repetition_penalty=1.0, max_new_tokens=cap,
                                       pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        torch.cuda.synchronize()
        seconds = time.monotonic() - before
        tokens = generated[0, batch.input_ids.shape[1]:].tolist()
        response = tokenizer.decode(tokens, skip_special_tokens=True)
        record = {"messages": messages, "formatted_prompt": text, "seed": seed, "cap": cap,
                  "response": response, "answer": last_box(response), "output_token_ids": tokens,
                  "input_tokens": int(batch.input_ids.shape[1]), "output_tokens": len(tokens),
                  "truncated": len(tokens) == cap and tokens[-1] not in eos_ids,
                  "inference_seconds": seconds, "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
                  "gpu": torch.cuda.get_device_name(0)}
        write_json(path, record)
        commit()
        return record

    interrupted = False
    for index, sample_id in enumerate(ids):
        folder = question_directory(output, sample_id)
        if (folder / "result.json").exists():
            continue
        estimated = max(durations[-3:], default=300)
        if time.monotonic() - started + estimated + 90 >= max_seconds:
            interrupted = True
            break
        folder.mkdir(parents=True, exist_ok=True)
        row = question_rows[sample_id]
        messages = [{"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": str(row["question_text"]) + "\n\n" + INSTRUCTION}]
        before_question = time.monotonic()
        try:
            parallel = [generate(messages, 1024, 6304 + 100 * index + offset,
                                 folder / f"parallel-{offset}.json") for offset in range(3)]
            single = generate(messages, 3072, 6304 + 100 * index + 3, folder / "single.json")
            review_messages = messages + [{"role": "assistant", "content": parallel[0]["response"]},
                                          {"role": "user", "content": REVIEW}]
            revision = generate(review_messages, 2048, 6304 + 100 * index + 4, folder / "revision.json")
        except TimeoutError:
            interrupted = True
            break
        scorer_context("math-live", sample_id)
        groups = answer_groups([r["answer"] for r in parallel], "math")
        voted = votes([{"answer_group": group} for group in groups])[0]
        gold = row["gold_answer"]
        initial_correct = is_correct(groups[0], gold, "math")
        revised_correct = is_correct(revision["answer"], gold, "math")

        def cost(calls):
            return {"input_tokens": sum(r["input_tokens"] for r in calls),
                    "output_tokens": sum(r["output_tokens"] for r in calls),
                    "inference_seconds": sum(r["inference_seconds"] for r in calls),
                    "peak_gpu_gib": max(r["peak_gpu_gib"] for r in calls),
                    "any_call_truncated": any(r["truncated"] for r in calls),
                    "call_truncation_rate": float(np.mean([r["truncated"] for r in calls])),
                    "call_extraction_rate": float(np.mean([r["answer"] is not None for r in calls]))}

        result = {"sample_id": sample_id, "gold": str(gold), "initial_correct": initial_correct,
                  "wrong_to_correct": not initial_correct and revised_correct,
                  "correct_to_wrong": initial_correct and not revised_correct,
                  "strategies": {
                      "single": {"answer": single["answer"], "answer_extracted": single["answer"] is not None,
                                 "correct": is_correct(single["answer"], gold, "math"), **cost([single])},
                      "parallel": {"answer": voted, "answer_extracted": voted != NO_ANSWER,
                                   "correct": is_correct(voted, gold, "math"), **cost(parallel)},
                      "refinement": {"answer": revision["answer"], "answer_extracted": revision["answer"] is not None,
                                     "correct": revised_correct, **cost([parallel[0], revision])}}}
        write_json(folder / "result.json", result)
        write_json(output / "scorer_diagnostics.json", previous_diagnostics + scorer_diagnostics())
        commit()
        durations.append(time.monotonic() - before_question)
        print(f"Live MATH: completed {sample_id} ({index + 1}/50)", flush=True)
    write_json(output / "chunk_status.json", {"interrupted": interrupted, "model_load_seconds": load_seconds,
               "wall_seconds": time.monotonic() - started,
               "completed": sum((question_directory(output, sid) / "result.json").exists() for sid in ids)})
    commit()
    result = summarize_live(output)
    close_scorer()
    return result


def summarize_live(output):
    output = Path(output)
    ids = read_json(output / "config.json")["ids"]
    results = [read_json(question_directory(output, sid) / "result.json") for sid in ids
               if (question_directory(output, sid) / "result.json").exists()]
    summary = {"completed": len(results), "required": 50, "complete": len(results) == 50,
               "wrong_to_correct": sum(r["wrong_to_correct"] for r in results),
               "correct_to_wrong": sum(r["correct_to_wrong"] for r in results), "strategies": {}}
    diagnostics_path = output / "scorer_diagnostics.json"
    diagnostics = read_json(diagnostics_path) if diagnostics_path.exists() else []
    summary["symbolic_failed_comparisons"] = len(diagnostics)
    summary["symbolic_timeout_comparisons"] = sum(row["status"] == "timeout" for row in diagnostics)
    if results:
        for name in ("single", "parallel", "refinement"):
            keys = ("correct", "input_tokens", "output_tokens", "inference_seconds", "peak_gpu_gib",
                    "call_truncation_rate", "call_extraction_rate", "answer_extracted", "any_call_truncated")
            summary["strategies"][name] = {key: float(np.mean([r["strategies"][name][key] for r in results])) for key in keys}
    summary["revision_examples"] = {
        "selection_rule": "first three saved held-out IDs in each observed transition category",
        "wrong_to_correct": [r["sample_id"] for r in results if r["wrong_to_correct"]][:3],
        "correct_to_wrong": [r["sample_id"] for r in results if r["correct_to_wrong"]][:3]}
    write_json(output / "summary.json", summary)
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/task3")
    parser.add_argument("--splits", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seconds", type=int, default=1800)
    args = parser.parse_args()
    print(run_live(args.data, args.splits, args.output, max_seconds=args.seconds))

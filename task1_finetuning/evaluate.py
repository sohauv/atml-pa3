"""Fixed benchmark protocol, qualitative exports, and local weight diagnostics."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common.io import append_jsonl, read_json, write_json
from common.training import seed_everything
from task1_finetuning.adapters import inject, merge_factors

LAYERS = (0, 13, 27)


def positive_batch_size(value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("Likelihood batch size must be a positive integer")
    return value


def task_cache_identity(model, task_name, batch_size, limit):
    return {"schema_version": 2, "task": task_name, "batch_size": batch_size,
            "example_limit": limit, "dtype": "float16",
            "checkpoint_identity": getattr(model, "_pa3_checkpoint_identity", None)}


def validate_completed_task(result, model, task_name, batch_size, limit):
    protocol = result.get("protocol", {})
    # All evaluations produced before batching support used batch size one.
    if protocol.get("batch_size", 1) != batch_size:
        raise ValueError(f"Cached {task_name} batch size differs; use a fresh evaluation directory")
    if protocol.get("example_limit") != limit:
        raise ValueError(f"Cached {task_name} example limit differs")
    expected = task_cache_identity(model, task_name, batch_size, limit)
    if protocol.get("cache_identity", expected) != expected:
        raise ValueError(f"Cached {task_name} model/protocol identity differs")
    return result


def json_value(value):
    if isinstance(value, (torch.dtype, torch.device)):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.tolist()
    return value


def load_model(config, checkpoint=None, diagnostic=False):
    meta = read_json("/data/task1/metadata.json")
    revision = meta["revisions"]["model"]
    tokenizer = AutoTokenizer.from_pretrained(config["model"], revision=revision)
    base = AutoModelForCausalLM.from_pretrained(config["model"], revision=revision,
                                               torch_dtype=torch.float32, attn_implementation="sdpa").cuda()
    identity = {"method": "baseline", "update": 0, "data_sha256": meta["sha256"],
                "model_revision": revision, "checkpoint": str(checkpoint) if checkpoint else None}
    verification = None
    if checkpoint:
        checkpoint = Path(checkpoint)
        if not (checkpoint / "complete.json").exists():
            raise ValueError("Incomplete checkpoint")
        saved = torch.load(checkpoint / "recovery.pt", map_location="cpu", weights_only=False)
        if saved["config"]["model_revision"] != revision or saved["data_sha256"] != meta["sha256"]:
            raise ValueError("Checkpoint provenance mismatch")
        spec = saved["config"]["method_config"]
        identity.update(method=saved["config"]["method"], update=saved["update"])
        if "rank" in spec:
            factors = torch.load(checkpoint / "projected_factors.pt", map_location="cpu", weights_only=True)
            merge_factors(base, factors)
            # Diagnostics retain exact FP32 master weights and avoid an extra
            # inference model. Benchmark jobs verify the actual FP16 export.
            if not diagnostic:
                unmerged = AutoModelForCausalLM.from_pretrained(config["model"], revision=revision,
                                                               torch_dtype=torch.float32, attn_implementation="sdpa").cuda()
                inject(unmerged, spec["rank"], spec["alpha"])
                with torch.no_grad():
                    for name, f in factors.items():
                        layer = unmerged.get_submodule(name)
                        layer.A.copy_(f["A"])
                        layer.B.copy_(f["B"])
                for p in unmerged.parameters():
                    if not p.requires_grad:
                        p.data = p.data.half()
                unmerged.eval()
                base.half().eval()
                batch = tokenizer("A small verification batch.", return_tensors="pt").to("cuda")
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                    a, b = unmerged(**batch).logits.float(), base(**batch).logits.float()
                verification = {"dtype": "float16", "max_absolute_logit_difference": (a - b).abs().max().item(),
                                "mean_absolute_logit_difference": (a - b).abs().mean().item(),
                                "tolerance_atol": .1, "tolerance_rtol": .02}
                if not torch.allclose(a, b, atol=.1, rtol=.02):
                    raise ValueError(f"Adapter merge logits failed: {verification}")
                del unmerged, a, b, batch
        else:
            base.load_state_dict(saved["model"], strict=True)
        del saved
    if not diagnostic:
        base.half()
    base.eval()
    base.config.use_cache = True
    base._pa3_checkpoint_identity = identity
    return base, tokenizer, verification


def harness_model(model, tokenizer, output=None, deadline=None, commit=lambda: None, *, batch_size=1):
    from lm_eval.models.huggingface import HFLM

    class NonThinkingHFLM(HFLM):
        def __init__(self):
            super().__init__(pretrained=model, tokenizer=tokenizer, batch_size=positive_batch_size(batch_size),
                             max_length=4096, dtype="float16")
            self.generation_records = []
            self.generation_cache = {}
            self.generation_cache_identity = {"schema_version": 2, "batch_size": 1,
                "dtype": "float16", "checkpoint_identity": getattr(model, "_pa3_checkpoint_identity", None)}
            cache_path = output / "generation_cache.jsonl" if output is not None else None
            if cache_path is not None and cache_path.exists():
                for line in cache_path.read_text(encoding="utf-8").splitlines():
                    record = json.loads(line)
                    if record.get("cache_identity", self.generation_cache_identity) != self.generation_cache_identity:
                        raise ValueError("Generation cache batch/model identity differs")
                    if record["metadata"].get("batch_size", 1) != 1:
                        raise ValueError("Generation cache must contain batch-one outputs")
                    record["metadata"]["batch_size"] = 1
                    self.generation_cache[record["request_sha256"]] = record

        def apply_chat_template(self, chat_history, add_generation_prompt=True, **kwargs):
            return tokenizer.apply_chat_template(chat_history, tokenize=False,
                                                add_generation_prompt=add_generation_prompt,
                                                enable_thinking=False)

        def _model_generate(self, context, max_length, stop, **kwargs):
            output = super()._model_generate(context, max_length, stop, **kwargs)
            for row in output:
                generated = row[context.shape[1]:].tolist()
                eos = tokenizer.eos_token_id in generated
                if eos:
                    generated = generated[:generated.index(tokenizer.eos_token_id) + 1]
                text = tokenizer.decode(generated, skip_special_tokens=False)
                stopped = eos or any(s and s in text for s in stop)
                self.generation_records.append({"generated_tokens": len(generated),
                    "batch_size": 1,
                    "cap": max_length - context.shape[1],
                    "truncated": len(generated) >= max_length - context.shape[1] and not stopped})
            return output

        def generate_until(self, requests, disable_tqdm=False):
            # Batch size one preserves an unambiguous sample -> truncation mapping.
            outputs = []
            for request in requests:
                if deadline is not None and time.monotonic() >= deadline:
                    commit()
                    raise TimeoutError("Evaluation deadline reached; generation cache committed")
                request_hash = hashlib.sha256(json.dumps(request.args, sort_keys=True).encode()).hexdigest()
                if request_hash in self.generation_cache:
                    record = self.generation_cache[request_hash]
                    outputs.append(record["response"])
                    self.generation_records.append(record["metadata"])
                    continue
                start = len(self.generation_records)
                result = super().generate_until([request], disable_tqdm=True)
                outputs.extend(result)
                records = self.generation_records[start:]
                for record in records:
                    record["prompt_sha256"] = hashlib.sha256(request.args[0].encode()).hexdigest()
                if output is not None:
                    if len(records) != 1 or len(result) != 1:
                        raise ValueError("Expected one generation per request")
                    record = {"request_sha256": request_hash, "response": result[0], "metadata": records[0],
                              "cache_identity": self.generation_cache_identity}
                    append_jsonl(output / "generation_cache.jsonl", record)
                    self.generation_cache[request_hash] = record
                    if len(self.generation_cache) % 25 == 0:
                        commit()
            return outputs

    return NonThinkingHFLM()


def benchmarks(model, tokenizer, output, deadline, *, tasks=None, limit=None,
               likelihood_batch_size=8, commit=lambda: None):
    import lm_eval
    import importlib.metadata
    tasks = tasks or ["gsm8k_cot", "hellaswag", "boolq", "arc_easy"]
    if not set(tasks) <= {"gsm8k_cot", "hellaswag", "boolq", "arc_easy"}:
        raise ValueError("Unknown Task 1 benchmark")
    if limit is not None and (not isinstance(limit, int) or limit < 1):
        raise ValueError("Smoke limit must be a positive example count")
    likelihood_batch_size = positive_batch_size(likelihood_batch_size)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    evaluation_config = {"schema_version": 2, "generation_batch_size": 1,
                         "likelihood_batch_size": likelihood_batch_size,
                         "example_limit": limit, "dtype": "float16", "seed": 6304,
                         "checkpoint_identity": getattr(model, "_pa3_checkpoint_identity", None)}
    config_path = output / "evaluation_config.json"
    if config_path.exists() and read_json(config_path) != evaluation_config:
        raise ValueError("Evaluation cache configuration differs; use a fresh output directory")
    # Validate all existing results before adopting a legacy output directory.
    # This prevents mixing old batch-one likelihood results with new batch-eight results.
    for task_name in ("gsm8k_cot", "hellaswag", "boolq", "arc_easy"):
        completed = output / f"{task_name}.json"
        if completed.exists():
            validate_completed_task(read_json(completed), model, task_name,
                                    1 if task_name == "gsm8k_cot" else likelihood_batch_size, limit)
    if not config_path.exists():
        write_json(config_path, evaluation_config)
    models = {}
    aggregates = {}
    for task_name in tasks:
        completed = output / f"{task_name}.json"
        if completed.exists():
            aggregates.update(read_json(completed)["results"])
            continue
        chat = task_name == "gsm8k_cot"
        batch_size = 1 if chat else likelihood_batch_size
        if time.monotonic() >= deadline:
            raise TimeoutError("Evaluation chunk exhausted; completed task files remain saved")
        if batch_size not in models:
            models[batch_size] = harness_model(model, tokenizer, output, deadline, commit,
                                               batch_size=batch_size)
        lm = models[batch_size]
        result = lm_eval.simple_evaluate(model=lm, tasks=[task_name], num_fewshot=0,
                                        apply_chat_template=chat, batch_size=batch_size,
                                        gen_kwargs="do_sample=False,max_gen_toks=1024" if chat else None,
                                        log_samples=True, random_seed=6304, numpy_random_seed=6304,
                                        torch_random_seed=6304, fewshot_random_seed=6304,
                                        limit=limit)
        samples = result.pop("samples", {})
        for task, records in samples.items():
            for record in records:
                append_jsonl(output / f"{task}.samples.jsonl", json_value(record))
        result["environment"] = {"lm_eval": importlib.metadata.version("lm_eval"),
                                 "transformers": importlib.metadata.version("transformers"),
                                 "torch": torch.__version__}
        result["protocol"] = {"complete_split": limit is None, "smoke_only": limit is not None,
                              "example_limit": limit, "thinking": False,
                              "chat_template": chat, "num_fewshot": 0, "batch_size": batch_size,
                              "cache_identity": task_cache_identity(model, task_name, batch_size, limit)}
        result.setdefault("config", {})["pa3_batch_sizes"] = {
            "generation": 1, "likelihood": likelihood_batch_size}
        write_json(completed, json_value(result))
        aggregates.update(result["results"])
        commit()
    for lm in models.values():
        for record in lm.generation_records:
            append_jsonl(output / "generation_metadata.jsonl", record)
    # Include previously completed tasks when a later call selects a new task.
    for task_name in ("gsm8k_cot", "hellaswag", "boolq", "arc_easy"):
        completed = output / f"{task_name}.json"
        if completed.exists():
            aggregates.update(read_json(completed)["results"])
    write_json(output / "scores.json", {"results": json_value(aggregates),
               "complete_split": limit is None, "smoke_only": limit is not None})
    commit()
    return aggregates


def probe_inputs(tokenizer, output):
    from lm_eval.tasks import TaskManager, get_task_dict
    tasks = get_task_dict(["gsm8k_cot", "hellaswag"], task_manager=TaskManager())
    rng = np.random.default_rng(6304)
    selected = []
    for name, task in tasks.items():
        docs = list(task.test_docs() if name == "gsm8k_cot" else task.validation_docs())
        for index in rng.choice(len(docs), 32, replace=False).tolist():
            text = task.doc_to_text(docs[index])
            if name == "gsm8k_cot":
                text = tokenizer.apply_chat_template([{"role": "user", "content": text}],
                       tokenize=False, add_generation_prompt=True, enable_thinking=False)
            selected.append({"task": name, "doc_id": index, "prompt": text})
    write_json(output / "probe_ids.json", selected)
    return selected


def cache_probes(model, tokenizer, output):
    selected = probe_inputs(tokenizer, output)
    cached = {str(layer): [] for layer in LAYERS}
    handles = []
    for layer in LAYERS:
        def capture(module, inputs, layer=layer):
            cached[str(layer)].append(inputs[0][0, -1].detach().float().cpu())
        handles.append(model.model.layers[layer].mlp.down_proj.register_forward_pre_hook(capture))
    try:
        with torch.no_grad():
            for row in selected:
                batch = tokenizer(row["prompt"], add_special_tokens=False, return_tensors="pt").to("cuda")
                model(**batch)
    finally:
        for handle in handles:
            handle.remove()
    torch.save({k: torch.stack(v) for k, v in cached.items()}, output / "activations.pt")
    return {"probes": len(selected), "path": str(output)}


@torch.no_grad()
def diagnostics(model, config, output, probes_dir):
    meta = read_json("/data/task1/metadata.json")
    original = AutoModelForCausalLM.from_pretrained(config["model"], revision=meta["revisions"]["model"],
                                                   torch_dtype=torch.float32).cuda().eval()
    probes = torch.load(Path(probes_dir) / "activations.pt", map_location="cuda", weights_only=True)
    ids = read_json(Path(probes_dir) / "probe_ids.json")
    result = []
    for layer in LAYERS:
        w0 = original.model.layers[layer].mlp.down_proj.weight.float()
        w = model.model.layers[layer].mlp.down_proj.weight.float()
        delta = w - w0
        u, _, _ = torch.linalg.svd(w0, full_matrices=False)
        norm = delta.square().sum()
        z = probes[str(layer)]
        correction = torch.linalg.vector_norm(z @ delta.T, dim=1)
        baseline = torch.linalg.vector_norm(z @ w0.T, dim=1)
        ratio = (correction / (baseline + 1e-8)).tolist()
        result.append({"layer": layer, "rho_128": ((u[:, :128].T @ delta).square().sum() / norm).item() if norm > 0 else None,
                       "eta": (norm.sqrt() / torch.linalg.vector_norm(w0)).item(),
                       "activation_ratios": [{**row, "ratio": value} for row, value in zip(ids, ratio)]})
    write_json(output / "diagnostics.json", result)
    return result


def qualitative(model, tokenizer, output, deadline, commit=lambda: None):
    import yaml
    source = Path("/opt/pa3/configs/task1_finetuning/first_plot_questions.yaml")
    questions = yaml.safe_load(source.read_text(encoding="utf-8"))
    if len(questions) != 24:
        raise ValueError("Expected all 24 assignment prompts")
    plain_ids = [row["id"] for row in questions if not row["id"].endswith(("_json", "_template"))]
    annotate_ids = {plain_ids[0], plain_ids[-1]}
    completed = set()
    path = output / "qualitative.jsonl"
    if path.exists():
        completed = {json.loads(line)["id"] for line in path.read_text(encoding="utf-8").splitlines()}
    for row in questions:
        if row["id"] in completed:
            continue
        if time.monotonic() >= deadline:
            raise TimeoutError("Qualitative chunk exhausted")
        if len(row["paraphrases"]) != 1:
            raise ValueError("Unexpected prompt schema")
        messages = []
        if row.get("system"):
            messages.append({"role": "system", "content": row["system"]})
        messages.append({"role": "user", "content": row["paraphrases"][0]})
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        batch = tokenizer(text, add_special_tokens=False, return_tensors="pt").to("cuda")
        with torch.no_grad():
            response = model.generate(**batch, do_sample=False, max_new_tokens=512, pad_token_id=tokenizer.pad_token_id)
        ids = response[0, batch.input_ids.shape[1]:].tolist()
        underlying = row["id"].removesuffix("_json").removesuffix("_template")
        format_name = "json" if row["id"].endswith("_json") else "template" if row["id"].endswith("_template") else "plain"
        append_jsonl(output / "qualitative.jsonl", {"id": row["id"], "messages": messages,
                     "question_id": underlying, "format": format_name,
                     "annotation_subset": underlying in annotate_ids,
                     "response": tokenizer.decode(ids, skip_special_tokens=True), "tokens": len(ids),
                     "truncated": len(ids) == 512 and ids[-1] != tokenizer.eos_token_id})
        commit()
    return {"responses": 24}


def evaluate_job(request, output, seconds, commit=lambda: None):
    seed_everything(6304)
    likelihood_batch_size = positive_batch_size(request.get("eval_likelihood_batch_size", 8))
    if output.exists():
        previous = read_json(output / "request.json")
        for key in ("job", "config", "checkpoint", "eval_limit", "probes_dir"):
            if previous.get(key) != request.get(key):
                raise ValueError(f"Evaluation resume protocol mismatch: {key}")
        if (request["job"] == "evaluate1" and
                positive_batch_size(previous.get("eval_likelihood_batch_size", 1)) != likelihood_batch_size):
            raise ValueError("Evaluation resume likelihood batch size differs; use a fresh output directory")
    else:
        output.mkdir(parents=True)
        write_json(output / "request.json", request)
    deadline = time.monotonic() + seconds - 60
    model, tokenizer, merge = load_model(request["config"], request.get("checkpoint"),
                                         diagnostic=request["job"] in ("probe1", "diagnostics1"))
    condition_path = output / "condition.json"
    if condition_path.exists() and read_json(condition_path) != model._pa3_checkpoint_identity:
        raise ValueError("Saved evaluation condition does not match the loaded checkpoint")
    if not condition_path.exists():
        write_json(condition_path, model._pa3_checkpoint_identity)
    if merge:
        write_json(output / "merge_verification.json", merge)
    job = request["job"]
    if job == "evaluate1":
        result = benchmarks(model, tokenizer, output, deadline, tasks=request.get("eval_tasks"),
                            limit=request.get("eval_limit"), likelihood_batch_size=likelihood_batch_size,
                            commit=commit)
    elif job == "probe1":
        if request.get("checkpoint"):
            raise ValueError("Probes must be cached from the unchanged starting model")
        result = cache_probes(model, tokenizer, output)
    elif job == "qualitative1":
        result = qualitative(model, tokenizer, output, deadline, commit)
    elif job == "diagnostics1":
        if not request.get("probes_dir"):
            raise ValueError("Diagnostics requires --probes-dir pointing to saved baseline probes")
        result = diagnostics(model, request["config"], output, request["probes_dir"])
    else:
        raise ValueError(job)
    commit()
    return {"output": str(output), "result": result}

"""Task 1 training in bounded, resumable chunks; no remote calls on import."""
import json
from pathlib import Path
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common.io import append_jsonl, read_json, sha256, write_json
from common.training import (causal_loss, collate, lr_factor, parameter_groups,
                             restore_rng, rng_state, seed_everything)
from task1_finetuning.adapters import export_factors, inject


def load_records(data_dir, expected=40000):
    data_dir = Path(data_dir)
    meta = read_json(data_dir / "metadata.json")
    if sha256(data_dir / "train.jsonl") != meta["sha256"]:
        raise ValueError("Training data checksum mismatch")
    with (data_dir / "train.jsonl").open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    if len(rows) != expected or len({r["source_id"] for r in rows}) != expected:
        raise ValueError("Incorrect example count or duplicate source IDs")
    if [r["source_id"] for r in rows] != sorted(r["source_id"] for r in rows):
        raise ValueError("Training data is out of source order")
    for row in rows:
        if len(row["input_ids"]) != len(row["labels"]) or len(row["labels"]) > 2048:
            raise ValueError("Invalid training sequence")
        if not any(x != -100 for x in row["labels"][1:]):
            raise ValueError("No causal response targets")
    return rows, meta


def save_checkpoint(model, optimizer, scaler, path, *, update, config,
                    metadata, wall_seconds, reference, commit,
                    previous_peak_gib=0.0, skipped_updates=0):
    checkpoint_started = time.monotonic()
    path = Path(path)
    # Never overwrite a checkpoint, even on resume or an interrupted prior write.
    if path.exists():
        raise FileExistsError(path)
    path.mkdir(parents=True)
    adapter = "rank" in config["method_config"]
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()
             if not adapter or k.endswith((".A", ".B", ".U", ".V"))}
    peak_gpu_gib = max(previous_peak_gib, torch.cuda.max_memory_allocated() / 2**30)
    torch.save({"model": state, "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(), "rng": rng_state(),
                "update": update, "data_position": update * 32,
                "scheduler": {"next_update": update},
                "config": config, "data_sha256": metadata["sha256"],
                "wall_seconds": wall_seconds, "peak_gpu_gib": peak_gpu_gib,
                "skipped_updates": skipped_updates}, path / "recovery.pt")
    export_started = time.monotonic()
    if adapter:
        torch.save(export_factors(model), path / "projected_factors.pt")
    else:
        model.save_pretrained(path / "model", safe_serialization=True)
    export_seconds = time.monotonic() - export_started
    displacement = None
    if reference is not None:
        with torch.no_grad():
            distance = sum((p.float() - reference[n]).square().sum()
                           for n, p in model.named_parameters())
        displacement = distance.sqrt().item()
    write_json(path / "summary.json", {
        "update": update, "examples": update * 32,
        "wall_seconds": wall_seconds + time.monotonic() - checkpoint_started,
        "checkpoint_write_seconds": time.monotonic() - checkpoint_started,
        "inference_export_seconds": export_seconds,
        "skipped_updates": skipped_updates,
        "successful_optimizer_updates": update - skipped_updates,
        "displacement_l2": displacement,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "peak_gpu_gib": peak_gpu_gib,
        "checkpoint_bytes": sum(p.stat().st_size for p in path.rglob("*") if p.is_file()),
    })
    # A completion marker is written last. Readers ignore partial checkpoints.
    write_json(path / "complete.json", {"update": update})
    commit()


def train(method, run_dir, config, *, data_dir="/data/task1", resume=None,
          max_seconds=1800, max_updates=None, commit=lambda: None):
    started = time.monotonic()
    if method not in config["methods"]:
        raise ValueError(method)
    if config["microbatch"] != 2 or config["accumulation"] != 16:
        raise ValueError("Task 1 requires microbatch 2, accumulation 16")
    run_dir = Path(run_dir)
    if resume is not None and not Path(resume).resolve().is_relative_to(run_dir.resolve()):
        raise ValueError("Resume checkpoint must belong to the original run directory")
    if resume is None and run_dir.exists():
        raise FileExistsError("New runs must use a fresh run directory")
    run_dir.mkdir(parents=True, exist_ok=resume is not None)
    if not torch.cuda.is_available():
        raise RuntimeError("FP16 experiment requires a CUDA GPU")
    # Include preprocessing (notably OPLoRA's SVD) in the method's peak memory.
    torch.cuda.reset_peak_memory_stats()
    seed_everything(config["seed"])
    rows, meta = load_records(data_dir, config["examples"])
    spec = config["methods"][method]
    resolved = {**config, "method": method, "method_config": spec,
                "model_revision": meta["revisions"]["model"]}
    saved = None
    if resume:
        resume = Path(resume)
        if not (resume / "complete.json").exists():
            raise ValueError("Cannot resume an incomplete checkpoint")
        saved = torch.load(resume / "recovery.pt", map_location="cpu", weights_only=False)
        if saved["config"] != resolved or saved["data_sha256"] != meta["sha256"]:
            raise ValueError("Resume config/data mismatch")
        if saved["data_position"] != saved["update"] * 32 or saved["scheduler"]["next_update"] != saved["update"]:
            raise ValueError("Resume scheduler/data position mismatch")
    tokenizer = AutoTokenizer.from_pretrained(config["model"], revision=resolved["model_revision"])
    model = AutoModelForCausalLM.from_pretrained(
        config["model"], revision=resolved["model_revision"],
        torch_dtype=torch.float32,
        attn_implementation="sdpa").cuda()
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    reference = None
    reference_started = time.monotonic()
    if method in ("full", "l2sp"):
        reference = {n: p.detach().clone() for n, p in model.named_parameters()}
    torch.cuda.synchronize()
    reference_seconds = time.monotonic() - reference_started
    preprocessing = time.monotonic()
    if "rank" in spec:
        inject(model, spec["rank"], spec["alpha"], spec.get("protection_rank", 0),
               basis_state=saved["model"] if saved is not None else None)
        # Compute bases from original FP32 starting weights, then round only the
        # frozen backbone to FP16. Adapter parameters and bases remain FP32.
        for parameter in model.parameters():
            if not parameter.requires_grad:
                parameter.data = parameter.data.half()
    torch.cuda.synchronize()
    preprocessing_seconds = time.monotonic() - preprocessing
    optimizer = torch.optim.AdamW(parameter_groups(model, config["weight_decay"]),
                                 lr=spec["lr"], betas=tuple(config["betas"]), eps=config["eps"])
    scaler = torch.amp.GradScaler("cuda")
    update, previous_seconds, previous_peak_gib, skipped_updates = 0, 0.0, 0.0, 0
    if saved is not None:
        incompatible = model.load_state_dict(saved["model"], strict="rank" not in spec)
        if incompatible.unexpected_keys:
            raise ValueError(incompatible.unexpected_keys)
        missing_adapter = [name for name in incompatible.missing_keys if name.endswith((".A", ".B", ".U", ".V"))]
        if missing_adapter:
            raise ValueError(f"Checkpoint missing adapter state: {missing_adapter}")
        optimizer.load_state_dict(saved["optimizer"])
        # Optimizer state follows parameter placement and remains FP32.
        for p, state in optimizer.state.items():
            for key, value in state.items():
                if isinstance(value, torch.Tensor) and key != "step":
                    state[key] = value.to(device=p.device, dtype=torch.float32)
        scaler.load_state_dict(saved["scaler"])
        restore_rng(saved["rng"])
        update = saved["update"]
        previous_seconds = read_json(resume / "summary.json")["wall_seconds"]
        previous_peak_gib = saved.get("peak_gpu_gib", 0.0)
        skipped_updates = saved.get("skipped_updates", 0)
        del saved
    else:
        write_json(run_dir / "config.json", resolved)
        write_json(run_dir / "provenance.json", {"data": meta,
                   "preprocessing_seconds": preprocessing_seconds,
                   "reference_storage_seconds": reference_seconds,
                   "gpu": torch.cuda.get_device_name(0),
                   "torch": torch.__version__,
                   "reference_bytes": sum(p.numel() * p.element_size() for p in (reference or {}).values()),
                   "basis_bytes": sum(b.numel() * b.element_size() for n, b in model.named_buffers()
                                      if n.endswith((".U", ".V")))})
    model.train()
    end_update = min(config["updates"], update + max_updates) if max_updates else config["updates"]
    last_saved = update
    last_checkpoint = Path(resume) if resume else None
    durations = []
    while update < end_update:
        elapsed = time.monotonic() - started
        estimate = max(durations[-4:], default=60)
        # Leave time to serialize and commit. Chunk duration includes model loading.
        if elapsed + estimate + 120 >= max_seconds:
            break
        step_started = time.monotonic()
        group = rows[update * 32:(update + 1) * 32]
        if len(group) != 32:
            raise ValueError("Incomplete effective batch")
        tokens = sum(sum(x != -100 for x in row["labels"][1:]) for row in group)
        lr = spec["lr"] * lr_factor(update, config["updates"], config["warmup_updates"])
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for offset in range(0, 32, 2):
            batch, labels = collate(group[offset:offset + 2], tokenizer.pad_token_id)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = causal_loss(model(**batch).logits, labels)
            scaler.scale(loss / tokens).backward()
            loss_sum += loss.detach().item()
            del loss, batch, labels
        omega = 0.0
        if spec.get("lambda", 0):
            # Backpropagate parameterwise to avoid a second model-sized autograd graph.
            for name, p in model.named_parameters():
                penalty = .5 * (p - reference[name]).square().sum()
                omega += penalty.detach().item()
                scaler.scale(spec["lambda"] * penalty).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], config["clip_norm"])
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        skipped = scaler.get_scale() < old_scale
        skipped_updates += int(skipped)
        torch.cuda.synchronize()
        update += 1
        duration = time.monotonic() - step_started
        durations.append(duration)
        append_jsonl(run_dir / "train.jsonl", {
            "update": update, "examples": update * 32, "sft_loss": loss_sum / tokens,
            "omega": omega, "lambda_omega": spec.get("lambda", 0) * omega,
            "total_loss": loss_sum / tokens + spec.get("lambda", 0) * omega,
            "lr": lr, "grad_norm": grad_norm.item() if torch.isfinite(grad_norm) else None,
            "skipped": skipped, "scale": scaler.get_scale(), "step_seconds": duration,
            "supervised_tokens": tokens})
        if update in config["checkpoints"]:
            checkpoint_path = run_dir / f"step-{update:04d}"
            if checkpoint_path.exists():
                # Retain partial snapshots after an interrupted write. A new
                # immutable snapshot at the same update is safe to resume.
                checkpoint_path = run_dir / f"step-{update:04d}-{time.time_ns()}"
            save_checkpoint(model, optimizer, scaler, checkpoint_path,
                            update=update, config=resolved, metadata=meta,
                            wall_seconds=previous_seconds + time.monotonic() - started,
                            reference=reference, commit=commit,
                            previous_peak_gib=previous_peak_gib, skipped_updates=skipped_updates)
            last_saved = update
            last_checkpoint = checkpoint_path
    if update != last_saved or last_checkpoint is None:
        # Extra recovery checkpoints do not add evaluation conditions.
        checkpoint = run_dir / f"resume-{update:04d}-{time.time_ns()}"
        save_checkpoint(model, optimizer, scaler, checkpoint, update=update,
                        config=resolved, metadata=meta,
                        wall_seconds=previous_seconds + time.monotonic() - started,
                        reference=reference, commit=commit,
                        previous_peak_gib=previous_peak_gib, skipped_updates=skipped_updates)
    else:
        checkpoint = last_checkpoint
    result = {"method": method, "update": update, "finished": update == config["updates"],
              "checkpoint": str(checkpoint) if checkpoint else None,
              "seconds": time.monotonic() - started}
    write_json(run_dir / "status.json", result)
    commit()
    return result

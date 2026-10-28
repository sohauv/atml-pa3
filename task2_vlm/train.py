"""Bounded three-stage VLM training with exact pair sampling and resume state."""
from collections import Counter
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from common.io import append_jsonl, read_json, sha256, write_json
from common.training import causal_loss, lr_factor, restore_rng, rng_state, seed_everything
from task2_vlm.data import validate_isolation
from task2_vlm.evaluate import DevelopmentDeadline, development, final_evaluation
from task2_vlm.model import VisualLanguageModel, load_feature


class CurriculumSampler:
    def __init__(self, manifest, seed=6304):
        self.manifest = manifest
        self.rng = np.random.default_rng(seed)
        self.caption_position = 0
        self.mixture = Counter()

    def state_dict(self):
        return {"rng": self.rng.bit_generator.state, "caption_position": self.caption_position,
                "mixture": dict(self.mixture)}

    def load_state_dict(self, value):
        self.rng.bit_generator.state = value["rng"]
        self.caption_position = value["caption_position"]
        self.mixture = Counter(value["mixture"])

    def batch(self, stage):
        captions = self.manifest["captions_train"]
        if stage == 0:
            if len(captions) != 18000:
                raise ValueError("Caption alignment requires exactly 18000 examples")
            rows = [captions[(self.caption_position + i) % len(captions)] for i in range(16)]
            self.caption_position += 16
            self.mixture["caption"] += 16
            return rows
        names = ["presence", "counting", "caption"] if stage == 1 else ["natural", "presence", "counting", "caption"]
        probabilities = [.5, .25, .25] if stage == 1 else [.7, .1, .1, .1]
        rows = []
        for _ in range(8):
            kind = str(self.rng.choice(names, p=probabilities))
            if kind == "caption":
                chosen = self.rng.integers(0, len(captions), size=2)
                pair_rows = [captions[int(i)] for i in chosen]
            else:
                pool = self.manifest[f"{kind}_train"]
                pair_rows = pool[int(self.rng.integers(len(pool)))]["examples"]
                if len(pair_rows) != 2:
                    raise ValueError("Every curriculum draw must contain two examples")
            rows.extend(pair_rows)
            self.mixture[kind] += 2
        return rows


def stage_at(update, stages):
    start = 0
    for index, stage in enumerate(stages):
        if update < start + stage["updates"]:
            return index, update - start
        start += stage["updates"]
    return len(stages) - 1, stages[-1]["updates"]


def build_model(config, model_revision):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["language_model"], revision=model_revision)
    backbone = AutoModelForCausalLM.from_pretrained(config["language_model"], revision=model_revision,
                torch_dtype=torch.float16, attn_implementation="sdpa").cuda()
    backbone.config.use_cache = False
    backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return VisualLanguageModel(backbone, tokenizer, config)


def optimizer_for(model, config):
    groups = [{"params": list(model.connector.parameters()), "name": "connector", "lr": .001}]
    if model.adapters_enabled:
        adapters = [p for p in model.language_model.parameters() if p.requires_grad]
        groups.append({"params": adapters, "name": "adapter", "lr": 2e-5})
    return torch.optim.AdamW(groups, betas=(.9, .999), eps=1e-8, weight_decay=config["weight_decay"])


def trainable_state(model):
    names = {name for name, p in model.named_parameters() if p.requires_grad}
    return {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()
            if name in names or name == "connector_scale"}


def save_checkpoint(model, optimizer, scaler, sampler, output, *, update, config,
                    identity, seconds, commit):
    if output.exists():
        if (output / "complete.json").exists():
            raise FileExistsError("Refusing to replace a complete checkpoint")
        # An interrupted serialization may leave the required named path. Keep
        # it intact and publish the retry under a discoverable unique suffix.
        output = output.with_name(output.name + f"-retry-{time.time_ns()}")
    output.mkdir(parents=True)
    torch.save({"model": trainable_state(model), "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(), "rng": rng_state(), "sampler": sampler.state_dict(),
                "update": update, "stage": stage_at(update, config["stages"]),
                "scheduler": {"next_update": update}, "adapters_enabled": model.adapters_enabled,
                "config": config, "identity": identity, "seconds": seconds}, output / "recovery.pt")
    torch.save(trainable_state(model), output / "connector_adapters.pt")
    write_json(output / "complete.json", {"update": update})
    append_jsonl(output.parent / "checkpoint_index.jsonl", {"update": update, "path": str(output)})
    commit()
    return output


def load_checkpoint(model, checkpoint, config, identity, optimizer=True):
    checkpoint = Path(checkpoint)
    if not (checkpoint / "complete.json").exists():
        raise ValueError("Incomplete VLM checkpoint")
    saved = torch.load(checkpoint / "recovery.pt", map_location="cpu", weights_only=False)
    if saved["config"] != config or saved["identity"] != identity:
        raise ValueError("VLM resume configuration/provenance mismatch")
    if saved["adapters_enabled"]:
        model.enable_adapters()
    invalid = model.load_state_dict(saved["model"], strict=False)
    if invalid.unexpected_keys:
        raise ValueError(invalid.unexpected_keys)
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    if any(name in trainable or name == "connector_scale" for name in invalid.missing_keys):
        raise ValueError("Checkpoint lacks trainable parameters")
    return saved


def has_development(run, update):
    path = Path(run) / "development.jsonl"
    if not path.exists():
        return False
    with path.open(encoding="utf-8") as stream:
        return any(json.loads(line)["update"] == update for line in stream if line.strip())


def train(config, manifest_path, feature_dir, model_revision, run_dir, *, resume=None,
          max_seconds=1800, max_updates=None, commit=lambda: None):
    if not torch.cuda.is_available():
        raise RuntimeError("VLM main experiment requires a CUDA GPU")
    if config["microbatch"] != 8 or config["accumulation"] != 2:
        raise ValueError("Manual requires microbatch8/accumulation2")
    if max_updates is not None and max_updates <= 0:
        raise ValueError("max_updates must be positive")
    started = time.monotonic()
    seed_everything(config["seed"])
    run = Path(run_dir)
    if run.exists() and resume is None:
        raise FileExistsError("Use a fresh VLM run directory")
    run.mkdir(parents=True, exist_ok=resume is not None)
    manifest = read_json(manifest_path)
    isolation = validate_isolation(manifest)
    identity = {"manifest_sha256": sha256(manifest_path), "model_revision": model_revision,
                "feature_provenance": read_json(Path(feature_dir) / "provenance.json")}
    model = build_model(config, model_revision)
    sampler = CurriculumSampler(manifest, config["seed"])
    scaler = torch.amp.GradScaler("cuda")
    update, previous_seconds = 0, 0.0
    if resume:
        saved = load_checkpoint(model, resume, config, identity)
        optimizer = optimizer_for(model, config)
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        sampler.load_state_dict(saved["sampler"])
        restore_rng(saved["rng"])
        update, previous_seconds = saved["update"], saved["seconds"]
    else:
        optimizer = optimizer_for(model, config)
        calibration_rows = manifest["captions_train"][:32]
        calibration_features = [load_feature(row, feature_dir, model.device) for row in calibration_rows]
        calibration = model.calibrate(calibration_rows, calibration_features)
        write_json(run / "config.json", config)
        write_json(run / "provenance.json", {**identity, "isolation": isolation, "calibration": calibration})
        # Preserve the calibrated untrained state before the first development
        # sweep, so an interrupted sweep can resume without changing calibration.
        initial_checkpoint = save_checkpoint(model, optimizer, scaler, sampler, run / "step-0000", update=0,
                        config=config, identity=identity, seconds=time.monotonic() - started, commit=commit)
    total = sum(stage["updates"] for stage in config["stages"])
    stop = min(total, update + max_updates) if max_updates else total
    boundaries = np.cumsum([stage["updates"] for stage in config["stages"]]).tolist()
    development_pending = False
    if (update == 0 or update % config["development_every"] == 0 or update in boundaries) and not has_development(run, update):
        try:
            development(model, manifest, feature_dir, update=update, output_dir=run,
                        deadline=started + max_seconds - 90, commit=commit)
        except DevelopmentDeadline:
            development_pending = True
    last_saved, step_durations = update, []
    last_checkpoint = Path(resume) if resume else initial_checkpoint
    model.train()
    torch.cuda.reset_peak_memory_stats()
    while update < stop and not development_pending:
        if time.monotonic() - started + max(step_durations[-4:], default=60) + 120 >= max_seconds:
            break
        step_start = time.monotonic()
        stage_index, stage_update = stage_at(update, config["stages"])
        stage = config["stages"][stage_index]
        if stage_index >= 1 and not model.adapters_enabled:
            model.enable_adapters()
            optimizer.add_param_group({"params": [p for p in model.language_model.parameters() if p.requires_grad],
                                       "name": "adapter", "lr": stage["adapter_lr"]})
        factor = lr_factor(stage_update, stage["updates"], math.ceil(.05 * stage["updates"]), floor=.1)
        for group in optimizer.param_groups:
            group["lr"] = stage[f'{group["name"]}_lr'] * factor
        rows = sampler.batch(stage_index)
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for offset in (0, 8):
            micro = rows[offset:offset + 8]
            features = [load_feature(row, feature_dir, model.device) for row in micro]
            with torch.autocast("cuda", dtype=torch.float16):
                batch, labels = model.batch(micro, features)
                loss = causal_loss(model(**batch).logits, labels, reduction="examples") / 16
            scaler.scale(loss).backward()
            loss_sum += loss.detach().item()
            del loss, batch, labels, features
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], config["clip_norm"])
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()
        update += 1
        duration = time.monotonic() - step_start
        step_durations.append(duration)
        append_jsonl(run / "train.jsonl", {"update": update, "stage": stage_index + 1,
            "stage_update": stage_update + 1, "loss": loss_sum, "examples": update * 16,
            "mixture_counts": dict(sampler.mixture), "step_seconds": duration,
            "grad_norm": norm.item() if torch.isfinite(norm) else None,
            "skipped": scaler.get_scale() < old_scale,
            "learning_rates": {group["name"]: group["lr"] for group in optimizer.param_groups},
            "trainable_parameters": {"connector": sum(p.numel() for p in model.connector.parameters()),
                 "adapter": sum(p.numel() for p in model.language_model.parameters() if p.requires_grad)},
            "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30})
        if update % config["checkpoint_every"] == 0 or update in boundaries:
            last_checkpoint = save_checkpoint(model, optimizer, scaler, sampler, run / f"step-{update:04d}",
                update=update, config=config, identity=identity,
                seconds=previous_seconds + time.monotonic() - started, commit=commit)
            last_saved = update
        if update % config["development_every"] == 0 or update in boundaries:
            try:
                development(model, manifest, feature_dir, update=update, output_dir=run,
                            deadline=started + max_seconds - 90, commit=commit)
            except DevelopmentDeadline:
                development_pending = True
    if update != last_saved:
        checkpoint = run / f"resume-{update:04d}-{time.time_ns()}"
        checkpoint = save_checkpoint(model, optimizer, scaler, sampler, checkpoint, update=update, config=config,
                        identity=identity, seconds=previous_seconds + time.monotonic() - started, commit=commit)
    else:
        checkpoint = last_checkpoint
    result = {"update": update, "finished": update == total and not development_pending,
              "training_finished": update == total, "development_pending": development_pending,
              "checkpoint": str(checkpoint) if checkpoint else None,
              "seconds": time.monotonic() - started}
    write_json(run / "status.json", result)
    commit()
    return result


def evaluate(config, manifest_path, feature_dir, model_revision, checkpoint, output_dir,
             *, max_seconds=1800, commit=lambda: None):
    seed_everything(config["seed"])
    model = build_model(config, model_revision)
    identity = {"manifest_sha256": sha256(manifest_path), "model_revision": model_revision,
                "feature_provenance": read_json(Path(feature_dir) / "provenance.json")}
    saved = load_checkpoint(model, checkpoint, config, identity)
    if saved["update"] != sum(stage["updates"] for stage in config["stages"]):
        raise ValueError("Final-test evaluation is restricted to completed Stage3 model")
    model.language_model.gradient_checkpointing_disable()
    model.language_model.config.use_cache = True
    return final_evaluation(model, read_json(manifest_path), feature_dir, output_dir,
                            max_seconds=max_seconds,
                            identity={**identity, "checkpoint_sha256": sha256(Path(checkpoint) / "recovery.pt"),
                                      "config": config}, commit=commit)

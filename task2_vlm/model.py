"""SigLIP feature cache and Qwen visual interface; no vision encoder in training."""
import gc
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time

import torch
from torch import nn

from common.io import read_json, sha256, write_json


def load_cache_image(image_root, row):
    from PIL import Image
    path = Path(image_root) / row["split"] / row["image_file"]
    with Image.open(path) as original:
        return original.convert("RGB")


def save_feature(path, value):
    """Save one independent tensor, never its parent batch storage."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(path.name + f".tmp-{time.time_ns()}")
    # A contiguous batch slice is still a view. torch.save otherwise includes
    # the entire batch's storage in every image file (32x duplication).
    isolated = value.detach().cpu().clone(memory_format=torch.contiguous_format)
    torch.save(isolated, temporary)
    if temporary.stat().st_size > 2 * isolated.numel() * isolated.element_size():
        raise RuntimeError("Feature serialization unexpectedly includes excess storage")
    if path.exists():
        raise FileExistsError(path)
    temporary.rename(path)


def cache_features(manifest_path, image_root, output_dir, revision, *, batch_size=32,
                   device="cuda", max_seconds=None, commit=lambda: None):
    """Cache final normalized patch outputs exactly once per image.

    Transformers 4.51.3 SiglipVisionTransformer already applies post_layernorm
    before returning last_hidden_state. Do not apply it again here.
    """
    from transformers import AutoImageProcessor, SiglipVisionModel
    started = time.monotonic()
    manifest = read_json(manifest_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    provenance = output / "provenance.json"
    identity = {"model": "google/siglip-base-patch16-224", "revision": revision,
                "manifest_sha256": sha256(manifest_path), "dtype": "float16",
                "shape": [196, 768], "normalization": "model post_layernorm once",
                "storage": "independent per-image tensor v2",
                "preprocessing": "pinned AutoImageProcessor deterministic resize/rescale/normalize"}
    if provenance.exists() and read_json(provenance) != identity:
        raise ValueError("Feature cache provenance mismatch")
    write_json(provenance, identity)
    unique = {}
    for items in manifest.values():
        for item in items:
            for row in item.get("examples", [item]):
                key = f'{row["split"]}-{row["image_id"]}'
                unique[key] = row
    pending = []
    for key, row in sorted(unique.items()):
        path = output / f"{key}.pt"
        if path.exists():
            value = torch.load(path, map_location="cpu", weights_only=True)
            if value.shape != (196, 768) or value.dtype != torch.float16 or not torch.isfinite(value).all():
                raise ValueError(f"Existing feature cache is invalid: {path}")
        else:
            pending.append((key, row))
    processor = AutoImageProcessor.from_pretrained(identity["model"], revision=revision)
    encoder = SiglipVisionModel.from_pretrained(identity["model"], revision=revision,
                                                torch_dtype=torch.float32).to(device).eval().requires_grad_(False)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    complete = 0
    timings = {key: 0.0 for key in ("input_seconds", "image_read_seconds", "preprocess_seconds",
                                   "vision_seconds", "write_seconds", "commit_seconds")}
    with ThreadPoolExecutor(max_workers=8) as image_pool, torch.no_grad():
        for offset in range(0, len(pending), batch_size):
            if max_seconds is not None and time.monotonic() - started > max_seconds - 90:
                break
            group = pending[offset:offset + batch_size]
            phase_started = time.monotonic()
            # map preserves manifest order while overlapping remote-volume I/O.
            images = list(image_pool.map(lambda item: load_cache_image(image_root, item[1]), group))
            timings["image_read_seconds"] += time.monotonic() - phase_started
            preprocess_started = time.monotonic()
            pixels = processor(images=images, return_tensors="pt")["pixel_values"].to(device)
            timings["preprocess_seconds"] += time.monotonic() - preprocess_started
            timings["input_seconds"] += time.monotonic() - phase_started
            phase_started = time.monotonic()
            features = encoder(pixel_values=pixels).last_hidden_state
            if features.shape[1:] != (196, 768) or not torch.isfinite(features).all():
                raise ValueError("Unexpected SigLIP patch features")
            features_cpu = features.half().cpu()
            timings["vision_seconds"] += time.monotonic() - phase_started
            phase_started = time.monotonic()
            for (key, _), value in zip(group, features_cpu):
                path = output / f"{key}.pt"
                save_feature(path, value)
            timings["write_seconds"] += time.monotonic() - phase_started
            complete += len(group)
            if complete % 128 == 0 or offset + batch_size >= len(pending):
                write_json(output / "progress.json", {
                    "new_images": complete, "existing_images": len(unique) - len(pending),
                    "total_images": len(unique), "seconds": time.monotonic() - started,
                    **timings})
                phase_started = time.monotonic()
                commit()
                timings["commit_seconds"] += time.monotonic() - phase_started
                print(f"[Task2 cache] {complete}/{len(pending)} new images; timings={timings}", flush=True)
    result = {"new_images": complete, "total_images": len(unique),
              "finished": complete == len(pending), "seconds": time.monotonic() - started,
              "timings": timings,
              "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else None}
    write_json(output / f"cache-run-{time.time_ns()}.json", result)
    del encoder, processor
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    commit()
    return result


def load_feature(row, feature_dir, device):
    path = Path(feature_dir) / f'{row["split"]}-{row["image_id"]}.pt'
    value = torch.load(path, map_location="cpu", weights_only=True)
    if value.shape != (196, 768) or value.dtype != torch.float16:
        raise ValueError(f"Invalid cached feature: {path}")
    return value.to(device)


class VisualLanguageModel(nn.Module):
    def __init__(self, language_model, tokenizer, config):
        super().__init__()
        self.language_model = language_model
        self.tokenizer = tokenizer
        self.settings = config
        if config.get("visual_tokens", 196) not in (196, 49):
            raise ValueError("Visual token count must be 196 (main) or 49 (optional repeat)")
        self.language_model.requires_grad_(False)
        self.connector = nn.Sequential(nn.Linear(768, 1024), nn.GELU(), nn.Linear(1024, 1024)).to(
            device=language_model.get_input_embeddings().weight.device, dtype=torch.float32)
        self.register_buffer("connector_scale", torch.tensor(1.0, device=self.connector[0].weight.device))
        if language_model.config.hidden_size != 1024:
            raise ValueError("Assignment connector requires Qwen hidden size 1024")
        marker = "ATML_VISUAL_EMBEDDINGS_PLACEHOLDER_6304"
        text = tokenizer.apply_chat_template([{"role": "user", "content": marker}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False)
        if text.count(marker) != 1:
            raise ValueError("Chat template does not preserve insertion marker")
        left, right = text.split(marker)
        self.prefix_ids = tokenizer(left, add_special_tokens=False)["input_ids"]
        self.suffix_ids = tokenizer(right, add_special_tokens=False)["input_ids"]
        self.end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.adapters_enabled = False

    @property
    def device(self):
        return self.connector[0].weight.device

    def enable_adapters(self):
        if self.adapters_enabled:
            return
        from task1_finetuning.adapters import inject
        inject(self.language_model, rank=16, alpha=32,
               targets={"q_proj", "k_proj", "v_proj", "o_proj"})
        self.adapters_enabled = True

    def project(self, features):
        if features.shape != (196, 768):
            raise ValueError("Expected 196 ordered SigLIP patch features")
        if self.settings.get("visual_tokens", 196) == 49:
            features = features.reshape(7, 2, 7, 2, 768).mean((1, 3)).reshape(49, 768)
        # Trainable connector remains FP32; surrounding autocast controls compute.
        return self.connector(features.float()) * self.connector_scale

    def build_example(self, row, features=None, *, target=True, include_end=True):
        embed = self.language_model.get_input_embeddings()
        question = self.tokenizer(row["question"], add_special_tokens=False)["input_ids"][:self.settings["question_cap"]]
        prefix = embed(torch.tensor(self.prefix_ids, device=self.device))
        suffix = embed(torch.tensor(question + self.suffix_ids, device=self.device))
        parts = [prefix]
        if features is not None:
            parts.append(self.project(features).to(prefix.dtype))
        parts.append(suffix)
        prompt_length = sum(part.shape[0] for part in parts)
        labels = [-100] * prompt_length
        if target:
            cap = self.settings["caption_cap"] if row["kind"] == "caption" else self.settings["answer_cap"]
            ids = self.tokenizer(row["answer"], add_special_tokens=False)["input_ids"][:cap]
            if not ids:
                raise ValueError("Empty tokenized supervision target")
            ids += [self.end_id] if include_end else []
            parts.append(embed(torch.tensor(ids, device=self.device)))
            labels.extend(ids)
        return torch.cat(parts), torch.tensor(labels, device=self.device), prompt_length

    def batch(self, rows, features, *, target=True, include_end=True, left_pad=False):
        built = [self.build_example(row, feature, target=target, include_end=include_end)
                 for row, feature in zip(rows, features)]
        if len(built) != len(rows):
            raise ValueError("Feature/example count mismatch")
        max_length = max(x[0].shape[0] for x in built)
        padded, labels, masks = [], [], []
        for embeddings, label, _ in built:
            pad = max_length - embeddings.shape[0]
            zeros = embeddings.new_zeros((pad, embeddings.shape[1]))
            ignored = label.new_full((pad,), -100)
            mask = torch.ones(label.shape[0], dtype=torch.long, device=self.device)
            empty = mask.new_zeros(pad)
            padded.append(torch.cat([zeros, embeddings] if left_pad else [embeddings, zeros]))
            labels.append(torch.cat([ignored, label] if left_pad else [label, ignored]))
            masks.append(torch.cat([empty, mask] if left_pad else [mask, empty]))
        mask = torch.stack(masks)
        positions = (mask.cumsum(-1) - 1).clamp(min=0)
        return {"inputs_embeds": torch.stack(padded), "attention_mask": mask, "position_ids": positions}, torch.stack(labels)

    def forward(self, **kwargs):
        # Do not use no_grad: frozen Qwen must propagate gradients to the connector.
        return self.language_model(**kwargs)

    @torch.no_grad()
    def calibrate(self, rows, features):
        if not rows or len(rows) != len(features):
            raise ValueError("Calibration needs matching training-only examples/features")
        visual_norms, text_norms = [], []
        for row, feature in zip(rows, features):
            if row["split"] != "train2014":
                raise ValueError("Connector calibration may use training images only")
            ids = self.tokenizer(row["question"], add_special_tokens=False)["input_ids"][:80]
            text = self.language_model.get_input_embeddings()(torch.tensor(ids, device=self.device))
            projected = self.project(feature)
            visual_norms.extend(projected.float().norm(dim=-1).tolist())
            text_norms.extend(text.float().norm(dim=-1).tolist())
        visual_mean = sum(visual_norms) / len(visual_norms)
        text_mean = sum(text_norms) / len(text_norms)
        scale = text_mean / max(visual_mean, 1e-8)
        self.connector_scale.fill_(scale)
        return {"rule": "mean text-token L2 norm / mean initial visual-token L2 norm",
                "examples": len(rows), "text_mean_l2": text_mean, "visual_mean_l2": visual_mean,
                "scale": scale, "training_only": True}

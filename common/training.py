"""Shared numerical primitives; losses explicitly handle causal shifting."""
import math
import random
import numpy as np
import torch
import torch.nn.functional as F


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])


def parameter_groups(model, decay):
    yes, no = [], []
    for name, p in model.named_parameters():  # removes tied duplicates
        if p.requires_grad:
            (yes if p.ndim >= 2 and not name.endswith("bias") else no).append(p)
    return [{"params": yes, "weight_decay": decay}, {"params": no, "weight_decay": 0.0}]


def lr_factor(update, total, warmup, floor=0.0):
    if update < warmup:
        return (update + 1) / warmup
    progress = min(1.0, (update - warmup) / max(1, total - warmup - 1))
    return floor + (1 - floor) * (1 + math.cos(math.pi * progress)) / 2


def causal_loss(logits, labels, reduction="sum"):
    # Compute CE in FP32; the model's output logits may be FP16.
    losses = F.cross_entropy(logits[:, :-1].float().transpose(1, 2),
                             labels[:, 1:], ignore_index=-100, reduction="none")
    if reduction == "examples":
        count = (labels[:, 1:] != -100).sum(1)
        if (count == 0).any():
            raise ValueError("Example has no supervised tokens")
        return (losses.sum(1) / count).sum()
    return losses.sum()


def collate(records, pad_id, device="cuda"):
    size = max(len(row["input_ids"]) for row in records)
    ids = torch.full((len(records), size), pad_id, dtype=torch.long)
    labels = torch.full_like(ids, -100)
    mask = torch.zeros_like(ids)
    for i, row in enumerate(records):
        n = len(row["input_ids"])
        ids[i, :n] = torch.tensor(row["input_ids"])
        labels[i, :n] = torch.tensor(row["labels"])
        mask[i, :n] = 1
    return {"input_ids": ids.to(device), "attention_mask": mask.to(device)}, labels.to(device)

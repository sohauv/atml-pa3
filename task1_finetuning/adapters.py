"""LoRA and factored OPLoRA, directly implementing the assignment equations."""
import math
import torch
from torch import nn

TARGETS = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}


class AdapterLinear(nn.Module):
    def __init__(self, base, rank, alpha, protection_rank=0, dropout=.05, bases=None):
        super().__init__()
        self.base = base
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.A = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=torch.float32))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        if protection_rank:
            # Exact FP32 SVD, no dense projection matrices. Bases are persisted.
            if bases is None:
                u, _, vh = torch.linalg.svd(base.weight.detach().float(), full_matrices=False)
                u, v = u[:, :protection_rank].contiguous(), vh[:protection_rank].T.contiguous()
            else:
                u, v = bases
                if tuple(u.shape) != (base.out_features, protection_rank) or tuple(v.shape) != (base.in_features, protection_rank):
                    raise ValueError("Checkpoint singular-basis shape mismatch")
                u = u.to(device=base.weight.device, dtype=torch.float32)
                v = v.to(device=base.weight.device, dtype=torch.float32)
            self.register_buffer("U", u)
            self.register_buffer("V", v)
        else:
            self.register_buffer("U", None)
            self.register_buffer("V", None)

    def factors(self):
        # Projection in FP32 even when the surrounding model uses autocast.
        with torch.autocast(self.A.device.type, enabled=False):
            b = self.B if self.U is None else self.B - self.U @ (self.U.T @ self.B)
            a = self.A if self.V is None else self.A - (self.A @ self.V) @ self.V.T
        return b, a

    def forward(self, x):
        b, a = self.factors()
        delta = torch.nn.functional.linear(torch.nn.functional.linear(self.dropout(x), a), b)
        return self.base(x) + delta * self.scale


def inject(model, rank, alpha, protection_rank=0, targets=TARGETS, basis_state=None):
    model.requires_grad_(False)
    # SVD may consume RNG in some implementations; isolate initialization per layer
    # so LoRA16 and OPLoRA16 get exactly the same unprojected A/B factors.
    selected = [(name, m) for name, m in model.named_modules()
                if name.rsplit(".", 1)[-1] in targets and isinstance(m, nn.Linear)]
    for index, (name, base) in enumerate(selected):
        with torch.random.fork_rng(devices=[base.weight.device] if base.weight.is_cuda else []):
            torch.manual_seed(6304 + index)
            bases = None
            if protection_rank and basis_state is not None:
                bases = (basis_state[name + ".U"], basis_state[name + ".V"])
            adapter = AdapterLinear(base, rank, alpha, protection_rank, bases=bases)
        parent_name, child = name.rsplit(".", 1)
        setattr(model.get_submodule(parent_name), child, adapter)
    if not selected:
        raise ValueError("No adapter target modules matched")
    model.enable_input_require_grads()
    return len(selected)


def export_factors(model):
    return {name: {"B": m.factors()[0].detach().cpu(),
                   "A": m.factors()[1].detach().cpu(), "scale": m.scale}
            for name, m in model.named_modules() if isinstance(m, AdapterLinear)}


def merge_factors(model, factors):
    with torch.no_grad():
        for name, saved in factors.items():
            layer = model.get_submodule(name)
            delta = saved["B"].float() @ saved["A"].float() * saved["scale"]
            layer.weight.add_(delta.to(layer.weight))
    return model

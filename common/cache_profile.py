"""CPU-only diagnostics for cached-tensor validation; never loads a GPU model."""
from pathlib import Path
import io
import time

import numpy as np
import torch


def load_feature_cpu(path):
    # torch.load seeks around ZIP records. One sequential remote-volume read
    # avoids issuing many lazy filesystem reads for the same small tensor.
    return torch.load(io.BytesIO(Path(path).read_bytes()), map_location="cpu", weights_only=True)


def valid_feature_cpu(value):
    return (isinstance(value, torch.Tensor) and value.shape == (196, 768)
            and value.dtype == torch.float16
            and bool(np.isfinite(value.detach().numpy()).all()))


def profile_feature(path, repeats=3, comparison_path=None):
    started = time.monotonic()
    value = torch.load(Path(path), map_location="cpu", weights_only=True)
    read_seconds = time.monotonic() - started
    started = time.monotonic()
    for _ in range(repeats):
        finite = bool(torch.isfinite(value).all())
    torch_seconds = (time.monotonic() - started) / repeats
    started = time.monotonic()
    for _ in range(repeats):
        valid = valid_feature_cpu(value)
    numpy_seconds = (time.monotonic() - started) / repeats
    started = time.monotonic()
    bulk = load_feature_cpu(comparison_path or path)
    bulk_seconds = time.monotonic() - started
    return {"shape": list(value.shape), "dtype": str(value.dtype), "finite": finite,
            "valid_feature": valid, "torch_threads": torch.get_num_threads(),
            "read_seconds": read_seconds, "torch_finite_seconds": torch_seconds,
            "numpy_validation_seconds": numpy_seconds, "repeats": repeats,
            "bulk_load_seconds": bulk_seconds, "bulk_feature_valid": valid_feature_cpu(bulk),
            "distinct_bulk_sample": comparison_path is not None and Path(comparison_path) != Path(path)}

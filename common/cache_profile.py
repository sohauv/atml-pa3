"""CPU-only diagnostics for cached-tensor validation; never loads a GPU model."""
from pathlib import Path
import io
import hashlib
from concurrent.futures import ThreadPoolExecutor
import time

import numpy as np
import torch

from common.io import read_json, sha256, write_json


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


def verify_feature_cache(root, max_seconds=600):
    """Attest immutable cache files on CPU so GPU resumes need only metadata."""
    root = Path(root)
    started = time.monotonic()
    files = sorted(root.glob("*.pt"))
    if not files:
        raise ValueError("No cached features to verify")

    def check(path):
        if time.monotonic() - started > max_seconds - 60:
            raise TimeoutError("CPU cache verification deadline; no index published")
        payload = path.read_bytes()
        value = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
        if not valid_feature_cpu(value) or len(payload) > 2 * value.numel() * value.element_size():
            raise ValueError(f"Invalid or oversized feature: {path.name}")
        return path.name, {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}

    with ThreadPoolExecutor(max_workers=8) as pool:
        entries = dict(pool.map(check, files))
    index = {"schema_version": 1, "provenance_sha256": sha256(root / "provenance.json"),
             "shape": [196, 768], "dtype": "float16", "all_finite": True, "files": entries}
    digest = hashlib.sha256(repr(sorted(entries.items())).encode()).hexdigest()[:16]
    path = root / f"verified-index-{digest}.json"
    if path.exists():
        if read_json(path) != index:
            raise ValueError("Existing verification index differs")
    else:
        write_json(path, index)
    return {"verified_images": len(entries), "index_path": str(path),
            "index_sha256": sha256(path), "seconds": time.monotonic() - started}


def load_verified_index(path, root):
    root, path = Path(root).resolve(), Path(path).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Verification index must belong to this cache")
    index = read_json(path)
    if (index["provenance_sha256"] != sha256(root / "provenance.json")
            or index.get("shape") != [196, 768] or index.get("dtype") != "float16"
            or index.get("all_finite") is not True):
        raise ValueError("Verification index provenance or schema differs")
    if any(Path(name).name != name or not name.endswith(".pt") for name in index["files"]):
        raise ValueError("Unsafe verification index filename")
    return index["files"]

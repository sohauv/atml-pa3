"""Run the unchanged Task1 trainer on Kaggle, without Modal or credentials."""
import argparse
import json
from pathlib import Path
import shutil
import zipfile

from common.io import read_json, sha256, write_json


def unpack_inputs(archive, destination):
    """Extract only the three expected data files, then verify original bytes."""
    destination = Path(destination)
    expected = {"task1/train.jsonl", "task1/metadata.json", "task1/source_ids.json"}
    with zipfile.ZipFile(archive) as source:
        if set(source.namelist()) != expected:
            raise ValueError("Unexpected transfer archive contents")
        for name in sorted(expected):
            target = destination / name
            if target.exists():
                with source.open(name) as stream:
                    import hashlib
                    if hashlib.file_digest(stream, "sha256").hexdigest() != sha256(target):
                        raise FileExistsError("Existing input differs; refusing overwrite")
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open(name) as stream, target.open("xb") as output:
                shutil.copyfileobj(stream, output)
    data = destination / "task1"
    if sha256(data / "train.jsonl") != read_json(data / "metadata.json")["sha256"]:
        raise ValueError("Transferred training data checksum mismatch")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--work", default="/kaggle/working/pa3")
    parser.add_argument("--method", choices=["lora4", "lora16", "oplora16"], default="lora4")
    parser.add_argument("--seconds", type=int, default=18000)
    parser.add_argument("--max-updates", type=int)
    parser.add_argument("--resume")
    parser.add_argument("--run-id", default="lora4-kaggle-20261030")
    parser.add_argument("--handoff-verified", action="store_true",
                        help="Only set after the previous Modal worker is terminal")
    args = parser.parse_args()
    if not args.handoff_verified:
        parser.error("Verify Modal has stopped before starting Kaggle training")
    if not 300 <= args.seconds <= 36000:
        parser.error("Use a bounded session with time to save outputs")
    if Path(args.run_id).name != args.run_id or args.run_id in {".", ".."}:
        parser.error("Unsafe run ID")
    data = unpack_inputs(args.archive, Path(args.work) / "data")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Select the free GPU accelerator before training")
    # Keep the existing one-device implementation and exact effective batch.
    # Two T4 devices do not form one 32GiB CUDA device.
    from task1_finetuning.train import train
    config = read_json(Path(__file__).resolve().parents[1] / "configs/task1_finetuning/shared.json")
    run = Path(args.work) / "runs" / args.run_id
    evidence = {"provider": "Kaggle", "gpu": torch.cuda.get_device_name(0),
                "device_count": torch.cuda.device_count(), "torch": torch.__version__,
                "data_sha256": sha256(data / "train.jsonl"), "method": args.method,
                "note": "Unchanged Task1 trainer; use one T4, preserve all output checkpoints"}
    write_json(Path(args.work) / f"{args.run_id}-environment.json", evidence)
    result = train(args.method, run, config, data_dir=data, resume=args.resume,
                   max_seconds=args.seconds, max_updates=args.max_updates)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

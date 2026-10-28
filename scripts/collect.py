"""Download only small experiment outputs, never checkpoint/model tensors."""
import argparse
from pathlib import Path, PurePosixPath
import modal


def collect(volume_name, remote_root, destination):
    volume = modal.Volume.from_name(volume_name)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    root = PurePosixPath("/" + remote_root.strip("/"))
    count = 0
    for entry in volume.iterdir(str(root), recursive=True):
        remote = PurePosixPath("/" + entry.path.lstrip("/"))
        if remote.suffix not in {".json", ".jsonl", ".csv", ".png", ".pdf"} or entry.size > 20 * 2**20:
            continue
        relative = remote.relative_to(root)
        target = destination.joinpath(*relative.parts)
        if ".." in relative.parts or not target.resolve().is_relative_to(destination.resolve()):
            raise ValueError("Unsafe volume path")
        payload = b"".join(volume.read_file(str(remote)))
        if target.exists():
            if target.read_bytes() == payload:
                continue
            # Version downloads instead of overwriting an earlier observation.
            import hashlib
            target = target.with_name(target.stem + "." + hashlib.sha256(payload).hexdigest()[:12] + target.suffix)
            if target.exists():
                continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(payload)
        count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--volume", default="atml-pa3-checkpoints")
    parser.add_argument("--remote", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(f"Collected {collect(args.volume, args.remote, args.output)} small artifacts")


if __name__ == "__main__":
    main()

"""Explicit, non-overwriting upload of prepared inputs to an existing volume."""
import argparse
from pathlib import Path, PurePosixPath
import modal

ALLOWED = {".json", ".jsonl", ".parquet", ".jpg", ".jpeg", ".png", ".md"}


def upload(source, remote_root, volume_name="atml-pa3-data"):
    source = Path(source).resolve()
    if not source.is_relative_to((Path.cwd() / "data").resolve()) or not source.is_dir():
        raise ValueError("Only explicit input directories inside project/data may be uploaded")
    remote_root = PurePosixPath("/" + remote_root.strip("/"))
    if str(remote_root) == "/" or ".." in remote_root.parts:
        raise ValueError("Choose a non-root volume destination")
    files = [p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in ALLOWED]
    if not files:
        raise ValueError("No prepared input files found")
    volume = modal.Volume.from_name(volume_name)
    # Modal rejects collisions; force=False never replaces existing user data.
    with volume.batch_upload(force=False) as batch:
        for path in files:
            relative = path.relative_to(source)
            batch.put_file(path, remote_root.joinpath(*relative.parts))
    return len(files)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--remote", required=True)
    parser.add_argument("--volume", default="atml-pa3-data")
    args = parser.parse_args()
    print(f"Uploaded {upload(args.source, args.remote, args.volume)} prepared input files")

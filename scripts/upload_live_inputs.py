"""Upload the three small/necessary Task3 live inputs without overwriting."""
from pathlib import Path
import argparse
import modal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", required=True)
    parser.add_argument("--remote", default="/task3/live-input-v1")
    args = parser.parse_args()
    analysis = Path(args.analysis)
    pairs = [(Path("data/task3/qwen25-math/generations.parquet"), "qwen25-math/generations.parquet"),
             (analysis / "splits.json", "analysis/math/splits.json"),
             (analysis / "data_provenance.json", "analysis/math/data_provenance.json")]
    if not all(source.is_file() for source, _ in pairs):
        raise FileNotFoundError("Prepare supplied MATH data and fixed split/provenance first")
    if ".." in args.remote.split("/") or args.remote.strip("/") == "":
        raise ValueError("A fresh, non-root destination is required")
    volume = modal.Volume.from_name("atml-pa3-data")
    with volume.batch_upload(force=False) as batch:
        for source, relative in pairs:
            batch.put_file(source, args.remote.rstrip("/") + "/" + relative)
    print("Uploaded pinned MATH live inputs and fixed split IDs; no GPU job launched")


if __name__ == "__main__":
    main()

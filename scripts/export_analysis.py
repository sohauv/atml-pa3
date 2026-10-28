"""Publish small completed-analysis artifacts; leave raw rollouts/results in place."""
import argparse
from pathlib import Path
import shutil

from common.io import read_json, sha256, write_json

SUMMARY_FILES = ("complete.json", "summary.csv", "adaptive_summary.csv", "policies.json",
                 "paired_intervals.json", "splits.json", "scorer.json", "scorer_diagnostics.json",
                 "data_provenance.json", "metadata_provenance.json", "input_summary.json",
                 "signal_summary.json", "selection_examples.json", "stopping_examples.json")


def export(source, output):
    source, output = Path(source), Path(output)
    complete = read_json(source / "complete.json")
    if (complete["questions"], complete["held_out"], complete["orders"]) != (1000, 800, 10):
        raise ValueError("Only the complete prescribed analysis may be exported")
    output.mkdir(parents=True, exist_ok=True)
    artifacts = []
    for name in SUMMARY_FILES:
        path = source / name
        if path.exists():
            artifacts.append(path)
    artifacts += list((source / "figures").glob("*.png")) + list((source / "figures").glob("*.pdf"))
    for path in artifacts:
        if path.stat().st_size > 5 * 2**20:
            raise ValueError(f"Summary unexpectedly large: {path}")
        target = output / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        # Preserve earlier exported observations if rendering changed.
        if target.exists() and sha256(target) != sha256(path):
            backup = target.with_name(target.stem + "." + sha256(target)[:12] + target.suffix)
            if not backup.exists():
                shutil.copyfile(target, backup)
        shutil.copyfile(path, target)
    manifest = {"raw_output_relative": source.as_posix(),
                "files": {path.relative_to(source).as_posix(): {
                    "bytes": path.stat().st_size, "sha256": sha256(path)}
                    for path in source.rglob("*") if path.is_file()}}
    write_json(output / "artifact_manifest.json", manifest)
    return {"output": str(output), "exported_artifacts": len(artifacts), "raw_source_files": len(manifest["files"])}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("output")
    args = parser.parse_args()
    print(export(args.source, args.output))

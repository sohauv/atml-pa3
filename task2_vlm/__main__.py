"""Local metadata preparation; GPU execution is managed by the parent launcher."""
import argparse
import json
from pathlib import Path

from common.io import read_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    metadata = sub.add_parser("download-metadata")
    metadata.add_argument("--output", default="data/task2/raw")
    extract = sub.add_parser("extract-metadata")
    extract.add_argument("--output", default="data/task2/raw")
    extract.add_argument("--allow-partial", action="store_true")
    archive = sub.add_parser("download-archive")
    archive.add_argument("--name", required=True, choices=["coco_annotations", "questions_train", "questions_val",
                        "annotations_train", "annotations_val", "pairs_train", "pairs_val"])
    archive.add_argument("--output", default="data/task2/raw/archives")
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--paths", default="data/task2/raw/paths.json")
    prepare.add_argument("--config", default="configs/task2_vlm/main.json")
    prepare.add_argument("--output", default="data/task2/prepared")
    export = sub.add_parser("export-ids")
    export.add_argument("--manifest", default="data/task2/prepared/manifest.json")
    export.add_argument("--provenance", default="data/task2/prepared/provenance.json")
    export.add_argument("--output", default="results/task2_vlm/selection_ids")
    images = sub.add_parser("download-images")
    images.add_argument("--manifest", default="data/task2/prepared/manifest.json")
    images.add_argument("--output", default="data/task2/images")
    images.add_argument("--workers", type=int, default=8)
    pin = sub.add_parser("pin-models")
    pin.add_argument("--output", default="data/task2/model_revisions.json")
    args = parser.parse_args()
    if args.command == "download-metadata":
        from task2_vlm.download import metadata
        result = metadata(args.output)
    elif args.command == "extract-metadata":
        from task2_vlm.download import extract_metadata
        result = extract_metadata(args.output, require_all=not args.allow_partial)
    elif args.command == "download-archive":
        from task2_vlm.download import ARCHIVES, download
        result = {"archive": str(download(ARCHIVES[args.name], Path(args.output) / f"{args.name}.zip", force_ranges=True))}
    elif args.command == "prepare":
        from task2_vlm.data import prepare_data
        result = prepare_data(read_json(args.paths), args.output, read_json(args.config))
    elif args.command == "export-ids":
        from task2_vlm.data import export_selection_ids
        result = export_selection_ids(args.manifest, args.provenance, args.output)
    elif args.command == "download-images":
        from task2_vlm.download import images
        result = images(args.manifest, args.output, workers=args.workers)
    else:
        from huggingface_hub import HfApi
        path = Path(args.output)
        if path.exists():
            result = read_json(path)
        else:
            api = HfApi()
            result = {name: api.model_info(name).sha for name in
                      ("Qwen/Qwen3-0.6B", "google/siglip-base-patch16-224")}
            write_json(path, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

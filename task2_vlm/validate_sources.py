"""Exercise original VQAv2 schemas/quotas before the larger COCO download ends."""
from collections import Counter
from pathlib import Path

import numpy as np

from common.io import read_json, sha256, write_json
from task2_vlm.data import compact_annotations, image_ids, natural_pool, select_natural, stream_items


def validate_natural(paths_path, output_path):
    paths = read_json(paths_path)
    pools = {}
    for split in ("train", "val"):
        questions = {"questions": list(stream_items(paths[f"questions_{split}"], "questions"))}
        split_name = "train2014" if split == "train" else "val2014"
        files = {q["image_id"]: f'COCO_{split_name}_{q["image_id"]:012d}.jpg' for q in questions["questions"]}
        pools[split] = natural_pool(questions, compact_annotations(paths[f"annotations_{split}"]),
                                   read_json(paths[f"pairs_{split}"]), split_name, files)
    rng = np.random.default_rng(6304)
    train, _ = select_natural(pools["train"], {"yes/no": 640, "number": 256, "other": 640}, rng)
    dev, reserved = select_natural(pools["val"], {"yes/no": 12, "number": 8, "other": 12}, rng, disjoint=True)
    test, _ = select_natural(pools["val"], {"yes/no": 36, "number": 24, "other": 36}, rng,
                             excluded=reserved, disjoint=True)
    if image_ids(train) & (image_ids(dev) | image_ids(test)) or image_ids(dev) & image_ids(test):
        raise ValueError("Natural source selection leaks images across splits")
    report = {"seed": 6304, "eligible_pairs": {split: dict(Counter(p["answer_type"] for p in pool)) for split, pool in pools.items()},
              "selected": {split: {"pair_counts": dict(Counter(p["answer_type"] for p in rows)),
                                   "image_count": len(image_ids(rows)),
                                   "pair_ids": [p["pair_id"] for p in rows]}
                           for split, rows in (("train", train), ("dev", dev), ("test", test))},
              "source_sha256": {key: sha256(value) for key, value in paths.items()
                                if key.startswith(("questions_", "annotations_", "pairs_"))},
              "status": "Natural original schema, consensus filtering, quotas and isolation verified; full COCO manifest pending"}
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(output)
    write_json(output, report)
    return {**report, "selected": {key: {k: v for k, v in value.items() if k != "pair_ids"}
                                   for key, value in report["selected"].items()}}


if __name__ == "__main__":
    import json
    print(json.dumps(validate_natural("data/task2/raw/paths.json", "data/task2/preflight_natural.json"), indent=2))

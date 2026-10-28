"""Deterministic COCO2014/VQAv2 preparation with strict image isolation.

Input keys: captions_train, captions_val, instances_train, instances_val,
questions_train, questions_val, annotations_train, annotations_val,
pairs_train, pairs_val. All values are paths to original released JSON files.
"""
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import sys

import numpy as np

from common.io import read_json, sha256, write_json
from task2_vlm.metrics import normalize_answer


def stream_items(path, prefix):
    """Read released JSON arrays without retaining segmentation/answer dicts."""
    import ijson
    with Path(path).open("rb") as stream:
        yield from ijson.items(stream, prefix + ".item" if prefix else "item", use_float=True)


def compact_instances(path):
    images = [{"id": row["id"], "file_name": row["file_name"]} for row in stream_items(path, "images")]
    categories = [{"id": row["id"], "name": row["name"]} for row in stream_items(path, "categories")]
    counts = defaultdict(Counter)
    for row in stream_items(path, "annotations"):
        counts[row["image_id"]][row["category_id"]] += 1
    return {"images": images, "categories": categories, "counts": counts}


def compact_captions(path, images):
    # Only image eligibility counts are needed before seeded image selection.
    # Actual caption strings are streamed later for the selected images alone.
    counts = Counter(row["image_id"] for row in stream_items(path, "annotations"))
    return {"images": images, "counts": counts, "source_path": str(path)}


def compact_annotations(path):
    return {"annotations": [{"question_id": row["question_id"], "answer_type": sys.intern(row["answer_type"]),
                              "answers": [sys.intern(answer["answer"]) for answer in row["answers"]]}
                             for row in stream_items(path, "annotations")]}


def natural_from_preflight(paths, preflight_path, rng, seed):
    """Reuse the verified original-source selection without reloading all QAs."""
    report = read_json(preflight_path)
    if report["seed"] != seed:
        raise ValueError("Natural preflight selection used a different seed")
    for key, expected in report["source_sha256"].items():
        if sha256(paths[key]) != expected:
            raise ValueError("Natural preflight source checksum changed")
    selected = {}
    for split, phases in (("train", ("train",)), ("val", ("dev", "test"))):
        desired = [pair_id for phase in phases for pair_id in report["selected"][phase]["pair_ids"]]
        desired_pairs = {tuple(map(int, pair_id.rsplit("-", 2)[1:])) for pair_id in desired}
        desired_questions = {question_id for pair_ids in desired_pairs for question_id in pair_ids}
        questions = {"questions": [row for row in stream_items(paths[f"questions_{split}"], "questions")
                                    if row["question_id"] in desired_questions]}
        annotations = {"annotations": [row for row in stream_items(paths[f"annotations_{split}"], "annotations")
                                        if row["question_id"] in desired_questions]}
        pairs = [row for row in stream_items(paths[f"pairs_{split}"], "")
                 if tuple(sorted(row)) in desired_pairs]
        split_name = "train2014" if split == "train" else "val2014"
        files = {row["image_id"]: f'COCO_{split_name}_{row["image_id"]:012d}.jpg' for row in questions["questions"]}
        pool = {pair["pair_id"]: pair for pair in natural_pool(questions, annotations, pairs, split_name, files)}
        if set(pool) != set(desired):
            raise ValueError("Natural preflight IDs fail original-source eligibility/membership checks")
        for phase in phases:
            selected[phase] = [pool[pair_id] for pair_id in report["selected"][phase]["pair_ids"]]
    # select_natural consumes one full permutation per type, regardless of how
    # many disjoint pairs are retained. Replay that exact RNG consumption.
    for split in ("train", "val", "val"):
        for kind in ("yes/no", "number", "other"):
            rng.permutation(report["eligible_pairs"][split][kind])
    for phase, quotas in (("train", {"yes/no": 640, "number": 256, "other": 640}),
                          ("dev", {"yes/no": 12, "number": 8, "other": 12}),
                          ("test", {"yes/no": 36, "number": 24, "other": 36})):
        if dict(Counter(pair["answer_type"] for pair in selected[phase])) != quotas:
            raise ValueError("Natural preflight does not satisfy the manual's exact pair quotas")
    return selected["train"], selected["dev"], selected["test"]


def consensus(answers):
    originals = [a["answer"] if isinstance(a, dict) else a for a in answers]
    normalized = [normalize_answer(a) for a in originals]
    if len(originals) != 10:
        raise ValueError("VQAv2 record lacks ten answers")
    winner, count = Counter(normalized).most_common(1)[0]
    target = next(answer for answer, value in zip(originals, normalized) if value == winner)
    return target, winner, count, originals


def natural_pool(questions, annotations, pairs, split, images):
    questions = {q["question_id"]: q for q in questions["questions"]}
    annotations = {a["question_id"]: a for a in annotations["annotations"]}
    output, seen = [], set()
    for raw in pairs:
        if not isinstance(raw, list) or len(raw) != 2:
            raise ValueError("Expected official complementary pair lists [question_id, question_id]")
        ids = tuple(sorted(map(int, raw)))
        if ids in seen:
            continue
        seen.add(ids)
        if any(i not in questions or i not in annotations for i in ids):
            raise ValueError("Complementary pair references an absent question")
        qa, qb = [questions[i] for i in ids]
        if " ".join(qa["question"].casefold().split()) != " ".join(qb["question"].casefold().split()):
            raise ValueError("Complementary pair has different question wording")
        if qa["image_id"] == qb["image_id"]:
            raise ValueError("Complementary pair uses the same image twice")
        values = [consensus(annotations[i]["answers"]) for i in ids]
        if any(v[2] < 8 for v in values) or values[0][1] == values[1][1]:
            continue
        types = [annotations[i]["answer_type"] for i in ids]
        if types[0] != types[1] or types[0] not in {"yes/no", "number", "other"}:
            continue
        examples = []
        for i, value in zip(ids, values):
            q = questions[i]
            examples.append({"question_id": i, "image_id": q["image_id"],
                "image_file": images[q["image_id"]], "split": split,
                "question": q["question"], "answer": value[0], "answers": value[3],
                "answer_type": types[0], "kind": "natural"})
        output.append({"pair_id": f"vqa-{split}-{ids[0]}-{ids[1]}", "kind": "natural",
                       "answer_type": types[0], "examples": examples})
    return sorted(output, key=lambda p: p["pair_id"])


def select_natural(pool, quotas, rng, excluded=None, disjoint=False):
    selected = []
    used = set(excluded or ())
    for kind, count in quotas.items():
        eligible = [p for p in pool if p["answer_type"] == kind]
        for index in rng.permutation(len(eligible)):
            pair = eligible[int(index)]
            images = {r["image_id"] for r in pair["examples"]}
            if images & used:
                continue
            selected.append(pair)
            if disjoint:
                used.update(images)
            if sum(p["answer_type"] == kind for p in selected) == count:
                break
        actual = sum(p["answer_type"] == kind for p in selected)
        if actual != count:
            raise ValueError(f"Insufficient eligible {kind} pairs: need {count}, found {actual}")
    return selected, used


def plural(category):
    irregular = {"person": "people", "mouse": "mice", "knife": "knives", "sheep": "sheep",
                 "skis": "skis", "scissors": "scissors"}
    if category in irregular:
        return irregular[category]
    if category.endswith(("s", "sh", "ch", "x", "z")):
        return category + "es"
    if category.endswith("y") and category[-2:-1] not in "aeiou":
        return category[:-1] + "ies"
    return category + "s"


def grounding_pool(instances, split, rng, count_per_kind, excluded=None, disjoint=False):
    images = {r["id"]: r["file_name"] for r in instances["images"]}
    categories = {r["id"]: r["name"] for r in instances["categories"]}
    counts = instances.get("counts")
    if counts is None:
        counts = defaultdict(Counter)
        for row in instances["annotations"]:
            counts[row["image_id"]][row["category_id"]] += 1
    used = set(excluded or ())
    output = {"presence": [], "counting": []}
    # Sorted category/image IDs make results independent of dict/hash order.
    available = sorted(set(images) - used)
    tables = {}
    for category in sorted(categories):
        tables[category] = {n: [i for i in available if counts[i][category] == n]
                            for n in range(1, 5)}
        tables[category][0] = [i for i in available if counts[i][category] == 0]
        tables[category]["positive"] = [i for i in available if counts[i][category] > 0]
    for kind in output:
        attempts = 0
        target = count_per_kind[kind] if isinstance(count_per_kind, dict) else count_per_kind
        while len(output[kind]) < target:
            attempts += 1
            if attempts > max(10000, target * 100):
                raise ValueError(f"Cannot construct {target} {kind} pairs without leakage")
            category = int(rng.choice(sorted(categories)))
            if kind == "presence":
                labels, question = ["yes", "no"], f"Does this image contain {categories[category]}?"
                pools = [tables[category]["positive"], tables[category][0]]
            else:
                values = rng.choice([1, 2, 3, 4], 2, replace=False).tolist()
                labels = list(map(str, values))
                question = f"How many {plural(categories[category])} can you see?"
                pools = [tables[category][value] for value in values]
            if any(not pool for pool in pools):
                continue
            picked = [int(rng.choice(pool)) for pool in pools]
            if len(set(picked)) != 2 or any(image in used for image in picked):
                continue
            pair_id = f"ground-{split}-{kind}-{category}-{picked[0]}-{picked[1]}"
            if any(p["pair_id"] == pair_id for p in output[kind]):
                continue
            rows = [{"image_id": image, "image_file": images[image], "split": split,
                     "question": question, "answer": label, "answers": [label],
                     "answer_type": kind, "kind": kind, "category_id": category}
                    for image, label in zip(picked, labels)]
            output[kind].append({"pair_id": pair_id, "kind": kind, "answer_type": kind, "examples": rows})
            if disjoint:
                used.update(picked)
    return output, used


def image_ids(rows):
    return {row["image_id"] for item in rows for row in item.get("examples", [item])}


def validate_isolation(manifest):
    train = image_ids(manifest["captions_train"] + manifest["natural_train"] +
                      manifest["presence_train"] + manifest["counting_train"])
    suites = {key: image_ids(manifest[key]) for key in
              ("captions_dev", "grounding_dev", "grounding_test", "natural_dev", "natural_test")}
    for key, images in suites.items():
        if train & images:
            raise ValueError(f"Training overlaps {key}")
        for other, values in suites.items():
            if key < other and images & values:
                raise ValueError(f"Evaluation image overlap: {key} and {other}")
    return {"train_images": len(train), **{key: len(value) for key, value in suites.items()}}


def prepare_data(paths, output_dir, config):
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a fresh Task 2 preparation directory")
    required = [f"{kind}_{split}" for kind in ("captions", "instances", "questions", "annotations", "pairs")
                for split in ("train", "val")]
    for key in required:
        if key not in paths or not Path(paths[key]).is_file():
            raise FileNotFoundError(f"Missing original Task 2 input: {key}")
    # Full released JSONs contain segmentation arrays and millions of repeated
    # answer dictionaries. Stream and retain only fields used in this task.
    raw = {}
    rng = np.random.default_rng(config["seed"])
    preflight = output.parent / "preflight_natural.json"
    if preflight.exists():
        print("[Task2] Verifying and reusing saved original-source natural pair selection", flush=True)
        natural_train, natural_dev, natural_test = natural_from_preflight(paths, preflight, rng, config["seed"])
    pools = {}
    for split in ("train", "val"):
        print(f"[Task2] Streaming compact {split} COCO metadata", flush=True)
        raw[f"instances_{split}"] = compact_instances(paths[f"instances_{split}"])
        raw[f"captions_{split}"] = compact_captions(paths[f"captions_{split}"], raw[f"instances_{split}"]["images"])
        if not preflight.exists():
            questions = {"questions": list(stream_items(paths[f"questions_{split}"], "questions"))}
            annotations = compact_annotations(paths[f"annotations_{split}"])
            pairs = read_json(paths[f"pairs_{split}"])
            images = {r["id"]: r["file_name"] for r in raw[f"instances_{split}"]["images"]}
            pools[split] = natural_pool(questions, annotations, pairs,
                                       "train2014" if split == "train" else "val2014", images)
            del questions, annotations, pairs
            print(f"[Task2] Eligible {split} natural pairs: {len(pools[split])}", flush=True)
    if not preflight.exists():
        natural_train, _ = select_natural(pools["train"], {"yes/no": 640, "number": 256, "other": 640}, rng)
        natural_dev, reserved = select_natural(pools["val"], {"yes/no": 12, "number": 8, "other": 12}, rng, disjoint=True)
        natural_test, reserved = select_natural(pools["val"], {"yes/no": 36, "number": 24, "other": 36}, rng,
                                                   excluded=reserved, disjoint=True)
    reserved = image_ids(natural_dev + natural_test)
    del pools
    dev_counts = dict(zip(("presence", "counting"), config["grounding_dev_presence_counting_pairs"]))
    test_counts = dict(zip(("presence", "counting"), config["grounding_test_presence_counting_pairs"]))
    dev, reserved = grounding_pool(raw["instances_val"], "val2014", rng, dev_counts, reserved, disjoint=True)
    test, reserved = grounding_pool(raw["instances_val"], "val2014", rng, test_counts, reserved, disjoint=True)
    training, _ = grounding_pool(raw["instances_train"], "train2014", rng, config["grounding_training_pairs_per_kind"])
    captions = {}
    for split, number, per_image in (("train", 6000, 3), ("val", 96, 1)):
        by_image = defaultdict(list)
        files = {row["id"]: row["file_name"] for row in raw[f"captions_{split}"]["images"]}
        eligible = sorted(i for i, count in raw[f"captions_{split}"]["counts"].items() if count >= per_image and
                          (split == "train" or i not in reserved))
        if len(eligible) < number:
            raise ValueError("Insufficient disjoint caption images")
        chosen = rng.choice(eligible, number, replace=False).tolist()
        chosen_set = set(chosen)
        for row in stream_items(raw[f"captions_{split}"]["source_path"], "annotations"):
            if row["image_id"] in chosen_set:
                by_image[row["image_id"]].append(row)
        captions[split] = [{"image_id": image, "image_file": files[image],
                            "split": "train2014" if split == "train" else "val2014",
                            "question": "Describe this image.", "answer": row["caption"],
                            "answers": [row["caption"]], "caption_id": row["id"],
                            "kind": "caption", "answer_type": "caption"}
                           for image in chosen for row in sorted(by_image[image], key=lambda r: r["id"])[:per_image]]
    manifest = {"captions_train": captions["train"], "captions_dev": captions["val"],
                "natural_train": natural_train, "natural_dev": natural_dev, "natural_test": natural_test,
                "presence_train": training["presence"], "counting_train": training["counting"],
                "grounding_dev": dev["presence"] + dev["counting"],
                "grounding_test": test["presence"] + test["counting"]}
    # COCO train/val numeric IDs are globally distinct; validate explicitly.
    audit = validate_isolation(manifest)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "manifest.json", manifest)
    write_json(output / "provenance.json", {"seed": config["seed"], "config": config,
               "source_sha256": {key: sha256(paths[key]) for key in required}, "isolation": audit,
               "normalization": "VQA official punctuation/number/article/contraction rules",
               "grounding_labels": "COCO annotation counts, including crowd annotations; absence means unannotated"})
    return {"output": str(output), "isolation": audit, "counts": {key: len(value) for key, value in manifest.items()}}


def export_selection_ids(manifest_path, provenance_path, output_dir):
    """Publish submission-safe IDs/hashes only; no questions, answers or images."""
    manifest = read_json(manifest_path)
    provenance = read_json(provenance_path)
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("ID export needs a fresh output directory")
    output.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, rows in manifest.items():
        path = output / f"{name}.csv"
        with path.open("x", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            if name.startswith("captions_"):
                writer.writerow(["image_id", "caption_id"])
                writer.writerows([row["image_id"], row["caption_id"]] for row in rows)
            elif name.startswith("natural_"):
                writer.writerow(["pair_id", "question_id_1", "question_id_2", "image_id_1", "image_id_2"])
                writer.writerows([pair["pair_id"], pair["examples"][0]["question_id"], pair["examples"][1]["question_id"],
                                  pair["examples"][0]["image_id"], pair["examples"][1]["image_id"]] for pair in rows)
            else:
                writer.writerow(["pair_id", "category_id", "image_id_1", "image_id_2"])
                writer.writerows([pair["pair_id"], pair["examples"][0]["category_id"],
                                  pair["examples"][0]["image_id"], pair["examples"][1]["image_id"]] for pair in rows)
        files[path.name] = {"rows": len(rows), "sha256": sha256(path), "bytes": path.stat().st_size}
    summary = {"schema_version": 1, "seed": provenance["seed"], "manifest_sha256": sha256(manifest_path),
               "source_sha256": provenance["source_sha256"], "counts": {key: len(value) for key, value in manifest.items()},
               "isolation": validate_isolation(manifest),
               "natural_pair_types": {key: dict(Counter(pair["answer_type"] for pair in manifest[key]))
                                      for key in ("natural_train", "natural_dev", "natural_test")},
               "files": files, "content": "ID-only CSVs and source hashes; no question/caption/answer/image content"}
    write_json(output / "provenance.json", summary)
    return {"output": str(output), "files": len(files) + 1,
            "bytes": sum(path.stat().st_size for path in output.iterdir()), "manifest_sha256": summary["manifest_sha256"]}

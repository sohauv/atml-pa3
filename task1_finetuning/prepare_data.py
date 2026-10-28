import hashlib
import json
import os
import unicodedata
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import HfApi
from transformers import AutoTokenizer

from common.io import sha256


def normalize(text):
    text = unicodedata.normalize("NFKC", text or "")
    return " ".join(text.casefold().split())


def prepare_data(output_dir="/data/task1", revisions=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if any(output_dir.iterdir()):
        # Reuse only a complete, verified dataset; never replace existing data.
        required = [output_dir / name for name in ("train.jsonl", "metadata.json", "source_ids.json")]
        if not all(path.exists() for path in required):
            raise FileExistsError("Partial/existing data found; use a fresh output directory")
        metadata = json.loads(required[1].read_text(encoding="utf-8"))
        if sha256(required[0]) != metadata["sha256"]:
            raise ValueError("Existing dataset checksum mismatch")
        if revisions and revisions != metadata["revisions"]:
            raise ValueError("Existing dataset revisions differ")
        return metadata

    api = HfApi()
    revisions = revisions or {
        "model": api.model_info("Qwen/Qwen3-0.6B").sha,
        "metamath": api.dataset_info("meta-math/MetaMathQA").sha,
        "gsm8k": api.dataset_info("openai/gsm8k").sha,
    }

    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-0.6B",
        revision=revisions["model"],
    )

    metamath = load_dataset(
        "meta-math/MetaMathQA",
        revision=revisions["metamath"],
        split="train",
    )
    gsm8k = load_dataset(
        "openai/gsm8k",
        "main",
        revision=revisions["gsm8k"],
        split="test",
    )

    test_questions = {
        normalize(row["question"]) for row in gsm8k
    }

    counts = Counter()
    source_ids = []
    lengths = []
    response_lengths = []

    temporary_path = output_dir / "train.jsonl.tmp"
    final_path = output_dir / "train.jsonl"

    with temporary_path.open("w", encoding="utf-8") as f:
        for source_id, row in enumerate(metamath):
            counts["scanned"] += 1

            question = row.get("query") or ""
            response = row.get("response") or ""

            if not question.strip() or not response.strip():
                counts["empty"] += 1
                continue

            if any(
                normalize(row.get(field)) in test_questions
                for field in ("query", "original_question")
            ):
                counts["gsm8k_overlap"] += 1
                continue

            messages = [
                {"role": "user", "content": question},
                {"role": "assistant", "content": response},
            ]

            prompt = tokenizer.apply_chat_template(
                messages[:1],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            full_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
            )

            input_ids = tokenizer(
                full_text, add_special_tokens=False
            )["input_ids"]

            if len(input_ids) > 2048:
                counts["too_long"] += 1
                continue

            prompt_ids = tokenizer(
                prompt, add_special_tokens=False
            )["input_ids"]

            assert input_ids[:len(prompt_ids)] == prompt_ids

            end_token = tokenizer.convert_tokens_to_ids(
                "<|im_end|>"
            )
            assert end_token in input_ids[len(prompt_ids):]

            # Mask the prompt and trailing template newline.
            # Train on the assistant response and end-of-turn.
            end_position = (
                len(input_ids)
                - 1
                - input_ids[::-1].index(end_token)
            )
            labels = [-100] * len(input_ids)
            labels[len(prompt_ids):end_position + 1] = (
                input_ids[len(prompt_ids):end_position + 1]
            )

            record = {
                "source_id": source_id,
                "input_ids": input_ids,
                "labels": labels,
            }
            f.write(json.dumps(record) + "\n")

            source_ids.append(source_id)
            lengths.append(len(input_ids))
            response_lengths.append(
                sum(label != -100 for label in labels)
            )
            counts["retained"] += 1

            if counts["retained"] % 5000 == 0:
                print(dict(counts), flush=True)

            if counts["retained"] == 40000:
                break

    assert counts["retained"] == 40000, dict(counts)
    assert source_ids == sorted(source_ids)
    assert max(lengths) <= 2048
    assert min(response_lengths) > 0

    os.replace(temporary_path, final_path)

    digest = hashlib.sha256()
    with final_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)

    metadata = {
        "seed": 6304,
        "revisions": revisions,
        "normalization": "NFKC + casefold + collapse whitespace",
        "max_sequence_length": 2048,
        "enable_thinking": False,
        "counts": dict(counts),
        "mean_sequence_length": sum(lengths) / len(lengths),
        "max_observed_length": max(lengths),
        "mean_response_tokens": (
            sum(response_lengths) / len(response_lengths)
        ),
        "sha256": digest.hexdigest(),
    }

    with (output_dir / "source_ids.json").open("w") as f:
        json.dump(source_ids, f)

    with (output_dir / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)

    return metadata

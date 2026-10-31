"""Run the existing full Task1 benchmark protocol on a free Kaggle GPU."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seconds", type=int, default=18000)
    parser.add_argument("--tasks", nargs="+", choices=["gsm8k_cot", "hellaswag", "boolq", "arc_easy"],
                        default=["hellaswag", "boolq", "arc_easy", "gsm8k_cot"])
    args = parser.parse_args()
    if not 300 <= args.seconds <= 36000:
        parser.error("Use a bounded evaluation job")
    checkpoint = Path(args.checkpoint)
    if not (checkpoint / "complete.json").is_file():
        parser.error("A complete checkpoint is required")
    from common.io import read_json
    from task1_finetuning.evaluate import evaluate_job
    config = read_json(Path(__file__).resolve().parents[1] / "configs/task1_finetuning/shared.json")
    request = {"job": "evaluate1", "config": config, "checkpoint": str(checkpoint),
               "data_dir": str(Path(args.data_dir)), "eval_tasks": args.tasks,
               "eval_likelihood_batch_size": 8}
    result = evaluate_job(request, Path(args.output), args.seconds)
    print(json.dumps(result, default=str), flush=True)


if __name__ == "__main__":
    main()

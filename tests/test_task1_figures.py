import csv
import json
from pathlib import Path
import tempfile
import unittest

from common.io import write_json
from task1_finetuning.figures import benchmark_records, condition_for, make_figures, training_runs


class Task1FigureTests(unittest.TestCase):
    def test_partial_run_and_smoke_evaluation_are_never_final_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "measurements"
            run = root / "test-full"
            run.mkdir(parents=True)
            write_json(run / "config.json", {"method": "full", "updates": 1250})
            records = [{"update": update, "sft_loss": 1 / update, "lr": .00001, "skipped": False}
                       for update in (1, 2)]
            (run / "train.jsonl").write_text("\n".join(json.dumps(row) for row in records))
            # Immutable collector snapshots must deduplicate their shared prefix.
            (run / "train.snapshot.jsonl").write_text(json.dumps(records[0]))
            checkpoint = run / "resume-0002-test"
            write_json(checkpoint / "summary.json", {"update": 2, "peak_gpu_gib": 1,
                        "trainable_parameters": 10, "wall_seconds": 3, "checkpoint_bytes": 1024})
            write_json(checkpoint / "complete.json", {"update": 2})
            evaluation = root / "baseline-smoke"
            write_json(evaluation / "request.json", {"job": "evaluate1", "checkpoint": None, "eval_limit": 5})
            write_json(evaluation / "gsm8k_cot.json", {"results": {"gsm8k_cot": {"exact_match,flexible-extract": .5}},
                "protocol": {"complete_split": False, "smoke_only": True},
                "n-samples": {"gsm8k_cot": {"effective": 5}}})
            output = Path(temporary) / "figures"
            result = make_figures(root, output)
            self.assertFalse(result["training_complete"])
            self.assertFalse(result["training_runs"][0]["complete"])
            self.assertEqual(result["training_runs"][0]["updates"], 2)
            self.assertEqual(result["completed_benchmark_cells"], 0)
            self.assertEqual(result["excluded_smoke_or_unverified_evaluations"], 1)
            self.assertEqual(len(result["missing_training_methods"]), 4)
            self.assertTrue((output / "learning_curves.png").exists())
            with (output / "benchmark_comparison.csv").open(newline="") as stream:
                table = list(csv.DictReader(stream))
            self.assertEqual(len(table), 7)
            self.assertTrue(all(row["gsm8k_cot_percent"] == "" for row in table))
            with self.assertRaises(FileExistsError):
                make_figures(root, output)

    def test_full_split_deltas_are_percentage_points(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for condition, checkpoint, score in [("baseline", None, .2),
                                                ("final", "/checkpoints/runs/full-run/step-1250", .3)]:
                folder = root / condition
                write_json(folder / "request.json", {"job": "evaluate1", "checkpoint": checkpoint, "method": "full"})
                write_json(folder / "gsm8k_cot.json", {"results": {"gsm8k_cot": {"exact_match,flexible-extract": score}},
                    "protocol": {"complete_split": True, "smoke_only": False},
                    "n-samples": {"gsm8k_cot": {"effective": 1319}}})
            accepted, smoke, table, _ = benchmark_records(root, [])
            self.assertEqual(len(accepted), 2)
            self.assertFalse(smoke)
            row = next(row for row in table if row["condition"] == "full_final")
            self.assertAlmostEqual(row["gsm8k_cot_delta_pp"], 10)
            self.assertEqual(row["status"], "incomplete")  # Three benchmarks absent.

    def test_conflicting_log_snapshots_fail_instead_of_picking_a_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            write_json(run / "config.json", {"method": "full", "updates": 1250})
            for name, loss in [("train.jsonl", 1.0), ("train.other.jsonl", 2.0)]:
                (run / name).write_text(json.dumps({"update": 1, "sft_loss": loss, "lr": .01}))
            with self.assertRaises(ValueError):
                training_runs(run)

    def test_loaded_checkpoint_identity_overrides_a_default_cli_method(self):
        request = {"checkpoint": "/checkpoints/runs/adapters/step-1250", "method": "full"}
        loaded = {"checkpoint": request["checkpoint"], "method": "lora16", "update": 1250}
        self.assertEqual(condition_for(request, [], loaded), "lora16_final")
        loaded["checkpoint"] = "/different/checkpoint"
        with self.assertRaises(ValueError):
            condition_for(request, [], loaded)


if __name__ == "__main__":
    unittest.main()

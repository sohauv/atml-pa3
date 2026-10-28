from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

from task3_test_time.offline import (load_tables, make_orders, make_splits,
                                    policy_stop, run_dataset, tune_policy)
from common.io import read_json
from task3_test_time.selection import votes
from task3_test_time.answers import answer_groups


class OfflineTests(unittest.TestCase):
    def test_rwc_float_sum_ties_keep_earliest_support(self):
        rows = [{"answer_group": "first"}, {"answer_group": "other"}, {"answer_group": "first"}]
        winner, margin, _, unique = votes(rows, [.3, .7, .4])
        self.assertEqual(winner, "first")
        self.assertEqual(margin, 0)
        self.assertFalse(unique)

    def test_splits_are_question_disjoint_and_reproducible(self):
        ids = [str(i) for i in range(1000)]
        first = make_splits(ids)
        second = make_splits(list(reversed(ids)))
        self.assertEqual(first, second)
        self.assertFalse(set(first["development"]) & set(first["evaluation"]))
        self.assertEqual(first["live"], first["evaluation"][:50])
        self.assertEqual(len(first["development"]), 200)
        self.assertEqual(len(first["evaluation"]), 800)

    def test_orders_have_all_candidates_and_nested_prefixes(self):
        orders = make_orders(["q1", "q2"])
        self.assertEqual(orders, make_orders(["q2", "q1"]))
        for pool in orders.values():
            self.assertEqual(len(pool), 10)
            for order in pool:
                self.assertEqual(sorted(order), list(range(32)))
                self.assertEqual(order[:4], order[:16][:4])

    def test_attention_duplicate_length_field_is_not_misjoined(self):
        with patch("task3_test_time.offline.pd.read_parquet") as read:
            folder = Path("unused-in-memory-fixture")
            keys = {"dataset": ["MathVista"] * 32, "sample_id": ["1"] * 32, "rollout_id": list(range(32))}
            generations = pd.DataFrame({**keys, "question_text": ["q"] * 32, "gold_answer": ["a"] * 32,
                "generated_token_ids": [[1, 2]] * 32, "n_generated_text_tokens": [2] * 32})
            probability = pd.DataFrame({**keys, "logp": [[-1, -2]] * 32, "kl_u_p": [[1, 2]] * 32})
            attention = pd.DataFrame({**keys, "n_generated_text_tokens": [[0, 1]] * 32,
                "img_attn_total": [[.5, .4]] * 32, "prompt_text_attn_total": [[.5, .4]] * 32,
                "generated_text_attn_total": [[0, .2]] * 32})
            frames = {"generations.parquet": generations, "probability_metrics.parquet": probability,
                      "attention_metrics.parquet": attention}
            read.side_effect = lambda path: frames[path.name].copy()
            result = load_tables(folder, vision=True, expected_questions=1)
            self.assertEqual(result.n_generated_text_tokens.iloc[0], 2)
            self.assertEqual(list(result.preceding_generated_token_counts.iloc[0]), [0, 1])
            frames["probability_metrics.parquet"] = probability.iloc[:31]
            with self.assertRaises(ValueError):
                load_tables(folder, vision=True, expected_questions=1)

    def test_policy_tuning_uses_development_accuracy_constraint(self):
        def prefix(correct, cost, agreement=1):
            return {"correct": {"consistency": correct}, "cost": cost, "consensus": "A",
                    "rwc": "A", "rwc_margin": 1, "agreement": agreement, "unique": True}
        order = {4: prefix(False, 4, .5), 8: prefix(True, 8, .75),
                 16: prefix(True, 16, 1), 32: prefix(True, 32, 1)}
        tuned = tune_policy({"dev": [order] * 10}, False)
        self.assertEqual(tuned["selected"]["tau"], .625)
        self.assertEqual(tuned["selected"]["mean_tokens"], 8)
        self.assertEqual(policy_stop(order, None, None), 32)

    def test_analysis_pipeline_saves_frozen_policies_and_paired_results(self):
        # Synthetic test data stay in a temporary directory; these outputs are
        # never presented as assignment measurements.
        rows = []
        metadata = {}
        for question in range(10):
            sid = str(question)
            metadata[sid] = {"question_type": "multi_choice", "choices": ["red", "blue"], "answer_type": "text", "precision": 0,
                             "fixture_sid": sid}
            for rollout in range(32):
                rows.append({"sample_id": sid, "rollout_id": rollout, "question_text": "test fixture",
                    "gold_answer": "red", "extracted_answer": "A" if rollout % 3 else "B",
                    "generated_text": "test fixture response", "n_generated_text_tokens": 2,
                    "hit_max_new_tokens": False, "logp": np.array([-1., -2.]), "kl_u_p": np.array([1., 2.]),
                    "img_attn_total": np.array([.5, .4]), "prompt_text_attn_total": np.array([.5, .4]),
                    "generated_text_attn_total": np.array([0., .2])})
        table = pd.DataFrame(rows)
        with tempfile.TemporaryDirectory() as tmp, patch("task3_test_time.offline.load_tables", return_value=table), \
             patch("task3_test_time.offline.scorer_manifest", return_value={"rules": "unit-test-only"}), \
             patch("task3_test_time.offline.make_splits", side_effect=lambda ids: make_splits(ids, development=4)):
            output = Path(tmp) / "analysis"
            development_ids = set(make_splits(table.sample_id.unique(), development=4)["development"])
            def checked_grouping(extractions, dataset, question_metadata):
                if question_metadata["fixture_sid"] not in development_ids:
                    self.assertTrue((output / "policies.json").exists(), "Held-out evaluation started before policy freeze")
                return answer_groups(extractions, dataset, question_metadata)
            with patch("task3_test_time.offline.answer_groups", side_effect=checked_grouping):
                run_dataset(tmp, "mathvista", output, metadata)
            self.assertEqual(read_json(output / "complete.json")["held_out"], 6)
            self.assertEqual(set(read_json(output / "policies.json")), {"agreement", "quality_aware"})
            summary = pd.read_csv(output / "summary.csv")
            self.assertTrue({"first", "certainty_rwc", "visconf", "token_fusion_rwc"}.issubset(set(summary.method)))
            self.assertTrue((output / "per_order.jsonl.gz").exists())
            self.assertEqual(len(read_json(output / "paired_intervals.json")), 6 * summary.method.nunique() + 2)


if __name__ == "__main__":
    unittest.main()

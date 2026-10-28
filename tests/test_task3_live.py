"""Exercise orchestration/resume with fake generation, never real GPU allocation."""
from contextlib import nullcontext, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd

from common.io import read_json, write_json
from task3_test_time.live import question_directory, run_live


class LiveTests(unittest.TestCase):
    def test_saved_calls_resume_and_shared_initial_is_charged_twice(self):
        calls = []
        class Tokens:
            shape = (1, 3)

        class Batch(dict):
            def __init__(self):
                super().__init__(input_ids=Tokens())
                self.input_ids = self["input_ids"]

            def to(self, device):
                return self

        class Generated:
            def __getitem__(self, key):
                return SimpleNamespace(tolist=lambda: [42, 151645])

        class Tokenizer:
            eos_token_id = 151645
            pad_token_id = 151643

            def apply_chat_template(self, messages, **kwargs):
                return json.dumps(messages)

            def __call__(self, text, **kwargs):
                return Batch()

            def decode(self, ids, **kwargs):
                return r"answer \boxed{1}"

        class Model:
            generation_config = SimpleNamespace(eos_token_id=[151645, 151643])

            def cuda(self):
                return self

            def eval(self):
                return self

            def generate(self, **kwargs):
                calls.append(kwargs["max_new_tokens"])
                return Generated()

        fake_torch = SimpleNamespace(float16="float16", inference_mode=nullcontext,
            cuda=SimpleNamespace(is_available=lambda: True, synchronize=lambda: None,
                reset_peak_memory_stats=lambda: None, max_memory_allocated=lambda: 2**30,
                get_device_name=lambda index: "unit-test fake GPU"))
        fake_transformers = SimpleNamespace(
            AutoModelForCausalLM=SimpleNamespace(from_pretrained=lambda *a, **k: Model()),
            AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: Tokenizer()))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "qwen25-math").mkdir()
            (root / "qwen25-math/generations.parquet").write_bytes(b"unit-test-placeholder")
            ids = [str(i) for i in range(50)]
            split = root / "splits.json"
            write_json(split, {"evaluation": ids, "live": ids})
            frame = pd.DataFrame({"sample_id": ids, "question_text": ["test problem"] * 50,
                                  "gold_answer": ["1"] * 50})
            with patch.dict("sys.modules", {"torch": fake_torch, "transformers": fake_transformers,
                             "common.training": SimpleNamespace(seed_everything=lambda seed: None)}), \
                 patch("pandas.read_parquet", return_value=frame), \
                 patch("task3_test_time.live.scorer_manifest", return_value={"rules": "unit-test-only"}), \
                 redirect_stdout(io.StringIO()):
                summary = run_live(root, split, root / "live", max_seconds=1800)
                self.assertTrue(summary["complete"])
                self.assertEqual(len(calls), 250)
                result = read_json(question_directory(root / "live", ids[0]) / "result.json")
                self.assertEqual(result["strategies"]["parallel"]["output_tokens"], 6)
                self.assertEqual(result["strategies"]["refinement"]["output_tokens"], 4)
                self.assertEqual(result["strategies"]["single"]["output_tokens"], 2)
                revision = read_json(question_directory(root / "live", ids[0]) / "revision.json")
                self.assertEqual(len(revision["messages"]), 4)
                self.assertEqual(revision["messages"][-2]["role"], "assistant")
                # A complete resumed run performs no additional model calls.
                run_live(root, split, root / "live", max_seconds=1800)
                self.assertEqual(len(calls), 250)


if __name__ == "__main__":
    unittest.main()

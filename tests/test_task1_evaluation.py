"""Check evaluation precision/identity using a real tiny CPU Qwen3 checkpoint."""
import copy
import json
from pathlib import Path
import tempfile
import time
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from common.io import read_json, write_json
from common.training import seed_everything
from task1_finetuning.adapters import AdapterLinear, export_factors, inject
from task1_finetuning.evaluate import benchmarks, evaluate_job, load_model, positive_batch_size


class EvaluationBatchingTests(unittest.TestCase):
    def test_likelihood_batches_eight_while_generation_remains_one_and_cache_reuses(self):
        calls, constructed = [], []
        model = SimpleNamespace(_pa3_checkpoint_identity={"method": "baseline", "update": 0})
        fake_module = ModuleType("lm_eval")

        def evaluate(**kwargs):
            calls.append(kwargs)
            task = kwargs["tasks"][0]
            return {"results": {task: {"acc,none": .5}}, "samples": {}, "config": {}}

        def harness(*args, batch_size, **kwargs):
            constructed.append(batch_size)
            return SimpleNamespace(generation_records=[], batch_size=batch_size)

        fake_module.simple_evaluate = evaluate
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            with patch.dict("sys.modules", {"lm_eval": fake_module}), \
                 patch("task1_finetuning.evaluate.harness_model", side_effect=harness), \
                 patch("importlib.metadata.version", return_value="unit-test-version"):
                benchmarks(model, None, output, time.monotonic() + 60,
                           tasks=["gsm8k_cot", "hellaswag", "boolq"])
                self.assertEqual(constructed, [1, 8])
                self.assertEqual([call["batch_size"] for call in calls], [1, 8, 8])
                self.assertEqual([call["apply_chat_template"] for call in calls], [True, False, False])
                self.assertEqual(calls[0]["gen_kwargs"], "do_sample=False,max_gen_toks=1024")
                self.assertIsNone(calls[1]["gen_kwargs"])
                generation = read_json(output / "gsm8k_cot.json")
                likelihood = read_json(output / "hellaswag.json")
                self.assertEqual(generation["protocol"]["batch_size"], 1)
                self.assertEqual(likelihood["protocol"]["batch_size"], 8)
                self.assertEqual(likelihood["protocol"]["cache_identity"]["batch_size"], 8)
                self.assertEqual(likelihood["config"]["pa3_batch_sizes"], {"generation": 1, "likelihood": 8})
                self.assertEqual(read_json(output / "evaluation_config.json")["likelihood_batch_size"], 8)
                before = (output / "evaluation_config.json").read_bytes()
                # A completed task incurs no repeated model/harness evaluation.
                benchmarks(model, None, output, time.monotonic() + 60,
                           tasks=["gsm8k_cot", "hellaswag", "boolq"])
                self.assertEqual(len(calls), 3)
                self.assertEqual(constructed, [1, 8])
                with self.assertRaisesRegex(ValueError, "configuration differs"):
                    benchmarks(model, None, output, time.monotonic() + 60,
                               tasks=["boolq"], likelihood_batch_size=4)
                self.assertEqual((output / "evaluation_config.json").read_bytes(), before)

    def test_legacy_cached_likelihood_is_batch_one_and_cannot_silently_mix(self):
        fake_module = ModuleType("lm_eval")
        fake_module.simple_evaluate = lambda **kwargs: self.fail("Completed legacy results should be reused")
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            result_path = output / "boolq.json"
            write_json(result_path, {"results": {"boolq": {"acc,none": .5}},
                                    "protocol": {"example_limit": None, "complete_split": True, "smoke_only": False}})
            before = result_path.read_bytes()
            with patch.dict("sys.modules", {"lm_eval": fake_module}):
                with self.assertRaisesRegex(ValueError, "batch size differs"):
                    benchmarks(None, None, output, time.monotonic() + 60, tasks=["boolq"])
                self.assertFalse((output / "evaluation_config.json").exists())
                benchmarks(None, None, output, time.monotonic() + 60,
                           tasks=["boolq"], likelihood_batch_size=1)
            self.assertEqual(result_path.read_bytes(), before)
            self.assertEqual(read_json(output / "evaluation_config.json")["likelihood_batch_size"], 1)

    def test_resume_batch_mismatch_is_rejected_before_loading_gpu_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            previous = {"job": "evaluate1", "config": {}, "checkpoint": None}
            write_json(output / "request.json", previous)
            with patch("task1_finetuning.evaluate.load_model") as load:
                with self.assertRaisesRegex(ValueError, "likelihood batch size differs"):
                    evaluate_job({**previous, "eval_likelihood_batch_size": 8}, output, 300)
                load.assert_not_called()

    def test_batch_size_requires_positive_integer(self):
        for value in (0, -1, True, False, 1.5, "8"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                positive_batch_size(value)


class Batch(dict):
    def to(self, device):
        return self


class Tokenizer:
    def __call__(self, text, **kwargs):
        return Batch(input_ids=torch.tensor([[1, 2, 3, 4]]), attention_mask=torch.ones(1, 4, dtype=torch.long))


class EvaluationPrecisionTests(unittest.TestCase):
    def test_adapter_inference_is_fp16_but_diagnostics_keep_master_updates_fp32(self):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous_threads)
        seed_everything(6304)
        config = Qwen3Config(vocab_size=97, hidden_size=32, intermediate_size=64,
                            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
                            head_dim=8, max_position_embeddings=64, sliding_window=None,
                            use_sliding_window=False, use_cache=False)
        config._attn_implementation = "eager"
        starting = Qwen3ForCausalLM(config)
        adapted = copy.deepcopy(starting)
        inject(adapted, 4, 8)
        with torch.no_grad():
            for module in adapted.modules():
                if isinstance(module, AdapterLinear):
                    module.B.normal_(std=.001)
        factors = export_factors(adapted)
        metadata = {"sha256": "test-hash", "revisions": {"model": "test-revision"}}
        original_autocast = torch.autocast

        def cpu_autocast(device_type, *args, **kwargs):
            return original_autocast("cpu" if device_type == "cuda" else device_type, *args, **kwargs)

        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "step-1250"
            checkpoint.mkdir()
            write_json(checkpoint / "complete.json", {"update": 1250})
            torch.save({"config": {"model_revision": "test-revision", "method": "lora4",
                         "method_config": {"rank": 4, "alpha": 8}}, "update": 1250,
                        "data_sha256": "test-hash"}, checkpoint / "recovery.pt")
            torch.save(factors, checkpoint / "projected_factors.pt")
            with patch("task1_finetuning.evaluate.read_json", return_value=metadata), \
                 patch("task1_finetuning.evaluate.AutoTokenizer.from_pretrained", return_value=Tokenizer()), \
                 patch("task1_finetuning.evaluate.AutoModelForCausalLM.from_pretrained", side_effect=lambda *a, **k: copy.deepcopy(starting)), \
                 patch.object(torch.nn.Module, "cuda", lambda self: self), \
                 patch.object(torch, "autocast", side_effect=cpu_autocast):
                inference, _, verified = load_model({"model": "random-test-qwen"}, checkpoint)
                diagnostic, _, no_inference_check = load_model({"model": "random-test-qwen"}, checkpoint, diagnostic=True)
            self.assertEqual(inference.dtype, torch.float16)
            self.assertEqual(verified["dtype"], "float16")
            self.assertEqual(diagnostic.dtype, torch.float32)
            self.assertIsNone(no_inference_check)
            self.assertEqual(inference._pa3_checkpoint_identity["method"], "lora4")
            self.assertEqual(inference._pa3_checkpoint_identity["update"], 1250)
            name = "model.layers.0.self_attn.q_proj"
            expected = starting.get_submodule(name).weight + factors[name]["B"] @ factors[name]["A"] * factors[name]["scale"]
            torch.testing.assert_close(diagnostic.get_submodule(name).weight, expected, atol=0, rtol=0)


class ResultSerializationTests(unittest.TestCase):
    def test_harness_torch_metadata_is_json_serializable(self):
        import json
        from task1_finetuning.evaluate import json_value
        original = {"config": {"dtype": torch.float16, "device": torch.device("cpu")},
                    "nested": [torch.tensor([1., 2.]), {"dtype": torch.float32}]}
        saved = json.loads(json.dumps(json_value(original)))
        self.assertEqual(saved["config"], {"dtype": "torch.float16", "device": "cpu"})
        self.assertEqual(saved["nested"], [[1., 2.], {"dtype": "torch.float32"}])


if __name__ == "__main__":
    unittest.main()

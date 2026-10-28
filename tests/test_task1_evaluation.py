"""Check evaluation precision/identity using a real tiny CPU Qwen3 checkpoint."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from common.io import write_json
from common.training import seed_everything
from task1_finetuning.adapters import AdapterLinear, export_factors, inject
from task1_finetuning.evaluate import load_model


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


if __name__ == "__main__":
    unittest.main()

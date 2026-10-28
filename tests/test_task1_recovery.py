"""CPU integration tests use random tiny Qwen3 models, never network weights."""
import copy
from contextlib import nullcontext
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from common.training import causal_loss, restore_rng, seed_everything
from task1_finetuning.adapters import export_factors, inject, merge_factors
from task1_finetuning.train import save_checkpoint, train


def tiny_qwen():
    return Qwen3ForCausalLM(Qwen3Config(
        vocab_size=97, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=64, tie_word_embeddings=True,
        attention_dropout=0.0, use_cache=False))


class Task1IntegrationTests(unittest.TestCase):
    def setUp(self):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous_threads)
        seed_everything(6304)
        self.ids = torch.tensor([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]])
        self.labels = self.ids.clone()
        self.labels[:, :3] = -100

    def test_tiny_qwen_all_methods_backward_and_merge(self):
        for method, rank, protected in [("full", 0, 0), ("l2sp", 0, 0),
                                         ("lora4", 4, 0), ("lora16", 16, 0), ("oplora16", 16, 3)]:
            with self.subTest(method=method):
                seed_everything(6304)
                model = tiny_qwen()
                fresh = copy.deepcopy(model)
                reference = {n: p.detach().clone() for n, p in model.named_parameters()}
                if rank:
                    count = inject(model, rank, rank * 2, protected)
                    self.assertEqual(count, 14)
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                model.train()
                optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
                loss = causal_loss(model(self.ids).logits, self.labels) / 6
                if method == "l2sp":
                    loss = loss + .001 * sum(.5 * (p - reference[n]).square().sum()
                                            for n, p in model.named_parameters())
                loss.backward()
                grads = [p.grad for p in model.parameters() if p.requires_grad]
                self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads))
                optimizer.step()
                for state in optimizer.state.values():
                    self.assertEqual(state["exp_avg"].dtype, torch.float32)
                    self.assertEqual(state["exp_avg_sq"].dtype, torch.float32)
                if rank:
                    for name, p in model.named_parameters():
                        if not p.requires_grad:
                            old_name = name.replace(".base.", ".")
                            torch.testing.assert_close(p, reference[old_name])
                    model.eval()
                    merge_factors(fresh, export_factors(model))
                    fresh.eval()
                    with torch.no_grad():
                        torch.testing.assert_close(model(self.ids).logits, fresh(self.ids).logits,
                                                   atol=2e-5, rtol=2e-4)

    def test_adapter_checkpoint_preserves_rng_optimizer_and_projected_export(self):
        model = tiny_qwen()
        inject(model, 4, 8, 3)
        model.train()
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
        scaler = torch.amp.GradScaler("cpu")
        scaler.scale(causal_loss(model(self.ids).logits, self.labels) / 6).backward()
        scaler.step(optimizer)
        scaler.update()
        calls = []
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "step-0001"
            save_checkpoint(model, optimizer, scaler, checkpoint, update=1,
                            config={"method_config": {"rank": 4}}, metadata={"sha256": "unit-test"},
                            wall_seconds=1.0, reference=None, commit=lambda: calls.append(True))
            saved = torch.load(checkpoint / "recovery.pt", weights_only=False)
            self.assertEqual(saved["data_position"], 32)
            self.assertEqual(saved["scheduler"]["next_update"], 1)
            self.assertTrue((checkpoint / "complete.json").exists())
            self.assertEqual(calls, [True])
            self.assertTrue(all(k.endswith((".A", ".B", ".U", ".V")) for k in saved["model"]))
            self.assertTrue((checkpoint / "projected_factors.pt").exists())
            expected = torch.rand(10)
            restore_rng(saved["rng"])
            torch.testing.assert_close(torch.rand(10), expected)
            with self.assertRaises(FileExistsError):
                save_checkpoint(model, optimizer, scaler, checkpoint, update=1,
                                config={"method_config": {"rank": 4}}, metadata={"sha256": "unit-test"},
                                wall_seconds=1.0, reference=None, commit=lambda: None)

    def test_training_loop_resume_matches_uninterrupted_optimizer_trajectory(self):
        from common.io import sha256, write_json
        from common.training import collate
        import task1_finetuning.train as training_module
        starting = tiny_qwen()
        scaler_type = torch.amp.GradScaler
        config = {"model": "unit-test-random-qwen3", "seed": 6304, "examples": 128,
                  "updates": 4, "microbatch": 2, "accumulation": 16,
                  "warmup_updates": 1, "weight_decay": .01, "betas": [.9, .999],
                  "eps": 1e-8, "clip_norm": 1.0, "checkpoints": [2, 4],
                  "methods": {"lora4": {"lr": .001, "rank": 4, "alpha": 8}}}
        class Tokenizer:
            pad_token_id = 0
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            data.mkdir()
            with (data / "train.jsonl").open("w", encoding="utf-8") as stream:
                for index in range(128):
                    ids = [1 + index % 8, 2, 3, 4, 5, 6]
                    stream.write(json.dumps({"source_id": index, "input_ids": ids,
                                             "labels": [-100, -100, -100, 4, 5, 6]}) + "\n")
            write_json(data / "metadata.json", {"sha256": sha256(data / "train.jsonl"),
                                               "revisions": {"model": "unit-test-revision"}})
            # Only device transport/AMP are replaced. The real loop, optimizer,
            # accumulated losses, dropout, data order, and checkpoint code run.
            patches = [patch.object(training_module.AutoTokenizer, "from_pretrained", return_value=Tokenizer()),
                       patch.object(training_module.AutoModelForCausalLM, "from_pretrained", side_effect=lambda *a, **k: copy.deepcopy(starting)),
                       patch.object(torch.nn.Module, "cuda", lambda self: self),
                       patch.object(torch.Tensor, "half", lambda self: self),
                       patch.object(torch.cuda, "is_available", return_value=True),
                       patch.object(torch.cuda, "is_current_stream_capturing", return_value=False),
                       patch.object(torch.cuda, "get_rng_state_all", return_value=[]),
                       patch.object(torch.cuda, "set_rng_state_all", return_value=None),
                       patch.object(torch.cuda, "manual_seed_all", return_value=None),
                       patch.object(torch.cuda, "reset_peak_memory_stats", return_value=None),
                       patch.object(torch.cuda, "synchronize", return_value=None),
                       patch.object(torch.cuda, "max_memory_allocated", return_value=0),
                       patch.object(torch.cuda, "get_device_name", return_value="CPU test device transport"),
                       patch.object(training_module, "collate", side_effect=lambda records, pad: collate(records, pad, "cpu")),
                       patch.object(torch, "autocast", side_effect=lambda *a, **k: nullcontext()),
                       patch.object(torch.amp, "GradScaler", side_effect=lambda *a, **k: scaler_type("cpu"))]
            from contextlib import ExitStack
            with ExitStack() as stack:
                for mocked in patches:
                    stack.enter_context(mocked)
                whole = train("lora4", root / "whole", config, data_dir=data)
                part = train("lora4", root / "split", config, data_dir=data, max_updates=2)
                resumed = train("lora4", root / "split", config, data_dir=data, resume=part["checkpoint"])
            self.assertTrue(whole["finished"])
            self.assertTrue(resumed["finished"])
            a = torch.load(Path(whole["checkpoint"]) / "recovery.pt", weights_only=False)
            b = torch.load(Path(resumed["checkpoint"]) / "recovery.pt", weights_only=False)
            for name in a["model"]:
                torch.testing.assert_close(a["model"][name], b["model"][name], atol=0, rtol=0)
            for index in a["optimizer"]["state"]:
                for name in ("step", "exp_avg", "exp_avg_sq"):
                    torch.testing.assert_close(a["optimizer"]["state"][index][name],
                                               b["optimizer"]["state"][index][name], atol=0, rtol=0)
            torch.testing.assert_close(a["rng"]["torch"], b["rng"]["torch"], atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()

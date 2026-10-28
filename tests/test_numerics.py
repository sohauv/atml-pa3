import unittest
from unittest.mock import patch
import torch
from torch import nn

from common.training import causal_loss, collate, lr_factor, parameter_groups
from task1_finetuning.adapters import AdapterLinear, export_factors, inject, merge_factors


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Module()
        self.layer.q_proj = nn.Linear(12, 10)

    def enable_input_require_grads(self):
        pass

    def forward(self, x):
        return self.layer.q_proj(x)


class NumericalTests(unittest.TestCase):
    def test_token_weighted_accumulation_matches_effective_batch(self):
        torch.manual_seed(12)
        logits = torch.randn(3, 6, 9, requires_grad=True)
        labels = torch.tensor([[-100, -100, 1, 2, 3, 4], [-100, -100, -100, -100, 1, 2],
                               [-100, 3, 4, 5, 6, 7]])
        n = (labels[:, 1:] != -100).sum()
        whole = causal_loss(logits, labels) / n
        expected = torch.autograd.grad(whole, logits, retain_graph=True)[0]
        pieces = sum(causal_loss(logits[i:i+1], labels[i:i+1]) / n for i in range(3))
        actual = torch.autograd.grad(pieces, logits)[0]
        torch.testing.assert_close(actual, expected)

    def test_example_weighting_is_distinct_and_correct(self):
        logits = torch.randn(2, 4, 5)
        labels = torch.tensor([[-100, 1, 2, 3], [-100, -100, -100, 4]])
        expected = (causal_loss(logits[:1], labels[:1]) / 3 + causal_loss(logits[1:], labels[1:])) / 2
        torch.testing.assert_close(causal_loss(logits, labels, "examples") / 2, expected)

    def test_oplora_constraints_and_gradients(self):
        layer = AdapterLinear(nn.Linear(12, 10), 4, 8, protection_rank=3)
        with torch.no_grad():
            layer.B.normal_()
        b, a = layer.factors()
        delta = b @ a
        self.assertLess((layer.U.T @ delta).abs().max().item(), 1e-5)
        self.assertLess((delta @ layer.V).abs().max().item(), 1e-5)
        layer(torch.randn(2, 12)).square().mean().backward()
        self.assertTrue(torch.isfinite(layer.A.grad).all())
        self.assertTrue(torch.isfinite(layer.B.grad).all())

    def test_initial_factors_match_and_export_merge_is_equivalent(self):
        torch.manual_seed(1)
        normal = TinyModel()
        protected = TinyModel()
        protected.load_state_dict(normal.state_dict())
        fresh = TinyModel()
        fresh.load_state_dict(normal.state_dict())
        inject(normal, 4, 8)
        inject(protected, 4, 8, 3)
        torch.testing.assert_close(normal.layer.q_proj.A, protected.layer.q_proj.A)
        torch.testing.assert_close(normal.layer.q_proj.B, protected.layer.q_proj.B)
        with torch.no_grad():
            protected.layer.q_proj.B.normal_()
        protected.eval()
        x = torch.randn(3, 12)
        merge_factors(fresh, export_factors(protected))
        torch.testing.assert_close(fresh(x), protected(x), atol=1e-5, rtol=1e-5)

    def test_padding_is_masked(self):
        batch, labels = collate([{"input_ids": [1, 2], "labels": [-100, 2]},
                                 {"input_ids": [1], "labels": [-100]}], 0, "cpu")
        self.assertEqual(labels.tolist(), [[-100, 2], [-100, -100]])
        self.assertEqual(batch["attention_mask"].tolist(), [[1, 1], [1, 0]])

    def test_resume_reuses_saved_bases_without_svd(self):
        original = TinyModel()
        restored = TinyModel()
        restored.load_state_dict(original.state_dict())
        inject(original, 4, 8, 3)
        with patch("torch.linalg.svd", side_effect=AssertionError("SVD must not repeat on resume")):
            inject(restored, 4, 8, 3, basis_state=original.state_dict())
        restored.load_state_dict(original.state_dict())
        torch.testing.assert_close(restored.layer.q_proj.U, original.layer.q_proj.U)
        torch.testing.assert_close(restored.layer.q_proj.V, original.layer.q_proj.V)

    def test_weight_decay_exclusions(self):
        model = nn.Sequential(nn.Linear(3, 4), nn.LayerNorm(4))
        groups = parameter_groups(model, .01)
        self.assertEqual([tuple(p.shape) for p in groups[0]["params"]], [(4, 3)])
        self.assertEqual(len(groups[1]["params"]), 3)

    def test_schedule_boundaries(self):
        self.assertAlmostEqual(lr_factor(0, 1250, 38), 1/38)
        self.assertAlmostEqual(lr_factor(37, 1250, 38), 1)
        self.assertAlmostEqual(lr_factor(1249, 1250, 38), 0)
        self.assertAlmostEqual(lr_factor(749, 750, 38, .1), .1)


if __name__ == "__main__":
    unittest.main()

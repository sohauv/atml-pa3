import copy
import unittest

import torch

from common.offload import CPUOffloadedAdamW


class OffloadAdamWTests(unittest.TestCase):
    def test_matches_adamw_with_clipping_groups_missing_gradients_and_resume(self):
        torch.manual_seed(6304)
        original = [torch.nn.Parameter(torch.randn(3, 4)), torch.nn.Parameter(torch.randn(4))]
        offloaded = [torch.nn.Parameter(p.detach().clone()) for p in original]
        def groups(parameters):
            return [{"params": [parameters[0]], "weight_decay": .01},
                    {"params": [parameters[1]], "weight_decay": 0.0}]
        regular = torch.optim.AdamW(groups(original), lr=2e-5, betas=(.9, .999), eps=1e-8, foreach=False)
        host = CPUOffloadedAdamW(groups(offloaded), lr=2e-5, betas=(.9, .999), eps=1e-8)
        for update in range(5):
            for a, b in zip(original, offloaded):
                grad = torch.randn_like(a)
                a.grad, b.grad = grad.clone(), grad.clone()
            if update == 1:
                original[1].grad = offloaded[1].grad = None
            torch.nn.utils.clip_grad_norm_(original, 1.0)
            torch.nn.utils.clip_grad_norm_(offloaded, 1.0)
            for optimizer in (regular, host):
                for group in optimizer.param_groups:
                    group["lr"] = 2e-5 * (1 - update / 10)
                optimizer.step()
            for a, b in zip(original, offloaded):
                torch.testing.assert_close(a, b, atol=0, rtol=0)
                for key in ("step", "exp_avg", "exp_avg_sq"):
                    torch.testing.assert_close(regular.state[a][key], host.state[b][key], atol=0, rtol=0)
                    self.assertEqual(host.state[b][key].device.type, "cpu")
            if update == 2:
                state = copy.deepcopy(host.state_dict())
                host = CPUOffloadedAdamW(groups(offloaded), lr=.1)
                host.load_state_dict(state)
        # Standard AdamW can consume the same state schema.
        plain = torch.optim.AdamW(groups(offloaded))
        plain.load_state_dict(host.state_dict())
        self.assertEqual(len(plain.state), 2)

    def test_grad_scaler_skips_infinite_device_gradients(self):
        p = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = CPUOffloadedAdamW([p], lr=.01)
        scaler = torch.amp.GradScaler("cpu")
        scaler.scale(p.square().sum()).backward()
        p.grad.fill_(float("inf"))
        before = p.detach().clone()
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()
        torch.testing.assert_close(p, before, atol=0, rtol=0)
        self.assertEqual(len(optimizer.state), 0)

    def test_fp16_master_and_capturable_modes_are_refused(self):
        with self.assertRaisesRegex(ValueError, "FP32"):
            CPUOffloadedAdamW([torch.nn.Parameter(torch.ones(2, dtype=torch.float16))])
        with self.assertRaisesRegex(ValueError, "noncapturable"):
            CPUOffloadedAdamW([torch.nn.Parameter(torch.ones(2))], capturable=True)

from pathlib import Path
import tempfile
import unittest

import torch

from common.cache_profile import load_feature_cpu, profile_feature, valid_feature_cpu


class CacheProfileTests(unittest.TestCase):
    def test_cpu_validation_rejects_nonfinite_and_wrong_layout(self):
        value = torch.zeros(196, 768, dtype=torch.float16)
        self.assertTrue(valid_feature_cpu(value))
        self.assertFalse(valid_feature_cpu(value.float()))
        self.assertFalse(valid_feature_cpu(value[:1]))
        for invalid in (float("nan"), float("inf"), -float("inf")):
            value[0, 0] = invalid
            self.assertFalse(valid_feature_cpu(value))
            self.assertFalse(bool(torch.isfinite(value).all()))

    def test_cpu_profile_records_validity_and_both_timing_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sample.pt"
            torch.save(torch.ones(196, 768, dtype=torch.float16), path)
            result = profile_feature(path)
            torch.testing.assert_close(load_feature_cpu(path), torch.load(path, weights_only=True), atol=0, rtol=0)
            self.assertTrue(result["bulk_feature_valid"])
            self.assertTrue(result["valid_feature"])
            self.assertTrue(result["finite"])
            self.assertEqual(result["shape"], [196, 768])
            self.assertGreaterEqual(result["read_seconds"], 0)
            self.assertGreaterEqual(result["torch_finite_seconds"], 0)
            self.assertGreaterEqual(result["numpy_validation_seconds"], 0)

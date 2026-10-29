from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from common.cache_profile import (load_feature_cpu, load_verified_index, profile_feature,
                                  valid_feature_cpu, verify_feature_cache)
from common.io import write_json
from task2_vlm.model import save_feature


class CacheProfileTests(unittest.TestCase):
    def test_incremental_verification_reads_only_new_tensor_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_json(root / "provenance.json", {"revision": "fixed"})
            save_feature(root / "old.pt", torch.ones(196, 768, dtype=torch.float16))
            original = verify_feature_cache(root)
            save_feature(root / "new.pt", torch.zeros(196, 768, dtype=torch.float16))
            read_bytes = Path.read_bytes
            def checked_read(path):
                if path.name == "old.pt":
                    self.fail("An immutable already-verified tensor was downloaded again")
                return read_bytes(path)
            with patch.object(Path, "read_bytes", checked_read):
                result = verify_feature_cache(root, verified_index=original["index_path"])
            self.assertEqual(result["verified_images"], 2)
            self.assertEqual(result["newly_verified_images"], 1)

    def test_cpu_index_attests_bytes_and_rejects_changed_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_json(root / "provenance.json", {"revision": "fixed"})
            save_feature(root / "train2014-1.pt", torch.ones(196, 768, dtype=torch.float16))
            save_feature(root / "train2014-2.pt", torch.zeros(196, 768, dtype=torch.float16))
            result = verify_feature_cache(root)
            self.assertEqual(result["verified_images"], 2)
            entries = load_verified_index(result["index_path"], root)
            self.assertEqual(entries["train2014-1.pt"]["bytes"], (root / "train2014-1.pt").stat().st_size)
            self.assertEqual(verify_feature_cache(root)["index_sha256"], result["index_sha256"])
            write_json(root / "provenance.json", {"revision": "different"})
            with self.assertRaisesRegex(ValueError, "provenance"):
                load_verified_index(result["index_path"], root)

    def test_invalid_cache_never_publishes_verification_index(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_json(root / "provenance.json", {"revision": "fixed"})
            value = torch.ones(196, 768, dtype=torch.float16)
            value[0, 0] = float("nan")
            torch.save(value, root / "bad.pt")
            with self.assertRaisesRegex(ValueError, "Invalid"):
                verify_feature_cache(root)
            self.assertEqual(list(root.glob("verified-index-*.json")), [])

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

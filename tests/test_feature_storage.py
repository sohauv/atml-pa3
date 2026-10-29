from pathlib import Path
import tempfile
import unittest

import torch

from task2_vlm.model import save_feature


class FeatureStorageTests(unittest.TestCase):
    def test_batch_view_serializes_only_one_image_and_preserves_values(self):
        batch = torch.arange(32 * 196 * 768, dtype=torch.float32).reshape(32, 196, 768).remainder(37).half()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.pt"
            save_feature(path, batch[17])
            self.assertLess(path.stat().st_size, 2 * batch[17].numel() * batch.element_size())
            actual = torch.load(path, weights_only=True)
            torch.testing.assert_close(actual, batch[17], atol=0, rtol=0)
            self.assertEqual(actual.untyped_storage().nbytes(), actual.numel() * actual.element_size())
            original = path.read_bytes()
            with self.assertRaises(FileExistsError):
                save_feature(path, batch[18])
            self.assertEqual(path.read_bytes(), original)

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import tempfile
import unittest

import torch

from task2_vlm.model import load_cache_image, save_feature


class FeatureStorageTests(unittest.TestCase):
    def test_parallel_image_reads_preserve_order_and_exact_rgb_pixels(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "train2014").mkdir()
            rows = []
            for index, mode in enumerate(("RGB", "L", "RGBA", "RGB")):
                color = index * 50 if mode == "L" else ((index * 50, 20, 30, 255) if mode == "RGBA" else (index * 50, 20, 30))
                name = f"{index}.png"
                Image.new(mode, (5, 7), color).save(root / "train2014" / name)
                rows.append({"split": "train2014", "image_file": name})
            expected = []
            for row in rows:
                with Image.open(root / row["split"] / row["image_file"]) as image:
                    expected.append(image.convert("RGB").tobytes())
            with ThreadPoolExecutor(max_workers=3) as pool:
                actual = list(pool.map(lambda row: load_cache_image(root, row), rows))
            self.assertEqual([image.mode for image in actual], ["RGB"] * 4)
            self.assertEqual([image.tobytes() for image in actual], expected)

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

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from scripts.kaggle_task1 import select_original_input, unpack_inputs


class KaggleTransferTests(unittest.TestCase):
    def archive(self, root, extra=False):
        data = b'{"source_id": 1}\n'
        path = root / "input.zip"
        with zipfile.ZipFile(path, "w") as z:
            z.writestr("task1/train.jsonl", data)
            z.writestr("task1/metadata.json", json.dumps({"sha256": hashlib.sha256(data).hexdigest()}))
            z.writestr("task1/source_ids.json", "[1]")
            if extra:
                z.writestr("../unsafe", "blocked")
        return path

    def test_exact_reuse_and_changed_bytes_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = self.archive(root)
            data = unpack_inputs(archive, root / "out")
            self.assertEqual(unpack_inputs(archive, root / "out"), data)
            (data / "train.jsonl").write_bytes(b"changed")
            with self.assertRaises(FileExistsError):
                unpack_inputs(archive, root / "out")

    def test_unexpected_or_traversal_member_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(ValueError):
                unpack_inputs(self.archive(root, extra=True), root / "out")
            self.assertFalse((root / "unsafe").exists())

    def test_identical_extracted_copies_are_safe_but_changed_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = self.archive(root)
            first = unpack_inputs(archive, root / "a")
            second = unpack_inputs(archive, root / "b")
            digest = hashlib.sha256((first / "train.jsonl").read_bytes()).hexdigest()
            self.assertEqual(select_original_input(root, digest), first)
            (second / "source_ids.json").write_text("[2]")
            with self.assertRaisesRegex(ValueError, "copies differ"):
                select_original_input(root, digest)

    def test_wrong_original_hash_is_rejected_even_with_matching_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            unpack_inputs(self.archive(root), root / "a")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                select_original_input(root, "0" * 64)

import hashlib
from pathlib import Path
import tempfile
import unittest
import zipfile

from scripts.kaggle_checkpoint import restore_verified_inputs


class CheckpointInputTests(unittest.TestCase):
    def fixture(self, root):
        files = {"run/resume-0804/complete.json": b'{"update":804}',
                 "run/resume-0804/recovery.pt": b"optimizer RNG adapters",
                 "run/status.json": b'{"update":804}'}
        archive = root / "input.zip"
        with zipfile.ZipFile(archive, "w") as z:
            for name, value in files.items():
                z.writestr(name, value)
        return archive, {name: hashlib.sha256(value).hexdigest() for name, value in files.items()}

    def test_raw_zip_restore_and_overwrite_guard(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            _, expected = self.fixture(root)
            checkpoint = restore_verified_inputs(root, root / "out", "resume-0804", expected)
            self.assertEqual((checkpoint / "recovery.pt").read_bytes(), b"optimizer RNG adapters")
            with self.assertRaises(FileExistsError):
                restore_verified_inputs(root, root / "out", "resume-0804", expected)

    def test_identical_extracted_and_zip_copies_then_corruption_fails_before_write(self):
        with tempfile.TemporaryDirectory() as t, tempfile.TemporaryDirectory() as output_temp:
            root = Path(t)
            archive, expected = self.fixture(root)
            with zipfile.ZipFile(archive) as z:
                z.extractall(root / "copy-a")
                z.extractall(root / "copy-b")
            restore_verified_inputs(root, Path(output_temp) / "verified", "resume-0804", expected)
            (root / "copy-b/run/resume-0804/recovery.pt").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                restore_verified_inputs(root, root / "rejected", "resume-0804", expected)
            self.assertFalse((root / "rejected").exists())

    def test_missing_checkpoint_and_unsafe_manifest_fail_closed(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            with self.assertRaisesRegex(ValueError, "No extracted or zipped"):
                restore_verified_inputs(root, root / "out", "resume-0804", {"run/resume-0804/complete.json": "0" * 64})
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                restore_verified_inputs(root, root / "out", "resume-0804", {"run/../complete.json": "0" * 64})

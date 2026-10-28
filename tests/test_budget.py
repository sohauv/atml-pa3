import json
from pathlib import Path
import tempfile
import unittest

from common.budget import ceiling_cost, reserve


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.approval = self.root / "approval.json"
        self.ledger = self.root / "budget.json"
        self.value = {"authorization_id": "unit-test-only", "approved": True,
                      "approved_by": "unit test", "limit_usd": 1.0,
                      "expires_at": "2099-01-01T00:00:00Z", "allowed_gpus": ["L4"]}
        self.approval.write_text(json.dumps(self.value))

    def tearDown(self):
        self.tmp.cleanup()

    def reserve(self):
        return reserve(self.approval, self.ledger, gpu="L4", seconds=1000, purpose="test")

    def test_reservations_are_cumulative_and_fail_closed(self):
        self.reserve()
        self.reserve()
        with self.assertRaises(PermissionError):
            self.reserve()
        self.assertEqual(len(json.loads(self.ledger.read_text())["jobs"]), 2)
        self.assertFalse(self.ledger.with_suffix(".lock").exists())

    def test_no_approval_no_call(self):
        self.value["approved"] = False
        self.approval.write_text(json.dumps(self.value))
        with self.assertRaises(PermissionError):
            self.reserve()
        self.assertFalse(self.ledger.exists())

    def test_expired_approval(self):
        self.value["expires_at"] = "2000-01-01T00:00:00Z"
        self.approval.write_text(json.dumps(self.value))
        with self.assertRaises(PermissionError):
            self.reserve()

    def test_gpu_not_authorized(self):
        with self.assertRaises(PermissionError):
            reserve(self.approval, self.ledger, gpu="H100", seconds=100, purpose="test")

    def test_lock_fails_closed(self):
        lock = self.ledger.with_suffix(".lock")
        lock.touch()
        with self.assertRaises(FileExistsError):
            self.reserve()
        self.assertTrue(lock.exists())

    def test_invalid_resource_ceiling(self):
        for seconds in (-1, 0, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                ceiling_cost("L4", seconds)
        with self.assertRaises(ValueError):
            ceiling_cost("L4", 100, cpu=-1)


if __name__ == "__main__":
    unittest.main()

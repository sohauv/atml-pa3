import copy
from pathlib import Path
import tempfile
import unittest

from common.budget import reserve, reserve_queue
from common.experiment_queue import run_queue
from common.io import read_json, write_json
from scripts.queue import plan_queue, sync_status


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.approval = {"approved": True, "approved_by": "unit test only", "authorization_id": "test",
                         "limit_usd": 2, "expires_at": "2099-01-01T00:00:00Z", "allowed_gpus": ["A100-40GB"]}
        write_json(self.root / "approval.json", self.approval)
        self.resources = [{"gpu": None, "seconds": 1000, "cpu": .125, "memory_gib": .5, "purpose": "controller"},
                          {"gpu": "A100-40GB", "seconds": 300, "cpu": 4, "memory_gib": 32, "purpose": "probe"}]
        self.request = {"queue_id": "testqueue", "existing_call_id": "old", "deadline": 10000,
                        "steps": [{"request": {"job": "probe1"}, "ticket": {"id": "ticket1", "gpu": "A100-40GB", "seconds": 300}}]}

    def run_queue(self, request=None, **overrides):
        defaults = {"submit": lambda payload, gpu, seconds: "newcall", "get": lambda call_id, timeout: {"complete": True},
                    "cancel": lambda call_id: None, "clock": lambda: 1000}
        defaults.update(overrides)
        return run_queue(request or self.request, self.root / "remote", **defaults)

    def test_existing_job_finishes_before_next_gpu_and_completed_queue_is_idempotent(self):
        events = []
        def get(call_id, timeout):
            events.append("get:" + call_id)
            return {"complete": True}
        def submit(payload, gpu, seconds):
            events.append("submit")
            self.assertLessEqual(payload["ticket"]["deadline"], self.request["deadline"])
            return "newcall"
        result = self.run_queue(get=get, submit=submit)
        self.assertEqual(events, ["get:old", "submit", "get:newcall"])
        self.assertEqual(result["phase"], "completed")
        self.run_queue(submit=lambda *a: self.fail("completed queue relaunched"), get=lambda *a: self.fail("unexpected call"))

    def test_resume_waiting_step_never_resubmits(self):
        path = self.root / "remote"
        write_json(path / "request.json", self.request)
        write_json(path / "status.json", {"queue_id": "testqueue", "phase": "waiting_for_step", "next_index": 0,
                                          "steps": [], "call_id": "already-submitted"})
        result = self.run_queue(submit=lambda *a: self.fail("duplicate GPU submission"))
        self.assertEqual(result["steps"][0]["call_id"], "already-submitted")

    def test_ambiguous_submission_stops_without_duplicate(self):
        path = self.root / "remote"
        write_json(path / "request.json", self.request)
        write_json(path / "status.json", {"queue_id": "testqueue", "phase": "submitting", "next_index": 0, "steps": []})
        result = self.run_queue(submit=lambda *a: self.fail("duplicate"))
        self.assertEqual(result["phase"], "needs_reconciliation")

    def test_failed_dependency_never_launches_gpu(self):
        result = self.run_queue(get=lambda *a: (_ for _ in ()).throw(RuntimeError("prior failed")),
                                submit=lambda *a: self.fail("must not launch"))
        self.assertEqual(result["phase"], "failed")

    def test_worker_timeout_cancels_and_preserves_state(self):
        cancelled = []
        def get(call_id, timeout):
            if call_id == "old":
                return {}
            raise TimeoutError("quota")
        result = self.run_queue(get=get, cancel=cancelled.append)
        self.assertEqual(cancelled, ["newcall"])
        self.assertEqual(result["phase"], "failed")
        self.assertEqual(read_json(self.root / "remote/status.json")["call_id"], "newcall")

    def test_controller_deadline_prevents_unfunded_time(self):
        request = copy.deepcopy(self.request)
        request["deadline"] = 1300
        result = self.run_queue(request=request, submit=lambda *a: self.fail("insufficient time"))
        self.assertEqual(result["phase"], "deadline_reached")

    def test_atomic_reservation_and_exclusive_queue_owner(self):
        tickets = reserve_queue(self.root / "approval.json", self.root / "ledger.json", "queue", self.resources)
        self.assertEqual(len(tickets), 2)
        with self.assertRaises(PermissionError):
            reserve(self.root / "approval.json", self.root / "ledger.json", gpu="A100-40GB", seconds=300, purpose="competing")
        with self.assertRaises(PermissionError):
            reserve_queue(self.root / "approval.json", self.root / "ledger.json", "other", self.resources)
        state = {"queue_id": "queue", "phase": "completed", "steps": [{"ticket_id": tickets[1]["id"]}]}
        self.assertTrue(sync_status("queue", self.root / "ledger.json", state))
        self.assertNotIn("active_queue", read_json(self.root / "ledger.json"))

    def test_insufficient_batch_reserves_nothing(self):
        self.approval["limit_usd"] = .01
        write_json(self.root / "approval.json", self.approval)
        with self.assertRaises(PermissionError):
            reserve_queue(self.root / "approval.json", self.root / "ledger.json", "queue", self.resources)
        self.assertFalse((self.root / "ledger.json").exists())

    def test_ambiguous_failure_does_not_unlock_spending_owner(self):
        reserve_queue(self.root / "approval.json", self.root / "ledger.json", "queue", self.resources)
        self.assertFalse(sync_status("queue", self.root / "ledger.json", {
            "queue_id": "queue", "phase": "needs_reconciliation", "steps": []}))
        self.assertEqual(read_json(self.root / "ledger.json")["active_queue"], "queue")

    def test_unaffordable_tail_is_not_implicitly_authorized(self):
        plan = {"controller_seconds": 1000, "steps": [
            {"name": "one", "gpu": "A100-40GB", "seconds": 300, "job": "probe1", "run_id": "probe-{short}"},
            {"name": "too_much", "gpu": "A100-40GB", "seconds": 3000, "job": "train1", "run_id": "full-{short}"}]}
        request, resources = plan_queue(plan, self.approval, {"jobs": []}, "testqueue", "old", now=1000)
        self.assertEqual([s["name"] for s in request["steps"]], ["one"])
        self.assertEqual([s["name"] for s in request["unfunded_steps"]], ["too_much"])
        request["steps"][0]["ticket"] = {"id": "x", **resources[1]}
        self.assertEqual(self.run_queue(request=request)["phase"], "budget_limited")


if __name__ == "__main__":
    unittest.main()

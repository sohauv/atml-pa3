import unittest

from common.runtime import worker_budget


class WorkerBudgetTests(unittest.TestCase):
    def test_setup_and_shutdown_are_inside_provider_timeout(self):
        ticket = {"seconds": 1350, "deadline": 2400}
        self.assertEqual(worker_budget(ticket, 1000, 1100), 1070)
        self.assertLessEqual(1100 + worker_budget(ticket, 1000, 1100) + 180, 2350)

    def test_reservation_deadline_can_be_earlier_than_provider_timeout(self):
        ticket = {"seconds": 1350, "deadline": 1200}
        self.assertEqual(worker_budget(ticket, 1000, 1100), -80)

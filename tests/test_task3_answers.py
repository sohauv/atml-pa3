import importlib.util
import unittest
import time

from task3_test_time.answers import (answer_groups, edit_distance, is_correct,
                                    last_box, math_equivalent, normalize_mathvista, _ScorerWorker)
from task3_test_time.selection import NO_ANSWER


def slow_fixture_worker(connection):
    connection.send("ready")
    connection.recv()
    time.sleep(2)
    connection.send(("result", True, None))


class AnswerTests(unittest.TestCase):
    def test_symbolic_timeout_returns_false_and_reaps_worker(self):
        worker = _ScorerWorker(seconds=.05, target=slow_fixture_worker)
        try:
            self.assertFalse(worker.compare("unit-test-a", "unit-test-b"))
            self.assertIsNone(worker.process)
        finally:
            worker.close()

    def test_last_box_balances_nested_braces(self):
        self.assertEqual(last_box(r"one \boxed{2}; final \boxed{\frac{1}{2}}"), r"\frac{1}{2}")
        self.assertIsNone(last_box(r"\boxed{1} then \boxed{unfinished"))
        self.assertIsNone(last_box(r"\boxed{}"))
        self.assertIsNone(last_box("no answer"))

    def test_mathvista_choices_use_letters_and_keep_nulls(self):
        meta = {"question_type": "multi_choice", "choices": ["red", "green", "blue"], "answer_type": "text", "precision": 0}
        self.assertEqual(normalize_mathvista("(B) green", meta), "B")
        self.assertEqual(normalize_mathvista("gren", meta), "B")
        self.assertEqual(normalize_mathvista(None, meta), NO_ANSWER)
        self.assertTrue(is_correct("B", "green", "mathvista", meta))
        self.assertFalse(is_correct(NO_ANSWER, "red", "mathvista", meta))

    def test_mathvista_numeric_rules(self):
        integer = {"question_type": "free_form", "answer_type": "integer"}
        floating = {"question_type": "free_form", "answer_type": "float", "precision": 2}
        self.assertEqual(normalize_mathvista("3.0", integer), "3")
        self.assertEqual(normalize_mathvista("1.236", floating), "1.24")
        self.assertEqual(normalize_mathvista("bad", floating), NO_ANSWER)
        self.assertEqual(edit_distance("green", "gren"), 1)

    @unittest.skipUnless(importlib.util.find_spec("math_verify"), "math-verify not installed")
    def test_math_groups_mathematically_equivalent_without_gold(self):
        values = ["0.5", r"\frac{1}{2}", None, "2", "2.0"]
        groups = answer_groups(values, "math")
        self.assertEqual(groups[0], groups[1])
        self.assertEqual(groups[2], NO_ANSWER)
        self.assertEqual(groups[3], groups[4])
        self.assertTrue(math_equivalent(r"\sqrt{4}", "2"))


if __name__ == "__main__":
    unittest.main()

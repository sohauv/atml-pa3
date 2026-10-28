import unittest
import numpy as np
from task3_test_time.selection import adaptive, rank_weights, scores, select, votes, NO_ANSWER


def candidate(answer, certainty, vision=.5):
    return {"answer_group": answer, "kl_u_p": [certainty, certainty + 1],
            "logp": [-1, -2], "img_attn_total": [vision, vision],
            "prompt_text_attn_total": [1 - vision, 1 - vision], "generated_text_attn_total": [0, 0]}


class SelectionTests(unittest.TestCase):
    def test_rank_ties_use_average(self):
        np.testing.assert_allclose(rank_weights([3, 3, 1]), [.75, .75, 0])
        np.testing.assert_allclose(rank_weights([1]), [1])

    def test_vote_tie_uses_first_support(self):
        rows = [candidate("b", 1), candidate("a", 1)]
        self.assertEqual(votes(rows)[0], "b")
        self.assertFalse(votes(rows)[3])

    def test_n1_all_rules_return_only_candidate(self):
        rows = [candidate("only", 1)]
        for method in ("first", "consistency", "certainty", "certainty_rwc", "logp", "vision", "visconf_rwc"):
            self.assertEqual(select(rows, method), "only")

    def test_future_candidates_do_not_change_prefix_scores(self):
        prefix = [candidate("a", 1), candidate("b", 2)]
        original = scores(prefix)["visconf"].copy()
        extended = prefix + [candidate("x", 10000)]
        np.testing.assert_allclose(scores(extended[:2])["visconf"], original)

    def test_no_answer_cannot_trigger_early_stop(self):
        rows = [candidate(NO_ANSWER, 1) for _ in range(32)]
        self.assertEqual(adaptive(rows, .5), (NO_ANSWER, 32))

    def test_stopping_uses_revealed_candidates_only(self):
        rows = [candidate("a", 1) for _ in range(4)] + [candidate("b", 100) for _ in range(28)]
        self.assertEqual(adaptive(rows, 1, 0), ("a", 4))
        self.assertEqual(select(rows, "consistency"), "b")

    def test_token_fusion_difference_equals_covariance(self):
        row = candidate("a", 1)
        row["img_attn_total"] = [.1, .9]
        row["prompt_text_attn_total"] = [.9, .1]
        measured = scores([row])
        # Two calibrated values [0,1] and vision [.1,.9] give population covariance .2.
        self.assertAlmostEqual(float(measured["token_fusion"][0] - measured["visconf"][0]), .2)


if __name__ == "__main__":
    unittest.main()

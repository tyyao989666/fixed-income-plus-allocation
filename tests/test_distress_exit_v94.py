import unittest

import numpy as np

from src.v94_distress_exit import adjust_targets, visible_exit_mask


def notice():
    return dict(ts_code="A", trigger_type="court_accepts_issuer_reorganization", known_at="2023-07-24", sha256="a"*64)


class ExitOverlayTests(unittest.TestCase):
    def test_future_notice_hidden(self):
        self.assertFalse(visible_exit_mask(["A", "B"], "2023-07-21", [notice()]).any())

    def test_known_notice_zeroes_target_not_raw_input(self):
        original = np.array([[.3, .7]])
        result, allowed, blocked = adjust_targets(original, [True, True], ["A", "B"], "2023-07-24", [notice()])
        np.testing.assert_array_equal(result, [[0., 1.]])
        np.testing.assert_array_equal(original, [[.3, .7]])
        self.assertFalse(allowed[0])
        self.assertTrue(blocked[0])

    def test_remaining_weights_relative_proportions_preserved(self):
        result, _, _ = adjust_targets([[.2, .3, .5]], [True]*3, ["A", "B", "C"], "2023-07-24", [notice()])
        np.testing.assert_allclose(result, [[0, .375, .625]])

    def test_all_affected_target_not_fake_liquidation(self):
        result, _, _ = adjust_targets([[1.]], [True], ["A"], "2023-07-24", [notice()])
        self.assertEqual(result.sum(), 0)

    def test_truncated_notices_same_past_output(self):
        left = adjust_targets([[.3, .7]], [True, True], ["A", "B"], "2023-07-21", [notice()])
        right = adjust_targets([[.3, .7]], [True, True], ["A", "B"], "2023-07-21", [])
        for a, b in zip(left, right):
            np.testing.assert_array_equal(a, b)

    def test_wrong_trigger_rejected(self):
        row = notice()
        row["trigger_type"] = "later_bad_returns"
        with self.assertRaises(ValueError):
            visible_exit_mask(["A"], "2023-07-24", [row])

    def test_no_notice_exact_identity(self):
        target = [[.3, .7]]
        result, allowed, _ = adjust_targets(target, [True, False], ["A", "B"], "2023-07-24", [])
        np.testing.assert_array_equal(result, target)
        np.testing.assert_array_equal(allowed, [True, False])


if __name__ == "__main__":
    unittest.main()

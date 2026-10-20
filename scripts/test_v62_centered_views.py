"""Small deterministic tests for the independent centered view interface."""
from pathlib import Path
import sys
import unittest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from src.v62_centered_views import centered_scores, fit_centered_slopes, comparison_mask


class CenteredViewTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(17)
        self.x = self.rng.normal(size=(120, 3, 6))
        self.m = np.ones(self.x.shape, bool)
        self.g = np.array([0, 0, 0, 1, 1, 1])

    def test_shift_invariance(self):
        offsets = np.repeat(self.rng.normal(size=(120, 3, 2)), 3, -1)
        np.testing.assert_allclose(centered_scores(self.x, self.m, self.g),
            centered_scores(self.x + offsets, self.m, self.g), atol=1e-14)

    def test_zero_group_sum_and_unavailable(self):
        self.m[:, :, 0] = False
        x = centered_scores(self.x, self.m, self.g)
        self.assertTrue((x[:, :, 0] == 0).all())
        for group in [0, 1]:
            np.testing.assert_allclose(x[:, :, self.g == group].sum(-1), 0, atol=1e-14)

    def test_single_member_zero(self):
        self.m[:, :, :2] = False
        x = centered_scores(self.x, self.m, self.g)
        self.assertTrue((x[:, :, :3] == 0).all())

    def test_units_ridge_and_scale(self):
        x = centered_scores(self.x, self.m, self.g)
        p = np.full((3, 6), .01)
        truth = p + .02 * x
        beta, _ = fit_centered_slopes(x, truth, self.m, p, self.g)
        np.testing.assert_allclose(beta, .02/1.1)
        scaled, _ = fit_centered_slopes(7*x, truth, self.m, p, self.g)
        np.testing.assert_allclose(x*beta, 7*x*scaled, atol=1e-14)

    def test_negative_relation_abstains(self):
        x = centered_scores(self.x, self.m, self.g)
        beta, rows = fit_centered_slopes(x, -x, self.m, np.zeros((3, 6)), self.g)
        self.assertTrue((beta == 0).all())
        self.assertEqual({r['reason'] for r in rows}, {'nonpositive_calibration_relation'})

    def test_short_calibration_abstains(self):
        x = centered_scores(self.x[:50], self.m[:50], self.g)
        beta, rows = fit_centered_slopes(x, x, self.m[:50], np.zeros((3, 6)), self.g)
        self.assertTrue((beta == 0).all())
        self.assertEqual({r['reason'] for r in rows}, {'insufficient_paired_dates'})

    def test_pairwise_mask_not_all_baselines(self):
        baseline = self.x.copy()
        baseline[:, :, 0] = np.nan
        full = comparison_mask(self.m, self.x)
        pair = comparison_mask(self.m, self.x, baseline)
        self.assertEqual(full.sum() - pair.sum(), 360)
        self.assertTrue(full[:, :, 0].all())

    def test_prefix_does_not_depend_on_future(self):
        full = centered_scores(self.x, self.m, self.g)
        prefix = centered_scores(self.x[:30], self.m[:30], self.g)
        np.testing.assert_array_equal(full[:30], prefix)

    def test_nonfinite_eligible_rejected(self):
        self.x[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            centered_scores(self.x, self.m, self.g)


if __name__ == '__main__':
    unittest.main()

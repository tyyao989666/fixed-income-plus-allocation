import unittest

import numpy as np
import pandas as pd
from scipy.optimize._numdiff import approx_derivative

from src.v75_strict_anchor import (solve_budget, objective_and_gradient,
    frozen_monthly_sleeves, covariance_complete, project_box)


class StrictAnchorTests(unittest.TestCase):
    def test_equal_risk_diagonal_is_inverse_vol(self):
        c = np.diag(np.array([1., 2., 3., 4., 5.])**2)
        w, diag = solve_budget(c, np.ones(5) / 5, np.zeros(5), np.ones(5))
        expected = 1 / np.sqrt(np.diag(c))
        np.testing.assert_allclose(w, expected / expected.sum(), atol=2e-6)
        self.assertTrue(diag["risk_target_exact"])

    def test_analytic_gradient(self):
        c = np.array([[2., .3], [.3, 1.]])
        w, target = np.array([.4, .6]), np.array([.3, .7])
        _, gradient = objective_and_gradient(w, c, target)
        numeric = approx_derivative(lambda x: objective_and_gradient(x, c, target)[0], w).ravel()
        np.testing.assert_allclose(gradient, numeric, atol=1e-7)

    def test_impossible_bounds_fail(self):
        with self.assertRaises(ValueError):
            project_box([.5, .5], [.6, .6], [1., 1.])

    def test_missing_held_path_not_renormalized(self):
        days = pd.to_datetime(["2020-01-31", "2020-02-03", "2020-02-04"])
        daily = pd.DataFrame({"a": [.01, .02, np.nan], "b": [.01, .03, .04]}, index=days)
        out, members = frozen_monthly_sleeves(daily, {"test": {"a": .5, "b": .5}}, "2020-02-29")
        feb = out.loc[out.date.eq(pd.Timestamp("2020-02-29"))].iloc[0]
        self.assertTrue(pd.isna(feb.return_1m))
        self.assertEqual(feb.status, "missing_held_path")
        self.assertEqual(len(members), 2)

    def test_future_does_not_reselect_month_membership(self):
        days = pd.to_datetime(["2020-01-31", "2020-02-03", "2020-02-04"])
        daily = pd.DataFrame({"a": [.01, .02, .03], "b": [np.nan, .04, .05]}, index=days)
        _, members = frozen_monthly_sleeves(daily, {"test": {"a": .5, "b": .5}}, "2020-02-29")
        self.assertEqual(members.asset.tolist(), ["a"])
        changed = daily.copy()
        changed.iloc[1:] = np.nan
        _, second = frozen_monthly_sleeves(changed, {"test": {"a": .5, "b": .5}}, "2020-02-29")
        pd.testing.assert_frame_equal(members, second)

    def test_covariance_no_pairwise_or_zero_fill(self):
        window = pd.DataFrame([[.1, .2], [np.nan, .3], [.2, .1], [.3, np.nan]])
        c, n = covariance_complete(window)
        self.assertEqual(n, 2)
        self.assertGreater(np.linalg.eigvalsh(c).min(), 0)


if __name__ == "__main__":
    unittest.main()

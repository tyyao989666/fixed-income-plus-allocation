import unittest
from unittest.mock import patch

import numpy as np

from src.v78_exact_risk_budget import exact_positive_budget


class ExactBudgetTests(unittest.TestCase):
    def test_diagonal_budget_solution(self):
        c = np.diag([1., 4., 9.])
        b = np.array([.2, .3, .5])
        w, audit = exact_positive_budget(c, b)
        expected = np.sqrt(b / np.diag(c))
        np.testing.assert_allclose(w, expected / expected.sum(), atol=1e-6)
        self.assertLess(audit["max_rc_error"], 1e-6)

    def test_erc_identity_is_equal_capital(self):
        w, _ = exact_positive_budget(np.eye(5), np.ones(5) / 5)
        np.testing.assert_allclose(w, np.ones(5) / 5, atol=1e-6)

    def test_zero_budget_and_non_spd_rejected(self):
        with self.assertRaises(ValueError):
            exact_positive_budget(np.eye(2), [1., 0.])
        with self.assertRaises(ValueError):
            exact_positive_budget(np.array([[1., 2.], [2., 1.]]), [.5, .5])

    def test_inaccurate_primary_is_rejected_not_reported_as_success(self):
        def inaccurate(problem, **kwargs):
            problem._status = "optimal_inaccurate"
        with patch("cvxpy.Problem.solve", inaccurate):
            w, audit = exact_positive_budget(np.diag([1., 4.]), np.array([.5, .5]))
        np.testing.assert_allclose(w, [2 / 3, 1 / 3], atol=1e-6)
        self.assertIn("LBFGSB", audit["solver"])
        self.assertEqual(audit["primary_status"], "optimal_inaccurate")

    def test_scalar_covariance_units_do_not_change_budget_weights(self):
        c = np.array([[2., .2], [.2, 1.]])
        first, _ = exact_positive_budget(c, [.4, .6])
        second, _ = exact_positive_budget(c * 12, [.4, .6])
        np.testing.assert_allclose(first, second, atol=1e-6)


if __name__ == "__main__":
    unittest.main()

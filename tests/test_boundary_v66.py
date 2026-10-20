import unittest
import numpy as np
import pandas as pd
from src.v66_boundary import monthly_boundary, align_completed_months, boundary_features


class BoundaryTests(unittest.TestCase):
    def history(self, n=160):
        return pd.DataFrame({'dividend_yield': np.arange(n, dtype=float),
                             'term_spread': np.arange(n, dtype=float)[::-1]},
                            index=pd.period_range('2000-01', periods=n, freq='M'))

    def test_formula_and_full_warmup(self):
        result = monthly_boundary(self.history())
        self.assertTrue(result.boundary.iloc[:130].isna().all())
        np.testing.assert_allclose(result.boundary.iloc[130:], 2)

    def test_constant_not_neutral(self):
        self.assertTrue(monthly_boundary(self.history()*0).boundary.isna().all())

    def test_hand_computed_midpoint(self):
        x = pd.DataFrame({'dividend_yield': [1, 3, 1, 2], 'term_spread': [3, 1, 3, 2]},
                         index=pd.period_range('2000-01', periods=4, freq='M'))
        result = monthly_boundary(x, history_months=3, smoothing_months=1)
        self.assertEqual(result.boundary.iloc[2], 2)
        self.assertEqual(result.boundary.iloc[3], 0)

    def test_missing_month_not_bridged(self):
        result = monthly_boundary(self.history().drop(pd.Period('2005-01', freq='M')))
        self.assertTrue(result.boundary.isna().all())

    def test_future_invariance(self):
        x = self.history()
        reference = monthly_boundary(x)
        x.iloc[150:] = 100000
        pd.testing.assert_frame_equal(reference.iloc[:150], monthly_boundary(x).iloc[:150])

    def test_completed_month_and_masks(self):
        x = monthly_boundary(self.history())
        dates = pd.to_datetime(['2012-01-03', '2012-01-31', '2012-02-01'])
        aligned = align_completed_months(x, dates)
        self.assertEqual(aligned.observation_month.iloc[0], '2011-12')
        self.assertEqual(aligned.observation_month.iloc[1], '2011-12')
        self.assertEqual(aligned.observation_month.iloc[2], '2012-01')
        self.assertFalse(aligned.strict_available.any())
        values, research, strict = boundary_features(aligned, [0.1, np.nan, -0.2])
        self.assertTrue(research[:, :3].all())
        self.assertFalse(research[1, 3:].any())
        self.assertFalse(strict.any())
        self.assertEqual(values.shape, (3, 5))
        self.assertFalse(boundary_features(aligned, [0.1, 0.2, 0.3], certified=True)[2].any())

    def test_twenty_year_warmup(self):
        x = monthly_boundary(self.history(270), history_months=240)
        self.assertTrue(x.boundary.iloc[:250].isna().all())
        self.assertEqual(x.boundary.first_valid_index(), x.index[250])


if __name__ == '__main__':
    unittest.main()

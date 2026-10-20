import unittest
import numpy as np
import pandas as pd
from src.v67_monthly_accounting import monthly_ledger


class AccountingTests(unittest.TestCase):
    def run_case(self, r, w):
        index = pd.date_range('2020-01-31', periods=len(r), freq='ME')
        return monthly_ledger(pd.DataFrame(r, index=index), pd.DataFrame(w, index=index), pd.Series(0., index=index))

    def test_entry_is_full_not_half_notional(self):
        z = self.run_case([[0], [0]], [[1], [1]])
        self.assertEqual(z.risky_traded_notional.iloc[1], 1)

    def test_price_drift_requires_rebalance(self):
        z = self.run_case([[0], [1], [0]], [[.5], [.5], [.5]])
        self.assertAlmostEqual(z.gross_return.iloc[1], .5)
        self.assertAlmostEqual(z.risky_traded_notional.iloc[2], 2/3-.5)

    def test_missing_held_return_flag(self):
        z = self.run_case([[0], [np.nan]], [[.4], [0]])
        self.assertEqual(z.unknown_held_weight.iloc[1], .4)
        self.assertFalse(z.strict_month_known.iloc[1])

    def test_current_target_cannot_earn_current_return(self):
        z = self.run_case([[0], [.1]], [[0], [1]])
        self.assertEqual(z.gross_return.iloc[1], 0)


if __name__ == '__main__':
    unittest.main()

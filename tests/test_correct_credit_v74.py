import unittest

import numpy as np
import pandas as pd

from src.v74_rebuild import exact_calendar_returns


class ExactSourceTests(unittest.TestCase):
    def setUp(self):
        self.days = pd.date_range("2020-01-01", periods=3)
        self.frame = pd.DataFrame(dict(date=self.days, close=[100., 101., 102.], source_identifier=["H11098.SH"] * 3))

    def test_no_fake_first_return(self):
        r = exact_calendar_returns(self.frame, self.days, "H11098.SH")
        self.assertTrue(np.isnan(r.iloc[0]))
        self.assertAlmostEqual(r.iloc[1], .01)

    def test_missing_date_never_filled(self):
        with self.assertRaises(ValueError):
            exact_calendar_returns(self.frame.iloc[[0, 2]], self.days, "H11098.SH")

    def test_wrong_identity_rejected_even_when_prices_look_like_bonds(self):
        f = self.frame.copy()
        f.source_identifier = "HSI.HI"
        with self.assertRaises(ValueError):
            exact_calendar_returns(f, self.days, "H11098.SH")

    def test_duplicate_dates_rejected(self):
        with self.assertRaises(ValueError):
            exact_calendar_returns(pd.concat([self.frame, self.frame.iloc[[2]]]), self.days, "H11098.SH")


if __name__ == "__main__":
    unittest.main()

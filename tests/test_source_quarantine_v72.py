import unittest

import numpy as np
import pandas as pd

from src.v72_source_repair import quarantine_returns, endpoint_reason


class QuarantineTests(unittest.TestCase):
    def test_masks_segment_and_boundary_without_mutating_parent(self):
        days = pd.date_range("2026-05-13", periods=5)
        frame = pd.DataFrame({"credit": [.01] * 5, "equity": [.02] * 5}, index=days)
        result = quarantine_returns(frame, "credit", "2026-05-15")
        self.assertTrue(result.credit.iloc[:3].isna().all())
        self.assertEqual(result.credit.iloc[3], .01)
        pd.testing.assert_series_equal(result.equity, frame.equity)
        self.assertTrue(frame.credit.notna().all())

    def test_bad_calendar_and_unknown_asset_rejected(self):
        frame = pd.DataFrame({"credit": [.1, .2]}, index=pd.to_datetime(["2020-01-01"] * 2))
        with self.assertRaises(ValueError):
            quarantine_returns(frame, "credit", "2020-01-01")
        with self.assertRaises(ValueError):
            quarantine_returns(frame, "unknown", "2020-01-01")

    def test_endpoint_missing_does_not_imply_zero_return_or_terminal_cash(self):
        self.assertEqual(endpoint_reason(100., np.nan), "current_source_endpoint_missing")
        self.assertEqual(endpoint_reason(np.nan, 101.), "previous_source_endpoint_missing")
        self.assertEqual(endpoint_reason(np.nan, np.nan), "both_source_endpoints_missing")
        self.assertEqual(endpoint_reason(100., 100.), "both_source_endpoints_present_requires_other_evidence")


if __name__ == "__main__":
    unittest.main()

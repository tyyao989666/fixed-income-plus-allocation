"""Causal source-splice, curve and visibility contract tests."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from src.v53_unified_data import (merge_verified_levels, curve_return, delayed_us_curve,
    independent_cb_relative_labels, harmonize_activity_units)


class UnifiedTests(unittest.TestCase):
    def test_volume_lots_to_shares_unit_conversion(self):
        old = pd.Series(np.arange(1., 11.))
        converted, new, audit = harmonize_activity_units(old, old * 100, "vol")
        np.testing.assert_array_equal(converted, new)
        self.assertEqual(audit["historical_to_refresh_unit_factor"], 100)

    def test_amount_thousands_to_yuan(self):
        old = pd.Series(np.arange(1., 11.))
        converted, new, audit = harmonize_activity_units(old, old * 1000, "amount")
        np.testing.assert_array_equal(converted, new)

    def test_arbitrary_activity_rescaling_not_allowed(self):
        old = pd.Series(np.arange(1., 11.))
        converted, new, audit = harmonize_activity_units(old, old * 137.5, "vol")
        self.assertFalse(audit["available"])
        self.assertTrue(converted.isna().all() and new.isna().all())

    def test_same_level_overlap(self):
        d = pd.bdate_range("2020-01-01", periods=12)
        a = pd.Series(np.arange(100., 110.), index=d[:10])
        b = pd.Series(np.arange(105., 112.), index=d[5:])
        merged, check = merge_verified_levels(a, b)
        self.assertEqual(len(merged), 12)
        self.assertEqual(check["overlap_rows"], 5)

    def test_different_basis_rejected(self):
        d = pd.bdate_range("2020-01-01", periods=10)
        with self.assertRaises(ValueError):
            merge_verified_levels(pd.Series(100., index=d), pd.Series(150., index=d))

    def test_no_missing_price_bridge(self):
        d = pd.bdate_range("2020-01-01", periods=10)
        a = pd.Series(100., index=d)
        b = a.copy()
        b.iloc[8] = np.nan
        merged, _ = merge_verified_levels(a, b)
        self.assertTrue(np.isnan(merged.pct_change(fill_method=None).iloc[9]))

    def test_curve_missing_not_zero(self):
        r = curve_return(pd.Series([2., np.nan, 2.1, 2.2]), 18.5)
        self.assertTrue(r.iloc[:3].isna().all())
        self.assertAlmostEqual(r.iloc[3], .021 / 252 - 18.5 * .001)

    def test_us_future_observation_not_same_day(self):
        d = pd.date_range("2020-01-01", periods=6)
        source = pd.DataFrame(dict(source_date=[d[0], d[2]], yield_percent=[2., 4.]))
        x = delayed_us_curve(source, d)
        self.assertTrue(np.isnan(x.yield_percent.iloc[1]))
        self.assertEqual(x.yield_percent.iloc[2], 2.)
        self.assertEqual(x.yield_percent.iloc[4], 4.)

    def test_future_perturbation_does_not_change_visible_history(self):
        d = pd.date_range("2020-01-01", periods=10)
        f = pd.DataFrame(dict(source_date=d, yield_percent=np.ones(10)))
        before = delayed_us_curve(f, d)
        f.loc[8:, "yield_percent"] = 100
        after = delayed_us_curve(f, d)
        pd.testing.assert_frame_equal(before, after)

    def test_us_staleness_bounded(self):
        d = pd.date_range("2020-01-01", periods=20)
        x = delayed_us_curve(pd.DataFrame(dict(source_date=[d[0]], yield_percent=[2.])), d)
        self.assertTrue(np.isnan(x.yield_percent.iloc[-1]))

    def test_duplicate_source_rejected(self):
        d = pd.date_range("2020-01-01", periods=4)
        f = pd.DataFrame(dict(source_date=[d[0], d[0]], yield_percent=[2., 3.]))
        with self.assertRaises(ValueError):
            delayed_us_curve(f, d)

    def test_cb_reference_does_not_spread_other_asset_missing(self):
        dates = pd.bdate_range("2020-01-01", periods=80).asi8
        r = np.full((80, 6), .001)
        mask = np.ones_like(r, dtype=bool)
        # Assets 2 and 3 both belong to CB; only asset 2 is missing tomorrow.
        modules = np.array([0, 1, 2, 2, 3, 4])
        mask[10, 2] = False
        weights = np.zeros((80, 5, 6))
        for m in range(5):
            weights[:,m,modules == m] = 1 / (modules == m).sum()
        result = independent_cb_relative_labels(dates, r, mask, modules, weights, np.full((80,5), .2), np.full(80, .0005))
        self.assertFalse(result["asset_target_mask"][9,0,2])
        self.assertTrue(result["asset_target_mask"][9,0,3])
        self.assertTrue(result["sleeve_target_mask"][9,0].all())

    def test_missing_independent_reference_still_invalidates_cb(self):
        dates = pd.bdate_range("2020-01-01", periods=80).asi8
        r = np.full((80,5), .001)
        w = np.broadcast_to(np.eye(5), (80,5,5))
        reference = np.full(80, .0005)
        reference[10] = np.nan
        result = independent_cb_relative_labels(dates, r, np.ones_like(r,dtype=bool), np.arange(5), w, np.full((80,5),.2), reference)
        self.assertFalse(result["asset_target_mask"][9,0,2])
        self.assertTrue(result["asset_target_mask"][9,0,0])
        self.assertFalse(result["sleeve_target_mask"][9,0].any())


if __name__ == "__main__":
    unittest.main()

"""Coverage and missing-member controls used by the scoped dependency audit."""
import unittest

import numpy as np
import pandas as pd

from scripts.audit_v71_asset_dependencies import coverage
from src.v49_point_in_time import cb_bucket_weights, realized_bucket_return


class AssetDependencyTests(unittest.TestCase):
    def test_coverage_separates_listing_edges_from_internal_gaps(self):
        days = pd.date_range("2020-01-01", periods=6)
        result = coverage(pd.Series([np.nan, .01, np.nan, .02, .03, np.nan], index=days))
        self.assertEqual(result["first_valid"], "2020-01-02")
        self.assertEqual(result["last_valid"], "2020-01-05")
        self.assertEqual(result["internal_missing_days"], 1)
        self.assertEqual(result["missing_days"], 3)

    def test_empty_and_invalid_price_returns_not_counted(self):
        series = pd.Series([np.nan, np.inf, -1.], index=pd.date_range("2020-01-01", periods=3))
        result = coverage(series)
        self.assertEqual(result["valid_days"], 0)
        self.assertEqual(result["first_valid"], "")
        self.assertEqual(result["internal_missing_days"], 0)

    def test_one_missing_member_invalidates_whole_return(self):
        weights = pd.Series([.4, .6], index=["a", "b"])
        observed = pd.Series([.01, np.nan], index=weights.index)
        self.assertTrue(np.isnan(realized_bucket_return(weights, observed)))
        self.assertAlmostEqual(realized_bucket_return(weights, observed.fillna(.02)), .016)

    def test_style_fallback_can_make_named_buckets_identical(self):
        codes = [f"bond{i}" for i in range(12)]
        price = pd.Series(np.arange(115., 127.), index=codes)
        size = pd.Series(np.arange(1., 13.), index=codes)
        buckets = cb_bucket_weights(price, size, size)
        # Twelve observations cannot support twelve ten-member disjoint groups.
        self.assertEqual(len(buckets), 12)
        weights = [tuple(g["weights"].items()) for g in buckets.values()]
        self.assertEqual(len(set(weights)), 1)
        self.assertTrue(all(g["weights"].max() <= .1 + 1e-12 for g in buckets.values()))


if __name__ == "__main__":
    unittest.main()

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.v49_cb_database import panel, historical_universe, observed_returns, rebalance_turnover, strict_nav, known_exit_exclusions


class SourceContracts(unittest.TestCase):
    def test_duplicate_dates_rejected(self):
        with self.assertRaises(ValueError):
            panel(pd.DataFrame({"date":["2026-01-01"]*2, "A":[1, 2]}))

    def test_cutoff_and_no_backfill(self):
        x = panel(pd.DataFrame({"date":["2026-01-01", "2026-10-15"], "A":[np.nan, 100]}))
        self.assertEqual(len(x), 1)
        self.assertTrue(x.A.isna().all())

    def test_retired_bonds_retained_eb_private_split_excluded(self):
        codes = ["110001.SH", "132001.SH", "124001.SZ", "126001.SH", "123001.SZ", "110815.SH"]
        b = pd.DataFrame(dict(ts_code=codes, cb_type=["CB", "EB", "CB", "CB", "CB", "CB"],
            bond_short_name=["历史转债", "交换", "私募", "分离", "定转", "九丰定01"],
            list_date=["20180101"]*6, issue_size=[1e9]*6, remain_size=[0]*6))
        result = historical_universe(b, codes)
        self.assertEqual(result.index[result.included_research_universe].tolist(), [codes[0]])

    def test_preclose_return_not_cross_gap_return_or_clipped(self):
        close = pd.DataFrame({"A":[100, np.nan, 150]})
        preclose = pd.DataFrame({"A":[100, 100, 100]})
        result = observed_returns(close, preclose)
        self.assertEqual(result.A.iloc[-1], .5)
        self.assertTrue(np.isnan(result.A.iloc[1]))

    def test_turnover_includes_drift(self):
        w = pd.Series([.5, .5], index=["A", "B"])
        r = pd.Series([.1, -.1], index=w.index)
        self.assertAlmostEqual(rebalance_turnover(w, r, w), .05)
        self.assertTrue(np.isnan(rebalance_turnover(w, r* np.nan, w)))

    def test_missing_nav_is_not_zero_filled(self):
        out = strict_nav(pd.Series([.1, np.nan, .2]))
        self.assertAlmostEqual(out.iloc[0], 110)
        self.assertTrue(out.iloc[1:].isna().all())

    def test_call_quarantine_not_before_announcement_and_future_invariant(self):
        days=pd.bdate_range("2026-07-01",periods=10)
        master=pd.DataFrame({"maturity_date":["20300101"]},index=["A"])
        calls=pd.DataFrame(dict(ts_code=["A"],call_type=["强赎"],is_call=["公告实施强赎"],ann_date=["20260703"]))
        a,_=known_exit_exclusions(calls,master,days)
        self.assertFalse(a.loc[:"2026-07-03"].any().any())
        self.assertTrue(a.loc["2026-07-06":].all().all())
        future=calls.copy();future["is_call"]="公告不强赎";future["ann_date"]="20260713"
        b,_=known_exit_exclusions(pd.concat([calls,future]),master,days)
        pd.testing.assert_frame_equal(a.loc[:"2026-07-13"],b.loc[:"2026-07-13"])
        self.assertFalse(b.loc["2026-07-14","A"])

    def test_maturity_buffer_does_not_use_future_delisting(self):
        days=pd.bdate_range("2026-07-01",periods=10)
        master=pd.DataFrame({"maturity_date":["20260715"],"delist_date":["20260703"]},index=["A"])
        calls=pd.DataFrame(columns=["ts_code","call_type","is_call","ann_date"])
        a,_=known_exit_exclusions(calls,master,days)
        self.assertFalse(a.loc["2026-07-07","A"])
        self.assertTrue(a.loc["2026-07-08","A"])


if __name__ == "__main__":
    unittest.main()

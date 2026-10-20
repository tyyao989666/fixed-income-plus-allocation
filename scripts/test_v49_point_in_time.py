from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.v49_point_in_time import (next_session_after, cb_disclosure_events, align_disclosed_balances,
    capped_weights, cb_bucket_weights, realized_bucket_return, dividend_cash_per_unit)


class Contracts(unittest.TestCase):
    def test_weekend_publication_next_session_only(self):
        calendar = pd.bdate_range("2026-07-03", periods=5)
        value = next_session_after(["2026-07-03", "2026-07-04"], calendar)
        self.assertEqual(value[0], pd.Timestamp("2026-07-06"))
        self.assertEqual(value[1], pd.Timestamp("2026-07-06"))

    def sample(self):
        return pd.DataFrame(dict(ts_code=["A"], publish_date=pd.to_datetime(["2026-07-06"]),
            end_date=pd.to_datetime(["2026-06-30"]), remain_size=[90.], issue_size=[100.]))

    def test_period_end_cannot_be_used_as_release(self):
        dates = pd.bdate_range("2026-07-01", periods=10)
        events = cb_disclosure_events(self.sample(), dates, "2026-07-14")
        result = align_disclosed_balances(events, dates, ["A"])["balance"]
        self.assertTrue(result.loc[:"2026-07-06", "A"].isna().all())
        self.assertEqual(result.loc["2026-07-07", "A"], 90.)

    def test_unpublished_or_bad_chronology_not_used(self):
        dates = pd.bdate_range("2026-07-01", periods=10)
        bad = self.sample()
        bad["publish_date"] = pd.Timestamp("2026-10-16")
        self.assertTrue(cb_disclosure_events(bad, dates, "2026-10-14").empty)
        bad["publish_date"] = pd.Timestamp("2026-06-01")
        with self.assertRaises(ValueError):
            cb_disclosure_events(bad, dates, "2026-10-14")

    def test_late_old_period_does_not_roll_back_latest_known_balance(self):
        dates = pd.bdate_range("2026-07-01", periods=15)
        frame = pd.concat([self.sample(), self.sample()], ignore_index=True)
        frame.loc[1, "publish_date"] = pd.Timestamp("2026-07-10")
        frame.loc[1, "end_date"] = pd.Timestamp("2026-03-31")
        frame.loc[1, "remain_size"] = 99.
        out = align_disclosed_balances(cb_disclosure_events(frame, dates, "2026-07-21"), dates, ["A"])
        self.assertEqual(out["balance"].loc["2026-07-13", "A"], 90.)

    def test_cap_after_normalization_and_infeasible_eight_names(self):
        size = pd.Series([900.]+[1.]*19)
        weights = capped_weights(size)
        self.assertAlmostEqual(weights.sum(), 1.)
        self.assertLessEqual(weights.max(), .1+1e-10)
        with self.assertRaises(ValueError):
            capped_weights(size.iloc[:8])

    def test_missing_realized_return_never_reweights_members(self):
        codes = [f"A{i}" for i in range(30)]
        prices = pd.Series(np.linspace(95, 160, 30), index=codes)
        balance = pd.Series(np.arange(1, 31)*100., index=codes)
        groups = cb_bucket_weights(prices, balance, balance)
        self.assertEqual(len(groups), 12)
        for bucket in groups.values():
            weight = bucket["weights"]
            self.assertGreaterEqual(len(weight), 10)
            self.assertLessEqual(weight.max(), .1+1e-10)
            returns = pd.Series(.001, index=codes)
            self.assertAlmostEqual(realized_bucket_return(weight, returns), .001)
            returns[weight.index[0]] = np.nan
            self.assertTrue(np.isnan(realized_bucket_return(weight, returns)))
            pd.testing.assert_series_equal(weight, bucket["weights"])

    def test_future_events_cannot_change_past(self):
        dates = pd.bdate_range("2026-07-01", periods=15)
        before = self.sample()
        after = pd.concat([before, before], ignore_index=True)
        after.loc[1, "publish_date"] = pd.Timestamp("2026-07-15")
        after.loc[1, "end_date"] = pd.Timestamp("2026-07-14")
        after.loc[1, "remain_size"] = 1.
        a = align_disclosed_balances(cb_disclosure_events(before, dates, "2026-07-21"), dates, ["A"])
        b = align_disclosed_balances(cb_disclosure_events(after, dates, "2026-07-21"), dates, ["A"])
        pd.testing.assert_frame_equal(a["balance"].loc[:"2026-07-14"], b["balance"].loc[:"2026-07-14"])

    def test_training_guard_cannot_be_overridden_by_ready_flag_alone(self):
        from scripts.train_v49_forecasts import require_ready
        with self.assertRaises(ValueError):
            require_ready(dict(training_ready=True, cutoff="2026-10-14", blockers=["source not audited"]))
        with self.assertRaises(ValueError):
            require_ready(dict(training_ready=False, cutoff="2026-10-14", blockers=[]))

    def test_duplicate_dividend_not_double_counted_and_future_ann_not_used(self):
        record = dict(ts_code="A", div_proc="实施", ann_date="20260701", imp_anndate="20260701",
                      base_date="20260630", record_date="20260702", ex_date="20260703", pay_date="20260706", div_cash=.1)
        frame = pd.DataFrame([record, record])
        dates = pd.bdate_range("2026-07-01", periods=5)
        cash = dividend_cash_per_unit(frame, dates, "A")
        self.assertAlmostEqual(cash.sum(), .1)
        frame["ann_date"], frame["imp_anndate"] = "20260704", "20260704"
        self.assertAlmostEqual(dividend_cash_per_unit(frame, dates, "A").sum(), 0.)


if __name__ == "__main__":
    unittest.main()

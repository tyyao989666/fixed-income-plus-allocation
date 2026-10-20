"""Executable-ledger regression tests, including entitlement timing."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from src.v52_cb_execution import QuantityBooks, validate_events, EVENT_COLUMNS


class ExecutionTests(unittest.TestCase):
    def book(self):
        return QuantityBooks(["bucket"], ["a", "b"], initial_capital=100., cost_rate=0.)

    def events(self, kind="coupon"):
        return validate_events(pd.DataFrame([dict(event_id="one", ts_code="a", kind=kind,
            known_at="2020-01-01", record_date="2020-01-02", ex_date="2020-01-03",
            pay_date="2020-01-06", confirmed_at="2020-01-06", cash_yuan_per_bond=2., source="test_evidence")]), ["a", "b"])

    def step(self, b, day, price, executable=(True, True), target=((1., 0.),), valuation=None, events=None):
        return b.step(day, target, [True, True], price, executable,
                      price if valuation is None else valuation,
                      pd.DataFrame(columns=EVENT_COLUMNS) if events is None else events)

    def test_buy_close_does_not_earn_prior_move(self):
        b = self.book()
        x = self.step(b, "2020-01-02", [200., 100.])
        self.assertEqual(x["observed_price_nav"][0], 100.)
        self.assertEqual(b.quantity[0, 0], .5)

    def test_halt_freezes_quantities_and_invalidates_nav(self):
        b = self.book()
        self.step(b, "2020-01-02", [100., 100.])
        x = self.step(b, "2020-01-03", [np.nan, 100.], executable=[False, True], target=[[0., 1.]])
        np.testing.assert_array_equal(b.quantity, [[1., 0.]])
        self.assertTrue(np.isnan(x["observed_price_nav"][0]))
        self.assertEqual(x["traded_notional"][0], 0)
        y = self.step(b, "2020-01-06", [60., 100.])
        self.assertEqual(y["observed_price_nav"][0], 60.)

    def test_receivable_then_payment_no_double_count(self):
        b, events = self.book(), self.events()
        self.step(b, "2020-01-02", [100., 100.], events=events)
        x = self.step(b, "2020-01-03", [98., 100.], events=events)
        self.assertEqual(x["observed_price_nav"][0], 100.)
        self.assertEqual(x["receivable"][0], 2.)
        self.assertEqual(x["payment_cash"][0], 0.)
        x = self.step(b, "2020-01-06", [98., 100.], events=events)
        self.assertAlmostEqual(x["observed_price_nav"][0], 100.)
        self.assertEqual(x["payment_cash"][0], 2.)
        x = self.step(b, "2020-01-07", [98., 100.], events=events)
        self.assertEqual(x["payment_cash"][0], 0.)

    def test_future_cash_announcement_rejected(self):
        f = self.events()
        f.loc[0, "known_at"] = pd.Timestamp("2020-01-04")
        with self.assertRaises(ValueError):
            validate_events(f, ["a", "b"])

    def test_duplicate_economic_event_rejected(self):
        f = self.events()
        copy = f.copy()
        copy["event_id"] = "two"
        with self.assertRaises(ValueError):
            validate_events(pd.concat([f, copy]), ["a", "b"])

    def test_terminal_and_coupon_same_payment_require_consolidation(self):
        f = self.events()
        copy = f.copy()
        copy["event_id"], copy["kind"] = "redemption_including_interest", "terminal"
        with self.assertRaises(ValueError):
            validate_events(pd.concat([f, copy]), ["a", "b"])

    def test_terminal_cancellation_preserves_receivable(self):
        b, events = self.book(), self.events("terminal")
        self.step(b, "2020-01-02", [100., 100.], events=events)
        x = self.step(b, "2020-01-03", [np.nan, 100.], executable=[False, True], events=events)
        self.assertEqual(b.quantity[0, 0], 0.)
        self.assertEqual(x["receivable"][0], 2.)
        self.assertEqual(x["observed_price_nav"][0], 2.)

    def test_blocked_buy_has_no_cost(self):
        b = self.book()
        b.cost_rate = .0005
        x = self.step(b, "2020-01-02", [100., 100.], executable=[False, True])
        self.assertEqual(x["cost_yuan"][0], 0)
        self.assertEqual(b.cash[0], 100.)

    def test_replayed_date_rejected(self):
        b = self.book()
        self.step(b, "2020-01-02", [100., 100.])
        with self.assertRaises(ValueError):
            self.step(b, "2020-01-02", [100., 100.])

    def test_fees_conserve_capital(self):
        b = self.book()
        b.cost_rate = .0005
        x = self.step(b, "2020-01-02", [100., 100.])
        self.assertAlmostEqual(x["observed_price_nav"][0] + x["cost_yuan"][0], 100.)
        self.assertGreaterEqual(b.cash[0], -1e-8)

    def test_halt_does_not_block_independent_full_exit(self):
        b = self.book()
        b.quantity[:] = [[.5, .5]]
        b.cash[:] = 0
        b.last_mark[:] = 100
        x = self.step(b, "2020-01-02", [np.nan, 100.], executable=[False, True], target=[[1., 0.]])
        np.testing.assert_array_equal(b.quantity, [[.5, 0.]])
        self.assertEqual(b.cash[0], 50.)
        self.assertTrue(np.isnan(x["observed_price_nav"][0]))

    def test_retired_security_cannot_be_repurchased(self):
        b, events = self.book(), self.events("terminal")
        self.step(b, "2020-01-02", [100., 100.], events=events)
        self.step(b, "2020-01-03", [100., 100.], events=events)
        self.step(b, "2020-01-06", [100., 100.], events=events)
        self.assertEqual(b.quantity[0, 0], 0.)
        self.assertEqual(b.cash[0], 2.)


if __name__ == "__main__":
    unittest.main()

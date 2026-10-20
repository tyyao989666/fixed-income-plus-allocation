"""Focused causal label and frozen-inventory contract tests."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from src.v51_cb_policy import Inventory, buy_eligibility, masked_forward_labels


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.index = pd.Index(["a", "b"])

    def series(self, a, b):
        return pd.Series([a, b], index=self.index)

    def test_signal_missing_volume_prevents_buy(self):
        ok, _ = buy_eligibility(self.series(100., 100.), self.series(1., np.nan),
            self.series(1., 1.), self.series(1., 1.), self.series(False, False))
        self.assertEqual(ok.tolist(), [True, False])

    def test_exit_is_snapshot_only(self):
        args = [self.series(100., 100.), self.series(1., 1.), self.series(1., 1.), self.series(1., 1.)]
        before, _ = buy_eligibility(*args, self.series(False, False))
        after, _ = buy_eligibility(*args, self.series(True, False))
        self.assertTrue(before.a)
        self.assertFalse(after.a)

    def test_missing_label_keeps_calendar_and_other_37(self):
        frame = pd.DataFrame(.001, index=pd.bdate_range("2020-01-01", periods=80), columns=range(38))
        frame.iloc[10, 0] = np.nan
        values, masks = masked_forward_labels(frame)
        self.assertEqual(values.shape, (80, 3, 38))
        self.assertFalse(masks[9, 0, 0])
        self.assertTrue(masks[9, 0, 1:].all())
        self.assertFalse(masks[-5:, 0].any())
        self.assertTrue(masks[10, 0, 0])  # Future starts at t+1, not t.
        self.assertEqual(values[9, 0, 0], 0)

    def test_future_perturbation_does_not_change_mature_past_label(self):
        frame = pd.DataFrame(.001, index=pd.bdate_range("2020-01-01", periods=80), columns=["a"])
        before, _ = masked_forward_labels(frame)
        frame.iloc[70:] = .5
        after, _ = masked_forward_labels(frame)
        np.testing.assert_array_equal(before[:10], after[:10])

    def test_frozen_quantity_and_missing_certified_nav(self):
        ledger = Inventory(self.series(1., 0.), self.series(100., 100.), 0.)
        result = ledger.close_step(self.series(0., 1.), self.series(np.nan, 100.),
            self.series(False, True), self.series(np.nan, 100.), self.series(False, True))
        self.assertEqual(ledger.quantity.a, 1.)
        self.assertEqual(ledger.quantity.b, 0.)
        self.assertEqual(result["cost"], 0.)
        self.assertTrue(np.isnan(result["certified_nav"]))
        self.assertEqual(result["provisional_nav"], 100.)

    def test_reopening_captures_full_loss(self):
        ledger = Inventory(self.series(1., 0.), self.series(100., 100.), 0.)
        ledger.close_step(self.series(1., 0.), self.series(np.nan, 100.),
            self.series(False, True), self.series(np.nan, 100.), self.series(False, True))
        result = ledger.close_step(self.series(1., 0.), self.series(50., 100.),
            self.series(True, True), self.series(50., 100.), self.series(True, True))
        self.assertEqual(result["certified_nav"], 50.)

    def test_nontransaction_mark_is_not_sell_authorization(self):
        ledger = Inventory(self.series(1., 0.), self.series(100., 100.), 0.)
        result = ledger.close_step(self.series(0., 1.), self.series(np.nan, 100.),
            self.series(False, True), self.series(80., 100.), self.series(False, True))
        self.assertEqual(ledger.quantity.a, 1.)
        self.assertEqual(result["certified_nav"], 80.)
        self.assertEqual(result["fills"].a, 0.)

    def test_cost_and_settlement_cash_nonnegative(self):
        ledger = Inventory(self.series(1., 0.), self.series(100., 100.), 0.)
        result = ledger.close_step(self.series(0., 1.), self.series(100., 100.),
            self.series(True, True), self.series(100., 100.), self.series(True, True))
        self.assertGreaterEqual(ledger.cash, -1e-10)
        self.assertAlmostEqual(result["provisional_nav"], 100. - result["cost"])
        self.assertGreater(result["cost"], 0.)

    def test_unknown_exit_is_conservative(self):
        ok, _ = buy_eligibility(self.series(100., 100.), self.series(1., 1.),
            self.series(1., 1.), self.series(1., 1.), self.series(np.nan, False))
        self.assertFalse(ok.a)


if __name__ == "__main__":
    unittest.main()

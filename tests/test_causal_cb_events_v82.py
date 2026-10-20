import unittest

import numpy as np
import pandas as pd

from src.v52_cb_execution import EVENT_COLUMNS
from src.v82_causal_cb_events import validate_causal_events, CausalQuantityBooks


class CausalCBTests(unittest.TestCase):
    def setUp(self):
        self.days = pd.bdate_range("2020-01-01", "2020-01-10")
        self.events = pd.DataFrame([dict(event_id="e", ts_code="X", kind="terminal", known_at="2020-01-01",
            record_date="2020-01-02", ex_date="2020-01-03", pay_date="2020-01-06", confirmed_at="2020-01-08",
            cash_yuan_per_bond=100., source="synthetic_test_fixture")])

    def new(self, events=None):
        return CausalQuantityBooks(["bucket"], ["X"], self.events if events is None else events,
                                   self.days, initial_capital=1000., cost_rate=0.)

    def step(self, book, day):
        i = self.days.get_loc(day)
        return book.step(day, [[1.]], [True], [100.], [True], [100.], signal_date=self.days[i - 1])

    def test_late_confirmation_is_valid_but_never_backdated(self):
        self.assertEqual(len(validate_causal_events(self.events, ["X"])), 1)
        book = self.new()
        self.step(book, "2020-01-02")
        ex = self.step(book, "2020-01-03")
        self.assertEqual(ex["receivable"][0], 1000)
        self.assertEqual(book.quantity.sum(), 0)
        early = self.step(book, "2020-01-06")
        self.assertEqual(early["payment_cash"][0], 0)
        self.step(book, "2020-01-07")
        paid = self.step(book, "2020-01-08")
        self.assertEqual(paid["payment_cash"][0], 1000)
        self.assertEqual(paid["receivable"][0], 0)
        self.assertEqual(self.step(book, "2020-01-09")["payment_cash"][0], 0)

    def test_removing_future_confirmation_does_not_change_past(self):
        pending = self.events.copy()
        pending.confirmed_at = pd.NaT
        a, b = self.new(), self.new(pending)
        for day in self.days[1:5]:
            x, y = self.step(a, day), self.step(b, day)
            for key in ("provisional_nav", "cash", "receivable", "payment_cash", "fills"):
                np.testing.assert_array_equal(x[key], y[key])

    def test_invalid_same_day_signal_rolls_back(self):
        book = self.new()
        with self.assertRaises(ValueError):
            book.step("2020-01-02", [[1.]], [True], [100.], [True], [100.], signal_date="2020-01-02")
        self.assertIsNone(book.last_day)
        self.assertEqual(book.cash[0], 1000)

    def test_unknown_schedule_not_used_for_retirement(self):
        events = self.events.iloc[:0].copy()
        a = self.new(events)
        self.step(a, "2020-01-02")
        self.step(a, "2020-01-03")
        self.assertEqual(a.quantity[0, 0], 10)

    def test_early_confirmation_cannot_accelerate_payment(self):
        events = self.events.copy()
        events.confirmed_at = "2020-01-02"
        book = self.new(events)
        self.step(book, "2020-01-02")
        result = self.step(book, "2020-01-03")
        self.assertEqual(result["payment_cash"][0], 0)
        self.assertEqual(result["receivable"][0], 1000)
        self.assertEqual(self.step(book, "2020-01-06")["payment_cash"][0], 1000)

    def test_failure_after_accrual_restores_inventory_and_rights(self):
        book = self.new()
        self.step(book, "2020-01-02")
        with self.assertRaises(ValueError):
            book.step("2020-01-03", [[-1.]], [True], [100.], [True], [100.],
                      signal_date="2020-01-02")
        self.assertEqual(book.quantity[0, 0], 10)
        self.assertEqual(book.last_day, pd.Timestamp("2020-01-02"))
        self.assertFalse(book.accrued)
        self.assertFalse(book.receivables)
        self.assertEqual(self.step(book, "2020-01-03")["receivable"][0], 1000)

    def test_coupon_keeps_security_inventory(self):
        events = self.events.copy()
        events.kind = "coupon"
        events.cash_yuan_per_bond = 1.
        book = self.new(events)
        self.step(book, "2020-01-02")
        result = self.step(book, "2020-01-03")
        self.assertEqual(book.quantity[0, 0], 10)
        self.assertEqual(result["receivable"][0], 10)
        self.assertEqual(result["cash"][0], 0)


if __name__ == "__main__":
    unittest.main()

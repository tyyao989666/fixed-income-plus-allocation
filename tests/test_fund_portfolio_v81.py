import unittest

import numpy as np
import pandas as pd

from src.v81_fund_portfolio import FundPortfolio


class FundPortfolioTests(unittest.TestCase):
    def setUp(self):
        self.codes = pd.Index(["A", "B"])
        self.days = pd.bdate_range("2020-01-01", "2020-01-10")
        self.prices = pd.Series([100., 100.], index=self.codes)
        self.volume = pd.Series([1000., 1000.], index=self.codes)
        self.empty = pd.DataFrame()

    def book(self, schedules=None, receipts=None):
        return FundPortfolio(self.codes, "2020-01-01", 1000.,
            self.empty if schedules is None else schedules, receipts=receipts, sessions=self.days)

    def trade(self, book, day, target, signal, prices=None, volume=None):
        return book.step(day, self.prices if prices is None else prices,
            self.volume if volume is None else volume, target=pd.Series(target, index=self.codes),
            signal_date=signal, signal_buy_allowed=pd.Series(True, index=self.codes))

    def test_buy_cost_and_self_financing(self):
        book = self.book()
        row = self.trade(book, "2020-01-02", [.5, .5], "2020-01-01")
        self.assertAlmostEqual(row["cost"], row["gross_traded_notional"] * .0005)
        self.assertAlmostEqual(row["observed_mark_nav"], 1000 - row["cost"])
        self.assertGreaterEqual(row["spendable_cash"], -1e-8)

    def test_same_day_and_stale_target_rejected_without_mutation(self):
        book = self.book()
        with self.assertRaises(ValueError):
            self.trade(book, "2020-01-02", [.5, .5], "2020-01-02")
        self.assertEqual(book.day, pd.Timestamp("2020-01-01"))
        with self.assertRaises(ValueError):
            self.trade(book, "2020-01-03", [.5, .5], "2020-01-01")

    def test_zero_volume_keeps_frozen_inventory(self):
        book = self.book()
        self.trade(book, "2020-01-02", [1., 0.], "2020-01-01")
        before = book.book.quantity.copy()
        row = self.trade(book, "2020-01-03", [0., 1.], "2020-01-02", volume=pd.Series([0., 1000.], index=self.codes))
        pd.testing.assert_series_equal(book.book.quantity, before)
        self.assertAlmostEqual(row["cost"], 0, places=10)
        self.assertGreater(row["frozen_market_value"], 0)

    def test_held_missing_quote_invalidates_marked_nav(self):
        book = self.book()
        self.trade(book, "2020-01-02", [1., 0.], "2020-01-01")
        row = book.step("2020-01-03", pd.Series([np.nan, 100.], index=self.codes), self.volume)
        self.assertTrue(np.isnan(row["observed_mark_nav"]))
        self.assertGreater(book.book.quantity.A, 0)

    def test_sale_after_record_retains_receivable_and_no_early_payment(self):
        schedules = pd.DataFrame([dict(ts_code="A", event_id="d", known_at=pd.Timestamp("2020-01-01"),
            record_date=pd.Timestamp("2020-01-02"), ex_date=pd.Timestamp("2020-01-03"),
            pay_date=pd.Timestamp("2020-01-06"), cash_per_unit=1., eligible_exdate_research=True)])
        book = self.book(schedules)
        self.trade(book, "2020-01-02", [1., 0.], "2020-01-01")
        qty = book.book.quantity.A
        row = self.trade(book, "2020-01-03", [0., 1.], "2020-01-02", prices=pd.Series([99., 100.], index=self.codes))
        self.assertAlmostEqual(row["receivable"], qty)
        self.assertAlmostEqual(book.book.quantity.A, 0)
        before_cash = row["spendable_cash"]
        later = book.step("2020-01-06", self.prices, self.volume)
        self.assertAlmostEqual(later["receivable"], qty)
        self.assertAlmostEqual(later["spendable_cash"], before_cash)

    def test_foreign_security_receipt_rejected_atomically(self):
        schedules = pd.DataFrame([dict(ts_code="A", event_id="d", known_at=pd.Timestamp("2020-01-01"),
            record_date=pd.Timestamp("2020-01-02"), ex_date=pd.Timestamp("2020-01-03"),
            pay_date=pd.Timestamp("2020-01-06"), cash_per_unit=1., eligible_exdate_research=True)])
        receipt = dict(ts_code="B", event_id="d", actual_payment_date="2020-01-06",
            receipt_available_at="2020-01-06", gross_cash_amount=1000 / 100.05, settlement_evidence_verified=True)
        book = self.book(schedules, receipts=[receipt])
        self.trade(book, "2020-01-02", [1., 0.], "2020-01-01")
        book.step("2020-01-03", self.prices, self.volume)
        cash = book.book.cash
        with self.assertRaises(ValueError):
            book.step("2020-01-06", self.prices, self.volume)
        self.assertEqual(book.day, pd.Timestamp("2020-01-03"))
        self.assertEqual(book.book.cash, cash)

    def test_receivable_cannot_finance_frozen_asset_rotation(self):
        schedules = pd.DataFrame([dict(ts_code="A", event_id="d", known_at=pd.Timestamp("2020-01-01"),
            record_date=pd.Timestamp("2020-01-02"), ex_date=pd.Timestamp("2020-01-03"),
            pay_date=pd.Timestamp("2020-01-06"), cash_per_unit=1., eligible_exdate_research=True)])
        book = self.book(schedules)
        self.trade(book, "2020-01-02", [1., 0.], "2020-01-01")
        row = self.trade(book, "2020-01-03", [0., 1.], "2020-01-02",
            prices=pd.Series([99., 100.], index=self.codes), volume=pd.Series([0., 1000.], index=self.codes))
        self.assertGreater(row["receivable"], 0)
        self.assertAlmostEqual(book.book.quantity.B, 0, places=10)

    def test_confirmed_receipt_preserves_nav_and_never_pays_twice(self):
        schedules = pd.DataFrame([dict(ts_code="A", event_id="d", known_at=pd.Timestamp("2020-01-01"),
            record_date=pd.Timestamp("2020-01-02"), ex_date=pd.Timestamp("2020-01-03"),
            pay_date=pd.Timestamp("2020-01-06"), cash_per_unit=1., eligible_exdate_research=True)])
        book = self.book(schedules)
        self.trade(book, "2020-01-02", [1., 0.], "2020-01-01")
        ex_prices = pd.Series([99., 100.], index=self.codes)
        before = book.step("2020-01-03", ex_prices, self.volume)
        book.receipts = [dict(ts_code="A", event_id="d", actual_payment_date="2020-01-06",
            receipt_available_at="2020-01-07", gross_cash_amount=before["receivable"],
            settlement_evidence_verified=True)]
        early = book.step("2020-01-06", ex_prices, self.volume)
        paid = book.step("2020-01-07", ex_prices, self.volume)
        self.assertAlmostEqual(early["observed_mark_nav"], paid["observed_mark_nav"])
        self.assertAlmostEqual(paid["spendable_cash"], early["spendable_cash"] + early["receivable"])
        self.assertEqual(paid["receivable"], 0)
        again = book.step("2020-01-08", ex_prices, self.volume)
        self.assertAlmostEqual(again["spendable_cash"], paid["spendable_cash"])


if __name__ == "__main__":
    unittest.main()

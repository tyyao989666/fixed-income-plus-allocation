import unittest

import pandas as pd

from src.v79_fund_entitlement_book import FundEntitlementBook


class EntitlementTests(unittest.TestCase):
    def setUp(self):
        self.events = pd.DataFrame([dict(ts_code="X", event_id="one", known_at=pd.Timestamp("2020-01-01"),
            record_date=pd.Timestamp("2020-01-02"), ex_date=pd.Timestamp("2020-01-03"),
            pay_date=pd.Timestamp("2020-01-06"), cash_per_unit=1., eligible_exdate_research=True)])

    def test_exdate_is_receivable_not_investable_cash(self):
        book = FundEntitlementBook("X", "2020-01-02", 10, 100, self.events)
        row = book.step("2020-01-03", 99)
        self.assertEqual(row["receivable"], 10)
        self.assertEqual(row["cash"], 0)
        self.assertEqual(row["observed_mark_nav"], 1000)
        row = book.step("2020-01-06", 99)
        self.assertEqual(row["cash"], 0)
        self.assertEqual(row["receivable"], 10)

    def test_verified_receipt_transfers_claim_once(self):
        receipt = dict(ts_code="X", event_id="one", actual_payment_date="2020-01-06",
            receipt_available_at="2020-01-07", gross_cash_amount=10., settlement_evidence_verified=True)
        book = FundEntitlementBook("X", "2020-01-02", 10, 100, self.events, receipts=[receipt])
        book.step("2020-01-03", 99)
        early = book.step("2020-01-06", 99)
        self.assertEqual(early["cash"], 0)
        paid = book.step("2020-01-07", 99)
        self.assertEqual(paid["cash"], 10)
        self.assertEqual(paid["receivable"], 0)
        self.assertEqual(book.step("2020-01-08", 99)["cash"], 10)

    def test_deleting_future_receipt_does_not_change_past(self):
        receipt = dict(ts_code="X", event_id="one", actual_payment_date="2020-01-06",
            receipt_available_at="2020-01-07", gross_cash_amount=10., settlement_evidence_verified=True)
        a = FundEntitlementBook("X", "2020-01-02", 10, 100, self.events, receipts=[receipt])
        b = FundEntitlementBook("X", "2020-01-02", 10, 100, self.events)
        self.assertEqual(a.step("2020-01-03", 99), b.step("2020-01-03", 99))

    def test_split_preserves_value_not_a_fake_profit(self):
        split = dict(ts_code="X", event_id="split", known_at="2020-01-01", effective_date="2020-01-03",
            new_units_per_old_unit=10, primary_source_verified=True)
        book = FundEntitlementBook("X", "2020-01-02", 1, 100, self.events.iloc[:0], splits=[split])
        row = book.step("2020-01-03", 10)
        self.assertEqual(row["quantity"], 10)
        self.assertEqual(row["observed_mark_nav"], 100)

    def test_missing_price_never_fabricates_observed_nav(self):
        book = FundEntitlementBook("X", "2020-01-02", 10, 100, self.events)
        row = book.step("2020-01-03", None)
        self.assertIsNone(row["observed_mark_nav"])
        self.assertEqual(row["quantity"], 10)

    def test_record_or_exdate_cannot_be_skipped(self):
        book = FundEntitlementBook("X", "2020-01-01", 10, 100, self.events)
        with self.assertRaises(ValueError):
            book.step("2020-01-03", 99)
        other = FundEntitlementBook("X", "2020-01-02", 10, 100, self.events)
        with self.assertRaises(ValueError):
            other.step("2020-01-06", 99)

    def test_failed_step_is_atomic(self):
        book = FundEntitlementBook("X", "2020-01-01", 10, 100, self.events)
        with self.assertRaises(ValueError):
            book.step("2020-01-03", 99)
        self.assertEqual(book.day, pd.Timestamp("2020-01-01"))
        self.assertEqual(book.history, {pd.Timestamp("2020-01-01"): 10.})
        book.step("2020-01-02", 100)
        self.assertEqual(book.step("2020-01-03", 99)["receivable"], 10)

    def test_late_known_split_never_changes_earlier_prefix(self):
        split = dict(ts_code="X", event_id="late_split", known_at="2020-01-06", effective_date="2020-01-03",
            new_units_per_old_unit=10, primary_source_verified=True)
        a = FundEntitlementBook("X", "2020-01-02", 1, 100, self.events.iloc[:0], splits=[split])
        b = FundEntitlementBook("X", "2020-01-02", 1, 100, self.events.iloc[:0])
        self.assertEqual(a.step("2020-01-03", 10), b.step("2020-01-03", 10))
        with self.assertRaises(ValueError):
            a.step("2020-01-06", 10)


if __name__ == "__main__":
    unittest.main()

import unittest

import pandas as pd

from src.v76_fund_cashflows import normalize_schedule


class FundScheduleTests(unittest.TestCase):
    def setUp(self):
        self.row = dict(ts_code="X", div_proc="实施", ann_date="20200101", imp_anndate="20200102",
            base_date="20191231", record_date="20200103", ex_date="20200106", pay_date="20200108", div_cash=.1)

    def test_duplicate_schedule_not_double_cash(self):
        f, status = normalize_schedule(pd.DataFrame([self.row, self.row]), "X", "2020-01-10")
        self.assertEqual(len(f), 1)
        self.assertEqual(status["eligible_events"], 1)
        self.assertFalse(f.settlement_certified.any())

    def test_conflict_blocks_event(self):
        f, status = normalize_schedule(pd.DataFrame([self.row, dict(self.row, div_cash=.2)]), "X", "2020-01-10")
        self.assertEqual(status["conflicting_events"], 1)
        self.assertFalse(f.eligible_exdate_research.any())

    def test_late_announcement_never_usable_at_earlier_exdate(self):
        f, _ = normalize_schedule(pd.DataFrame([dict(self.row, imp_anndate="20200107")]), "X", "2020-01-10")
        self.assertFalse(f.eligible_exdate_research.any())
        empty, _ = normalize_schedule(pd.DataFrame([self.row]), "X", "2019-12-31")
        self.assertTrue(empty.empty)

    def test_payment_before_exdate_rejected(self):
        f, _ = normalize_schedule(pd.DataFrame([dict(self.row, pay_date="20200103")]), "X", "2020-01-10")
        self.assertFalse(f.eligible_exdate_research.any())


if __name__ == "__main__":
    unittest.main()

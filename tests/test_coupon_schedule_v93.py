import unittest

import pandas as pd

from src.v93_coupon_events import augment_coupon_events


def event(kind="coupon"):
    return dict(event_id="coupon", ts_code="A", kind=kind, known_at="2019-01-29",
                record_date="2019-02-01", ex_date="2019-02-11", pay_date="2019-02-15",
                confirmed_at=pd.NaT, cash_yuan_per_bond=1.5, source="reviewed_schedule")


class CouponTests(unittest.TestCase):
    def empty(self):
        return pd.DataFrame(columns=list(event()))

    def test_ex_and_pay_are_distinct(self):
        out = augment_coupon_events(self.empty(), [event()], ["A"])
        self.assertLess(out.ex_date.iloc[0], out.pay_date.iloc[0])
        self.assertTrue(pd.isna(out.confirmed_at.iloc[0]))

    def test_no_assumed_confirmation(self):
        row = event()
        row["confirmed_at"] = "2019-02-15"
        with self.assertRaises(ValueError):
            augment_coupon_events(self.empty(), [row], ["A"])

    def test_terminal_overlap_rejected(self):
        terminal = event("terminal")
        terminal.update(event_id="terminal", record_date="2019-01-30", ex_date="2019-01-31", pay_date="2019-01-31")
        with self.assertRaises(ValueError):
            augment_coupon_events(pd.DataFrame([terminal]), [event()], ["A"])

    def test_duplicate_rejected(self):
        with self.assertRaises(ValueError):
            augment_coupon_events(pd.DataFrame([event()]), [event()], ["A"])

    def test_future_schedule_cannot_create_prior_entitlement(self):
        row = event()
        row["known_at"] = "2019-02-12"
        with self.assertRaises(ValueError):
            augment_coupon_events(self.empty(), [row], ["A"])


if __name__ == "__main__":
    unittest.main()

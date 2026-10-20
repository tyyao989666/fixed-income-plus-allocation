import unittest
import pandas as pd
from src.v88_terminal_controls import terminal_controls
from scripts.replay_v88_inventory_maturities import CASES, exact_total


class TerminalControlTests(unittest.TestCase):
    def setUp(self):
        self.calendar = pd.bdate_range("2020-01-01", "2020-01-15")
        self.schedule = pd.DataFrame([dict(ts_code="X", known_at="2020-01-06", last_trade_date="2020-01-08",
            source="synthetic_test_fixture", schedule_sha256="a" * 64)])

    def run_control(self, signal, execution, schedule=None):
        return terminal_controls(["X", "Y"], self.schedule if schedule is None else schedule,
                                 signal, execution, self.calendar)

    def test_unseen_schedule_does_not_change_past_instructions(self):
        actual = self.run_control("2020-01-03", "2020-01-06")
        self.assertFalse(actual.to_numpy().any())

    def test_known_schedule_requests_exit_only_in_remaining_trade_window(self):
        actual = self.run_control("2020-01-06", "2020-01-07")
        self.assertTrue(actual.at["X", "block_new_buy"])
        self.assertTrue(actual.at["X", "request_liquidation"])
        self.assertFalse(actual.at["X", "block_execution"])
        self.assertFalse(actual.loc["Y"].any())

    def test_final_trading_day_allowed_but_next_day_cannot_fake_exit(self):
        self.assertTrue(self.run_control("2020-01-07", "2020-01-08").at["X", "request_liquidation"])
        late = self.run_control("2020-01-08", "2020-01-09")
        self.assertFalse(late.at["X", "request_liquidation"])
        self.assertTrue(late.at["X", "block_execution"])

    def test_late_disclosure_never_backdates_exit(self):
        schedule = self.schedule.copy()
        schedule.known_at = "2020-01-09"
        late = self.run_control("2020-01-09", "2020-01-10", schedule)
        self.assertTrue(late.at["X", "needs_frozen_inventory_review"])
        self.assertFalse(late.at["X", "request_liquidation"])

    def test_results_and_future_returns_have_no_influence(self):
        schedule = self.schedule.assign(confirmed_at="2030-01-01", future_return=1000)
        pd.testing.assert_frame_equal(self.run_control("2020-01-06", "2020-01-07"),
                                     self.run_control("2020-01-06", "2020-01-07", schedule))

    def test_same_day_signal_and_ambiguous_schedule_rejected(self):
        with self.assertRaises(ValueError):
            self.run_control("2020-01-06", "2020-01-06")
        with self.assertRaises(ValueError):
            self.run_control("2020-01-06", "2020-01-07", pd.concat([self.schedule, self.schedule]))

    def test_hope_conflict_and_two_exact_totals(self):
        expected = {"113545.SH": True, "127015.SZ": False, "128134.SZ": True}
        for case in CASES:
            self.assertEqual(exact_total(case)["exact_total_check"], expected[case["code"]])
        hope = exact_total(next(r for r in CASES if r["code"] == "127015.SZ"))
        self.assertEqual(hope["discrepancy_yuan"], "-30004.80")


if __name__ == "__main__":
    unittest.main()

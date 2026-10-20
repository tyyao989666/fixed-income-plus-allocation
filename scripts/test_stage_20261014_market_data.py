"""No-network regression tests for the acquisition quarantine."""
from pathlib import Path
import sys
import unittest
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.stage_20261014_market_data import normalize_bars, overlap_check, tables, tasks


def response(zone="+08:00", currency="元", date="2026-10-14"):
    return {"data": {"columns": [{"name": "TIME"}, {"name": "MATCH"}],
                     "rows": [[date+"T00:00:00.000"+zone, "304.9493"]], "unit": {"MATCH": currency}}}


class QuarantineTests(unittest.TestCase):
    def test_domestic_positive_bar(self):
        frame = normalize_bars(response())
        self.assertEqual(frame.date.iloc[0], pd.Timestamp("2026-10-14"))

    def test_silent_us_mapping_blocked(self):
        with self.assertRaisesRegex(ValueError, "timezone"):
            normalize_bars(response(zone="-04:00", currency="美元"))

    def test_currency_blocked(self):
        with self.assertRaisesRegex(ValueError, "currency"):
            normalize_bars(response(currency="USD"))

    def test_no_future_bar(self):
        with self.assertRaisesRegex(ValueError, "cutoff"):
            normalize_bars(response(date="2026-10-15"))

    def test_overlap_not_automatically_rebased(self):
        a = pd.DataFrame({"date": pd.bdate_range("2026-07-01", periods=8), "close": range(100, 108)})
        self.assertTrue(overlap_check(a, a)["passed"])
        b = a.copy()
        b["close"] *= 2
        self.assertFalse(overlap_check(a, b)["passed"])
        self.assertFalse(overlap_check(a.iloc[:2], a)["passed"])

    def test_vendor_codes_are_separate(self):
        task = next(t for t in tasks() if t["asset"] == "rate_30y")
        self.assertEqual(task["code"], "CBA00661.CS")
        self.assertEqual(task["source_code"], "CBA00661.CB")

    def test_edb_schema_preserves_missing_and_does_not_invent_release(self):
        edb = {"data": {"date": ["20261013", "20261014"], "indicatorInfo": [
            {"code": "X", "name": "test", "data": [None, 1.2], "unit": "%", "source": "provider"}]}}
        frame, units = list(tables(edb))[0]
        self.assertEqual(len(frame), 2)
        self.assertTrue(pd.isna(frame.value.iloc[0]))
        self.assertNotIn("release_date", frame)
        self.assertEqual(units["value"], "%")


if __name__ == "__main__":
    unittest.main()

"""Do not confuse auxiliary direct labels with additional market observations."""
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd

from src.v47_history_supplement import aggregate_auxiliary, merge_supplement


def fixture():
    return pd.DataFrame([dict(as_of_date=pd.Timestamp("2022-01-04"), task="sleeve", entity="equity",
        horizon_days=20, prediction=.02, training_dates=422, available=True,
        train_end=pd.Timestamp("2020-03-25"), training_label_end=pd.Timestamp("2020-06-24"),
        encoder_validation_end=pd.Timestamp("2020-12-28"), issue_cutoff=pd.Timestamp("2021-12-31"))])


class HistoryTests(unittest.TestCase):
    def test_fixed_horizon_units(self):
        q = aggregate_auxiliary(fixture())
        self.assertAlmostEqual(q.increment_21.iloc[0], .02*.5*21/20*.25)

    def test_actual_300_minimum_unchanged(self):
        q = fixture()
        q["training_dates"] = 299
        self.assertTrue(aggregate_auxiliary(q).empty)

    def test_current_missing_not_filled(self):
        q = fixture()
        q["available"] = False
        self.assertTrue(aggregate_auxiliary(q).empty)

    def test_future_label_rejected(self):
        for field in ["training_label_end", "train_end", "encoder_validation_end", "issue_cutoff"]:
            q = fixture()
            q[field] = q.as_of_date
            with self.subTest(field=field), self.assertRaises(ValueError):
                aggregate_auxiliary(q)

    def test_cannot_change_recent_forecasts(self):
        q = fixture()
        q["as_of_date"] = pd.Timestamp("2024-01-02")
        with self.assertRaises(ValueError):
            aggregate_auxiliary(q)

    def test_duplicates_rejected(self):
        with self.assertRaises(ValueError):
            aggregate_auxiliary(pd.concat([fixture(), fixture()]))

    def test_existing_good_forecast_cannot_be_replaced(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "aux.parquet"
            fixture().to_parquet(path)
            with self.assertRaises(ValueError):
                merge_supplement(aggregate_auxiliary(fixture()), path)


if __name__ == "__main__":
    unittest.main()

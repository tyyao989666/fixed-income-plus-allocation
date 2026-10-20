import unittest
import pandas as pd
from src.v70_observation_dates import month_end_session


class ObservationTests(unittest.TestCase):
    def test_weekend_label_uses_prior_actual_session(self):
        dates=pd.to_datetime(['2021-07-29','2021-07-30','2021-08-02'])
        self.assertEqual(month_end_session('2021-07-31',dates),pd.Timestamp('2021-07-30'))

    def test_not_filled_from_another_month(self):
        with self.assertRaises(ValueError):
            month_end_session('2021-07-31',pd.to_datetime(['2021-06-30','2021-08-02']))

    def test_no_future_session(self):
        self.assertEqual(month_end_session('2021-07-29',pd.to_datetime(['2021-07-29','2021-07-30'])),pd.Timestamp('2021-07-29'))


if __name__=='__main__':
    unittest.main()

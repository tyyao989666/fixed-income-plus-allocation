"""Independent reporting arithmetic and missing-input guards."""
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_v45_research import performance, paired_bootstrap, validate_daily, ReportInputError


class ReportTests(unittest.TestCase):
    def test_standard_sharpe(self):
        x = np.array([.001, -.002, .003, .001])
        self.assertAlmostEqual(performance(x)['sharpe'], x.mean()/x.std(ddof=1)*np.sqrt(252))
        self.assertAlmostEqual(performance(x)['cagr'], np.prod(1+x)**(252/len(x))-1)

    def test_paired_zero_increment(self):
        x = np.tile([.001, -.0005, .002], 30)
        result = paired_bootstrap(x, x, 20, 100)
        self.assertEqual(result['cagr_positive_fraction'], 0)
        self.assertEqual(result['sharpe_positive_fraction'], 0)

    def test_missing_inputs_rejected(self):
        with self.assertRaises(ReportInputError):
            validate_daily(pd.DataFrame())
        with self.assertRaises(ReportInputError):
            performance([.01, np.nan])


if __name__ == '__main__':
    unittest.main(verbosity=2)

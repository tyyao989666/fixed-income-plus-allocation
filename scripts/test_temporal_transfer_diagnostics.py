import sys
from pathlib import Path
import unittest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from src.temporal_transfer_diagnostics import residual_alignment


class TransferTests(unittest.TestCase):
    def test_perfect_and_reversed(self):
        y = np.array([[1., 2.], [3., 4.]])
        m = np.ones_like(y, bool)
        correct = residual_alignment(y, y, m)
        reversed_ = residual_alignment(-y, y, m)
        self.assertAlmostEqual(correct['alignment'], 1.)
        self.assertAlmostEqual(correct['raw_MSE_ratio'], 0.)
        self.assertAlmostEqual(reversed_['alignment'], -1.)
        self.assertAlmostEqual(reversed_['raw_MSE_ratio'], 4.)

    def test_zero_increment_is_control(self):
        y = np.ones((3, 2))
        r = residual_alignment(np.zeros_like(y), y, np.ones_like(y, bool))
        self.assertEqual(r['raw_MSE_ratio'], 1.)
        self.assertTrue(np.isnan(r['alignment']))

    def test_missing_and_empty(self):
        y = np.array([[1., np.nan], [2., 999.]])
        m = np.array([[True, False], [True, False]])
        r = residual_alignment(y, y, m)
        self.assertEqual(r['valid_cells'], 2)
        self.assertAlmostEqual(r['alignment'], 1.)
        self.assertTrue(np.isnan(residual_alignment(y, y, m & False)['raw_MSE_ratio']))

    def test_date_not_observation_weighted(self):
        x = np.array([[0., 0.], [1., 0.]])
        y = np.array([[1., 1.], [1., 0.]])
        m = np.array([[True, True], [True, False]])
        self.assertAlmostEqual(residual_alignment(x, y, m)['raw_MSE_ratio'], .5)


if __name__ == '__main__':
    unittest.main()

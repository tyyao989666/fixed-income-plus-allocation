from pathlib import Path
import sys
import unittest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from src.v65_online_grid_control import online_grid_control


class OnlineGridTests(unittest.TestCase):
    def setUp(self):
        self.d = (np.arange(40)+1)*100
        self.e = self.d[:, None]+200
        self.y = np.arange(80).reshape(40, 1, 2).astype(float)/100
        self.m = np.ones_like(self.y, bool)
        self.g = np.zeros(2, int)
        self.r = np.arange(40) % 9

    def run_control(self, y=None, r=None, q=None, m=None):
        return online_grid_control(self.d, self.e, self.y if y is None else y, self.m if m is None else m,
            self.g, self.r if r is None else r, np.arange(40), np.array([20]) if q is None else q,
            minimum_group_dates=2, half_life_sessions=5, shrinkage=4)

    def test_future_and_same_day_ignored(self):
        before = self.run_control()
        y = self.y.copy()
        y[18:] = 999
        after = self.run_control(y=y)
        for key in ['online_mean', 'online_grid']:
            np.testing.assert_array_equal(before[key], after[key])

    def test_prefix_replay(self):
        full = self.run_control()
        prefix = online_grid_control(self.d[:21], self.e[:21], self.y[:21], self.m[:21], self.g, self.r[:21],
            np.arange(21), np.array([20]), minimum_group_dates=2, half_life_sessions=5, shrinkage=4)
        np.testing.assert_array_equal(full['online_grid'], prefix['online_grid'])

    def test_unknown_grid_falls_back(self):
        r = self.r.copy()
        r[20] = -1
        result = self.run_control(r=r)
        np.testing.assert_array_equal(result['online_mean'], result['online_grid'])

    def test_same_constant_history_returns_constant(self):
        result = self.run_control(y=np.full_like(self.y, .03))
        np.testing.assert_allclose(result['online_mean'], .03)
        np.testing.assert_allclose(result['online_grid'], .03)

    def test_missing_not_zero(self):
        result = self.run_control(m=self.m & False)
        self.assertTrue(np.isnan(result['online_mean']).all())
        self.assertTrue(np.isnan(result['online_grid']).all())

    def test_unmatured_and_future_regime_mutation(self):
        before = self.run_control()
        r = self.r.copy()
        r[21:] = -1
        r[18:20] = -1
        after = self.run_control(r=r)
        np.testing.assert_array_equal(before['online_grid'], after['online_grid'])

    def test_online_update_waits_for_maturity(self):
        y = self.y.copy()
        y[18] = 9
        old = self.run_control(q=np.array([20, 21]))
        changed = self.run_control(y=y, q=np.array([20, 21]))
        np.testing.assert_array_equal(old['online_mean'][0], changed['online_mean'][0])
        self.assertGreater(np.abs(old['online_mean'][1]-changed['online_mean'][1]).sum(), 0)

    def test_old_label_window_and_inputs(self):
        result = self.run_control()
        self.assertTrue((result['latest_maturity'] < self.d[20]).all())
        self.assertTrue((result['joint_weight_ESS'] >= 0).all())
        with self.assertRaises(ValueError):
            self.run_control(r=np.full_like(self.r, 10))


if __name__ == '__main__':
    unittest.main()

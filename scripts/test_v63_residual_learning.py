from pathlib import Path
import sys
import unittest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from src.v63_residual_learning import historical_control, fit_residual_scale, residual_loss, fit_gain


class ResidualTests(unittest.TestCase):
    def setUp(self):
        self.d = np.arange(1, 31, dtype=np.int64)*100
        self.e = self.d[:, None]+200
        self.y = np.arange(60, dtype=float).reshape(30, 1, 2)/100
        self.m = np.ones_like(self.y, bool)
        self.g = np.zeros(2, int)

    def control(self, y=None, sources=None, queries=None, mask=None):
        return historical_control(self.d, self.e, self.y if y is None else y,
            self.m if mask is None else mask, self.g,
            np.arange(30) if sources is None else sources,
            np.array([10, 15, 20]) if queries is None else queries,
            minimum_group_dates=2, minimum_entity_labels=2)

    def test_future_and_same_day_label_ignored(self):
        before = self.control(queries=np.array([10]))
        altered = self.y.copy()
        altered[8:] = 999  # row 8 matures on query day; not allowed
        after = self.control(y=altered, queries=np.array([10]))
        np.testing.assert_array_equal(before[0], after[0])
        self.assertTrue((before[2] < self.d[10]).all())

    def test_frozen_sources_ignore_calibration_truth(self):
        source = np.arange(10)
        before = self.control(sources=source)
        altered = self.y.copy()
        altered[10:] = -999
        np.testing.assert_array_equal(before[0], self.control(y=altered, sources=source)[0])

    def test_prefix_causality(self):
        full = self.control(queries=np.array([10]))[0]
        short = historical_control(self.d[:11], self.e[:11], self.y[:11], self.m[:11], self.g,
            np.arange(11), np.array([10]), minimum_group_dates=2, minimum_entity_labels=2)[0]
        np.testing.assert_array_equal(full, short)

    def test_warmup_is_missing(self):
        value, _, _ = self.control(queries=np.array([0, 1, 2]))
        self.assertTrue(np.isnan(value).all())

    def test_masked_labels_never_enter(self):
        m = self.m.copy()
        m[:8, :, 0] = False
        altered = self.y.copy()
        altered[:8, :, 0] = 999
        np.testing.assert_array_equal(self.control(mask=m)[0], self.control(y=altered, mask=m)[0])

    def test_gain_units_and_no_direction_flip(self):
        x = np.ones((110, 3, 2))*.02
        m = np.ones_like(x, bool)
        beta = fit_gain(x, .5*x, m, self.g)
        np.testing.assert_allclose(beta, .5/1.1, rtol=1e-6)
        self.assertTrue((fit_gain(x, -x, m, self.g) == 0).all())
        self.assertTrue((fit_gain(x, 9*x, m, self.g) == 1).all())

    def test_scale_train_only_and_group_shared(self):
        scale = fit_residual_scale(self.y, self.m, self.g)
        self.assertEqual(scale[0, 0], scale[0, 1])
        self.assertTrue(np.isfinite(scale).all())

    def test_loss_prefers_correct_residual_and_gradients(self):
        target = {t: torch.ones(4, 3, 2)*.1 for t in ['asset', 'sleeve']}
        mask = {t: torch.ones_like(v, dtype=torch.bool) for t, v in target.items()}
        group = {t: torch.zeros(2, dtype=torch.long) for t in target}
        pred = {t: torch.zeros_like(v, requires_grad=True) for t, v in target.items()}
        zero = residual_loss(pred, target, mask, group)
        exact = residual_loss(target, target, mask, group)
        self.assertLess(exact.item(), zero.item())
        zero.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for p in pred.values()))


if __name__ == '__main__':
    unittest.main()

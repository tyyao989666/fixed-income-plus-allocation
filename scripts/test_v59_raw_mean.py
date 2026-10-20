import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from src.v59_raw_mean import RawMeanHead, fit_shared_scale, raw_mean_loss, ep_mean_view


class RawMeanTests(unittest.TestCase):
    def test_shared_scaling_preserves_rank_and_roundtrip(self):
        y = np.arange(60, dtype=np.float32).reshape(5, 3, 4)/100
        c, s, ok = fit_shared_scale(y[:3], np.ones_like(y[:3], bool), [0, 0, 1, 1])
        self.assertTrue(ok.all())
        self.assertTrue(np.array_equal(s[:, 0], s[:, 1]))
        self.assertTrue(np.array_equal(c[:, 0], c[:, 1]))
        p = (y-c)/s
        np.testing.assert_allclose(p*s+c, y, atol=1e-6)
        self.assertTrue(np.array_equal(np.argsort(p[:, :, :2]), np.argsort(y[:, :, :2])))

    def test_missing_does_not_fit_zero(self):
        y = np.ones((5, 3, 2), np.float32)
        mask = np.zeros_like(y, bool)
        c, s, ok = fit_shared_scale(y, mask, [0, 0])
        self.assertFalse(ok.any())
        self.assertTrue(np.isfinite(c).all() and (s > 0).all())

    def test_future_not_used_in_scale(self):
        y = np.arange(60, dtype=np.float32).reshape(5, 3, 4)
        first = fit_shared_scale(y[:3], np.ones_like(y[:3], bool), [0, 0, 1, 1])
        y[3:] = 1e20
        second = fit_shared_scale(y[:3], np.ones_like(y[:3], bool), [0, 0, 1, 1])
        for x, z in zip(first, second):
            np.testing.assert_array_equal(x, z)

    def test_heads_loss_and_gradients(self):
        cfg = dict(mean_weight=1., pinball_weight=.25, rank_weight=.05)
        for kind in ["linear", "regularized"]:
            model = RawMeanHead(40, kind)
            mean, q = model(torch.randn(4, 5, 40))
            self.assertEqual(mean.shape, (4, 3, 5))
            self.assertTrue((q[..., 0] <= q[..., 1]).all() and (q[..., 1] <= q[..., 2]).all())
            target = torch.randn_like(mean)
            mask = torch.ones_like(mean, dtype=torch.bool)
            value = raw_mean_loss(mean, q, target, mask, torch.arange(5), cfg)
            self.assertTrue(torch.isfinite(value))
            value.backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_EP_rejects_unapproved_scores(self):
        base = dict(units="cumulative_raw_relative_price_return", horizon_days=20,
                    as_of="2023-01-03", available_after="2023-01-04", mean=[.01])
        with self.assertRaises(ValueError):
            ep_mean_view(base)
        with self.assertRaises(ValueError):
            ep_mean_view(dict(base, units="risk_normalized_opportunity", approved=True, execution_certified=True))
        np.testing.assert_array_equal(ep_mean_view(dict(base, approved=True, execution_certified=True)), [.01])

    def test_matched_validation_identity(self):
        from scripts.run_v59_raw_mean import score
        rng = np.random.default_rng(17)
        base = {"asset": rng.normal(size=(8, 3, 4))}
        target = {"asset": rng.normal(size=(8, 3, 4))}
        mask = {"asset": np.ones((8, 3, 4), bool)}
        ratio, delta = score(base, base, target, mask, {"asset": np.array([0, 0, 1, 1])})
        self.assertAlmostEqual(ratio, 1.)
        self.assertAlmostEqual(delta, 0.)


if __name__ == "__main__":
    unittest.main()

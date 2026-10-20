from pathlib import Path
import sys
import unittest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from src.research_rank_metrics import rank_metrics_named
from scripts.run_v61_state_ranker import rank_metrics


class RankMetricTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(17)
        self.y = {t: rng.normal(size=(8, 3, n)) for t, n in [('asset', 6), ('sleeve', 5)]}
        self.p = {t: v + rng.normal(size=v.shape)*.2 for t, v in self.y.items()}
        self.m = {t: np.ones(v.shape, bool) for t, v in self.y.items()}
        self.g = dict(asset=np.repeat(np.arange(3), 2), sleeve=np.zeros(5, int))
        self.b = {t: np.zeros_like(v) for t, v in self.y.items()}

    def calculate(self, baselines, reference):
        return rank_metrics_named(self.p, baselines, self.y, self.m, self.g, reference)

    def test_baseline_rename_does_not_drop_absolute_ic(self):
        a = self.calculate({'prior': self.b}, 'prior')
        b = self.calculate({'historical_control': self.b}, 'historical_control')
        np.testing.assert_allclose([a[0], a[2]], [b[0], b[2]])
        self.assertTrue(np.isfinite(b[2]))

    def test_legacy_aggregation_matches(self):
        old = rank_metrics(self.p, {'prior': self.b}, self.y, self.m, self.g)
        new = self.calculate({'prior': self.b}, 'prior')
        np.testing.assert_allclose([old[0], old[2]], [new[0], new[2]])
        self.assertEqual(old[1], new[1])

    def test_explicit_missing_reference_fails(self):
        with self.assertRaises(ValueError):
            self.calculate({'momentum': self.b}, 'missing')

    def test_constant_prediction_is_zero_information(self):
        self.p = self.b
        result = self.calculate({'control': self.b}, 'control')
        self.assertEqual(result[0], 0.)
        self.assertEqual(result[2], 0.)

    def test_missing_values_use_shared_support(self):
        self.b['asset'][:, :, 0] = np.nan
        result = self.calculate({'control': self.b}, 'control')
        self.assertTrue(np.isfinite(result[2]))


if __name__ == '__main__':
    unittest.main()

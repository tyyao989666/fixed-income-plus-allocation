import sys
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.v61_state_momentum import monthly_cohort_votes, price_rule_signals, fit_calibrator, rank_loss


class StateTests(unittest.TestCase):
    def test_positive_cohorts_and_prefix_causality(self):
        dates = pd.bdate_range("2010-01-01", "2015-12-31")
        f = pd.DataFrame(.001, index=dates, columns=["a", "b"])
        full = monthly_cohort_votes(f)
        self.assertAlmostEqual(full.iloc[-1, 0], 1.)
        cut = f.index.get_loc("2014-03-17")
        prefix = monthly_cohort_votes(f.iloc[:cut+1])
        pd.testing.assert_series_equal(full.loc[f.index[cut]], prefix.iloc[-1], check_names=False)

    def test_missing_month_does_not_carry_stale_vote(self):
        f = pd.DataFrame(.001, index=pd.bdate_range("2010-01-01", "2016-12-30"), columns=["a"])
        f.loc["2014-02-03", "a"] = np.nan
        result = monthly_cohort_votes(f)
        self.assertTrue(pd.isna(result.loc["2014-03-03", "a"]))

    def test_boundary_missing_and_price_rule_named(self):
        f = pd.DataFrame(.01, index=pd.bdate_range("2010-01-01", "2016-12-30"), columns=["cb", "bond"])
        s = price_rule_signals(f, np.array([2, 0]))
        self.assertTrue(s["value_boundary"].isna().all().all())
        pd.testing.assert_series_equal(s["price_soft"].bond, s["cohort"].bond)

    def test_calibrator_monotonic_and_units(self):
        x = np.tile(np.arange(5, dtype=np.float32)[:, None, None], (1, 3, 2))
        prior = np.zeros((3, 2), np.float32)
        mask = np.ones_like(x, bool)
        beta = fit_calibrator(x, .02*x, mask, prior, np.array([0, 0]), ridge=0, minimum_dates=0)
        np.testing.assert_allclose(beta, .02, atol=1e-7)
        np.testing.assert_array_equal(fit_calibrator(x, -.02*x, mask, prior, np.array([0, 0])), 0)

    def test_rank_loss_prefers_correct_order_and_gradient(self):
        y = torch.tensor([[[0., .01, .02]]]*3).transpose(0, 1)
        y = y.repeat(2, 1, 1)
        p = y.clone().requires_grad_(True)
        mask = torch.ones_like(y, dtype=torch.bool)
        group = {"asset": torch.zeros(3, dtype=torch.long)}
        good = rank_loss({"asset":p}, {"asset":y}, {"asset":mask}, group, 0)
        bad = rank_loss({"asset":-p}, {"asset":y}, {"asset":mask}, group, 0)
        self.assertLess(good.item(), bad.item())
        good.backward()
        self.assertTrue(torch.isfinite(p.grad).all())

    def test_nested_time_split_isolated(self):
        from src import v45_forecast as engine
        from src.v61_state_momentum import nested_split
        root = Path(__file__).resolve().parents[1]
        with np.load(root / "data_processed/corporate_action_v58/dataset.npz") as z:
            a = {k:z[k] for k in z.files}
        cfg = engine.configuration(root / "config/v45_research.yaml")
        ends = engine.end_times(a)
        for year in [2023, 2024]:
            split = nested_split(a, engine.shared_split(a, year, cfg))
            self.assertGreaterEqual(len(split["train"]), 300)
            self.assertTrue((ends[split["train"]].max(1) < a["dates"][split["calibration"][0]]).all())
            self.assertTrue((ends[split["calibration"]].max(1) < a["dates"][split["validation"][0]]).all())
        with self.assertRaises(ValueError):
            nested_split(a, engine.shared_split(a, 2022, cfg))

    def test_constant_baseline_is_zero_information(self):
        from scripts.run_v61_state_ranker import rank_metrics
        y = np.tile(np.array([0., 1., 2.])[None, None, :], (5, 3, 1))
        p = {"asset": y}
        metric, delta, ic = rank_metrics(p, {"prior":{"asset":np.zeros_like(y)}}, p,
            {"asset":np.ones_like(y, bool)}, {"asset":np.zeros(3, int)})
        self.assertAlmostEqual(metric, 1.)

    def test_full_encoder_receives_gradient(self):
        from src import v45_forecast as engine
        from src.v49_masked_forecast import training_implementation
        from src.v61_state_momentum import StateRanker
        cfg = engine.configuration(Path(__file__).resolve().parents[1] / "config/v45_research.yaml")
        arrays = dict(local_values=np.zeros((130, 5, 3), np.float32), macro_values=np.zeros((130, 2), np.float32), module_index=np.arange(5))
        with training_implementation():
            model = StateRanker(engine.ForecastNetwork(arrays, cfg), 40)
            x = torch.randn(2, 120, 5, 3)
            p = model((x, torch.ones_like(x), torch.randn(2, 120, 2)))
            y = {t:torch.randn_like(v) for t,v in p.items()}
            m = {t:torch.ones_like(v, dtype=torch.bool) for t,v in p.items()}
            loss = rank_loss(p, y, m, dict(asset=torch.zeros(5, dtype=torch.long), sleeve=torch.zeros(5, dtype=torch.long)))
            loss.backward()
            first = model.network.core.encoder.layers[0]
            self.assertTrue(any(p.grad is not None and torch.count_nonzero(p.grad) > 0 for p in first.parameters()))
            self.assertFalse(model.network.local_reconstruct.weight.requires_grad)


if __name__ == "__main__":
    unittest.main()

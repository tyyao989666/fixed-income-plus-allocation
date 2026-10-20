"""Focused causal/label/split tests; optionally verify prepared frozen output."""
from pathlib import Path
import argparse
import json
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from src.v45_data import (HORIZONS, MODULES, NAT, LABEL_REFERENCE_VERSION, annual_splits, label_reference_policy, local_features,
                          observable_grid, reference_weights, relative_labels, robust_scale)


class DataContractTests(unittest.TestCase):
    def test_scaler_shift_and_prefix_invariance(self):
        raw = pd.DataFrame({"x": np.arange(360, dtype=float)})
        values, masks = robust_scale(raw)
        self.assertFalse(masks[:126].any())
        self.assertTrue(masks[126:].all())
        self.assertAlmostEqual(float(values[126, 0]), (126 - 62.5) / 62.5, places=6)
        changed = raw.copy()
        changed.iloc[250:] = 1e6
        other, other_masks = robust_scale(changed)
        np.testing.assert_array_equal(values[:250], other[:250])
        np.testing.assert_array_equal(masks[:250], other_masks[:250])
        changed = raw.copy()
        changed.iloc[200] = 10000
        other, _ = robust_scale(changed)
        self.assertAlmostEqual(float(other[200, 0]), (10000 - 99.5) / 99.5, places=5)

    def test_features_do_not_bridge_gaps_or_prelisting(self):
        idx = pd.bdate_range("2018-01-02", periods=400)
        returns = pd.DataFrame({"a": np.linspace(-0.01, 0.01, 400), "prelisting": np.nan}, index=idx)
        returns.loc[idx[200], "a"] = np.nan
        bars = {k: pd.DataFrame(np.nan, index=idx, columns=returns.columns)
                for k in ("open", "high", "low", "close", "vol", "amount")}
        values, masks, names = local_features(returns, bars)
        self.assertFalse(masks[:, 1].any())
        self.assertTrue((values[:, 1] == 0).all())
        self.assertFalse(masks[200:205, 0, names.index("mom_5d")].any())
        for name in ("overnight_ret", "intraday_ret", "log_vol", "log_amount"):
            self.assertFalse(masks[:, :, names.index(name)].any())
        changed = returns.copy()
        changed.iloc[320:, 0] = 0.8
        v2, m2, _ = local_features(changed, bars)
        np.testing.assert_array_equal(values[:320], v2[:320])
        np.testing.assert_array_equal(masks[:320], m2[:320])

    def fixture(self):
        dates = pd.bdate_range("2020-01-01", periods=80).asi8
        # 6 assets, two rates and one in each other sleeve.
        modules = np.array([0, 0, 1, 2, 3, 4])
        returns = np.full((80, 6), 0.001)
        returns[:, 0] = 0.01
        returns[:, 1] = -0.004
        observed = np.ones_like(returns, dtype=bool)
        preferred = np.zeros((5, 6))
        preferred[0, :2] = [0.6, 0.4]
        for s in range(1, 5):
            preferred[s, s + 1] = 1
        sw = reference_weights(returns, observed, preferred, modules)
        bw = np.tile(np.array([0.28, 0.32, 0.06, 0.14, 0.20]), (80, 1))
        return dates, returns, observed, modules, preferred, sw, bw

    def test_labels_exact_buy_hold_drift_and_units(self):
        dates, r, mask, modules, _, sw, bw = self.fixture()
        out = relative_labels(dates, r, mask, modules, sw, bw)
        growth = np.prod(1 + r[1:6], axis=0) - 1
        rate_ref = 0.6 * growth[0] + 0.4 * growth[1]
        self.assertAlmostEqual(float(out["asset_target"][0, 0, 0]), growth[0] - rate_ref, places=7)
        self.assertNotAlmostEqual(rate_ref, (1 + 0.6 * 0.01 + 0.4 * -0.004) ** 5 - 1, places=7)
        sleeve_ref = sw[0] @ growth
        np.testing.assert_allclose(out["sleeve_target"][0, 0], sleeve_ref - bw[0] @ sleeve_ref, atol=1e-8)
        self.assertEqual(out["label_end"][0, 0], dates[5])
        self.assertTrue((out["label_end"][-5:] == NAT).all())
        self.assertFalse(out["asset_target_mask"][-5:].any())
        # r[t] is not in the forward payoff.
        r[0] = 0.7
        out2 = relative_labels(dates, r, mask, modules, sw, bw)
        np.testing.assert_array_equal(out["asset_target"][0], out2["asset_target"][0])

    def test_gap_is_not_future_renormalized(self):
        dates, r, mask, modules, _, sw, bw = self.fixture()
        mask[3, 1] = False
        r[3, 1] = 0  # frozen placeholder must not be treated as an observation
        out = relative_labels(dates, r, mask, modules, sw, bw)
        self.assertFalse(out["asset_target_mask"][0, 0, :2].any())
        self.assertTrue(out["asset_target_mask"][0, 0, 2:].all())
        self.assertFalse(out["sleeve_target_mask"][0, 0].any())
        # The missing rate return is outside this subsequent horizon.
        self.assertTrue(out["asset_target_mask"][4, 0].all())

    def test_reits_prelisting_only_affects_own_component(self):
        dates, r, mask, modules, preferred, _, bw = self.fixture()
        mask[:30, -1] = False
        r[:30, -1] = 0
        sw = reference_weights(r, mask, preferred, modules)
        self.assertTrue((sw[:30, 4] == 0).all())
        out = relative_labels(dates, r, mask, modules, sw, bw)
        self.assertTrue(out["asset_target_mask"][0, 0, :-1].all())
        self.assertFalse(out["asset_target_mask"][0, 0, -1])
        self.assertTrue(out["asset_target_mask"][30, 0, -1])

    def test_membership_at_t_not_future_or_input_mutation(self):
        _, r, mask, modules, preferred, _, _ = self.fixture()
        before = mask.copy()
        sw = reference_weights(r, mask, preferred, modules)
        mask[1:, 1] = False
        other = reference_weights(r, mask, preferred, modules)
        np.testing.assert_array_equal(sw[0], other[0])
        np.testing.assert_array_equal(before[0], mask[0])
        self.assertEqual(other[1, 0, 0], 1)

    def test_missing_us_does_not_kill_other_alt_benchmark(self):
        dates, r, mask, modules, preferred, _, bw = self.fixture()
        # Existing alternative is the economic core; append a context-only US token.
        r = np.column_stack([r, np.full(len(r), 0.002)])
        mask = np.column_stack([mask, np.ones(len(mask), dtype=bool)])
        modules = np.r_[modules, 4]
        preferred = np.column_stack([preferred, np.zeros(5)])
        sw = reference_weights(r, mask, preferred, modules)
        before = relative_labels(dates, r, mask, modules, sw, bw)
        mask[3, -1] = False
        r[3, -1] = 0
        sw2 = reference_weights(r, mask, preferred, modules)
        after = relative_labels(dates, r, mask, modules, sw2, bw)
        np.testing.assert_array_equal(sw, sw2)
        np.testing.assert_array_equal(before["sleeve_target_mask"], after["sleeve_target_mask"])
        np.testing.assert_array_equal(before["asset_target_mask"][:, :, :-1], after["asset_target_mask"][:, :, :-1])
        np.testing.assert_array_equal(before["asset_target"][:, :, :-1], after["asset_target"][:, :, :-1])
        self.assertTrue(after["asset_target_mask"][0, 0, -2])
        self.assertFalse(after["asset_target_mask"][0, 0, -1])

    def test_observable_grid_only_history_and_missing_unknown(self):
        idx = pd.bdate_range("2018-01-02", periods=100)
        r = pd.DataFrame({"rate_7_10y": 0.001, "equity_broad": -0.001}, index=idx)
        grid, masks, regime = observable_grid(r)
        self.assertFalse(masks[:19].any())
        self.assertTrue((regime[:19] == -1).all())
        self.assertEqual(grid.shape, (100, 11))
        self.assertTrue((grid[19:, :9].sum(axis=1) == 1).all())
        np.testing.assert_allclose(grid[19:, 9], 0.5 + 0.5*np.tanh(((1.001**20)-1)/.001), atol=1e-7)
        np.testing.assert_allclose(grid[19:, 10], 0.5 + 0.5*np.tanh(((.999**20)-1)/.005), atol=1e-7)
        r.iloc[70:] = 0.5
        grid2, _, _ = observable_grid(r)
        np.testing.assert_array_equal(grid[:70], grid2[:70])
        r.iloc[50] = np.nan
        _, mask2, _ = observable_grid(r)
        self.assertFalse(mask2[50:70].any())
        r = pd.DataFrame({"rate_7_10y": 0.00001, "equity_broad": 0.00002}, index=idx)
        r.iloc[50, 0] = np.nan
        values, masks, _ = observable_grid(r)
        self.assertFalse(masks[50:70, :10].any())
        self.assertTrue(masks[50:70, 10].all())
        self.assertTrue((values[50:70, :10] == 0).all())

    def test_annual_splits_maturity_purges_and_dict_api(self):
        dates = pd.bdate_range("2018-01-02", "2026-07-15").asi8
        ends = np.full((len(dates), 3), NAT, dtype=np.int64)
        for j, h in enumerate(HORIZONS):
            ends[:-h, j] = dates[h:]
        arrays = dict(dates=dates, label_end=ends, sample_indices=np.arange(119, len(dates)))
        split = annual_splits(arrays, 2024)
        self.assertEqual(split["reason"], "")
        self.assertEqual(len(split["validation"]), 126)
        self.assertEqual(len(split["gate"]), 126)
        self.assertEqual(split["validation"][0] - split["train"][-1] - 1, 60)
        self.assertEqual(split["gate"][0] - split["validation"][-1] - 1, 60)
        self.assertTrue((ends[split["train"][-1]] < dates[split["validation"][0]]).all())
        self.assertTrue((ends[split["validation"][-1]] < dates[split["gate"][0]]).all())
        self.assertTrue((ends[split["gate"], -1] <= split["cutoff"]).all())
        self.assertGreaterEqual(split["train"][0], 119)
        self.assertTrue((pd.to_datetime(dates[split["test"]]).year == 2024).all())
        self.assertNotEqual(annual_splits(arrays, 2020)["reason"], "")
        # All REIT labels can be absent without dropping every training date.
        arrays["asset_target_mask"] = np.ones((len(dates), 3, 38), dtype=bool)
        arrays["asset_target_mask"][:, :, 4] = False
        again = annual_splits(arrays, 2024)
        np.testing.assert_array_equal(split["train"], again["train"])


def verify_prepared(root=ROOT):
    with np.load(root / "data_processed/v45_dataset.npz", allow_pickle=False) as z, np.load(
            root / "data_processed/hierarchical_asset_itransformer_v43_clean_calendar_dataset.npz", allow_pickle=False) as frozen:
        meta = json.loads((root / "output/reports/v45_metadata.json").read_text(encoding="utf-8"))
        for name in ("dates", "returns", "return_masks", "module_index"):
            np.testing.assert_array_equal(z[name], frozen[name])
        assert z["dates"].dtype == np.int64
        assert z["label_end"].dtype == np.int64
        assert z["local_values"].shape == z["local_masks"].shape == (2069, 38, len(meta["local_features"]))
        assert z["macro_values"].shape == z["macro_masks"].shape == (2069, 11)
        assert len(meta["macro_features"]) == 11
        assert meta["macro_features"][-2:] == ["grid_bond_score", "grid_equity_score"]
        assert (z["macro_values"] >= 0).all() and (z["macro_values"] <= 1).all()
        assert z["asset_target"].shape == z["asset_target_mask"].shape == (2069, 3, 38)
        assert z["sleeve_target"].shape == z["sleeve_target_mask"].shape == (2069, 3, 5)
        for value, mask in (("local_values", "local_masks"), ("macro_values", "macro_masks"),
                            ("asset_target", "asset_target_mask"), ("sleeve_target", "sleeve_target_mask")):
            assert np.isfinite(z[value]).all(), value
            assert (z[value][~z[mask].astype(bool)] == 0).all(), value
        reits = meta["assets"].index("alt_reits")
        prelisting = z["dates"] < pd.Timestamp("2021-06-22").value
        assert not z["return_masks"][prelisting, reits].any()
        assert not z["local_masks"][prelisting, reits].any()
        assert not z["asset_target_mask"][prelisting, :, reits].any()
        assert z["asset_target_mask"][prelisting].any()
        assert meta["label_policy"]["version"] == LABEL_REFERENCE_VERSION
        _, policy = label_reference_policy(meta["assets"], meta["asset_modules"])
        assert meta["label_policy"]["preferred_weights"] == policy
        for module in ("rates", "convertible_bonds", "equity", "alternatives"):
            assert len(set(policy[module].values())) == 1, module
        assert policy["credit"] == {"credit_ind_aaa": 1.0}
        for context_only in ("alt_us_bond", "alt_reits", "equity_tech_growth"):
            i = meta["assets"].index(context_only)
            assert not z["sleeve_reference_weights"][:, :, i].any()
        split2024 = annual_splits(z, 2024)
        valid60 = z["sleeve_target_mask"][split2024["train"], 2].sum(axis=0)
        assert (valid60 >= 300).all(), ("all five sleeve h60 train dates must be usable by 2024", valid60)
        for year in range(2022, 2027):
            split = annual_splits(z, year)
            assert split["reason"] == "", (year, split["reason"])
        print("Prepared snapshot: frozen returns/masks unchanged; tensor shapes, finite placeholders, REIT gaps and annual splits passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", action="store_true")
    args, remaining = parser.parse_known_args()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(DataContractTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
    if args.prepared:
        verify_prepared()

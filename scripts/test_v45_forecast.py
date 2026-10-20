"""Focused chronology, model-training and artifact-contract tests (synthetic only)."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.v45_forecast import (DEFAULTS, PREDICTION_COLUMNS, ForecastNetwork, GroupRidge,
    annual_splits, batch_at, bh_adjust, bootstrap_lower, clip_fold_features,
    component_gate, configuration, historical_signature, prediction_frame,
    quarterly_oof, raw_features, residual_scaling, shared_split, synthetic_arrays,
    targets_at, train_fold, train_network, validate_arrays, candidate_selection,
    forecast_preflight, require_selectable_preflight)


class ForecastTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.a = synthetic_arrays(1100)
        cls.cfg = copy.deepcopy(DEFAULTS)

    def test_shared_config_mapping(self):
        cfg = configuration(ROOT / "config/v45_research.yaml")
        self.assertEqual((cfg["pretrain_max_epochs"], cfg["max_epochs"]), (1000, 1000))
        self.assertEqual((cfg["pretrain_minimum_epochs"], cfg["supervised_minimum_epochs"]), (100, 50))
        self.assertEqual(cfg["causal_suffix_days"], [1, 5, 20])
        self.assertEqual(cfg["full_path_mask_fraction"], .25)
        self.assertEqual((cfg["patience"], cfg["pretrain_patience"]), (40, 40))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"bad.json"
            path.write_text(json.dumps(dict(model=dict(pairwise_loss_weight=.3))))
            with self.assertRaises(ValueError):
                configuration(path)
            path.write_text(json.dumps(dict(model=dict(ignored_learning_rate=.1))))
            with self.assertRaises(ValueError):
                configuration(path)

    def test_split_chronology_and_no_future_maturity(self):
        validate_arrays(self.a)
        s = annual_splits(self.a, 2021)
        self.assertEqual(len(s["validation"]), 126)
        self.assertEqual(len(s["gate"]), 126)
        for left, right in (("train", "validation"), ("validation", "gate")):
            self.assertGreater(s[right][0]-s[left][-1], 60)
            self.assertTrue((self.a["label_end"][s[left]] < self.a["dates"][s[right][0]]).all())
        for key in ("train", "validation", "gate"):
            self.assertTrue((self.a["label_end"][s[key]] <= self.a["dates"][s["cutoff_index"]]).all())
        self.assertFalse(annual_splits(self.a, 2019)["valid"])

    def test_signature_is_historical_and_config_sensitive(self):
        s = shared_split(self.a, 2021, self.cfg)
        original = historical_signature(self.a, s, self.cfg, {})
        changed = {k: v.copy() for k, v in self.a.items()}
        changed["local_values"][s["test"][0]:] += 5000
        changed["asset_target"][s["test"][0]:] += 20
        self.assertEqual(original, historical_signature(changed, s, self.cfg, {}))
        changed["local_values"][s["train"][0], 0, 0] += 1
        self.assertNotEqual(original, historical_signature(changed, s, self.cfg, {}))
        cfg = dict(self.cfg, head_lr=.1)
        self.assertNotEqual(original, historical_signature(self.a, s, cfg, {}))

    def test_oof_baseline_is_past_matured_and_sparse_groups_unavailable(self):
        train = np.arange(119, 700)
        a = {k: v.copy() for k, v in self.a.items()}
        a["asset_target_mask"][:, :, a["module_index"] == 4] = False
        a["asset_target_mask"][600:650, :, a["module_index"] == 4] = True
        oof, audit = quarterly_oof(a, train, self.cfg)
        self.assertTrue(np.isnan(oof["asset"][:, :, a["module_index"] == 4]).all())
        self.assertTrue(np.isnan(oof["asset"][:126]).all())
        for block in audit:
            if block["status"] != "unavailable":
                self.assertLess(block["latest_label_end"], block["block_start"])
        # Mutating a later quarter cannot change an earlier OOF prediction.
        changed = {k: v.copy() for k, v in a.items()}
        changed["asset_target"][650:] += 100
        changed["local_values"][650:] += 5e6
        second, _ = quarterly_oof(changed, train, self.cfg)
        np.testing.assert_allclose(oof["asset"][train < 650], second["asset"][train < 650], equal_nan=True)

    def test_residual_raw_unit_inverse_and_mask(self):
        indices = np.arange(200, 330)
        y, m = targets_at(self.a, indices)
        base = {t: np.full_like(x, .002) for t, x in y.items()}
        base["asset"][0] = np.nan
        r, good, center, scale = residual_scaling(y, base, m, self.a["module_index"])
        self.assertFalse(good["asset"][0].any())
        for task in y:
            np.testing.assert_allclose((r[task]*scale[task]+center[task]+base[task])[good[task]],
                y[task][good[task]], atol=2e-8)

    def test_t_only_feature_clipping(self):
        train = np.arange(150, 350)
        a, audit = clip_fold_features(self.a, train, self.cfg)
        changed = {k: v.copy() for k, v in self.a.items()}
        changed["local_values"][500:] = 5e6
        b, second = clip_fold_features(changed, train, self.cfg)
        np.testing.assert_array_equal(audit["lower"], second["lower"])
        np.testing.assert_array_equal(audit["upper"], second["upper"])
        self.assertLess(np.max(np.abs(b["local_values"][500:])), 100.)
        self.assertTrue(np.isfinite(a["local_values"]).all())

    def test_ridge_fit_only_clipping_is_saved_and_not_refit_on_predict(self):
        train = np.arange(200, 400)
        model = GroupRidge(self.a["module_index"]).fit(raw_features(self.a, train), *targets_at(self.a, train))
        before = {key: (values[3].copy(), values[4].copy()) for key, values in model.models.items()}
        future = raw_features(self.a, np.arange(500, 510))
        for values in future.values():
            values[:] = 5e6
        result = model.predict(future)
        for key, (_, _, _, lower, upper) in model.models.items():
            np.testing.assert_array_equal(lower, before[key][0])
            np.testing.assert_array_equal(upper, before[key][1])
            self.assertLess(float(np.max(np.abs(upper))), 100.)
        self.assertTrue(np.isfinite(result["asset"]).all())

    def test_masked_reconstruction_really_trains_encoder(self):
        model = ForecastNetwork(self.a, self.cfg)
        before = model.core.encoder.layers[0].linear1.weight.detach().clone()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        loss = model.reconstruction_loss(batch_at(self.a, np.array([200, 201]), self.cfg, torch.device("cpu")))
        loss.backward()
        optimizer.step()
        self.assertTrue(torch.isfinite(loss))
        self.assertFalse(torch.equal(before, model.core.encoder.layers[0].linear1.weight))

    def test_nonfinite_gradient_fails_before_optimizer(self):
        cfg = dict(self.cfg, pretrain_max_epochs=1)
        split = dict(train=np.array([200, 201]), validation=np.arange(300, 340), year=2021)
        model = ForecastNetwork(self.a, cfg)
        handle = model.core.encoder.layers[0].linear1.weight.register_hook(lambda g: torch.full_like(g, float("inf")))
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(torch.optim.AdamW, "step") as optimizer_step:
                with self.assertRaises(RuntimeError):
                    train_network(model, self.a, split, cfg, torch.device("cpu"), Path(tmp)/"bad.pt", "bad", "pretraining", 17)
                optimizer_step.assert_not_called()
            self.assertFalse((Path(tmp)/"bad.pt").exists())
        handle.remove()

    def test_insufficient_evidence_and_constant_baseline_fail_gate(self):
        y, m = targets_at(self.a, np.arange(300, 426))
        base = {t: np.zeros_like(v) for t, v in y.items()}
        predictions = [{t: v.copy() for t, v in y.items()} for _ in range(3)]
        gate = component_gate(("asset", 0, 1), predictions, base, y, m,
            self.a["module_index"], [], self.cfg, True, True)
        self.assertFalse(gate["active"])
        self.assertIn("prior_matured_oos_two_years", gate["reason"])
        self.assertIn("paired_delta_rank_ic", gate["reason"])
        self.assertTrue(np.isnan(gate["delta_rank_ic"]))
        self.assertTrue(np.isnan(bootstrap_lower(np.ones(59), self.cfg)[0]))

    def test_output_all_candidates_and_gated_source(self):
        indices = np.arange(800, 805)
        y, _ = targets_at(self.a, indices)
        base = {t: np.full_like(v, .001) for t, v in y.items()}
        frame = prediction_frame(self.a, indices, y, base, "F2", -1, {}, self.cfg, "sig", "F2", {})
        self.assertEqual(list(frame), PREDICTION_COLUMNS)
        self.assertTrue((frame.active_increment == 0).all())
        np.testing.assert_allclose(frame.gated_prediction, frame.baseline_prediction)
        self.assertTrue(frame.is_oof.all())
        self.assertFalse(frame.component_gate_passed.any())
        self.assertTrue((bh_adjust([.001, np.nan, .2]) >= 0).all())

    def test_group_pass_cannot_activate_nan_or_unavailable_entity(self):
        indices = np.arange(800, 805)
        a = {k: v.copy() for k, v in self.a.items()}
        y, _ = targets_at(a, indices)
        base = {t: np.full_like(v, .001) for t, v in y.items()}
        y["asset"][0, 1, 0] = np.nan
        a["return_masks"][indices[1], 0] = False
        gate = {("asset", 0, 1): dict(active=True, reason="passed")}
        metadata = dict(forecast_timing=dict(entity_residual_training_dates={
            "asset": np.full((3, 10), 300), "sleeve": np.full((3, 5), 300)}))
        frame = prediction_frame(a, indices, y, base, "F2", -1, metadata, self.cfg, "sig", "F2", gate)
        entity = frame.loc[(frame.entity == "asset_0") & (frame.horizon_days == 20)].sort_values("as_of_date")
        self.assertTrue(entity.component_gate_passed.all())
        self.assertFalse(entity.active_component.iloc[:2].any())
        self.assertTrue((entity.active_increment.iloc[:2] == 0).all())
        self.assertTrue((entity.gate_reason.iloc[:2] == "missing_entity_forecast").all())
        self.assertTrue(entity.active_component.iloc[2:].all())
        self.assertTrue(np.isnan(entity.prediction.iloc[0]))

    def test_sparse_entity_cannot_piggyback_group_gate(self):
        indices = np.arange(800, 805)
        y, _ = targets_at(self.a, indices)
        base = {t: np.full_like(v, .001) for t, v in y.items()}
        counts = np.full((3, 10), 500)
        counts[1, 0] = 23
        metadata = dict(forecast_timing=dict(entity_residual_training_dates={"asset": counts}))
        frame = prediction_frame(self.a, indices, y, base, "F2", -1, metadata, self.cfg,
            "sig", "F2", {("asset", 0, 1): dict(active=True, reason="passed")})
        group = frame.loc[(frame.task == "asset") & (frame.module == "rates") & (frame.horizon_days == 20)]
        sparse = group.loc[group.entity == "asset_0"]
        mature = group.loc[group.entity == "asset_1"]
        self.assertTrue(group.component_gate_passed.all())
        self.assertTrue((sparse.entity_training_dates == 23).all())
        self.assertTrue((sparse.gate_reason == "insufficient_entity_training_history").all())
        self.assertTrue((sparse.active_increment == 0).all())
        self.assertFalse(sparse.active_component.any())
        self.assertTrue(np.isfinite(sparse.prediction).all())
        self.assertTrue(mature.active_component.all())

    def test_unavailable_f1_does_not_blank_valid_f2_f3_selection(self):
        y, m = targets_at(self.a, np.arange(300, 426))
        rng = np.random.default_rng(17)
        base = {task: values*.5+rng.normal(0, .005, values.shape) for task, values in y.items()}
        missing = {task: np.full_like(values, np.nan) for task, values in y.items()}
        pred = {"F1": [missing]*3, "F2": [y]*3, "F3": [y]*3}
        selected, scores, available = candidate_selection(pred, base, y, m, self.a["module_index"], self.cfg)
        self.assertTrue(available)
        self.assertTrue(np.isnan(scores["F1"]))
        self.assertTrue(np.isfinite(scores["F2"]))
        self.assertEqual(selected, "F2")

    def test_all_structurally_unavailable_folds_rejected(self):
        with self.assertRaisesRegex(ValueError, "All forecast folds"):
            require_selectable_preflight(dict(any_structurally_selectable_fold=False))
        require_selectable_preflight(dict(any_structurally_selectable_fold=True))

    def test_preflight_does_not_use_gate_targets(self):
        cfg = dict(self.cfg, years=[2021])
        split = shared_split(self.a, 2021, cfg)
        changed = {k: v.copy() for k, v in self.a.items()}
        for task in ("asset", "sleeve"):
            changed[task+"_target"][split["gate"]] = np.nan
        a = forecast_preflight(self.a, cfg)
        b = forecast_preflight(changed, cfg)
        self.assertEqual(a["folds"][0]["T_V_label_signature"], b["folds"][0]["T_V_label_signature"])
        self.assertEqual(a["any_structurally_selectable_fold"], b["any_structurally_selectable_fold"])

    def test_small_end_to_end_fold_and_resume(self):
        cfg = dict(self.cfg, max_epochs=1, pretrain_max_epochs=1)
        with tempfile.TemporaryDirectory(prefix="v45_contract_") as tmp:
            report, history = train_fold(self.a, {}, 2021, cfg, tmp, [], torch.device("cpu"), debug=True)
            path = Path(tmp)/"2021/predictions.parquet"
            frame = pd.read_parquet(path)
            self.assertEqual(set(frame.model), {"F0", "F1", "F2", "F3"})
            self.assertEqual(set(frame.loc[frame.model == "F3", "seed"]), {-1, 17, 29, 47})
            self.assertFalse(frame.active_component.any())
            self.assertTrue((frame.active_increment == 0).all())
            self.assertTrue((frame.train_end_date < frame.as_of_date).all())
            self.assertTrue((frame.gate_as_of_date < frame.as_of_date).all())
            self.assertFalse(report["representation_contract_pass"])
            self.assertEqual(len(report["stages"]), 9)
            again, _ = train_fold(self.a, {}, 2021, cfg, tmp, [], torch.device("cpu"), resume=True, debug=True)
            self.assertEqual(report["selected_candidate"], again["selected_candidate"])
            self.assertTrue(history)
            changed = dict(cfg, dropout=.35)
            with self.assertRaises(ValueError):
                train_fold(self.a, {}, 2021, changed, tmp, [], torch.device("cpu"), resume=True, debug=True)

    def test_three_spawn_workers_match_single_worker_and_keep_seed_order(self):
        cfg = dict(self.cfg, max_epochs=1, pretrain_max_epochs=1, seed_workers=1)
        with tempfile.TemporaryDirectory(prefix="v45_parallel_") as tmp:
            serial_dir, parallel_dir = Path(tmp)/"serial", Path(tmp)/"parallel"
            serial, _ = train_fold(self.a, {}, 2021, cfg, serial_dir, [], torch.device("cpu"), debug=True)
            parallel_cfg = dict(cfg, seed_workers=3)
            parallel, _ = train_fold(self.a, {}, 2021, parallel_cfg, parallel_dir, [], torch.device("cpu"), debug=True)
            self.assertEqual(parallel["completed_seeds"], [17, 29, 47])
            self.assertEqual([s["seed"] for s in parallel["stages"]], [17]*3+[29]*3+[47]*3)
            self.assertEqual(len({v["pid"] for v in parallel["worker_runtime"].values()}), 3)
            for runtime in parallel["worker_runtime"].values():
                self.assertEqual(runtime["torch_threads"], 2)
                self.assertTrue(all(v["threads"] <= 2 for v in runtime["blas_threads"]))
            left = pd.read_parquet(serial_dir/"2021/predictions.parquet")
            right = pd.read_parquet(parallel_dir/"2021/predictions.parquet")
            np.testing.assert_allclose(left.prediction, right.prediction, atol=1e-7, rtol=1e-6, equal_nan=True)
            np.testing.assert_allclose(left.baseline_prediction, right.baseline_prediction, atol=1e-7, rtol=1e-6, equal_nan=True)
            self.assertEqual(serial["selected_candidate"], parallel["selected_candidate"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

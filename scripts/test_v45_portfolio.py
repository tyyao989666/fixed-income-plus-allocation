"""Synthetic v45 contracts; never launches production training or refreshes data."""
from pathlib import Path
import copy
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from src.v45_portfolio import (ExecutionEnvironment, PROFILES, RISK_SLEEVES,
    allocate_target, bootstrap_comparison, config, dynamic_bounds, fit_seed,
    frame_hash, prediction_adapter, product_box_anchor, restore_model, simulate, split_history,
    output, statistics, target_cache_matches, target_signature, validate_effective_config)
from src.v45_solver import SolverContractError, causal_anchor, solve


def fixture(rows=70, profile="balanced"):
    cfg = config()
    dates = pd.bdate_range("2021-01-04", periods=rows)
    assets = [f"{s}_{j}" for s in RISK_SLEEVES for j in range(2)]
    modules = {f"{s}_{j}": s for s in RISK_SLEEVES for j in range(2)}
    anchor = pd.DataFrame(.1, index=dates, columns=assets)
    target = anchor.copy()
    target.iloc[:, 0] += .02
    target.iloc[:, 1] -= .02
    rng = np.random.default_rng(8)
    returns = pd.DataFrame(rng.normal(.0002, .004, (rows, 10)), index=dates, columns=assets)
    quotes = returns.notna()
    state = pd.DataFrame({"state_liquidity_shock": 0., "state_grid_bond": .2,
        "state_grid_equity": -.1, "state_card_credit_mean": .5}, index=dates)
    diag = pd.DataFrame({"active_components": 0, "forecast_confidence": 0.,
        "mean_prediction_uncertainty": 0., "forecast_available": False}, index=dates)
    for s in RISK_SLEEVES:
        diag[f"gated_score_{s}"] = 0.
        diag[f"gated_uncertainty_{s}"] = 0.
    return ExecutionEnvironment(profile, cfg, target, diag, state, anchor, returns, quotes, modules, anchor.iloc[0].values)


def forecast_row(**overrides):
    row = dict(as_of_date="2024-01-03", task="sleeve", entity="rates", module="rates", horizon_days=5,
        model="F1", seed=-1, prediction=.03, baseline_prediction=.01, active_increment=.02,
        selected_candidate="F1", train_end_date="2023-06-01", gate_as_of_date="2023-12-29",
        is_oof=True, component_gate_passed=True)
    return {**row, **overrides}


class PortfolioContracts(unittest.TestCase):
    def test_registered_config(self):
        cfg = config()
        validate_effective_config(cfg)
        self.assertEqual(cfg["controller"]["mechanical_no_trade_l1"], {"low_vol": .008, "balanced": .01, "high_vol": .012})
        self.assertEqual(cfg["portfolio"]["max_asset_active_weight"], cfg["portfolio"]["asset_active_limits"])
        self.assertEqual(cfg["controller"]["minimum_hold_days"], {"low_vol": 3, "balanced": 2, "high_vol": 1})
        self.assertIn("v43_clean_calendar", cfg["outputs"]["dataset_npz"])

    def test_fixed_horizon_conversion(self):
        f = pd.DataFrame([forecast_row(), forecast_row(horizon_days=20, active_increment=.04),
                          forecast_row(horizon_days=60, active_increment=.06)])
        result, status = prediction_adapter(f)
        self.assertAlmostEqual(result.increment_21.iloc[0], 21*(.15*.02/5+.5*.04/20+.35*.06/60))
        self.assertEqual(status["active_rows"], 3)

    def test_selected_ensemble_only(self):
        f = pd.DataFrame([forecast_row(), forecast_row(seed=17, active_increment=4.),
                          forecast_row(model="F2", active_increment=5.)])
        result, _ = prediction_adapter(f)
        self.assertAlmostEqual(result.increment_21.iloc[0], .02*.15*21/5)

    def test_unavailable_nan_not_valid_forecast(self):
        f = pd.DataFrame([forecast_row(model="F0", active_increment=0., prediction=np.nan, baseline_prediction=np.nan)])
        result, status = prediction_adapter(f)
        self.assertTrue(result.empty)
        self.assertEqual(status["unavailable_prediction_rows"], 1)

    def test_no_active_components_still_valid_shadow(self):
        result, status = prediction_adapter(pd.DataFrame([forecast_row(active_increment=0., component_gate_passed=False)]))
        self.assertTrue(result.empty)
        self.assertEqual(status["status"], "no_active_learned_forecasts")

    def test_time_contract_rejects_backfill(self):
        for field, value in [("train_end_date", "2024-01-03"), ("gate_as_of_date", "2024-02-01"), ("is_oof", False)]:
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "time contract"):
                prediction_adapter(pd.DataFrame([forecast_row(**{field: value})]))

    def test_missing_timing_not_faked_zero_valid(self):
        f = pd.DataFrame([forecast_row()]).drop(columns=["is_oof"])
        with self.assertRaisesRegex(ValueError, "lack timing evidence"):
            prediction_adapter(f)
        f["active_increment"] = 0.
        result, status = prediction_adapter(f)
        self.assertTrue(result.empty)
        self.assertEqual(status["missing_items"], ["is_oof"])

    def test_nonfinite_active_rejected(self):
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            prediction_adapter(pd.DataFrame([forecast_row(prediction=np.nan)]))

    def test_f0_cannot_contribute(self):
        with self.assertRaisesRegex(ValueError, "F0"):
            prediction_adapter(pd.DataFrame([forecast_row(model="F0", selected_candidate="F0")]))

    def test_dynamic_constraints_same_solver(self):
        raw = pd.Series(.2, index=RISK_SLEEVES)
        row = {"state_grid_block_credit": 1, "state_trend_equity": -1.0}
        trend = {"enabled": True, "affected_sleeves": ["equity"], "weak_threshold": -.35,
                 "max_underweight_from_base": {"balanced": .008}}
        box = dynamic_bounds(raw, row, {s:(0.,1.) for s in RISK_SLEEVES}, trend, "balanced")
        anchor = causal_anchor(raw, raw, box, .022, .08)
        mu = pd.Series([0.,1.,0.,1.,0.], index=RISK_SLEEVES)
        cov = pd.DataFrame(np.eye(5)*.001, index=RISK_SLEEVES, columns=RISK_SLEEVES)
        result = solve(anchor.weights, mu, cov, raw, box, None,
            {"max_active_deviation": .022, "max_sleeve_turnover": .08}, backend="cvxpy")
        self.assertLessEqual(result.weights.credit, .2+1e-8)
        self.assertLessEqual(result.weights.equity, box["equity"][1]+1e-8)
        self.assertLessEqual(np.abs(result.weights-raw).sum(), .08+1e-6)

    def test_product_box_priority_before_grid_cap(self):
        raw = pd.Series([.3265277149869643,.4388004505580496,.03008850023578156,
                         .059999999999999984,.14458333421920436], index=RISK_SLEEVES)
        bounds = {"rates":(.23,.53), "credit":(.23,.47), "convertible_bonds":(.03,.08),
            "equity":(.06,.16), "alternatives":(.1,.24)}
        projected = product_box_anchor(raw,bounds)
        self.assertGreaterEqual(projected.equity,.06)
        self.assertAlmostEqual(projected.equity,.06,places=14)
        self.assertLess(float((projected-raw).abs().sum()),1e-14)
        trend={"enabled":False,"affected_sleeves":[]}
        box = dynamic_bounds(projected,{"state_grid_block_equity":1.},bounds,trend,"balanced")
        self.assertEqual(box["equity"][0],.06)
        self.assertAlmostEqual(box["equity"][1],.06,places=14)
        effective = causal_anchor(projected,projected,box,.022,.08)
        np.testing.assert_allclose(effective.weights,projected,atol=1e-12)
        bad = raw.copy()
        bad.equity -= .01
        bad.rates += .01
        with self.assertRaisesRegex(SolverContractError,"requires_policy"):
            product_box_anchor(bad,bounds)
        audit = {}
        box = dynamic_bounds(raw,{"state_grid_block_equity":1.},bounds,trend,"balanced",audit)
        self.assertEqual(box["equity"],(.06,.06))
        self.assertGreater(audit["equity"],0.)

    def test_target_cache_independent_of_PPO(self):
        cfg = config()
        first, _ = target_signature(cfg,"low_vol",{"status":"synthetic"})
        cfg["controller"]["learning_rates"] = [.1,.3]
        cfg["controller"]["max_epochs"] = 1
        second, _ = target_signature(cfg,"low_vol",{"status":"synthetic"})
        self.assertEqual(first,second)

    def test_resume_reuses_only_matching_target_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            frame = fixture().target
            saved = {"signature":"current", "target_hashes":{}}
            for layer in ["A0","A1","A2","A3","R0","R1"]:
                path=output("balanced",layer,"targets",directory)
                path.parent.mkdir(parents=True,exist_ok=True)
                frame.to_parquet(path)
                saved["target_hashes"][layer]=frame_hash(frame)
            self.assertTrue(target_cache_matches("balanced",saved,"current",directory)[0])
            self.assertFalse(target_cache_matches("balanced",saved,"new_forecast",directory)[0])
            changed=frame.copy()
            changed.iloc[0,0]+=.001
            changed.to_parquet(output("balanced","A2","targets",directory))
            self.assertFalse(target_cache_matches("balanced",saved,"current",directory)[0])

    def test_asset_infeasible_not_relaxed(self):
        cfg = config()
        base = pd.Series({"a": .99, "b": .01})
        sleeves = pd.Series([1.,0.,0.,0.,0.], index=RISK_SLEEVES)
        with self.assertRaises(SolverContractError):
            allocate_target(base, sleeves, {"a":"rates", "b":"rates"}, pd.Series(dtype=float), "low_vol", cfg)

    def test_four_rates_split_preserved(self):
        cfg = config()
        split = pd.Series(cfg["portfolio"]["formal_anchor_rate_split"]["balanced"])
        sleeves = pd.Series([1.,0.,0.,0.,0.], index=RISK_SLEEVES)
        result = allocate_target(split, sleeves, dict.fromkeys(split.index,"rates"), split*100, "balanced", cfg)
        np.testing.assert_allclose(result, split, atol=1e-10)

    def test_true_mechanical_benchmark_own_state(self):
        env = fixture()
        rows = simulate(env, np.arange(40))
        np.testing.assert_allclose([r["strategy_return"] for r in rows], [r["mechanical_return"] for r in rows], atol=1e-14)
        again = simulate(env, np.arange(40))
        np.testing.assert_allclose([r["nav"] for r in rows], [r["nav"] for r in again])

    def test_cost_actual_drift_and_quote_freeze(self):
        env = fixture()
        context = env.initial_context()
        context.age = 20
        current = env.initial_weights(0)
        current[0] += .03
        current[1] -= .03
        env._quotes[1,0] = False
        _, _, drifted, _, info = env.step(0, 3, current, context)
        self.assertAlmostEqual(info["weights"][0], current[0])
        self.assertAlmostEqual(info["transaction_cost"], np.abs(info["weights"]-current).sum()*.0005)
        self.assertAlmostEqual(drifted.sum(),1.)
        self.assertFalse(np.allclose(drifted, info["weights"]))

    def test_missing_held_returns_fail(self):
        env = fixture()
        env._returns[1, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "Missing execution return"):
            env.step(0, 0, env.initial_weights(0), env.initial_context())

    def test_quote_rescaling_cannot_create_concentration_breach(self):
        env = fixture()
        current = np.r_[.03,.01,.48,.48,np.zeros(6)]
        target = np.r_[.30,.10,.30,.30,np.zeros(6)]
        env._target[0] = target
        env._anchor[0] = target
        env._quotes[1, 1] = False
        env._returns[1] = 0.
        context = env.initial_context()
        context.age = 20
        _, _, end, _, info = env.step(0, 3, current, context)
        self.assertTrue(info["quote_order_blocked"])
        self.assertGreater(info["proposed_concentration_residual"], 0.)
        self.assertEqual(info["executed_concentration_residual"], 0.)
        self.assertEqual(info["action_fraction"], 0.)
        self.assertEqual(info["transaction_cost"], 0.)
        np.testing.assert_array_equal(current, end)

    def test_existing_other_sleeve_breach_cannot_mask_new_quote_breach(self):
        env = fixture()
        current = np.r_[.03,.01,.50,0.,.23,.23,np.zeros(4)]
        target = np.r_[.30,.10,.15,.15,.15,.15,np.zeros(4)]
        env._target[0], env._anchor[0] = target, target
        env._quotes[1,1] = False
        env._returns[1] = 0.
        context=env.initial_context()
        context.age=20
        _,_,end,_,info=env.step(0,3,current,context)
        self.assertTrue(info["quote_order_blocked"])
        self.assertIn("concentration:rates_0",info["quote_violation_detail"])
        np.testing.assert_array_equal(current,end)

    def test_minimum_holds_all_profiles_and_exception(self):
        for profile, minimum in zip(PROFILES, [3,2,1]):
            env = fixture(profile=profile)
            context = env.initial_context()
            context.age = minimum-1
            self.assertEqual(env.allowed_action(0, 3, context), 0)
            env.state.iloc[0, env.state.columns.get_loc("state_liquidity_shock")] = 1.
            self.assertEqual(env.allowed_action(0, 3, context), 3)
            with self.assertRaises(ValueError):
                env.allowed_action(0, -1, context)

    def test_teacher_no_rollout_mutation(self):
        env = fixture()
        current, context = env.initial_weights(0), env.initial_context()
        before = current.copy()
        action = env.teacher_action(0,current,5,context)
        self.assertIn(action, range(4))
        np.testing.assert_array_equal(before,current)
        self.assertEqual(context.nav,1.)
        self.assertIsNone(context.formal_weights)

    def test_nested_split_and_teacher_purge(self):
        env = fixture(800)
        split = split_history(env, 2024)
        self.assertIsNotNone(split)
        train, selection, gate = split
        self.assertGreaterEqual(len(train),252)
        self.assertGreaterEqual(selection[0]-train[-1]-1,20)
        self.assertGreaterEqual(gate[0]-selection[-1]-1,20)
        self.assertLess(env.dates[gate[-1]+1],pd.Timestamp("2024-01-01"))
        self.assertIsNone(split_history(env,2021))

    def test_actual_ppo_smoke(self):
        env = fixture()
        env.cfg["controller"].update(max_epochs=1, updates_per_epoch=1, teacher_pretrain_epochs=1)
        torch.set_num_threads(1)
        split = np.arange(32), np.arange(38,49), np.arange(55,66)
        record, logs = fit_seed(env,2024,17,1e-4,.004,split)
        self.assertEqual(len(logs),1)
        self.assertTrue(np.isfinite(logs[0]["loss"]))
        self.assertGreater(record["teacher_count"],0)
        self.assertTrue(all(r["label_end_date"]<=r["train_last_execution"] for r in record["teacher_audit"]))
        model = restore_model(record,32)
        self.assertEqual(len(simulate(env,split[1],model,record["mean"],record["std"])),len(split[1]))

    def test_bootstrap_pairing_and_standard_sharpe(self):
        values = np.array([.001,-.0005,.002,-.001]*25)
        result = bootstrap_comparison(values,values,20,100)
        self.assertEqual(result["return_positive_probability"],0.)
        self.assertEqual(result["sharpe_positive_probability"],0.)
        self.assertAlmostEqual(statistics(values)["sharpe"],values.mean()/values.std(ddof=1)*np.sqrt(252))

    def test_content_hash_object_and_target_identity(self):
        frame = pd.DataFrame({"a":[[1,2]],"b":[{"x":1}]})
        self.assertEqual(frame_hash(frame),frame_hash(copy.deepcopy(frame)))
        env = fixture()
        self.assertEqual(frame_hash(env.target),frame_hash(env.target.copy()))
        mutated = env.target.copy()
        mutated.iloc[0,0] += .0001
        self.assertNotEqual(frame_hash(env.target),frame_hash(mutated))


if __name__ == "__main__":
    unittest.main(verbosity=2)

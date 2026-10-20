"""Causal, explicitly unpromoted counterfactuals for the frozen v45 models."""
from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from src import v45_portfolio as engine

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "output/experiments/v46_shadow"


def shadow_predictions(frame, shrinkage=.25, monthly_cap=.01):
    """Use past-V selected neural residuals, NOT fabricate passed formal gates.

    Missing horizons contribute zero with their original weights; never reweight
    surviving horizons. Thin individual entities remain excluded from research.
    """
    required = {"as_of_date", "train_end_date", "gate_as_of_date", "is_oof",
                "selected_candidate", "model", "seed", "task", "entity",
                "horizon_days", "raw_increment", "prediction", "baseline_prediction",
                "entity_training_dates", "gate_reason"}
    if required - set(frame):
        raise ValueError(f"Missing forecast evidence: {sorted(required-set(frame))}")
    if not 0 < shrinkage <= 1 or not 0 < monthly_cap <= .01:
        raise ValueError("Shadow view bounds exceed the registered policy")
    work = frame.loc[frame.seed.eq(-1) & frame.model.isin(["F1", "F2", "F3"])
                     & frame.model.eq(frame.selected_candidate)].copy()
    keys = ["as_of_date", "task", "entity", "horizon_days"]
    if work.duplicated(keys).any():
        raise ValueError("Duplicate selected ensemble forecast")
    for column in ["as_of_date", "train_end_date", "gate_as_of_date"]:
        work[column] = pd.to_datetime(work[column], errors="raise")
    causal = (work.train_end_date.lt(work.as_of_date)
              & work.gate_as_of_date.lt(work.as_of_date) & work.is_oof.eq(True))
    if not causal.all():
        raise ValueError("Selected forecast violates historical issuance contract")
    if not work.horizon_days.isin(engine.HORIZONS).all() or not work.task.isin(["asset", "sleeve"]).all():
        raise ValueError("Unknown task/horizon")
    finite = np.isfinite(work[["raw_increment", "prediction", "baseline_prediction"]].to_numpy(float)).all(1)
    available = work.gate_reason.ne("missing_entity_forecast")
    eligible = finite & available & work.entity_training_dates.ge(300)
    rejected = {"nonfinite_rows": int((~finite).sum()),
                "thin_entity_rows": int(work.entity_training_dates.lt(300).sum()),
                "currently_unavailable_rows": int((~available).sum())}
    work = work.loc[eligible].copy()
    if not np.allclose(work.raw_increment, work.prediction-work.baseline_prediction, atol=1e-10):
        raise ValueError("Residual does not reconcile to model minus F0")
    work["increment_21"] = (work.raw_increment * work.horizon_days.map(engine.HORIZONS)
                             * 21 / work.horizon_days * shrinkage)
    result = work.groupby(keys[:3], as_index=False).agg(
        increment_21=("increment_21", "sum"), active_components=("horizon_days", "size"))
    result["increment_21"] = result.increment_21.clip(-monthly_cap, monthly_cap)
    status = {"status": "UNPROMOTED_causal_neural_shadow", "formal_forecast_gate_bypassed_for_research_only": True,
              "active_rows": len(work), "input_rows": len(frame), "output_rows": len(result),
              "shrinkage": shrinkage, "monthly_cap": monthly_cap, "minimum_entity_dates": 300,
              "horizon_weights": engine.HORIZONS, "missing_horizons": "zero_not_renormalized",
              "selected_models": sorted(work.model.unique().tolist()), **rejected}
    return result, status


class RelativeExecutionEnvironment(engine.ExecutionEnvironment):
    """Same books/constraints/accounting; only replace asymmetric active cost."""

    def step(self, index, action, current, context):
        _, next_index, weights, new_context, info = super().step(index, action, current, context)
        # formal_weights are the benchmark's actual *end* book. Reverse its
        # already-known next-session mark to recover its executed weights.
        next_returns = np.nan_to_num(self._returns[index+1], nan=0.)
        benchmark_executed = new_context.formal_weights / (1 + next_returns)
        benchmark_executed /= benchmark_executed.sum()
        benchmark_active = float(np.abs(benchmark_executed-self._anchor[index]).sum())
        coefficient = self.cfg["controller"]["active_weight_penalty"]
        relative_penalty = 100 * coefficient * max(0., info["active_l1"]-benchmark_active)
        info["legacy_reward"] = info["reward"]
        info["legacy_reward_active_penalty"] = info["reward_active_penalty"]
        info["mechanical_active_l1"] = benchmark_active
        info["reward_active_penalty"] = relative_penalty
        info["reward"] = (info["reward_return_edge"] - info["reward_downside_penalty"]
                          - info["reward_volatility_penalty"] - info["reward_drawdown_penalty"]
                          - relative_penalty)
        if not np.isfinite(info["reward"]):
            raise ValueError("Nonfinite relative execution reward")
        return info["reward"], next_index, weights, new_context, info


def experiment_config():
    cfg = copy.deepcopy(engine.config())
    cfg["controller"]["learning_rates"] = [1e-4]
    cfg["controller"]["entropy_weights"] = [.004]
    cfg["research_shadow"] = {
        "version": "v46_shadow_1", "formal_promotion": False,
        "source_code_hash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "view_shrinkage": .25, "monthly_cap": .01,
        "reward": "relative_positive_active_deviation_vs_same_target_mechanical",
        "selection": "past_V_only_fixed_lr_entropy_three_seeds",
        "replay": "all_selected_seeds_in_shadow_even_if_G_rejects; formal_G_unchanged",
        "data_end": "2026-07-15", "untouched_holdout": False}
    return cfg


def stage_inputs(cfg, root=EXPERIMENT):
    """Copy only immutable dependencies, keeping every v45 result untouched."""
    root = Path(root)
    files = {"src/v45_solver.py", "src/portfolio_accounting.py", "src/hierarchical_asset_portfolio_v7.py",
             "src/integrated_daily_view_ppo.py", "config/v1_strategy_config.yaml"}
    files.update(cfg["v45"].values())
    files.update([cfg["outputs"]["dataset_npz"], cfg["outputs"]["metadata"],
                  cfg["portfolio"]["integrated_base_config"], cfg["portfolio"]["frozen_state_cards"],
                  cfg["portfolio"]["frozen_rule_views"]])
    for profile in engine.PROFILES:
        files.add(cfg["portfolio"]["formal_anchor_pattern"].format(profile=profile))
        files.add(cfg["portfolio"]["risk_budget_pattern"].format(profile=profile))
    hashes = {}
    for name in sorted(files):
        source, target = ROOT / name, root / name
        if not source.exists() and name == cfg["v45"]["forecast_gate"]:
            continue
        if not source.is_file():
            raise FileNotFoundError(source)
        digest = engine.sha_file(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or engine.sha_file(target) != digest:
            shutil.copy2(source, target)
        hashes[name] = digest
    (root / "output/reports").mkdir(parents=True, exist_ok=True)
    engine.write_json(root / "output/reports/v46_input_manifest.json", hashes)
    return hashes

"""Run isolated v46 shadow counterfactuals; never replace a formal product."""
from __future__ import annotations

import argparse
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch

from src import v45_portfolio as engine
from src.v46_shadow_research import (EXPERIMENT, RelativeExecutionEnvironment,
                                     experiment_config, shadow_predictions, stage_inputs)


def run_profile(profile, train, resume, experiment_root=None, history_supplement=None):
    torch.set_num_threads(1)
    root = Path(experiment_root) if experiment_root else EXPERIMENT
    cfg = experiment_config()
    # Validate the unchanged v45 risk/accounting contract independently of the
    # explicitly narrower v46 learning grid. Do not edit the old validator.
    contract = copy.deepcopy(cfg)
    contract["controller"]["learning_rates"] = [1e-4, 3e-4]
    contract["controller"]["entropy_weights"] = [.004, .01]
    engine.validate_effective_config(contract)
    if cfg["controller"]["learning_rates"] != [1e-4] or cfg["controller"]["entropy_weights"] != [.004]:
        raise ValueError("Unregistered v46 controller candidates")
    forecasts, status = shadow_predictions(pd.read_parquet(ROOT / cfg["v45"]["predictions"]))
    if history_supplement:
        from src.v47_history_supplement import merge_supplement
        forecasts, supplement_status = merge_supplement(forecasts, Path(history_supplement))
        cfg["history_supplement"] = supplement_status
        status["history_supplement"] = supplement_status
    status["research_code_hash"] = cfg["research_shadow"]["source_code_hash"]
    returns, _, quotes, _ = engine.frozen_accounting(cfg, root=root)
    manifest = engine.prepare_profile(profile, cfg, returns, quotes, forecasts, status,
                                      resume=resume, root=root)
    engine.ExecutionEnvironment = RelativeExecutionEnvironment
    summary, folds = [], []
    for layer in ["A0", "A1", "A2"]:
        env = engine.load_environment(profile, layer, cfg, returns, quotes, root=root)
        daily, holdings = engine.replay(env)
        daily.to_csv(engine.output(profile, layer, "daily", root), index=False)
        holdings.to_parquet(engine.output(profile, layer, "holdings", root), index=False)
        summary.append(measure(profile, layer, daily))
    if train:
        env = engine.load_environment(profile, "A3", cfg, returns, quotes, root=root)
        research_models, formal_models = {}, {}
        for year in cfg["controller"]["years"]:
            approved, fold = engine.train_fold(env, year, resume=resume, root=root)
            formal_models[year] = approved
            records = []
            if "selected_lr" in fold:
                folder = root / "data_processed/v45_controller_models" / profile / str(year)
                for seed in cfg["controller"]["seeds"]:
                    path = folder / f"lr{fold['selected_lr']}_entropy{fold['selected_entropy']}_seed{seed}.pt"
                    record = torch.load(path, map_location="cpu", weights_only=False)
                    if record["signature"] != engine.model_signature(env, year, fold["selected_lr"], fold["selected_entropy"]):
                        raise ValueError("Research checkpoint signature changed")
                    records.append((record, engine.restore_model(record, cfg["controller"]["hidden_dim"])))
            research_models[year] = records
            folds.append(fold)
        for layer, models in [("A3_shadow", research_models), ("A3_execution_gated", formal_models)]:
            daily, holdings = engine.replay(env, models)
            daily["formal_promotion"] = False
            daily["research_policy_used"] = daily.model_used
            daily.to_csv(engine.output(profile, layer, "daily", root), index=False)
            holdings.to_parquet(engine.output(profile, layer, "holdings", root), index=False)
            summary.append(measure(profile, layer, daily))
    return {"profile": profile, "metrics": summary, "folds": folds,
            "forecast": status, "active_target_days": manifest["active_days"]}


def measure(profile, layer, daily):
    sub = daily.loc[pd.to_datetime(daily.date).ge("2024-01-01")]
    stats = engine.statistics(sub.strategy_return.to_numpy())
    return {"profile": profile, "layer": layer, "start": str(sub.date.iloc[0]),
            "end": str(sub.date.iloc[-1]), "days": len(sub), **stats,
            "annual_cost_bp": float(sub.cost.sum()*252/len(sub)*1e4),
            "model_used_days": int(sub.model_used.sum()),
            "forecast_days": int(sub.active_components.gt(0).sum()),
            "max_end_sleeve_residual": float(sub.end_sleeve_bounds_residual.max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-ppo", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=3, choices=[1, 2, 3])
    args = parser.parse_args()
    cfg = experiment_config()
    stage_inputs(cfg)
    engine.write_json(EXPERIMENT / "output/reports/v46_registered_config.json", cfg)
    status_path = EXPERIMENT / "output/reports/v46_progress.json"
    completed = []
    engine.write_json(status_path, {"status": "running", "pid": os.getpid(), "completed_profiles": [],
                                   "train_ppo": args.train_ppo, "formal_promotion": False})
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(run_profile, p, args.train_ppo, args.resume): p for p in engine.PROFILES}
            for future in as_completed(futures):
                result = future.result()
                completed.append(result)
                engine.write_json(status_path, {"status": "running", "pid": os.getpid(),
                                               "completed_profiles": [r["profile"] for r in completed]})
                print(f"[v46] completed {result['profile']}", flush=True)
    except Exception:
        engine.write_json(status_path, {"status": "failed", "error": traceback.format_exc(),
                                       "completed_profiles": [r["profile"] for r in completed]})
        raise
    table = pd.DataFrame([row for result in completed for row in result["metrics"]])
    table.to_csv(EXPERIMENT / "output/reports/v46_shadow_comparison.csv", index=False)
    engine.write_json(EXPERIMENT / "output/reports/v46_shadow_results.json", completed)
    engine.write_json(status_path, {"status": "completed", "train_ppo": args.train_ppo,
                                   "completed_profiles": [r["profile"] for r in completed],
                                   "formal_promotion": False})
    print(table.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

"""Recover honest 2022 neural-history coverage, then retrain matched PPO."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
    os.environ.setdefault(key, "1")

import pandas as pd

from src import v45_portfolio as engine
from src.v46_shadow_research import experiment_config, stage_inputs
from src.v47_history_supplement import EXPERIMENT, build_supplement
from scripts.run_v46_shadow_research import run_profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, choices=[1, 2, 3], default=3)
    args = parser.parse_args()
    reports = EXPERIMENT / "output/reports"
    reports.mkdir(parents=True, exist_ok=True)
    progress = reports / "v47_progress.json"
    engine.write_json(progress, {"status": "running", "stage": "auxiliary_heads", "pid": os.getpid(),
                                 "train_ppo": not args.prepare_only})
    completed = []
    try:
        path, auxiliary_report = build_supplement(resume=args.resume)
        if args.prepare_only:
            engine.write_json(progress, {"status": "prepared", "stage": "auxiliary_heads",
                                         "train_ppo": False, "auxiliary": auxiliary_report})
            print(json.dumps(auxiliary_report, indent=2), flush=True)
            return
        cfg = experiment_config()
        stage_inputs(cfg, root=EXPERIMENT)
        engine.write_json(reports / "v47_registered_config.json", {"base": cfg,
            "change": "2022 direct relative-return auxiliary views only; subsequent original views preserved",
            "formal_promotion": False, "auxiliary": auxiliary_report,
            "ppo_retrained": True, "reward_changed_vs_v46": False,
            "new_raw_market_data": False})
        engine.write_json(progress, {"status": "running", "stage": "targets_and_ppo", "pid": os.getpid(),
                                     "train_ppo": True, "completed_profiles": []})
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(run_profile, p, True, args.resume, str(EXPERIMENT), str(path)): p
                       for p in engine.PROFILES}
            for future in as_completed(futures):
                completed.append(future.result())
                engine.write_json(progress, {"status": "running", "stage": "targets_and_ppo", "pid": os.getpid(),
                                             "train_ppo": True, "completed_profiles": [r["profile"] for r in completed]})
                print(f"[v47] completed {completed[-1]['profile']}", flush=True)
        table = pd.DataFrame([m for r in completed for m in r["metrics"]])
        table.to_csv(reports / "v47_shadow_comparison.csv", index=False)
        engine.write_json(reports / "v47_shadow_results.json", completed)
        engine.write_json(progress, {"status": "completed", "train_ppo": True,
                                     "completed_profiles": [r["profile"] for r in completed], "formal_promotion": False})
        print(table.to_string(index=False), flush=True)
    except BaseException:
        engine.write_json(progress, {"status": "failed", "error": traceback.format_exc(),
                                     "completed_profiles": [r["profile"] for r in completed]})
        raise


if __name__ == "__main__":
    main()

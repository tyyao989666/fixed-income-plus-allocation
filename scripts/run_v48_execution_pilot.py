"""Pre-test PPO ablation: balanced/2024, three seeds, unchanged deployment gate."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
    os.environ.setdefault(key, "1")

import numpy as np
import torch
from src import v45_portfolio as engine
from src import v48_execution_learning as learning

INPUT = ROOT / "output/experiments/v47_history_supplement"
OUTPUT = ROOT / "output/experiments/v48_execution_pilot"


def run(variant, resume):
    torch.set_num_threads(1)
    cfg = copy.deepcopy(json.loads((INPUT / "output/reports/v47_registered_config.json").read_text(encoding="utf-8"))["base"])
    cfg["v48"] = dict(variant=variant, source_hash=learning.source_hash(),
                      scope="balanced_2024_preissue_T_V_G_only_no_test_replay",
                      formal_promotion=False)
    profile, year = "balanced", 2024
    root = OUTPUT / variant
    (root / "output/reports").mkdir(parents=True, exist_ok=True)
    returns, _, quotes, _ = engine.frozen_accounting(cfg, root=INPUT)
    engine.ExecutionEnvironment = learning.ScaledExecutionEnvironment
    engine.fit_seed = learning.fit_seed
    env = engine.load_environment(profile, "A3", cfg, returns, quotes, root=INPUT)
    split = engine.split_history(env, year)
    if split is None:
        raise ValueError("Pilot training history unavailable")
    source_files = [INPUT / f"data_processed/v45_{profile}_{layer}_{kind}.parquet"
                    for layer in ["A0", "A3"] for kind in ["targets", "diagnostics"]]
    source_files.append(INPUT / f"data_processed/v45_{profile}_states.parquet")
    cfg["v48"]["input_hashes"] = {str(p.relative_to(ROOT)): engine.sha_file(p) for p in source_files}
    engine.write_json(root / "output/reports/registered_config.json", cfg)
    approved, fold = engine.train_fold(env, year, resume=resume, root=root)
    mean, std = learning.training_normalizer(env, split[0])
    observations, probabilities, actions, audit, temperature = learning.teacher_data(env, split[0], mean, std, variant)
    if any(a["label_end_date"] > a["train_last_execution"] for a in audit):
        raise ValueError("Teacher label exceeded T")
    # G is read only after V checkpoint selection. Never run 2024+ test here.
    mechanical = engine.simulate(env, split[2])
    folder = root / "data_processed/v45_controller_models" / profile / str(year)
    diagnostics = []
    for seed in cfg["controller"]["seeds"]:
        path = folder / f"lr{fold['selected_lr']}_entropy{fold['selected_entropy']}_seed{seed}.pt"
        record = torch.load(path, weights_only=False, map_location="cpu")
        model = engine.restore_model(record, cfg["controller"]["hidden_dim"])
        rows = engine.simulate(env, split[2], model, record["mean"], record["std"])
        diagnostics.append(dict(seed=seed, raw_hold_fraction=float(np.mean([r["raw_policy_action"] == 0 for r in rows])),
                                executed_hold_fraction=float(np.mean([r["action"] == 0 for r in rows])),
                                minimum_hold_suppressed_days=sum(r["minimum_hold_suppressed"] for r in rows),
                                gate_full_segment_mean_net_edge=float(np.mean([p["strategy_return"]-b["strategy_return"] for p, b in zip(rows, mechanical)]))))
    result = dict(variant=variant, fold=fold, qualified_models=len(approved), seed_diagnostics=diagnostics,
                  teacher_count=len(audit), teacher_temperature=temperature,
                  teacher_mean_probabilities=probabilities.mean(0).tolist(),
                  teacher_class_counts=np.bincount(actions, minlength=4).tolist(),
                  observation_dimension=len(mean), test_replayed=False, formal_promotion=False,
                  next_step="full_research_eligible_not_live_promotion" if fold["deployment_gate_passed"] else "stop_expansion_gate_failed")
    engine.write_json(root / "output/reports/pilot_result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    reports = OUTPUT / "output/reports"
    reports.mkdir(parents=True, exist_ok=True)
    progress = reports / "progress.json"
    engine.write_json(progress, dict(status="running", pid=os.getpid(), test_replayed=False,
                                    variants=["scaled_hard", "scaled_value"], completed=[]))
    results = []
    try:
        with ProcessPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(run, v, args.resume) for v in ["scaled_hard", "scaled_value"]]
            for future in as_completed(futures):
                results.append(future.result())
                engine.write_json(progress, dict(status="running", completed=[r["variant"] for r in results], test_replayed=False))
        engine.write_json(reports / "pilot_results.json", results)
        engine.write_json(progress, dict(status="completed", completed=[r["variant"] for r in results], test_replayed=False,
                                        formal_promotion=False))
        print(json.dumps(results, indent=2), flush=True)
    except BaseException:
        engine.write_json(progress, dict(status="failed", error=traceback.format_exc()))
        raise


if __name__ == "__main__":
    main()

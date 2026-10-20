"""Bounded completion audit; never starts or promotes another experiment."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import torch
from src import v45_forecast as engine
from src.v49_masked_forecast import training_implementation
from src.v61_state_momentum import StateRanker, nested_split
from scripts.run_v45_research_pipeline import process_alive


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    out = ROOT / "output/reports/state_momentum_v61"
    started = time.monotonic()
    while True:
        progress = json.loads((out / "progress.json").read_text())
        if progress["status"] == "completed_validation_only":
            break
        if not args.wait or progress["status"] == "failed" or not process_alive(progress["pid"]) or time.monotonic()-started > 10800:
            (out / "artifact_audit.json").write_text(json.dumps(dict(status="not_ready", training_status=progress["status"], auto_restart=False)), encoding="utf-8")
            return
        time.sleep(10)
    cfg = json.loads((ROOT / "config/v61_state_momentum.json").read_text())
    manifest = json.loads((out / "manifest.json").read_text())
    checks = []
    for name,digest in manifest.items():
        p = ROOT / ("data_processed/corporate_action_v58/dataset.npz" if name == "dataset_sha256" else name)
        checks.append(dict(check="source_hash", source=name, passed=hashlib.sha256(p.read_bytes()).hexdigest() == digest))
    with np.load(ROOT / "data_processed/corporate_action_v58/dataset.npz") as z:
        a = {k:z[k].copy() for k in z.files}
    with np.load(ROOT / "data_processed/state_momentum_v61/signals.npz") as z:
        signal = {k:z[k] for k in z.files}
    checks.append(dict(check="value_boundary_not_fabricated", passed=bool(np.isnan(signal["value_boundary"]).all())))
    extra = np.stack([2*signal["cohort"]-1, np.tanh(signal["extension"]/3), signal["value_boundary"]], -1)
    a["local_values"] = np.concatenate([a["local_values"], np.nan_to_num(extra)], -1).astype(np.float32)
    a["local_masks"] = np.concatenate([a["local_masks"], np.isfinite(extra)], -1)
    old = engine.configuration(ROOT / "config/v45_research.yaml")
    old["seed_workers"] = 1
    gradient_rows = []
    with training_implementation():
        for year in cfg["years"]:
            split = nested_split(a, engine.shared_split(a, year, old))
            end = engine.end_times(a)
            checks.append(dict(check="T_C_V_label_maturity", year=year, passed=bool(
                (end[split["train"]].max(1) < a["dates"][split["calibration"][0]]).all()
                and (end[split["calibration"]].max(1) < a["dates"][split["validation"][0]]).all())))
            for seed in cfg["seeds"]:
                folder = ROOT / f"output/models/state_momentum_v61/{year}/{seed}"
                current = torch.load(folder / "current.pt", map_location="cpu", weights_only=False)
                selected = torch.load(folder / "selected.pt", map_location="cpu", weights_only=False)
                summary = json.loads((folder / "summary.json").read_text())
                checks.append(dict(check="identity", year=year, seed=seed, passed=current["identity"] == selected["identity"] == summary["identity"] and current["identity"]["manifest"] == manifest))
                model = StateRanker(engine.ForecastNetwork(a, old), old["d_model"])
                ids = current["optimizer"]["param_groups"][1]["params"]
                names = [n for n,p in model.network.core.named_parameters()]
                if len(ids) != len(names):
                    raise ValueError("Optimizer parameter mapping changed")
                grads = {}
                for name,i in zip(names, ids):
                    state = current["optimizer"]["state"].get(i, {})
                    v = state.get("exp_avg", torch.zeros(1))
                    grads[name] = bool(torch.count_nonzero(v).item() > 0)
                first = any(v for n,v in grads.items() if n.startswith("encoder.layers.0."))
                last = any(v for n,v in grads.items() if n.startswith("encoder.layers.1."))
                projection = any(v for n,v in grads.items() if n.startswith("local_projection."))
                checks.append(dict(check="full_encoder_supervision", year=year, seed=seed, passed=first and last and projection))
                checks.append(dict(check="calibrator_nonnegative_finite", year=year, seed=seed,
                    passed=all(np.isfinite(v).all() and (v >= 0).all() for v in selected["calibrator"].values())))
                with np.load(folder / "validation_predictions.npz") as z:
                    checks.append(dict(check="finite_validation_predictions", year=year, seed=seed,
                        passed=all(np.isfinite(z[k]).all() for k in z.files)))
                gradient_rows.append(dict(year=year, seed=seed, first_layer_gradient=first, last_layer_gradient=last,
                    local_projection_gradient=projection, selected_epoch=summary["selected_epoch"], trained_epochs=summary["trained_epochs"]))
    pd.DataFrame(checks).to_csv(out / "artifact_checks.csv", index=False)
    pd.DataFrame(gradient_rows).to_csv(out / "full_encoder_gradient_audit.csv", index=False)
    result = dict(status="passed" if all(c["passed"] for c in checks) else "failed", checks=len(checks),
        full_encoder_tasks_verified=sum(r["first_layer_gradient"] and r["last_layer_gradient"] and r["local_projection_gradient"] for r in gradient_rows),
        value_boundary_enabled=False, formal_promotion=False, ppo_training_started=False)
    (out / "artifact_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)
    if result["status"] != "passed":
        raise ValueError("Ranker audit failed")


if __name__ == "__main__":
    main()

"""Post-experiment T/V diagnostics only; no fitting, search or trading approval."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import torch
from src import v45_forecast as engine
from src.v59_raw_mean import fit_shared_scale


def feature_similarity(x, y):
    """Centered covariance alignment, invariant to feature rotations and scale."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2 or x.shape[0] != y.shape[0]:
        return None
    x, y = x-x.mean(0), y-y.mean(0)
    denom = np.linalg.norm(x.T@x)*np.linalg.norm(y.T@y)
    return float(np.linalg.norm(x.T@y)**2/denom) if denom > 0 else None


def main():
    out = ROOT / "output/reports/raw_mean_v59"
    settings = json.loads((ROOT / "config/v59_raw_mean.json").read_text())
    manifest = json.loads((out / "manifest.json").read_text())
    checks = []
    for name, digest in manifest.items():
        p = ROOT / ("data_processed/corporate_action_v58/dataset.npz" if name == "dataset_sha256" else name)
        checks.append(dict(check="frozen_source_hash", source=name,
                           passed=p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest() == digest))
    with np.load(ROOT / "data_processed/corporate_action_v58/dataset.npz") as z:
        a = {k: z[k].copy() for k in z.files}
    meta = json.loads((ROOT / "output/reports/corporate_action_v58/metadata.json").read_text())
    cfg = engine.configuration(ROOT / "config/v45_research.yaml")
    data_rows, embedding_rows, dispersion_rows = [], [], []
    groups = dict(asset=a["module_index"], sleeve=np.zeros(5, int))
    for year in settings["years"]:
        split = engine.shared_split(a, year, cfg)
        for part in ["train", "validation"]:
            ix = split[part]
            for task in groups:
                for h, horizon in enumerate([5, 20, 60]):
                    m = a[task+"_target_mask"][ix, h]
                    data_rows.append(dict(year=year, part=part, task=task, horizon_days=horizon,
                        dates=len(ix), dates_with_any_label=int(m.any(1).sum()), valid_cells=int(m.sum()),
                        approximate_nonoverlap_windows=int(m.any(1).sum())//horizon,
                        note="rough count, not statistical effective sample size; shared markets reduce independence"))
        caches = {}
        for seed in settings["seeds"]:
            cache = ROOT / f"output/models/raw_mean_v59/{year}/{seed}"
            caches[seed] = torch.load(cache / "historical_embeddings.pt", map_location="cpu", weights_only=False)["features"]
            for kind in settings["candidates"]:
                folder = cache / kind
                saved = torch.load(folder / "selected.pt", map_location="cpu", weights_only=False)
                summary = json.loads((folder / "summary.json").read_text())
                checks.append(dict(check="candidate_identity", year=year, seed=seed, candidate=kind,
                    passed=saved["identity"] == summary["identity"] and saved["identity"]["provenance"] == manifest))
                with np.load(folder / "validation_predictions.npz") as z:
                    for task in groups:
                        c, s, good = fit_shared_scale(a[task+"_raw_relative_target"][split["train"]],
                            a[task+"_target_mask"][split["train"]], groups[task])
                        checks.append(dict(check="scale_fit_T_only", year=year, seed=seed, candidate=kind, task=task,
                            passed=bool(np.array_equal(c, saved["centers"][task]) and np.array_equal(s, saved["scales"][task]))))
                        pred, q = z[task], z[task+"_quantiles"]
                        checks.append(dict(check="finite_ordered_outputs", year=year, seed=seed, candidate=kind, task=task,
                            passed=bool(np.isfinite(pred).all() and np.isfinite(q).all() and (q[..., 0] <= q[..., 1]).all() and (q[..., 1] <= q[..., 2]).all())))
                        vi = split["validation"]
                        truth, mask = a[task+"_raw_relative_target"][vi], a[task+"_target_mask"][vi].astype(bool)
                        names = meta["assets"] if task == "asset" else meta["modules"]
                        for h, horizon in enumerate([5, 20, 60]):
                            for entity, name in enumerate(names):
                                m = mask[:, h, entity] & np.isfinite(pred[:, h, entity])
                                y, p = truth[m, h, entity], pred[m, h, entity]
                                if len(y) < 10:
                                    continue
                                ys, ps = float(y.std()), float(p.std())
                                corr = float(pd.Series(p).rank().corr(pd.Series(y).rank())) if ys > 0 and ps > 0 else None
                                dispersion_rows.append(dict(year=year, seed=seed, candidate=kind, task=task,
                                    entity=name, horizon_days=horizon, observed_dates=len(y),
                                    forecast_std_over_target_std=ps/ys if ys > 0 else None,
                                    temporal_rank_ic=corr, constant_T_mean=float(c[h, entity]),
                                    forecast_mean=float(p.mean()), target_mean=float(y.mean()),
                                    note="historical_validation_diagnostic_not_tradeable_oos"))
        for part in ["train", "validation"]:
            for task in groups:
                ix = split[part]
                valid = a["local_masks"][ix].any(-1) if task == "asset" else np.ones((len(ix), 5), bool)
                for i, left in enumerate(settings["seeds"]):
                    for right in settings["seeds"][i+1:]:
                        x, y = caches[left][part][task], caches[right][part][task]
                        shared = valid & np.isfinite(x).all(-1) & np.isfinite(y).all(-1)
                        temporal = []
                        for values in [x, y]:
                            z = np.asarray(values, float).copy()
                            for entity in range(z.shape[1]):
                                m = shared[:, entity]
                                if m.any():
                                    z[:, entity] -= z[m, entity].mean(0)
                            temporal.append(z[shared])
                        embedding_rows.append(dict(year=year, part=part, task=task, seed_left=left, seed_right=right,
                            pooled_feature_similarity=feature_similarity(x[shared], y[shared]),
                            within_entity_temporal_similarity=feature_similarity(*temporal),
                            matched_context_rows=int(shared.sum()), similarity_is_not_predictive_skill=True))
    views = pd.read_parquet(out / "validation_research_mean_views.parquet")
    checks.extend([
        dict(check="research_views_not_authorized", passed=bool((~views.approved).all() and (~views.execution_certified).all() and (~views.point_in_time_eligible).all())),
        dict(check="selection_maturity_not_backdated", passed=bool((views.available_after >= views.model_selection_known_after).all() and (views.available_after > views.as_of).all())),
        dict(check="no_test_partition_exported", passed=bool(views.evaluation_partition.eq("historical_validation").all())),
        dict(check="38_assets_retained", passed=views.loc[views.task.eq("asset"), "entity"].nunique() == 38),
    ])
    for name, rows in [("sample_support_diagnostics", data_rows), ("embedding_seed_diagnostics", embedding_rows),
                       ("prediction_dispersion_diagnostics", dispersion_rows), ("completed_experiment_checks", checks)]:
        pd.DataFrame(rows).to_csv(out / (name+".csv"), index=False)
    control = pd.read_csv(out / "constant_control_diagnostics.csv")
    training = pd.read_csv(out / "training_summary.csv")
    dispersion = pd.DataFrame(dispersion_rows)
    stats = dict(audit_passed=all(r["passed"] for r in checks), check_count=len(checks),
        models_trained=12, candidate_passed=0, automatic_new_training_started=False,
        constant_control=control.groupby("candidate").agg(mean_improvement=("improvement_vs_T_group_mean", "mean"),
            positive_runs=("improvement_vs_T_group_mean", lambda x:int((x>0).sum()))).reset_index().to_dict("records"),
        median_forecast_to_target_std=dispersion.groupby("candidate").forecast_std_over_target_std.median().to_dict(),
        selected_epoch_min=int(training.best_epoch.min()), selected_epoch_max=int(training.best_epoch.max()),
        effective_sample_warning="overlapping horizons and shared asset exposures; raw row counts overstate independence")
    (out / "signal_diagnosis.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    if not stats["audit_passed"]:
        raise ValueError("Completed experiment artifact audit failed")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()

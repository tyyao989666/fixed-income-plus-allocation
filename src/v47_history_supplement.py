"""2022 causal auxiliary views: real direct labels, never invented residuals."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from src import v45_forecast as forecast
from src import v45_portfolio as portfolio
from src.v46_shadow_research import ROOT

EXPERIMENT = ROOT / "output/experiments/v47_history_supplement"
KEY = ["as_of_date", "task", "entity"]


def aggregate_auxiliary(frame):
    required = {*KEY, "horizon_days", "prediction", "training_dates", "available",
                "train_end", "training_label_end", "encoder_validation_end", "issue_cutoff"}
    if required-set(frame):
        raise ValueError(f"Missing auxiliary evidence: {required-set(frame)}")
    data = frame.copy()
    for column in ["as_of_date", "train_end", "training_label_end", "encoder_validation_end", "issue_cutoff"]:
        data[column] = pd.to_datetime(data[column], errors="raise")
    if not data.as_of_date.dt.year.eq(2022).all():
        raise ValueError("Supplement must not change forecasts after 2022")
    for column in ["train_end", "training_label_end", "encoder_validation_end", "issue_cutoff"]:
        if not data[column].lt(data.as_of_date).all():
            raise ValueError(f"Auxiliary future information: {column}")
    if data.duplicated(KEY+["horizon_days"]).any():
        raise ValueError("Duplicate auxiliary ensemble component")
    if not data.horizon_days.isin(portfolio.HORIZONS).all() or not data.task.isin(["asset", "sleeve"]).all():
        raise ValueError("Unknown auxiliary task/horizon")
    good = data.available.eq(True) & data.training_dates.ge(300) & np.isfinite(data.prediction)
    data = data.loc[good].copy()
    data["increment_21"] = data.prediction*data.horizon_days.map(portfolio.HORIZONS)*21/data.horizon_days*.25
    result = data.groupby(KEY, as_index=False).agg(increment_21=("increment_21", "sum"),
                                                active_components=("horizon_days", "size"))
    result["increment_21"] = result.increment_21.clip(-.01, .01)
    return result


def merge_supplement(base, path):
    data = pd.read_parquet(path)
    auxiliary = aggregate_auxiliary(data)
    if base.duplicated(KEY).any():
        raise ValueError("Duplicate original forecast")
    if len(base.merge(auxiliary[KEY], on=KEY)):
        raise ValueError("Cannot replace an existing eligible neural forecast")
    combined = pd.concat([base, auxiliary], ignore_index=True).sort_values(KEY).reset_index(drop=True)
    return combined, {"version": "v47_direct_history_auxiliary_1", "formal_promotion": False,
                      "path": str(path), "sha256": portfolio.sha_file(path),
                      "source_hash": portfolio.sha_file(Path(__file__)), "added_entity_days": len(auxiliary),
                      "added_calendar_days": int(auxiliary.as_of_date.nunique()),
                      "policy": "2022_only_direct_relative_return_not_F0_residual; original_300_min_unchanged",
                      "encoding": "reuse_historical_SSL_iTransformer; newly_fit_direct_Ridge_readout",
                      "test_2024_plus_forecasts_changed": False}


def build_supplement(root=EXPERIMENT, resume=False):
    root = Path(root)
    folder = root / "output/models/2022_auxiliary"
    folder.mkdir(parents=True, exist_ok=True)
    output = root / "data_processed/v47_auxiliary_2022.parquet"
    a = dict(np.load(ROOT / "data_processed/v45_dataset.npz", allow_pickle=False))
    metadata = json.loads((ROOT / "output/reports/v45_metadata.json").read_text(encoding="utf-8"))
    old_probe = torch.load(ROOT / "output/models/v45_forecast/2022/seed_17_F1.pt", map_location="cpu", weights_only=False)
    cfg = old_probe["cfg"]
    split = forecast.shared_split(a, 2022, cfg)
    if not split["valid"] or len(split["train"]) < 300:
        raise ValueError("No honest historical encoder split")
    signature = forecast.historical_signature(a, split, cfg, metadata)
    if old_probe["signature"] != signature:
        raise ValueError("Original historical encoder inputs/config/source signature changed")
    own_signature = hashlib.sha256((signature+portfolio.sha_file(Path(__file__))).encode()).hexdigest()
    report_path = folder / "report.json"
    if resume and output.exists() and report_path.exists():
        saved = json.loads(report_path.read_text())
        if saved["signature"] == own_signature and portfolio.sha_file(output) == saved["prediction_sha256"]:
            aggregate_auxiliary(pd.read_parquet(output))
            return output, saved
    clipped, clipping = forecast.clip_fold_features(a, split["train"], cfg)
    y, masks = forecast.targets_at(a, split["train"])
    indices = np.r_[split["train"], split["test"]]
    predictions, models = [], []
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        for seed in [17, 29, 47]:
            checkpoint = ROOT / f"output/models/v45_forecast/2022/seed_{seed}_pretraining.pt"
            old = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if old["signature"] != signature or not old["complete"] or not old["summary"]["selection_available"]:
                raise ValueError("Historical encoder identity or selection unavailable")
            network = forecast.ForecastNetwork(clipped, cfg)
            network.load_state_dict(old["best_state"])
            encoded = forecast.infer(network, clipped, indices, cfg, torch.device("cpu"), True)
            representations = {k: v[:len(split["train"])] for k, v in encoded.items()}
            head = forecast.GroupRidge(a["module_index"], cfg["ridge_alpha_per_training_cell"],
                                      minimum_dates=300).fit(representations, y, masks)
            predicted = head.predict({k: v[len(split["train"]):] for k, v in encoded.items()})
            predictions.append(predicted)
            models.append({"seed": seed, "encoder_sha256": portfolio.sha_file(checkpoint)})
            torch.save({"signature": own_signature, "head": head, "encoder_signature": signature,
                        "encoder_sha256": portfolio.sha_file(checkpoint), "seed": seed,
                        "train_indices": split["train"], "feature_clipping": clipping,
                        "label_type": "direct_relative_return_not_residual"}, folder / f"seed_{seed}_direct_head.pt")
            print(f"[v47/auxiliary] fitted direct head seed={seed} T={len(split['train'])}", flush=True)
    ensemble = {k: np.mean([p[k] for p in predictions], axis=0) for k in y}
    dates = pd.to_datetime(a["dates"])
    all_ends = forecast.end_times(a)
    rows = []
    names = {"asset": forecast.entity_names(metadata, len(a["module_index"])), "sleeve": portfolio.RISK_SLEEVES}
    for task in ["asset", "sleeve"]:
        counts = masks[task].sum(axis=0)
        for entity, name in enumerate(names[task]):
            members = np.array([entity]) if task == "asset" else np.flatnonzero(a["module_index"] == entity)
            available = (a["local_masks"][split["test"]][:, members].any(2)
                         & a["return_masks"][split["test"]][:, members].astype(bool)).any(1)
            for h, horizon in enumerate([5, 20, 60]):
                used = split["train"][masks[task][:, h, entity]]
                latest_label = pd.Timestamp(all_ends[used, h].max()) if len(used) else pd.NaT
                rows.append(pd.DataFrame({"as_of_date": dates[split["test"]], "task": task, "entity": name,
                    "horizon_days": horizon, "prediction": ensemble[task][:, h, entity],
                    "training_dates": int(counts[h, entity]), "available": available,
                    "train_end": dates[split["train"][-1]], "training_label_end": latest_label,
                    "encoder_validation_end": dates[split["validation"][-1]],
                    "issue_cutoff": dates[split["cutoff_index"]],
                    "source": "historical_iTransformer_direct_auxiliary; NOT_F0_residual"}))
    frame = pd.concat(rows, ignore_index=True)
    # Untrained entities have no issued auxiliary forecast, rather than fake dates.
    frame = frame.loc[frame.training_dates.ge(300)].copy()
    aggregated = aggregate_auxiliary(frame)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)
    report = {"signature": own_signature, "historical_encoder_signature": signature,
              "train_dates": len(split["train"]), "test_dates": len(split["test"]),
              "training_start": str(dates[split["train"][0]]), "training_end": str(dates[split["train"][-1]]),
              "maximum_training_label_end": str(pd.to_datetime(frame.training_label_end).max()),
              "models": models, "raw_rows": len(frame), "issued_entity_days": len(aggregated),
              "issued_calendar_days": int(aggregated.as_of_date.nunique()),
              "prediction_sha256": portfolio.sha_file(output), "formal_promotion": False,
              "new_raw_market_observations": 0, "new_independent_market_days": 0,
              "purpose": "honest_alternative_learning_target_recovers_existing_2022_history"}
    portfolio.write_json(report_path, report)
    return output, report

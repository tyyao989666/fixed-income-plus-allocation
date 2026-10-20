"""Inspect real frozen inputs/checkpoints; do not silently retrain or promote."""
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import torch
from src import v45_forecast as old
from src.v49_masked_forecast import MaskSafeForecastNetwork, validate_dataset, safe_batch_at, source_hash


def main():
    torch.set_num_threads(2)
    reports = ROOT / "output/reports/v49_project_review"
    reports.mkdir(parents=True, exist_ok=True)
    with np.load(ROOT / "data_processed/v45_dataset.npz", allow_pickle=False) as z:
        arrays = {k:z[k].copy() for k in z.files}
    metadata = json.loads((ROOT / "output/reports/v45_metadata.json").read_text(encoding="utf-8"))
    validate_dataset(arrays)
    checkpoint = ROOT / "output/models/v45_forecast/2026/seed_17_F1.pt"
    readout = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = readout["cfg"]
    splits, checks, epochs = [], [], []
    for year in cfg["years"]:
        split = old.shared_split(arrays, year, cfg)
        counts = {name:len(split[name]) for name in ["train", "validation", "gate", "test"]}
        row = dict(year=year, **counts)
        for part in ["train", "validation", "gate"]:
            ix = split[part]
            row[part+"_signal_end"] = str(pd.Timestamp(arrays["dates"][ix[-1]]).date()) if len(ix) else None
            row[part+"_latest_label_end"] = str(pd.Timestamp(old.end_times(arrays)[ix].max()).date()) if len(ix) else None
        splits.append(row)
        signature = old.historical_signature(arrays, split, cfg, metadata)
        first = ROOT / f"output/models/v45_forecast/{year}/seed_17_F1.pt"
        saved = torch.load(first, map_location="cpu", weights_only=False)
        checks.append(dict(check=f"{year}:original_historical_signature", passed=signature == saved["signature"]))
        for seed in cfg["seeds"]:
            for stage in ["pretraining", "F2", "F3"]:
                path = ROOT / f"output/models/v45_forecast/{year}/seed_{seed}_{stage}.pt"
                record = torch.load(path, map_location="cpu", weights_only=False)
                epochs.append(dict(year=year, seed=seed, stage=stage, **record["summary"]))
                checks.append(dict(check=f"{year}:{seed}:{stage}:finite_best_state", passed=bool(
                    record.get("complete") and all(torch.isfinite(x).all().item() for x in record["best_state"].values()))))
    pd.DataFrame(splits).to_csv(reports / "forecast_time_splits.csv", index=False)
    pd.DataFrame(epochs).to_csv(reports / "actual_training_epochs.csv", index=False)
    saved = torch.load(ROOT / "output/models/v45_forecast/2026/seed_17_pretraining.pt", map_location="cpu", weights_only=False)
    legacy = old.ForecastNetwork(arrays, cfg).eval()
    fixed = MaskSafeForecastNetwork(arrays, cfg).eval()
    legacy.load_state_dict(saved["best_state"])
    fixed.load_state_dict(saved["best_state"])
    split = old.shared_split(arrays, 2026, cfg)
    clipped, _ = old.clip_fold_features(arrays, split["train"], cfg)
    batch = list(safe_batch_at(clipped, split["test"][:2], cfg, "cpu"))
    # Counterfactual isolation test on real fitted weights, not a trading result.
    batch[0][:, :, 0], batch[1][:, :, 0] = 0., 0.
    measurements = {}
    for name, model in [("legacy", legacy), ("masked_repair", fixed)]:
        with torch.no_grad():
            before = model.encode(tuple(batch))
            model.core.asset_embedding.weight[0] += torch.linspace(-50., 50., cfg["d_model"])
            after = model.encode(tuple(batch))
        measurements[name] = dict(observed_asset_max_change=float((before[0][:, 1:]-after[0][:, 1:]).abs().max()),
                                  sleeve_max_change=float((before[3]-after[3]).abs().max()),
                                  units="hidden_representation_not_return_or_portfolio_PnL")
    checks.append(dict(check="fixed_missing_identity_isolation", passed=measurements["masked_repair"]["observed_asset_max_change"] < 1e-6
                      and measurements["masked_repair"]["sleeve_max_change"] < 1e-6))
    checks.append(dict(check="exact_38_context_assets", passed=len(metadata["assets"]) == 38))
    result = dict(engineering_checks_passed=all(c["passed"] for c in checks), checks=checks,
        original_data_end=metadata["end"], local_features=metadata["local_features"],
        macro_features=metadata["macro_features"], full_asset_cards_used_as_model_features=False,
        raw_macro_used_as_model_features=False, parameters=sum(p.numel() for p in legacy.parameters()),
        architecture=dict(sequence_length=cfg["sequence_length"], d_model=cfg["d_model"],
            global_attention_layers=cfg["num_layers"], heads=cfg["n_heads"],
            local_feature_attention_layers=cfg.get("local_num_layers", 0),
            description="hierarchical inverted-history variant; not unmodified official iTransformer"),
        counterfactual_missing_token_isolation=measurements, new_code_hash=source_hash(),
        new_model_trained=False, new_architecture_promoted=False, source_PIT_certified=False,
        limitations=["Finite outputs and chronology checks do not prove predictive alpha",
            "2024+ history has been repeatedly reviewed; not an untouched holdout",
            "Annual longest-label T/V/G windows leave supervised training stale",
            "New mask semantics require fresh training and evaluation; old weights used for isolation diagnostics only"])
    old.write_json(reports / "forecast_integrity_audit.json", result)
    print(json.dumps(dict(checks_passed=sum(c["passed"] for c in checks), checks=len(checks),
                         isolation=measurements, parameters=result["parameters"], splits=splits), indent=2))
    if not result["engineering_checks_passed"]:
        raise ValueError("Forecast integrity audit failed; do not train/promote")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.itransformer_daily_signal import load_daily_config, load_daily_dataset_arrays  # noqa: E402
from src.itransformer_signal import InvertedTimeSeriesTransformer, RISK_SLEEVES, prediction_metrics, set_seed  # noqa: E402


def fit_fold(x: np.ndarray, y: np.ndarray, train: np.ndarray, val: np.ndarray, cfg: dict, seed: int) -> nn.Module:
    set_seed(seed)
    model = InvertedTimeSeriesTransformer(x.shape[1], x.shape[2], y.shape[1], int(cfg["model"]["d_model"]), int(cfg["model"]["n_heads"]), int(cfg["model"]["num_layers"]), float(cfg["model"]["dropout"]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["model"]["learning_rate"]), weight_decay=float(cfg["model"]["weight_decay"]))
    tx = torch.tensor(x[train], dtype=torch.float32)
    ty = torch.tensor(y[train], dtype=torch.float32)
    vx = torch.tensor(x[val], dtype=torch.float32)
    vy = torch.tensor(y[val], dtype=torch.float32)
    best, best_state, wait = float("inf"), None, 0
    epochs = min(int(cfg["model"]["epochs"]), 150)
    patience = min(int(cfg["model"]["patience"]), 15)
    for _ in range(epochs):
        model.train(); optimizer.zero_grad()
        loss = nn.SmoothL1Loss()(model(tx), ty)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        model.eval()
        with torch.no_grad():
            score = float(nn.SmoothL1Loss()(model(vx), vy).item())
        if score < best:
            best = score; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}; wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Expanding-window walk-forward test for the daily iTransformer.")
    parser.add_argument("--config", default="config/itransformer_daily_ohlcv_config.yaml")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    cfg = load_daily_config(ROOT / args.config)
    df, x, y, _ = load_daily_dataset_arrays(cfg)
    dates = pd.to_datetime(df["as_of_date"])
    prediction_rows, fold_rows = [], []
    for test_year in [2023, 2024, 2025, 2026]:
        train = dates.dt.year.le(test_year - 2).to_numpy()
        val = dates.dt.year.eq(test_year - 1).to_numpy()
        test = dates.dt.year.eq(test_year).to_numpy()
        if train.sum() < 20 or val.sum() == 0 or test.sum() == 0:
            continue
        model = fit_fold(x, y, train, val, cfg, args.seed)
        model.eval()
        with torch.no_grad():
            pred = model(torch.tensor(x[test], dtype=torch.float32)).cpu().numpy()
        fold = prediction_metrics(pred, y[test])
        fold.update({"test_year": test_year, "train_rows": int(train.sum()), "val_rows": int(val.sum()), "test_rows": int(test.sum()), "seed": args.seed})
        fold_rows.append(fold)
        for local_i, global_i in enumerate(np.where(test)[0]):
            for j, sleeve in enumerate(RISK_SLEEVES):
                prediction_rows.append({"as_of_date": df.iloc[global_i]["as_of_date"], "label_date": df.iloc[global_i]["label_date"], "sleeve": sleeve, "pred_return": float(pred[local_i, j]), "realized_return": float(y[global_i, j]), "test_year": test_year})
    predictions = pd.DataFrame(prediction_rows)
    if predictions.empty:
        raise RuntimeError("No valid walk-forward folds.")
    pred_matrix = predictions.pivot(index="as_of_date", columns="sleeve", values="pred_return").reindex(columns=RISK_SLEEVES)
    real_matrix = predictions.pivot(index="as_of_date", columns="sleeve", values="realized_return").reindex(columns=RISK_SLEEVES)
    aggregate = prediction_metrics(pred_matrix.to_numpy(), real_matrix.to_numpy())
    aggregate.update({"test_year": "aggregate", "train_rows": np.nan, "val_rows": np.nan, "test_rows": len(pred_matrix), "seed": args.seed})
    report = pd.DataFrame(fold_rows + [aggregate])
    report_path = ROOT / "output" / "reports" / "itransformer_daily_ohlcv_walkforward_report.csv"
    prediction_path = ROOT / "data_processed" / "itransformer_daily_ohlcv_walkforward_predictions.parquet"
    report.to_csv(report_path, index=False, encoding="utf-8-sig")
    predictions.to_parquet(prediction_path, index=False)
    print(json.dumps({"report": str(report_path.relative_to(ROOT)), "predictions": str(prediction_path.relative_to(ROOT)), "folds": len(fold_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

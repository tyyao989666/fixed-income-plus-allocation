from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.itransformer_signal import (  # noqa: E402
    RISK_SLEEVES,
    load_config,
    load_dataset_arrays,
    out_path,
    prediction_metrics,
    split_masks,
    write_prediction_frame,
)


def expanding_mean_baseline(y: np.ndarray, masks: dict[str, np.ndarray]) -> np.ndarray:
    pred = np.zeros_like(y)
    train_idx = np.where(masks["train"])[0]
    fallback = y[train_idx].mean(axis=0) if len(train_idx) else y.mean(axis=0)
    for i in range(len(y)):
        hist = y[:i]
        if len(hist) == 0:
            pred[i] = fallback
        else:
            pred[i] = hist[-6:].mean(axis=0)
    return pred


def main() -> None:
    cfg = load_config()
    df, _, y, _ = load_dataset_arrays(cfg)
    masks = split_masks(df, cfg)
    pred = expanding_mean_baseline(y, masks)
    pred_df = write_prediction_frame(cfg, df, pred, y, "baseline_predictions")

    rows = []
    for split in ["train", "val", "test", "prediction"]:
        metrics = prediction_metrics(pred[masks[split]], y[masks[split]])
        metrics["split"] = split
        metrics["model"] = "rolling_6m_mean"
        rows.append(metrics)
    report = pd.DataFrame(rows)[["model", "split", "rows", "mse", "rank_ic", "top1_hit_rate", "top2_realized_return", "long_short_spread"]]
    path = out_path(cfg, "baseline_report")
    path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(path, index=False, encoding="utf-8-sig")
    print(
        json.dumps(
            {
                "predictions": cfg["outputs"]["baseline_predictions"],
                "report": cfg["outputs"]["baseline_report"],
                "prediction_rows": len(pred_df),
                "target_sleeves": RISK_SLEEVES,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

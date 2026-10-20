from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.itransformer_signal import RISK_SLEEVES, prediction_metrics, softmax_np  # noqa: E402


BASE = ROOT / "data_processed" / "itransformer_predictions_daily_path_monthly.parquet"
ENHANCED = ROOT / "data_processed" / "v4_itransformer_daily_ohlcv_ensemble_predictions.parquet"
OUT = ROOT / "data_processed" / "itransformer_predictions_daily_blended_monthly.parquet"
GRID_OUT = ROOT / "output" / "reports" / "itransformer_daily_blend_grid.csv"
REPORT_OUT = ROOT / "output" / "reports" / "itransformer_daily_blended_validation_report.csv"
MANIFEST_OUT = ROOT / "output" / "reports" / "itransformer_daily_blended_manifest.json"


def matrix_metrics(frame: pd.DataFrame, pred_col: str) -> dict[str, float]:
    pred = frame.pivot(index="as_of_date", columns="sleeve", values=pred_col).reindex(columns=RISK_SLEEVES)
    realized = frame.pivot(index="as_of_date", columns="sleeve", values="realized_return").reindex(columns=RISK_SLEEVES)
    common = pred.dropna().index.intersection(realized.dropna().index)
    return prediction_metrics(pred.loc[common].to_numpy(), realized.loc[common].to_numpy())


def main() -> None:
    base = pd.read_parquet(BASE)
    enhanced = pd.read_parquet(ENHANCED)
    keys = ["as_of_date", "label_date", "sleeve"]
    base = base.dropna(subset=["realized_return"]).drop_duplicates(keys, keep="last")
    enhanced = enhanced.drop_duplicates(keys, keep="last")
    merged = base[keys + ["pred_return", "realized_return"]].merge(
        enhanced[keys + ["pred_return", "pred_std"]], on=keys, how="inner", suffixes=("_base", "_enhanced")
    )
    merged["as_of_date"] = pd.to_datetime(merged["as_of_date"])

    val_mask = merged["as_of_date"].gt(pd.Timestamp("2022-12-31")) & merged["as_of_date"].le(pd.Timestamp("2023-12-31"))
    grid_rows = []
    for alpha in np.linspace(0.0, 1.0, 11):
        merged["candidate"] = (1.0 - alpha) * merged["pred_return_base"] + alpha * merged["pred_return_enhanced"]
        row = matrix_metrics(merged[val_mask], "candidate")
        row["enhanced_weight"] = float(alpha)
        row["selection_score"] = row["rank_ic"] + 0.25 * row["top1_hit_rate"] + 5.0 * row["long_short_spread"] - 50.0 * row["mse"]
        grid_rows.append(row)
    grid = pd.DataFrame(grid_rows).sort_values(["selection_score", "mse"], ascending=[False, True])
    GRID_OUT.parent.mkdir(parents=True, exist_ok=True)
    grid.to_csv(GRID_OUT, index=False, encoding="utf-8-sig")
    alpha = float(grid.iloc[0]["enhanced_weight"])

    merged["pred_return"] = (1.0 - alpha) * merged["pred_return_base"] + alpha * merged["pred_return_enhanced"]
    probs = []
    ranks = []
    for _, active in merged.groupby("as_of_date", sort=True):
        values = active["pred_return"].to_numpy(dtype=float)[None, :]
        probs.extend(softmax_np(values, temperature=0.03)[0].tolist())
        order = (-values[0]).argsort().tolist()
        ranks.extend([order.index(i) + 1 for i in range(len(active))])
    merged["pred_prob"] = probs
    merged["pred_rank"] = ranks
    merged["model_disagreement"] = merged["pred_std"].fillna(0.0)
    merged["model_version"] = f"daily_blend_base_{1-alpha:.1f}_ohlcv_{alpha:.1f}"
    out = merged[keys + ["pred_return", "pred_prob", "realized_return", "pred_rank", "model_disagreement", "model_version"]]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)

    report_rows = []
    masks = {
        "train": merged["as_of_date"].le(pd.Timestamp("2022-12-31")),
        "val": val_mask,
        "test": merged["as_of_date"].gt(pd.Timestamp("2023-12-31")),
        "prediction": merged["as_of_date"].ge(pd.Timestamp("2022-01-31")),
    }
    for split, mask in masks.items():
        row = matrix_metrics(merged[mask], "pred_return")
        row.update({"model": "itransformer_daily_blended", "split": split, "enhanced_weight": alpha})
        report_rows.append(row)
    report = pd.DataFrame(report_rows)
    report.to_csv(REPORT_OUT, index=False, encoding="utf-8-sig")
    MANIFEST_OUT.write_text(json.dumps({"selected_enhanced_weight": alpha, "selection_split": "2023 validation", "output": str(OUT.relative_to(ROOT)), "report": str(REPORT_OUT.relative_to(ROOT))}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"selected_enhanced_weight": alpha, "output": str(OUT.relative_to(ROOT)), "report": str(REPORT_OUT.relative_to(ROOT))}, ensure_ascii=False))


if __name__ == "__main__":
    main()

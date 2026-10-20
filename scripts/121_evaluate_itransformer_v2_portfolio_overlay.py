from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.itransformer_daily_signal import load_daily_config  # noqa: E402
from src.itransformer_daily_timing_v2 import v2_path  # noqa: E402
from src.v1_strategy import (  # noqa: E402
    build_asset_weights,
    build_risk_budget_base,
    build_sleeve_weights,
    calc_metrics,
    load_config,
    run_backtest,
)


PROFILES = ["low_vol", "balanced", "high_vol"]
TILTS = [0.0, 0.003, 0.006, 0.010, 0.015, 0.020]
TEMP_DATA = ROOT / "data_processed" / "itransformer_v2_overlay_search"
TEMP_REPORTS = ROOT / "output" / "reports" / "itransformer_v2_overlay_search"


def materialize_monthly_predictions() -> Path:
    cfg = load_daily_config()
    source = pd.read_parquet(v2_path(cfg, "timing_v2_predictions"))
    source["as_of_date"] = pd.to_datetime(source["as_of_date"], errors="coerce")
    source["score"] = pd.to_numeric(source["combined_oos_adaptive_score"], errors="coerce")
    source = source.dropna(subset=["as_of_date", "sleeve", "score"]).copy()
    source["month"] = source["as_of_date"].dt.to_period("M")
    last_dates = source.groupby(["month", "sleeve"])["as_of_date"].transform("max")
    monthly = source[source["as_of_date"].eq(last_dates)].copy()
    monthly["as_of_date"] = monthly["month"].dt.to_timestamp("M")
    monthly["pred_return"] = monthly["score"]
    monthly["itransformer_model_weight"] = pd.to_numeric(
        monthly["adaptive_model_weight"], errors="coerce"
    ).fillna(0.65)
    monthly["training_cutoff"] = monthly["as_of_date"].map(
        lambda value: (
            pd.Timestamp(year=value.year - 1, month=12, day=31)
            if value.year <= 2024
            else value.to_period("Q").start_time - pd.Timedelta(days=1)
        )
    )

    def probability(values: pd.Series) -> pd.Series:
        raw = values.to_numpy(dtype=float)
        z = raw / 0.75
        z -= np.max(z)
        exp = np.exp(z)
        return pd.Series(exp / exp.sum(), index=values.index)

    monthly["pred_prob"] = monthly.groupby("as_of_date")["pred_return"].transform(probability)
    monthly["model_version"] = "identified_itransformer_v2_walkforward_adaptive"
    monthly["input_frequency"] = "daily_to_month_end"
    keep = [
        "as_of_date",
        "sleeve",
        "pred_return",
        "pred_prob",
        "itransformer_model_weight",
        "model_version",
        "input_frequency",
        "training_cutoff",
    ]
    path = v2_path(cfg, "timing_v2_monthly_oos")
    path.parent.mkdir(parents=True, exist_ok=True)
    monthly[keep].sort_values(["as_of_date", "pred_return"], ascending=[True, False]).to_parquet(
        path, index=False
    )
    return path


def period_metrics(nav: pd.DataFrame, start: str, end: str) -> dict[str, float]:
    sub = nav[pd.to_datetime(nav["date"]).between(pd.Timestamp(start), pd.Timestamp(end))].copy()
    if sub.empty:
        return {"annual_return": np.nan, "annual_vol": np.nan, "sharpe": np.nan, "max_drawdown": np.nan, "avg_turnover": np.nan}
    sub["nav"] = (1.0 + sub["net_return"]).cumprod()
    return calc_metrics(sub).set_index("metric")["value"].to_dict()


def case_config(profile: str, tilt: float, prediction_path: Path) -> dict[str, Any]:
    cfg = copy.deepcopy(load_config())
    cfg["strategy_version"] = "no_agent"
    cfg["product"]["selected_profile"] = profile
    overlay = cfg["signal_overlays"]["itransformer_rank"]
    overlay.update(
        {
            "enabled": True,
            "direct_tilt_enabled": True,
            "ep_mean_enabled": False,
            "predictions_path": str(prediction_path.relative_to(ROOT)).replace("\\", "/"),
            "enabled_profiles": [profile],
            "top_n": 1,
            "bottom_n": 1,
            "tilt": float(tilt),
            "min_top_prob": 0.0,
        }
    )
    prefix = f"{profile}_tilt_{int(round(tilt * 10000)):03d}bp"
    cfg["outputs"] = {
        "weights": f"data_processed/itransformer_v2_overlay_search/{prefix}_weights.parquet",
        "sleeve_weights": f"data_processed/itransformer_v2_overlay_search/{prefix}_sleeves.parquet",
        "risk_budget_base": f"data_processed/itransformer_v2_overlay_search/{prefix}_risk.parquet",
        "nav": f"output/reports/itransformer_v2_overlay_search/{prefix}_nav.csv",
        "metrics": f"output/reports/itransformer_v2_overlay_search/{prefix}_metrics.csv",
        "data_quality": f"output/reports/itransformer_v2_overlay_search/{prefix}_data_quality.csv",
        "ep_posterior": f"data_processed/itransformer_v2_overlay_search/{prefix}_posterior.parquet",
        "ep_signals": f"output/reports/itransformer_v2_overlay_search/{prefix}_signals.csv",
        "optimizer_diagnostics": f"output/reports/itransformer_v2_overlay_search/{prefix}_diagnostics.csv",
    }
    return cfg


def run_case(profile: str, tilt: float, prediction_path: str) -> dict[str, Any]:
    cfg = case_config(profile, tilt, Path(prediction_path))
    try:
        build_risk_budget_base(cfg)
        build_sleeve_weights(cfg)
        build_asset_weights(cfg)
        nav, metrics_frame = run_backtest(cfg)
        full = metrics_frame.set_index("metric")["value"].to_dict()
        row: dict[str, Any] = {"profile": profile, "tilt": tilt, "error": ""}
        row.update({f"full_{key}": value for key, value in full.items()})
        for name, start, end in [
            ("oos_2022", "2022-01-01", "2022-12-31"),
            ("oos_2023", "2023-01-01", "2023-12-31"),
            ("oos_2024", "2024-01-01", "2024-12-31"),
            ("development", "2022-01-01", "2024-12-31"),
            ("holdout", "2025-01-01", "2026-12-31"),
        ]:
            values = period_metrics(nav, start, end)
            row.update({f"{name}_{key}": value for key, value in values.items()})
        yearly_sharpes = [row[f"oos_{year}_sharpe"] for year in [2022, 2023, 2024]]
        row["development_score"] = float(
            np.nanmean(yearly_sharpes)
            + 0.30 * np.nanmin(yearly_sharpes)
            - 0.10 * row["development_avg_turnover"]
        )
        return row
    except Exception as exc:
        return {"profile": profile, "tilt": tilt, "error": repr(exc), "development_score": -np.inf}


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize and evaluate the iTransformer V2 portfolio overlay.")
    parser.add_argument("--materialize-only", action="store_true")
    args = parser.parse_args()
    prediction_path = materialize_monthly_predictions()
    if args.materialize_only:
        print(json.dumps({"monthly_predictions": str(prediction_path.relative_to(ROOT))}, ensure_ascii=False))
        return
    TEMP_DATA.mkdir(parents=True, exist_ok=True)
    TEMP_REPORTS.mkdir(parents=True, exist_ok=True)
    cases = [(profile, tilt, str(prediction_path)) for profile in PROFILES for tilt in TILTS]
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(run_case, *case): case for case in cases}
        for index, future in enumerate(as_completed(futures), 1):
            result = future.result()
            rows.append(result)
            print(
                f"done {index}/{len(cases)} {result['profile']} tilt={result['tilt']:.3%} score={result.get('development_score')}",
                flush=True,
            )
    detail = pd.DataFrame(rows).sort_values(["profile", "development_score"], ascending=[True, False])
    selected_rows: list[dict[str, Any]] = []
    for profile in PROFILES:
        group = detail[detail["profile"].eq(profile) & detail["error"].fillna("").eq("")].copy()
        selected = group.sort_values("development_score", ascending=False).iloc[0]
        baseline = group[group["tilt"].eq(0.0)].iloc[0]
        row = selected.to_dict()
        for metric in ["annual_return", "sharpe", "max_drawdown", "avg_turnover"]:
            row[f"development_delta_{metric}"] = float(selected[f"development_{metric}"] - baseline[f"development_{metric}"])
            row[f"holdout_delta_{metric}"] = float(selected[f"holdout_{metric}"] - baseline[f"holdout_{metric}"])
            row[f"full_delta_{metric}"] = float(selected[f"full_{metric}"] - baseline[f"full_{metric}"])
        row["passes_portfolio_gate"] = bool(
            float(selected["tilt"]) > 0.0
            and row["development_delta_sharpe"] >= 0.02
            and row["holdout_delta_sharpe"] >= 0.0
            and row["holdout_delta_annual_return"] >= -0.001
            and row["full_delta_sharpe"] >= 0.0
        )
        selected_rows.append(row)
    selected = pd.DataFrame(selected_rows)
    detail_path = ROOT / "output" / "reports" / "itransformer_v2_portfolio_overlay_search.csv"
    selected_path = ROOT / "output" / "reports" / "itransformer_v2_portfolio_overlay_selection.csv"
    detail.to_csv(detail_path, index=False, encoding="utf-8-sig")
    selected.to_csv(selected_path, index=False, encoding="utf-8-sig")
    print(selected[["profile", "tilt", "development_delta_sharpe", "holdout_delta_sharpe", "holdout_delta_annual_return", "passes_portfolio_gate"]].to_string(index=False))
    print(json.dumps({"monthly_predictions": str(prediction_path.relative_to(ROOT)), "detail": str(detail_path.relative_to(ROOT)), "selection": str(selected_path.relative_to(ROOT))}, ensure_ascii=False))
    shutil.rmtree(TEMP_DATA, ignore_errors=True)
    shutil.rmtree(TEMP_REPORTS, ignore_errors=True)


if __name__ == "__main__":
    main()

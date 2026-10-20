from __future__ import annotations

import argparse
import importlib
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_v1_full_pipeline import parse_profiles  # noqa: E402
from src.v1_strategy import calc_metrics, load_config, run_backtest  # noqa: E402


grid = importlib.import_module("scripts.61_grid_search_v1_strategy")


WINDOW_STARTS = ["2022-02-28", "2022-07-31", "2023-01-31", "2023-07-31", "2024-01-31"]
TEMP_DATA = ROOT / "data_processed" / "walkforward_grid_v1"
TEMP_REPORTS = ROOT / "output" / "reports" / "walkforward_grid_v1"


def metrics_for_window(nav: pd.DataFrame, start: str) -> dict[str, float]:
    sub = nav[pd.to_datetime(nav["date"]).ge(pd.Timestamp(start))].copy()
    if sub.empty:
        return {"months": 0}
    sub = sub.reset_index(drop=True)
    sub["nav"] = (1.0 + sub["net_return"]).cumprod()
    metrics = calc_metrics(sub)
    return dict(zip(metrics["metric"], metrics["value"]))


def set_outputs(cfg: dict[str, Any], case_id: str, profile: str) -> None:
    prefix = f"{case_id}_{profile}"
    cfg["outputs"] = {
        "weights": f"data_processed/walkforward_grid_v1/{prefix}_weights.parquet",
        "sleeve_weights": f"data_processed/walkforward_grid_v1/{prefix}_sleeves.parquet",
        "risk_budget_base": f"data_processed/walkforward_grid_v1/{prefix}_risk_base.parquet",
        "nav": f"output/reports/walkforward_grid_v1/{prefix}_nav.csv",
        "metrics": f"output/reports/walkforward_grid_v1/{prefix}_metrics.csv",
        "attribution": f"output/reports/walkforward_grid_v1/{prefix}_attribution.csv",
        "robustness": f"output/reports/walkforward_grid_v1/{prefix}_robustness.csv",
        "data_quality": f"output/reports/walkforward_grid_v1/{prefix}_data_quality.csv",
    }


def run_case(case_id: str, params: dict[str, Any], profiles: list[str]) -> list[dict[str, Any]]:
    rows = []
    for profile in profiles:
        cfg = load_config()
        grid.apply_params(cfg, profile, params)
        set_outputs(cfg, case_id, profile)
        try:
            nav, full_metrics = run_backtest(cfg)
            full = full_metrics.set_index("metric")["value"].to_dict()
            for start in WINDOW_STARTS:
                window = metrics_for_window(nav, start)
                rows.append(
                    {
                        "case_id": case_id,
                        "profile": profile,
                        "window_start": start,
                        **params,
                        "full_annual_return": full.get("annual_return", np.nan),
                        "full_sharpe": full.get("sharpe", np.nan),
                        "annual_return": window.get("annual_return", np.nan),
                        "annual_vol": window.get("annual_vol", np.nan),
                        "sharpe": window.get("sharpe", np.nan),
                        "max_drawdown": window.get("max_drawdown", np.nan),
                        "avg_turnover": window.get("avg_turnover", np.nan),
                        "months": window.get("months", 0),
                        "error": "",
                    }
                )
        except Exception as exc:
            rows.append({"case_id": case_id, "profile": profile, "window_start": "", **params, "error": repr(exc)})
    return rows


def summarize(detail: pd.DataFrame, profiles: list[str]) -> pd.DataFrame:
    ok = detail[detail["error"].fillna("").eq("")].copy()
    rows = []
    for case_id, group in ok.groupby("case_id"):
        expected = len(profiles) * len(WINDOW_STARTS)
        if len(group) != expected:
            continue
        row = {
            "case_id": case_id,
            "cases": len(group),
            "avg_window_return": group["annual_return"].mean(),
            "avg_window_sharpe": group["sharpe"].mean(),
            "min_window_sharpe": group["sharpe"].min(),
            "worst_window_drawdown": group["max_drawdown"].min(),
            "avg_window_turnover": group["avg_turnover"].mean(),
            "return_win_rate_vs_zero": float((group["annual_return"] > 0).mean()),
        }
        latest = group.groupby("profile", as_index=False).tail(1)
        row["avg_full_return"] = latest["full_annual_return"].mean()
        row["avg_full_sharpe"] = latest["full_sharpe"].mean()
        first = group.iloc[0]
        for col in [
            "rate_30y_weight",
            "itr_tilt",
            "min_top_prob",
            "trend_tilt",
            "trend_max",
            "trend_high_vol",
            "asset_blend",
            "asset_temp",
            "cb_scale",
            "equity_scale",
            "credit_share",
        ]:
            row[col] = first[col]
        row["score"] = (
            row["avg_window_sharpe"]
            + 1.5 * row["avg_window_return"]
            + 0.6 * row["min_window_sharpe"]
            + 0.5 * row["worst_window_drawdown"]
            - 0.15 * row["avg_window_turnover"]
            + 0.25 * row["return_win_rate_vs_zero"]
        )
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["score", "avg_window_sharpe", "avg_window_return"], ascending=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward grid-search V1 strategy parameters.")
    parser.add_argument("--profiles", default="all")
    parser.add_argument("--rate-30y-weights", default="0.66,0.68,0.70")
    parser.add_argument("--itr-tilts", default="0.060,0.065,0.070")
    parser.add_argument("--min-top-probs", default="0.35")
    parser.add_argument("--trend-pairs", default="0.030:0.080,0.034:0.090")
    parser.add_argument("--trend-high-vol", default="true")
    parser.add_argument("--asset-blends", default="0.80")
    parser.add_argument("--asset-temps", default="0.035")
    parser.add_argument("--cb-scales", default="0.80,1.00")
    parser.add_argument("--equity-scales", default="0.80,1.00")
    parser.add_argument("--credit-shares", default="0.55")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--keep-temp", action="store_true")
    args = parser.parse_args()

    profiles = parse_profiles(args.profiles)
    TEMP_DATA.mkdir(parents=True, exist_ok=True)
    TEMP_REPORTS.mkdir(parents=True, exist_ok=True)
    param_grid = grid.build_param_grid(args)
    if args.limit:
        param_grid = param_grid[: args.limit]

    detail_path = ROOT / "output" / "reports" / "v1_walkforward_grid_detail.csv"
    summary_path = ROOT / "output" / "reports" / "v1_walkforward_grid_summary.csv"
    rows = []
    for idx, params in enumerate(param_grid, 1):
        case_id = f"wf_case_{idx:04d}"
        rows.extend(run_case(case_id, params, profiles))
        if args.checkpoint_every and idx % args.checkpoint_every == 0:
            detail = pd.DataFrame(rows)
            detail.to_csv(detail_path, index=False, encoding="utf-8-sig")
            summarize(detail, profiles).to_csv(summary_path, index=False, encoding="utf-8-sig")
        if idx % 10 == 0 or idx == len(param_grid):
            print(f"done {idx}/{len(param_grid)}", flush=True)

    detail = pd.DataFrame(rows)
    detail.to_csv(detail_path, index=False, encoding="utf-8-sig")
    summary = summarize(detail, profiles)
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    if not summary.empty:
        print(summary.head(20).to_string(index=False))
    print({"detail": str(detail_path.relative_to(ROOT)), "summary": str(summary_path.relative_to(ROOT)), "cases": len(param_grid)})

    if not args.keep_temp:
        shutil.rmtree(TEMP_DATA, ignore_errors=True)
        shutil.rmtree(TEMP_REPORTS, ignore_errors=True)


if __name__ == "__main__":
    main()

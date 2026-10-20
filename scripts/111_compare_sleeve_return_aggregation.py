from __future__ import annotations

import argparse
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

from src.v1_strategy import calc_metrics, load_config, run_backtest  # noqa: E402


PROFILES = ["low_vol", "balanced", "high_vol"]
METHODS = ["equal_assets", "preferred_assets_equal", "preferred_asset_weights"]
DEV_FOLDS = {
    "dev_2018_2020": ("2018-01-01", "2020-12-31"),
    "dev_2021_2022": ("2021-01-01", "2022-12-31"),
    "validation_2023_2024": ("2023-01-01", "2024-12-31"),
}
HOLDOUT = ("2025-01-01", "2026-12-31")
TEMP_DATA = ROOT / "data_processed" / "sleeve_return_aggregation_comparison"
TEMP_REPORTS = ROOT / "output" / "reports" / "sleeve_return_aggregation_comparison"
DETAIL_PATH = ROOT / "output" / "reports" / "v1_sleeve_return_aggregation_comparison.csv"
NOTES_PATH = ROOT / "output" / "reports" / "v1_sleeve_return_aggregation_comparison.md"


def configure_case(method: str, profile: str) -> dict[str, Any]:
    cfg = load_config()
    cfg["strategy_version"] = "no_agent"
    cfg["product"]["selected_profile"] = profile
    cfg["risk_budget"]["sleeve_return_aggregation"] = method
    cfg["signal_overlays"]["itransformer_rank"]["enabled"] = False
    cfg["signal_overlays"]["asset_itransformer"]["enabled"] = False
    cfg["signal_overlays"]["risk_on_rotation"]["enabled"] = False
    cfg["signal_overlays"]["rule_output"]["apply_alt_internal_overlay"] = False
    prefix = f"{method}__{profile}"
    cfg["outputs"] = {
        "weights": f"data_processed/sleeve_return_aggregation_comparison/{prefix}_weights.parquet",
        "sleeve_weights": f"data_processed/sleeve_return_aggregation_comparison/{prefix}_sleeves.parquet",
        "risk_budget_base": f"data_processed/sleeve_return_aggregation_comparison/{prefix}_risk.parquet",
        "nav": f"output/reports/sleeve_return_aggregation_comparison/{prefix}_nav.csv",
        "metrics": f"output/reports/sleeve_return_aggregation_comparison/{prefix}_metrics.csv",
        "data_quality": f"output/reports/sleeve_return_aggregation_comparison/{prefix}_data_quality.csv",
    }
    return cfg


def period_metrics(nav: pd.DataFrame, start: str, end: str) -> dict[str, float]:
    sub = nav[pd.to_datetime(nav["date"]).between(pd.Timestamp(start), pd.Timestamp(end))].copy()
    if sub.empty:
        return {"annual_return": np.nan, "annual_vol": np.nan, "sharpe": np.nan, "max_drawdown": np.nan, "avg_turnover": np.nan}
    sub["nav"] = (1.0 + sub["net_return"]).cumprod()
    return calc_metrics(sub).set_index("metric")["value"].to_dict()


def run_case(method: str, profile: str) -> dict[str, Any]:
    row: dict[str, Any] = {"method": method, "profile": profile}
    try:
        cfg = configure_case(method, profile)
        nav, metrics = run_backtest(cfg)
        full = metrics.set_index("metric")["value"].to_dict()
        windows = {
            **{name: period_metrics(nav, start, end) for name, (start, end) in DEV_FOLDS.items()},
            "development_2018_2024": period_metrics(nav, "2018-01-01", "2024-12-31"),
            "holdout_2025_plus": period_metrics(nav, *HOLDOUT),
        }
        for window, values in windows.items():
            for metric, value in values.items():
                row[f"{window}_{metric}"] = value
        for metric, value in full.items():
            row[f"full_{metric}"] = value
        sleeves = pd.read_parquet(ROOT / cfg["outputs"]["sleeve_weights"])
        cols = ["rates", "credit", "convertible_bonds", "equity", "alternatives", "cash"]
        row["max_sleeve_sum_error"] = float((sleeves[cols].sum(axis=1) - 1.0).abs().max())
        row["cash_weight_max"] = float(sleeves["cash"].max())
        row["status"] = "ok"
    except Exception as exc:  # noqa: BLE001
        row["status"] = "error"
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def add_scores(detail: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ok = detail[detail["status"].eq("ok")].copy()
    baseline = ok[ok["method"].eq("equal_assets")].set_index("profile")
    importance = {"low_vol": 0.20, "balanced": 0.35, "high_vol": 0.45}
    fold_cols = [f"{name}_sharpe" for name in DEV_FOLDS]
    rows: list[dict[str, Any]] = []
    for method, group in ok.groupby("method"):
        score = 0.0
        worst = np.inf
        summary: dict[str, float] = {}
        for _, item in group.iterrows():
            profile = str(item["profile"])
            base = baseline.loc[profile]
            ds = float(item["development_2018_2024_sharpe"] - base["development_2018_2024_sharpe"])
            dr = float(item["development_2018_2024_annual_return"] - base["development_2018_2024_annual_return"])
            dd = float(item["development_2018_2024_max_drawdown"] - base["development_2018_2024_max_drawdown"])
            dt = float(item["development_2018_2024_avg_turnover"] - base["development_2018_2024_avg_turnover"])
            folds = [float(item[c] - base[c]) for c in fold_cols]
            worst = min(worst, min(folds))
            score += importance[profile] * (ds + 0.35 * np.mean(folds) + 3.0 * dr + 1.5 * dd - 0.15 * max(dt, 0.0) - 0.35 * max(-min(folds) - 0.10, 0.0))
            summary.update(
                {
                    f"{profile}_dev_sharpe_delta": ds,
                    f"{profile}_worst_fold_sharpe_delta": min(folds),
                    f"{profile}_holdout_sharpe": float(item["holdout_2025_plus_sharpe"]),
                    f"{profile}_full_return": float(item["full_annual_return"]),
                    f"{profile}_full_sharpe": float(item["full_sharpe"]),
                    f"{profile}_full_drawdown": float(item["full_max_drawdown"]),
                    f"{profile}_full_turnover": float(item["full_avg_turnover"]),
                }
            )
        vols = group.set_index("profile")["full_annual_vol"]
        vol_order = bool(vols["low_vol"] < vols["balanced"] < vols["high_vol"])
        if not vol_order:
            score -= 2.0
        rows.append({"method": method, "selection_score": score, "worst_dev_fold_sharpe_delta": worst, "volatility_order_ok": vol_order, **summary})
    summary = pd.DataFrame(rows).sort_values(["selection_score", "worst_dev_fold_sharpe_delta"], ascending=False).reset_index(drop=True)
    return detail.merge(summary[["method", "selection_score"]], on="method", how="left"), summary


def write_notes(detail: pd.DataFrame, summary: pd.DataFrame) -> None:
    best = str(summary.iloc[0]["method"])
    baseline = detail[detail["method"].eq("equal_assets")].set_index("profile")
    selected = detail[detail["method"].eq(best)].set_index("profile")
    robust = bool(
        best != "equal_assets"
        and summary.iloc[0]["worst_dev_fold_sharpe_delta"] >= -0.10
        and all(selected.loc[p, "development_2018_2024_sharpe"] >= baseline.loc[p, "development_2018_2024_sharpe"] for p in PROFILES)
    )
    lines = [
        "# V1 袖套收益聚合口径比较",
        "",
        "## 结论",
        "",
        f"- 发现期最优口径：`{best}`。",
        f"- 是否达到自动替换门槛：`{'是' if robust else '否'}`。",
        "- `equal_assets` 是旧口径；`preferred_assets_equal` 只保留正式资产池后等权；`preferred_asset_weights` 按正式 profile 资产锚聚合。",
        "- 选口径只看 2018-2024，2025+ 不参与排序。",
        "",
        "| 产品 | 旧口径发现期 Sharpe | 候选发现期 Sharpe | 旧口径2025+ Sharpe | 候选2025+ Sharpe | 候选全样本收益 | 候选全样本 Sharpe |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for profile in PROFILES:
        b = baseline.loc[profile]
        s = selected.loc[profile]
        lines.append(
            f"| {profile} | {b['development_2018_2024_sharpe']:.3f} | {s['development_2018_2024_sharpe']:.3f} | "
            f"{b['holdout_2025_plus_sharpe']:.3f} | {s['holdout_2025_plus_sharpe']:.3f} | {s['full_annual_return']:.2%} | {s['full_sharpe']:.3f} |"
        )
    lines.extend(["", "## 判断", "", "若一致性更高的正式资产口径不能同时改善三个产品的发现期 Sharpe，本轮只保留代码开关和问题披露，不替换正式回测口径。"])
    NOTES_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare sleeve return aggregation methods.")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--keep-temp", action="store_true")
    args = parser.parse_args()
    TEMP_DATA.mkdir(parents=True, exist_ok=True)
    TEMP_REPORTS.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(run_case, method, profile): (method, profile) for method in METHODS for profile in PROFILES}
        for index, future in enumerate(as_completed(futures), start=1):
            method, profile = futures[future]
            row = future.result()
            rows.append(row)
            print(f"[{index:02d}/{len(futures)}] {method} {profile}: {row['status']}", flush=True)
    detail = pd.DataFrame(rows)
    errors = detail[detail["status"].ne("ok")]
    if not errors.empty:
        raise RuntimeError(errors[["method", "profile", "error"]].to_string(index=False))
    detail, summary = add_scores(detail)
    detail.sort_values(["selection_score", "method", "profile"], ascending=[False, True, True]).to_csv(DETAIL_PATH, index=False, encoding="utf-8-sig")
    write_notes(detail, summary)
    print("\nMethod summary (selection uses 2018-2024 only):")
    print(summary.to_string(index=False))
    print(f"\nDetail: {DETAIL_PATH.relative_to(ROOT)}")
    print(f"Notes: {NOTES_PATH.relative_to(ROOT)}")
    if not args.keep_temp:
        shutil.rmtree(TEMP_DATA, ignore_errors=True)
        shutil.rmtree(TEMP_REPORTS, ignore_errors=True)


if __name__ == "__main__":
    main()

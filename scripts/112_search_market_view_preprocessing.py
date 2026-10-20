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

from src.v1_strategy import calc_metrics, load_config, run_backtest  # noqa: E402


PROFILES = ["low_vol", "balanced", "high_vol"]
DEV_FOLDS = {
    "dev_2018_2020": ("2018-01-01", "2020-12-31"),
    "dev_2021_2022": ("2021-01-01", "2022-12-31"),
    "validation_2023_2024": ("2023-01-01", "2024-12-31"),
}
HOLDOUT = ("2025-01-01", "2026-12-31")
TEMP_DATA = ROOT / "data_processed" / "view_preprocessing_search"
TEMP_REPORTS = ROOT / "output" / "reports" / "view_preprocessing_search"
DETAIL_PATH = ROOT / "output" / "reports" / "v1_view_preprocessing_search_detail.csv"
POLICY_PATH = ROOT / "output" / "reports" / "v1_view_preprocessing_search_policy_summary.csv"
NOTES_PATH = ROOT / "output" / "reports" / "v1_view_preprocessing_search_summary.md"


def build_policies() -> list[dict[str, Any]]:
    return [
        {"policy_id": "baseline_raw_month", "mode": "raw_month", "recent_days": 5, "persistence_power": 0.5},
        {"policy_id": "last_day", "mode": "last_day", "recent_days": 1, "persistence_power": 0.5},
        {"policy_id": "recent_3d_raw", "mode": "recent_days_raw", "recent_days": 3, "persistence_power": 0.5},
        {"policy_id": "recent_5d_raw", "mode": "recent_days_raw", "recent_days": 5, "persistence_power": 0.5},
        {"policy_id": "monthly_pair_mean", "mode": "monthly_pair_mean", "recent_days": 5, "persistence_power": 0.0},
        {"policy_id": "monthly_pair_persistence_sqrt", "mode": "monthly_pair_persistence", "recent_days": 5, "persistence_power": 0.5},
        {"policy_id": "monthly_pair_persistence_linear", "mode": "monthly_pair_persistence", "recent_days": 5, "persistence_power": 1.0},
        {"policy_id": "recent_5d_pair_mean", "mode": "recent_pair_mean", "recent_days": 5, "persistence_power": 0.0},
        {"policy_id": "recent_5d_pair_persistence", "mode": "recent_pair_persistence", "recent_days": 5, "persistence_power": 0.5},
    ]


def configure_case(spec: dict[str, Any], profile: str) -> dict[str, Any]:
    cfg = load_config()
    cfg["strategy_version"] = "no_agent"
    cfg["product"]["selected_profile"] = profile
    cfg["signal_overlays"]["itransformer_rank"]["enabled"] = False
    cfg["signal_overlays"]["asset_itransformer"]["enabled"] = False
    cfg["signal_overlays"]["view_preprocessing"] = {
        "mode": str(spec["mode"]),
        "recent_days": int(spec["recent_days"]),
        "persistence_power": float(spec["persistence_power"]),
    }
    prefix = f"{spec['policy_id']}__{profile}"
    cfg["outputs"] = {
        "weights": f"data_processed/view_preprocessing_search/{prefix}_weights.parquet",
        "sleeve_weights": f"data_processed/view_preprocessing_search/{prefix}_sleeves.parquet",
        "risk_budget_base": f"data_processed/view_preprocessing_search/{prefix}_risk.parquet",
        "nav": f"output/reports/view_preprocessing_search/{prefix}_nav.csv",
        "metrics": f"output/reports/view_preprocessing_search/{prefix}_metrics.csv",
        "data_quality": f"output/reports/view_preprocessing_search/{prefix}_data_quality.csv",
    }
    return cfg


def period_metrics(nav: pd.DataFrame, start: str, end: str) -> dict[str, float]:
    sub = nav[pd.to_datetime(nav["date"]).between(pd.Timestamp(start), pd.Timestamp(end))].copy()
    if sub.empty:
        return {k: np.nan for k in ["annual_return", "annual_vol", "sharpe", "max_drawdown", "avg_turnover"]}
    sub["nav"] = (1.0 + sub["net_return"]).cumprod()
    return calc_metrics(sub).set_index("metric")["value"].to_dict()


def run_case(spec: dict[str, Any], profile: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "policy_id": spec["policy_id"],
        "profile": profile,
        "spec_json": json.dumps(spec, sort_keys=True),
    }
    try:
        cfg = configure_case(spec, profile)
        nav, metrics = run_backtest(cfg)
        for metric, value in metrics.set_index("metric")["value"].to_dict().items():
            row[f"full_{metric}"] = value
        windows = {
            **{name: period_metrics(nav, start, end) for name, (start, end) in DEV_FOLDS.items()},
            "development_2018_2024": period_metrics(nav, "2018-01-01", "2024-12-31"),
            "holdout_2025_plus": period_metrics(nav, *HOLDOUT),
        }
        for window, values in windows.items():
            for metric, value in values.items():
                row[f"{window}_{metric}"] = value
        weights = pd.read_parquet(ROOT / cfg["outputs"]["weights"])
        row["max_weight_sum_error"] = float((weights.groupby("date")["weight"].sum() - 1.0).abs().max())
        row["cash_weight_max"] = float(weights.loc[weights["asset_code"].eq("cash"), "weight"].max()) if (weights["asset_code"] == "cash").any() else 0.0
        row["status"] = "ok"
    except Exception as exc:  # noqa: BLE001
        row["status"] = "error"
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def score_policies(detail: pd.DataFrame) -> pd.DataFrame:
    ok = detail[detail["status"].eq("ok")].copy()
    baseline = ok[ok["policy_id"].eq("baseline_raw_month")].set_index("profile")
    rows: list[dict[str, Any]] = []
    importance = {"low_vol": 0.20, "balanced": 0.35, "high_vol": 0.45}
    for policy_id, group in ok.groupby("policy_id"):
        if set(group["profile"]) != set(PROFILES):
            continue
        score = 0.0
        worst_fold = np.inf
        record: dict[str, Any] = {"policy_id": policy_id}
        for _, item in group.iterrows():
            profile = str(item["profile"])
            base = baseline.loc[profile]
            dev_sharpe_delta = float(item["development_2018_2024_sharpe"] - base["development_2018_2024_sharpe"])
            dev_return_delta = float(item["development_2018_2024_annual_return"] - base["development_2018_2024_annual_return"])
            dev_dd_delta = float(item["development_2018_2024_max_drawdown"] - base["development_2018_2024_max_drawdown"])
            turnover_delta = float(item["development_2018_2024_avg_turnover"] - base["development_2018_2024_avg_turnover"])
            fold_deltas = [float(item[f"{fold}_sharpe"] - base[f"{fold}_sharpe"]) for fold in DEV_FOLDS]
            profile_score = (
                dev_sharpe_delta
                + 0.35 * float(np.mean(fold_deltas))
                + 3.0 * dev_return_delta
                + 1.5 * dev_dd_delta
                - 0.15 * max(turnover_delta, 0.0)
                - 0.35 * max(-min(fold_deltas) - 0.10, 0.0)
            )
            score += importance[profile] * profile_score
            worst_fold = min(worst_fold, min(fold_deltas))
            for name, value in {
                "dev_sharpe_delta": dev_sharpe_delta,
                "dev_return_delta": dev_return_delta,
                "worst_fold_sharpe_delta": min(fold_deltas),
                "holdout_sharpe": float(item["holdout_2025_plus_sharpe"]),
                "full_return": float(item["full_annual_return"]),
                "full_vol": float(item["full_annual_vol"]),
                "full_sharpe": float(item["full_sharpe"]),
                "full_drawdown": float(item["full_max_drawdown"]),
                "full_turnover": float(item["full_avg_turnover"]),
            }.items():
                record[f"{profile}_{name}"] = value
        vol_order = record["low_vol_full_vol"] < record["balanced_full_vol"] < record["high_vol_full_vol"]
        gate = bool(
            vol_order
            and worst_fold >= -0.10
            and all(record[f"{profile}_dev_sharpe_delta"] >= -0.02 for profile in PROFILES)
        )
        if not vol_order:
            score -= 2.0
        record.update(
            {
                "selection_score": float(score),
                "worst_dev_fold_sharpe_delta": float(worst_fold),
                "volatility_order_ok": bool(vol_order),
                "formal_gate_pass": gate,
            }
        )
        rows.append(record)
    return pd.DataFrame(rows).sort_values(["formal_gate_pass", "selection_score"], ascending=[False, False]).reset_index(drop=True)


def write_notes(summary: pd.DataFrame) -> None:
    eligible = summary[summary["formal_gate_pass"]]
    winner = eligible.iloc[0] if not eligible.empty else summary.iloc[0]
    lines = [
        "# V1 Market View 月末取样消融结果",
        "",
        "## 结论",
        "",
        f"- 2018-2024 发现期首选：`{winner['policy_id']}`。",
        "- 2025+ 仅作冻结后的样本外展示，不参与策略排序。",
        "- 对比月内全部日卡片、最后一张日卡片、最近若干日和按观点对去重/持久度加权。",
        "- 无论采用哪种模式，月末形成的目标权重仍滞后一月执行。",
        "",
        "| 策略 | 门槛 | 选择分 | 最差开发期 Sharpe 差 | 低/中/高全样本 Sharpe | 低/中/高 2025+ Sharpe |",
        "|---|---|---:|---:|---|---|",
    ]
    for _, row in summary.head(9).iterrows():
        full = "/".join(f"{row[f'{p}_full_sharpe']:.3f}" for p in PROFILES)
        holdout = "/".join(f"{row[f'{p}_holdout_sharpe']:.3f}" for p in PROFILES)
        lines.append(
            f"| {row['policy_id']} | {'通过' if row['formal_gate_pass'] else '未通过'} | {row['selection_score']:.4f} | "
            f"{row['worst_dev_fold_sharpe_delta']:.4f} | {full} | {holdout} |"
        )
    NOTES_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Search causal Market View preprocessing using 2018-2024 only.")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    TEMP_DATA.mkdir(parents=True, exist_ok=True)
    TEMP_REPORTS.mkdir(parents=True, exist_ok=True)
    jobs = [(spec, profile) for spec in build_policies() for profile in PROFILES]
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = {pool.submit(run_case, copy.deepcopy(spec), profile): (spec["policy_id"], profile) for spec, profile in jobs}
        for index, future in enumerate(as_completed(futures), start=1):
            policy_id, profile = futures[future]
            row = future.result()
            rows.append(row)
            print(f"[{index:02d}/{len(jobs)}] {policy_id} {profile}: {row['status']}", flush=True)
    detail = pd.DataFrame(rows).sort_values(["policy_id", "profile"]).reset_index(drop=True)
    detail.to_csv(DETAIL_PATH, index=False, encoding="utf-8-sig")
    summary = score_policies(detail)
    summary.to_csv(POLICY_PATH, index=False, encoding="utf-8-sig")
    write_notes(summary)
    shutil.rmtree(TEMP_DATA, ignore_errors=True)
    shutil.rmtree(TEMP_REPORTS, ignore_errors=True)
    print("\nTop policies (selection uses 2018-2024 only):")
    print(summary.head(9).to_string(index=False))
    print(f"\nDetail: {DETAIL_PATH.relative_to(ROOT)}")
    print(f"Policy summary: {POLICY_PATH.relative_to(ROOT)}")
    print(f"Notes: {NOTES_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

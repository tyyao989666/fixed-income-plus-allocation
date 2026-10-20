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
TEMP_DATA = ROOT / "data_processed" / "view_application_search"
TEMP_REPORTS = ROOT / "output" / "reports" / "view_application_search"
DETAIL_PATH = ROOT / "output" / "reports" / "v1_view_application_search_detail.csv"
POLICY_PATH = ROOT / "output" / "reports" / "v1_view_application_search_policy_summary.csv"
NOTES_PATH = ROOT / "output" / "reports" / "v1_view_application_search_summary.md"


def build_policies() -> list[dict[str, Any]]:
    return [
        {"policy_id": "baseline_both", "direct": True, "ep": True, "ep_scale": 1.00, "direct_scale": 1.00},
        {"policy_id": "both_half_direct", "direct": True, "ep": True, "ep_scale": 1.00, "direct_scale": 0.50},
        {"policy_id": "ep_only_050", "direct": False, "ep": True, "ep_scale": 0.50, "direct_scale": 0.00},
        {"policy_id": "ep_only_075", "direct": False, "ep": True, "ep_scale": 0.75, "direct_scale": 0.00},
        {"policy_id": "ep_only_100", "direct": False, "ep": True, "ep_scale": 1.00, "direct_scale": 0.00},
        {"policy_id": "ep_only_125", "direct": False, "ep": True, "ep_scale": 1.25, "direct_scale": 0.00},
        {"policy_id": "ep_only_150", "direct": False, "ep": True, "ep_scale": 1.50, "direct_scale": 0.00},
        {"policy_id": "direct_only_050", "direct": True, "ep": False, "ep_scale": 0.00, "direct_scale": 0.50},
        {"policy_id": "direct_only_100", "direct": True, "ep": False, "ep_scale": 0.00, "direct_scale": 1.00},
        {"policy_id": "no_market_view", "direct": False, "ep": False, "ep_scale": 0.00, "direct_scale": 0.00},
    ]


def configure_case(spec: dict[str, Any], profile: str) -> dict[str, Any]:
    cfg = load_config()
    cfg["strategy_version"] = "no_agent"
    cfg["product"]["selected_profile"] = profile
    cfg["signal_overlays"]["itransformer_rank"]["enabled"] = False
    cfg["signal_overlays"]["asset_itransformer"]["enabled"] = False
    cfg["signal_overlays"]["view_reliability"]["enabled"] = False
    application = {
        "direct_tilt_enabled": bool(spec["direct"]),
        "ep_mean_enabled": bool(spec["ep"]),
        "direct_tilt_scale": float(spec["direct_scale"]),
    }
    cfg["signal_overlays"]["view_application"] = copy.deepcopy(application)
    cfg["product"]["profiles"][profile].setdefault("signal_overlay_overrides", {})["view_application"] = copy.deepcopy(application)
    current_ep_scale = float(cfg["ep_skfolio"].get("view_mu_scale", 0.004))
    cfg["ep_skfolio"]["view_mu_scale"] = current_ep_scale * float(spec["ep_scale"])
    prefix = f"{spec['policy_id']}__{profile}"
    cfg["outputs"] = {
        "weights": f"data_processed/view_application_search/{prefix}_weights.parquet",
        "sleeve_weights": f"data_processed/view_application_search/{prefix}_sleeves.parquet",
        "risk_budget_base": f"data_processed/view_application_search/{prefix}_risk.parquet",
        "nav": f"output/reports/view_application_search/{prefix}_nav.csv",
        "metrics": f"output/reports/view_application_search/{prefix}_metrics.csv",
        "data_quality": f"output/reports/view_application_search/{prefix}_data_quality.csv",
    }
    return cfg


def period_metrics(nav: pd.DataFrame, start: str, end: str) -> dict[str, float]:
    sub = nav[pd.to_datetime(nav["date"]).between(pd.Timestamp(start), pd.Timestamp(end))].copy()
    if sub.empty:
        return {k: np.nan for k in ["annual_return", "annual_vol", "sharpe", "max_drawdown", "avg_turnover"]}
    sub["nav"] = (1.0 + sub["net_return"]).cumprod()
    return calc_metrics(sub).set_index("metric")["value"].to_dict()


def run_case(spec: dict[str, Any], profile: str) -> dict[str, Any]:
    row: dict[str, Any] = {"policy_id": spec["policy_id"], "profile": profile, "spec_json": json.dumps(spec, sort_keys=True)}
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
    ok = detail[detail["status"].eq("ok")]
    baseline = ok[ok["policy_id"].eq("baseline_both")].set_index("profile")
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
            ds = float(item["development_2018_2024_sharpe"] - base["development_2018_2024_sharpe"])
            dr = float(item["development_2018_2024_annual_return"] - base["development_2018_2024_annual_return"])
            dd = float(item["development_2018_2024_max_drawdown"] - base["development_2018_2024_max_drawdown"])
            dt = float(item["development_2018_2024_avg_turnover"] - base["development_2018_2024_avg_turnover"])
            fold_deltas = [float(item[f"{fold}_sharpe"] - base[f"{fold}_sharpe"]) for fold in DEV_FOLDS]
            score += importance[profile] * (
                ds + 0.35 * float(np.mean(fold_deltas)) + 3.0 * dr + 1.5 * dd
                - 0.15 * max(dt, 0.0) - 0.35 * max(-min(fold_deltas) - 0.10, 0.0)
            )
            worst_fold = min(worst_fold, min(fold_deltas))
            values = {
                "dev_sharpe_delta": ds,
                "dev_return_delta": dr,
                "worst_fold_sharpe_delta": min(fold_deltas),
                "holdout_return": float(item["holdout_2025_plus_annual_return"]),
                "holdout_sharpe": float(item["holdout_2025_plus_sharpe"]),
                "full_return": float(item["full_annual_return"]),
                "full_vol": float(item["full_annual_vol"]),
                "full_sharpe": float(item["full_sharpe"]),
                "full_drawdown": float(item["full_max_drawdown"]),
                "full_turnover": float(item["full_avg_turnover"]),
            }
            record.update({f"{profile}_{name}": value for name, value in values.items()})
        vol_order = record["low_vol_full_vol"] < record["balanced_full_vol"] < record["high_vol_full_vol"]
        gate = bool(vol_order and worst_fold >= -0.10 and all(record[f"{p}_dev_sharpe_delta"] >= -0.02 for p in PROFILES))
        if not vol_order:
            score -= 2.0
        record.update({"selection_score": float(score), "worst_dev_fold_sharpe_delta": float(worst_fold), "volatility_order_ok": bool(vol_order), "formal_gate_pass": gate})
        rows.append(record)
    return pd.DataFrame(rows).sort_values(["formal_gate_pass", "selection_score"], ascending=[False, False]).reset_index(drop=True)


def write_notes(summary: pd.DataFrame) -> None:
    eligible = summary[summary["formal_gate_pass"]]
    winner = eligible.iloc[0] if not eligible.empty else summary.iloc[0]
    lines = [
        "# V1 Market View 传导通道消融结果",
        "",
        "## 结论",
        "",
        f"- 2018-2024 发现期首选：`{winner['policy_id']}`。",
        "- 对比 direct sleeve tilt 与 EP posterior mean 两条通道，检查同一 view 是否应被双重使用。",
        "- 2025+ 仅作冻结验收，不参与排序。",
        "",
        "| 策略 | 门槛 | 选择分 | 最差开发期 Sharpe 差 | 低/中/高全样本 Sharpe | 低/中/高 2025+ Sharpe |",
        "|---|---|---:|---:|---|---|",
    ]
    for _, row in summary.head(10).iterrows():
        full = "/".join(f"{row[f'{p}_full_sharpe']:.3f}" for p in PROFILES)
        holdout = "/".join(f"{row[f'{p}_holdout_sharpe']:.3f}" for p in PROFILES)
        lines.append(f"| {row['policy_id']} | {'通过' if row['formal_gate_pass'] else '未通过'} | {row['selection_score']:.4f} | {row['worst_dev_fold_sharpe_delta']:.4f} | {full} | {holdout} |")
    NOTES_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ablate direct and EP Market View transmission channels on 2018-2024.")
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
    print(summary.head(10).to_string(index=False))
    print(f"\nDetail: {DETAIL_PATH.relative_to(ROOT)}")
    print(f"Policy summary: {POLICY_PATH.relative_to(ROOT)}")
    print(f"Notes: {NOTES_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

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
TEMP_DATA = ROOT / "data_processed" / "risk_on_rotation_search"
TEMP_REPORTS = ROOT / "output" / "reports" / "risk_on_rotation_search"
DETAIL_PATH = ROOT / "output" / "reports" / "v1_risk_on_rotation_search_detail.csv"
POLICY_PATH = ROOT / "output" / "reports" / "v1_risk_on_rotation_search_policy_summary.csv"
NOTES_PATH = ROOT / "output" / "reports" / "v1_risk_on_rotation_search_summary.md"


def policy(
    policy_id: str,
    caps: list[float],
    lookbacks: list[int] = [6, 12],
    require_all: bool = True,
    top_n: int = 1,
    momentum_scale: float = 0.10,
    threshold: float = 0.0,
) -> dict[str, Any]:
    return {
        "policy_id": policy_id,
        "enabled": policy_id != "baseline_disabled",
        "profile_max_tilt": dict(zip(PROFILES, caps)),
        "lookbacks": lookbacks,
        "require_all": require_all,
        "top_n": top_n,
        "momentum_scale": momentum_scale,
        "threshold": threshold,
    }


def build_policies() -> list[dict[str, Any]]:
    return [
        policy("baseline_disabled", [0.0, 0.0, 0.0]),
        policy("dual_6_12_tiny", [0.0, 0.010, 0.020]),
        policy("dual_6_12_small", [0.0, 0.015, 0.035]),
        policy("dual_6_12_medium", [0.0, 0.020, 0.050]),
        policy("dual_6_12_large", [0.0, 0.025, 0.065]),
        policy("dual_6_12_all_profiles", [0.005, 0.020, 0.050]),
        policy("dual_3_6_small", [0.0, 0.015, 0.035], [3, 6]),
        policy("dual_3_6_medium", [0.0, 0.020, 0.050], [3, 6]),
        policy("dual_3_6_12_medium", [0.0, 0.020, 0.050], [3, 6, 12]),
        policy("average_6_12_small", [0.0, 0.015, 0.035], [6, 12], False),
        policy("dual_6_12_top2", [0.0, 0.020, 0.050], [6, 12], True, 2),
        policy("dual_6_12_fast_scale", [0.0, 0.020, 0.050], [6, 12], True, 1, 0.05),
        policy("dual_6_12_slow_scale", [0.0, 0.020, 0.050], [6, 12], True, 1, 0.15),
        policy("dual_6_12_confirm_1pct", [0.0, 0.020, 0.050], [6, 12], True, 1, 0.10, 0.01),
    ]


def configure_case(spec: dict[str, Any], profile: str) -> dict[str, Any]:
    cfg = load_config()
    cfg["strategy_version"] = "no_agent"
    cfg["product"]["selected_profile"] = profile
    cfg["signal_overlays"]["itransformer_rank"]["enabled"] = False
    cfg["signal_overlays"]["asset_itransformer"]["enabled"] = False
    cfg["signal_overlays"]["risk_on_rotation"].update(
        {
            "enabled": bool(spec["enabled"]),
            "lookback_months": list(spec["lookbacks"]),
            "require_all_lookbacks_positive": bool(spec["require_all"]),
            "positive_threshold": float(spec["threshold"]),
            "top_n": int(spec["top_n"]),
            "momentum_scale": float(spec["momentum_scale"]),
            "profile_max_tilt": copy.deepcopy(spec["profile_max_tilt"]),
        }
    )
    prefix = f"{spec['policy_id']}__{profile}"
    cfg["outputs"] = {
        "weights": f"data_processed/risk_on_rotation_search/{prefix}_weights.parquet",
        "sleeve_weights": f"data_processed/risk_on_rotation_search/{prefix}_sleeves.parquet",
        "risk_budget_base": f"data_processed/risk_on_rotation_search/{prefix}_risk.parquet",
        "nav": f"output/reports/risk_on_rotation_search/{prefix}_nav.csv",
        "metrics": f"output/reports/risk_on_rotation_search/{prefix}_metrics.csv",
        "data_quality": f"output/reports/risk_on_rotation_search/{prefix}_data_quality.csv",
    }
    return cfg


def period_metrics(nav: pd.DataFrame, start: str, end: str) -> dict[str, float]:
    sub = nav[pd.to_datetime(nav["date"]).between(pd.Timestamp(start), pd.Timestamp(end))].copy()
    if sub.empty:
        return {"annual_return": np.nan, "annual_vol": np.nan, "sharpe": np.nan, "max_drawdown": np.nan, "avg_turnover": np.nan}
    sub["nav"] = (1.0 + sub["net_return"]).cumprod()
    return calc_metrics(sub).set_index("metric")["value"].to_dict()


def run_case(spec: dict[str, Any], profile: str) -> dict[str, Any]:
    row: dict[str, Any] = {"policy_id": spec["policy_id"], "profile": profile, "spec_json": json.dumps(spec, ensure_ascii=False, sort_keys=True)}
    try:
        cfg = configure_case(spec, profile)
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
        row["mean_risky_weight"] = float((sleeves["convertible_bonds"] + sleeves["equity"] + sleeves["alternatives"]).mean())
        row["status"] = "ok"
    except Exception as exc:  # noqa: BLE001
        row["status"] = "error"
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def add_scores(detail: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ok = detail[detail["status"].eq("ok")].copy()
    baseline = ok[ok["policy_id"].eq("baseline_disabled")].set_index("profile")
    importance = {"low_vol": 0.15, "balanced": 0.35, "high_vol": 0.50}
    fold_cols = [f"{name}_sharpe" for name in DEV_FOLDS]
    rows: list[dict[str, Any]] = []
    for policy_id, group in ok.groupby("policy_id"):
        if set(group["profile"]) != set(PROFILES):
            continue
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
            profile_score = ds + 0.30 * np.mean(folds) + 5.0 * dr + 1.5 * dd - 0.15 * max(dt, 0.0) - 0.40 * max(-min(folds) - 0.10, 0.0)
            score += importance[profile] * profile_score
            summary.update(
                {
                    f"{profile}_dev_return_delta": dr,
                    f"{profile}_dev_sharpe_delta": ds,
                    f"{profile}_worst_fold_sharpe_delta": min(folds),
                    f"{profile}_holdout_return": float(item["holdout_2025_plus_annual_return"]),
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
        rows.append({"policy_id": policy_id, "selection_score": score, "worst_dev_fold_sharpe_delta": worst, "volatility_order_ok": vol_order, **summary})
    policies = pd.DataFrame(rows).sort_values(["selection_score", "worst_dev_fold_sharpe_delta"], ascending=False).reset_index(drop=True)
    return detail.merge(policies[["policy_id", "selection_score"]], on="policy_id", how="left"), policies


def write_notes(detail: pd.DataFrame, policies: pd.DataFrame) -> None:
    best = policies.iloc[0]
    selected_id = str(best["policy_id"])
    selected = detail[detail["policy_id"].eq(selected_id)].set_index("profile")
    baseline = detail[detail["policy_id"].eq("baseline_disabled")].set_index("profile")
    robust = bool(
        selected_id != "baseline_disabled"
        and best["worst_dev_fold_sharpe_delta"] >= -0.10
        and selected.loc["low_vol", "development_2018_2024_sharpe"] >= baseline.loc["low_vol", "development_2018_2024_sharpe"] - 0.02
        and selected.loc["balanced", "development_2018_2024_sharpe"] >= baseline.loc["balanced", "development_2018_2024_sharpe"]
        and selected.loc["high_vol", "development_2018_2024_sharpe"] >= baseline.loc["high_vol", "development_2018_2024_sharpe"]
    )
    lines = [
        "# V1 风险进攻轮动消融结果",
        "",
        "## 结论",
        "",
        f"- 发现期最优策略：`{selected_id}`。",
        f"- 是否达到自动转正式门槛：`{'是' if robust else '否'}`。",
        "- 参数选择只使用 2018-2024；2025+ 只披露，不参与排序。",
        "- 该层只在 6/12 月趋势共同确认后增加风险，不替代现有风险刹车。",
        "",
        "| 产品 | 发现期基线收益 | 发现期候选收益 | 发现期基线 Sharpe | 发现期候选 Sharpe | 2025+ 候选收益 | 2025+ 候选 Sharpe | 全样本 Sharpe |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for profile in PROFILES:
        b = baseline.loc[profile]
        s = selected.loc[profile]
        lines.append(
            f"| {profile} | {b['development_2018_2024_annual_return']:.2%} | {s['development_2018_2024_annual_return']:.2%} | "
            f"{b['development_2018_2024_sharpe']:.3f} | {s['development_2018_2024_sharpe']:.3f} | "
            f"{s['holdout_2025_plus_annual_return']:.2%} | {s['holdout_2025_plus_sharpe']:.3f} | {s['full_sharpe']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## 门槛",
            "",
            "balanced 和 high_vol 的发现期 Sharpe 均不得下降，low_vol 最多下降 0.02，任一开发阶段相对基线不得恶化超过 0.10，且风险层级保持 low < balanced < high。未通过则正式配置保持关闭。",
        ]
    )
    NOTES_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Search causal risk-on sleeve rotation policies.")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--keep-temp", action="store_true")
    args = parser.parse_args()
    TEMP_DATA.mkdir(parents=True, exist_ok=True)
    TEMP_REPORTS.mkdir(parents=True, exist_ok=True)
    specs = build_policies()
    cases = [(spec, profile) for spec in specs for profile in PROFILES]
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(run_case, spec, profile): (spec["policy_id"], profile) for spec, profile in cases}
        for index, future in enumerate(as_completed(futures), start=1):
            policy_id, profile = futures[future]
            row = future.result()
            rows.append(row)
            print(f"[{index:02d}/{len(cases)}] {policy_id} {profile}: {row['status']}", flush=True)
    detail = pd.DataFrame(rows)
    errors = detail[detail["status"].ne("ok")]
    if not errors.empty:
        raise RuntimeError(errors[["policy_id", "profile", "error"]].to_string(index=False))
    detail, policies = add_scores(detail)
    detail.sort_values(["selection_score", "policy_id", "profile"], ascending=[False, True, True]).to_csv(DETAIL_PATH, index=False, encoding="utf-8-sig")
    policies.to_csv(POLICY_PATH, index=False, encoding="utf-8-sig")
    write_notes(detail, policies)
    print("\nTop policies (selection uses 2018-2024 only):")
    print(policies.head(8).to_string(index=False))
    print(f"\nDetail: {DETAIL_PATH.relative_to(ROOT)}")
    print(f"Policy summary: {POLICY_PATH.relative_to(ROOT)}")
    print(f"Notes: {NOTES_PATH.relative_to(ROOT)}")
    if not args.keep_temp:
        shutil.rmtree(TEMP_DATA, ignore_errors=True)
        shutil.rmtree(TEMP_REPORTS, ignore_errors=True)


if __name__ == "__main__":
    main()

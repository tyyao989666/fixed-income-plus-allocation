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
TEMP_DATA = ROOT / "data_processed" / "asset_momentum_search"
TEMP_REPORTS = ROOT / "output" / "reports" / "asset_momentum_search"
DETAIL_PATH = ROOT / "output" / "reports" / "v1_asset_momentum_search_detail.csv"
POLICY_PATH = ROOT / "output" / "reports" / "v1_asset_momentum_search_policy_summary.csv"
NOTES_PATH = ROOT / "output" / "reports" / "v1_asset_momentum_search_summary.md"


def policy(
    policy_id: str,
    modules: list[str],
    lookbacks: list[int],
    method: str,
    blends: list[float],
    concentration: float = 2.0,
    negative_multiplier: float = 0.50,
    max_asset_share: float = 0.75,
) -> dict[str, Any]:
    return {
        "policy_id": policy_id,
        "enabled": policy_id != "baseline_disabled",
        "modules": modules,
        "lookbacks": lookbacks,
        "method": method,
        "profile_blend": dict(zip(PROFILES, blends)),
        "concentration": concentration,
        "negative_multiplier": negative_multiplier,
        "max_asset_share": max_asset_share,
    }


def build_policies() -> list[dict[str, Any]]:
    eq_alt = ["equity", "alternatives"]
    eq_cb_alt = ["convertible_bonds", "equity", "alternatives"]
    rates_eq_alt = ["rates", "equity", "alternatives"]
    all_active = ["rates", "convertible_bonds", "equity", "alternatives"]
    return [
        policy("baseline_disabled", [], [6, 12], "raw_momentum", [0.0, 0.0, 0.0]),
        policy("raw_eq_alt_6_12_conservative", eq_alt, [6, 12], "raw_momentum", [0.02, 0.10, 0.18]),
        policy("raw_eq_alt_6_12", eq_alt, [6, 12], "raw_momentum", [0.05, 0.15, 0.25]),
        policy("raw_eq_alt_6_12_strong", eq_alt, [6, 12], "raw_momentum", [0.06, 0.22, 0.38]),
        policy("raw_eq_alt_3_6", eq_alt, [3, 6], "raw_momentum", [0.04, 0.14, 0.24]),
        policy("raw_eq_alt_3_6_12", eq_alt, [3, 6, 12], "raw_momentum", [0.05, 0.15, 0.25]),
        policy("raw_eq_alt_slow_rank", eq_alt, [6, 12], "raw_momentum", [0.05, 0.15, 0.25], concentration=1.0),
        policy("raw_eq_alt_fast_rank", eq_alt, [6, 12], "raw_momentum", [0.05, 0.15, 0.25], concentration=3.0),
        policy("raw_eq_alt_hard_negative", eq_alt, [6, 12], "raw_momentum", [0.05, 0.15, 0.25], negative_multiplier=0.25),
        policy("raw_eq_alt_soft_negative", eq_alt, [6, 12], "raw_momentum", [0.05, 0.15, 0.25], negative_multiplier=0.75),
        policy("raw_rates_eq_alt", rates_eq_alt, [6, 12], "raw_momentum", [0.04, 0.14, 0.24]),
        policy("raw_eq_cb_alt", eq_cb_alt, [6, 12], "raw_momentum", [0.04, 0.14, 0.24]),
        policy("raw_all_active", all_active, [6, 12], "raw_momentum", [0.04, 0.14, 0.24]),
        policy("risk_adj_eq_alt", eq_alt, [6, 12], "risk_adjusted_momentum", [0.04, 0.14, 0.24]),
        policy("risk_adj_rates_eq_alt", rates_eq_alt, [6, 12], "risk_adjusted_momentum", [0.04, 0.14, 0.24]),
        policy("risk_adj_all_active", all_active, [6, 12], "risk_adjusted_momentum", [0.04, 0.14, 0.24]),
    ]


def configure_case(spec: dict[str, Any], profile: str) -> dict[str, Any]:
    cfg = load_config()
    cfg["strategy_version"] = "no_agent"
    cfg["product"]["selected_profile"] = profile
    cfg["signal_overlays"]["itransformer_rank"]["enabled"] = False
    cfg["signal_overlays"]["asset_itransformer"]["enabled"] = False
    overlay = cfg["signal_overlays"]["asset_momentum"]
    overlay.update(
        {
            "enabled": bool(spec["enabled"]),
            "enabled_profiles": PROFILES,
            "enabled_modules": list(spec["modules"]),
            "lookback_months": list(spec["lookbacks"]),
            "min_history_months": max(spec["lookbacks"]),
            "score_method": str(spec["method"]),
            "rank_concentration": float(spec["concentration"]),
            "negative_momentum_multiplier": float(spec["negative_multiplier"]),
            "profile_blend": copy.deepcopy(spec["profile_blend"]),
            "max_asset_share": float(spec["max_asset_share"]),
        }
    )
    prefix = f"{spec['policy_id']}__{profile}"
    cfg["outputs"] = {
        "weights": f"data_processed/asset_momentum_search/{prefix}_weights.parquet",
        "sleeve_weights": f"data_processed/asset_momentum_search/{prefix}_sleeves.parquet",
        "risk_budget_base": f"data_processed/asset_momentum_search/{prefix}_risk.parquet",
        "nav": f"output/reports/asset_momentum_search/{prefix}_nav.csv",
        "metrics": f"output/reports/asset_momentum_search/{prefix}_metrics.csv",
        "data_quality": f"output/reports/asset_momentum_search/{prefix}_data_quality.csv",
    }
    return cfg


def period_metrics(nav: pd.DataFrame, start: str, end: str) -> dict[str, float]:
    sub = nav[pd.to_datetime(nav["date"]).between(pd.Timestamp(start), pd.Timestamp(end))].copy()
    if sub.empty:
        return {
            "annual_return": np.nan,
            "annual_vol": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
            "avg_turnover": np.nan,
        }
    sub["nav"] = (1.0 + sub["net_return"]).cumprod()
    return calc_metrics(sub).set_index("metric")["value"].to_dict()


def run_case(spec: dict[str, Any], profile: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "policy_id": spec["policy_id"],
        "profile": profile,
        "spec_json": json.dumps(spec, ensure_ascii=False, sort_keys=True),
    }
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
        weights = pd.read_parquet(ROOT / cfg["outputs"]["weights"])
        weight_sum = weights.groupby("date")["weight"].sum()
        row["max_weight_sum_error"] = float((weight_sum - 1.0).abs().max())
        row["max_asset_weight"] = float(weights.loc[weights["asset_code"].ne("cash"), "weight"].max())
        row["cash_weight_max"] = float(weights.loc[weights["asset_code"].eq("cash"), "weight"].max()) if (weights["asset_code"] == "cash").any() else 0.0
        row["status"] = "ok"
    except Exception as exc:  # noqa: BLE001
        row["status"] = "error"
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def add_policy_scores(detail: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ok = detail[detail["status"].eq("ok")].copy()
    baseline = ok[ok["policy_id"].eq("baseline_disabled")].set_index("profile")
    fold_sharpe_cols = [f"{name}_sharpe" for name in DEV_FOLDS]
    rows: list[dict[str, Any]] = []
    profile_importance = {"low_vol": 0.20, "balanced": 0.35, "high_vol": 0.45}
    for policy_id, group in ok.groupby("policy_id"):
        if set(group["profile"]) != set(PROFILES):
            continue
        score = 0.0
        worst_fold_delta = np.inf
        profile_rows: dict[str, dict[str, float]] = {}
        for _, item in group.iterrows():
            profile = str(item["profile"])
            base = baseline.loc[profile]
            importance = profile_importance[profile]
            dev_sharpe_delta = float(item["development_2018_2024_sharpe"] - base["development_2018_2024_sharpe"])
            dev_return_delta = float(item["development_2018_2024_annual_return"] - base["development_2018_2024_annual_return"])
            dev_dd_delta = float(item["development_2018_2024_max_drawdown"] - base["development_2018_2024_max_drawdown"])
            turnover_delta = float(item["development_2018_2024_avg_turnover"] - base["development_2018_2024_avg_turnover"])
            fold_deltas = [float(item[col] - base[col]) for col in fold_sharpe_cols]
            worst_fold_delta = min(worst_fold_delta, min(fold_deltas))
            profile_score = (
                dev_sharpe_delta
                + 0.35 * float(np.mean(fold_deltas))
                + 3.0 * dev_return_delta
                + 1.5 * dev_dd_delta
                - 0.15 * max(turnover_delta, 0.0)
                - 0.35 * max(-min(fold_deltas) - 0.10, 0.0)
            )
            score += importance * profile_score
            profile_rows[profile] = {
                "dev_sharpe_delta": dev_sharpe_delta,
                "dev_return_delta": dev_return_delta,
                "worst_fold_sharpe_delta": min(fold_deltas),
                "holdout_sharpe": float(item["holdout_2025_plus_sharpe"]),
                "full_sharpe": float(item["full_sharpe"]),
                "full_return": float(item["full_annual_return"]),
                "full_drawdown": float(item["full_max_drawdown"]),
                "full_turnover": float(item["full_avg_turnover"]),
            }
        low = group[group["profile"].eq("low_vol")].iloc[0]
        balanced = group[group["profile"].eq("balanced")].iloc[0]
        high = group[group["profile"].eq("high_vol")].iloc[0]
        vol_order = bool(low["full_annual_vol"] < balanced["full_annual_vol"] < high["full_annual_vol"])
        if not vol_order:
            score -= 2.0
        rows.append(
            {
                "policy_id": policy_id,
                "selection_score": score,
                "worst_dev_fold_sharpe_delta": worst_fold_delta,
                "volatility_order_ok": vol_order,
                **{
                    f"{profile}_{metric}": value
                    for profile, values in profile_rows.items()
                    for metric, value in values.items()
                },
            }
        )
    policies = pd.DataFrame(rows).sort_values(
        ["selection_score", "worst_dev_fold_sharpe_delta"], ascending=False
    ).reset_index(drop=True)
    return detail.merge(policies[["policy_id", "selection_score"]], on="policy_id", how="left"), policies


def write_notes(detail: pd.DataFrame, policies: pd.DataFrame) -> None:
    best = policies.iloc[0]
    baseline = policies[policies["policy_id"].eq("baseline_disabled")].iloc[0]
    selected_id = str(best["policy_id"])
    selected = detail[detail["policy_id"].eq(selected_id)].set_index("profile")
    base_detail = detail[detail["policy_id"].eq("baseline_disabled")].set_index("profile")
    robust = bool(
        selected_id != "baseline_disabled"
        and best["worst_dev_fold_sharpe_delta"] >= -0.10
        and selected.loc["balanced", "development_2018_2024_sharpe"] >= base_detail.loc["balanced", "development_2018_2024_sharpe"]
        and selected.loc["high_vol", "development_2018_2024_sharpe"] >= base_detail.loc["high_vol", "development_2018_2024_sharpe"]
    )
    lines = [
        "# V1 袖套内资产动量消融结果",
        "",
        "## 结论",
        "",
        f"- 发现期最优策略：`{selected_id}`。",
        f"- 是否达到自动转正式门槛：`{'是' if robust else '否'}`。",
        "- 参数选择只使用 2018-2024；2025+ 仅作为最终留出期披露，不参与排序。",
        "- 所有信号只使用决策日及以前的月度收益，目标权重再滞后一月执行。",
        "",
        "## 三产品对比",
        "",
        "| 产品 | 发现期基线 Sharpe | 发现期候选 Sharpe | 2025+ 基线 Sharpe | 2025+ 候选 Sharpe | 全样本年化 | 全样本 Sharpe | 最大回撤 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for profile in PROFILES:
        s = selected.loc[profile]
        b = base_detail.loc[profile]
        lines.append(
            f"| {profile} | {b['development_2018_2024_sharpe']:.3f} | {s['development_2018_2024_sharpe']:.3f} | "
            f"{b['holdout_2025_plus_sharpe']:.3f} | {s['holdout_2025_plus_sharpe']:.3f} | "
            f"{s['full_annual_return']:.2%} | {s['full_sharpe']:.3f} | {s['full_max_drawdown']:.2%} |"
        )
    lines.extend(
        [
            "",
            "## 采用规则",
            "",
            "候选必须同时满足 balanced 和 high_vol 的 2018-2024 Sharpe 不低于基线，任一开发阶段 Sharpe 相对基线不得恶化超过 0.10，且三产品波动率顺序保持 low < balanced < high。未通过时正式配置继续保持关闭。",
            "",
            "## 说明",
            "",
            f"基线策略综合选择分为 {baseline['selection_score']:.4f}，最优候选为 {best['selection_score']:.4f}。该分数只用于参数筛选，不是可投资业绩指标。",
        ]
    )
    NOTES_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Search causal within-sleeve asset momentum overlays.")
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
        error_text = errors[["policy_id", "profile", "error"]].to_string(index=False)
        raise RuntimeError(f"Asset momentum search has failed cases:\n{error_text}")
    detail, policies = add_policy_scores(detail)
    DETAIL_PATH.parent.mkdir(parents=True, exist_ok=True)
    detail.sort_values(["selection_score", "policy_id", "profile"], ascending=[False, True, True]).to_csv(
        DETAIL_PATH, index=False, encoding="utf-8-sig"
    )
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

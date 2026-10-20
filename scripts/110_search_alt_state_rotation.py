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
TEMP_DATA = ROOT / "data_processed" / "alt_state_rotation_search"
TEMP_REPORTS = ROOT / "output" / "reports" / "alt_state_rotation_search"
DETAIL_PATH = ROOT / "output" / "reports" / "v1_alt_state_rotation_search_detail.csv"
POLICY_PATH = ROOT / "output" / "reports" / "v1_alt_state_rotation_search_policy_summary.csv"
NOTES_PATH = ROOT / "output" / "reports" / "v1_alt_state_rotation_search_summary.md"


ALT1_DEFAULT = {
    "alt_gold": 0.62,
    "alt_us_bond": 0.18,
    "alt_reits": 0.10,
    "alt_commodity": 0.04,
    "alt_nonferrous": 0.03,
    "alt_chemical": 0.03,
}
ALT1_HEDGE = {
    "alt_gold": 0.65,
    "alt_us_bond": 0.20,
    "alt_reits": 0.10,
    "alt_commodity": 0.02,
    "alt_nonferrous": 0.02,
    "alt_chemical": 0.01,
}
ALT1_GOLD = {
    "alt_gold": 0.75,
    "alt_us_bond": 0.15,
    "alt_reits": 0.06,
    "alt_commodity": 0.02,
    "alt_nonferrous": 0.01,
    "alt_chemical": 0.01,
}


def policy(policy_id: str, strengths: list[float], alt1: dict[str, float] | None = None) -> dict[str, Any]:
    return {
        "policy_id": policy_id,
        "enabled": policy_id != "baseline_disabled",
        "profile_strength": dict(zip(PROFILES, strengths)),
        "alt1": copy.deepcopy(alt1 or ALT1_DEFAULT),
    }


def build_policies() -> list[dict[str, Any]]:
    return [
        policy("baseline_disabled", [0.0, 0.0, 0.0]),
        policy("alt1_uniform_05", [0.05, 0.05, 0.05]),
        policy("alt1_uniform_10", [0.10, 0.10, 0.10]),
        policy("alt1_uniform_15", [0.15, 0.15, 0.15]),
        policy("alt1_uniform_20", [0.20, 0.20, 0.20]),
        policy("alt1_uniform_30", [0.30, 0.30, 0.30]),
        policy("alt1_profile_soft", [0.05, 0.10, 0.15]),
        policy("alt1_profile_medium", [0.05, 0.15, 0.25]),
        policy("alt1_profile_strong", [0.10, 0.20, 0.35]),
        policy("alt1_hedge_profile", [0.05, 0.15, 0.25], ALT1_HEDGE),
        policy("alt1_gold_profile", [0.05, 0.15, 0.25], ALT1_GOLD),
    ]


def configure_case(spec: dict[str, Any], profile: str) -> dict[str, Any]:
    cfg = load_config()
    cfg["strategy_version"] = "no_agent"
    cfg["product"]["selected_profile"] = profile
    cfg["signal_overlays"]["itransformer_rank"]["enabled"] = False
    cfg["signal_overlays"]["asset_itransformer"]["enabled"] = False
    cfg["signal_overlays"]["risk_on_rotation"]["enabled"] = False
    rule = cfg["signal_overlays"]["rule_output"]
    rule["apply_alt_internal_overlay"] = bool(spec["enabled"])
    rule["alt_overlay_strength"] = float(spec["profile_strength"][profile])
    rule["alt_state_weights"]["ALT1"] = copy.deepcopy(spec["alt1"])
    profile_rule = cfg["product"]["profiles"][profile].setdefault("signal_overlay_overrides", {}).setdefault("rule_output", {})
    profile_rule.update(
        {
            "apply_alt_internal_overlay": bool(spec["enabled"]),
            "alt_overlay_strength": float(spec["profile_strength"][profile]),
            "alt_state_weights": copy.deepcopy(rule["alt_state_weights"]),
        }
    )
    prefix = f"{spec['policy_id']}__{profile}"
    cfg["outputs"] = {
        "weights": f"data_processed/alt_state_rotation_search/{prefix}_weights.parquet",
        "sleeve_weights": f"data_processed/alt_state_rotation_search/{prefix}_sleeves.parquet",
        "risk_budget_base": f"data_processed/alt_state_rotation_search/{prefix}_risk.parquet",
        "nav": f"output/reports/alt_state_rotation_search/{prefix}_nav.csv",
        "metrics": f"output/reports/alt_state_rotation_search/{prefix}_metrics.csv",
        "data_quality": f"output/reports/alt_state_rotation_search/{prefix}_data_quality.csv",
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
        weights = pd.read_parquet(ROOT / cfg["outputs"]["weights"])
        row["max_weight_sum_error"] = float((weights.groupby("date")["weight"].sum() - 1.0).abs().max())
        row["cash_weight_max"] = float(weights.loc[weights["asset_code"].eq("cash"), "weight"].max()) if (weights["asset_code"] == "cash").any() else 0.0
        row["status"] = "ok"
    except Exception as exc:  # noqa: BLE001
        row["status"] = "error"
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def add_scores(detail: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ok = detail[detail["status"].eq("ok")].copy()
    baseline = ok[ok["policy_id"].eq("baseline_disabled")].set_index("profile")
    importance = {"low_vol": 0.20, "balanced": 0.35, "high_vol": 0.45}
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
            profile_score = ds + 0.35 * np.mean(folds) + 3.0 * dr + 1.5 * dd - 0.15 * max(dt, 0.0) - 0.35 * max(-min(folds) - 0.10, 0.0)
            score += importance[profile] * profile_score
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
        and all(selected.loc[p, "development_2018_2024_sharpe"] >= baseline.loc[p, "development_2018_2024_sharpe"] for p in PROFILES)
    )
    lines = [
        "# V1 另类状态轮动消融结果",
        "",
        "## 结论",
        "",
        f"- 发现期最优策略：`{selected_id}`。",
        f"- 是否达到自动转正式门槛：`{'是' if robust else '否'}`。",
        "- 月末状态样本中只有 ALT1/ALT2；因此本轮只验证 ALT1 防守模板，ALT2 保持原权重，未对未出现的 ALT3/ALT4 事后拟合。",
        "- 参数选择只使用 2018-2024；2025+ 不参与排序。",
        "",
        "| 产品 | 发现期基线 Sharpe | 发现期候选 Sharpe | 2025+ 基线 Sharpe | 2025+ 候选 Sharpe | 全样本收益 | 全样本 Sharpe | 最大回撤 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for profile in PROFILES:
        b = baseline.loc[profile]
        s = selected.loc[profile]
        lines.append(
            f"| {profile} | {b['development_2018_2024_sharpe']:.3f} | {s['development_2018_2024_sharpe']:.3f} | "
            f"{b['holdout_2025_plus_sharpe']:.3f} | {s['holdout_2025_plus_sharpe']:.3f} | "
            f"{s['full_annual_return']:.2%} | {s['full_sharpe']:.3f} | {s['full_max_drawdown']:.2%} |"
        )
    lines.extend(["", "## 门槛", "", "三个产品的发现期 Sharpe 均不得下降，任一开发阶段相对基线不得恶化超过 0.10，且风险层级保持 low < balanced < high。未通过则正式状态轮动继续关闭。"])
    NOTES_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Search causal alternative state rotations.")
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

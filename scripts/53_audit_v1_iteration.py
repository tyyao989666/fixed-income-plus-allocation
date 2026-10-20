from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "output" / "reports"
PROCESSED = ROOT / "data_processed"
PROFILES = ["low_vol", "balanced", "high_vol"]
SLEEVES = ["rates", "credit", "convertible_bonds", "equity", "alternatives", "cash"]


def metric_map(profile: str) -> dict[str, float]:
    df = pd.read_csv(REPORTS / f"v1_{profile}_backtest_metrics.csv")
    return dict(zip(df["metric"], df["value"]))


def load_config() -> dict[str, Any]:
    with (ROOT / "config" / "v1_strategy_config.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def max_drawdown_info(nav: pd.DataFrame) -> dict[str, Any]:
    nav = nav.copy()
    nav["drawdown"] = nav["nav"] / nav["nav"].cummax() - 1.0
    idx = int(nav["drawdown"].idxmin())
    return {
        "max_drawdown_date": str(pd.to_datetime(nav.loc[idx, "date"]).date()),
        "max_drawdown": float(nav.loc[idx, "drawdown"]),
        "max_drawdown_month_return": float(nav.loc[idx, "net_return"]),
    }


def sleeve_bound_violations(profile: str, sleeve_weights: pd.DataFrame, cfg: dict[str, Any]) -> int:
    bounds = cfg["product"]["profiles"][profile]["sleeve_bounds"]
    count = 0
    for sleeve in SLEEVES:
        lo, hi = map(float, bounds[sleeve])
        s = pd.to_numeric(sleeve_weights[sleeve], errors="coerce")
        count += int((s.lt(lo - 1e-8) | s.gt(hi + 1e-8)).sum())
    return count


def audit_profile(profile: str, label: str, cfg: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    metrics = metric_map(profile)
    nav = pd.read_csv(REPORTS / f"v1_{profile}_backtest_nav.csv", parse_dates=["date"])
    weights = pd.read_parquet(PROCESSED / f"v1_{profile}_target_weights_monthly.parquet")
    sleeves = pd.read_parquet(PROCESSED / f"v1_{profile}_sleeve_weights_monthly.parquet")
    attr = pd.read_csv(REPORTS / f"v1_{profile}_attribution_by_sleeve.csv")
    monthly_attr = pd.read_csv(REPORTS / f"v1_{profile}_monthly_attribution.csv", parse_dates=["date"])

    weight_sum = weights.groupby("date")["weight"].sum()
    cash = weights.loc[weights["asset_code"].eq("cash"), "weight"] if "asset_code" in weights else pd.Series(dtype=float)
    worst = nav.nsmallest(1, "net_return").iloc[0]
    worst_attr = monthly_attr.loc[monthly_attr["date"].eq(worst["date"]), SLEEVES].sum().sort_values()
    avg_sleeves = sleeves[SLEEVES].mean()
    max_sleeves = sleeves[SLEEVES].max()
    min_sleeves = sleeves[SLEEVES].min()
    top_positive = attr.sort_values("total_contribution", ascending=False).iloc[0]
    top_negative = attr.sort_values("total_contribution", ascending=True).iloc[0]

    row = {
        "label": label,
        "profile": profile,
        "months": metrics.get("months", np.nan),
        "annual_return": metrics.get("annual_return", np.nan),
        "annual_vol": metrics.get("annual_vol", np.nan),
        "sharpe": metrics.get("sharpe", np.nan),
        "max_drawdown_metric": metrics.get("max_drawdown", np.nan),
        "calmar": metrics.get("calmar", np.nan),
        "avg_turnover": metrics.get("avg_turnover", np.nan),
        "weight_sum_min": float(weight_sum.min()),
        "weight_sum_max": float(weight_sum.max()),
        "cash_max": float(cash.max()) if len(cash) else 0.0,
        "sleeve_bound_violations": sleeve_bound_violations(profile, sleeves, cfg),
        "worst_month": str(pd.to_datetime(worst["date"]).date()),
        "worst_month_return": float(worst["net_return"]),
        "worst_month_main_drag": str(worst_attr.index[0]),
        "worst_month_main_drag_contribution": float(worst_attr.iloc[0]),
        "top_positive_sleeve": str(top_positive["module"]),
        "top_positive_contribution": float(top_positive["total_contribution"]),
        "top_negative_sleeve": str(top_negative["module"]),
        "top_negative_contribution": float(top_negative["total_contribution"]),
    }
    row.update(max_drawdown_info(nav))
    for sleeve in SLEEVES:
        row[f"avg_{sleeve}"] = float(avg_sleeves[sleeve])
        row[f"min_{sleeve}"] = float(min_sleeves[sleeve])
        row[f"max_{sleeve}"] = float(max_sleeves[sleeve])

    checks = []
    if row["weight_sum_min"] < 1.0 - 1e-6 or row["weight_sum_max"] > 1.0 + 1e-6:
        checks.append(f"{profile}: 权重合计异常 {row['weight_sum_min']:.8f}/{row['weight_sum_max']:.8f}")
    if abs(row["cash_max"]) > 1e-10:
        checks.append(f"{profile}: 现金权重非零 {row['cash_max']:.6f}")
    if row["sleeve_bound_violations"]:
        checks.append(f"{profile}: 袖套越界 {row['sleeve_bound_violations']} 行")
    if abs(row["worst_month_return"]) > 0.08:
        checks.append(f"{profile}: 单月跌幅偏大 {row['worst_month']} {row['worst_month_return']:.2%}")
    return row, checks


def append_summary(rows: list[dict[str, Any]]) -> pd.DataFrame:
    path = REPORTS / "v1_iteration_summary.csv"
    current = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_csv(path)
        out = pd.concat([old, current], ignore_index=True)
        out = out.drop_duplicates(["label", "profile"], keep="last")
    else:
        out = current
    out.to_csv(path, index=False, encoding="utf-8-sig")
    return out


def write_notes(label: str, rows: list[dict[str, Any]], checks: list[str]) -> None:
    path = REPORTS / "v1_iteration_notes.md"
    lines = []
    if path.exists():
        lines.append(path.read_text(encoding="utf-8").rstrip())
        lines.append("")
    lines.append(f"## {label}")
    lines.append("")
    lines.append("| profile | ann_ret | ann_vol | sharpe | max_dd | avg_turnover | worst_month | main_drag | top_positive | top_negative |")
    lines.append("|---|---:|---:|---:|---:|---:|---|---|---|---|")
    for row in rows:
        lines.append(
            "| {profile} | {annual_return:.2%} | {annual_vol:.2%} | {sharpe:.2f} | {max_drawdown_metric:.2%} | {avg_turnover:.2%} | {worst_month} | {worst_month_main_drag} | {top_positive_sleeve} | {top_negative_sleeve} |".format(
                **row
            )
        )
    lines.append("")
    if checks:
        lines.append("自动检查：")
        for check in checks:
            lines.append(f"- {check}")
    else:
        lines.append("自动检查：权重合计、现金、袖套边界和单月异常检查均通过。")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit V1 iteration outputs and append summary tables.")
    parser.add_argument("--label", required=True, help="Iteration label, e.g. baseline or conservative_v1.")
    args = parser.parse_args()

    cfg = load_config()
    rows: list[dict[str, Any]] = []
    checks: list[str] = []
    for profile in PROFILES:
        row, profile_checks = audit_profile(profile, args.label, cfg)
        rows.append(row)
        checks.extend(profile_checks)
    summary = append_summary(rows)
    write_notes(args.label, rows, checks)
    print(
        json.dumps(
            {
                "label": args.label,
                "summary": "output/reports/v1_iteration_summary.csv",
                "notes": "output/reports/v1_iteration_notes.md",
                "rows": len(summary),
                "checks": checks,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

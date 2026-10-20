from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DETAIL_PATH = ROOT / "output" / "reports" / "v1_view_application_search_detail.csv"
OUTPUT_PATH = ROOT / "output" / "reports" / "v1_profile_view_channel_selection.csv"
NOTES_PATH = ROOT / "output" / "reports" / "v1_profile_view_channel_selection.md"
PROFILES = ["low_vol", "balanced", "high_vol"]
DEV_FOLDS = ["dev_2018_2020", "dev_2021_2022", "validation_2023_2024"]
IMPORTANCE = {"low_vol": 0.20, "balanced": 0.35, "high_vol": 0.45}


COMBINATIONS = {
    "baseline_both_all": {p: "baseline_both" for p in PROFILES},
    "clean_ep_low_bal_direct_high": {
        "low_vol": "ep_only_100",
        "balanced": "ep_only_100",
        "high_vol": "direct_only_100",
    },
    "clean_ep_all": {p: "ep_only_100" for p in PROFILES},
    "clean_direct_all": {p: "direct_only_100" for p in PROFILES},
    "ep_low_direct_bal_high": {
        "low_vol": "ep_only_100",
        "balanced": "direct_only_100",
        "high_vol": "direct_only_100",
    },
    "ep_low_half_bal_direct_high": {
        "low_vol": "ep_only_100",
        "balanced": "direct_only_050",
        "high_vol": "direct_only_100",
    },
    "low_none_ep_bal_direct_high": {
        "low_vol": "no_market_view",
        "balanced": "ep_only_100",
        "high_vol": "direct_only_100",
    },
    "low_none_half_bal_direct_high": {
        "low_vol": "no_market_view",
        "balanced": "direct_only_050",
        "high_vol": "direct_only_100",
    },
}


def main() -> None:
    detail = pd.read_csv(DETAIL_PATH)
    detail = detail[detail["status"].eq("ok")].set_index(["policy_id", "profile"])
    baseline = {profile: detail.loc[("baseline_both", profile)] for profile in PROFILES}
    rows: list[dict[str, object]] = []
    for combination, mapping in COMBINATIONS.items():
        record: dict[str, object] = {"combination": combination}
        score = 0.0
        worst_fold = np.inf
        vols: list[float] = []
        for profile in PROFILES:
            policy_id = mapping[profile]
            item = detail.loc[(policy_id, profile)]
            base = baseline[profile]
            dev_sharpe_delta = float(item["development_2018_2024_sharpe"] - base["development_2018_2024_sharpe"])
            dev_return_delta = float(item["development_2018_2024_annual_return"] - base["development_2018_2024_annual_return"])
            dev_dd_delta = float(item["development_2018_2024_max_drawdown"] - base["development_2018_2024_max_drawdown"])
            turnover_delta = float(item["development_2018_2024_avg_turnover"] - base["development_2018_2024_avg_turnover"])
            fold_deltas = [float(item[f"{fold}_sharpe"] - base[f"{fold}_sharpe"]) for fold in DEV_FOLDS]
            score += IMPORTANCE[profile] * (
                dev_sharpe_delta
                + 0.35 * float(np.mean(fold_deltas))
                + 3.0 * dev_return_delta
                + 1.5 * dev_dd_delta
                - 0.15 * max(turnover_delta, 0.0)
                - 0.35 * max(-min(fold_deltas) - 0.10, 0.0)
            )
            worst_fold = min(worst_fold, min(fold_deltas))
            vols.append(float(item["full_annual_vol"]))
            record.update(
                {
                    f"{profile}_policy": policy_id,
                    f"{profile}_dev_sharpe_delta": dev_sharpe_delta,
                    f"{profile}_worst_fold_sharpe_delta": min(fold_deltas),
                    f"{profile}_full_return": float(item["full_annual_return"]),
                    f"{profile}_full_vol": float(item["full_annual_vol"]),
                    f"{profile}_full_sharpe": float(item["full_sharpe"]),
                    f"{profile}_full_drawdown": float(item["full_max_drawdown"]),
                    f"{profile}_full_turnover": float(item["full_avg_turnover"]),
                    f"{profile}_holdout_sharpe": float(item["holdout_2025_plus_sharpe"]),
                }
            )
        vol_order = bool(vols[0] < vols[1] < vols[2])
        gate = bool(
            vol_order
            and worst_fold >= -0.10
            and all(float(record[f"{p}_dev_sharpe_delta"]) >= -0.02 for p in PROFILES)
        )
        record.update(
            {
                "selection_score": float(score),
                "worst_dev_fold_sharpe_delta": float(worst_fold),
                "volatility_order_ok": vol_order,
                "formal_gate_pass": gate,
            }
        )
        rows.append(record)
    summary = pd.DataFrame(rows).sort_values(["formal_gate_pass", "selection_score"], ascending=[False, False]).reset_index(drop=True)
    summary.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    lines = [
        "# V1 分产品 Market View 传导选择",
        "",
        "选择只使用 2018-2024；2025+ 仅展示冻结结果。",
        "",
        "| 组合 | 门槛 | 选择分 | 最差开发期差 | 低/中/高通道 | 低/中/高全样本 Sharpe | 低/中/高 2025+ Sharpe |",
        "|---|---|---:|---:|---|---|---|",
    ]
    for _, row in summary.iterrows():
        channels = "/".join(str(row[f"{p}_policy"]) for p in PROFILES)
        full = "/".join(f"{row[f'{p}_full_sharpe']:.3f}" for p in PROFILES)
        holdout = "/".join(f"{row[f'{p}_holdout_sharpe']:.3f}" for p in PROFILES)
        lines.append(
            f"| {row['combination']} | {'通过' if row['formal_gate_pass'] else '未通过'} | {row['selection_score']:.4f} | "
            f"{row['worst_dev_fold_sharpe_delta']:.4f} | {channels} | {full} | {holdout} |"
        )
    NOTES_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"\nSummary: {OUTPUT_PATH.relative_to(ROOT)}")
    print(f"Notes: {NOTES_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

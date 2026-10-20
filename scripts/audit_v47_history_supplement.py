"""Audit added historical coverage and isolate it from recent forecast changes."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src import v45_portfolio as engine
from src.v46_shadow_research import EXPERIMENT as OLD, shadow_predictions
from src.v47_history_supplement import EXPERIMENT, merge_supplement
from scripts.audit_v46_shadow_research import main as shared_audit


def main():
    shared_audit(EXPERIMENT, "v47")
    reports = EXPERIMENT / "output/reports"
    auxiliary = EXPERIMENT / "data_processed/v47_auxiliary_2022.parquet"
    base, _ = shadow_predictions(pd.read_parquet(ROOT / "data_processed/v45_predictions.parquet"))
    joined, status = merge_supplement(base, auxiliary)
    key = ["as_of_date", "task", "entity"]
    pd.testing.assert_frame_equal(base.loc[base.as_of_date.ge("2023-01-01")].sort_values(key).reset_index(drop=True),
        joined.loc[joined.as_of_date.ge("2023-01-01")].sort_values(key).reset_index(drop=True))
    checks = [{"name": "all_original_2023_plus_forecast_rows_exactly_unchanged", "passed": True}]
    auxiliary_report = json.loads((EXPERIMENT / "output/models/2022_auxiliary/report.json").read_text())
    checks.append({"name": "auxiliary_prediction_file_matches_fit_report",
                   "passed": engine.sha_file(auxiliary) == auxiliary_report["prediction_sha256"]})
    comparison = []
    coverage = pd.read_csv(reports / "v47_controller_signal_coverage.csv")
    old_coverage = pd.read_csv(OLD / "output/reports/v46_controller_signal_coverage.csv")
    for profile in engine.PROFILES:
        manifest = json.loads((reports / f"v45_{profile}_target_manifest.json").read_text())
        used = manifest["forecast"]["history_supplement"]
        checks.append({"name": f"{profile}:actually_used_registered_auxiliary_file",
                       "passed": used["sha256"] == status["sha256"] and used["source_hash"] == status["source_hash"]})
        historical_a2 = pd.read_parquet(engine.output(profile, "A2", "targets", EXPERIMENT)).loc["2022"]
        historical_a1 = pd.read_parquet(engine.output(profile, "A1", "targets", EXPERIMENT)).loc["2022"]
        movement = (historical_a2-historical_a1).abs().sum(axis=1)
        checks.append({"name": f"{profile}:auxiliary_views_change_historical_targets",
                       "passed": bool(movement.gt(1e-8).all()), "calendar_days": len(movement),
                       "changed_days": int(movement.gt(1e-8).sum()), "mean_target_l1": float(movement.mean())})
        for year in [2024, 2025, 2026]:
            row = coverage.loc[coverage.profile.eq(profile) & coverage.year.eq(year) & coverage.segment.eq("train")].iloc[0]
            before = old_coverage.loc[old_coverage.profile.eq(profile) & old_coverage.year.eq(year) & old_coverage.segment.eq("train")].iloc[0]
            checks.append({"name": f"{profile}/{year}:neural_train_coverage_increased",
                           "passed": bool(row.neural_signal_days > before.neural_signal_days),
                           "before": int(before.neural_signal_days), "after": int(row.neural_signal_days), "train_rows": int(row.rows)})
        for layer in ["A0", "A1", "A2", "A3_shadow", "A3_execution_gated"]:
            new = pd.read_csv(engine.output(profile, layer, "daily", EXPERIMENT), parse_dates=["date"])
            old = pd.read_csv(engine.output(profile, layer, "daily", OLD), parse_dates=["date"])
            new, old = new.loc[new.date.ge("2024-01-01")], old.loc[old.date.ge("2024-01-01")]
            if not new.date.equals(old.date):
                raise ValueError("Changed comparison calendar")
            n, o = engine.statistics(new.strategy_return), engine.statistics(old.strategy_return)
            comparison.append({"profile": profile, "layer": layer,
                "old_annual_return": o["annual_return"], "new_annual_return": n["annual_return"],
                "annual_return_delta": n["annual_return"]-o["annual_return"],
                "old_sharpe": o["sharpe"], "new_sharpe": n["sharpe"], "sharpe_delta": n["sharpe"]-o["sharpe"],
                "old_drawdown": o["max_drawdown"], "new_drawdown": n["max_drawdown"]})
            if layer in ["A0", "A1"]:
                checks.append({"name": f"{profile}/{layer}:original_baseline_unchanged",
                               "passed": bool(np.allclose(new.strategy_return, old.strategy_return, atol=1e-10))})
        new_target = pd.read_parquet(engine.output(profile, "A2", "targets", EXPERIMENT)).loc["2024-01-01":]
        old_target = pd.read_parquet(engine.output(profile, "A2", "targets", OLD)).loc["2024-01-01":]
        difference = float((new_target-old_target).abs().to_numpy().max())
        checks.append({"name": f"{profile}:deployment_target_difference_disclosed", "passed": True,
                       "maximum_asset_weight_difference": difference,
                       "reason": "same recent forecasts; initial historical books can propagate through causal constraints"})
    pd.DataFrame(comparison).to_csv(reports / "v47_vs_v46_comparison.csv", index=False)
    pd.DataFrame(checks).to_csv(reports / "v47_history_checks.csv", index=False)
    failed = [c for c in checks if not c["passed"]]
    engine.write_json(reports / "v47_history_audit.json", {"passed": not failed, "failed": failed,
                                                          "supplement": status, "checks": checks,
                                                          "new_raw_observations": 0, "formal_promotion": False})
    lines = ["", "## 补齐历史覆盖的核查", "",
             "没有新增独立市场数据；复用当年历史 iTransformer 编码器，新拟合直接相对收益读出头，",
             "只恢复 2022 年辅助观点。它不等同于原 F0 残差预测，也不代表通过原正式门。", "",
             "| 产品 | PPO年度 | 训练决策日 | 原神经观点日 | 补后神经观点日 |",
             "|---|---:|---:|---:|---:|"]
    for row in checks:
        if ":neural_train_coverage_increased" in row["name"]:
            profile, year = row["name"].split(":")[0].split("/")
            lines.append(f"| {profile} | {year} | {row['train_rows']} | {row['before']} | {row['after']} |")
    lines += ["", "## 与 v46 同层比较", "",
              "| 产品 | 层级 | 原年化 | 新年化 | 原Sharpe | 新Sharpe |",
              "|---|---|---:|---:|---:|---:|"]
    for row in comparison:
        if row["layer"] in ["A2", "A3_shadow", "A3_execution_gated"]:
            lines.append(f"| {row['profile']} | {row['layer']} | {row['old_annual_return']:.2%} | {row['new_annual_return']:.2%} | {row['old_sharpe']:.3f} | {row['new_sharpe']:.3f} |")
    lines += ["", "2023 年以后原观点逐行保持不变；因历史组合会沿因果约束传递，另表披露实际目标差。",
              "任务来源变化可能带来分布差异，不能将全部效果归因为样本数增加。实际持仓漂移仍为未认证风险。"]
    with (reports / "v47_shadow_report.md").open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines)+"\n")
    print(pd.DataFrame(comparison).to_string(index=False), flush=True)
    if failed:
        raise ValueError("Historical coverage audit failed")


if __name__ == "__main__":
    main()

"""Independent accounting, transmission and matched-control checks for v46."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src import v45_portfolio as engine
from src.v46_shadow_research import EXPERIMENT


def main(experiment_root=None, prefix="v46"):
    root = Path(experiment_root) if experiment_root else EXPERIMENT
    reports = root / "output/reports"
    status = json.loads((reports / f"{prefix}_progress.json").read_text())
    if status["status"] != "completed":
        raise ValueError("Cannot audit an unfinished experiment")
    checks, edges, stress, transmission, coverage = [], [], [], [], []

    def check(name, passed, **details):
        checks.append({"name": name, "passed": bool(passed), **details})

    for name, digest in json.loads((reports / "v46_input_manifest.json").read_text()).items():
        check(f"source_unchanged:{name}", engine.sha_file(ROOT/name) == digest)
        check(f"snapshot_unchanged:{name}", engine.sha_file(root/name) == digest)
    for profile in engine.PROFILES:
        targets = {l: pd.read_parquet(engine.output(profile, l, "targets", root))
                   for l in ["A0", "A1", "A2", "A3"]}
        check(f"{profile}:same_execution_target", targets["A2"].equals(targets["A3"]))
        delta = (targets["A2"]-targets["A1"]).abs().sum(axis=1)
        transmission.append({"profile": profile, "changed_target_days": int(delta.gt(1e-8).sum()),
                             "mean_neural_target_l1": float(delta.mean()),
                             "max_neural_target_l1": float(delta.max())})
        for layer, target in targets.items():
            check(f"{profile}/{layer}:budget", np.allclose(target.sum(axis=1), 1., atol=1e-8))
            check(f"{profile}/{layer}:long_only", target.min().min() >= -1e-10)
            diag = pd.read_parquet(engine.output(profile, layer, "diagnostics", root))
            manifest = json.loads((reports / f"v45_{profile}_target_manifest.json").read_text())
            check(f"{profile}/{layer}:content_hash", engine.frame_hash(target) == manifest["target_hashes"][layer])
            for sleeve in engine.RISK_SLEEVES:
                assets = [a for a in target if manifest["module_map"][a] == sleeve]
                weight = target[assets].sum(axis=1)
                check(f"{profile}/{layer}/{sleeve}:target_box",
                      ((weight >= diag[f"lower_{sleeve}"]-1e-7)
                       & (weight <= diag[f"upper_{sleeve}"]+1e-7)).all())
        dailies = {}
        layers = ["A0", "A1", "A2"] + (["A3_shadow", "A3_execution_gated"] if status["train_ppo"] else [])
        for layer in layers:
            daily = pd.read_csv(engine.output(profile, layer, "daily", root), parse_dates=["date", "signal_date"])
            check(f"{profile}/{layer}:unique_sorted_calendar", daily.date.is_unique and daily.date.is_monotonic_increasing)
            check(f"{profile}/{layer}:full_calendar", len(daily) == 1095
                  and daily.date.iloc[0] == pd.Timestamp("2022-01-05")
                  and daily.date.iloc[-1] == pd.Timestamp("2026-07-15"))
            check(f"{profile}/{layer}:finite", np.isfinite(daily[["strategy_return", "nav", "cost", "turnover"]]).all().all())
            check(f"{profile}/{layer}:execution_after_signal", daily.date.gt(daily.signal_date).all())
            check(f"{profile}/{layer}:actual_L1_cost", np.allclose(daily.cost, daily.turnover*.0005, atol=1e-10))
            check(f"{profile}/{layer}:nav_reconciles", np.allclose(daily.nav, (1+daily.strategy_return).cumprod(), atol=1e-10))
            holdings = pd.read_parquet(engine.output(profile, layer, "holdings", root))
            grouped = holdings.groupby("date", sort=True)
            for column in ["pretrade_weight", "weight", "end_weight"]:
                check(f"{profile}/{layer}:{column}_budget", np.allclose(grouped[column].sum(), 1., atol=1e-8))
                check(f"{profile}/{layer}:{column}_long_only", holdings[column].min() >= -1e-10)
            contribution = grouped.return_contribution.sum().to_numpy()
            check(f"{profile}/{layer}:gross_attribution", np.allclose(contribution, daily.gross_return, atol=1e-10))
            turnover = (holdings.weight-holdings.pretrade_weight).abs().groupby(holdings.date).sum().to_numpy()
            check(f"{profile}/{layer}:actual_drift_turnover", np.allclose(turnover, daily.turnover, atol=1e-10))
            if layer in ["A0", "A1", "A2"]:
                check(f"{profile}/{layer}:mechanical_self_reward_zero", daily.reward.abs().max() < 1e-10)
            if layer in ["A0", "A1"]:
                old = pd.read_csv(engine.output(profile, layer, "daily", ROOT))
                check(f"{profile}/{layer}:unchanged_baseline", np.allclose(old.strategy_return, daily.strategy_return, atol=1e-10))
            dailies[layer] = daily.set_index("date").loc["2024-01-01":]
            for bps in [5, 10, 20]:
                # Fixed execution path stress, NOT retraining the actor at new costs.
                x = dailies[layer]
                net = x.strategy_return-x.turnover*((bps-5)/1e4)
                stress.append({"profile": profile, "layer": layer, "cost_bps": bps,
                               "scope": "fixed_path_not_retrained", **engine.statistics(net)})
        pairs = [("A2", "A1"), ("A1", "A0")]
        if status["train_ppo"]:
            pairs += [("A3_shadow", "A2"), ("A3_shadow", "A1"), ("A3_execution_gated", "A2")]
        for candidate, baseline in pairs:
            x, y = dailies[candidate], dailies[baseline]
            check(f"{profile}/{candidate}-{baseline}:same_calendar", x.index.equals(y.index))
            sx, sy = engine.statistics(x.strategy_return), engine.statistics(y.strategy_return)
            for block in [20, 60]:
                edges.append({"profile": profile, "candidate": candidate, "baseline": baseline,
                              "delta_annual_return": sx["annual_return"]-sy["annual_return"],
                              "delta_sharpe": sx["sharpe"]-sy["sharpe"],
                              **engine.bootstrap_comparison(x.strategy_return, y.strategy_return, block_days=block)})
        if status["train_ppo"]:
            for year in [2024, 2025, 2026]:
                fold = json.loads((reports / f"v45_{profile}_{year}_ppo_gate.json").read_text())
                diag = pd.read_parquet(engine.output(profile, "A3", "diagnostics", root))
                for segment, bounds in fold["split"].items():
                    part = diag.loc[(diag.index >= pd.Timestamp(bounds["signal_start"]))
                                    & (diag.index < pd.Timestamp(bounds["last_execution"]))]
                    coverage.append({"profile": profile, "year": year, "segment": segment,
                                     "rows": len(part), "neural_signal_days": int(part.active_components.gt(0).sum()),
                                     "qualified_seeds": fold["qualified_seeds"]})
                folder = root / "data_processed/v45_controller_models" / profile / str(year)
                paths = sorted(folder.glob("*.pt"))
                check(f"{profile}/{year}:three_real_seed_checkpoints", len(paths) == 3)
                for path in paths:
                    record = engine.torch.load(path, map_location="cpu", weights_only=False)
                    engine.restore_model(record, 32)
                    teacher = pd.DataFrame(record["teacher_audit"])
                    signal = pd.to_datetime(teacher.signal_date)
                    end = pd.to_datetime(teacher.label_end_date)
                    train_end = pd.to_datetime(teacher.train_last_execution)
                    check(f"{profile}/{year}/{record['seed']}:matured_teacher_labels",
                          (signal < end).all() and (end <= train_end).all()
                          and (train_end < pd.Timestamp(year, 1, 1)).all())
    pd.DataFrame(checks).to_csv(reports / f"{prefix}_contract_checks.csv", index=False)
    pd.DataFrame(edges).to_csv(reports / f"{prefix}_increment_bootstrap.csv", index=False)
    pd.DataFrame(stress).to_csv(reports / f"{prefix}_cost_stress.csv", index=False)
    pd.DataFrame(transmission).to_csv(reports / f"{prefix}_transmission.csv", index=False)
    pd.DataFrame(coverage).to_csv(reports / f"{prefix}_controller_signal_coverage.csv", index=False)
    failed = [c for c in checks if not c["passed"]]
    result = {"engineering_checks_passed": not failed, "checks": len(checks), "failed": failed,
              "formal_promotion": False, "source_PIT_certified": False, "untouched_holdout": False,
              "actual_drift_contract": "disclosed_not_certified", "bootstrap_scope": "conditional_reused_research_history_not_selection_adjusted"}
    engine.write_json(reports / f"{prefix}_audit.json", result)
    metrics = pd.read_csv(reports / f"{prefix}_shadow_comparison.csv")
    text = [f"# {prefix} 影子实验实际结果", "",
            "冻结数据截至 2026-07-15；比较区间从 2024 年开始。历史已多次研发使用，不是未触碰样本外。",
            "正式晋升仍为 False；跳过研究阶段门控是反事实实验，不代表证据门通过。", "",
            "## 净表现（实际 L1 成交 5bp）", "",
            "| 产品 | 层级 | 年化收益 | 波动 | Sharpe | 最大回撤 | PPO使用日 |",
            "|---|---|---:|---:|---:|---:|---:|"]
    for row in metrics.itertuples():
        text.append(f"| {row.profile} | {row.layer} | {row.annual_return:.2%} | {row.annual_volatility:.2%} | {row.sharpe:.3f} | {row.max_drawdown:.2%} | {row.model_used_days} |")
    text += ["", "## 模块净增量", "",
             "| 产品 | 比较 | 年化差（百分点） | Sharpe差 | 60日分块收益正概率 | 60日分块Sharpe正概率 |",
             "|---|---|---:|---:|---:|---:|"]
    for row in pd.DataFrame(edges).query("block_days == 60").itertuples():
        text.append(f"| {row.profile} | {row.candidate} - {row.baseline} | {100*row.delta_annual_return:+.3f} | {row.delta_sharpe:+.3f} | {row.return_positive_probability:.1%} | {row.sharpe_positive_probability:.1%} |")
    text += ["", "A2-A1 检验神经观点，A3_shadow-A2 检验同目标 PPO 执行。正值不自动等于统计可靠。",
             "分块概率是固定路径、重复使用历史上的条件重采样，不是成功率预测，也未校正历次模型选择。", "",
             "## 审计与限制", "",
             f"工程检查 {len(checks)} 项，通过 {len(checks)-len(failed)} 项。",
             "保留未通过正式预测门及执行门的影子结果，不依据部署收益选中获胜模型。",
             "来源 PIT（历史成分、修订、分红财富及代理）仍未独立认证，实际持仓漂移仍披露而非认证全部合规。",
             "本轮复用已训练 iTransformer；PPO 为本轮实际重新拟合。1000 epoch 上限不等于必须训练到1000。",
             "成本压力文件为固定成交路径重估，不是不同成本下重训 PPO。", "",
             f"详见 {prefix}_contract_checks.csv、{prefix}_increment_bootstrap.csv、{prefix}_cost_stress.csv、{prefix}_transmission.csv。"]
    (reports / f"{prefix}_shadow_report.md").write_text("\n".join(text)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    if failed:
        raise ValueError("v46 independent audit failed")


if __name__ == "__main__":
    main()

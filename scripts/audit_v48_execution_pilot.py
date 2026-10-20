"""Read-only result audit plus concise v48 ablation report; no training."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import v45_portfolio as engine
from src.v48_execution_learning import source_hash


def main():
    root = ROOT / "output/experiments/v48_execution_pilot"
    reports = root / "output/reports"
    progress = json.loads((reports / "progress.json").read_text(encoding="utf-8"))
    if progress["status"] != "completed":
        raise ValueError("Pilot not completed")
    results = json.loads((reports / "pilot_results.json").read_text(encoding="utf-8"))
    checks, rows = [], []
    for result in results:
        variant = result["variant"]
        local = root / variant
        cfg = json.loads((local / "output/reports/registered_config.json").read_text(encoding="utf-8"))
        checks.extend([dict(check=f"{variant}:no_test", passed=result["test_replayed"] is False),
                       dict(check=f"{variant}:source_hash", passed=cfg["v48"]["source_hash"] == source_hash()),
                       dict(check=f"{variant}:three_seeds", passed=len(result["fold"]["seed_gates"]) == 3)])
        for name, digest in cfg["v48"]["input_hashes"].items():
            checks.append(dict(check=f"{variant}:immutable:{name}", passed=engine.sha_file(ROOT / name) == digest))
        for seed_gate in result["fold"]["seed_gates"]:
            seed = seed_gate["seed"]
            fold = result["fold"]
            path = local / f"data_processed/v45_controller_models/balanced/2024/lr{fold['selected_lr']}_entropy{fold['selected_entropy']}_seed{seed}.pt"
            record = torch.load(path, map_location="cpu", weights_only=False)
            cutoff = pd.Timestamp(record["training_cutoff"])
            checks.append(dict(check=f"{variant}:{seed}:teacher_T_only", passed=all(
                pd.Timestamp(a["signal_date"]) < pd.Timestamp(a["label_end_date"])
                <= pd.Timestamp(a["train_last_execution"]) < cutoff for a in record["teacher_audit"])))
            checks.append(dict(check=f"{variant}:{seed}:normalization_T_only", passed=
                pd.Timestamp(record["normalization_train_last_execution"]) == pd.Timestamp(fold["split"]["train"]["last_execution"])
                and pd.Timestamp(record["normalization_train_last_execution"]) < cutoff))
            checks.append(dict(check=f"{variant}:{seed}:finite_checkpoint", passed=bool(
                np.isfinite(record["mean"]).all() and np.isfinite(record["std"]).all()
                and (record["std"] > 0).all()
                and all(torch.isfinite(x).all().item() for x in record["state_dict"].values()))))
            diagnostics = next(d for d in result["seed_diagnostics"] if d["seed"] == seed)
            rows.append(dict(variant=variant, seed=seed, V_reward=record["selection_reward"],
                             G_reward_mean=np.mean(seed_gate["block_edges_reward_units"]),
                             G_net_edge_daily=np.mean(seed_gate["block_edges_net_return"]),
                             passed=seed_gate["passed"], **{k:v for k,v in diagnostics.items() if k != "seed"}))
    baseline_path = ROOT / "output/experiments/v47_history_supplement/output/reports/v45_balanced_2024_ppo_gate.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    for seed_gate in baseline["seed_gates"]:
        rows.append(dict(variant="v47_original", seed=seed_gate["seed"],
                         G_reward_mean=np.mean(seed_gate["block_edges_reward_units"]),
                         G_net_edge_daily=np.mean(seed_gate["block_edges_net_return"]), passed=seed_gate["passed"]))
    passed = all(c["passed"] for c in checks)
    engine.write_json(reports / "audit.json", dict(passed=passed, checks=checks,
        source_data_PIT_certified=False, live_execution_certified=False,
        test_replayed=False, formal_promotion=False))
    frame = pd.DataFrame(rows)
    frame.to_csv(reports / "gate_ablation.csv", index=False)
    lines = ["# V48 执行学习试验结果", "", "## 结论", "",
             f"工程检查：{sum(c['passed'] for c in checks)}/{len(checks)}。本轮未回放 2024+ 测试收益，不生成新的测试夏普。", "",
             "| 组别 | 通过 G 的种子 | 平均 G 奖励增量 | 平均 G 日净收益增量（bp） |",
             "|---|---:|---:|---:|"]
    for variant, group in frame.groupby("variant", sort=False):
        lines.append(f"| {variant} | {int(group.passed.sum())}/3 | {group.G_reward_mean.mean():.8f} | {group.G_net_edge_daily.mean()*10000:.5f} |")
    lines.extend(["", "奖励增量包含风险惩罚，不能当作收益率或夏普。G 为 2023-10-23 信号至 2023-12-29 执行的历史门控，三个分块等权；短区间仅用于拒绝明显无增量方案，不代表稳健盈利。", "",
                  "## 处理", ""])
    for result in results:
        lines.append(f"- {result['variant']}：{result['next_step']}，正式晋升始终为 False。")
    lines.extend(["", "两组使用同一 V47 目标、风险约束及成本。scaled_hard 仅改动态缩放，scaled_value 再改教师后续策略和软标签。", "",
                  "如均失败，按预设规则停止扩跑，不降低门槛，也不试探 2024+ 曲线来反向选参数。下一步应检查动作任务/奖励是否真的能区分有价值的执行选择，并补真实的预测不确定性与历史成熟标签校准。", "",
                  "原有数据源 PIT、代理收益、持仓漂移与次日收盘近似问题并未因此解决。旧正式结果不覆盖。", "",
                  "设计说明：`docs/v48_execution_learning_pilot_20261015.md`。逐种子结果见同目录 `gate_ablation.csv`；审计见 `audit.json`。"])
    (reports / "v48_pilot_report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(frame.to_string(index=False))
    print(f"Audit: {sum(c['passed'] for c in checks)}/{len(checks)}", flush=True)
    if not passed:
        raise ValueError("Pilot audit failed")


if __name__ == "__main__":
    main()

"""Read-only model diagnostics; write review tables, never tune or authorize trading."""
import json
import sys
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import torch
from src import v45_forecast as engine


def main():
    parent = ROOT / "output/reports/corporate_action_v58"
    out = ROOT / "output/reports/multitask_v58"
    meta = json.loads((parent / "metadata.json").read_text())
    registry = json.loads((parent / "status.json").read_text())
    gates = json.loads((out / "forecast_gates.json").read_text())["gates"]
    cfg = engine.configuration(ROOT / "config/v45_research.yaml")
    with np.load(ROOT / "data_processed/corporate_action_v58/dataset.npz") as z:
        a = {k: z[k] for k in z.files}
    dates = pd.to_datetime(a["dates"])
    rows, splits, losses = [], [], []
    for year in range(2022, 2027):
        split = engine.shared_split(a, year, cfg)
        splits.append({"year": year, **{
            f"{part}_{suffix}": value
            for part in ["train", "validation", "gate", "test"]
            for suffix, value in [("n", len(split[part])), ("start", str(dates[split[part][0]].date())), ("end", str(dates[split[part][-1]].date()))]
        }})
        bundles = []
        for seed in cfg["seeds"]:
            folder = ROOT / f"output/models/multitask_v58/{year}/{seed}"
            with np.load(folder / "forecast_arrays.npz") as z:
                bundles.append({k: z[k].copy() for k in z.files})
            loss = pd.read_csv(folder / "loss.csv")
            summary = json.loads((folder / "summary.json").read_text())
            losses.append(dict(year=year, seed=seed, selected_epoch=summary["selected_epoch"],
                trained_epochs=summary["trained_epochs"], selected_validation_loss=summary["validation_loss"],
                last_validation_loss=float(loss.iloc[-1].validation_loss),
                first_validation_loss=float(loss.iloc[0].validation_loss),
                late_validation_deterioration=float(loss.iloc[-1].validation_loss)-summary["validation_loss"]))
        baseline = torch.load(ROOT / f"output/models/corporate_action_v58/{year}/seed_17_F1.pt",
                              map_location="cpu", weights_only=False)["baseline"]
        ix = split["test"]
        base = baseline.predict(engine.raw_features(a, ix))
        for task in ["asset", "sleeve"]:
            pred = np.mean([b["test_" + task] for b in bundles], axis=0)
            scale, offset = a[task + "_opportunity_scale"][ix], a[task + "_opportunity_offset"][ix]
            raw = (pred + offset) * scale
            raw_base = (base[task] + offset) * scale
            truth = a[task + "_raw_relative_target"][ix]
            score_truth = a[task + "_target"][ix]
            mask = a[task + "_target_mask"][ix].astype(bool)
            groups = a["module_index"] if task == "asset" else np.zeros(5, dtype=int)
            for group in np.unique(groups):
                members = groups == group
                for j, horizon in enumerate([5, 20, 60]):
                    good = mask[:, j, members] & np.isfinite(raw[:, j, members]) & np.isfinite(raw_base[:, j, members])
                    score_ic = engine.daily_ic(pred[:, j, members], score_truth[:, j, members], good)
                    raw_ic = engine.daily_ic(raw[:, j, members], truth[:, j, members], good)
                    baseline_ic = engine.daily_ic(raw_base[:, j, members], truth[:, j, members], good)
                    g = next(g for g in gates if g["year"] == year and g["key"] == [task, int(group), j])
                    rows.append(dict(year=year, task=task,
                        module=meta["modules"][int(group)] if task == "asset" else "cross_sleeve",
                        horizon_days=horizon, asset_count=int(members.sum()), gate_active=g["active"],
                        preyear_gate_score_rank_ic=g.get("rank_ic"),
                        test_score_rank_ic=engine.finite_mean(score_ic), test_raw_rank_ic=engine.finite_mean(raw_ic),
                        test_raw_baseline_rank_ic=engine.finite_mean(baseline_ic),
                        test_raw_delta_rank_ic=engine.finite_mean(raw_ic-baseline_ic),
                        observed_ic_dates=int(np.isfinite(raw_ic).sum()),
                        approximate_nonoverlap_blocks=int(np.isfinite(raw_ic).sum())//horizon,
                        mean_raw_error=engine.finite_mean(np.where(good, raw[:, j, members]-truth[:, j, members], np.nan)),
                        test_previously_inspected=True))
    table = pd.DataFrame(rows)
    table.to_csv(out / "completed_test_diagnostics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(splits).to_csv(out / "actual_annual_splits.csv", index=False)
    pd.DataFrame(losses).to_csv(out / "early_stopping_diagnostics.csv", index=False)
    failures = Counter(k for g in gates for k, passed in g["checks"].items() if not passed)
    pd.DataFrame(failures.most_common(), columns=["failed_check", "components"]).to_csv(out / "gate_failure_counts.csv", index=False)
    reconciliation = dict(metadata_dataset_sha256=meta.get("dataset_sha256"),
        authoritative_dataset_sha256=registry["dataset_sha256"],
        stale_inherited_metadata_hash=meta.get("dataset_sha256") != registry["dataset_sha256"],
        actual_rows=len(dates), actual_start=str(dates[0].date()), actual_end=str(dates[-1].date()),
        annual_splits_recomputed_from_arrays=True, original_metadata_preserved=True,
        note="Inherited metadata hash/split descriptions are not authoritative; training used actual arrays and shared_split")
    (out / "metadata_reconciliation.json").write_text(json.dumps(reconciliation, indent=2), encoding="utf-8")
    passed = table[table.gate_active]
    lines = ["# V58 完成后的客观复核", "", "## 执行状态", "",
        "新编码器及原F系列15/15、分布头15/15全部完成；产物审计通过，预测明细446,985行。",
        "流水线报告完成且无最终异常堆栈；监督进程保存的exit_code为null，不能将它表述为已核实退出码0。", "",
        "## 预测结果", "", f"原门槛通过{len(passed)}/{len(table)}个年度/任务/袖套/期限组件。不是资产通过率或交易胜率。",
        "该门控衡量风险归一化机会分数的横截面排序，不直接认证原始收益预测、择时能力或组合夏普。", "",
        "| 年度 | 组件 | 期限 | 年前门控分数IC | 当年研究区间原始收益IC | 同口径相对F0增量 |",
        "|---|---|---|---|---|---|"]
    for r in passed.itertuples():
        lines.append(f"| {r.year} | {r.module} | {r.horizon_days}日 | {r.preyear_gate_score_rank_ic:.4f} | {r.test_raw_rank_ic:.4f} | {r.test_raw_delta_rank_ic:.4f} |")
    lines += ["", "上述当年数据仅为诊断；2024–2026反复观察，不能再称全新样本外，不能据此选参数或撤销门槛。",
        "信用组件资产较少且存在代理共线，60日期限的126日门控窗仅约2个非重叠区块。即便原bootstrap门通过，也不应把很小的p值解释成强独立证据。",
        "归一化分数与逐资产反变换后的原始收益可能产生不同排序，两者已经分列，不混用。", "", "## 主要瓶颈", ""]
    lines += [f"- {k}：{v}/{len(gates)}个组件未通过。" for k, v in failures.most_common(6)]
    lines += ["", "## 文档一致性问题", "",
        "发现metadata.json继承了上游dataset_sha256及部分年度拆分描述，未同步为V58数据。真实V58哈希由status.json及已通过数据审计确定，训练按实际数组重新生成拆分，不使用旧拆分描述。",
        "为保留已训练产物来源，不原地改写旧元数据；metadata_reconciliation.json及actual_annual_splits.csv提供纠正对照。下一次重建须在生成阶段修正此缺陷。", "",
        "## 正式执行仍阻断", "",
        "转债实际票息/退出现金流、未决持仓估值、REIT分派、代理到可交易产品映射和真实成本仍未完整认证。正式PPO未启动；新EP目标尚未获准，不提供新三产品收益或夏普。", "",
        "下一步先补执行证据与原始收益单位校准，核查信用代理相关性和尾部风险；不立刻加epoch或根据当年收益盲调模型。保留38资产、九宫格及iTransformer→EP/skfolio→PPO设计。"]
    (out / "completion_review.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(table[table.gate_active].to_string(index=False))
    print("Review written; no models, gates or trading permissions changed.")


if __name__ == "__main__":
    main()

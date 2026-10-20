"""Export typed research views and finish report after the bounded T/V experiment."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import torch
from src import v45_forecast as engine
from scripts.run_v45_research_pipeline import process_alive
from scripts.run_v59_raw_mean import score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    out = ROOT / "output/reports/raw_mean_v59"
    start = time.monotonic()
    while True:
        progress = json.loads((out / "progress.json").read_text())
        if progress["status"] == "completed_validation_only":
            break
        alive = process_alive(progress["pid"])
        if not args.watch or progress["status"] == "failed" or not alive or time.monotonic()-start > 7200:
            reason = dict(status="report_not_ready", training_status=progress["status"], process_alive=alive,
                          timeout=time.monotonic()-start > 7200, models_modified=False)
            (out / "report_status.json").write_text(json.dumps(reason, indent=2), encoding="utf-8")
            print(json.dumps(reason), flush=True)
            return
        time.sleep(5)
    cfg = engine.configuration(ROOT / "config/v45_research.yaml")
    settings = json.loads((ROOT / "config/v59_raw_mean.json").read_text())
    meta = json.loads((ROOT / "output/reports/corporate_action_v58/metadata.json").read_text())
    with np.load(ROOT / "data_processed/corporate_action_v58/dataset.npz") as z:
        a = {k: z[k] for k in z.files}
    dates = pd.to_datetime(a["dates"])
    frames, controls = [], []
    for year in settings["years"]:
        split = engine.shared_split(a, year, cfg)
        ix = split["validation"]
        train_end = str(dates[split["train"][-1]].date())
        # Selected weights depend on the whole V-label window. Historical V
        # predictions are retrospective, not tradable as-of their feature date.
        maturity = int(engine.end_times(a)[ix].max())
        known_index = int(np.searchsorted(a["dates"], maturity, side="right"))
        if known_index >= len(dates):
            raise ValueError("No session after selection-label maturity")
        selection_known = dates[known_index]
        available = pd.to_datetime(np.maximum(dates[ix+1].asi8, selection_known.value))
        baseline = torch.load(ROOT / f"output/models/corporate_action_v58/{year}/seed_17_F1.pt", map_location="cpu", weights_only=False)["baseline"]
        b = baseline.predict(engine.raw_features(a, ix))
        truth = {t: a[t+"_raw_relative_target"][ix] for t in ["asset", "sleeve"]}
        masks = {t: a[t+"_target_mask"][ix].astype(bool) & np.isfinite(b[t]) for t in truth}
        groups = dict(asset=a["module_index"], sleeve=np.zeros(5, int))
        for seed in settings["seeds"]:
            for kind in settings["candidates"]:
                folder = ROOT / f"output/models/raw_mean_v59/{year}/{seed}/{kind}"
                saved = torch.load(folder / "selected.pt", map_location="cpu", weights_only=False)
                with np.load(folder / "validation_predictions.npz") as z:
                    constant = {t: np.broadcast_to(saved["centers"][t], truth[t].shape) for t in truth}
                    raw_predictions = {t: z[t] for t in truth}
                    ratio, _ = score(raw_predictions, constant, truth, masks, groups)
                    controls.append(dict(year=year, seed=seed, candidate=kind,
                        MSE_ratio_to_T_group_mean=ratio, improvement_vs_T_group_mean=1-ratio,
                        diagnostic_only_not_a_new_selection_rule=True))
                    for task in ["asset", "sleeve"]:
                        mean, q = z[task], z[task+"_quantiles"]
                        names = meta["assets"] if task == "asset" else meta["modules"]
                        for h, horizon in enumerate([5, 20, 60]):
                            for entity, name in enumerate(names):
                                frames.append(pd.DataFrame(dict(as_of=dates[ix], available_after=available,
                                    hypothetical_next_session=dates[ix+1], model_selection_known_after=selection_known,
                                    year=year, seed=seed, candidate=kind, task=task, entity=name,
                                    horizon_days=horizon, mean=mean[:, h, entity], lower=q[:, h, entity, 0],
                                    median=q[:, h, entity, 1], upper=q[:, h, entity, 2],
                                    units="cumulative_raw_relative_price_return",
                                    reference_kind="within_sleeve_reference" if task == "asset" else "balanced_reference",
                                    train_end=train_end, evaluation_partition="historical_validation",
                                    point_in_time_eligible=False, approved=False, execution_certified=False)))
    views = pd.concat(frames, ignore_index=True)
    if not (views.available_after > views.as_of).all():
        raise ValueError("Noncausal availability")
    if not ((views.lower <= views['median']) & (views['median'] <= views.upper)).all():
        raise ValueError("Quantile crossing")
    views.to_parquet(out / "validation_research_mean_views.parquet", index=False)
    pd.DataFrame(controls).to_csv(out / "constant_control_diagnostics.csv", index=False)
    decision = json.loads((out / "decision.json").read_text())
    summary = pd.read_csv(out / "training_summary.csv")
    evidence = json.loads((out / "local_evidence_inventory.json").read_text())["summary"]
    lines = ["# V59 原始相对收益均值实验", "", "## 做了什么", "",
        "保留38资产、九宫格及V58冻结iTransformer；两种固定预测头，2022/2023两折各3种子，共12次训练。只用T/V，不查看G/test表现选模型。",
        "均值MSE提升为主损失，改用T内共享组尺度，避免逐资产归一化改变组内收益排序。输出明确为5/20/60日累计相对价格收益，不是日收益、年化收益或认证总收益。", "",
        "## 验证结果", "", "| 候选 | 相对F0平均MSE改进 | 最差种子/折改进 | 下一轮研究资格 |", "|---|---|---|---|"]
    for item in decision["candidates"]:
        f = summary[summary.candidate == item["candidate"]]
        lines.append(f"| {item['candidate']} | {f.validation_improvement.mean():.2%} | {f.validation_improvement.min():.2%} | {item['research_followup_allowed']} |")
    for item in decision["candidates"]:
        lines += ["", item["candidate"]+"失败项：" + ("、".join(k for k,v in item["checks"].items() if not v) or "无"), ""]
    lines += ["额外导出constant_control_diagnostics.csv，对比仅用T组均值的常数预测。如果模型只是修正F0偏差而未胜过常数，不将其包装成学到了择时；该补充诊断不改预登记选择门槛。", ""]
    lines += [f"下一轮研究候选：{decision['selected_for_followup']}。没有候选通过则停止，不改变门槛，不自动增加搜索。",
        "验证MSE改善不等于组合夏普改善，不能绕过原预测门控及执行认证。历史验证已经用于研究，未来不能称为未观察测试。", "",
        "## EP与PPO接口", "",
        "validation_research_mean_views.parquet包含as_of、hypothetical_next_session、model_selection_known_after、available_after、期限、均值、分位数、参考基准和收益单位。V预测使用V标签选出的模型，属于事后诊断，不是当时可交易预测；available_after至少晚于V全部标签成熟，point_in_time_eligible/approved/execution_certified全部为False。禁止直接拿V预测回填历史交易。",
        "本轮风险头沿用V58原版未重训，V59未宣称风险预测已经改进。PPO仍负责每日执行比例与调仓时点，账本未认证时不启动正式奖励训练。", "",
        "## 执行证据", "",
        f"只读整理9只REIT的{evidence['reit_candidate_records']}条分派候选记录、{evidence['CB_contract_schedule_rows']}行转债合同计划及{evidence['held_CB_requests']}只曾持有转债的付款证据需求。",
        "来源文件哈希、行号和字段缺口已保留；付款单位、公告和实际到账尚需独立核验，未伪造认证。", "",
        "## 复核入口", "",
        "docs/v59_raw_mean_preregistration.md；config/v59_raw_mean.json；manifest.json；training_summary.csv；decision.json；execution_review_tickets.csv。原V58模型和报告保留，不自动替换。"]
    (out / "experiment_review.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    (out / "report_status.json").write_text(json.dumps(dict(status="completed", view_rows=len(views),
        formal_promotion=False, execution_certified=False, test_labels_used=False), indent=2), encoding="utf-8")
    print(json.dumps(dict(status="report_complete", decision=decision, view_rows=len(views))), flush=True)


if __name__ == "__main__":
    main()

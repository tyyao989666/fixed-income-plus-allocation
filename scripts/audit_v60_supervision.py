"""Verify actual last-block updates, not merely requires_grad configuration."""
import argparse
import json
from pathlib import Path
import sys
import time
import hashlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
import torch
from scripts.run_v45_research_pipeline import process_alive


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    out = ROOT / "output/reports/supervised_adapter_v60"
    start = time.monotonic()
    while True:
        progress = json.loads((out / "progress.json").read_text())
        if progress["status"] == "completed_validation_only":
            break
        if not args.wait or progress["status"] == "failed" or not process_alive(progress["pid"]) or time.monotonic()-start > 3600:
            (out / "supervision_audit_status.json").write_text(json.dumps(dict(status="not_ready", training_status=progress["status"])), encoding="utf-8")
            return
        time.sleep(5)
    cfg = json.loads((ROOT / "config/v60_supervised_adapter.json").read_text())
    manifest = json.loads((out / "manifest.json").read_text())
    checks, rows = [], []
    for name,digest in manifest.items():
        p = ROOT / ("data_processed/corporate_action_v58/dataset.npz" if name == "dataset_sha256" else name)
        checks.append(dict(check="source_hash", source=name, passed=hashlib.sha256(p.read_bytes()).hexdigest() == digest))
    for year in cfg["years"]:
        source = torch.load(ROOT / f"output/models/corporate_action_v58/{year}/seed_{cfg['encoder_seed']}_pretraining.pt", map_location="cpu", weights_only=False)
        cached = torch.load(ROOT / f"output/models/supervised_adapter_v60/{year}/prefix.pt", map_location="cpu", weights_only=False)
        checks.append(dict(check="prefix_parity_and_identity", year=year, passed=cached["replay_parity_passed"] is True and cached["manifest"] == manifest))
        for seed in cfg["training_seeds"]:
            for mode in cfg["modes"]:
                folder = ROOT / f"output/models/supervised_adapter_v60/{year}/{seed}/{mode}"
                current = torch.load(folder / "current.pt", map_location="cpu", weights_only=False)
                selected = torch.load(folder / "selected.pt", map_location="cpu", weights_only=False)
                summary = json.loads((folder / "summary.json").read_text())
                checks.append(dict(check="identity", year=year, seed=seed, mode=mode,
                    passed=current["identity"] == selected["identity"] == summary["identity"] and current["identity"]["manifest"] == manifest))
                last = "core.encoder.layers.1."
                changed = [k for k,v in current["state"].items() if k.startswith("core.") and not torch.equal(v, source["best_state"][k])]
                allowed = all(k.startswith(last) for k in changed) if mode == "last_block_adapter" else not changed
                checks.append(dict(check="current_frozen_boundary", year=year, seed=seed, mode=mode, passed=allowed))
                grad_nonzero = False
                if mode == "last_block_adapter":
                    ids = current["optimizer"]["param_groups"][1]["params"]
                    grad_nonzero = any(torch.count_nonzero(current["optimizer"]["state"][i].get("exp_avg", torch.zeros(1))).item() > 0 for i in ids)
                    checks.append(dict(check="last_block_received_gradient", year=year, seed=seed, mode=mode, passed=grad_nonzero))
                rows.append(dict(year=year, seed=seed, mode=mode, trained_epochs=current["epoch"],
                    selected_epoch=summary["selected_epoch"], last_block_current_parameters_changed=bool(changed),
                    last_block_gradient_verified=grad_nonzero, last_block_selected_parameters_changed=summary["last_block_changed"],
                    current_frozen_boundary_passed=allowed))
    pd.DataFrame(rows).to_csv(out / "actual_supervision_audit.csv", index=False)
    pd.DataFrame(checks).to_csv(out / "supervision_checks.csv", index=False)
    passed = all(c["passed"] for c in checks)
    decision = json.loads((out / "decision.json").read_text())
    result = dict(status="passed" if passed else "failed", checks=len(checks),
        adapted_tasks_with_gradient=sum(r["last_block_gradient_verified"] for r in rows),
        adapted_tasks_with_selected_updates=sum(r["last_block_selected_parameters_changed"] for r in rows if r["mode"] == "last_block_adapter"),
        research_followup_allowed=decision["research_followup_allowed"], formal_promotion=False)
    (out / "supervision_audit_status.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (out / "experiment_review.md").open("a", encoding="utf-8") as f:
        f.write("\n## 实际监督更新复核\n\n")
        f.write(f"独立监督审计：{result['status']}；{result['adapted_tasks_with_gradient']}/6个适配任务验证最后层收到非零优化器梯度；最终入选最后层更新的任务数{result['adapted_tasks_with_selected_updates']}/6。训练过不等于验证支持使用。\n")
        f.write("epoch0反变换产生约1e-9级MSE比值舍入误差时，只是与先验数值等价，不解释为实质恶化；未据此修改预登记判断，也不绕过2%改进和配对比较要求。\n")
    print(json.dumps(result), flush=True)
    if not passed:
        raise ValueError("Supervision audit failed")


if __name__ == "__main__":
    main()

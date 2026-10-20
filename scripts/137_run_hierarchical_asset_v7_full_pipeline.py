from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_step(name: str, args: list[str]) -> dict[str, object]:
    started = time.perf_counter()
    print(f"\n=== {name} ===", flush=True)
    print(" ".join(args), flush=True)
    subprocess.run(args, cwd=ROOT, check=True)
    elapsed = time.perf_counter() - started
    print(f"=== {name} done in {elapsed:.1f}s ===", flush=True)
    return {"step": name, "elapsed_seconds": elapsed}


def main() -> None:
    parser = argparse.ArgumentParser(description="One-click hierarchical asset iTransformer + adaptive PPO v7 pipeline.")
    parser.add_argument("--skip-dataset", action="store_true")
    parser.add_argument("--skip-itransformer", action="store_true")
    parser.add_argument(
        "--resume-itransformer",
        action="store_true",
        help="Reuse compatible completed annual/seed checkpoints when training the iTransformer.",
    )
    parser.add_argument("--skip-controller-training", action="store_true")
    parser.add_argument("--profiles", default="low_vol,balanced,high_vol")
    args = parser.parse_args()
    python = sys.executable
    logs: list[dict[str, object]] = []
    if not args.skip_dataset:
        logs.append(run_step("build point-in-time v7 dataset", [python, "scripts/134_build_hierarchical_asset_itransformer_v7.py"]))
    if not args.skip_itransformer:
        training = [python, "scripts/135_train_hierarchical_asset_itransformer_v7.py"]
        if args.resume_itransformer:
            training.append("--resume")
        logs.append(run_step("annual walk-forward hierarchical iTransformer", training))
    training_log = ROOT / "output/reports/hierarchical_asset_itransformer_v7_training_log.csv"
    if training_log.exists():
        logs.append(
            run_step(
                "iTransformer convergence audit",
                [python, "scripts/138_plot_hierarchical_asset_itransformer_v7_training.py"],
            )
        )
    portfolio = [python, "scripts/136_run_hierarchical_asset_itransformer_ppo_v7.py", "--profiles", args.profiles]
    if args.skip_controller_training:
        portfolio.append("--skip-training")
    logs.append(run_step("daily targets, adaptive PPO and backtest", portfolio))
    required = [
        ROOT / "output/reports/hierarchical_asset_itransformer_v7_training_log.csv",
        ROOT / "output/reports/hierarchical_asset_itransformer_v7_convergence_summary.csv",
        ROOT / "output/reports/hierarchical_asset_itransformer_v7_loss_curves.png",
        ROOT / "output/reports/hierarchical_asset_itransformer_v7_gate.json",
        ROOT / "output/reports/hierarchical_asset_itransformer_v7_three_profile_summary.csv",
        ROOT / "output/reports/hierarchical_asset_itransformer_v7_vs_v6_comparison.csv",
        ROOT / "output/reports/hierarchical_asset_itransformer_v7_point_in_time_audit.csv",
        ROOT / "output/reports/hierarchical_asset_itransformer_v7_portfolio_gate.json",
        ROOT / "output/reports/hierarchical_asset_itransformer_v7_portfolio_results.md",
    ]
    manifest = {
        "steps": logs,
        "outputs": [str(path.relative_to(ROOT)) for path in required],
        "all_required_outputs_exist": all(path.exists() for path in required),
    }
    path = ROOT / "output/reports/hierarchical_asset_itransformer_v7_pipeline_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

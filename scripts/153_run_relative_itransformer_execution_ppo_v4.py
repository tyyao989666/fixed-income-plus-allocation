from __future__ import annotations

"""One-click causal relative-view iTransformer + execution-only PPO pipeline."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
IT_CONFIG = ROOT / "config" / "itransformer_daily_v3_residual_robust.yaml"
PPO_CONFIG = ROOT / "config" / "view_execution_ppo_v4_config.yaml"
IT_PREDICTIONS = (
    ROOT
    / "data_processed"
    / "itransformer_daily_timing_v3_residual_robust_predictions.parquet"
)
IT_GATE = (
    ROOT
    / "output"
    / "reports"
    / "itransformer_daily_timing_v3_residual_robust_gate.json"
)
MANIFEST = ROOT / "output" / "reports" / "view_execution_ppo_v4_run_manifest.json"


def run_stage(label: str, arguments: list[str]) -> float:
    started = time.perf_counter()
    print(f"\n=== {label} ===", flush=True)
    print(f"$ {sys.executable} {' '.join(arguments)}", flush=True)
    subprocess.run([sys.executable, *arguments], cwd=ROOT, check=True)
    elapsed = time.perf_counter() - started
    print(f"=== {label} done in {elapsed:.1f}s ===", flush=True)
    return elapsed


def read_itransformer_gate() -> dict:
    if not IT_GATE.exists():
        return {
            "enabled_for_formal_strategy": False,
            "reason": f"missing gate report: {IT_GATE.relative_to(ROOT)}",
        }
    return json.loads(IT_GATE.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build the point-in-time daily dataset, train relative iTransformer, "
            "then train the execution-only PPO controller."
        )
    )
    parser.add_argument("--skip-dataset", action="store_true")
    parser.add_argument("--skip-itransformer-training", action="store_true")
    parser.add_argument("--skip-ppo-training", action="store_true")
    parser.add_argument("--profiles", nargs="+", default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument(
        "--require-itransformer-gate",
        action="store_true",
        help="Stop before PPO if the global iTransformer research gate fails.",
    )
    args = parser.parse_args()

    durations: dict[str, float] = {}
    if not args.skip_dataset:
        durations["dataset"] = run_stage(
            "build point-in-time daily iTransformer dataset",
            [
                "scripts/119_build_itransformer_daily_timing_v2.py",
                "--override",
                str(IT_CONFIG.relative_to(ROOT)),
            ],
        )
    if not args.skip_itransformer_training:
        durations["itransformer"] = run_stage(
            "train and gate relative iTransformer",
            [
                "scripts/120_train_itransformer_daily_timing_v2.py",
                "--override",
                str(IT_CONFIG.relative_to(ROOT)),
            ],
        )
    if not IT_PREDICTIONS.exists():
        raise FileNotFoundError(
            f"Missing iTransformer predictions: {IT_PREDICTIONS.relative_to(ROOT)}"
        )

    itransformer_gate = read_itransformer_gate()
    gate_passed = bool(
        itransformer_gate.get("enabled_for_formal_strategy", False)
    )
    print(
        json.dumps(
            {
                "itransformer_gate_passed": gate_passed,
                "gate_report": str(IT_GATE.relative_to(ROOT)),
                "fallback_if_failed": "mechanical_market_views",
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if args.require_itransformer_gate and not gate_passed:
        raise RuntimeError(
            "iTransformer gate failed; PPO was not started because "
            "--require-itransformer-gate was supplied."
        )

    ppo_arguments = [
        "scripts/149_run_robust_execution_ppo.py",
        "--config",
        str(PPO_CONFIG.relative_to(ROOT)),
    ]
    if args.profiles:
        ppo_arguments.extend(["--profiles", *args.profiles])
    if args.max_epochs is not None:
        ppo_arguments.extend(["--max-epochs", str(args.max_epochs)])
    if args.seeds:
        ppo_arguments.extend(["--seeds", *[str(value) for value in args.seeds]])
    if args.skip_ppo_training:
        ppo_arguments.append("--skip-training")
    durations["execution_ppo"] = run_stage(
        "train, backtest and audit execution-only PPO", ppo_arguments
    )

    manifest = {
        "pipeline": "relative_itransformer_ep_execution_ppo_v4",
        "python": sys.executable,
        "itransformer_config": str(IT_CONFIG.relative_to(ROOT)),
        "ppo_config": str(PPO_CONFIG.relative_to(ROOT)),
        "itransformer_gate_passed": gate_passed,
        "durations_seconds": durations,
        "outputs": {
            "itransformer_predictions": str(IT_PREDICTIONS.relative_to(ROOT)),
            "itransformer_gate": str(IT_GATE.relative_to(ROOT)),
            "ppo_summary": "output/reports/view_execution_ppo_v4_three_profile_summary.csv",
            "ppo_formal_gate": "output/reports/view_execution_ppo_v4_formal_gate.json",
            "report": "output/reports/view_execution_ppo_v4_method_and_results.md",
        },
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

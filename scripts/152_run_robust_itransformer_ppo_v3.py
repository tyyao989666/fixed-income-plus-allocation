from __future__ import annotations

"""Build, train and gate the guarded iTransformer + PPO challenger."""

from pathlib import Path
import subprocess
import sys
import argparse


ROOT = Path(__file__).resolve().parents[1]
IT_CONFIG = "config/itransformer_daily_v3_residual_robust.yaml"
PPO_CONFIG = "config/adaptive_daily_ppo_v3_config.yaml"


def run(label: str, args: list[str]) -> None:
    print(f"\n=== {label} ===", flush=True)
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the guarded iTransformer + PPO v3 challenger.")
    parser.add_argument("--skip-ppo", action="store_true", help="Build and train iTransformer only.")
    args = parser.parse_args()
    run(
        "build point-in-time iTransformer v3 dataset",
        ["scripts/119_build_itransformer_daily_timing_v2.py", "--override", IT_CONFIG],
    )
    run(
        "train residual iTransformer v3",
        ["scripts/120_train_itransformer_daily_timing_v2.py", "--override", IT_CONFIG],
    )
    if not args.skip_ppo:
        run(
            "train and gate adaptive PPO v3",
            ["scripts/126_run_adaptive_daily_itransformer_ppo.py", "--config", PPO_CONFIG],
        )

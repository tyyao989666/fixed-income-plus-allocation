from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(label: str, args: list[str]) -> None:
    print(f"\n=== {label} ===", flush=True)
    subprocess.run(args, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="One-click daily state-card, residual iTransformer and adaptive PPO pipeline.")
    parser.add_argument("--skip-state-cards", action="store_true")
    parser.add_argument("--skip-itransformer", action="store_true")
    parser.add_argument("--skip-ppo", action="store_true")
    args = parser.parse_args()
    python = sys.executable
    if not args.skip_state_cards:
        run("rebuild point-in-time state cards", [python, "scripts/10_generate_market_state_cards.py"])
    if not args.skip_itransformer:
        run(
            "train daily residual iTransformer",
            [python, "scripts/120_train_itransformer_daily_timing_v2.py", "--override", "config/itransformer_daily_v2_residual_challenger.yaml"],
        )
    run("compare iTransformer variants", [python, "scripts/127_compare_daily_itransformer_variants.py"])
    if not args.skip_ppo:
        run("train and backtest adaptive daily PPO", [python, "scripts/126_run_adaptive_daily_itransformer_ppo.py"])


if __name__ == "__main__":
    main()

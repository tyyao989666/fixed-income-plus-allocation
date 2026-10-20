"""Isolated v45 portfolio runner. Default runs full PPO after target preflight.

Use --prepare-targets to build deterministic targets only; --baselines-only
also replays mechanical portfolios without fitting any PPO models.
"""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.v45_portfolio import config, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/v45_research.yaml")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-targets", action="store_true")
    mode.add_argument("--train-ppo", action="store_true", help="Explicit full training (also the default mode)")
    mode.add_argument("--baselines-only", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Reuse valid targets; rebuild stale targets; reject incompatible PPO checkpoints")
    args = parser.parse_args()
    cfg = config(args.config)
    result = run(cfg, prepare_only=args.prepare_targets,
        train_ppo=not args.prepare_targets and not args.baselines_only, resume=args.resume)
    print(f"v45 portfolios: {result['status']}; reports: {ROOT / 'output/reports'}", flush=True)


if __name__ == "__main__":
    main()

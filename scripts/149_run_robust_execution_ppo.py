from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.robust_execution_ppo import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the purged walk-forward PPO execution controller."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "robust_execution_ppo_v1_config.yaml",
    )
    parser.add_argument("--profiles", nargs="+", default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--skip-training", action="store_true")
    args = parser.parse_args()
    result = run_pipeline(
        config_path=args.config,
        profiles=args.profiles,
        train_models=not args.skip_training,
        max_epochs=args.max_epochs,
        seeds=args.seeds,
    )
    print(result, flush=True)


if __name__ == "__main__":
    main()

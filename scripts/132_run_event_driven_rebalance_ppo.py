from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
torch.set_num_threads(8)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

from src.event_driven_rebalance_ppo import run_pipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run event-driven adaptive-rebalancing PPO research challenger.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--profiles", default="low_vol,balanced,high_vol")
    parser.add_argument("--skip-training", action="store_true")
    args = parser.parse_args()
    result = run_pipeline(
        args.config,
        [value.strip() for value in args.profiles.split(",") if value.strip()],
        train_models=not args.skip_training,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

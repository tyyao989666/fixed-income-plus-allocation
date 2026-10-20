from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.market_state_vae import load_config, run_pipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and audit the point-in-time market-state VAE diagnostic layer."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "market_state_vae.yaml",
    )
    args = parser.parse_args()
    result = run_pipeline(load_config(args.config))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

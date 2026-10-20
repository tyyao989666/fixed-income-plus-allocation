from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.hierarchical_asset_itransformer import build_dataset, load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the point-in-time hierarchical asset iTransformer v7 dataset.")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    metadata = build_dataset(load_config(args.config))
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

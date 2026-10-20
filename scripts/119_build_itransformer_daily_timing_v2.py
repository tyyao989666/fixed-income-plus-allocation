from __future__ import annotations

import json
import sys
from pathlib import Path

import argparse
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.itransformer_daily_signal import load_daily_config  # noqa: E402
from src.itransformer_daily_timing_v2 import build_v2_dataset  # noqa: E402


def merge_config(base: dict, override: dict) -> dict:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merge_config(base[key], value)
        else:
            base[key] = value
    return base


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the daily multi-horizon iTransformer dataset.")
    parser.add_argument("--override", type=Path, help="Optional YAML override shared with training.")
    args = parser.parse_args()
    cfg = load_daily_config()
    if args.override:
        with args.override.open("r", encoding="utf-8") as handle:
            cfg = merge_config(cfg, yaml.safe_load(handle) or {})
    metadata = build_v2_dataset(cfg)
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()

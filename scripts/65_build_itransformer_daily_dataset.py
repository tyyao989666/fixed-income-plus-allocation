from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.itransformer_daily_signal import build_daily_path_dataset, daily_out_path, load_daily_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the daily-path iTransformer dataset.")
    parser.add_argument("--config", default=None, help="Optional daily iTransformer YAML config path.")
    args = parser.parse_args()
    cfg = load_daily_config(Path(args.config) if args.config else None)
    df = build_daily_path_dataset(cfg)
    meta = json.loads(daily_out_path(cfg, "metadata").read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "dataset": cfg["outputs"]["dataset"],
                "metadata": cfg["outputs"]["metadata"],
                "rows": len(df),
                "start": meta["start"],
                "end": meta["end"],
                "seq_len_days": meta["seq_len"],
                "feature_count": meta["feature_count"],
                "input_frequency": meta["input_frequency"],
                "output_frequency": meta["output_frequency"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

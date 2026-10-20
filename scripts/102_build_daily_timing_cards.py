from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.daily_timing_cards import build_daily_timing_cards, load_timing_card_config, out_path  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build daily timing state cards for fixed-income-plus profiles.")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = load_timing_card_config(Path(args.config) if args.config else None)
    cards = build_daily_timing_cards(cfg)
    print(
        json.dumps(
            {
                "cards": str(out_path(cfg, "cards").relative_to(ROOT)),
                "latest_csv": str(out_path(cfg, "latest_csv").relative_to(ROOT)),
                "summary_csv": str(out_path(cfg, "summary_csv").relative_to(ROOT)),
                "rows": len(cards),
                "dates": int(cards["date"].nunique()),
                "start": str(cards["date"].min().date()) if not cards.empty else "",
                "end": str(cards["date"].max().date()) if not cards.empty else "",
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.daily_rl_dataset import build_daily_rl_overlay_dataset, load_overlay_config, overlay_out_path  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build daily-state / next-horizon-label data for the PPO overlay.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--profile", default=None, choices=["low_vol", "balanced", "high_vol"])
    parser.add_argument("--strategy-version", default=None, choices=["no_agent", "agentic"])
    parser.add_argument("--decision-frequency", default=None, choices=["month_end", "daily"])
    args = parser.parse_args()

    cfg = load_overlay_config(Path(args.config) if args.config else None)
    if args.profile:
        cfg["data"]["profile"] = args.profile
    if args.strategy_version:
        cfg["data"]["strategy_version"] = args.strategy_version
    if args.decision_frequency:
        cfg["data"]["decision_frequency"] = args.decision_frequency

    df = build_daily_rl_overlay_dataset(cfg)
    meta = json.loads(overlay_out_path(cfg, "metadata").read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "dataset": str(overlay_out_path(cfg, "dataset").relative_to(ROOT)),
                "metadata": str(overlay_out_path(cfg, "metadata").relative_to(ROOT)),
                "rows": len(df),
                "start": meta["start"],
                "end": meta["end"],
                "feature_count": meta["feature_count"],
                "decision_frequency": meta["decision_frequency"],
                "label_frequency": meta["label_frequency"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

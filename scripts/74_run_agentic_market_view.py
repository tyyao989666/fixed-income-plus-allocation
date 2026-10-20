import argparse
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agentic_market_view import run_agentic_market_views  # noqa: E402


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the bounded Agentic Market View graph.")
    parser.add_argument("--frequency", choices=["month_end", "latest", "all"], default=None)
    parser.add_argument("--config", type=Path, default=None, help="Optional agentic Market View YAML config.")
    args = parser.parse_args()

    cards = pd.read_parquet(ROOT / "data_processed" / "market_state_cards_daily_v1.parquet")
    views = run_agentic_market_views(cards, frequency=args.frequency, config_path=args.config)
    valid = int(views["validator_status"].eq("valid").sum()) if not views.empty else 0
    print(f"Agentic Market Views generated: rows={len(views)}, valid={valid}")

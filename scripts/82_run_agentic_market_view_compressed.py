import argparse
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agentic_market_view import run_compressed_agentic_market_views  # noqa: E402


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run compressed single-call LangGraph Agentic Market View.")
    parser.add_argument("--frequency", choices=["month_end", "latest", "all"], default=None)
    parser.add_argument("--config", type=Path, default=None, help="Optional agentic Market View YAML config.")
    parser.add_argument("--max-dates", type=int, default=0, help="Optional first-N dates smoke test.")
    args = parser.parse_args()

    cards = pd.read_parquet(ROOT / "data_processed" / "market_state_cards_daily_v1.parquet")
    if args.max_dates and args.max_dates > 0:
        from src.market_skill import cards_for_llm_frequency

        selected = cards_for_llm_frequency(cards, args.frequency or "month_end").sort_values("date").head(args.max_dates)
        cards = selected.copy()
        frequency = "all"
    else:
        frequency = args.frequency
    views = run_compressed_agentic_market_views(cards, frequency=frequency, config_path=args.config)
    valid = int(views["validator_status"].eq("valid").sum()) if not views.empty else 0
    print(f"Compressed Agentic Market Views generated: rows={len(views)}, valid={valid}")

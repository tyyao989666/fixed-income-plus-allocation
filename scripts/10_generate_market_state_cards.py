from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.market_skill import build_market_state_cards, load_config


if __name__ == "__main__":
    cards = build_market_state_cards(load_config())
    print(f"market state cards generated: rows={len(cards)}, latest={cards['date'].max().date()}")

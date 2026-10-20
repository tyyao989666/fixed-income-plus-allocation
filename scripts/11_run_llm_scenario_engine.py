import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.market_skill import build_llm_views, build_market_state_cards, build_views, load_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate MarketSkill relative views.")
    parser.add_argument("--mode", choices=["mechanical", "llm", "both"], default=None)
    parser.add_argument("--active-source", choices=["mechanical", "llm"], default=None)
    parser.add_argument("--llm-frequency", choices=["month_end", "latest", "all"], default=None)
    args = parser.parse_args()

    cfg = load_config()
    default_source = str(cfg.get("view_generation", {}).get("active_source", "mechanical"))
    mode = args.mode or default_source
    active_source = args.active_source or default_source
    cards = build_market_state_cards(cfg)

    if mode in {"mechanical", "both"}:
        views = build_views(cards, cfg=cfg, write_active=(active_source == "mechanical"))
        valid = int(views["validator_status"].eq("valid").sum()) if not views.empty else 0
        print(f"mechanical MarketSkill views generated: rows={len(views)}, valid={valid}")

    if mode in {"llm", "both"}:
        llm_views = build_llm_views(cards, cfg=cfg, frequency=args.llm_frequency, write_active=(active_source == "llm"))
        valid = int(llm_views["validator_status"].eq("valid").sum()) if not llm_views.empty else 0
        print(f"GLM MarketSkill views generated: rows={len(llm_views)}, valid={valid}")

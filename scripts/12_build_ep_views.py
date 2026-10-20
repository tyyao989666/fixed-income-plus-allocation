import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.market_skill import build_ep_views, build_views


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build EP-ready MarketSkill views.")
    parser.add_argument("--source", choices=["active", "mechanical", "llm"], default="active")
    args = parser.parse_args()

    views = None
    if args.source == "mechanical":
        views = build_views(write_active=False)
    ep_views = build_ep_views(views, source=args.source)
    print(f"EP-ready views generated: rows={len(ep_views)}")

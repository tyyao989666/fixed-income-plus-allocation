from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agentic_market_view import run_agentic_market_views


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one hidden-date Agentic Market View date into isolated outputs.")
    parser.add_argument("--date", required=True, help="Calendar date used only for local card selection, YYYY-MM-DD.")
    parser.add_argument("--config", default="config/agentic_market_view_deepseek.yaml")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_path = ROOT / args.config
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    cfg["paths"] = {
        "agentic_views": str((out_dir / "market_skill_views_agentic_one.parquet").relative_to(ROOT)),
        "agentic_ep_views": str((out_dir / "ep_views_agentic_one.parquet").relative_to(ROOT)),
        "audit_jsonl": str((out_dir / "agentic_audit_one.jsonl").relative_to(ROOT)),
        "raw_responses": str((out_dir / "agentic_raw_one.csv").relative_to(ROOT)),
        "report": str((out_dir / "agentic_report_one.md").relative_to(ROOT)),
    }

    tmp_cfg = out_dir / "agentic_config_one.yaml"
    tmp_cfg.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")

    cards = pd.read_parquet(ROOT / "data_processed" / "market_state_cards_daily_v1.parquet")
    date_key = pd.Timestamp(args.date).strftime("%Y-%m-%d")
    selected = cards[pd.to_datetime(cards["date"]).dt.strftime("%Y-%m-%d").eq(date_key)].copy()
    if selected.empty:
        raise SystemExit(f"No card found for date {date_key}")

    out = run_agentic_market_views(selected, frequency="all", config_path=tmp_cfg)
    valid = int(out["validator_status"].eq("valid").sum()) if not out.empty else 0
    print(json.dumps({"date": date_key, "rows": len(out), "valid": valid}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

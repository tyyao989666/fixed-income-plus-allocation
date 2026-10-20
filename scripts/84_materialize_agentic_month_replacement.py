from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.market_skill import MarketSkillPaths, build_ep_views  # noqa: E402


def main() -> None:
    paths = MarketSkillPaths(
        ep_views=ROOT / "data_processed" / "ep_views_agentic_daily_v1.parquet",
    )
    mechanical_path = paths.mechanical_views
    agentic_path = ROOT / "data_processed" / "market_skill_views_agentic_daily_v1.parquet"
    audit_path = ROOT / "output" / "reports" / "agentic_market_view_audit.jsonl"
    report_path = ROOT / "output" / "reports" / "agentic_month_replacement_report.md"

    mechanical = pd.read_parquet(mechanical_path)
    current = pd.read_parquet(agentic_path)
    audits = [json.loads(x) for x in audit_path.read_text(encoding="utf-8").splitlines() if x.strip()]

    mechanical["date"] = pd.to_datetime(mechanical["date"], errors="coerce")
    current["date"] = pd.to_datetime(current["date"], errors="coerce")
    nonfallback_dates = {
        pd.Timestamp(item["date"])
        for item in audits
        if not bool(item.get("fallback")) and item.get("date")
    }
    replace_months = {date + pd.offsets.MonthEnd(0) for date in nonfallback_dates}

    agentic_rows = current[
        current["date"].isin(nonfallback_dates)
        & current.get("generator", "").astype(str).eq("agentic_market_view_v1")
    ].copy()
    mechanical_month = mechanical["date"] + pd.offsets.MonthEnd(0)
    untouched = mechanical[~mechanical_month.isin(replace_months)].copy()
    out = pd.concat([untouched, agentic_rows], ignore_index=True, sort=False)
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.sort_values(["date", "validator_status", "source_asset", "target_asset"]).reset_index(drop=True)
    out.to_parquet(agentic_path, index=False)
    ep = build_ep_views(out, paths=paths)

    report_path.write_text(
        "# Agentic Month Replacement Materialization\n\n"
        f"- audit_dates: {len(audits)}\n"
        f"- agentic_success_dates: {len(nonfallback_dates)}\n"
        f"- replaced_months: {len(replace_months)}\n"
        f"- mechanical_rows_before: {len(mechanical)}\n"
        f"- final_agentic_view_rows: {len(out)}\n"
        f"- agentic_rows_kept: {len(agentic_rows)}\n"
        f"- ep_views: {len(ep)}\n",
        encoding="utf-8",
    )
    print(
        {
            "audit_dates": len(audits),
            "agentic_success_dates": len(nonfallback_dates),
            "replaced_months": len(replace_months),
            "final_agentic_view_rows": len(out),
            "agentic_rows_kept": len(agentic_rows),
            "ep_views": len(ep),
        }
    )


if __name__ == "__main__":
    main()

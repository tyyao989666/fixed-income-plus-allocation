"""Acquire 16 scoped fund dividend histories and audit cached REIT construction."""
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import tushare as ts

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.v76_fund_cashflows import normalize_schedule
from scripts.acquire_v53_unified_sources import REITS

RAW = ROOT / "data_raw/fund_cashflows_v76"
OUT = ROOT / "output/reports/fund_cashflows_v76"
ETFS = ["511220.SH", "510300.SH", "159915.SZ", "159940.SZ", "510170.SH", "510880.SH", "518880.SH"]
CUTOFF = "2026-10-14"


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pro = ts.pro_api(ts.get_token(), timeout=20)
    summary, events, errors = [], [], []
    for code in ETFS + REITS:
        path = RAW / f"fund_div_{code}.parquet"
        try:
            if path.exists():
                cached = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
                if (cached["endpoint"] != "fund_div" or cached["ts_code"] != code
                        or cached["sha256"] != hashlib.sha256(path.read_bytes()).hexdigest()):
                    raise ValueError("Cached source request or hash mismatch")
                source = pd.read_parquet(path)
            else:
                source = pro.fund_div(ts_code=code)
                if source is None or len(source) >= 2000:
                    raise ValueError("Absent or potentially truncated response")
                source.to_parquet(path, index=False)
                path.with_suffix(".json").write_text(json.dumps(dict(endpoint="fund_div", ts_code=code,
                    fetched_at=pd.Timestamp.now(tz="Asia/Shanghai").isoformat(), rows=len(source),
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest()), indent=2), encoding="utf-8")
                time.sleep(.2)
            normalized, info = normalize_schedule(source, code, CUTOFF)
            summary.append(dict(code=code, category="ETF" if code in ETFS else "REIT", **info))
            if not normalized.empty:
                normalized["category"] = "ETF" if code in ETFS else "REIT"
                events.append(normalized)
            print(f"V76 {code}: source={len(source)} events={info['unique_events']} eligible={info['eligible_events']}", flush=True)
        except Exception as exc:
            errors.append(dict(code=code, error_type=type(exc).__name__, action="blocked_no_retry_no_zero_assumption"))
            print(f"V76 {code}: blocked ({type(exc).__name__})", flush=True)
    merged = pd.concat(events, ignore_index=True) if events else pd.DataFrame()
    merged.to_parquet(OUT / "announced_distribution_schedules.parquet", index=False)
    merged.to_csv(OUT / "announced_distribution_schedules.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summary).to_csv(OUT / "16_fund_coverage.csv", index=False)
    # Compare current schedule extraction to the frozen REIT exdate cash history.
    previous = pd.read_parquet(ROOT / "data_processed/unified_v53/reits_deduplicated_announced_distributions.parquet")
    difference = []
    for code in REITS:
        subset = merged.loc[merged.ts_code.eq(code) & merged.eligible_exdate_research] if not merged.empty else pd.DataFrame()
        if subset.empty:
            continue
        new = subset.groupby("ex_date").cash_per_unit.sum().reindex(previous.index, fill_value=0)
        for day in previous.index[~np.isclose(previous[code], new, atol=1e-10, rtol=0)]:
            difference.append(dict(code=code, ex_date=day, frozen_cash_per_unit=previous.at[day, code],
                                   fresh_schedule_cash_per_unit=new.at[day], applied_to_returns=False))
    pd.DataFrame(difference, columns=["code", "ex_date", "frozen_cash_per_unit", "fresh_schedule_cash_per_unit", "applied_to_returns"]).to_csv(
        OUT / "reit_frozen_vs_fresh_schedule.csv", index=False)
    status = dict(status="scoped_fund_schedules_audited_not_settlement_certified", cutoff=CUTOFF,
        requested_funds=16, successful_funds=len(summary), errors=errors,
        eligible_events=int(merged.eligible_exdate_research.sum()) if not merged.empty else 0,
        reit_differences=len(difference), dividend_schedule_is_not_payment_confirmation=True,
        all_splits_certified=False, settlement_certified=False, returns_modified=False,
        portfolio_promoted=False, ppo_started=False)
    (OUT / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    paths = list(RAW.glob("*")) + [ROOT / "data_processed/unified_v53/reits_deduplicated_announced_distributions.parquet",
                                    ROOT / "src/v76_fund_cashflows.py", Path(__file__)]
    (OUT / "source_hashes.json").write_text(json.dumps({str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in paths if p.is_file()}, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2), flush=True)


if __name__ == "__main__":
    main()

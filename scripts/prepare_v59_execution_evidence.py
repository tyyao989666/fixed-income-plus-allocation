"""Local evidence inventory only; schedules never become certified cash payments."""
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    out = ROOT / "output/reports/raw_mean_v59"
    out.mkdir(parents=True, exist_ok=True)
    sources, events, tickets = [], [], []
    for p in sorted((ROOT / "data_raw/unified_v53").glob("reit_div_*.parquet")):
        f = pd.read_parquet(p)
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        sources.append(dict(path=str(p.relative_to(ROOT)), sha256=digest, rows=len(f), source_type="REIT_distribution_vendor_records"))
        for index, row in f.iterrows():
            def date(key):
                value = row.get(key)
                if pd.isna(value):
                    return None
                parsed = pd.to_datetime(str(value).replace(".0", ""), format="%Y%m%d", errors="coerce")
                return None if pd.isna(parsed) else str(parsed.date())
            ann = date("imp_anndate") or date("ann_date")
            record, ex, pay = date("record_date"), date("ex_date"), date("pay_date")
            cash = row.get("div_cash")
            known = ann is not None and all(v is not None for v in [record, ex, pay]) and pd.notna(cash)
            events.append(dict(ts_code=row.get("ts_code"), announcement_date=ann, record_date=record,
                ex_date=ex, pay_date=pay, div_cash_raw=cash, div_proc=row.get("div_proc"),
                source_path=str(p.relative_to(ROOT)), source_row=int(index), source_sha256=digest,
                dates_and_amount_present=known, amount_unit_verified=False,
                actual_receipt_independently_verified=False, incorporated_into_NAV=False, certified=False,
                synthetic=False))
    frame = pd.DataFrame(events)
    frame.to_csv(out / "reit_payment_candidates.csv", index=False, encoding="utf-8-sig")
    contracts = []
    for p in sorted((ROOT / "data_raw/unified_v53").glob("cb_coupon_contract_*.parquet")):
        f = pd.read_parquet(p)
        contracts.append(f)
        sources.append(dict(path=str(p.relative_to(ROOT)), sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                            rows=len(f), source_type="CB_contract_schedule_NOT_payment"))
    all_contracts = pd.concat(contracts, ignore_index=True).drop_duplicates()
    requested = pd.read_csv(ROOT / "output/reports/multitask_v57/cashflow_evidence_requests.csv")
    counts = all_contracts.groupby("ts_code").size()
    requested["local_contract_rows_current"] = requested.ts_code.map(counts).fillna(0).astype(int)
    requested["payment_history_certified"] = False
    requested["synthetic"] = False
    requested.to_csv(out / "cb_payment_evidence_requests.csv", index=False, encoding="utf-8-sig")
    terminal = ROOT / "data_processed/cb_execution_v54/reviewed_terminal_event_with_evidence.parquet"
    reviewed = pd.read_parquet(terminal)
    sources.append(dict(path=str(terminal.relative_to(ROOT)), sha256=hashlib.sha256(terminal.read_bytes()).hexdigest(),
                        rows=len(reviewed), source_type="reviewed_terminal_event_not_full_coverage"))
    for code, group in frame.groupby("ts_code"):
        tickets.append(dict(scope=code, issue="REIT_payment_verification", records=len(group),
            required="verify div_cash unit, implementation announcement, payment status, record-date units held and accounting convention",
            owner="execution_data_review", status="open", synthetic=False))
    tickets += [dict(scope="all_CB_holdings", issue="contracts_are_not_payment_history", records=len(requested),
        required="actual coupons and terminal notices, holdings on record dates; unresolved inventory marks", owner="CB_data_review", status="open", synthetic=False),
        dict(scope="rate_US_bond_proxies", issue="executable_mapping_missing", records=None,
        required="instrument mapping, FX/financing/roll conventions and instrument-specific costs", owner="execution_model_review", status="open", synthetic=False)]
    pd.DataFrame(tickets).to_csv(out / "execution_review_tickets.csv", index=False, encoding="utf-8-sig")
    result = dict(reit_candidate_records=len(frame), reit_assets=int(frame.ts_code.nunique()),
        reit_records_with_dates_amount=int(frame.dates_and_amount_present.sum()),
        reit_certified_records=0, CB_contract_schedule_rows=len(all_contracts), held_CB_requests=len(requested),
        reviewed_terminal_events=len(reviewed), execution_certificate_created=False, formal_PPO_allowed=False,
        fetched_external_data=False, original_sources_modified=False, synthetic=False)
    (out / "local_evidence_inventory.json").write_text(json.dumps(dict(summary=result, sources=sources), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Prioritize missing coupons from actual V92 fills, not hypothetical flat holdings.

Anniversary dates are lookup hints only; do not create payment entitlements here.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/reports/cb_coupons_v93"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    bound = {}
    def bind(path):
        path = ROOT / path if isinstance(path, str) else path
        bound[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path
    parent = json.loads(bind("output/reports/cb_cashflows_v92/status.json").read_text(encoding="utf-8"))
    if parent["status"] != "retrospective_screen_replayed_formal_gate_blocked":
        raise ValueError("V92 not complete")
    for relative, expected in parent["source_sha256"].items():
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
            raise ValueError("Parent source changed: " + relative)
        bound[relative] = expected
    frames = [pd.read_parquet(bind(p)) for p in sorted((ROOT / "data_raw/unified_v53").glob("cb_coupon_contract_*.parquet"))]
    contracts = pd.concat(frames, ignore_index=True).drop_duplicates()
    keys = ["ts_code", "rate_start_date", "rate_end_date"]
    if contracts.duplicated(keys).any() or not contracts.rate_freq.eq(1).all():
        raise ValueError("Conflicting contracts or unsupported nonannual frequency")
    for col in ["rate_start_date", "rate_end_date"]:
        contracts[col] = pd.to_datetime(contracts[col].astype(str), format="%Y%m%d", errors="raise")
    contracts.coupon_rate = pd.to_numeric(contracts.coupon_rate, errors="raise")
    if not np.isfinite(contracts.coupon_rate).all() or contracts.coupon_rate.lt(0).any():
        raise ValueError("Invalid rates")
    fills = pd.read_parquet(bind("data_processed/cb_cashflows_v92/actual_fills.parquet"))
    events = pd.read_parquet(bind("data_processed/cb_cashflows_v88/reviewed_terminal_events.parquet"))
    eligibility = pd.read_parquet(bind("data_processed/cb_cashflows_v92/signal_buy_eligibility.parquet"))
    basic = pd.read_parquet(bind("data_raw/tushare/cb_basic_all.parquet")).drop_duplicates("ts_code").set_index("ts_code")
    exclusions = set(parent["excluded_securities"])
    dates = eligibility.index
    rows, ignored = [], []
    for contract in contracts.itertuples(index=False):
        code = contract.ts_code
        if code in exclusions or code not in eligibility:
            continue
        hint = contract.rate_end_date + pd.Timedelta(days=1)
        if hint < dates[0] or hint > dates[-1]:
            continue
        payments = dates[dates >= hint]
        if not len(payments):
            continue
        approximate_ex = payments[0]
        prior = dates[dates < approximate_ex]
        if not len(prior):
            continue
        approximate_record = prior[-1]
        terminal = events[(events.ts_code == code) & (events.kind == "terminal") & (events.ex_date <= approximate_ex)]
        if len(terminal):
            ignored.append(dict(ts_code=code, anniversary_hint=hint, reason="terminal_cash_includes_last_coupon_or_security_already_retired"))
            continue
        owned = fills[(fills.ts_code == code) & (fills.date <= approximate_record)].groupby("asset_code").quantity_change.sum()
        for basket, qty in owned[owned > 1e-8].items():
            rows.append(dict(ts_code=code, asset_code=basket,
                name=basic.at[code, "bond_short_name"] if code in basic.index else None,
                interest_start=contract.rate_start_date, interest_end=contract.rate_end_date,
                anniversary_lookup_hint=hint, approximate_record_NOT_verified=approximate_record,
                approximate_ex_NOT_verified=approximate_ex, provisional_owned_quantity=qty,
                contract_rate_percent=contract.coupon_rate,
                gross_reference_yuan_NOT_entitlement=qty*contract.coupon_rate,
                actual_receipt_yuan=None, payment_verified=False,
                purpose="forensic_query_priority_only_no_historical_signal_or_cash_posting"))
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "owned_coupon_lookup_queue_NOT_entitlements.csv", index=False, encoding="utf-8-sig")
    summary = frame.groupby(["ts_code", "name", "anniversary_lookup_hint"], as_index=False).agg(
        baskets=("asset_code", "size"), gross_reference_yuan_NOT_entitlement=("gross_reference_yuan_NOT_entitlement", "sum"))
    summary = summary.sort_values("gross_reference_yuan_NOT_entitlement", ascending=False)
    summary.to_csv(OUT / "coupon_event_priority.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(ignored).to_csv(OUT / "terminal_overlap_lookup_exclusions.csv", index=False)
    bind(Path(__file__))
    status = dict(status="owned_coupon_lookup_queue_built_not_cashflows", contract_rows=len(contracts),
                  owned_basket_event_candidates=len(frame), candidate_events=len(summary),
                  candidate_securities=frame.ts_code.nunique(), excluded_securities_not_reintroduced=True,
                  original_38_asset_structure_preserved=True, cash_postings=0, training_started=False,
                  formal_returns_certified=False, retrospective_universe_selection=True,
                  source_sha256=bound)
    (OUT / "queue_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.head(10).to_string(index=False))
    print(json.dumps({k: v for k, v in status.items() if k != "source_sha256"}, ensure_ascii=False))


if __name__ == "__main__":
    main()

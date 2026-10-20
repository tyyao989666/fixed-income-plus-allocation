"""Compare V86 with V85 and independently verify event timing and traded costs."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from scripts.run_v52_execution_preflight import sha, save
from scripts.replay_v86_inventory_maturities import CASES, exact_total


def main():
    report = ROOT / "output/reports/cb_cashflows_v86"
    out = ROOT / "data_processed/cb_cashflows_v86"
    status_path = report / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status["status"] != "research_accounting_completed_formal_gate_blocked":
        raise ValueError("Incomplete replay")
    for name, expected in status["source_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"Replay source changed: {name}")
    paths = [status_path, Path(__file__), ROOT / "scripts/replay_v86_inventory_maturities.py",
             out / "daily_accounting_diagnostics.parquet", out / "actual_fills.parquet",
             out / "reviewed_terminal_events.parquet", out / "unresolved_inventory.parquet",
             report / "actual_registered_entitlements.csv", report / "event_transitions.csv",
             ROOT / "data_processed/cb_cashflows_v85/unresolved_inventory.parquet",
             ROOT / "data_processed/cb_evidence_v50/last_known_balance_yuan.parquet"]
    daily, fills, events, missing = [pd.read_parquet(p) for p in paths[3:7]]
    rights = pd.read_csv(paths[7])
    transitions = pd.read_csv(paths[8], parse_dates=["day"])
    before = pd.read_parquet(paths[9])
    sessions = pd.read_parquet(paths[10]).index
    pos = sessions.get_indexer(pd.DatetimeIndex(daily.date))
    if not (pos > 0).all():
        raise ValueError("Date outside execution calendar")
    np.testing.assert_array_equal(pd.DatetimeIndex(daily.signal_date), sessions.take(pos-1))
    costs = float((fills.quantity_change.abs() * fills.execution_close * .0005).sum())
    np.testing.assert_allclose(costs, daily.cost_yuan.sum(), atol=1e-6, rtol=1e-12)
    payments = transitions.loc[transitions.transition.eq("confirmed_receivable_to_cash")]
    payment_checks = payments.merge(events[["event_id", "confirmed_at", "pay_date"]], on="event_id",
                                   validate="one_to_one", suffixes=("_transition", "_contract"))
    if not (payment_checks.day.ge(payment_checks.confirmed_at_contract)
            & payment_checks.day.ge(payment_checks.pay_date)).all():
        raise ValueError("Premature or duplicate payment")
    claims = rights.loc[rights.ts_code.eq("113507.SH")]
    if claims.settled.any() or payments.ts_code.eq("113507.SH").any():
        raise ValueError("Tianma contradictory payment was released")
    summaries = []
    for case in CASES:
        code = case["code"]
        event = events.loc[events.ts_code.eq(code)].iloc[0]
        r = rights.loc[rights.ts_code.eq(code)]
        late_fills = fills.loc[fills.ts_code.eq(code) & fills.date.ge(event.ex_date)]
        after = missing.loc[missing.ts_code.eq(code) & missing.date.ge(event.ex_date)]
        if len(late_fills) or len(after):
            raise ValueError(f"Retired inventory or execution remains: {code}")
        paid = payments.loc[payments.ts_code.eq(code)]
        np.testing.assert_allclose(paid.amount_all_books.sum(), r.gross_claim_yuan.sum(), atol=1e-7)
        summaries.append(dict(ts_code=code, exact_issuer_total_yuan=exact_total(case),
            record_date=event.record_date, cash_confirmed_at=event.confirmed_at,
            registered_quantity=r.registered_quantity.sum(), nonzero_baskets=int(r.registered_quantity.gt(1e-12).sum()),
            gross_receipt_yuan=r.gross_claim_yuan.sum(), missing_inventory_after_ex_date=len(after),
            fills_after_ex_date=len(late_fills),
            pre_ex_missing_position_days=int((missing.ts_code.eq(code) & missing.date.lt(event.ex_date)).sum())))
    pd.DataFrame(summaries).to_csv(report / "three_maturity_actual_inventory.csv", index=False)
    keys = ["date", "asset_code", "ts_code"]
    changes = before[keys].merge(missing[keys], on=keys, how="outer", indicator=True, validate="one_to_one")
    changes.groupby(["ts_code", "_merge"], observed=True).size().rename("position_days").reset_index().to_csv(
        report / "incremental_gap_changes_vs_v85.csv", index=False)
    latest = daily.date.max()
    known_codes = set(events.ts_code)
    final = missing.loc[missing.date.eq(latest)].copy()
    final["has_reviewed_terminal_event"] = final.ts_code.isin(known_codes)
    final["cashflow_or_quote_evidence_still_required"] = True
    final.to_csv(report / "ending_unresolved_inventory.csv", index=False)
    result = dict(three_contract_totals_exact=True, three_maturities_cancelled_without_post_ex_trades=True,
        prior_domestic_close_signal=True, fee_per_gross_fill=.0005, fee_reconciled_yuan=costs,
        no_early_or_duplicate_payment=True, tianma_pending_claim_preserved=True,
        tianma_pending_claim_yuan=float(claims.gross_claim_yuan.sum()),
        missing_position_days_v85=len(before), missing_position_days_v86=len(missing),
        removed_old_gap_days=int(changes._merge.eq("left_only").sum()),
        new_path_gap_days=int(changes._merge.eq("right_only").sum()),
        ending_missing_codes=int(final.ts_code.nunique()), formal_training_permitted=False,
        note="Per-basket hypothetical inventory accounting; not certified product performance.",
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths})
    save(report / "maturity_closure_audit.json", result)
    print(json.dumps({k:v for k,v in result.items() if k != "source_sha256"}, indent=2))
    print(pd.DataFrame(summaries).to_string(index=False))
    print(final[["asset_code", "ts_code", "quantity", "provisional_mark"]].to_string(index=False))


if __name__ == "__main__":
    main()

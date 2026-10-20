"""Independent timing, cost, extinction and last-traded-price accounting checks."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from scripts.run_v52_execution_preflight import sha, save
from scripts.replay_v87_inventory_maturities import CASES, exact_total


def main():
    report = ROOT / "output/reports/cb_cashflows_v87"
    out = ROOT / "data_processed/cb_cashflows_v87"
    status_path = report / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status["status"] != "research_accounting_completed_formal_gate_blocked":
        raise ValueError("Replay incomplete")
    for name, expected in status["source_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"Source changed: {name}")
    paths = [status_path, Path(__file__), ROOT / "scripts/replay_v87_inventory_maturities.py",
        out / "daily_accounting_diagnostics.parquet", out / "actual_fills.parquet",
        out / "reviewed_terminal_events.parquet", out / "unresolved_inventory.parquet",
        report / "actual_registered_entitlements.csv", report / "event_transitions.csv",
        ROOT / "data_processed/cb_cashflows_v86/unresolved_inventory.parquet",
        ROOT / "data_processed/cb_evidence_v50/last_known_balance_yuan.parquet",
        ROOT / "data_processed/cb_evidence_v50/quotes/close_individual.parquet",
        ROOT / "data_processed/cb_evidence_v50/quotes/volume_individual.parquet"]
    daily, fills, events, missing = [pd.read_parquet(p) for p in paths[3:7]]
    rights = pd.read_csv(paths[7])
    transitions = pd.read_csv(paths[8], parse_dates=["day"])
    before = pd.read_parquet(paths[9])
    sessions = pd.read_parquet(paths[10]).index
    close, volume = [pd.read_parquet(p) for p in paths[11:13]]
    positions = sessions.get_indexer(pd.DatetimeIndex(daily.date))
    if not (positions > 0).all():
        raise ValueError("Execution outside domestic calendar")
    np.testing.assert_array_equal(pd.DatetimeIndex(daily.signal_date), sessions.take(positions-1))
    cost = float((fills.quantity_change.abs() * fills.execution_close * .0005).sum())
    np.testing.assert_allclose(cost, daily.cost_yuan.sum(), atol=1e-6, rtol=1e-12)
    payments = transitions.loc[transitions.transition.eq("confirmed_receivable_to_cash")]
    p = payments.merge(events[["event_id", "confirmed_at", "pay_date"]], on="event_id",
                       validate="one_to_one", suffixes=("_transition", "_contract"))
    if not (p.day.ge(p.confirmed_at_contract) & p.day.ge(p.pay_date)).all():
        raise ValueError("Payment chronology invalid")
    tianma = rights.loc[rights.ts_code.eq("113507.SH")]
    if tianma.settled.any() or payments.ts_code.eq("113507.SH").any():
        raise ValueError("Tianma unresolved amount was released")
    summaries, exposure = [], []
    for case in CASES:
        exact_total(case)
        code = case["code"]
        e = events.loc[events.ts_code.eq(code)].iloc[0]
        r = rights.loc[rights.ts_code.eq(code)]
        late_fills = fills.loc[fills.ts_code.eq(code) & fills.date.ge(e.ex_date)]
        if len(late_fills) or (missing.ts_code.eq(code) & missing.date.ge(e.ex_date)).any():
            raise ValueError("Retired security traded or still in missing inventory")
        paid = payments.loc[payments.ts_code.eq(code)]
        np.testing.assert_allclose(paid.amount_all_books.sum(), r.gross_claim_yuan.sum(), atol=1e-7)
        last = pd.Timestamp(case["last_trade"])
        price = close.at[last, code]
        if not (np.isfinite(price) and price > 0 and np.isfinite(volume.at[last, code]) and volume.at[last, code] > 0):
            raise ValueError("Documented last trade date lacks executable quote")
        summaries.append(dict(ts_code=code, registered_quantity=r.registered_quantity.sum(),
            nonzero_baskets=int(r.registered_quantity.gt(1e-12).sum()), gross_receipt_yuan=r.gross_claim_yuan.sum(),
            confirmed_at=e.confirmed_at, missing_after_ex_date=0, trades_after_ex_date=0))
        for row in r.loc[r.registered_quantity.gt(1e-12)].itertuples(index=False):
            exposure.append(dict(ts_code=code, asset_code=row.asset_code, registered_quantity=row.registered_quantity,
                last_trade_date=last, last_traded_close=float(price), contractual_gross_redemption=float(case["unit"]),
                redemption_vs_last_quote_pct=float(case["unit"])/float(price)-1,
                registered_quantity_times_quote_difference_yuan=row.registered_quantity*(float(case["unit"])-float(price)),
                note="Mark-basis diagnostic, not realized product PnL, conversion value or optimal exit counterfactual."))
    pd.DataFrame(summaries).to_csv(report / "five_maturity_actual_inventory.csv", index=False)
    pd.DataFrame(exposure).to_csv(report / "maturity_mark_basis_risk.csv", index=False)
    keys = ["date", "asset_code", "ts_code"]
    delta = before[keys].merge(missing[keys], on=keys, how="outer", indicator=True, validate="one_to_one")
    delta.groupby(["ts_code", "_merge"], observed=True).size().rename("position_days").reset_index().to_csv(
        report / "incremental_gap_changes_vs_v86.csv", index=False)
    end = missing.loc[missing.date.eq(daily.date.max())].copy()
    end.to_csv(report / "ending_unresolved_inventory.csv", index=False)
    quality = daily.assign(observed_mark_complete=daily.missing_held_count.eq(0),
        total_return_certified=False, eligible_for_formal_ppo_reward=False)
    quality[["date", "asset_code", "observed_mark_complete", "total_return_certified",
             "eligible_for_formal_ppo_reward"]].to_parquet(out / "accounting_quality_masks.parquet", index=False)
    result = dict(five_issuer_totals_exact=True, five_events_no_post_extinction_fills_or_inventory=True,
        prior_domestic_close_signal=True, fee_per_gross_fill=.0005, fee_reconciled_yuan=cost,
        no_early_or_duplicate_payment=True, tianma_unconfirmed_receivable_yuan=float(tianma.gross_claim_yuan.sum()),
        missing_position_days_v86=len(before), missing_position_days_v87=len(missing),
        removed_old_gap_days=int(delta._merge.eq("left_only").sum()),
        new_path_gap_days=int(delta._merge.eq("right_only").sum()), ending_missing_codes=int(end.ts_code.nunique()),
        formal_training_permitted=False,
        limitation="Real historical quote reconstruction in hypothetical basket books, not complete certified total returns.",
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths})
    save(report / "maturity_closure_audit.json", result)
    print(json.dumps({k:v for k,v in result.items() if k != "source_sha256"}, indent=2))
    print(pd.DataFrame(summaries).to_string(index=False))
    print(end[["ts_code", "asset_code", "quantity", "provisional_mark"]].to_string(index=False))


if __name__ == "__main__":
    main()

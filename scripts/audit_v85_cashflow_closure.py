"""Exact contract totals, actual fee reconciliation and unresolved-claim checks."""
from decimal import Decimal
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from scripts.run_v52_execution_preflight import sha, save
from scripts.replay_v85_remaining_terminal_events import CASES


def main():
    report = ROOT / "output/reports/cb_cashflows_v85"
    out = ROOT / "data_processed/cb_cashflows_v85"
    status_path = report / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status["status"] != "research_accounting_completed_formal_gate_blocked":
        raise ValueError("Replay incomplete")
    paths = [status_path, Path(__file__), report / "actual_registered_entitlements.csv",
             report / "event_transitions.csv", out / "reviewed_terminal_events.parquet",
             out / "daily_accounting_diagnostics.parquet", out / "actual_fills.parquet",
             out / "unresolved_inventory.parquet",
             ROOT / "data_processed/cb_evidence_v50/last_known_balance_yuan.parquet",
             ROOT / "data_processed/cb_cashflows_v83/unresolved_inventory.parquet",
             ROOT / "output/reports/cb_cashflows_v84/primary_evidence_review.json"]
    for name, digest in status["source_sha256"].items():
        if sha(ROOT / name) != digest:
            raise ValueError(f"Input changed: {name}")
    exact_checks = []
    for case in CASES:
        if case["bonds"] is None:
            continue
        expected = Decimal(case["bonds"]) * Decimal(case["unit"])
        if expected != Decimal(case["aggregate"]):
            raise ValueError("Contractual unit price does not reconcile exactly to published aggregate")
        exact_checks.append(dict(ts_code=case["code"], contractual_total_exact_match=True,
                                 amount_yuan=str(expected)))
    review = json.loads(paths[-1].read_text(encoding="utf-8"))
    geli = next(r for r in review["events"] if r["ts_code"] == "110030.SH")
    assert Decimal(geli["issuer_bonds"]) * Decimal(str(geli["scheduled_gross_yuan_per_bond"])) == Decimal(str(geli["reported_gross_yuan"]))
    rights = pd.read_csv(paths[2])
    transitions = pd.read_csv(paths[3], parse_dates=["day"])
    events, daily, fills, missing = [pd.read_parquet(p) for p in paths[4:8]]
    calendar = pd.read_parquet(paths[8]).index
    positions = calendar.get_indexer(pd.DatetimeIndex(daily.date))
    if not (positions > 0).all():
        raise ValueError("Execution outside domestic calendar")
    np.testing.assert_array_equal(pd.DatetimeIndex(daily.signal_date), calendar.take(positions - 1))
    trade_costs = (fills.quantity_change.abs() * fills.execution_close * .0005).sum()
    np.testing.assert_allclose(daily.cost_yuan.sum(), trade_costs, atol=1e-6, rtol=1e-12)
    payments = transitions.loc[transitions.transition.eq("confirmed_receivable_to_cash")]
    checks = payments.merge(events[["event_id", "confirmed_at", "pay_date"]], on="event_id",
                            validate="one_to_one", suffixes=("_transition", "_contract"))
    if not (checks.day.ge(checks.confirmed_at_contract) & checks.day.ge(checks.pay_date)).all():
        raise ValueError("Early or unconfirmed payment")
    tianma = rights.loc[rights.ts_code.eq("113507.SH")]
    if tianma.settled.any() or payments.ts_code.eq("113507.SH").any():
        raise ValueError("Tianma contradictory aggregate incorrectly paid")
    # Retirements may change later holdings; preserve newly exposed failures.
    before = pd.read_parquet(paths[9])
    keys = ["date", "asset_code", "ts_code"]
    delta = before[keys].merge(missing[keys], on=keys, how="outer", indicator=True, validate="one_to_one")
    delta.groupby(["ts_code", "_merge"], observed=True).size().rename("position_days").reset_index().to_csv(
        report / "incremental_gap_changes_vs_v83.csv", index=False)
    result = dict(exact_contract_totals=exact_checks, geli_exact_contract_total=True,
        tianma_confirmation_blocked=True, tianma_outstanding_gross_claim_yuan=float(tianma.gross_claim_yuan.sum()),
        fee_rate_per_gross_fill=.0005, fee_reconciled_yuan=float(trade_costs),
        prior_domestic_close_signal_every_day=True, no_early_or_duplicate_event_payment=True,
        missing_position_days_v83=len(before), missing_position_days_v85=len(missing),
        old_gap_days_removed_vs_v83=int(delta._merge.eq("left_only").sum()),
        new_path_gap_days_vs_v83=int(delta._merge.eq("right_only").sum()),
        statement="Accounting diagnostics only; no Sharpe, formal training or promotion.",
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths})
    save(report / "cashflow_closure_audit.json", result)
    print(json.dumps({k:v for k,v in result.items() if k != "source_sha256"}, indent=2))


if __name__ == "__main__":
    main()

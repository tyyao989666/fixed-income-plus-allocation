"""Validate V88 accounting and shadow-run schedule-only controls without trading them."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from scripts.run_v52_execution_preflight import sha, save
from scripts.replay_v88_inventory_maturities import CASES, exact_total
from src.v88_terminal_controls import terminal_controls


def main():
    report = ROOT / "output/reports/cb_cashflows_v88"
    out = ROOT / "data_processed/cb_cashflows_v88"
    status_path = report / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status["status"] != "research_accounting_completed_formal_gate_blocked":
        raise ValueError("Replay incomplete")
    for name, expected in status["source_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"Source changed: {name}")
    paths = [status_path, Path(__file__), ROOT / "scripts/replay_v88_inventory_maturities.py",
        ROOT / "src/v88_terminal_controls.py", out / "daily_accounting_diagnostics.parquet",
        out / "actual_fills.parquet", out / "reviewed_terminal_events.parquet",
        out / "unresolved_inventory.parquet", report / "actual_registered_entitlements.csv",
        report / "event_transitions.csv", ROOT / "data_processed/cb_cashflows_v87/unresolved_inventory.parquet",
        ROOT / "data_processed/cb_evidence_v50/last_known_balance_yuan.parquet",
        report / "schedule_only_execution_controls.parquet"]
    daily, fills, events, missing = [pd.read_parquet(p) for p in paths[4:8]]
    rights = pd.read_csv(paths[8])
    transitions = pd.read_csv(paths[9], parse_dates=["day"])
    before = pd.read_parquet(paths[10])
    sessions = pd.read_parquet(paths[11]).index
    schedules = pd.read_parquet(paths[12])
    positions = sessions.get_indexer(pd.DatetimeIndex(daily.date))
    if not (positions > 0).all():
        raise ValueError("Execution not in domestic calendar")
    np.testing.assert_array_equal(pd.DatetimeIndex(daily.signal_date), sessions.take(positions-1))
    cost = float((fills.quantity_change.abs() * fills.execution_close * .0005).sum())
    np.testing.assert_allclose(cost, daily.cost_yuan.sum(), atol=1e-6, rtol=1e-12)
    payments = transitions.loc[transitions.transition.eq("confirmed_receivable_to_cash")]
    checks = payments.merge(events[["event_id", "confirmed_at", "pay_date"]], on="event_id",
                            validate="one_to_one", suffixes=("_transition", "_contract"))
    if not (checks.day.ge(checks.confirmed_at_contract) & checks.day.ge(checks.pay_date)).all():
        raise ValueError("Early payment or duplicate confirmation")
    pending_codes = {"113507.SH", "127015.SZ"}
    pending = rights.loc[rights.ts_code.isin(pending_codes)]
    if pending.settled.any() or payments.ts_code.isin(pending_codes).any():
        raise ValueError("Unresolved aggregate contradiction paid")
    np.testing.assert_allclose(daily.loc[daily.date.eq(daily.date.max()), "receivable_yuan"].sum(),
                               pending.gross_claim_yuan.sum(), atol=1e-7)
    summaries = []
    for case in CASES:
        code = case["code"]
        e = events.loc[events.ts_code.eq(code)].iloc[0]
        r = rights.loc[rights.ts_code.eq(code)]
        if ((missing.ts_code.eq(code) & missing.date.ge(e.ex_date)).any()
                or (fills.ts_code.eq(code) & fills.date.ge(e.ex_date)).any()):
            raise ValueError("Extinguished bond inventory/trades remain")
        is_paid = pd.notna(e.confirmed_at)
        expected_cash = float(r.gross_claim_yuan.sum()) if is_paid else 0.
        np.testing.assert_allclose(payments.loc[payments.ts_code.eq(code), "amount_all_books"].sum(),
                                   expected_cash, atol=1e-7)
        summaries.append(dict(ts_code=code, registered_quantity=r.registered_quantity.sum(),
            registered_gross_claim_yuan=r.gross_claim_yuan.sum(), gross_confirmed_cash_yuan=expected_cash,
            confirmed_at=e.confirmed_at, pending_conflict=not is_paid, **exact_total(case)))
    pd.DataFrame(summaries).to_csv(report / "three_maturity_actual_inventory.csv", index=False)
    keys = ["date", "asset_code", "ts_code"]
    delta = before[keys].merge(missing[keys], on=keys, how="outer", indicator=True, validate="one_to_one")
    delta.groupby(["ts_code", "_merge"], observed=True).size().rename("position_days").reset_index().to_csv(
        report / "incremental_gap_changes_vs_v87.csv", index=False)
    end = missing.loc[missing.date.eq(daily.date.max())].copy()
    end.to_csv(report / "ending_unresolved_inventory.csv", index=False)

    codes = list(schedules.ts_code)
    controls = []
    for i in range(1, len(sessions)):
        signal, execution = sessions[i-1], sessions[i]
        frame = terminal_controls(codes, schedules, signal, execution, sessions)
        frame["signal_date"], frame["execution_date"] = signal, execution
        controls.append(frame.reset_index())
    control_path = pd.concat(controls, ignore_index=True)
    control_path.to_parquet(out / "schedule_controls_shadow_only.parquet", index=False)
    control_summary = []
    for code in codes:
        rows = control_path.loc[control_path.ts_code.eq(code)]
        requests = rows.loc[rows.request_liquidation]
        frozen = rows.loc[rows.needs_frozen_inventory_review]
        schedule = schedules.loc[schedules.ts_code.eq(code)].iloc[0]
        known = pd.Timestamp(schedule.known_at)
        if rows.loc[rows.signal_date.lt(known), ["block_new_buy", "request_liquidation", "block_execution"]].to_numpy().any():
            raise ValueError("Schedule controls used unseen evidence")
        if requests.execution_date.gt(pd.Timestamp(schedule.last_trade_date)).any():
            raise ValueError("Proposed exit after trading ended")
        control_summary.append(dict(ts_code=code, known_at=known, last_trade_date=schedule.last_trade_date,
            first_possible_exit_request=requests.execution_date.min(), exit_request_days=len(requests),
            first_frozen_review_day=frozen.execution_date.min(), used_in_accounting_replay=False,
            guaranteed_executable_fill=False))
    pd.DataFrame(control_summary).to_csv(report / "schedule_controls_shadow_summary.csv", index=False)
    result = dict(two_new_paid_events_exact=True, hope_conflict_pending_not_paid=True,
        no_post_extinction_trades_or_inventory=True, prior_domestic_close_signal=True,
        fee_per_gross_fill=.0005, fee_reconciled_yuan=cost, no_early_or_duplicate_payment=True,
        pending_conflict_codes=sorted(pending_codes), pending_gross_claim_yuan=float(pending.gross_claim_yuan.sum()),
        missing_position_days_v87=len(before), missing_position_days_v88=len(missing),
        removed_old_gap_days=int(delta._merge.eq("left_only").sum()),
        new_path_gap_days=int(delta._merge.eq("right_only").sum()), ending_missing_codes=int(end.ts_code.nunique()),
        shadow_controls_ignore_unseen_schedules=True, shadow_controls_rows=len(control_path),
        shadow_controls_used_in_trading=False, formal_training_permitted=False,
        limitation="Accounting and shadow instruction audit, not a certified strategy backtest or control-policy performance test.",
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths})
    save(report / "maturity_and_controls_audit.json", result)
    print(json.dumps({k:v for k,v in result.items() if k != "source_sha256"}, indent=2))
    print(pd.DataFrame(summaries).to_string(index=False))
    print(pd.DataFrame(control_summary).to_string(index=False))


if __name__ == "__main__":
    main()

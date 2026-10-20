"""Apply reviewed redemption evidence to actual V51 basket inventory, not toy holdings."""
from pathlib import Path
import copy
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from scripts.run_v52_execution_preflight import save, sha
from src.v49_point_in_time import cb_bucket_weights
from src.v51_cb_policy import buy_eligibility
from src.v52_cb_execution import QuantityBooks, EVENT_COLUMNS
from src.v82_causal_cb_events import CausalQuantityBooks, validate_causal_events


def main(extra_events=None, extra_sources=(), version="v82"):
    started = time.monotonic()
    report = ROOT / f"output/reports/cb_cashflows_{version}"
    output = ROOT / f"data_processed/cb_cashflows_{version}"
    report.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    status_path = report / "status.json"
    status = dict(status="running", training_started=False, ppo_started=False, formal_promotion=False)
    save(status_path, status)
    sources = set()
    try:
        registrations = ["output/reports/cb_masked_research_v51/status.json",
                         "output/reports/terminal_books_v69/status.json",
                         "output/reports/terminal_batch_v70/status.json"]
        for relative in registrations:
            path = ROOT / relative
            sources.add(path)
            data = json.loads(path.read_text(encoding="utf-8"))
            for name, expected in data.get("source_sha256", data.get("source_hashes", {})).items():
                original = ROOT / name
                if sha(original) != expected:
                    raise ValueError(f"Immutable source changed: {name}")
                sources.add(original)
        registration = json.loads((ROOT / registrations[0]).read_text(encoding="utf-8"))
        base = ROOT / "data_processed/cb_evidence_v50"

        def read(path):
            sources.add(path)
            return pd.read_parquet(path)

        balance = read(base / "last_known_balance_yuan.parquet")
        size = read(base / "last_known_issue_size_yuan.parquet").reindex_like(balance)
        close = read(base / "quotes/close_individual.parquet").reindex_like(balance)
        volume = read(base / "quotes/volume_individual.parquet").reindex_like(balance)
        excluded = read(base / "known_exit_exclusions.parquet").reindex_like(balance)
        eligibility = read(ROOT / "data_processed/cb_masked_research_v51/signal_buy_eligibility.parquet")
        if not (eligibility.index.equals(close.index) and eligibility.columns.equals(close.columns)):
            raise ValueError("Eligibility coordinate mismatch")
        dates, codes, names = close.index, list(close.columns), registration["cb_assets"]
        executable = close.gt(0) & np.isfinite(close) & volume.gt(0) & np.isfinite(volume)
        valuation = close.where(executable)
        marks = read(base / "nontransaction_marks_for_inventory_review_only.parquet")
        verified = marks.loc[marks.anchor_verified.eq(True) & marks.positive_volume.eq(False)]
        if verified.groupby(["date", "ts_code"]).close.nunique().gt(1).any():
            raise ValueError("Conflicting nontransaction marks")
        for row in verified.drop_duplicates(["date", "ts_code"]).itertuples(index=False):
            if row.date in dates and row.ts_code in codes and pd.isna(valuation.at[row.date, row.ts_code]):
                valuation.at[row.date, row.ts_code] = row.close

        reviewed_path = ROOT / "output/reports/terminal_batch_v70/reviewed_events.json"
        sources.add(reviewed_path)
        reviewed = json.loads(reviewed_path.read_text(encoding="utf-8"))
        event_rows = []
        for code, pair in reviewed.items():
            schedule, result = pair["schedule"], pair["result"]
            published = pd.Timestamp(result["result_publication_date"])
            later = dates[dates > published]
            if not len(later):
                raise ValueError("Result confirmation is outside available calendar")
            for key in ("record_date", "redemption_date", "payment_date", "gross_cash_yuan_per_bond"):
                if schedule[key] != result[key]:
                    raise ValueError(f"Schedule/result conflict: {code} {key}")
            event_rows.append(dict(event_id=f"reviewed_terminal_{code}_{schedule['record_date']}",
                ts_code=code, kind="terminal", known_at=schedule["conservative_available_after_close"],
                record_date=schedule["record_date"], ex_date=schedule["redemption_date"],
                pay_date=schedule["payment_date"], confirmed_at=later[0],
                cash_yuan_per_bond=schedule["gross_cash_yuan_per_bond"],
                source=str(reviewed_path.relative_to(ROOT)), result_publication_date=published,
                cash_basis="gross_before_investor_specific_tax",
                confirmation_basis="issuer_result_announcement_not_personal_account_statement"))
        if extra_events is not None:
            event_rows.extend(extra_events)
            sources.update(extra_sources)
        events = validate_causal_events(pd.DataFrame(event_rows), codes)
        events.to_parquet(output / "reviewed_terminal_events.parquet", index=False)
        events.to_csv(report / "reviewed_terminal_events.csv", index=False)
        sources.update([Path(__file__), ROOT / "src/v82_causal_cb_events.py",
                        ROOT / "src/v52_cb_execution.py", ROOT / "src/v49_point_in_time.py",
                        ROOT / "src/v51_cb_policy.py"])
        hashes = {str(p.relative_to(ROOT)): sha(p) for p in sorted(sources)}
        book = CausalQuantityBooks(names, codes, events, dates)
        baseline = QuantityBooks(names, codes)
        empty = pd.DataFrame(columns=EVENT_COLUMNS)
        records, transitions, unresolved, trades, causal_checks = [], [], [], [], []
        checkpoints = {dates[dates.get_loc(t) - 1] for t in events.confirmed_at.dropna()}
        for i in range(1, len(dates)):
            signal, day = dates[i - 1], dates[i]
            allowed = eligibility.loc[signal]
            expected, _ = buy_eligibility(close.loc[signal], volume.loc[signal], balance.loc[signal],
                                           size.loc[signal], excluded.loc[signal])
            if not allowed.equals(expected):
                raise ValueError("Stored signal differs from previous-close eligibility")
            groups = cb_bucket_weights(close.loc[signal].where(allowed), balance.loc[signal], size.loc[signal])
            target = np.zeros_like(book.quantity)
            for a, name in enumerate(names):
                if name in groups:
                    target[a] = groups[name]["weights"].reindex(codes, fill_value=0).to_numpy()
            arguments = (day, target, allowed.to_numpy(), close.loc[day].to_numpy(),
                         executable.loc[day].to_numpy(), valuation.loc[day].to_numpy())
            truncated = None
            if day in checkpoints:
                truncated = copy.deepcopy(book)
                truncated.events = truncated.events.loc[truncated.events.known_at <= day].copy()
                truncated.events.loc[truncated.events.confirmed_at > day, "confirmed_at"] = pd.NaT
            result = book.step(*arguments, signal_date=signal)
            control = baseline.step(*arguments, empty)
            if truncated is not None:
                check = truncated.step(*arguments, signal_date=signal)
                for key in ("provisional_nav", "cash", "receivable", "fills", "payment_cash"):
                    np.testing.assert_allclose(result[key], check[key], atol=1e-8, rtol=1e-12)
                np.testing.assert_allclose(book.quantity, truncated.quantity, atol=1e-10, rtol=1e-12)
                causal_checks.append(dict(date=day, future_confirmations_removed=True, passed=True))
            transitions.extend(result["event_transitions"])
            for a, name in enumerate(names):
                records.append(dict(date=day, signal_date=signal, asset_code=name,
                    provisional_nav=result["provisional_nav"][a] / 1e6,
                    baseline_provisional_nav=control["provisional_nav"][a] / 1e6,
                    observed_price_nav=result["observed_price_nav"][a] / 1e6,
                    certified_total_return_nav=np.nan,
                    settlement_cash_yuan=result["cash"][a], receivable_yuan=result["receivable"][a],
                    payment_cash_yuan=result["payment_cash"][a], cost_yuan=result["cost_yuan"][a],
                    missing_held_count=int(result["missing_held_count"][a]),
                    baseline_missing_held_count=int(control["missing_held_count"][a])))
                for c in np.flatnonzero(result["stale_valuation_positions"][a]):
                    unresolved.append(dict(date=day, asset_code=name, ts_code=codes[c],
                        quantity=book.quantity[a, c], provisional_mark=book.last_mark[c]))
                for c in np.flatnonzero(np.abs(result["fills"][a]) > 1e-12):
                    trades.append((day, signal, name, codes[c], result["fills"][a, c], close.at[day, codes[c]]))
            if i % 200 == 0 or i == len(dates) - 1:
                status.update(completed_days=i, total_days=len(dates)-1,
                    elapsed_seconds=round(time.monotonic()-started, 1))
                save(status_path, status)
                print(f"{version.upper()} actual inventory {i}/{len(dates)-1}, {day.date()}", flush=True)
        frame = pd.DataFrame(records)
        frame.to_parquet(output / "daily_accounting_diagnostics.parquet", index=False)
        pd.DataFrame(unresolved).to_parquet(output / "unresolved_inventory.parquet", index=False)
        pd.DataFrame(trades, columns=["date", "signal_date", "asset_code", "ts_code", "quantity_change",
                                      "execution_close"]).to_parquet(output / "actual_fills.parquet", index=False)
        pd.DataFrame(book.quantity, index=names, columns=codes).to_parquet(output / "final_quantities.parquet")
        pd.DataFrame(transitions).to_csv(report / "event_transitions.csv", index=False)
        pd.DataFrame(causal_checks).to_csv(report / "future_confirmation_truncation_checks.csv", index=False)
        entitlements = []
        for row in events.itertuples(index=False):
            owned = book.entitlements[row.event_id]
            for a, name in enumerate(names):
                entitlements.append(dict(event_id=row.event_id, ts_code=row.ts_code, asset_code=name,
                    record_date=row.record_date, registered_quantity=owned[a],
                    gross_claim_yuan=owned[a]*row.cash_yuan_per_bond, confirmed_at=row.confirmed_at,
                    settled=row.event_id in book.paid))
        rights = pd.DataFrame(entitlements)
        rights.to_csv(report / "actual_registered_entitlements.csv", index=False)
        summary = frame.groupby("asset_code").agg(days=("date", "size"),
            missing_position_days=("missing_held_count", "sum"),
            baseline_missing_position_days=("baseline_missing_held_count", "sum"),
            gross_confirmed_payment_yuan=("payment_cash_yuan", "sum"),
            execution_cost_yuan=("cost_yuan", "sum"))
        summary.to_csv(report / "basket_accounting_comparison_NOT_performance.csv")
        for name, expected in hashes.items():
            if sha(ROOT / name) != expected:
                raise ValueError(f"Source changed during replay: {name}")
        paid_transitions = [t for t in transitions if t["transition"] == "confirmed_receivable_to_cash"]
        assert all(t["day"] >= max(t["confirmed_at"], t["scheduled_pay_date"]) for t in paid_transitions)
        assert len({t["event_id"] for t in paid_transitions}) == len(paid_transitions)
        status.update(status="research_accounting_completed_formal_gate_blocked",
            calendar_start=str(dates[0].date()), calendar_end=str(dates[-1].date()),
            basket_count=len(names), security_count=len(codes), reviewed_terminal_events=len(events),
            nonzero_event_basket_entitlements=int(rights.registered_quantity.gt(1e-12).sum()),
            nonzero_event_codes=int(rights.loc[rights.registered_quantity.gt(1e-12), "ts_code"].nunique()),
            gross_confirmed_payment_yuan=float(frame.payment_cash_yuan.sum()),
            unresolved_position_days=int(frame.missing_held_count.sum()),
            baseline_unresolved_position_days=int(frame.baseline_missing_held_count.sum()),
            ending_receivable_yuan=float(frame.loc[frame.date.eq(dates[-1]), "receivable_yuan"].sum()),
            future_confirmation_truncation_checks=len(causal_checks), source_sha256=hashes,
            blockers=["complete_coupon_history_not_certified", "remaining_terminal_events_not_certified",
                      "remaining_missing_inventory_valuations", "other_38_asset_cashflow_execution_contracts_incomplete"],
            limitation="Issuer-confirmed gross redemption reconstruction, not personal cash receipts or certified strategy performance; no Sharpe calculated.",
            elapsed_seconds=round(time.monotonic()-started, 1))
        save(status_path, status)
        print(json.dumps({k:v for k,v in status.items() if k != "source_sha256"}, indent=2))
    except Exception as exc:
        status.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        save(status_path, status)
        raise


if __name__ == "__main__":
    main()

"""Rebuild all 12 CB books after the user's explicit three-name exclusion.

Preserves historical evidence and old outputs. This retrospective research screen
must never be presented as an untouched out-of-sample universe.
"""
from pathlib import Path
import hashlib
import json
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.v92_cb_research_universe import apply_exclusions
from src.v49_point_in_time import cb_bucket_weights
from src.v51_cb_policy import buy_eligibility
from src.v82_causal_cb_events import CausalQuantityBooks

OUT = ROOT / "output/reports/cb_cashflows_v92"
DATA = ROOT / "data_processed/cb_cashflows_v92"


def main():
    started = time.monotonic()
    for folder in [OUT, DATA]:
        folder.mkdir(parents=True, exist_ok=True)
    bound = {}
    def bind(relative):
        path = ROOT / relative
        bound[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path
    def save(status):
        (OUT / "status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    policy = json.loads(bind("config/cb_research_exclusions_v92.json").read_text(encoding="utf-8"))
    removed = policy["excluded_securities"]
    if set(removed) != {"123015.SZ", "128100.SZ", "110072.SH"} or not policy["retrospective_universe_selection"]:
        raise ValueError("Unexpected authorized scope or missing retrospective disclosure")
    parent = json.loads(bind("output/reports/cb_cashflows_v88/status.json").read_text(encoding="utf-8"))
    for relative, expected in parent["source_sha256"].items():
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Parent source changed: {relative}")
        bound[relative] = expected
    registration = json.loads(bind("output/reports/cb_masked_research_v51/status.json").read_text(encoding="utf-8"))
    def read(relative):
        return pd.read_parquet(bind(relative))
    base = "data_processed/cb_evidence_v50/"
    balance = read(base + "last_known_balance_yuan.parquet")
    raw_close = read(base + "quotes/close_individual.parquet")
    if not balance.index.equals(raw_close.index) or not set(balance.columns).issubset(raw_close.columns):
        raise ValueError("Original research coordinates absent from quote source")
    close = raw_close.reindex_like(balance)
    volume = read(base + "quotes/volume_individual.parquet").reindex_like(close)
    size = read(base + "last_known_issue_size_yuan.parquet").reindex_like(close)
    exits = read(base + "known_exit_exclusions.parquet").reindex_like(close)
    old_allowed = read("data_processed/cb_masked_research_v51/signal_buy_eligibility.parquet")
    if not old_allowed.index.equals(close.index) or not old_allowed.columns.equals(close.columns):
        raise ValueError("Input coordinates differ")
    allowed = apply_exclusions(old_allowed, removed)
    allowed.to_parquet(DATA / "signal_buy_eligibility.parquet")
    retained = [c for c in close if c not in removed]
    pd.testing.assert_frame_equal(allowed[retained], old_allowed[retained])
    executable = close.gt(0) & np.isfinite(close) & volume.gt(0) & np.isfinite(volume)
    valuation = close.where(executable)
    marks = read(base + "nontransaction_marks_for_inventory_review_only.parquet")
    marks = marks.loc[marks.anchor_verified.eq(True) & marks.positive_volume.eq(False)]
    if marks.groupby(["date", "ts_code"]).close.nunique().gt(1).any():
        raise ValueError("Conflicting valuation evidence")
    for row in marks.drop_duplicates(["date", "ts_code"]).itertuples(index=False):
        if row.date in close.index and row.ts_code in close.columns and pd.isna(valuation.at[row.date, row.ts_code]):
            valuation.at[row.date, row.ts_code] = row.close
    events = read("data_processed/cb_cashflows_v88/reviewed_terminal_events.parquet")
    codes, names, dates = list(close.columns), registration["cb_assets"], close.index
    if len(names) != 12 or len(registration["context_assets"]) != 38:
        raise ValueError("Asset structure changed")
    book = CausalQuantityBooks(names, codes, events, dates)
    blocked_indices = [codes.index(c) for c in removed]
    records, gaps, fills, transitions = [], [], [], []
    status = dict(status="running", retrospective_universe_selection=True, training_started=False,
                  strict_no_lookahead_claim_allowed=False, excluded_securities=removed)
    save(status)
    try:
        for i in range(1, len(dates)):
            signal, day = dates[i - 1], dates[i]
            expected, _ = buy_eligibility(close.loc[signal], volume.loc[signal], balance.loc[signal], size.loc[signal], exits.loc[signal])
            if not expected.equals(old_allowed.loc[signal]):
                raise ValueError("Original point-in-time eligibility mismatch")
            selected = allowed.loc[signal]
            groups = cb_bucket_weights(close.loc[signal].where(selected), balance.loc[signal], size.loc[signal])
            target = np.zeros_like(book.quantity)
            for a, name in enumerate(names):
                if name in groups:
                    target[a] = groups[name]["weights"].reindex(codes, fill_value=0).to_numpy()
            if np.any(target[:, blocked_indices] != 0):
                raise ValueError("Excluded security entered target")
            if not np.isfinite(target).all() or (target < 0).any() or not np.isclose(target.sum(1)[target.sum(1) > 0], 1).all():
                raise ValueError("Invalid basket target")
            result = book.step(day, target, selected.to_numpy(), close.loc[day].to_numpy(),
                               executable.loc[day].to_numpy(), valuation.loc[day].to_numpy(), signal_date=signal)
            if np.any(book.quantity[:, blocked_indices] != 0) or np.any(result["fills"][:, blocked_indices] != 0):
                raise ValueError("Excluded security has inventory/fills")
            if (result["cash"] < -1e-6).any():
                raise ValueError("Negative settlement cash")
            transitions.extend(result["event_transitions"])
            for a, name in enumerate(names):
                records.append(dict(date=day, signal_date=signal, asset_code=name,
                    provisional_nav=result["provisional_nav"][a] / 1e6,
                    observed_price_nav=result["observed_price_nav"][a] / 1e6,
                    certified_total_return_nav=np.nan, settlement_cash_yuan=result["cash"][a],
                    receivable_yuan=result["receivable"][a], payment_cash_yuan=result["payment_cash"][a],
                    cost_yuan=result["cost_yuan"][a], missing_held_count=int(result["missing_held_count"][a])))
                for c in np.flatnonzero(result["stale_valuation_positions"][a]):
                    gaps.append((day, name, codes[c], book.quantity[a, c], book.last_mark[c]))
                for c in np.flatnonzero(np.abs(result["fills"][a]) > 1e-12):
                    fills.append((day, signal, name, codes[c], result["fills"][a, c], close.at[day, codes[c]]))
            if i % 300 == 0 or i == len(dates) - 1:
                status.update(completed_days=i, total_days=len(dates) - 1, elapsed_seconds=round(time.monotonic()-started, 1))
                save(status)
                print(f"V92 excluded universe {i}/{len(dates)-1} {day.date()}", flush=True)
        frame = pd.DataFrame(records)
        frame.to_parquet(DATA / "daily_accounting_diagnostics.parquet", index=False)
        gap = pd.DataFrame(gaps, columns=["date", "asset_code", "ts_code", "quantity", "provisional_mark"])
        gap.to_parquet(DATA / "unresolved_inventory.parquet", index=False)
        trades = pd.DataFrame(fills, columns=["date", "signal_date", "asset_code", "ts_code", "quantity_change", "execution_close"])
        trades.to_parquet(DATA / "actual_fills.parquet", index=False)
        pd.DataFrame(book.quantity, index=names, columns=codes).to_parquet(DATA / "final_quantities.parquet")
        ending = gap.loc[gap.date.eq(dates[-1])]
        ending.to_csv(OUT / "ending_unresolved_inventory.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(transitions).to_csv(OUT / "event_transitions.csv", index=False)
        frame.groupby("asset_code").agg(missing_position_days=("missing_held_count", "sum"),
            cost_yuan=("cost_yuan", "sum")).to_csv(OUT / "basket_accounting_NOT_performance.csv")
        for relative in ["scripts/replay_v92_excluded_cb_universe.py", "src/v92_cb_research_universe.py", "tests/test_research_exclusions_v92.py"]:
            bind(relative)
        for relative, expected in bound.items():
            if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Input changed during replay: {relative}")
        status.update(status="retrospective_screen_replayed_formal_gate_blocked", asset_structure_count=38,
                      basket_count=len(names), universe_before=len(codes), universe_after=len(retained),
                      excluded_fill_rows=int(trades.ts_code.isin(removed).sum()),
                      excluded_final_quantity=float(book.quantity[:, blocked_indices].sum()),
                      unresolved_position_days=len(gap), previous_unresolved_position_days=parent["unresolved_position_days"],
                      ending_unresolved_securities=sorted(ending.ts_code.unique()),
                      ending_receivable_yuan=float(frame.loc[frame.date.eq(dates[-1]), "receivable_yuan"].sum()),
                      raw_data_deleted=False, previous_outputs_overwritten=False,
                      formal_training_allowed=False, certified_total_return=False,
                      calendar_start=str(dates[0].date()), calendar_end=str(dates[-1].date()),
                      source_sha256=bound, elapsed_seconds=round(time.monotonic()-started, 1))
        save(status)
        print(json.dumps({k: v for k, v in status.items() if k != "source_sha256"}, ensure_ascii=False))
    except Exception as exc:
        status.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        save(status)
        raise


if __name__ == "__main__":
    main()

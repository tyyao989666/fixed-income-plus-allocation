"""Replay all CB quantity books before permitting any retraining or portfolio test."""
from pathlib import Path
import hashlib
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from src.v49_point_in_time import cb_bucket_weights
from src.v51_cb_policy import buy_eligibility
from src.v52_cb_execution import QuantityBooks, EVENT_COLUMNS, validate_events


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    start = time.monotonic()
    source = ROOT / "data_processed/cb_evidence_v50"
    out = ROOT / "data_processed/cb_execution_v52"
    report = ROOT / "output/reports/cb_execution_v52"
    out.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    status_path = report / "pipeline_status.json"
    status = dict(status="running", stage="execution_replay", completed_days=0,
        training_started=False, ppo_started=False, portfolio_backtest_started=False,
        formal_promotion=False, stages=[dict(name="data_and_execution", status="running")])
    save(status_path, status)
    try:
        previous = ROOT / "output/reports/cb_masked_research_v51/status.json"
        registration = json.loads(previous.read_text(encoding="utf-8"))
        for name, expected in registration["source_sha256"].items():
            if sha(ROOT / name) != expected:
                raise ValueError(f"V51 source changed: {name}")
        eligibility = pd.read_parquet(ROOT / "data_processed/cb_masked_research_v51/signal_buy_eligibility.parquet")
        balance = pd.read_parquet(source / "last_known_balance_yuan.parquet")
        size = pd.read_parquet(source / "last_known_issue_size_yuan.parquet").reindex_like(balance)
        close = pd.read_parquet(source / "quotes/close_individual.parquet").reindex_like(balance)
        volume = pd.read_parquet(source / "quotes/volume_individual.parquet").reindex_like(balance)
        preclose = pd.read_parquet(source / "quotes/preclose_individual.parquet").reindex_like(balance)
        excluded = pd.read_parquet(source / "known_exit_exclusions.parquet").reindex_like(balance)
        if not (eligibility.index.equals(close.index) and eligibility.columns.equals(close.columns)):
            raise ValueError("Eligibility and execution source must have identical coordinates")
        names, codes, dates = registration["cb_assets"], list(close.columns), close.index
        executable = close.gt(0) & np.isfinite(close) & volume.gt(0) & np.isfinite(volume)
        valuation = close.where(executable)
        marks = pd.read_parquet(source / "nontransaction_marks_for_inventory_review_only.parquet")
        verified = marks.loc[marks.anchor_verified.eq(True) & marks.positive_volume.eq(False)]
        if verified.duplicated(["date", "ts_code"]).any():
            counts = verified.groupby(["date", "ts_code"]).close.nunique()
            if counts.gt(1).any():
                raise ValueError("Conflicting nontransaction valuation evidence")
            verified = verified.drop_duplicates(["date", "ts_code"])
        for row in verified.itertuples(index=False):
            if row.date in dates and row.ts_code in codes and pd.isna(valuation.at[row.date, row.ts_code]):
                valuation.at[row.date, row.ts_code] = row.close
        events_path = out / "adjudicated_cashflows.parquet"
        # An absent event database is UNKNOWN, never an asserted history of zero coupons.
        events = validate_events(pd.read_parquet(events_path) if events_path.exists()
                                 else pd.DataFrame(columns=EVENT_COLUMNS), codes)
        pd.DataFrame(columns=EVENT_COLUMNS).to_csv(report / "required_cashflow_schema.csv", index=False)
        book = QuantityBooks(names, codes)
        records, unresolved, trades, gaps = [], [], [], []
        held_codes = set()
        position_days = np.zeros(len(codes), dtype=np.int64)
        for i in range(1, len(dates)):
            signal, day = dates[i - 1], dates[i]
            allowed = eligibility.loc[signal]
            expected, _ = buy_eligibility(close.loc[signal], volume.loc[signal], balance.loc[signal], size.loc[signal], excluded.loc[signal])
            if not allowed.equals(expected):
                raise ValueError("Stored eligibility differs from causal signal snapshot")
            groups = cb_bucket_weights(close.loc[signal].where(allowed), balance.loc[signal], size.loc[signal])
            target = np.zeros_like(book.quantity)
            for a, name in enumerate(names):
                if name in groups:
                    target[a] = groups[name]["weights"].reindex(codes, fill_value=0).to_numpy()
            previous_quantity = book.quantity.copy()
            old_marks = book.last_mark.copy()
            data = book.step(day, target, allowed.to_numpy(), close.loc[day].to_numpy(),
                executable.loc[day].to_numpy(), valuation.loc[day].to_numpy(), events)
            held_any = (previous_quantity > 1e-12).any(axis=0)
            position_days += held_any
            held_codes.update(np.asarray(codes)[(book.quantity > 1e-12).any(axis=0)])
            p = preclose.loc[day].to_numpy()
            mismatch = held_any & np.isfinite(old_marks) & np.isfinite(p) & (p > 0) & ~np.isclose(old_marks, p, rtol=1e-6, atol=1e-5)
            for c in np.flatnonzero(mismatch):
                gaps.append(dict(date=day, ts_code=codes[c], previous_inventory_mark=float(old_marks[c]),
                    observed_preclose=float(p[c]), resolution="price_basis_or_corporate_action_requires_review"))
            for a, name in enumerate(names):
                records.append(dict(date=day, signal_date=signal, asset_code=name,
                    provisional_nav=data["provisional_nav"][a] / 1e6,
                    observed_price_nav=data["observed_price_nav"][a] / 1e6,
                    # Price evidence does not certify coupons or terminal settlement.
                    certified_total_return_nav=np.nan,
                    transaction_cost_yuan=data["cost_yuan"][a], traded_notional=data["traded_notional"][a],
                    settlement_cash_yuan=data["cash"][a], receivable_yuan=data["receivable"][a],
                    frozen_held_count=int(data["frozen_held_count"][a]),
                    missing_held_count=int(data["missing_held_count"][a]),
                    blocked_requested_notional=data["blocked_requested_notional"][a]))
                for c in np.flatnonzero(data["stale_valuation_positions"][a]):
                    unresolved.append(dict(date=day, asset_code=name, ts_code=codes[c],
                        quantity=float(previous_quantity[a, c]), last_mark=float(old_marks[c]),
                        resolution="inventory_retained_missing_current_valuation"))
                for c in np.flatnonzero(np.abs(data["fills"][a]) > 1e-12):
                    trades.append((day, signal, name, codes[c], float(data["fills"][a, c]), float(close.at[day, codes[c]])))
            if i % 200 == 0 or i == len(dates) - 1:
                status.update(completed_days=i, total_days=len(dates) - 1,
                    unresolved_position_days=len(unresolved), elapsed_seconds=round(time.monotonic() - start, 1))
                save(status_path, status)
                print(f"V52 inventory {i}/{len(dates)-1} {day.date()}; unresolved={len(unresolved)}", flush=True)
        frame = pd.DataFrame(records)
        frame.groupby("asset_code").agg(
            days=("date", "size"),
            missing_valuation_days=("missing_held_count", lambda x: int(x.gt(0).sum())),
            cumulative_execution_cost_yuan=("transaction_cost_yuan", "sum"),
            cumulative_two_way_notional_yuan=("traded_notional", "sum"),
            ending_provisional_nav=("provisional_nav", "last"),
        ).to_csv(report / "bucket_execution_summary_NOT_performance.csv")
        frame.to_parquet(out / "daily_price_only_execution_diagnostics.parquet", index=False)
        pd.DataFrame(trades, columns=["date", "signal_date", "asset_code", "ts_code", "quantity_change", "execution_close"]).to_parquet(out / "actual_close_fills.parquet", index=False)
        pd.DataFrame(unresolved, columns=["date", "asset_code", "ts_code", "quantity", "last_mark", "resolution"]).to_parquet(out / "unresolved_frozen_inventory.parquet", index=False)
        pd.DataFrame(gaps, columns=["date", "ts_code", "previous_inventory_mark", "observed_preclose", "resolution"]).to_csv(report / "price_basis_and_event_review.csv", index=False)
        pd.DataFrame(book.quantity, index=names, columns=codes).to_parquet(out / "final_quantities.parquet")
        covered = set(events.ts_code) if len(events) else set()
        pd.DataFrame(dict(ts_code=codes, held_book_days=position_days,
            ever_held=[c in held_codes for c in codes], has_any_adjudicated_cashflow=[c in covered for c in codes])).to_csv(report / "cashflow_evidence_coverage.csv", index=False)
        meta = json.loads((ROOT / "output/reports/v45_metadata.json").read_text(encoding="utf-8"))
        with np.load(ROOT / "data_processed/v45_dataset.npz", allow_pickle=False) as dataset:
            timeline = pd.to_datetime(dataset["dates"], unit="ns")
            coverage = []
            for a, asset in enumerate(meta["assets"]):
                idx = np.flatnonzero(dataset["return_masks"][:, a].astype(bool))
                coverage.append(dict(asset_code=asset, training_first=str(timeline[idx[0]].date()) if len(idx) else None,
                    training_last=str(timeline[idx[-1]].date()) if len(idx) else None,
                    v51_cb_research_cutoff="2026-10-14" if asset.startswith("cb_") else None,
                    formally_refreshed_to_requested_cutoff=False))
        pd.DataFrame(coverage).to_csv(report / "all_38_assets_training_coverage.csv", index=False)
        readiness = json.loads((ROOT / "output/reports/v49_project_review/data_readiness.json").read_text(encoding="utf-8"))
        # A documented proxy or an explicitly excluded macro field is not by
        # itself a reason to prohibit learning. Unverified actual P&L is.
        blockers = ["CB_complete_coupon_and_terminal_cashflow_history_not_certified",
            "REIT_historical_distribution_and_execution_returns_not_certified",
            "US_bond_return_currency_and_domestic_information_clock_not_certified",
            "38_asset_training_dataset_not_rebuilt_to_2026_10_14",
            "relative_labels_and_internal_CB_execution_costs_not_integrated_into_new_38_asset_dataset",
            "cashflow_event_presence_does_not_certify_complete_historical_payment_coverage"]
        if len(unresolved):
            blockers.append("actual_quantity_replay_has_unresolved_current_valuations")
        status.update(status="blocked_before_retraining", stage="data_and_execution_gate",
            stages=[dict(name="data_and_execution", status="research_replay_completed_formal_gate_failed"),
                    dict(name="rebuild_relative_training_set", status="blocked"),
                    dict(name="itransformer_training", status="not_started"),
                    dict(name="ppo_training", status="not_started"),
                    dict(name="three_product_ablation_backtest", status="not_started")],
            blockers=blockers, trading_books=len(names), calendar_days=len(dates),
            daily_records=len(frame), trade_rows=len(trades), ever_held_codes=len(held_codes),
            adjudicated_cashflow_events=len(events), unresolved_position_days=len(unresolved),
            price_basis_review_rows=len(gaps), elapsed_seconds=round(time.monotonic()-start, 1),
            limitation="Price-only inventory replay is diagnostic; no new formal Sharpe or total-return labels.",
            nonblocking_disclosures=["Unverified macro fields remain excluded from strict forecast inputs",
                "A consistently defined and disclosed 30Y duration-curve proxy is allowed; it is not an ETF"],
            legacy_readiness_blockers_for_traceability=readiness["blockers"],
            source_sha256={str(p.relative_to(ROOT)): sha(p) for p in [previous, Path(__file__), ROOT / "src/v52_cb_execution.py",
                source / "nontransaction_marks_for_inventory_review_only.parquet",
                ROOT / "data_processed/cb_masked_research_v51/signal_buy_eligibility.parquet"]})
        save(status_path, status)
        print(json.dumps({k:v for k,v in status.items() if k not in ["source_sha256", "blockers"]}, ensure_ascii=False, indent=2))
        return 2
    except Exception as exc:
        status.update(status="failed", error=f"{type(exc).__name__}: {exc}", elapsed_seconds=round(time.monotonic()-start, 1))
        save(status_path, status)
        raise


if __name__ == "__main__":
    sys.exit(main())

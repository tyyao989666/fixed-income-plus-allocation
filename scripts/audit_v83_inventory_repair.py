"""Reconcile the combined inventory replay and rank only actually held gaps."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from scripts.run_v52_execution_preflight import save, sha


def main(version="v83"):
    output = ROOT / f"data_processed/cb_cashflows_{version}"
    report = ROOT / f"output/reports/cb_cashflows_{version}"
    status_path = report / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status["status"] != "research_accounting_completed_formal_gate_blocked":
        raise ValueError("Replay incomplete")
    for name, expected in status["source_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"Replay source changed: {name}")
    paths = [output / "daily_accounting_diagnostics.parquet", output / "actual_fills.parquet",
             output / "final_quantities.parquet", output / "unresolved_inventory.parquet",
             report / "actual_registered_entitlements.csv", output / "reviewed_terminal_events.parquet",
             ROOT / "data_processed/cb_execution_v52/daily_price_only_execution_diagnostics.parquet",
             ROOT / "data_raw/cb_database_20261014_v49/basic.parquet", status_path, Path(__file__),
             ROOT / "data_processed/cb_execution_v52/unresolved_frozen_inventory.parquet"]
    daily, fills, final, missing = [pd.read_parquet(p) for p in paths[:4]]
    rights = pd.read_csv(paths[4])
    events = pd.read_parquet(paths[5])
    old = pd.read_parquet(paths[6])
    master = pd.read_parquet(paths[7])
    keys = ["date", "asset_code"]
    compared = daily.merge(old[keys + ["provisional_nav", "missing_held_count"]], on=keys,
                          validate="one_to_one", suffixes=("_new", "_old"))
    if len(compared) != len(daily) or len(old) != len(daily):
        raise ValueError("Baseline dates or basket coordinates changed")
    np.testing.assert_allclose(compared.baseline_provisional_nav, compared.provisional_nav_old,
                               atol=1e-10, rtol=1e-12)
    np.testing.assert_array_equal(compared.baseline_missing_held_count, compared.missing_held_count_old)
    summed_fills = fills.pivot_table(index="asset_code", columns="ts_code", values="quantity_change", aggfunc="sum")
    net = summed_fills.reindex(index=final.index, columns=final.columns).fillna(0.)
    for row in rights.itertuples(index=False):
        net.at[row.asset_code, row.ts_code] -= row.registered_quantity
    np.testing.assert_allclose(final.to_numpy(), net.to_numpy(), atol=1e-7, rtol=1e-10)
    paid_rights = rights.loc[rights.settled]
    pending_rights = rights.loc[~rights.settled]
    np.testing.assert_allclose(daily.payment_cash_yuan.sum(), paid_rights.gross_claim_yuan.sum(), atol=1e-7)
    np.testing.assert_allclose(daily.loc[daily.date.eq(daily.date.max()), "receivable_yuan"].sum(),
                               pending_rights.gross_claim_yuan.sum(), atol=1e-7)
    if not (daily.certified_total_return_nav.isna().all() and daily.settlement_cash_yuan.ge(-1e-7).all()):
        raise ValueError("Uncertified NAV or borrowing gate failed")

    missing["provisional_inventory_value_yuan"] = missing.quantity * missing.provisional_mark
    priority = missing.groupby("ts_code").agg(
        missing_basket_security_days=("date", "size"), distinct_missing_dates=("date", "nunique"),
        affected_baskets=("asset_code", "nunique"), first_missing=("date", "min"),
        last_missing=("date", "max"),
        stale_inventory_value_days_yuan=("provisional_inventory_value_yuan", "sum"))
    priority = priority.join(master.set_index("ts_code")[["bond_short_name", "delist_date"]], how="left")
    latest = daily.date.max()
    end = missing.loc[missing.date.eq(latest)].groupby("ts_code").agg(
        ending_missing_baskets=("asset_code", "nunique"),
        ending_stale_inventory_value_yuan=("provisional_inventory_value_yuan", "sum"))
    priority = priority.join(end).fillna({"ending_missing_baskets": 0, "ending_stale_inventory_value_yuan": 0})
    # Rank by observed accounting impact; current metadata only helps locate documents.
    priority["metadata_used_in_historical_signal"] = False
    priority["resolution"] = "requires_quote_or_primary_terminal_evidence_not_automatic_par_redemption"
    priority = priority.sort_values("missing_basket_security_days", ascending=False)
    priority.to_csv(report / "actual_inventory_repair_priority.csv")

    old_missing = pd.read_parquet(paths[-1])
    changes = old_missing[keys + ["ts_code"]].merge(missing[keys + ["ts_code"]],
        on=keys + ["ts_code"], how="outer", indicator=True, validate="one_to_one")
    changes["change"] = changes["_merge"].map({"left_only": "old_gap_removed",
        "right_only": "new_path_gap", "both": "gap_retained"})
    changes.drop(columns="_merge").to_parquet(output / "missing_inventory_path_changes.parquet", index=False)
    changes.groupby(["ts_code", "change"], observed=True).size().rename("position_days").reset_index().to_csv(
        report / "missing_inventory_change_by_security.csv", index=False)

    exposure = []
    for event in events.itertuples(index=False):
        traded = fills.loc[fills.ts_code.eq(event.ts_code)]
        owned = rights.loc[rights.event_id.eq(event.event_id)]
        exposure.append(dict(ts_code=event.ts_code, record_date=event.record_date,
            registered_quantity_all_baskets=owned.registered_quantity.sum(),
            gross_registered_claim_yuan=owned.gross_claim_yuan.sum(),
            first_trade=traded.date.min(), last_trade=traded.date.max(),
            last_trade_before_record=bool(traded.date.max() < event.record_date)))
    pd.DataFrame(exposure).to_csv(report / "reviewed_events_actual_inventory_exposure.csv", index=False)
    audit = dict(baseline_reproduces_original_v52=True, fill_minus_terminal_inventory_reconciled=True,
        event_claims_equal_once_only_payments=True, no_cash_borrowing=True,
        certified_total_return_nav_remains_unavailable=True,
        remaining_missing_codes=len(priority), remaining_ending_missing_codes=len(end),
        missing_basket_security_days_before=int(daily.baseline_missing_held_count.sum()),
        missing_basket_security_days_after=int(daily.missing_held_count.sum()),
        old_gap_days_removed=int(changes._merge.eq("left_only").sum()),
        new_path_gap_days=int(changes._merge.eq("right_only").sum()),
        gross_confirmed_redemption_yuan=float(daily.payment_cash_yuan.sum()),
        ending_unconfirmed_receivable_yuan=float(pending_rights.gross_claim_yuan.sum()),
        note="Twelve separate hypothetical 1m-yuan basket ledgers; cross-basket sums are diagnostics, not a combined product portfolio.",
        formal_training_permitted=False, source_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths})
    save(report / "accounting_audit.json", audit)
    print(json.dumps({k:v for k,v in audit.items() if k != "source_sha256"}, indent=2))
    print(priority.head(10).to_string())


if __name__ == "__main__":
    main()

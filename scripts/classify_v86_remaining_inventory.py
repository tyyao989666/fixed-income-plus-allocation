"""Investigation routes only; current master metadata never changes historical trades."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from scripts.run_v52_execution_preflight import sha, save

ORDINARY = {"127003.SZ", "127016.SZ", "110070.SH", "123049.SZ", "113589.SH"}
SPECIAL = {"123015.SZ", "128100.SZ", "128114.SZ", "110072.SH"}


def main():
    report = ROOT / "output/reports/cb_cashflows_v86"
    path = report / "actual_inventory_repair_priority.csv"
    master_path = ROOT / "data_raw/cb_database_20261014_v49/basic.parquet"
    data = pd.read_csv(path)
    current = data.loc[data.ending_missing_baskets.gt(0)].copy()
    if set(current.ts_code) != ORDINARY | SPECIAL:
        raise ValueError("Remaining positions changed; do not silently reuse classification")
    master = pd.read_parquet(master_path)
    current = current.merge(master[["ts_code", "maturity_date", "delist_date"]], on="ts_code",
                            validate="one_to_one", suffixes=("_audit", "_current_master"))
    current["investigation_route"] = current.ts_code.map(
        lambda c: "ordinary_maturity_primary_payment_evidence_pending" if c in ORDINARY
        else "special_claim_recovery_and_off_exchange_price_review")
    current["route_is_verified_settlement"] = False
    current["automatic_cash_credit_permitted"] = False
    current["metadata_used_for_historical_signals"] = False
    current["scope"] = "current_evidence_work_queue_not_historical_exclusion_list"
    current["next_evidence"] = current.ts_code.map(lambda c:
        "schedule_record_date_unit_amount_actual_payment_result_and_pre_maturity_quote"
        if c in ORDINARY else "rights_register_actual_recovery_cash_stock_trust_and_independent_valuation")
    current.loc[current.ts_code.eq("128114.SZ"), "next_evidence"] = (
        "creditor_identity_and_aggregated_admitted_claim_then_cash_stock_trust_allocations_and_actual_receipts")
    current.to_csv(report / "remaining_inventory_investigation_routes.csv", index=False)
    save(report / "remaining_inventory_investigation_status.json", dict(
        ordinary_maturity_candidates=len(ORDINARY), special_claim_review_candidates=len(SPECIAL),
        routes_are_investigation_not_verified_facts=True, original_returns_or_weights_changed=False,
        tianma_pending_receivable_is_separate_from_missing_security_count=True,
        zhengbang_cash_cap_must_not_be_applied_independently_to_each_asset_basket=True,
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [path, master_path, Path(__file__)]}))
    print(current[["ts_code", "bond_short_name", "investigation_route"]].to_string(index=False))


if __name__ == "__main__":
    main()

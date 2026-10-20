"""Bind the five unresolved issuers to evidence and actual simulated fills.

This is an ending-inventory investigation, never an input to historical signals.
It does not retire quantities, recognize receivables or credit any cash.
"""
from pathlib import Path
import hashlib
import json
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.v89_distressed_claims import creditor_tiers

OUT = ROOT / "output/reports/distressed_claims_v89"
CASES = {
    "128114.SZ": dict(name="正邦", checkpoint="2023-08-18", cash_cap=100000,
                      stock_cap=20000000, ratio="8.70", checkpoint_kind="last_conversion_not_claim_register",
                      blocker="accepted_claim_legal_creditor_aggregate_and_cash_stock_delivery_missing"),
    "127033.SZ": dict(name="中装转2", checkpoint="2025-09-18", cash_cap=50315,
                      checkpoint_kind="last_conversion_not_claim_register",
                      blocker="accepted_claim_legal_creditor_aggregate_cash_stock_trust_delivery_missing"),
    "110072.SH": dict(name="广汇", checkpoint="2026-08-17",
                      checkpoint_kind="proposed_maturity_not_claim_register",
                      blocker="proposal_approval_election_allotment_actual_payment_and_alias_bridge_missing"),
    "128100.SZ": dict(name="搜特", checkpoint="2024-08-07",
                      checkpoint_kind="bankruptcy_acceptance_not_claim_register",
                      blocker="accepted_claim_priority_recovery_distribution_and_alias_bridge_missing"),
    "123015.SZ": dict(name="蓝盾", checkpoint="2024-08-13",
                      checkpoint_kind="maturity_reference_requires_independent_verification",
                      blocker="old_new_code_bridge_accepted_claim_and_recovery_evidence_missing"),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    base = ROOT / "output/reports/cb_cashflows_v88/status.json"
    status = json.loads(base.read_text(encoding="utf-8"))
    bound = {str(base.relative_to(ROOT)): sha(base)}
    for relative, expected in status["source_sha256"].items():
        path = ROOT / relative
        if sha(path) != expected:
            raise ValueError(f"Parent evidence changed: {relative}")
        bound[relative] = expected
    def bind(relative):
        path = ROOT / relative
        bound[relative] = sha(path)
        return path
    ending = pd.read_csv(bind("output/reports/cb_cashflows_v88/ending_unresolved_inventory.csv"))
    fills = pd.read_parquet(bind("data_processed/cb_cashflows_v88/actual_fills.parquet"))
    manifest = json.loads(bind("data_raw/distressed_claims_v89/manifest.json").read_text(encoding="utf-8"))
    evidence = {r["ts_code"]: r for r in manifest}
    for r in manifest:
        for leaf in ["source.pdf", "source.txt", "source.html"]:
            relative = f'data_raw/distressed_claims_v89/{r["ts_code"]}/{leaf}'
            if (ROOT / relative).exists():
                bind(relative)
        if r.get("sha256") and sha(ROOT / r["pdf"]) != r["sha256"]:
            raise ValueError("Acquired PDF changed")
    if set(ending.ts_code) != set(CASES):
        raise ValueError("Ending inventory scope changed")
    rows, checks = [], []
    for item in ending.to_dict("records"):
        code, basket = item["ts_code"], item["asset_code"]
        case = CASES[code]
        selected = fills[(fills.ts_code == code) & (fills.asset_code == basket)]
        before = pd.to_datetime(selected.date) <= pd.Timestamp(case["checkpoint"])
        checkpoint_qty = float(selected.loc[before, "quantity_change"].sum())
        final_qty = float(selected.quantity_change.sum())
        if abs(final_qty - item["quantity"]) > 1e-7:
            raise ValueError(f"Fill/ending mismatch: {code}/{basket}")
        checks.append(dict(ts_code=code, asset_code=basket, quantities_reconcile=True,
                           fill_rows_after_checkpoint=int((~before).sum())))
        rows.append(dict(**item, name=case["name"], diagnostic_asof=item["date"],
                         checkpoint_date=case["checkpoint"], checkpoint_kind=case["checkpoint_kind"],
                         checkpoint_quantity_NOT_confirmed_entitlement=checkpoint_qty,
                         checkpoint_face_reference_yuan=checkpoint_qty * 100,
                         ending_stale_mark_reference_NOT_NAV_yuan=item["quantity"] * item["provisional_mark"],
                         evidence_status=evidence[code]["status"], source_url=evidence[code]["url"],
                         assumed_legal_creditor_id=None, accepted_claim_yuan=None,
                         cash_received_yuan=None, stock_received_units=None, trust_received_units=None,
                         recovery_value_yuan=None, blocker=case["blocker"], posting_allowed=False))
    pd.DataFrame(rows).to_csv(OUT / "ending_claim_investigation_NOT_NAV.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(checks).to_csv(OUT / "historical_quantity_checks.csv", index=False)
    # Synthetic legal scopes demonstrate why splitting a creditor is not neutral.
    scenarios = []
    for code in ["128114.SZ", "127033.SZ"]:
        case = CASES[code]
        for scope in ["one_legal_creditor", "two_independent_legal_creditors"]:
            claims = [dict(plan_id=f"synthetic_{code}", claim_id=f"synthetic_{i}", creditor_id="A" if scope == "one_legal_creditor" else str(i),
                           accepted_claim_yuan=80000, scope_verified=True, aggregate_complete=True) for i in range(2)]
            out = creditor_tiers(claims, case["cash_cap"], case.get("stock_cap"), case.get("ratio"))
            scenarios.append(dict(ts_code=code, synthetic_only=True, scope=scope,
                                  total_accepted_claim_yuan=160000,
                                  cash_tier_NOT_received_yuan=float(sum(x["contractual_cash_tier_yuan"] for x in out)),
                                  posting_allowed=False))
    pd.DataFrame(scenarios).to_csv(OUT / "synthetic_scope_sensitivity_NOT_recovery.csv", index=False)
    for relative in ["scripts/audit_v89_distressed_claims.py", "scripts/fetch_v89_distressed_evidence.py",
                     "src/v89_distressed_claims.py", "tests/test_distressed_claims_v89.py"]:
        bind(relative)
    result = dict(status="investigation_only_recovery_not_certified", unique_securities=5,
                  ending_positions=len(rows), original_pdfs_identity_checked=sum(r["status"] == "original_pdf_acquired_identity_checked" for r in manifest),
                  identity_blocked=[r["ts_code"] for r in manifest if r["status"] != "original_pdf_acquired_identity_checked"],
                  quantity_checks=len(checks), parent_source_hash_checks=len(status["source_sha256"]),
                  historical_missing_days_unchanged=7481, pending_existing_receivable_unchanged_yuan=status["ending_receivable_yuan"],
                  cash_postings=0, quantities_retired=0, formal_training_started=False,
                  recovery_certified=False, historical_signal_input=False, source_sha256=bound)
    (OUT / "status.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "source_sha256"}, ensure_ascii=False))


if __name__ == "__main__":
    main()

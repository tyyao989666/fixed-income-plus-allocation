"""Audit dated query routes and conditional cash references; no live book writes."""
from pathlib import Path
import hashlib
import json
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.v90_retired_security_routes import route_at_signal, conditional_cash_reference

OUT = ROOT / "output/reports/distressed_claims_v90"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    bound = {}
    def bind(relative):
        path = ROOT / relative
        bound[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path
    parent = json.loads(bind("output/reports/distressed_claims_v89/status.json").read_text(encoding="utf-8"))
    for relative, expected in parent["source_sha256"].items():
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Bound parent source changed: {relative}")
        bound[relative] = expected
    manifest = json.loads(bind("data_raw/distressed_claims_v90/manifest.json").read_text(encoding="utf-8"))
    evidence = {r["ts_code"]: r for r in manifest}
    for row in manifest:
        if row["status"] != "original_pdf_acquired_identity_checked":
            raise ValueError(f"Incomplete new evidence: {row['ts_code']}")
        pdf = bind(row["pdf"])
        if bound[row["pdf"]] != row["sha256"]:
            raise ValueError("New evidence hash changed")
        bind(str(pdf.with_suffix(".txt").relative_to(ROOT)))
        html = pdf.with_suffix(".html")
        if html.exists():
            bind(str(html.relative_to(ROOT)))
    calendar = pd.read_parquet(bind("data_processed/cb_evidence_v50/quotes/close_individual.parquet"), columns=[]).index
    if not isinstance(calendar, pd.DatetimeIndex) or calendar.has_duplicates or not calendar.is_monotonic_increasing:
        raise ValueError("Expected the complete dated replay calendar, not dates with fills only")
    def next_session(date):
        eligible = calendar[calendar > pd.Timestamp(date)]
        if not len(eligible):
            raise ValueError("No next session in verified replay calendar")
        return eligible[0].date().isoformat()
    routes = []
    for old, new, doc, start, days in [
        ("123015.SZ", "404001", "blue_transfer", "2023-09-18", [4]),
        ("128100.SZ", "404002", "soute_transfer", "2023-10-20", [0, 2, 4]),
        ("110072.SH", "404004", "guanghui_bridge", "2024-11-01", [0, 1, 2, 3, 4]),
    ]:
        proof = evidence[doc]
        routes.append(dict(old_code=old, new_code=new, transfer_start=start,
                           publication_date=proof["publication_date"],
                           known_at=next_session(proof["publication_date"]),
                           scheduled_weekdays=days, source_sha256=proof["sha256"],
                           source_url=proof["url"], purpose="query_only_not_execution"))
    (OUT / "reviewed_query_routes.json").write_text(json.dumps(routes, ensure_ascii=False, indent=2), encoding="utf-8")
    daily = []
    for day in calendar:
        for route in routes:
            out = route_at_signal(route["old_code"], day, routes)
            without_future = [r for r in routes if pd.Timestamp(r["known_at"]) <= day]
            assert out == route_at_signal(route["old_code"], day, without_future)
            daily.append(dict(signal_date=day, old_code=route["old_code"], **out))
    pd.DataFrame(daily).to_parquet(OUT / "daily_query_routes_NOT_execution.parquet", index=False)
    gaps = pd.read_parquet(bind("data_processed/cb_cashflows_v88/unresolved_inventory.parquet"))
    queue = []
    for route in routes:
        relevant = gaps[(gaps.ts_code == route["old_code"]) &
                        (pd.to_datetime(gaps.date) >= max(pd.Timestamp(route["known_at"]), pd.Timestamp(route["transfer_start"])))]
        queue.append(dict(old_code=route["old_code"], query_code=route["new_code"],
                          query_start=relevant.date.min(), query_end=relevant.date.max(),
                          missing_basket_security_days=len(relevant),
                          distinct_dates=relevant.date.nunique(), observed_otc_quotes_acquired=False,
                          execution_authorized=False))
    pd.DataFrame(queue).to_csv(OUT / "off_exchange_data_gap_queue.csv", index=False, encoding="utf-8-sig")
    quantities = pd.read_csv(bind("output/reports/distressed_claims_v89/ending_claim_investigation_NOT_NAV.csv"))
    scenarios, timeline = [], []
    for code, doc, unit, economic_day, unit_basis in [
        ("128114.SZ", "zhengbang_delivery", 100.14, "2024-01-31", "schedule_explicit_gross_cash_per_bond"),
        ("127033.SZ", "zhongzhuang_delivery", 100.63, "2026-02-12", "uniform_interest_scenario_not_holder_specific_verified_claim"),
    ]:
        proof = evidence[doc]
        known = next_session(proof["publication_date"])
        selected = quantities[quantities.ts_code == code]
        if len(selected) != 1:
            raise ValueError("Scope changed; cannot silently apply a per-basket creditor scenario")
        row = selected.iloc[0]
        qty = row.checkpoint_quantity_NOT_confirmed_entitlement
        cap = 100000 if code == "128114.SZ" else 50315
        if qty * unit > cap:
            raise ValueError("Scenario is no longer an all-cash tier")
        assumptions = "isolated_hypothetical_legal_creditor_no_other_claims_accepted_unrestricted_no_conversion_fractional_quantity_gross_tax_unmodeled"
        scenarios.append(dict(ts_code=code, asset_code=row.asset_code, quantity=qty,
                              conditional_unit_yuan=unit, unit_basis=unit_basis,
                              economic_payment_date=economic_day, conservative_confirmation_known_at=known,
                              assumptions=assumptions, issuer_class_delivery_reported=True,
                              investor_receipt_verified=False, source_url=proof["url"],
                              **conditional_cash_reference(qty, unit, calendar[-1], known, assumptions_explicit=True)))
        for day in calendar:
            state = conditional_cash_reference(qty, unit, day, known, assumptions_explicit=True)
            if day < pd.Timestamp(known):
                assert state["scenario_gross_cash_reference_yuan"] is None
            timeline.append(dict(date=day, ts_code=code, **state))
    pd.DataFrame(scenarios).to_csv(OUT / "conditional_actual_quantity_cash_NOT_receipts.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(timeline).to_parquet(OUT / "conditional_cash_timeline_NOT_NAV.parquet", index=False)
    for relative in ["scripts/fetch_v90_transfer_and_delivery.py", "scripts/audit_v90_transfer_and_delivery.py",
                     "src/v90_retired_security_routes.py", "tests/test_retired_routes_v90.py"]:
        bind(relative)
    status = dict(status="dated_routes_and_conditional_reference_completed_not_certified_returns",
                  parent_hash_checks=len(parent["source_sha256"]), primary_documents=len(manifest),
                  code_bridges=len(routes), daily_future_truncation_checks=len(daily),
                  conditional_positions=len(scenarios), conditional_cash_reference_NOT_receipts_yuan=sum(r["scenario_gross_cash_reference_yuan"] for r in scenarios),
                  acquired_otc_quote_rows=0, actual_cash_postings=0, retired_positions=0,
                  historical_missing_days_unchanged=parent["historical_missing_days_unchanged"],
                  training_started=False, formal_promotion=False, source_sha256=bound)
    (OUT / "status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in status.items() if k != "source_sha256"}, ensure_ascii=False))


if __name__ == "__main__":
    main()

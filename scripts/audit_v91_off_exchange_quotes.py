"""Audit raw samples, isolate unsafe marks, and date the issuer nonpayment flag."""
from pathlib import Path
import hashlib
import json
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.v91_off_exchange_quote_contract import review_receipt, default_state_at_signal

OUT = ROOT / "output/reports/off_exchange_quotes_v91"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    bound = {}
    def bind(relative):
        path = ROOT / relative
        bound[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path
    parent = json.loads(bind("output/reports/distressed_claims_v90/status.json").read_text(encoding="utf-8"))
    for relative, expected in parent["source_sha256"].items():
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Bound parent changed: {relative}")
        bound[relative] = expected
    routes = json.loads(bind("output/reports/distressed_claims_v90/reviewed_query_routes.json").read_text(encoding="utf-8"))
    calendar = pd.read_parquet(bind("data_processed/cb_evidence_v50/quotes/close_individual.parquet"), columns=[]).index
    probes = json.loads(bind("output/reports/off_exchange_quotes_v91/probe_status.json").read_text(encoding="utf-8"))
    reviewed, summaries = [], []
    for probe in probes:
        path = bind(probe["raw_file"])
        if bound[probe["raw_file"]] != probe["sha256"]:
            raise ValueError("Probe receipt changed")
        receipt = json.loads(path.read_text(encoding="utf-8"))
        route = next(r for r in routes if r["new_code"] == probe["code"])
        frame = review_receipt(receipt, route, calendar)
        frame["source_file"] = probe["raw_file"]
        frame["source_sha256"] = probe["sha256"]
        for field in ["observed_close", "observed_volume"]:
            frame[field] = pd.array(frame[field], dtype="Float64")
        reviewed.append(frame)
        summaries.append(dict(query_code=probe["code"], rows=len(frame),
                              missing_quantity=int(frame.observed_volume.isna().sum()),
                              non_session_rows=int(frame.reasons.str.contains("not_domestic_session").sum()),
                              unit_unverified_rows=int(frame.reasons.str.contains("quantity_unit_not_verified").sum()),
                              execution_accepted_rows=int(frame.execution_allowed.sum()),
                              nav_accepted_rows=int(frame.approved_for_nav.sum())))
    merged = pd.concat(reviewed, ignore_index=True)
    merged.to_csv(OUT / "quarantined_vendor_observations.csv", index=False, encoding="utf-8-sig")
    merged.to_parquet(OUT / "quarantined_vendor_observations.parquet", index=False)
    pd.DataFrame(summaries).to_csv(OUT / "sample_semantics_audit.csv", index=False, encoding="utf-8-sig")
    manifest = json.loads(bind("data_raw/distressed_default_v91/manifest.json").read_text(encoding="utf-8"))
    if len(manifest) != 1 or manifest[0]["status"] != "original_pdf_acquired_identity_checked":
        raise ValueError("Default notice not identity checked")
    proof = manifest[0]
    pdf = bind(proof["pdf"])
    if bound[proof["pdf"]] != proof["sha256"]:
        raise ValueError("Default proof changed")
    text = bind(str(pdf.with_suffix(".txt").relative_to(ROOT))).read_text(encoding="utf-8")
    if "本息无法按期兑付" not in "".join(text.split()):
        raise ValueError("Nonpayment term not found")
    visible = calendar[calendar > pd.Timestamp(proof["publication_date"])][0]
    notice = dict(old_code="110072.SH", new_code="404004", publication_date=proof["publication_date"],
                  known_at=str(visible.date()), source_url=proof["url"], source_sha256=proof["sha256"],
                  conclusion="issuer_reports_nonpayment_at_maturity_not_a_recovery_valuation")
    (OUT / "reviewed_default_notice.json").write_text(json.dumps(notice, ensure_ascii=False, indent=2), encoding="utf-8")
    states = []
    for day in calendar:
        state = default_state_at_signal("110072.SH", day, [notice])
        past_only = [notice] if pd.Timestamp(notice["known_at"]) <= day else []
        assert state == default_state_at_signal("110072.SH", day, past_only)
        states.append(dict(signal_date=day, old_code="110072.SH", **state))
    pd.DataFrame(states).to_parquet(OUT / "default_notice_shadow_NOT_execution.parquet", index=False)
    wind_probes = {}
    for name in ["wind_sdk_probe.json", "wind_legacy_runtime_probe.json"]:
        wind_probes[name] = json.loads(bind("data_raw/off_exchange_quotes_v91/" + name).read_text(encoding="utf-8"))
    for name in ["probe_v91_off_exchange_quotes.py", "probe_v91_wind_sdk.py", "probe_v91_wind_legacy_runtime.py", "fetch_v91_default_notice.py", "audit_v91_off_exchange_quotes.py"]:
        bind("scripts/" + name)
    bind("src/v91_off_exchange_quote_contract.py")
    bind("tests/test_otc_quotes_v91.py")
    status = dict(status="samples_quarantined_default_notice_dated_formal_returns_blocked",
                  provider_queries=len(probes), candidate_rows=len(merged),
                  missing_quantity_rows=int(merged.observed_volume.isna().sum()),
                  non_session_rows=int(merged.reasons.str.contains("not_domestic_session").sum()),
                  invalid_quantity_unit_rows=int(merged.reasons.str.contains("quantity_unit_not_verified").sum()),
                  accepted_trade_rows=int(merged.execution_allowed.sum()), accepted_nav_rows=int(merged.approved_for_nav.sum()),
                  daily_default_future_truncation_checks=len(states), default_known_at=notice["known_at"],
                  wind_legacy_import_fixed=bool(wind_probes["wind_legacy_runtime_probe.json"].get("sdk_imported")),
                  wind_connected=bool(wind_probes["wind_legacy_runtime_probe.json"].get("connected")),
                  historical_missing_days_unchanged=parent["historical_missing_days_unchanged"],
                  training_started=False, cash_postings=0, parent_hash_checks=len(parent["source_sha256"]),
                  source_sha256=bound)
    (OUT / "status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in status.items() if k != "source_sha256"}, ensure_ascii=False))


if __name__ == "__main__":
    main()

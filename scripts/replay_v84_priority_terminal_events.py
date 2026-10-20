"""Integrate Geli; retain Tianma aggregate contradiction as an unpaid scheduled claim."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from pypdf import PdfReader
from scripts.run_v52_execution_preflight import sha, save
from scripts.replay_v82_actual_cb_cashflows import main as replay
from scripts.replay_v83_combined_cb_cashflows import reviewed_gac
from scripts.audit_v83_inventory_repair import main as audit
from src.v84_receipt_check import reconcile_redemption_total


CASES = [dict(code="113507.SH", name="天马转债", schedule_id="1207055100", result_id="1207101007",
              record="2019-11-18", ex="2019-11-19", payment="2019-11-19", unit="100.353",
              bonds=29670, reported="2978962.25", decimals=3,
              schedule_tokens=["2019年11月18日", "2019年11月19日", "100.353"],
              result_tokens=["29670张", "2978962.25元", "100.353元/张", "2019年11月19日"]),
         dict(code="110030.SH", name="格力转债", schedule_id="1207176419", result_id="1207194448",
              record="2019-12-24", ex="2019-12-25", payment="2019-12-25", unit="106.00",
              bonds=3968820, reported="420694920.00", decimals=2,
              schedule_tokens=["2019年12月24日", "2019年12月25日", "106.00元"],
              result_tokens=["3968820张", "420694920.00元", "2019年12月25日兑付完毕"])]


def prepare():
    manifest_path = ROOT / "data_raw/inventory_terminal_v84/candidate_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    calendar_path = ROOT / "data_processed/cb_evidence_v50/last_known_balance_yuan.parquet"
    dates = pd.read_parquet(calendar_path).index
    events, sources = reviewed_gac()
    sources.extend([manifest_path, calendar_path, Path(__file__), ROOT / "src/v84_receipt_check.py",
                    ROOT / "scripts/audit_v83_inventory_repair.py"])
    checks = []
    for case in CASES:
        documents = {}
        for kind in ["schedule", "result"]:
            matches = [r for r in manifest if r["ts_code"] == case["code"]
                       and r["kind"] == kind and Path(r["pdf"]).stem == case[kind + "_id"]]
            if len(matches) != 1:
                raise ValueError("Missing/ambiguous reviewed primary evidence")
            row = matches[0]
            path = ROOT / row["pdf"]
            if sha(path) != row["sha256"]:
                raise ValueError("Primary PDF changed")
            compact = "".join("".join(p.extract_text() or "" for p in PdfReader(path).pages).split())
            compact = compact.replace(",", "").replace("，", "")
            if not all(t in compact for t in [case["code"][:6]] + case[kind + "_tokens"]):
                raise ValueError(f"Reviewed facts absent: {case['code']} {kind}")
            documents[kind] = row
            sources.append(path)
        reconciliation = reconcile_redemption_total(case["bonds"], case["unit"], case["reported"], case["decimals"])
        schedule_published = pd.Timestamp(documents["schedule"]["publication_date"])
        result_published = pd.Timestamp(documents["result"]["publication_date"])
        known_at = dates[dates > schedule_published][0]
        confirmed_at = dates[dates > result_published][0] if reconciliation["aggregate_reconciled"] else pd.NaT
        events.append(dict(event_id=f"reviewed_terminal_{case['code']}_{case['record']}",
            ts_code=case["code"], kind="terminal", known_at=known_at, record_date=case["record"],
            ex_date=case["ex"], pay_date=case["payment"], confirmed_at=confirmed_at,
            cash_yuan_per_bond=float(case["unit"]), result_publication_date=result_published,
            source=" | ".join(x["url"] for x in documents.values()),
            cash_basis="gross_before_investor_specific_tax",
            confirmation_basis="issuer_result_announcement_not_personal_account_statement" if pd.notna(confirmed_at)
                               else "blocked_issuer_aggregate_conflicts_with_quantity_times_unit_price"))
        checks.append(dict(ts_code=case["code"], name=case["name"], schedule_verified=True,
            result_document_verified=True, payment_confirmation_permitted=pd.notna(confirmed_at),
            schedule_publication_date=str(schedule_published.date()),
            result_publication_date=str(result_published.date()),
            confirmation_available_date=str(confirmed_at.date()) if pd.notna(confirmed_at) else None,
            scheduled_gross_yuan_per_bond=float(case["unit"]), issuer_bonds=case["bonds"],
            **reconciliation, documents=documents))
    review_path = ROOT / "output/reports/cb_cashflows_v84/primary_evidence_review.json"
    save(review_path, dict(events=checks, no_aggregate_override=True,
        tianma_note="Published price and aggregate contradict beyond rounding; no paid cash inferred; this is not a finding of issuer default."))
    sources.append(review_path)
    return events, sources


if __name__ == "__main__":
    events, sources = prepare()
    replay(events, sources, version="v84")
    audit(version="v84")

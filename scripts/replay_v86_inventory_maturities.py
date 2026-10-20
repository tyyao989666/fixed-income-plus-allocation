"""Replay three source-reconciled maturities on the actual basket inventory."""
from decimal import Decimal
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from pypdf import PdfReader
from scripts.run_v52_execution_preflight import sha, save
from scripts.replay_v82_actual_cb_cashflows import main as replay
from scripts.audit_v83_inventory_repair import main as audit

CASES = [dict(code="113008.SH", schedule_id="1209195023", result_id="1209233314",
    record="2021-02-01", ex="2021-02-02", payment="2021-02-02", unit="106.60",
    bonds=13421530, aggregate="1430735098.00",
    schedule_tokens=["2021年2月1日", "2021年2月2日", "106.60元"],
    result_tokens=["13421530张", "1430735098.00元", "已经于2021年2月2日兑付完毕"]),
    dict(code="110031.SH", schedule_id="1210184054", result_id="1210238828",
    record="2021-06-11", ex="2021-06-15", payment="2021-06-15", unit="107.00",
    bonds=23979520, aggregate="2565808640",
    schedule_tokens=["2021年6月11日", "2021年6月15日", "107.00元"],
    result_tokens=["23979520张", "2565808640元", "2021年6月15日兑付完毕"]),
    dict(code="110034.SH", schedule_id="1212147248", result_id="1212201645",
    record="2022-01-14", ex="2022-01-17", payment="2022-01-17", unit="108.00",
    bonds=14979190, aggregate="1617752520.00",
    schedule_tokens=["2022年1月14日", "2022年1月17日", "108.00元/张"],
    result_tokens=["14979190张", "1617752520.00元", "已经于2022年1月17日兑付完毕"])]


def exact_total(case):
    amount = Decimal(str(case["bonds"])) * Decimal(case["unit"])
    if amount != Decimal(case["aggregate"]):
        raise ValueError("Published contractual quantity, price and total conflict")
    return str(amount)


def prepare():
    parent_path = ROOT / "output/reports/cb_cashflows_v85/status.json"
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    if parent["status"] != "research_accounting_completed_formal_gate_blocked":
        raise ValueError("Parent accounting replay incomplete")
    sources = [parent_path, Path(__file__), ROOT / "scripts/audit_v83_inventory_repair.py"]
    for name, expected in parent["source_sha256"].items():
        path = ROOT / name
        if sha(path) != expected:
            raise ValueError(f"Parent source changed: {name}")
        sources.append(path)
    event_path = ROOT / "data_processed/cb_cashflows_v85/reviewed_terminal_events.parquet"
    sources.append(event_path)
    inherited = pd.read_parquet(event_path)
    seven_codes = set(json.loads((ROOT / "output/reports/terminal_batch_v70/reviewed_events.json").read_text(encoding="utf-8")))
    inherited = inherited.loc[~inherited.ts_code.isin(seven_codes)]
    if len(inherited) != 7:
        raise ValueError("Unexpected inherited event coverage")
    events = inherited.to_dict("records")
    manifest_path = ROOT / "data_raw/inventory_terminal_v86/candidate_manifest.json"
    sources.append(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    calendar_path = ROOT / "data_processed/cb_evidence_v50/last_known_balance_yuan.parquet"
    dates = pd.read_parquet(calendar_path).index
    sources.append(calendar_path)
    reviews = []
    for case in CASES:
        docs = {}
        for kind in ["schedule", "result"]:
            matches = [r for r in manifest if r["ts_code"] == case["code"] and r["kind"] == kind
                       and Path(r["pdf"]).stem == case[kind + "_id"]]
            if len(matches) != 1:
                raise ValueError("Missing/ambiguous primary evidence")
            row = matches[0]
            path = ROOT / row["pdf"]
            if sha(path) != row["sha256"]:
                raise ValueError("Primary PDF changed")
            text = "".join("".join(p.extract_text() or "" for p in PdfReader(path).pages).split())
            text = text.replace(",", "").replace("，", "")
            if not all(t in text for t in [case["code"][:6]] + case[kind + "_tokens"]):
                raise ValueError(f"Reviewed facts absent: {case['code']} {kind}")
            docs[kind] = row
            sources.append(path)
        total = exact_total(case)
        publication = pd.Timestamp(docs["result"]["publication_date"])
        known = dates[dates > pd.Timestamp(docs["schedule"]["publication_date"])][0]
        confirmed = dates[dates > publication][0]
        events.append(dict(event_id=f"reviewed_terminal_{case['code']}_{case['record']}", ts_code=case["code"],
            kind="terminal", known_at=known, record_date=case["record"], ex_date=case["ex"],
            pay_date=case["payment"], confirmed_at=confirmed, cash_yuan_per_bond=float(case["unit"]),
            result_publication_date=publication, source=" | ".join(r["url"] for r in docs.values()),
            cash_basis="gross_before_investor_specific_tax",
            confirmation_basis="issuer_result_exact_quantity_times_contract_price_and_paid_statement"))
        reviews.append(dict(ts_code=case["code"], exact_aggregate_reconciliation=True, total_yuan=total,
            unit_price_yuan=case["unit"], issuer_bond_quantity=case["bonds"],
            record_date=case["record"], economic_payment_date=case["payment"],
            schedule_known_at=str(known.date()), confirmed_at=str(confirmed.date()), documents=docs))
    review_path = ROOT / "output/reports/cb_cashflows_v86/primary_evidence_review.json"
    save(review_path, dict(events=reviews, inherited_tianma_conflict_not_overridden=True,
                           no_synthetic_suspension_quotes=True, investor_specific_tax_not_modeled=True))
    sources.append(review_path)
    return events, sources


if __name__ == "__main__":
    events, sources = prepare()
    replay(events, sources, version="v86")
    audit(version="v86")

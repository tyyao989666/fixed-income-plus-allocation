"""Add four reviewed ordinary terminations; do not erase suspended-period gaps."""
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
from src.v84_receipt_check import reconcile_redemption_total

CASES = [dict(code="128012.SZ", schedule_id="1209624346", result_id="1209837200",
    record="2021-04-19", ex="2021-04-20", payment="2021-04-27", unit="103", bonds=None, aggregate=None,
    schedule_tokens=["2021年4月19日", "2021年4月20日", "103元/张", "2021年4月27日"],
    result_tokens=["公司已全额赎回", "2021年4月19日", "2021年4月27日",
                   "赎回款已通过可转债托管券商直接划入"]),
    dict(code="123029.SZ", schedule_id="1224444238", result_id="1224502694",
    record="2025-08-15", ex="2025-08-18", payment="2025-08-18", unit="128", bonds=1788, aggregate="228864",
    schedule_tokens=["2025年8月15日", "2025年8月18日", "128元人民币/张", "2021年6月29日停止交易"],
    result_tokens=["1788张", "228864元", "18日兑付完毕"]),
    dict(code="128013.SZ", schedule_id="1214121947", result_id="1214173673",
    record="2022-07-28", ex="2022-07-29", payment="2022-07-29", unit="108", bonds=652363, aggregate="70455204.00",
    schedule_tokens=["2022年7月28日", "2022年7月29日", "108元/张"],
    result_tokens=["652363.00张", "70455204.00元", "已于2022年7月29日兑付完毕"]),
    dict(code="128076.SZ", schedule_id="1224692514", result_id="1224714912",
    record="2025-10-14", ex="2025-10-15", payment="2025-10-15", unit="110", bonds=26941, aggregate="2963510.00",
    schedule_tokens=["2025年10月14日", "2025年10月15日", "110元人民币/张", "2025年10月9日"],
    result_tokens=["26941张", "2963510.00元", "15日兑付完毕"])]


def prepare():
    parent_path = ROOT / "output/reports/cb_cashflows_v84/status.json"
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    if parent["status"] != "research_accounting_completed_formal_gate_blocked":
        raise ValueError("V84 parent replay incomplete")
    sources = [parent_path, Path(__file__), ROOT / "src/v84_receipt_check.py",
               ROOT / "scripts/audit_v83_inventory_repair.py"]
    for name, digest in parent["source_sha256"].items():
        path = ROOT / name
        if sha(path) != digest:
            raise ValueError(f"V84 source changed: {name}")
        sources.append(path)
    events_path = ROOT / "data_processed/cb_cashflows_v84/reviewed_terminal_events.parquet"
    sources.append(events_path)
    inherited = pd.read_parquet(events_path)
    # The generic runner independently supplies the seven V70 events.
    inherited = inherited.loc[inherited.ts_code.isin(["113009.SH", "113507.SH", "110030.SH"])]
    if len(inherited) != 3:
        raise ValueError("Inherited GAC/Geli/Tianma events missing")
    events = inherited.to_dict("records")
    manifest_path = ROOT / "data_raw/inventory_terminal_v85/candidate_manifest.json"
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
                raise ValueError("Missing exact reviewed document")
            row = matches[0]
            path = ROOT / row["pdf"]
            if sha(path) != row["sha256"]:
                raise ValueError("Primary PDF hash mismatch")
            text = "".join("".join(p.extract_text() or "" for p in PdfReader(path).pages).split())
            text = text.replace(",", "").replace("，", "")
            if not all(t in text for t in [case["code"][:6]] + case[kind + "_tokens"]):
                raise ValueError(f"Reviewed source facts missing: {case['code']} {kind}")
            docs[kind] = row
            sources.append(path)
        if case["bonds"] is not None:
            reconciled = reconcile_redemption_total(case["bonds"], case["unit"], case["aggregate"], 0)
            if not reconciled["aggregate_reconciled"]:
                raise ValueError("Aggregate payment contradiction")
            basis = "issuer_result_quantity_price_total_and_paid_statement"
        else:
            reconciled = dict(aggregate_reconciled=None,
                              note="No final aggregate stated; full-class payment explicitly confirmed in issuer notice; no invented issuer quantity or total.")
            basis = "issuer_explicit_full_class_payment_and_schedule_unit_price_no_aggregate_available"
        published = pd.Timestamp(docs["result"]["publication_date"])
        schedule_published = pd.Timestamp(docs["schedule"]["publication_date"])
        events.append(dict(event_id=f"reviewed_terminal_{case['code']}_{case['record']}",
            ts_code=case["code"], kind="terminal", known_at=dates[dates > schedule_published][0],
            record_date=case["record"], ex_date=case["ex"], pay_date=case["payment"],
            confirmed_at=dates[dates > published][0], cash_yuan_per_bond=float(case["unit"]),
            source=" | ".join(r["url"] for r in docs.values()), result_publication_date=published,
            cash_basis="gross_before_investor_specific_tax", confirmation_basis=basis))
        reviews.append(dict(ts_code=case["code"], schedule_verified=True, payment_confirmation_permitted=True,
            confirmation_basis=basis, unit_price=float(case["unit"]), record_date=case["record"],
            economic_payment_date=case["payment"], result_publication_date=str(published.date()),
            confirmed_at=str(dates[dates > published][0].date()), documents=docs, **reconciled))
    review_path = ROOT / "output/reports/cb_cashflows_v85/primary_evidence_review.json"
    save(review_path, dict(events=reviews, inherited_tianma_confirmation_remains_blocked=True,
                           no_synthetic_suspension_prices=True))
    sources.append(review_path)
    return events, sources


if __name__ == "__main__":
    events, sources = prepare()
    replay(events, sources, version="v85")
    audit(version="v85")

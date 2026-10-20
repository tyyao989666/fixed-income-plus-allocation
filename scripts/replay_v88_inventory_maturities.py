"""Three additional issuer-reconciled maturities; no changes to basket trading rules."""
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

CASES = [dict(code="113545.SH", schedule_id="1224682020", result_id="1224711536",
    record="2025-10-13", ex="2025-10-14", payment="2025-10-14", unit="110",
    bonds=9524400, aggregate="1047684000.00", last_trade="2025-09-30",
    schedule_tokens=["2025年10月13日", "2025年10月14日", "110元", "2025年9月30日"],
    result_tokens=["9524400张", "1047684000.00元", "根据中登上海分公司提供的数据"], note=""),
    dict(code="127015.SZ", schedule_id="1224893977", result_id="1224918013",
    record="2025-12-31", ex="2026-01-05", payment="2026-01-05", unit="106",
    bonds=9472781, aggregate="1004084781.20", last_trade="2025-12-26",
    schedule_tokens=["2025年12月31日", "2026年1月5日", "106元", "2025年12月26日"],
    result_tokens=["9472781张", "1004084781.20元", "已于2026年1月5日兑付完毕",
                   "债券简称“希望转债”债券代码“127015”"],
    note="Header names 127049 Hope2; body, 2020 issue and settlement section identify 127015 Hope1. Do not retire 127049."),
    dict(code="128134.SZ", schedule_id="1225581072", result_id="1225598597",
    record="2026-10-08", ex="2026-10-09", payment="2026-10-09", unit="110",
    bonds=3841785, aggregate="422596350", last_trade="2026-09-28",
    schedule_tokens=["2026年10月8日", "2026年10月9日", "110元", "2026年9月28日"],
    result_tokens=["3841785张", "422596350元", "2026年10月9日兑付完毕"],
    note="Selected detailed schedule becomes conservatively visible after the last trading day; do not infer a feasible prior exit.")]


def exact_total(case):
    expected = Decimal(case["bonds"]) * Decimal(case["unit"])
    difference = Decimal(case["aggregate"]) - expected
    return dict(exact_total_check=difference == 0, expected_total_yuan=str(expected),
                reported_total_yuan=case["aggregate"], discrepancy_yuan=str(difference))


def prepare():
    parent_path = ROOT / "output/reports/cb_cashflows_v87/status.json"
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    if parent["status"] != "research_accounting_completed_formal_gate_blocked":
        raise ValueError("Parent replay incomplete")
    sources = [parent_path, Path(__file__), ROOT / "scripts/audit_v83_inventory_repair.py"]
    for name, expected in parent["source_sha256"].items():
        path = ROOT / name
        if sha(path) != expected:
            raise ValueError(f"Parent source changed: {name}")
        sources.append(path)
    inherited_path = ROOT / "data_processed/cb_cashflows_v87/reviewed_terminal_events.parquet"
    inherited = pd.read_parquet(inherited_path)
    seven_path = ROOT / "output/reports/terminal_batch_v70/reviewed_events.json"
    seven = set(json.loads(seven_path.read_text(encoding="utf-8")))
    inherited = inherited.loc[~inherited.ts_code.isin(seven)]
    if len(inherited) != 15:
        raise ValueError("Unexpected inherited events")
    events = inherited.to_dict("records")
    manifest_path = ROOT / "data_raw/inventory_terminal_v88/candidate_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    calendar_path = ROOT / "data_processed/cb_evidence_v50/last_known_balance_yuan.parquet"
    dates = pd.read_parquet(calendar_path).index
    sources.extend([inherited_path, manifest_path, calendar_path, ROOT / "scripts/fetch_v88_inventory_maturities.py"])
    sources.extend((ROOT / "data_raw/inventory_terminal_v88").glob("*/index.json"))
    reviews, schedules = [], []
    for case in CASES:
        docs = {}
        for kind in ["schedule", "result"]:
            match = [r for r in manifest if r["ts_code"] == case["code"] and r["kind"] == kind
                     and Path(r["pdf"]).stem == case[kind + "_id"]]
            if len(match) != 1:
                raise ValueError("Missing exact reviewed document")
            row = match[0]
            path = ROOT / row["pdf"]
            if sha(path) != row["sha256"]:
                raise ValueError("Primary PDF changed")
            text = "".join("".join(p.extract_text() or "" for p in PdfReader(path).pages).split())
            text = text.replace(",", "").replace("，", "")
            if not all(t in text for t in [case["code"][:6]] + case[kind + "_tokens"]):
                raise ValueError(f"Reviewed facts absent: {case['code']} {kind}")
            docs[kind] = row
            sources.append(path)
        reconciliation = exact_total(case)
        if not reconciliation["exact_total_check"] and case["code"] != "127015.SZ":
            raise ValueError("New unresolved issuer aggregate conflict")
        publication = pd.Timestamp(docs["result"]["publication_date"])
        known = dates[dates > pd.Timestamp(docs["schedule"]["publication_date"])][0]
        confirmed = dates[dates > publication][0] if reconciliation["exact_total_check"] else pd.NaT
        events.append(dict(event_id=f"reviewed_terminal_{case['code']}_{case['record']}",
            ts_code=case["code"], kind="terminal", known_at=known, record_date=case["record"],
            ex_date=case["ex"], pay_date=case["payment"], confirmed_at=confirmed,
            cash_yuan_per_bond=float(case["unit"]), source=" | ".join(r["url"] for r in docs.values()),
            result_publication_date=publication, cash_basis="gross_before_investor_specific_tax",
            confirmation_basis="issuer_result_exact_quantity_unit_total_and_settlement_result_disclosure"
                if pd.notna(confirmed) else "blocked_issuer_quantity_unit_total_conflict"))
        reviews.append(dict(ts_code=case["code"], **reconciliation, unit_yuan=case["unit"],
            issuer_bonds=case["bonds"], total_yuan=case["aggregate"], record_date=case["record"],
            ex_date=case["ex"], known_at=str(known.date()),
            confirmed_at=str(confirmed.date()) if pd.notna(confirmed) else None,
            note=case["note"], documents=docs))
        schedules.append(dict(ts_code=case["code"], known_at=known, last_trade_date=case["last_trade"],
            source=docs["schedule"]["url"], schedule_sha256=docs["schedule"]["sha256"]))
    report = ROOT / "output/reports/cb_cashflows_v88"
    review_path = report / "primary_evidence_review.json"
    save(review_path, dict(events=reviews, no_hindsight_exit_or_conversion=True,
                           tianma_confirmation_not_overridden=True, hope_confirmation_blocked_by_amount_conflict=True))
    schedule_path = report / "schedule_only_execution_controls.parquet"
    pd.DataFrame(schedules).to_parquet(schedule_path, index=False)
    sources.extend([review_path, schedule_path])
    return events, sources


if __name__ == "__main__":
    events, sources = prepare()
    replay(events, sources, version="v88")
    audit(version="v88")

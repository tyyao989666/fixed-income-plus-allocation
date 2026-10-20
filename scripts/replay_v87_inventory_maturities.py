"""Five exact-identity maturity events, using unchanged basket targets and prices."""
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

CASES = [dict(code="127003.SZ", schedule_id="1213573848", result_id="1213645424",
    record="2022-06-07", ex="2022-06-08", payment="2022-06-08", unit="110",
    bonds=892946, aggregate="98224060", last_trade="2022-05-23",
    schedule_tokens=["2022年6月7日", "2022年6月8日", "110元/张"],
    result_tokens=["892946张", "98224060元", "2022年6月8日兑付完毕"],
    note="Notice labels 96438168 as principal, equal to 108 percent of par; do not use it as par face amount."),
    dict(code="127016.SZ", schedule_id="1225071643", result_id="1225088833",
    record="2026-04-08", ex="2026-04-09", payment="2026-04-09", unit="111",
    bonds=13980019, aggregate="1551782109.00", last_trade="2026-04-02",
    schedule_tokens=["2026年4月8日", "2026年4月9日", "111元人民币/张"],
    result_tokens=["13980019张", "1551782109.00元", "已于2026年4月9日兑付完毕"], note=""),
    dict(code="110070.SH", schedule_id="1225070868", result_id="1225097880",
    record="2026-04-10", ex="2026-04-13", payment="2026-04-13", unit="112",
    bonds=1726840, aggregate="193406080", last_trade="2026-04-07",
    schedule_tokens=["2026年4月10日", "2026年4月13日", "112元人民币/张"],
    result_tokens=["1726840张", "193406080元", "根据中登上海分公司提供的数据"], note=""),
    dict(code="123049.SZ", schedule_id="1225079428", result_id="1225097981",
    record="2026-04-10", ex="2026-04-13", payment="2026-04-13", unit="118",
    bonds=2143704, aggregate="252957072", last_trade="2026-04-07",
    schedule_tokens=["2026年4月10日", "2026年4月13日", "118元人民币/张"],
    result_tokens=["2143704张", "252957072元", "13日兑付完毕"], note=""),
    dict(code="113589.SH", schedule_id="1225367825", result_id="1225386428",
    record="2026-06-23", ex="2026-06-24", payment="2026-06-24", unit="110",
    bonds=12880, aggregate="1416800", last_trade="2026-06-17",
    schedule_tokens=["2026年6月23日", "2026年6月24日", "110.00元"],
    result_tokens=["12880张", "1416800元", "根据中登上海分公司提供的数据"], note="")]


def exact_total(case):
    if Decimal(case["bonds"]) * Decimal(case["unit"]) != Decimal(case["aggregate"]):
        raise ValueError("Contract quantity times unit price differs from issuer aggregate")


def prepare():
    parent_path = ROOT / "output/reports/cb_cashflows_v86/status.json"
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    if parent["status"] != "research_accounting_completed_formal_gate_blocked":
        raise ValueError("Parent replay not complete")
    sources = [parent_path, Path(__file__), ROOT / "scripts/audit_v83_inventory_repair.py"]
    for name, expected in parent["source_sha256"].items():
        path = ROOT / name
        if sha(path) != expected:
            raise ValueError(f"Parent immutable source changed: {name}")
        sources.append(path)
    inherited_path = ROOT / "data_processed/cb_cashflows_v86/reviewed_terminal_events.parquet"
    sources.append(inherited_path)
    inherited = pd.read_parquet(inherited_path)
    seven_path = ROOT / "output/reports/terminal_batch_v70/reviewed_events.json"
    seven = set(json.loads(seven_path.read_text(encoding="utf-8")))
    inherited = inherited.loc[~inherited.ts_code.isin(seven)]
    if len(inherited) != 10:
        raise ValueError("Inherited event coverage changed")
    events = inherited.to_dict("records")
    raw = ROOT / "data_raw/inventory_terminal_v87"
    manifest = []
    for path in [raw / "candidate_manifest.json", raw / "issuer_fallback/candidate_manifest.json"]:
        manifest.extend(json.loads(path.read_text(encoding="utf-8")))
        sources.append(path)
    sources.extend(raw.glob("*/index.json"))
    sources.extend((raw / "issuer_fallback").glob("*/index*.json"))
    calendar_path = ROOT / "data_processed/cb_evidence_v50/last_known_balance_yuan.parquet"
    sources.append(calendar_path)
    dates = pd.read_parquet(calendar_path).index
    reviews = []
    for case in CASES:
        docs = {}
        for kind in ["schedule", "result"]:
            matches = [r for r in manifest if r["ts_code"] == case["code"] and r["kind"] == kind
                       and Path(r["pdf"]).stem == case[kind + "_id"]]
            if len(matches) != 1:
                raise ValueError(f"Ambiguous or absent reviewed source: {case['code']} {kind}")
            row = matches[0]
            path = ROOT / row["pdf"]
            if sha(path) != row["sha256"]:
                raise ValueError("Primary PDF changed")
            text = "".join("".join(p.extract_text() or "" for p in PdfReader(path).pages).split())
            text = text.replace(",", "").replace("，", "")
            if not all(t in text for t in [case["code"][:6]] + case[kind + "_tokens"]):
                raise ValueError(f"Reviewed text facts absent: {case['code']} {kind}")
            sources.append(path)
            docs[kind] = row
        exact_total(case)
        published = pd.Timestamp(docs["result"]["publication_date"])
        known = dates[dates > pd.Timestamp(docs["schedule"]["publication_date"])][0]
        confirmed = dates[dates > published][0]
        events.append(dict(event_id=f"reviewed_terminal_{case['code']}_{case['record']}",
            ts_code=case["code"], kind="terminal", known_at=known, record_date=case["record"],
            ex_date=case["ex"], pay_date=case["payment"], confirmed_at=confirmed,
            cash_yuan_per_bond=float(case["unit"]), result_publication_date=published,
            source=" | ".join(r["url"] for r in docs.values()), cash_basis="gross_before_investor_specific_tax",
            confirmation_basis="issuer_result_exact_quantity_unit_total_and_settlement_result_disclosure"))
        reviews.append(dict(ts_code=case["code"], exact_total_check=True, unit_yuan=case["unit"],
            issuer_bonds=case["bonds"], issuer_total_yuan=case["aggregate"], record_date=case["record"],
            ex_date=case["ex"], economic_pay_date=case["payment"], last_trade_date=case["last_trade"],
            last_trade_date_is_diagnostic_only_not_new_policy=True,
            known_at=str(known.date()), confirmed_at=str(confirmed.date()), note=case["note"], documents=docs))
    review_path = ROOT / "output/reports/cb_cashflows_v87/primary_evidence_review.json"
    save(review_path, dict(events=reviews, tianma_confirmation_still_blocked=True,
        no_automatic_conversion_or_hindsight_exit=True, source_search_failures_preserved=True,
        issuer_search_fallback_used_for=["127016.SZ", "110070.SH", "123049.SZ"]))
    sources.extend([review_path, ROOT / "scripts/fetch_v87_inventory_maturities.py",
                    ROOT / "scripts/fetch_v87_issuer_fallback.py"])
    return events, sources


if __name__ == "__main__":
    events, sources = prepare()
    replay(events, sources, version="v87")
    audit(version="v87")

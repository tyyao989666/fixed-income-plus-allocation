"""Merge the existing primary-evidence GAC maturity with V70 redemptions."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from pypdf import PdfReader
from scripts.run_v52_execution_preflight import sha
from scripts.replay_v82_actual_cb_cashflows import main


def reviewed_gac():
    status_path = ROOT / "output/reports/cb_execution_v54/status.json"
    prior = json.loads(status_path.read_text(encoding="utf-8"))
    sources = [status_path, Path(__file__)]
    text = {}
    for kind, evidence in prior["evidence"].items():
        path = ROOT / "data_raw/cb_terminal_evidence_v54" / f"{kind}.pdf"
        if sha(path) != evidence["sha256"]:
            raise ValueError("V54 primary PDF changed")
        sources.append(path)
        text[kind] = "".join("".join(p.extract_text() or "" for p in PdfReader(path).pages).split())
        if "113009" not in text[kind]:
            raise ValueError("GAC exact security code missing")
    if not all(x in text["gac_schedule"] for x in ["2022年1月21日", "106.00"]):
        raise ValueError("GAC schedule fact mismatch")
    if not all(x in text["gac_result"] for x in ["1,136,020,020", "1,071,717,000", "兑付完毕"]):
        raise ValueError("GAC payment confirmation fact mismatch")
    if abs(1071717000 / 100 * 106 - 1136020020) > .01:
        raise ValueError("GAC per-bond gross payment reconciliation failed")
    calendar_path = ROOT / "data_processed/cb_evidence_v50/last_known_balance_yuan.parquet"
    sources.append(calendar_path)
    dates = pd.read_parquet(calendar_path).index
    published = pd.Timestamp("2022-01-24")
    confirmed = dates[dates > published][0]
    event = dict(event_id="reviewed_terminal_113009.SH_2022-01-21", ts_code="113009.SH", kind="terminal",
        known_at="2022-01-19", record_date="2022-01-21", ex_date="2022-01-24",
        pay_date="2022-01-24", confirmed_at=confirmed, cash_yuan_per_bond=106.,
        source=" | ".join(v["url"] for v in prior["evidence"].values()),
        result_publication_date=published, cash_basis="gross_before_investor_specific_tax",
        confirmation_basis="issuer_result_announcement_not_personal_account_statement")
    return [event], sources


if __name__ == "__main__":
    events, sources = reviewed_gac()
    main(events, sources, version="v83")

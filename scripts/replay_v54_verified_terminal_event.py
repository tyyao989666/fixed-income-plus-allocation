"""Source-backed terminal-event replay; never changes the active V53 training inputs."""
from pathlib import Path
from io import BytesIO
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import requests
from pypdf import PdfReader
from scripts.run_v52_execution_preflight import save, sha
from src.v49_point_in_time import cb_bucket_weights
from src.v52_cb_execution import QuantityBooks, validate_events


def main():
    raw = ROOT / "data_raw/cb_terminal_evidence_v54"
    out = ROOT / "data_processed/cb_execution_v54"
    report = ROOT / "output/reports/cb_execution_v54"
    for p in [raw, out, report]:
        p.mkdir(parents=True, exist_ok=True)
    sources = {
        "gac_schedule": "https://static.cninfo.com.cn/finalpage/2022-01-18/1212201646.PDF",
        "gac_result": "https://www.hkexnews.hk/listedco/listconews/sehk/2022/0124/2022012400800_c.pdf",
    }
    for name, url in sources.items():
        path = raw / (name + ".pdf")
        if not path.exists():
            response = requests.get(url, timeout=40)
            response.raise_for_status()
            if not response.content.startswith(b"%PDF"):
                raise ValueError("Announcement transport is not a PDF")
            path.write_bytes(response.content)
        text = "\n".join(page.extract_text() or "" for page in PdfReader(BytesIO(path.read_bytes())).pages)
        (raw / (name + ".txt")).write_text(text, encoding="utf-8")
        compact = "".join(text.split())
        if "113009" not in compact:
            raise ValueError("Exact security identifier absent from source")
        if name == "gac_result" and not all(s in compact for s in ["1,136,020,020", "1,071,717,000", "兑付完毕"]):
            raise ValueError("Settlement result evidence differs from reviewed facts")
        if name == "gac_schedule" and not all(s in compact for s in ["2022年1月21日", "106.00"]):
            raise ValueError("Record date/redemption amount differs from reviewed facts")
    source = ROOT / "data_processed/cb_evidence_v50"
    balance = pd.read_parquet(source / "last_known_balance_yuan.parquet")
    close = pd.read_parquet(source / "quotes/close_individual.parquet").reindex_like(balance)
    volume = pd.read_parquet(source / "quotes/volume_individual.parquet").reindex_like(balance)
    size = pd.read_parquet(source / "last_known_issue_size_yuan.parquet").reindex_like(balance)
    eligible = pd.read_parquet(ROOT / "data_processed/cb_masked_research_v51/signal_buy_eligibility.parquet")
    names = json.loads((ROOT / "output/reports/cb_masked_research_v51/status.json").read_text(encoding="utf-8"))["cb_assets"]
    dates, codes = close.index, list(close.columns)
    # Do not feed a result published on payment day into an earlier signal.
    # Cash recognition is conservatively delayed until the next local session.
    confirm = dates[dates > pd.Timestamp("2022-01-24")][0]
    event = dict(event_id="113009_terminal_20220124_verified", ts_code="113009.SH", kind="terminal",
        known_at="2022-01-19", record_date="2022-01-21", ex_date="2022-01-24",
        pay_date=confirm, confirmed_at=confirm, cash_yuan_per_bond=106.,
        source=sources["gac_schedule"] + " | " + sources["gac_result"],
        economic_payment_date="2022-01-24", result_publication_date="2022-01-24",
        schedule_publication_date="2022-01-18", amount_basis="gross_before_investor_specific_tax",
        ledger_convention="next_session_cash_recognition_after_payment_result_publication_NOT_actual_same_day_cash_NAV")
    raw_events = pd.DataFrame([event])
    raw_events.to_parquet(out / "reviewed_terminal_event_with_evidence.parquet", index=False)
    events = validate_events(raw_events, codes)
    executable = close.gt(0) & np.isfinite(close) & volume.gt(0) & np.isfinite(volume)
    valuation = close.where(executable)
    marks = pd.read_parquet(source / "nontransaction_marks_for_inventory_review_only.parquet")
    for row in marks.loc[marks.anchor_verified.eq(True) & marks.positive_volume.eq(False)].drop_duplicates(["date","ts_code"]).itertuples():
        if row.date in dates and row.ts_code in codes and pd.isna(valuation.at[row.date,row.ts_code]):
            valuation.at[row.date,row.ts_code] = row.close
    book = QuantityBooks(names, codes)
    rows, gaps = [], []
    for i in range(1, len(dates)):
        signal, day = dates[i-1], dates[i]
        allowed = eligible.loc[signal]
        groups = cb_bucket_weights(close.loc[signal].where(allowed), balance.loc[signal], size.loc[signal])
        target = np.zeros_like(book.quantity)
        for a, name in enumerate(names):
            if name in groups:
                target[a] = groups[name]["weights"].reindex(codes, fill_value=0).to_numpy()
        values = book.step(day, target, allowed.to_numpy(), close.loc[day].to_numpy(),
            executable.loc[day].to_numpy(), valuation.loc[day].to_numpy(), events)
        for a,name in enumerate(names):
            rows.append(dict(date=day, signal_date=signal, asset_code=name,
                provisional_nav=values["provisional_nav"][a]/1e6,
                observed_price_nav=values["observed_price_nav"][a]/1e6,
                certified_total_return_nav=np.nan, missing_held_count=int(values["missing_held_count"][a]),
                transaction_cost_yuan=values["cost_yuan"][a], payment_cash_yuan=values["payment_cash"][a],
                receivable_yuan=values["receivable"][a], settlement_cash_yuan=values["cash"][a]))
            for c in np.flatnonzero(values["stale_valuation_positions"][a]):
                gaps.append((day,name,codes[c],float(book.quantity[a,c])))
        if i % 400 == 0:
            print(f"V54 verified-event inventory {i}/{len(dates)-1}",flush=True)
    daily = pd.DataFrame(rows)
    daily.to_parquet(out / "verified_event_research_replay.parquet",index=False)
    missing = pd.DataFrame(gaps,columns=["date","asset_code","ts_code","quantity"])
    missing.to_parquet(out / "remaining_missing_valuation_inventory.parquet",index=False)
    original = pd.read_parquet(ROOT / "data_processed/cb_execution_v52/unresolved_frozen_inventory.parquet")
    result = dict(status="verified_one_terminal_event_replayed_not_formally_promoted", formal_promotion=False,
        reviewed_terminal_events=1, preexisting_missing_book_security_days=len(original),
        current_missing_book_security_days=len(missing),
        missing_GAC_inventory_after_terminal=int((missing.ts_code.eq("113009.SH") & missing.date.ge(pd.Timestamp("2022-01-24"))).sum()),
        payment_cash_yuan_total=float(daily.payment_cash_yuan.sum()),
        current_source_date_cutoff="2026-10-14", training_dataset_changed=False,
        limitations=["Only one verified terminal event; complete coupons and other terminations remain unverified",
            "Gross before investor-specific tax; not a net-of-tax product NAV",
            "Payment result recognized next domestic session, not exact intraday bank receipt",
            "Full rerun changes later target sizing; difference in missing counts is not a performance improvement"],
        evidence={name:dict(url=url,sha256=sha(raw / (name+'.pdf'))) for name,url in sources.items()})
    if result["missing_GAC_inventory_after_terminal"] != 0:
        raise ValueError("Terminal position failed to cancel")
    if not np.isclose(daily.loc[daily.date.eq(confirm),"payment_cash_yuan"].sum(), daily.payment_cash_yuan.sum()):
        raise ValueError("Unexpected double or mistimed terminal cashflow")
    save(report / "status.json",result)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__ == "__main__":
    main()

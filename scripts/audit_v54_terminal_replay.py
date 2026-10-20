"""Check the actual isolated terminal replay against evidence and recognition dates."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from scripts.run_v52_execution_preflight import save, sha
from src.v52_cb_execution import validate_events


def main():
    out = ROOT / "data_processed/cb_execution_v54"
    report = ROOT / "output/reports/cb_execution_v54"
    state = json.loads((report / "status.json").read_text(encoding="utf-8"))
    daily = pd.read_parquet(out / "verified_event_research_replay.parquet")
    missing = pd.read_parquet(out / "remaining_missing_valuation_inventory.parquet")
    raw = pd.read_parquet(out / "reviewed_terminal_event_with_evidence.parquet")
    events = validate_events(raw,["113009.SH"])
    checks = []

    def check(name, condition):
        checks.append(dict(check=name, passed=bool(condition)))

    check("all_twelve_books_on_entire_calendar",len(daily)==2127*12 and daily.asset_code.nunique()==12)
    check("one_record_per_book_date",not daily.duplicated(["date","asset_code"]).any())
    check("signal_precedes_execution",daily.signal_date.lt(daily.date).all())
    check("no_post_cancellation_GAC_missing_inventory",not (missing.ts_code.eq("113009.SH") & missing.date.ge(pd.Timestamp("2022-01-24"))).any())
    check("no_early_payment_recognition",daily.loc[daily.date.lt(pd.Timestamp(events.pay_date.iloc[0])),"payment_cash_yuan"].eq(0).all())
    check("one_payment_booking_date",daily.loc[daily.payment_cash_yuan.gt(0),"date"].nunique()==1)
    check("payment_on_registered_conservative_date",daily.loc[daily.payment_cash_yuan.gt(0),"date"].eq(events.pay_date.iloc[0]).all())
    check("gross_per_bond_matches_result_aggregate",np.isclose(1136020020 / 10717170,raw.cash_yuan_per_bond.iloc[0]))
    check("no_negative_cash_or_cost",daily.settlement_cash_yuan.ge(-1e-7).all() and daily.transaction_cost_yuan.ge(0).all())
    check("not_promoted_to_total_return",daily.certified_total_return_nav.isna().all() and not state["formal_promotion"])
    check("future_outcome_not_moved_back_to_schedule",pd.Timestamp(events.confirmed_at.iloc[0]) > pd.Timestamp(raw.result_publication_date.iloc[0]))
    for key,item in state["evidence"].items():
        check("original_PDF_hash:"+key,sha(ROOT / "data_raw/cb_terminal_evidence_v54" / (key+".pdf"))==item["sha256"])
    registry=json.loads((ROOT / "output/reports/unified_v53/status.json").read_text(encoding="utf-8"))
    check("active_forecast_dataset_not_modified",sha(ROOT / "data_processed/unified_v53/dataset.npz")==registry["dataset_sha256"])
    pd.DataFrame(checks).to_csv(report / "artifact_checks.csv",index=False)
    result=dict(passed=all(x["passed"] for x in checks),checks=len(checks),formal_promotion=False,
        current_missing_book_security_days=len(missing),verified_terminal_events=1)
    save(report / "audit.json",result)
    print(json.dumps(result,indent=2))
    if not result["passed"]:
        raise SystemExit("Terminal replay audit failed")


if __name__ == "__main__":
    main()

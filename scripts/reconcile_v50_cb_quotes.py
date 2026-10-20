"""Accept only identified positive-volume observations with neighboring anchors."""
from pathlib import Path
import json
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from scripts.v50_cb_vendor_evidence import ifind_tables
from src.v50_cb_evidence import quote_table,requested_codes
from src.v45_forecast import write_json


def main():
    base=ROOT/"data_processed/cb_rebuild_20261014_v49/quote_repair"
    out=ROOT/"data_processed/cb_evidence_v50/quotes"
    reports=ROOT/"output/reports/cb_evidence_v50"
    out.mkdir(parents=True,exist_ok=True);reports.mkdir(parents=True,exist_ok=True)
    fields=["open","high","low","close","preclose","volume","amount"]
    frames={k:pd.read_parquet(base/f"{k}_individual.parquet") for k in fields}
    original_close=frames["close"].copy()
    accepted=[];review=[];evidence=[]
    for path in sorted((ROOT/"data_raw/cb_evidence_v50/ifind").glob("remaining_*.json")):
        record=json.loads(path.read_text(encoding="utf-8"))
        if record.get("exit_code")!=0:continue
        query=record["request"]["params"]["query"]
        codes=requested_codes(query)
        if not codes:raise ValueError("Request lacks explicit security codes")
        day=pd.Timestamp(path.stem.split("_")[1])
        for table,attrs in ifind_tables(record["response"]):
            f=quote_table(table,codes,day-pd.Timedelta(days=5),day+pd.Timedelta(days=5))
            if f.empty:continue
            if f.duplicated(["date","ts_code"]).any():raise ValueError("Duplicated quote table keys")
            for code,g in f.groupby("ts_code"):
                if code not in original_close:continue
                paired=g.set_index("date").close.to_frame("new").join(original_close[code].rename("old"),how="inner").dropna()
                anchor_error=float((paired.new/paired.old-1).abs().max()) if len(paired) else np.inf
                verified=len(paired)>=2 and anchor_error<=5e-5
                for row in g.itertuples():
                    if row.date not in original_close.index:continue
                    item=dict(source=path.name,date=row.date,ts_code=code,close=row.close,preclose=row.preclose,
                        positive_volume=bool(row.positive_volume),anchor_rows=len(paired),anchor_max_error=anchor_error,
                        anchor_verified=verified)
                    evidence.append(item)
                    missing=pd.isna(frames["close"].at[row.date,code]) or pd.isna(frames["preclose"].at[row.date,code])
                    if not missing:continue
                    if not verified or not row.positive_volume or not np.isfinite(row.preclose) or row.preclose<=0:
                        review.append(dict(**item,decision="not_filled_unverified_or_no_transaction_mark"))
                        continue
                    conflict=False
                    for field,value in [("close",row.close),("preclose",row.preclose)]:
                        old=frames[field].at[row.date,code]
                        if pd.notna(old) and not np.isclose(old,value,rtol=5e-5,atol=1e-5):conflict=True
                    if conflict:
                        review.append(dict(**item,decision="not_filled_existing_field_conflict"));continue
                    for field,value in [("close",row.close),("preclose",row.preclose)]:
                        if pd.isna(frames[field].at[row.date,code]):
                            frames[field].at[row.date,code]=value
                            accepted.append(dict(date=row.date,ts_code=code,field=field,value=value,source=path.name,
                                role="positive_volume_historical_quote_not_current_status"))
    if not evidence:raise ValueError("No dated quote evidence; parser/source contract requires review")
    for field,frame in frames.items():frame.to_parquet(out/f"{field}_individual.parquet")
    pd.DataFrame(evidence).to_parquet(out/"ifind_quote_evidence.parquet",index=False)
    pd.DataFrame(accepted).to_csv(reports/"ifind_verified_quote_repairs.csv",index=False)
    pd.DataFrame(review).to_csv(reports/"ifind_quote_quarantine.csv",index=False)
    result=dict(status="completed_observed_quote_reconciliation",accepted_field_cells=len(accepted),
        accepted_security_days=len(pd.DataFrame(accepted)[["date","ts_code"]].drop_duplicates()) if accepted else 0,
        source_evidence_rows=len(evidence),quarantined_missing_observations=len(review),
        zero_volume_marks_filled=False,current_status_used_as_historical=False,training_ready=False)
    write_json(reports/"quote_reconciliation_status.json",result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":main()

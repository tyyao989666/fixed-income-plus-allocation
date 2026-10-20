"""Full historical CB announcement retrieval with split-on-cap and checkpoints."""
from pathlib import Path
import json
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import pandas as pd
import numpy as np
import tushare as ts
from src.v45_forecast import write_json

DEST=ROOT/"data_raw/cb_evidence_v50/tushare/disclosures"
REPORT=ROOT/"output/reports/cb_evidence_v50"


def fetch(pro,codes,start,end,tag):
    path=DEST/f"{tag}_{start}_{end}.parquet"
    request=dict(ts_code=",".join(codes),start_date=start,end_date=end)
    if path.exists():
        f=pd.read_parquet(path)
        cached=json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        if cached["request"]!=request:raise ValueError("Disclosure cache request mismatch")
    else:
        f=pro.cb_share(**request)
        if f is None:raise ValueError("No disclosure response")
        f.to_parquet(path,index=False)
        write_json(path.with_suffix(".json"),dict(request=request,rows=len(f),fetched_at=pd.Timestamp.now(tz="UTC").isoformat()))
        time.sleep(.25)
    if len(f)>=2000:
        if len(codes)>1:
            m=len(codes)//2
            return pd.concat([fetch(pro,codes[:m],start,end,tag+"a"),fetch(pro,codes[m:],start,end,tag+"b")],ignore_index=True)
        a,b=pd.Timestamp(start),pd.Timestamp(end)
        if a>=b:raise ValueError("Single-day disclosure cap requires pagination")
        m=a+(b-a)//2
        return pd.concat([fetch(pro,codes,start,m.strftime("%Y%m%d"),tag+"l"),
            fetch(pro,codes,(m+pd.Timedelta(days=1)).strftime("%Y%m%d"),end,tag+"r")],ignore_index=True)
    if len(f) and not f.ts_code.isin(codes).all():raise ValueError("Wrong code in disclosure response")
    return f


def main():
    DEST.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    progress=REPORT/"disclosure_progress.json"
    write_json(progress,dict(status="running",completed_codes=0))
    try:
        base=ROOT/"data_processed/cb_rebuild_20261014_v49"
        master=pd.read_parquet(base/"exit_aware_repaired/historical_security_master_audited.parquet")
        close=pd.read_parquet(base/"quote_repair/close_individual.parquet")
        codes=sorted(set(master.index)&set(close.columns[close.notna().any()]))
        pro=ts.pro_api(ts.get_token(),timeout=40)
        frames=[]
        for start in range(0,len(codes),12):
            group=codes[start:start+12]
            f=fetch(pro,group,"20170101","20261014",f"group_{start:04d}")
            frames.append(f)
            print(f"Disclosures {min(start+12,len(codes))}/{len(codes)} codes; rows collected={sum(len(x) for x in frames)}",flush=True)
            write_json(progress,dict(status="running",completed_codes=min(start+12,len(codes)),total_codes=len(codes),rows=sum(len(x) for x in frames)))
        raw=pd.concat(frames,ignore_index=True)
        raw.to_parquet(DEST/"all_historical_disclosures_raw.parquet",index=False)
        f=raw.copy()
        for col in ["publish_date","end_date"]:
            f[col]=pd.to_datetime(f[col].astype(str),format="%Y%m%d",errors="coerce")
        valid=f.publish_date.notna()&f.end_date.notna()&f.end_date.le(f.publish_date)&f.publish_date.le(pd.Timestamp("2026-10-14"))
        for col in ["remain_size","issue_size"]:
            f[col]=pd.to_numeric(f[col],errors="coerce")
            valid &= np.isfinite(f[col])&f[col].ge(0)
        valid &= f.issue_size.gt(0)&f.remain_size.le(f.issue_size*(1+1e-8))
        f.loc[~valid].to_parquet(DEST/"invalid_disclosures_quarantined.parquet",index=False)
        clean=f.loc[valid].drop_duplicates(["ts_code","publish_date","end_date","remain_size","issue_size"])
        conflicting=clean.duplicated(["ts_code","publish_date","end_date"],keep=False)
        clean.loc[conflicting].to_parquet(DEST/"conflicting_disclosures_quarantined.parquet",index=False)
        clean=clean.loc[~conflicting]
        clean.to_parquet(DEST/"all_historical_disclosures_valid.parquet",index=False)
        clean.groupby(clean.publish_date.dt.year).agg(rows=("ts_code","size"),codes=("ts_code","nunique")).to_csv(REPORT/"historical_disclosure_coverage.csv")
        result=dict(status="completed_raw_disclosures_not_promoted",requested_codes=len(codes),raw_rows=len(raw),valid_rows=len(clean),
            codes=clean.ts_code.nunique(),invalid_rows=int((~valid).sum()),conflicting_rows=int(conflicting.sum()),
            first_publication=str(clean.publish_date.min()),last_publication=str(clean.publish_date.max()),training_ready=False)
        write_json(REPORT/"disclosure_status.json",result);write_json(progress,result)
        print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    except Exception as exc:
        write_json(progress,dict(status="failed",error=f"{type(exc).__name__}: {exc}"))
        raise


if __name__=="__main__":main()

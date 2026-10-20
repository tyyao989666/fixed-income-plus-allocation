"""Fill only vendor-observed missing CB cells; never interpolate or alter source DB."""
from pathlib import Path
import json
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import tushare as ts
from src.v45_forecast import write_json
from src.v49_cb_database import historical_universe


def main():
    base=ROOT/"data_processed/cb_rebuild_20261014_v49"
    report=ROOT/"output/reports/cb_rebuild_20261014_v49/quote_repair"
    raw=ROOT/"data_raw/cb_database_20261014_v49/targeted_market_queries"
    out=base/"quote_repair"
    for p in [report,raw,out]:p.mkdir(parents=True,exist_ok=True)
    progress=report/"progress.json"
    write_json(progress,dict(status="running",completed=0))
    try:
        missing=pd.read_csv(ROOT/"output/reports/cb_rebuild_20261014_v49/exit_aware/unresolved_held_positions.csv",parse_dates=["date"])
        master=pd.read_parquet(base/"historical_security_master_audited.parquet")
        eligible=historical_universe(master.rename_axis("ts_code").reset_index(),list(master.index))
        missing=missing.loc[missing.ts_code.isin(eligible.index[eligible.included_research_universe])]
        fields={"open":"open","high":"high","low":"low","close":"close","preclose":"pre_close","volume":"vol","amount":"amount"}
        frames={k:pd.read_parquet(base/f"{k}_individual.parquet") for k in fields}
        pro=None
        tasks=list(missing.groupby("ts_code"))
        if len(tasks)>60:raise ValueError("Targeted request budget exceeded")
        records=[];conflicts=[];files=[]
        for i,(code,g) in enumerate(tasks,1):
            start=g.date.min()-pd.Timedelta(days=3)
            end=min(g.date.max()+pd.Timedelta(days=1),pd.Timestamp("2026-10-14"))
            # Repair the whole interior quote gap, not just a held endpoint.
            # Otherwise a newly restored quote can reveal another missing
            # held mark one day later. Future observations choose query extent
            # only; they never select prior holdings or training labels.
            later=frames["close"].loc[frames["close"].index>g.date.max(),code].first_valid_index()
            if later is not None:
                end=max(end,later)
            request=dict(ts_code=code,start_date=start.strftime("%Y%m%d"),end_date=end.strftime("%Y%m%d"))
            key=f"{code}_{request['start_date']}_{request['end_date']}"
            cached=raw/f"{key}.parquet"
            if cached.exists():
                f=pd.read_parquet(cached)
            else:
                if pro is None:
                    pro=ts.pro_api(ts.get_token(),timeout=30)
                f=pro.cb_daily(**request)
                if f is None:raise ValueError("Vendor returned no response")
                if len(f)>=2000:raise ValueError("Possible vendor pagination truncation")
                f.to_parquet(cached,index=False)
                write_json(raw/f"{key}.json",dict(endpoint="cb_daily",parameters=request,rows=len(f),fetched_at=pd.Timestamp.now(tz="UTC").isoformat()))
                time.sleep(.3)
            files.append(str(cached.relative_to(ROOT)))
            if not f.empty:
                if not f.ts_code.eq(code).all() or f.trade_date.duplicated().any():raise ValueError("Ambiguous quote identity")
                f=f.copy();f["date"]=pd.to_datetime(f.trade_date,format="%Y%m%d",errors="raise")
                if not f.date.between(start,end).all():raise ValueError("Quote outside registered window")
                for _,row in f.iterrows():
                    day=row.date
                    if day not in frames["close"].index:continue
                    for local,vendor in fields.items():
                        value=pd.to_numeric(row.get(vendor),errors="coerce")
                        if not np.isfinite(value) or value<0 or (local not in {"volume","amount"} and value==0):continue
                        old=frames[local].at[day,code]
                        if pd.isna(old):
                            frames[local].at[day,code]=float(value)
                            records.append(dict(date=day,ts_code=code,field=local,value=float(value),source=cached.name))
                        elif not np.isclose(old,value,rtol=1e-7,atol=1e-6):
                            conflicts.append(dict(date=day,ts_code=code,field=local,database_value=float(old),vendor_value=float(value),action="preserved_database_observation"))
            write_json(progress,dict(status="running",completed=i,total=len(tasks),filled_cells=len(records)))
            print(f"Quote verification {i}/{len(tasks)} {code}, filled cells={len(records)}",flush=True)
        for field,f in frames.items():f.to_parquet(out/f"{field}_individual.parquet")
        pd.DataFrame(records).to_csv(report/"observed_cell_repairs.csv",index=False)
        pd.DataFrame(conflicts).to_csv(report/"conflicting_observations_not_overwritten.csv",index=False)
        import hashlib
        manifest=[dict(path=p,sha256=hashlib.sha256((ROOT/p).read_bytes()).hexdigest()) for p in files]
        write_json(report/"status.json",dict(status="completed_observed_quotes_only",securities=len(tasks),
            filled_cells=len(records),conflicting_observations=len(conflicts),training_ready=False,
            interpolated_cells=0,external_database_modified=False,files=manifest))
        write_json(progress,dict(status="completed",completed=len(tasks),total=len(tasks),filled_cells=len(records)))
    except Exception as exc:
        write_json(progress,dict(status="failed",error=f"{type(exc).__name__}: {exc}"))
        raise


if __name__=="__main__":main()

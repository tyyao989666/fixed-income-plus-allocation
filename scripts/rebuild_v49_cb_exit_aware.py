"""Exit/universe/quote repair comparison, isolated from frozen production data.

The stored initial control predates private-name screening and optional quote
repairs; differences are not a single-variable estimate of the exit rule.
"""
from pathlib import Path
import argparse
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from src.v49_cb_database import known_exit_exclusions, rebalance_turnover, strict_nav, historical_universe, observed_returns
from src.v49_point_in_time import cb_bucket_weights, realized_bucket_return, align_disclosed_balances
from scripts.rebuild_v49_cb_from_latest_database import snapshot, NAMES
from src.v45_forecast import write_json


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--use-quote-repairs",action="store_true")
    args=ap.parse_args()
    label="exit_aware_repaired" if args.use_quote_repairs else "exit_aware"
    base=ROOT/"data_processed/cb_rebuild_20261014_v49"
    raw=ROOT/"data_raw/cb_database_20261014_v49/exit_contract"
    report=ROOT/"output/reports/cb_rebuild_20261014_v49"/label
    out=base/label
    report.mkdir(parents=True,exist_ok=True)
    out.mkdir(parents=True,exist_ok=True)
    progress=report/"progress.json"
    write_json(progress,dict(status="running",stage="exit_contract",completed_days=0))
    try:
        files=[]
        calls=[]
        paths=[ROOT/"data_raw/cb_database_20261014_v49/1_Base_Data/_meta/tushare_cb_call_all.parquet",
               ROOT/"data_raw/refresh_20261014_wind_verified/supplement_v49/tushare/cb_calls.parquet"]
        # The older full-history cache was not part of the price import.
        if not paths[0].exists():
            paths[0]=raw/"calls_0.parquet" if (raw/"calls_0.parquet").exists() else Path.home()/"Desktop/db_download/修改/Data/可转债数据/1_Base_Data/_meta/tushare_cb_call_all.parquet"
        if not paths[1].exists() and (raw/"calls_1.parquet").exists():
            paths[1]=raw/"calls_1.parquet"
        for i,p in enumerate(paths):
            calls.append(pd.read_parquet(snapshot(p,raw/f"calls_{i}.parquet",files)))
        calls=pd.concat(calls,ignore_index=True)
        master=pd.read_parquet(base/"historical_security_master_audited.parquet")
        master=historical_universe(master.rename_axis("ts_code").reset_index(),list(master.index))
        master=master.loc[master.included_research_universe]
        quote_source=base/"quote_repair" if args.use_quote_repairs else base
        close=pd.read_parquet(quote_source/"close_individual.parquet").reindex(columns=master.index)
        preclose=pd.read_parquet(quote_source/"preclose_individual.parquet").reindex(columns=master.index)
        returns=observed_returns(close,preclose)
        dates=close.index
        excluded,events=known_exit_exclusions(calls,master,dates,maturity_buffer_days=7)
        for code in close:
            close.loc[dates<master.at[code,"list_date_parsed"],code]=np.nan
        signal_close=close.mask(excluded)
        events.to_parquet(out/"known_call_state_events.parquet",index=False)
        excluded.to_parquet(out/"known_exit_exclusions.parquet")
        balance=pd.read_parquet(base/"balance_individual.parquet").reindex(columns=master.index)*1e8
        issue=pd.DataFrame(np.broadcast_to(master.issue_size.to_numpy(),balance.shape),index=dates,columns=master.index)
        disclosed=pd.read_parquet(base/"dated_balance_announcements.parquet")
        known=align_disclosed_balances(disclosed,dates,list(master.index),max_age_days=120)
        variants={"historical_balance_diagnostic":(balance,issue),
                  "announced_balance_strict_candidate":(known["balance"],known["issue_size"])}
        rows=[];members=[];missing=[];books={}
        for i in range(1,len(dates)):
            day,signal=dates[i],dates[i-1]
            for variant,(b,s) in variants.items():
                groups=cb_bucket_weights(signal_close.loc[signal],b.loc[signal],s.loc[signal])
                for name in NAMES:
                    detail=groups.get(name)
                    w=detail["weights"] if detail else None
                    if w is None:
                        rows.append(dict(variant=variant,date=day,signal_date=signal,asset_code=name,
                            member_count=0,return_1d=np.nan,max_weight=np.nan,weight_sum=np.nan,
                            fallback_level=-1,one_way_turnover=np.nan,cost_5bp=np.nan))
                        books.pop((variant,name),None)
                        continue
                    turnover=rebalance_turnover(books.get((variant,name)),returns.loc[signal],w)
                    value=realized_bucket_return(w,returns.loc[day])
                    rows.append(dict(variant=variant,date=day,signal_date=signal,asset_code=name,
                        member_count=len(w),return_1d=value,max_weight=float(w.max()),weight_sum=float(w.sum()),
                        fallback_level=detail["fallback_level"],one_way_turnover=turnover,cost_5bp=turnover*.0005))
                    for code,weight in w.items():
                        members.append((variant,day,signal,name,code,float(weight)))
                        r=returns.at[day,code]
                        if not np.isfinite(r) or r<=-1:
                            missing.append(dict(variant=variant,date=day,asset_code=name,ts_code=code,
                                precommitted_weight=float(weight),resolution="unresolved_no_renormalization"))
                    books[(variant,name)]=w
            if i%200==0 or i==len(dates)-1:
                print(f"Exit-aware {i}/{len(dates)-1}: {day.date()}, unresolved positions={len(missing)}",flush=True)
                write_json(progress,dict(status="running",completed_days=i,total_days=len(dates)-1))
        result=pd.DataFrame(rows)
        result["net_return_5bp_diagnostic"]=result.return_1d-result.cost_5bp
        result["nav_no_gap_fill"]=result.groupby(["variant","asset_code"]).return_1d.transform(strict_nav)
        result.to_parquet(out/"cb_12_buckets_rebuilt.parquet",index=False)
        holdings=pd.DataFrame(members,columns=["variant","date","signal_date","asset_code","ts_code","weight"])
        holdings.to_parquet(out/"precommitted_holdings.parquet",index=False)
        pd.DataFrame(missing).to_csv(report/"unresolved_held_positions.csv",index=False)
        original=pd.read_parquet(base/"cb_12_buckets_rebuilt.parquet")
        comparison=result.merge(original,on=["variant","date","asset_code"],suffixes=("_exit","_control"),validate="one_to_one")
        comparison.to_parquet(out/"controlled_exit_comparison.parquet",index=False)
        stats=[]
        for variant,g in comparison.groupby("variant"):
            stats.append(dict(variant=variant,bucket_days=len(g),
                missing_before=int(g.return_1d_control.isna().sum()),missing_after=int(g.return_1d_exit.isna().sum()),
                mean_one_way_turnover_before=float(g.one_way_turnover_control.mean()),
                mean_one_way_turnover_after=float(g.one_way_turnover_exit.mean())))
        pd.DataFrame(stats).to_csv(report/"controlled_exit_comparison.csv",index=False)
        # This validates only the precommitted exclusion, not next-open fills.
        idx=excluded.index.get_indexer(holdings.signal_date)
        col=excluded.columns.get_indexer(holdings.ts_code)
        if excluded.to_numpy()[idx,col].any():
            raise ValueError("Known excluded security re-entered holdings")
        status=dict(status="completed_research_candidate_not_promoted",training_ready=False,
            quote_repairs_used=args.use_quote_repairs,public_candidate_codes=len(master),
            rows=len(result),unresolved_positions=len(missing),comparison=stats,
            exclusion_violations=0,nonzero_holdings=len(holdings),
            call_events=len(events),maturity_buffer_calendar_days=7,
            selection_basis="prior available implemented-call announcement and fixed contractual maturity, not future delisting or return",
            announcement_availability="strictly next trading session; then next-day research return",
            future_return_membership_filter=False,
            warning="Research exit convention chosen during development; not untouched OOS or certified executable fills.",
            blockers=["historical_issue_method_and_balance_publication_coverage",
                "remaining_missing_marks_and_total_return_cashflows",
                "known_halt_tradeability_and_actual_execution_prices"],
            original_results_overwritten=False)
        write_json(raw/"manifest.json",dict(files=files))
        write_json(report/"status.json",status)
        write_json(progress,dict(status="completed_candidates_blocked",completed_days=len(dates)-1,total_days=len(dates)-1))
        print(status,flush=True)
    except Exception as exc:
        write_json(progress,dict(status="failed",error=f"{type(exc).__name__}: {exc}"))
        raise


if __name__=="__main__":
    main()

"""Rebuild all 12 CB price-return candidates from full dated disclosures."""
from pathlib import Path
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from src.v45_forecast import write_json
from src.v49_cb_database import known_exit_exclusions,observed_returns,rebalance_turnover
from src.v49_point_in_time import cb_disclosure_events,align_disclosed_balances,cb_bucket_weights,realized_bucket_return
from src.v50_cb_evidence import issuance_registry,fill_preconversion_known_balance
from scripts.rebuild_v49_cb_from_latest_database import NAMES


def main():
    old=ROOT/"data_processed/cb_rebuild_20261014_v49"
    out=ROOT/"data_processed/cb_evidence_v50"
    report=ROOT/"output/reports/cb_evidence_v50"
    report.mkdir(exist_ok=True,parents=True);out.mkdir(exist_ok=True,parents=True)
    progress=report/"rebuild_progress.json"
    write_json(progress,dict(status="running",stage="source_alignment"))
    try:
        master=pd.read_parquet(old/"exit_aware_repaired/historical_security_master_audited.parquet")
        quotes=out/"quotes"
        close=pd.read_parquet(quotes/"close_individual.parquet").reindex(columns=master.index)
        dates=close.index
        issues=pd.read_parquet(ROOT/"data_raw/cb_evidence_v50/tushare/cb_issues_full.parquet")
        registry=issuance_registry(issues,master,dates)
        registry.to_parquet(out/"historical_issuance_registry.parquet")
        public=registry.index[registry.public_method_verified]
        observed_codes=close.columns[close.notna().any()]
        unknown=master.loc[observed_codes.difference(public),["bond_short_name","list_date","issue_size"]]
        unknown.to_csv(report/"issuance_method_unverified_active_codes.csv")
        master=master.loc[master.index.intersection(public)]
        close=close.reindex(columns=master.index)
        for code in close:
            close.loc[dates<master.at[code,"list_date_parsed"],code]=np.nan
        pre=pd.read_parquet(quotes/"preclose_individual.parquet").reindex(columns=master.index)
        returns=observed_returns(close,pre)
        returns.to_parquet(out/"individual_observed_price_returns.parquet")
        disclosures=pd.read_parquet(ROOT/"data_raw/cb_evidence_v50/tushare/disclosures/all_historical_disclosures_valid.parquet")
        disclosures=disclosures.loc[disclosures.ts_code.isin(master.index)]
        # Validate scale against immutable issue size instead of assuming all
        # endpoints/eras have the same units (cb_issue is mixed yuan/100m).
        paired=disclosures.join(master.issue_size.rename("master_issue_size"),on="ts_code")
        if not np.isclose(paired.issue_size,paired.master_issue_size,rtol=1e-6,atol=1.).all():
            raise ValueError("Share disclosure issuance units disagree with master")
        events=cb_disclosure_events(disclosures,dates,"2026-10-14")
        known=align_disclosed_balances(events,dates,list(master.index),max_age_days=120)
        known,origin=fill_preconversion_known_balance(known,registry.reindex(master.index),dates)
        origin.to_parquet(out/"known_balance_origin.parquet")
        known["balance"].to_parquet(out/"last_known_balance_yuan.parquet")
        known["issue_size"].to_parquet(out/"last_known_issue_size_yuan.parquet")
        events.to_parquet(out/"dated_balance_events.parquet",index=False)
        annual=pd.DataFrame({"quoted":close.gt(0).sum(axis=1),
            "known_balance_available_quoted":(known["balance"].notna()&close.gt(0)).sum(axis=1),
            "known_positive_balance_quoted":(known["balance"].gt(0)&close.gt(0)).sum(axis=1)})
        annual["coverage_fraction"]=annual.known_balance_available_quoted/annual.quoted.replace(0,np.nan)
        annual.groupby(annual.index.year).agg(["mean","min","max"]).to_csv(report/"annual_balance_availability.csv")
        calls=pd.concat([pd.read_parquet(ROOT/f"data_raw/cb_database_20261014_v49/exit_contract/calls_{i}.parquet") for i in [0,1]],ignore_index=True)
        excluded,call_events=known_exit_exclusions(calls,master,dates,maturity_buffer_days=7)
        signal_close=close.mask(excluded)
        excluded.to_parquet(out/"known_exit_exclusions.parquet")
        master.to_parquet(out/"historical_public_universe.parquet")
        historical=pd.read_parquet(old/"balance_individual.parquet").reindex(columns=master.index)*1e8
        hist_issue=pd.DataFrame(np.broadcast_to(master.issue_size.to_numpy(),historical.shape),index=dates,columns=master.index)
        variants={"dated_disclosures_and_preconversion_contract":(known["balance"],known["issue_size"]),
            "historical_balance_diagnostic":(historical,hist_issue)}
        records=[];holdings=[];missing=[];books={}
        for i in range(1,len(dates)):
            day,signal=dates[i],dates[i-1]
            for variant,(balance,size) in variants.items():
                groups=cb_bucket_weights(signal_close.loc[signal],balance.loc[signal],size.loc[signal])
                for name in NAMES:
                    detail=groups.get(name)
                    if detail is None:
                        records.append(dict(variant=variant,date=day,signal_date=signal,asset_code=name,return_1d=np.nan,
                            member_count=0,max_weight=np.nan,weight_sum=np.nan,fallback_level=-1,one_way_turnover=np.nan,cost_5bp=np.nan))
                        books.pop((variant,name),None);continue
                    w=detail["weights"]
                    r=realized_bucket_return(w,returns.loc[day])
                    turnover=rebalance_turnover(books.get((variant,name)),returns.loc[signal],w)
                    records.append(dict(variant=variant,date=day,signal_date=signal,asset_code=name,return_1d=r,
                        member_count=len(w),max_weight=float(w.max()),weight_sum=float(w.sum()),fallback_level=detail["fallback_level"],
                        one_way_turnover=turnover,cost_5bp=turnover*.0005))
                    for code,weight in w.items():
                        holdings.append((variant,day,signal,name,code,float(weight)))
                        if not np.isfinite(returns.at[day,code]) or returns.at[day,code]<=-1:
                            missing.append(dict(variant=variant,date=day,signal_date=signal,asset_code=name,ts_code=code,
                                precommitted_weight=float(weight),resolution="unresolved_not_renormalized"))
                    books[(variant,name)]=w
            if i%200==0 or i==len(dates)-1:
                print(f"V50 rebuilt {i}/{len(dates)-1}, {day.date()}, unresolved holdings={len(missing)}",flush=True)
                write_json(progress,dict(status="running",completed_days=i,total_days=len(dates)-1))
        result=pd.DataFrame(records)
        result["net_return_5bp_diagnostic"]=result.return_1d-result.cost_5bp
        result.to_parquet(out/"cb_12_price_return_buckets.parquet",index=False)
        pd.DataFrame(holdings,columns=["variant","date","signal_date","asset_code","ts_code","weight"]).to_parquet(out/"precommitted_holdings.parquet",index=False)
        pd.DataFrame(missing).to_csv(report/"unresolved_held_positions.csv",index=False)
        result.groupby(["variant",result.date.dt.year]).return_1d.agg(["size","count"]).to_csv(report/"bucket_availability_by_year.csv")
        statistics=[]
        for variant,g in result.groupby("variant"):
            statistics.append(dict(variant=variant,total_bucket_days=len(g),valid_bucket_days=int(g.return_1d.notna().sum()),
                missing_bucket_days=int(g.return_1d.isna().sum()),no_eligible_bucket_days=int(g.member_count.eq(0).sum()),
                first_valid_day=str(g.loc[g.return_1d.notna(),"date"].min()),mean_one_way_turnover=float(g.one_way_turnover.mean())))
        status=dict(status="rebuilt_full_disclosure_candidates_not_promoted",cutoff="2026-10-14",
            public_issue_method_verified_codes=len(master),active_codes_without_verified_method=len(unknown),
            dated_balance_event_codes=events.ts_code.nunique(),dated_balance_events=len(events),
            preconversion_contract_balance_cells=int(origin.eq("published_issue_result_before_contract_conversion").sum().sum()),
            unresolved_position_rows=len(missing),statistics=statistics,
            training_ready=False,formal_data_replaced=False,
            blockers=["remaining_halt_and_terminal_marks_need_tradeability_ledger",
                "price_return_not_yet_coupon_and_terminal_total_return",
                "daily_bucket_reconstitution_not_yet_certified_executable"])
        write_json(report/"rebuild_status.json",status);write_json(progress,status)
        print(json.dumps(status,ensure_ascii=False,indent=2),flush=True)
    except Exception as exc:
        write_json(progress,dict(status="failed",error=f"{type(exc).__name__}: {exc}"));raise


if __name__=="__main__":main()

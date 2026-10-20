"""Audit a completed local-database rebuild without approving model deployment."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from src.v45_forecast import write_json


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--tag",default="20261014_v49")
    args=ap.parse_args()
    if not args.tag.replace("_","").isalnum():
        raise ValueError("Invalid tag")
    out=ROOT/"data_processed"/f"cb_rebuild_{args.tag}"
    report=ROOT/"output/reports"/f"cb_rebuild_{args.tag}"
    raw=ROOT/"data_raw"/f"cb_database_{args.tag}"
    status=json.loads((report/"rebuild_status.json").read_text(encoding="utf-8"))
    checks=[]
    def check(name,condition):
        checks.append(dict(check=name,passed=bool(condition)))
    manifest=json.loads((raw/"manifest.json").read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        h=hashlib.sha256()
        with (ROOT/entry["snapshot"]).open("rb") as f:
            for chunk in iter(lambda:f.read(1024*1024),b""):
                h.update(chunk)
        check("snapshot_hash:"+entry["snapshot"],h.hexdigest()==entry["sha256"])
    result=pd.read_parquet(out/"cb_12_buckets_rebuilt.parquet")
    held=pd.read_parquet(out/"precommitted_holdings.parquet")
    master=pd.read_parquet(out/"historical_security_master_audited.parquet")
    close=pd.read_parquet(out/"close_individual.parquet")
    pre=pd.read_parquet(out/"preclose_individual.parquet")
    check("unique_return_keys",not result.duplicated(["variant","date","asset_code"]).any())
    check("strictly_prior_signal",result.signal_date.lt(result.date).all())
    check("all_12_buckets_each_day",result.groupby(["variant","date"]).asset_code.nunique().eq(12).all())
    check("cutoff",result.date.max()==pd.Timestamp(status["cutoff"]))
    active=result.member_count.gt(0)
    check("normalized",np.allclose(result.loc[active,"weight_sum"],1,atol=1e-10))
    check("minimum_10_members",result.loc[active,"member_count"].ge(10).all())
    check("true_10pct_cap",result.loc[active,"max_weight"].le(.1+1e-10).all())
    check("no_inf_returns",not np.isinf(result.return_1d.to_numpy()).any())
    check("no_cash_token",not held.ts_code.str.contains("cash",case=False).any())
    listing=held.ts_code.map(master.list_date_parsed)
    check("not_before_listing",held.signal_date.ge(listing).all())
    check("no_unapproved_universe_member",held.ts_code.isin(master.index[master.included_research_universe]).all())
    check("held_quote_known_at_signal",np.isfinite(
        close.to_numpy()[close.index.get_indexer(held.signal_date),close.columns.get_indexer(held.ts_code)]).all())
    missing=pd.read_csv(report/"unresolved_held_positions.csv")
    if not missing.empty:
        missing["date"]=pd.to_datetime(missing.date)
        bad=missing[["variant","date","asset_code"]].drop_duplicates()
        joined=bad.merge(result,on=["variant","date","asset_code"],validate="one_to_one")
        check("missing_held_return_invalidates_bucket",joined.return_1d.isna().all())
    for _,g in result.sort_values("date").groupby(["variant","asset_code"]):
        gaps=g.return_1d.isna().to_numpy().cumsum()>0
        check("no_gap_bridge:"+str(g.asset_code.iloc[0])+":"+str(g.variant.iloc[0]),g.loc[gaps,"nav_no_gap_fill"].isna().all())
    old_missing=ROOT/"output/reports/v49_project_review/cb_missing_held_components.csv"
    old_missing_status={}
    if old_missing.exists():
        prior=pd.read_csv(old_missing,parse_dates=["date"])
        cells=prior[["date","ts_code"]].drop_duplicates().copy()
        cells["new_close"]=[close.at[d,c] if d in close.index and c in close else np.nan for d,c in zip(cells.date,cells.ts_code)]
        cells["new_preclose"]=[pre.at[d,c] if d in pre.index and c in pre else np.nan for d,c in zip(cells.date,cells.ts_code)]
        cells["observed_price_return_now_available"]=cells.new_close.gt(0)&cells.new_preclose.gt(0)
        cells.to_csv(report/"previous_missing_quote_resolution.csv",index=False)
        old_missing_status=dict(unique_prior_missing_security_days=len(cells),
            now_observed=int(cells.observed_price_return_now_available.sum()),
            still_missing=int((~cells.observed_price_return_now_available).sum()))
    observed=pd.read_parquet(out/"individual_observed_price_returns.parquet")
    extreme=observed.where(observed.abs().gt(.25)).stack().rename("return_1d").reset_index()
    extreme.columns=["date","ts_code","return_1d"]
    extreme.to_csv(report/"large_observed_moves_not_clipped.csv",index=False)
    audit=dict(engineering_passed=all(x["passed"] for x in checks),checks=checks,
        training_ready=False,promotion_passed=False,old_missing_quote_resolution=old_missing_status,
        individual_moves_over_25pct=len(extreme),
        note="Observation availability does not certify trading, terminal payouts or publication-time balance vintages.")
    write_json(report/"audit.json",audit)
    print(json.dumps({k:v for k,v in audit.items() if k!="checks"},ensure_ascii=False,indent=2))
    print(f"Checks: {sum(x['passed'] for x in checks)}/{len(checks)}")
    if not audit["engineering_passed"]:
        raise SystemExit(1)


if __name__=="__main__":
    main()

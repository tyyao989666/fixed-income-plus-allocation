"""Final reconstruction checks, not an approval of source PIT or portfolio alpha."""
from pathlib import Path
import hashlib
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from src.v49_cb_database import historical_universe
from src.v45_forecast import write_json


def main():
    base=ROOT/"data_processed/cb_rebuild_20261014_v49"
    report=ROOT/"output/reports/cb_rebuild_20261014_v49"
    out=base/"exit_aware_repaired"
    status=json.loads((report/"exit_aware_repaired/status.json").read_text(encoding="utf-8"))
    repair=json.loads((report/"quote_repair/status.json").read_text(encoding="utf-8"))
    checks=[]
    def check(name,value):checks.append(dict(check=name,passed=bool(value)))
    for entry in repair["files"]:
        check("quote_hash:"+Path(entry["path"]).name,
            hashlib.sha256((ROOT/entry["path"]).read_bytes()).hexdigest()==entry["sha256"])
    original=pd.read_parquet(base/"historical_security_master_audited.parquet")
    master=historical_universe(original.rename_axis("ts_code").reset_index(),list(original.index))
    master.loc[master.included_research_universe].to_parquet(out/"historical_security_master_audited.parquet")
    r=pd.read_parquet(out/"cb_12_buckets_rebuilt.parquet")
    h=pd.read_parquet(out/"precommitted_holdings.parquet")
    mask=pd.read_parquet(out/"known_exit_exclusions.parquet")
    check("unique_holdings",not h.duplicated(["variant","date","asset_code","ts_code"]).any())
    check("unique_returns",not r.duplicated(["variant","date","asset_code"]).any())
    check("12_slots_every_day",r.groupby(["variant","date"]).asset_code.nunique().eq(12).all())
    check("prior_signal_only",h.signal_date.lt(h.date).all())
    check("cutoff",r.date.max()==pd.Timestamp("2026-10-14"))
    check("public_candidate_screen",h.ts_code.isin(master.index[master.included_research_universe]).all())
    check("private_110815_excluded",not h.ts_code.eq("110815.SH").any())
    check("listing_known",h.signal_date.ge(h.ts_code.map(master.list_date_parsed)).all())
    check("announced_exclusion_obeyed",not mask.to_numpy()[mask.index.get_indexer(h.signal_date),mask.columns.get_indexer(h.ts_code)].any())
    active=r.member_count.gt(0)
    check("weight_sum",np.allclose(r.loc[active,"weight_sum"],1,atol=1e-10))
    check("weight_cap",r.loc[active,"max_weight"].le(.1+1e-10).all())
    check("member_floor",r.loc[active,"member_count"].ge(10).all())
    check("nonnegative_costs",r.cost_5bp.dropna().ge(0).all())
    check("half_l1_turnover_bounded",r.one_way_turnover.dropna().between(-1e-10,1+1e-10).all())
    check("net_cost_identity",np.allclose(r.net_return_5bp_diagnostic,r.return_1d-r.cost_5bp,equal_nan=True))
    check("no_infinite_returns",not np.isinf(r.return_1d.to_numpy()).any())
    missing=pd.read_csv(report/"exit_aware_repaired/unresolved_held_positions.csv",parse_dates=["date"])
    keys=missing[["variant","date","asset_code"]].drop_duplicates()
    j=keys.merge(r,on=["variant","date","asset_code"],validate="one_to_one")
    check("missing_held_marks_not_renormalized",j.return_1d.isna().all())
    for key,g in r.sort_values("date").groupby(["variant","asset_code"]):
        gap=g.return_1d.isna().cumsum().gt(0)
        check("no_nav_bridge:"+str(key),g.loc[gap,"nav_no_gap_fill"].isna().all())
    remaining=missing[["date","ts_code"]].drop_duplicates().join(
        master[["bond_short_name","maturity_date","delist_date","conv_stop_date"]],on="ts_code")
    remaining.to_csv(report/"exit_aware_repaired/remaining_security_days.csv",index=False)
    accounting=r.groupby("variant").agg(mean_one_way_turnover=("one_way_turnover","mean"),
        mean_cost_5bp=("cost_5bp","mean"),valid_bucket_days=("return_1d","count"))
    accounting["annualized_cost_252day_diagnostic"]=accounting.mean_cost_5bp*252
    accounting.to_csv(report/"exit_aware_repaired/internal_cost_diagnostic.csv")
    close=pd.read_parquet(base/"quote_repair/close_individual.parquet").reindex(columns=mask.columns)
    quoted=close.gt(0)
    features=pd.DataFrame(index=close.index)
    features["quoted_public_candidate_count"]=quoted.sum(axis=1)
    features["exit_eligible_quoted_count"]=(quoted&~mask).sum(axis=1)
    features["cb_price_median"]=close.where(quoted).median(axis=1)
    for field in ["premium","parity","ytm","remaining_years","iv_strict"]:
        f=pd.read_parquet(base/f"{field}_individual.parquet").reindex(index=close.index,columns=close.columns).where(quoted)
        features[f"cb_{field}_median"]=f.median(axis=1)
        features[f"cb_{field}_observed_count"]=f.notna().sum(axis=1)
    amount=pd.read_parquet(base/"quote_repair/amount_individual.parquet").reindex(columns=close.columns)
    features["cb_amount_yuan"]=(amount*10000).where(quoted).sum(axis=1,min_count=1)
    features.to_parquet(out/"cb_market_observed_features_no_interpolation.parquet")
    audit=dict(engineering_passed=all(x["passed"] for x in checks),checks=checks,
        training_ready=False,promotion_passed=False,remaining_unique_security_days=len(remaining),
        filled_observed_cells=repair["filled_cells"],remaining_missing_bucket_days=status["comparison"],
        external_database_modified=False,old_model_and_portfolio_overwritten=False,
        warning="Release-vintage gaps, remaining marks, cashflows and tradability remain unapproved.")
    audit["implementation_sha256"]={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [ROOT/"src/v49_cb_database.py",ROOT/"src/v49_point_in_time.py",
            ROOT/"scripts/rebuild_v49_cb_from_latest_database.py",ROOT/"scripts/rebuild_v49_cb_exit_aware.py",
            ROOT/"scripts/repair_v49_cb_targeted_quotes.py",ROOT/"scripts/audit_v49_cb_exit_repair.py"]}
    write_json(report/"exit_aware_repaired/audit.json",audit)
    if audit["engineering_passed"]:
        registration=dict(cutoff="2026-10-14",status="research_candidate_only",training_ready=False,
            source_workbook="data_raw/cb_database_20261014_v49/转债数据库_20261014.xlsx",
            latest_cb_candidate=str((out/"cb_12_buckets_rebuilt.parquet").relative_to(ROOT)),
            latest_cb_features=str((out/"cb_market_observed_features_no_interpolation.parquet").relative_to(ROOT)),
            latest_cb_security_master=str((out/"historical_security_master_audited.parquet").relative_to(ROOT)),
            audit=str((report/"exit_aware_repaired/audit.json").relative_to(ROOT)),
            historical_CB_reconstruction_verified=False,legacy_38_asset_dataset_replaced=False)
        write_json(ROOT/"data_processed/cb_latest_reconstruction.json",registration)
        readiness_path=ROOT/"output/reports/v49_project_review/data_readiness.json"
        if readiness_path.exists():
            readiness=json.loads(readiness_path.read_text(encoding="utf-8"))
            readiness["latest_local_cb_reconstruction"]=registration
            readiness["training_ready"]=False
            write_json(readiness_path,readiness)
    print(json.dumps({k:v for k,v in audit.items() if k!="checks"},ensure_ascii=False,indent=2))
    print(f"Checks: {sum(x['passed'] for x in checks)}/{len(checks)}")
    if not audit["engineering_passed"]:raise SystemExit(1)


if __name__=="__main__":main()

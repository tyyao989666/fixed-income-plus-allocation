"""Audit real vendor observations and PIT reconstruction without claiming tradability."""
from pathlib import Path
import hashlib
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from src.v45_forecast import write_json


def main():
    raw=ROOT/"data_raw/cb_evidence_v50"
    out=ROOT/"data_processed/cb_evidence_v50"
    report=ROOT/"output/reports/cb_evidence_v50"
    checks=[]
    def check(name,condition):checks.append(dict(check=name,passed=bool(condition)))
    status=json.loads((report/"rebuild_status.json").read_text(encoding="utf-8"))
    result=pd.read_parquet(out/"cb_12_price_return_buckets.parquet")
    held=pd.read_parquet(out/"precommitted_holdings.parquet")
    master=pd.read_parquet(out/"historical_public_universe.parquet")
    registry=pd.read_parquet(out/"historical_issuance_registry.parquet")
    origin=pd.read_parquet(out/"known_balance_origin.parquet")
    events=pd.read_parquet(out/"dated_balance_events.parquet")
    excluded=pd.read_parquet(out/"known_exit_exclusions.parquet")
    close=pd.read_parquet(out/"quotes/close_individual.parquet").reindex(columns=master.index)
    balance=pd.read_parquet(out/"last_known_balance_yuan.parquet")
    annual=pd.DataFrame({"quoted":close.gt(0).sum(axis=1),
        "known_balance_available_quoted":(balance.notna()&close.gt(0)).sum(axis=1),
        "known_positive_balance_quoted":(balance.gt(0)&close.gt(0)).sum(axis=1)})
    annual["coverage_fraction"]=annual.known_balance_available_quoted/annual.quoted.replace(0,np.nan)
    annual.groupby(annual.index.year).agg(["mean","min","max"]).to_csv(report/"annual_balance_availability.csv")
    quoted=close.gt(0)
    features=pd.DataFrame(index=close.index)
    features["quoted_public_cb_count"]=quoted.sum(axis=1)
    features["exit_rule_eligible_quote_count"]=(quoted&~excluded).sum(axis=1)
    features["cb_price_median"]=close.where(quoted).median(axis=1)
    for field in ["premium","parity","ytm","remaining_years","iv_strict"]:
        f=pd.read_parquet(ROOT/f"data_processed/cb_rebuild_20261014_v49/{field}_individual.parquet").reindex(index=close.index,columns=master.index).where(quoted)
        features[f"cb_{field}_median"]=f.median(axis=1)
        features[f"cb_{field}_observed_count"]=f.notna().sum(axis=1)
    amount=pd.read_parquet(out/"quotes/amount_individual.parquet").reindex(columns=master.index).where(quoted)
    features["cb_amount_observed_count"]=amount.notna().sum(axis=1)
    features["cb_amount_yuan_partial"]=(amount*10000).sum(axis=1,min_count=1)
    features["cb_amount_yuan_complete_only"]=features.cb_amount_yuan_partial.where(features.cb_amount_observed_count.eq(features.quoted_public_cb_count))
    features["cb_published_balance_coverage"]=annual.coverage_fraction
    features.to_parquet(out/"cb_market_observed_features_no_interpolation.parquet")
    check("unique_target_keys",not result.duplicated(["variant","date","asset_code"]).any())
    check("unique_holdings",not held.duplicated(["variant","date","asset_code","ts_code"]).any())
    check("12_buckets_every_day",result.groupby(["variant","date"]).asset_code.nunique().eq(12).all())
    check("full_daily_span",result.date.min()==pd.Timestamp("2018-01-03") and result.date.max()==pd.Timestamp("2026-10-14"))
    check("prior_signal",held.signal_date.lt(held.date).all())
    check("all_held_issue_methods_verified",held.ts_code.map(registry.public_method_verified).all())
    check("known_listing",held.signal_date.ge(held.ts_code.map(master.list_date_parsed)).all())
    check("no_private_names",not held.ts_code.map(master.bond_short_name).str.contains("定向|定转|定[0-9一二三四五六七八九十]|可交换|分离").any())
    active=result.member_count.gt(0)
    check("normalized_targets",np.allclose(result.loc[active,"weight_sum"],1,atol=1e-10))
    check("10pct_target_cap",result.loc[active,"max_weight"].le(.1+1e-10).all())
    check("10_member_floor",result.loc[active,"member_count"].ge(10).all())
    check("nonnegative_costs",result.cost_5bp.dropna().ge(0).all())
    check("one_way_turnover_range",result.one_way_turnover.dropna().between(-1e-10,1+1e-10).all())
    check("net_cost_identity",np.allclose(result.net_return_5bp_diagnostic,result.return_1d-result.cost_5bp,equal_nan=True))
    check("no_infinite_returns",not np.isinf(result.return_1d.to_numpy()).any())
    check("publication_after_period",events.end_date.le(events.publish_date).all())
    check("visibility_after_announcement",events.available_at.gt(events.publish_date).all())
    check("no_future_publication",events.publish_date.le(pd.Timestamp("2026-10-14")).all())
    strict=held.loc[held.variant.eq("dated_disclosures_and_preconversion_contract")]
    ix=origin.index.get_indexer(strict.signal_date);col=origin.columns.get_indexer(strict.ts_code)
    check("known_balance_at_all_strict_signals",origin.to_numpy()[ix,col].astype(str).tolist().count("missing")==0)
    baseline=origin.to_numpy()[ix,col]=="published_issue_result_before_contract_conversion"
    initial=strict.loc[baseline]
    check("initial_size_only_after_result",initial.signal_date.ge(initial.ts_code.map(registry.result_known_at)).all())
    check("initial_size_only_before_conversion",initial.signal_date.lt(initial.ts_code.map(registry.conversion_start)).all())
    check("announced_exclusions_obeyed",not excluded.to_numpy()[excluded.index.get_indexer(held.signal_date),excluded.columns.get_indexer(held.ts_code)].any())
    missing=pd.read_csv(report/"unresolved_held_positions.csv",parse_dates=["date"])
    keys=missing[["variant","date","asset_code"]].drop_duplicates()
    check("missing_held_returns_not_renormalized",keys.merge(result,on=["variant","date","asset_code"],validate="one_to_one").return_1d.isna().all())
    # All uncapped request leaves must contribute once to the assembled raw
    # table. Capped parents are diagnostic only, never treated as complete.
    total_leaf_rows=0;request_count=0;capped_count=0
    for path in sorted((raw/"tushare/disclosures").glob("group_*.json")):
        record=json.loads(path.read_text(encoding="utf-8"))
        f=pd.read_parquet(path.with_suffix(".parquet"))
        request_count+=1
        check("request_row_count:"+path.stem,len(f)==record["rows"])
        if len(f)>=2000:capped_count+=1
        else:total_leaf_rows+=len(f)
    all_raw=pd.read_parquet(raw/"tushare/disclosures/all_historical_disclosures_raw.parquet")
    check("all_uncapped_leaves_assembled",len(all_raw)==total_leaf_rows)
    repairs=pd.read_csv(report/"ifind_verified_quote_repairs.csv",parse_dates=["date"])
    evidence=pd.read_parquet(out/"quotes/ifind_quote_evidence.parquet")
    if len(repairs):
        proof=repairs.merge(evidence,on=["date","ts_code","source"],validate="many_to_one",suffixes=("_repair","_evidence"))
        check("only_positive_volume_repairs",proof.positive_volume.all())
        check("quote_identity_anchors",proof.anchor_verified.all() and proof.anchor_rows.ge(2).all())
        check("only_missing_fields_patched",all(pd.isna(pd.read_parquet(ROOT/f"data_processed/cb_rebuild_20261014_v49/quote_repair/{field}_individual.parquet").at[day,code])
            for day,code,field in repairs[["date","ts_code","field"]].itertuples(index=False,name=None)))
    needs=missing[["date","ts_code"]].drop_duplicates()
    details=needs.merge(evidence,on=["date","ts_code"],how="left")
    details=details.join(master[["bond_short_name","maturity_date","delist_date","conv_stop_date"]],on="ts_code")
    details["classification"]=np.select([
        details.positive_volume.eq(False)&details.anchor_verified.eq(True),
        details.positive_volume.eq(False)&details.close.notna(),
        details.close.isna()],
        ["vendor_no_transaction_mark_requires_frozen_inventory_ledger",
         "no_transaction_or_inconsistent_mark_requires_review",
         "no_dated_vendor_quote_requires_event_or_other_evidence"],default="unresolved")
    details.to_csv(report/"remaining_security_day_adjudication.csv",index=False)
    evidence.loc[evidence.anchor_verified&evidence.positive_volume.eq(False)].to_parquet(
        out/"nontransaction_marks_for_inventory_review_only.parquet",index=False)
    unit_counts=registry.unit_factor.value_counts(dropna=False).to_dict()
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in raw.rglob("*") if p.is_file()}
    source_files=[ROOT/"src/v50_cb_evidence.py",ROOT/"scripts/v50_cb_vendor_evidence.py",
        ROOT/"scripts/acquire_v50_cb_disclosures.py",ROOT/"scripts/reconcile_v50_cb_quotes.py",
        ROOT/"scripts/build_v50_cb_candidates.py",ROOT/"scripts/audit_v50_cb_data.py"]
    implementation={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    write_json(report/"source_manifest.json",dict(sha256=hashes,implementation_sha256=implementation))
    audit=dict(engineering_passed=all(x["passed"] for x in checks),checks=checks,
        training_ready=False,promotion_passed=False,source_request_count=request_count,capped_requests_split=capped_count,
        raw_disclosures=len(all_raw),remaining_unique_security_days=len(needs),
        quote_repaired_security_days=len(repairs[["date","ts_code"]].drop_duplicates()),
        issue_unit_factors={str(k):int(v) for k,v in unit_counts.items()},
        publication_coverage_rebuilt=True,active_issuance_method_gap_closed=status["active_codes_without_verified_method"]==0,
        wind_online_status="BALANCE_INSUFFICIENT_no_further_requests",
        unresolved_reason="no_transaction_marks_and_terminal_cashflows_are_not_tradable_prices",
        old_models_and_formal_38_asset_data_overwritten=False)
    write_json(report/"audit.json",audit)
    if audit["engineering_passed"]:
        registration=dict(cutoff="2026-10-14",status="full_disclosure_research_candidate_only",training_ready=False,
            latest_cb_candidate="data_processed/cb_evidence_v50/cb_12_price_return_buckets.parquet",
            latest_cb_security_master="data_processed/cb_evidence_v50/historical_public_universe.parquet",
            latest_cb_features="data_processed/cb_evidence_v50/cb_market_observed_features_no_interpolation.parquet",
            audit="output/reports/cb_evidence_v50/audit.json",historical_publication_coverage_rebuilt=True,
            active_issuance_method_gap_closed=True,legacy_38_asset_dataset_replaced=False,
            historical_CB_execution_and_total_return_verified=False)
        write_json(ROOT/"data_processed/cb_latest_reconstruction.json",registration)
        ready_path=ROOT/"output/reports/v49_project_review/data_readiness.json"
        if ready_path.exists():
            ready=json.loads(ready_path.read_text(encoding="utf-8"))
            ready["latest_local_cb_reconstruction"]=registration
            ready["training_ready"]=False
            ready["blockers"]=["CB_execution_and_cashflows_not_certified_after_full_disclosure_rebuild" if b=="full_history_CB_constituent_rebuild_required_before_old_metrics_can_be_revalidated" else b for b in ready["blockers"]]
            write_json(ready_path,ready)
    print(json.dumps({k:v for k,v in audit.items() if k!="checks"},ensure_ascii=False,indent=2))
    print(f"Checks: {sum(x['passed'] for x in checks)}/{len(checks)}")
    if not audit["engineering_passed"]:raise SystemExit(1)


if __name__=="__main__":main()

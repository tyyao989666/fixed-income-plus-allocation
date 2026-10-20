"""Observed-data predictability triage and evidence-only ledger readiness."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_v55_common_predictions import attach_truth
from scripts.run_v52_execution_preflight import save, sha


def main():
    out = ROOT/"output/reports/multitask_v57"
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((ROOT/"output/reports/temporal_opportunity_v55/metadata.json").read_text())
    with np.load(ROOT/"data_processed/temporal_opportunity_v55/dataset.npz") as z:
        a = {k:z[k] for k in z.files if "target" in k or "opportunity" in k or k == "dates"}
    rows = []
    for year in range(2022,2027):
        f = pd.read_parquet(ROOT/f"output/models/total_view_rank_v56/{year}/predictions.parquet",
                            filters=[("seed","=",-1)])
        f = attach_truth(f, a, meta, True)
        f = f[f.observed_label & np.isfinite(f.truth) & np.isfinite(f.prediction)]
        for (model,task,entity,h), g in f.groupby(["model","task","entity","horizon_days"]):
            residual = g.truth-g.baseline_prediction
            increment = g.prediction-g.baseline_prediction
            def corr(x,y):
                return float(x.rank().corr(y.rank())) if x.nunique()>1 and y.nunique()>1 and len(x)>10 else None
            rows.append(dict(year=year,model=model,task=task,entity=entity,horizon=int(h),
                observed_days=len(g),approx_nonoverlap_blocks=len(g)//h,
                timing_rank_ic=corr(g.prediction,g.truth),
                baseline_timing_rank_ic=corr(g.baseline_prediction,g.truth),
                residual_timing_rank_ic=corr(increment,residual),
                increment_std=float(increment.std()),baseline_std=float(g.baseline_prediction.std()),
                target_std=float(g.truth.std()),raw_return_mae=float((g.truth-g.prediction).abs().mean())))
    pd.DataFrame(rows).to_csv(out/"predictability_by_entity_horizon.csv",index=False)

    coverage = pd.read_csv(ROOT/"output/reports/cb_execution_v52/cashflow_evidence_coverage.csv")
    contracts = pd.concat([pd.read_parquet(p) for p in (ROOT/"data_raw/unified_v53").glob("cb_coupon_contract_*.parquet")],ignore_index=True)
    contracts = contracts.drop_duplicates()
    planned = contracts.groupby("ts_code").size()
    held = coverage[coverage.ever_held.eq(True)].copy()
    held["contract_schedule_rows"] = held.ts_code.map(planned).fillna(0).astype(int)
    held["payment_history_certified"] = False
    held["required_evidence"] = "coupon record/ex/payment dates; per-unit gross amount; actual payment notice; terminal cancellation/payment; source and publication timestamp"
    held.to_csv(out/"cashflow_evidence_requests.csv",index=False)
    terminal = json.loads((ROOT/"output/reports/cb_execution_v54/status.json").read_text())
    ledger = dict(formal_ppo_allowed=False,certified_total_return_dataset=False,
        dataset_sha256=sha(ROOT/"data_processed/temporal_opportunity_v55/dataset.npz"),
        requested_assets=38,ever_held_individual_CBs=len(held),
        local_contract_rows=len(contracts),contract_schedule_is_not_actual_payment=True,
        verified_terminal_events=terminal["reviewed_terminal_events"],
        unresolved_book_security_days=terminal["current_missing_book_security_days"],
        blockers=["complete_actual_CB_coupon_and_terminal_payment_history_not_certified",
                  "unresolved_inventory_marks", "REIT_payment_and_execution_ledger_not_certified",
                  "synthetic_rate_and_US_bond_proxies_are_not_executable_instruments",
                  "security_specific_costs_and_slippage_not_independently_verified"],
        no_synthetic_events_added=True,original_sources_modified=False)
    save(out/"ledger_readiness.json",ledger)
    save(out/"pipeline_status.json",dict(status="audit_complete_training_pending",forecast_training_started=False,
        ppo_training_started=False,formal_promotion=False,
        known_grid_limitation=meta["grid_definition"]["distinction"],
        previously_inspected_period="2024-2026 not an untouched holdout",
        predictability_rows=len(rows),ledger=ledger))
    print(json.dumps(ledger,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()

"""Audit actual unified inputs, references, cashflow contracts and forecast readiness."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from scripts.run_v52_execution_preflight import sha, save
from src.v53_unified_data import validate_relative_dataset, independent_cb_relative_labels


def main():
    out = ROOT / "data_processed/unified_v53"
    report = ROOT / "output/reports/unified_v53"
    status = json.loads((report / "status.json").read_text(encoding="utf-8"))
    meta = json.loads((report / "metadata.json").read_text(encoding="utf-8"))
    with np.load(out / "dataset.npz", allow_pickle=False) as f:
        a = {k:f[k] for k in f.files}
    validate_relative_dataset(a)
    checks = []

    def check(name, value):
        checks.append(dict(check=name, passed=bool(value)))

    check("dataset_registration_hash", sha(out / "dataset.npz") == status["dataset_sha256"])
    check("metadata_registration_hash", sha(report / "metadata.json") == status["metadata_sha256"])
    for name, item in meta["source_manifest"].items():
        check("source_hash:" + name, sha(ROOT / item["path"]) == item["sha256"])
    dates = pd.to_datetime(a["dates"], unit="ns")
    check("calendar_and_all_38_tokens", len(dates)==2128 and a["returns"].shape==(2128,38))
    check("requested_end", dates[-1] == pd.Timestamp("2026-10-14"))
    check("observations_at_final_session", a["return_masks"][-1].all())
    check("nine_grid_and_two_continuous_axes", a["macro_values"].shape==(2128,11))
    check("missing_ret_not_fabricated", (a["returns"][~a["return_masks"]] == 0).all())
    check("all_38_assets_retain_individual_targets", a["asset_target"].shape==(2128,3,38))
    for h, horizon in enumerate([5,20,60]):
        check(f"tail_{horizon}_unmatured", not a["asset_target_mask"][-horizon:,h].any())
        check(f"label_end_{horizon}", np.array_equal(a["label_end"][:-horizon,h], a["dates"][horizon:]))
    us = pd.read_parquet(out / "us_curve_visibility_audit.parquet")
    fx = pd.read_parquet(out / "cny_parity_visibility_audit.parquet")
    check("US_source_observed_before_domestic_signal", (us.source_date.dropna() < us.source_date.dropna().index).all())
    check("FX_source_observed_before_domestic_signal", (fx.source_date.dropna() < fx.source_date.dropna().index).all())
    for asset, start in [("alt_reits","2021-06-22"),("equity_tech_growth","2020-07-23")]:
        index = meta["assets"].index(asset)
        check("no_prelisting_fill:" + asset, not a["return_masks"][dates < pd.Timestamp(start), index].any())
    # Recompute exact actual labels, not only their shape or non-NaN counts.
    refs = pd.read_parquet(out / "cb_reference_returns.parquet").iloc[:,0].to_numpy()
    labels = independent_cb_relative_labels(a["dates"], a["returns"], a["return_masks"], a["module_index"],
        a["sleeve_reference_weights"], a["balanced_reference_weights"], refs)
    for key in labels:
        check("exact_relative_label_reconstruction:" + key, np.array_equal(labels[key], a[key]))
    # Past targets must be invariant to returns beyond their horizon endpoint.
    altered = a["returns"].copy()
    altered[-100:] = .1
    changed = independent_cb_relative_labels(a["dates"], altered, a["return_masks"], a["module_index"],
        a["sleeve_reference_weights"], a["balanced_reference_weights"], refs)
    check("mature_past_labels_future_perturbation_invariance", np.array_equal(changed["asset_target"][:-160], a["asset_target"][:-160]))
    pd.DataFrame(checks).to_csv(report / "data_artifact_checks.csv", index=False)
    # Preserve contractual schedules separately from realized payment events.
    rates = pd.concat([pd.read_parquet(p) for p in (ROOT / "data_raw/unified_v53").glob("cb_coupon_contract_*.parquet")],ignore_index=True)
    rates.to_parquet(out / "cb_coupon_contracts_NOT_payment_events.parquet", index=False)
    calls = pd.concat([pd.read_parquet(ROOT / f"data_raw/cb_database_20261014_v49/exit_contract/calls_{i}.parquet") for i in [0,1]],ignore_index=True)
    unresolved = pd.read_parquet(ROOT / "data_processed/cb_execution_v52/unresolved_frozen_inventory.parquet")
    master = pd.read_parquet(ROOT / "data_processed/cb_evidence_v50/historical_public_universe.parquet")
    queue = unresolved.groupby("ts_code").agg(first_unresolved=("date","min"),last_unresolved=("date","max"),
        book_security_days=("date","size")).join(master[["bond_short_name","maturity_date","stk_code"]])
    for code in queue.index:
        rows = calls.loc[calls.ts_code.eq(code)]
        queue.loc[code,"coupon_contract_rows"] = int(rates.ts_code.eq(code).sum())
        queue.loc[code,"redemption_announcement_rows"] = len(rows)
        queue.loc[code,"has_payment_date_field"] = rows.payment_date.notna().any()
        queue.loc[code,"has_record_date_field"] = rows.call_reg_date.notna().any()
    queue.sort_values("book_security_days", ascending=False).to_csv(report / "cb_remaining_evidence_queue.csv")
    summary = dict(passed=all(x["passed"] for x in checks), checks=len(checks),
        dataset_sha256=sha(out / "dataset.npz"), metadata_sha256=sha(report / "metadata.json"),
        unified_calendar_assets=38, final_date=str(dates[-1].date()),
        coupon_contract_rows=len(rates), coupon_contract_codes=rates.ts_code.nunique(),
        actual_coupon_payment_history_certified=False,
        remaining_inventory_evidence_codes=len(queue), formal_portfolio_promoted=False)
    save(report / "data_audit.json", summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    if not summary["passed"]:
        raise SystemExit("Unified source/label audit failed")


if __name__ == "__main__":
    main()

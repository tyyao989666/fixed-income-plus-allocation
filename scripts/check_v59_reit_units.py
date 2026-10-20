"""Vendor schema and arithmetic reconciliation, not payment certification."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    out = ROOT / "output/reports/raw_mean_v59"
    source = out / "reit_field_units_source.json"
    metadata = json.loads(source.read_text())
    rows = []
    for p in sorted((ROOT / "data_raw/unified_v53").glob("reit_div_*.parquet")):
        f = pd.read_parquet(p)
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        for index, r in f.iterrows():
            per_unit = pd.to_numeric(r.get("div_cash"), errors="coerce")
            units = pd.to_numeric(r.get("base_unit"), errors="coerce")*10000
            amount = pd.to_numeric(r.get("ear_amount"), errors="coerce")
            expected = per_unit*units
            usable = bool(np.isfinite([per_unit, units, amount]).all() and units > 0 and amount > 0)
            difference = abs(expected-amount)/amount if usable else None
            row = dict(ts_code=r.get("ts_code"), ex_date=r.get("ex_date"), pay_date=r.get("pay_date"),
                vendor_ann_date=r.get("ann_date"), vendor_implementation_ann_date=r.get("imp_anndate"),
                vendor_status=r.get("div_proc"),
                cash_CNY_per_unit_vendor=per_unit, base_fund_units=units, total_CNY_vendor=amount,
                reconstructed_total_CNY=expected, relative_amount_difference=difference,
                arithmetic_check_available=usable, arithmetic_within_0_1pct=bool(usable and difference <= .001),
                source_path=str(p.relative_to(ROOT)), source_row=int(index), source_sha256=digest,
                unit_documentation_url=metadata["url"], vendor_unit_documented=True,
                issuer_notice_independently_verified=False, actual_receipt_verified=False,
                execution_certified=False, synthetic=False)
            rows.append(row)
    frame = pd.DataFrame(rows)
    keys = ["ts_code", "ex_date", "pay_date"]
    frame["same_event_key_record_count"] = frame.groupby(keys, dropna=False).ts_code.transform("size")
    frame["event_version_requires_review"] = frame.same_event_key_record_count > 1
    frame.to_csv(out / "reit_unit_reconciliation.csv", index=False, encoding="utf-8-sig")
    exceptions = frame[~frame.arithmetic_within_0_1pct].copy()
    exceptions["priority"] = np.where(exceptions.relative_amount_difference > .05, "large_difference_review_first", "rounding_or_field_basis_review")
    exceptions["action"] = "do_not_infer_correct_cash_amount; verify_issuer_notice_and_record_version"
    exceptions.to_csv(out / "reit_amount_review_queue.csv", index=False, encoding="utf-8-sig")
    frame[frame.event_version_requires_review].to_csv(out / "reit_event_version_review.csv", index=False, encoding="utf-8-sig")
    result = dict(records=len(frame), vendor_unit_documented_records=len(frame),
        arithmetic_check_available=int(frame.arithmetic_check_available.sum()),
        arithmetic_within_0_1pct=int(frame.arithmetic_within_0_1pct.sum()),
        arithmetic_requires_review=int((~frame.arithmetic_within_0_1pct).sum()),
        differences_above_5pct=int((frame.relative_amount_difference > .05).sum()),
        distinct_vendor_event_keys=len(frame[keys].drop_duplicates()),
        rows_in_repeated_event_keys=int(frame.event_version_requires_review.sum()),
        duplicate_records_not_automatically_counted_as_cash_payments=True,
        actual_receipts_certified=0, formal_PPO_allowed=False,
        documentation_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        limitations="Arithmetic agreement only checks unit consistency, not event completeness, PIT revisions, investor tax or actual receipt")
    (out / "reit_unit_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

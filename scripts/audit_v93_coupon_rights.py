"""Recompute coupon entitlements from the rerun's actual fills, not the lookup queue."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/reports/cb_cashflows_v93"
DATA = ROOT / "data_processed/cb_cashflows_v93"


def main():
    files = [OUT / "status.json", OUT / "actual_registered_coupon_claims_NOT_receipts.csv",
             OUT / "event_transitions.csv", DATA / "reviewed_events.parquet", DATA / "actual_fills.parquet",
             DATA / "daily_accounting_diagnostics.parquet", Path(__file__)]
    status = json.loads(files[0].read_text(encoding="utf-8"))
    for relative, expected in status["source_sha256"].items():
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
            raise ValueError("Source changed: " + relative)
    claims = pd.read_csv(files[1])
    transitions = pd.read_csv(files[2])
    events = pd.read_parquet(files[3])
    fills = pd.read_parquet(files[4])
    new = pd.read_parquet(files[5])
    coupons = events.loc[events.kind.eq("coupon")]
    if len(coupons) != 2 or coupons.confirmed_at.notna().any():
        raise ValueError("Unexpected coupon batch/confirmation")
    checks = []
    for event in coupons.itertuples(index=False):
        owned = fills[(fills.ts_code == event.ts_code) & (fills.date <= event.record_date)].groupby("asset_code").quantity_change.sum()
        registered = claims.loc[claims.event_id.eq(event.event_id)].set_index("asset_code")
        expected = owned.reindex(registered.index, fill_value=0)
        np.testing.assert_allclose(registered.quantity, expected, atol=1e-8, rtol=1e-10)
        np.testing.assert_allclose(registered.gross_claim_yuan, expected*event.cash_yuan_per_bond, atol=1e-7, rtol=1e-10)
        assert registered.actual_coupon_cash_yuan.eq(0).all()
        state = transitions.loc[transitions.event_id.eq(event.event_id)]
        assert len(state) == 1 and state.transition.eq("entitlement_to_receivable").all()
        assert pd.to_datetime(state.day).eq(event.ex_date).all()
        np.testing.assert_allclose(state.amount_all_books.iloc[0], registered.gross_claim_yuan.sum(), atol=1e-7)
        checks.append(dict(event_id=event.event_id, nonzero_baskets=int(registered.quantity.gt(1e-8).sum()),
                           gross_claim_yuan=float(registered.gross_claim_yuan.sum()), confirmed_coupon_cash_yuan=0,
                           record_date=str(event.record_date.date()), ex_date=str(event.ex_date.date()),
                           scheduled_pay_date=str(event.pay_date.date())))
    before_first_ex = coupons.ex_date.min()
    previous_path = ROOT / "data_processed/cb_cashflows_v92/daily_accounting_diagnostics.parquet"
    previous = pd.read_parquet(previous_path)
    left = previous.loc[previous.date < before_first_ex].reset_index(drop=True)
    right = new.loc[new.date < before_first_ex].reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right)
    assert new.certified_total_return_nav.isna().all()
    files.append(previous_path)
    result = dict(passed=True, scope="two_coupon_rights_and_pre_first_ex_accounting_prefix_not_full_strategy_certification",
                  events=checks, unchanged_prefix_basket_days=len(left),
                  schedule_only_coupon_cash_not_posted=True, full_product_backtest_completed=False,
                  training_started=False, goal_complete=False,
                  source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    (OUT / "coupon_rights_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "source_sha256"}, ensure_ascii=False))


if __name__ == "__main__":
    main()

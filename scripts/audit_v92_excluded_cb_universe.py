"""Verify the requested screen without claiming removal repairs the original NAV."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/reports/cb_cashflows_v92"
DATA = ROOT / "data_processed/cb_cashflows_v92"


def main():
    status_path = OUT / "status.json"
    state = json.loads(status_path.read_text(encoding="utf-8"))
    if state["status"] != "retrospective_screen_replayed_formal_gate_blocked":
        raise ValueError("Replay incomplete")
    for relative, expected in state["source_sha256"].items():
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Replay input changed: {relative}")
    removed = state["excluded_securities"]
    new = pd.read_parquet(DATA / "signal_buy_eligibility.parquet")
    old = pd.read_parquet(ROOT / "data_processed/cb_masked_research_v51/signal_buy_eligibility.parquet")
    assert not new[removed].to_numpy().any()
    retained = [c for c in old if c not in removed]
    pd.testing.assert_frame_equal(new[retained], old[retained])
    fills = pd.read_parquet(DATA / "actual_fills.parquet")
    final = pd.read_parquet(DATA / "final_quantities.parquet")
    daily = pd.read_parquet(DATA / "daily_accounting_diagnostics.parquet")
    assert not fills.ts_code.isin(removed).any()
    assert not final[removed].to_numpy().any()
    assert np.isfinite(final.to_numpy()).all() and (final.to_numpy() >= -1e-8).all()
    assert daily.certified_total_return_nav.isna().all()
    assert (daily.settlement_cash_yuan >= -1e-6).all()
    calendar = new.index
    previous = pd.Series(calendar[:-1], index=calendar[1:])
    assert pd.to_datetime(fills.signal_date).reset_index(drop=True).equals(fills.date.map(previous).reset_index(drop=True))
    fees = fills.assign(cost=lambda x: x.quantity_change.abs()*x.execution_close*.0005).groupby(["date", "asset_code"]).cost.sum()
    costs = daily.set_index(["date", "asset_code"]).cost_yuan
    np.testing.assert_allclose(costs, fees.reindex(costs.index, fill_value=0), atol=1e-6, rtol=1e-10)
    before = pd.read_parquet(ROOT / "data_processed/cb_cashflows_v88/unresolved_inventory.parquet")
    after = pd.read_parquet(DATA / "unresolved_inventory.parquet")
    keys = ["date", "asset_code", "ts_code"]
    comparison = before[keys].merge(after[keys], on=keys, how="outer", indicator=True, validate="one_to_one")
    comparison.to_csv(OUT / "gap_key_comparison_NOT_returns.csv", index=False, encoding="utf-8-sig")
    inputs = [status_path, Path(__file__), DATA / "signal_buy_eligibility.parquet", DATA / "actual_fills.parquet",
              DATA / "final_quantities.parquet", DATA / "daily_accounting_diagnostics.parquet", DATA / "unresolved_inventory.parquet"]
    result = dict(passed=True, source_hash_checks=len(state["source_sha256"]),
                  retained_security_eligibility_unchanged=True, excluded_fills_and_inventory_zero=True,
                  execution_signals_previous_session=True, five_bp_two_way_cost_check=True,
                  training_started=False, strict_no_lookahead_claim_allowed=False,
                  retrospective_screen_not_original_nav_repair=True,
                  old_gap_keys=len(before), new_gap_keys=len(after),
                  removed_gap_keys=int(comparison._merge.eq("left_only").sum()),
                  added_gap_keys=int(comparison._merge.eq("right_only").sum()),
                  ending_unresolved_securities=state["ending_unresolved_securities"],
                  source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs})
    (OUT / "audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "source_sha256"}, ensure_ascii=False))


if __name__ == "__main__":
    main()

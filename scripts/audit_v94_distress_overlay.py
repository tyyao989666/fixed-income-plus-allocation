"""Same-basis V93/V94 comparison: real quoted exits, no certified performance claim."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/reports/cb_cashflows_v94"
DATA = ROOT / "data_processed/cb_cashflows_v94"


def main():
    inputs = []
    def read(relative, json_file=False):
        path = ROOT / relative
        inputs.append(path)
        return json.loads(path.read_text(encoding="utf-8")) if json_file else pd.read_parquet(path)
    status = read("output/reports/cb_cashflows_v94/status.json", True)
    if status.get("version") != "v94_fixed_distress_exit_research":
        raise ValueError("V94 replay not finalized")
    for relative, expected in status["source_sha256"].items():
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
            raise ValueError("Bound source changed: " + relative)
    notes = read("output/reports/cb_cashflows_v94/reviewed_trigger_rules.json", True)
    fills = read("data_processed/cb_cashflows_v94/actual_fills.parquet")
    final = read("data_processed/cb_cashflows_v94/final_quantities.parquet")
    new = read("data_processed/cb_cashflows_v94/daily_accounting_diagnostics.parquet")
    old = read("data_processed/cb_cashflows_v93/daily_accounting_diagnostics.parquet")
    close = read("data_processed/cb_evidence_v50/quotes/close_individual.parquet")
    volume = read("data_processed/cb_evidence_v50/quotes/volume_individual.parquet")
    actions_path = OUT / "risk_exit_inventory_actions.csv"
    inputs.append(actions_path)
    actions = pd.read_csv(actions_path, parse_dates=["date", "signal_date"])
    summaries = []
    for note in notes:
        code, known = note["ts_code"], pd.Timestamp(note["known_at"])
        selected = fills[(fills.ts_code == code) & (fills.signal_date >= known)]
        assert not selected.quantity_change.gt(1e-8).any()
        sale = selected[selected.quantity_change < -1e-8]
        for row in sale.itertuples(index=False):
            assert close.at[row.date, code] == row.execution_close
            assert volume.at[row.date, code] > 0 and np.isfinite(volume.at[row.date, code])
        observed = actions.loc[actions.ts_code.eq(code)]
        assert not observed.loc[~observed.executable_flag, "fill_quantity"].abs().gt(1e-8).any()
        summaries.append(dict(ts_code=code, publication_date=note["publication_date"], known_at=known,
                              first_actual_sell_date=sale.date.min(), sell_rows=len(sale),
                              sold_quantity=float(-sale.quantity_change.sum()),
                              final_quantity=float(final[code].sum()),
                              unavailable_target_rows=int(observed.unavailable_target.sum()),
                              price_positive_volume_checks_passed=True,
                              order_book_capacity_certified=False))
    first_known = min(pd.Timestamp(n["known_at"]) for n in notes)
    # Signal at first-known close cannot affect execution before the next close.
    prefix_old = old.loc[old.date <= first_known].reset_index(drop=True)
    prefix_new = new.loc[new.date <= first_known].reset_index(drop=True)
    pd.testing.assert_frame_equal(prefix_old, prefix_new)
    before = read("data_processed/cb_cashflows_v93/unresolved_inventory.parquet")
    after = read("data_processed/cb_cashflows_v94/unresolved_inventory.parquet")
    keys = ["date", "asset_code", "ts_code"]
    gaps = before[keys].merge(after[keys], on=keys, how="outer", indicator=True, validate="one_to_one")
    gaps.to_csv(OUT / "gap_comparison_against_v93_NOT_performance.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summaries).to_csv(OUT / "actual_exit_summary.csv", index=False, encoding="utf-8-sig")
    attribution = old.groupby("asset_code").missing_held_count.sum().rename("v93_missing_position_days").to_frame()
    attribution["v94_missing_position_days"] = new.groupby("asset_code").missing_held_count.sum()
    attribution["v93_execution_cost_yuan"] = old.groupby("asset_code").cost_yuan.sum()
    attribution["v94_execution_cost_yuan"] = new.groupby("asset_code").cost_yuan.sum()
    attribution.to_csv(OUT / "same_basis_accounting_comparison_NOT_performance.csv")
    assert new.certified_total_return_nav.isna().all()
    result = dict(passed=True, prior_goal_turn_classification="progress", objective_complete=False,
                  unchanged_pre_trigger_basket_days=len(prefix_old),
                  future_risk_signals_not_backdated=True, post_trigger_buys_zero=True,
                  v93_missing_position_days=len(before), v94_missing_position_days=len(after),
                  removed_gap_keys=int(gaps._merge.eq("left_only").sum()),
                  added_gap_keys=int(gaps._merge.eq("right_only").sum()),
                  ending_unresolved_securities=status["ending_unresolved_securities"],
                  comprehensive_all_issuer_event_coverage=False, formal_promotion=False,
                  full_three_product_backtest_completed=False, training_started=False,
                  source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs+[Path(__file__)]})
    (OUT / "distress_overlay_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "source_sha256"}, ensure_ascii=False))


if __name__ == "__main__":
    main()

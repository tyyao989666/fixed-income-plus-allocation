"""Verify real replay artifacts and refusal to train on uncertified total returns."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from scripts.run_v52_execution_preflight import sha, save


def main():
    out = ROOT / "data_processed/cb_execution_v52"
    report = ROOT / "output/reports/cb_execution_v52"
    state = json.loads((report / "pipeline_status.json").read_text(encoding="utf-8"))
    checks = []

    def check(name, condition):
        checks.append(dict(check=name, passed=bool(condition)))

    frame = pd.read_parquet(out / "daily_price_only_execution_diagnostics.parquet")
    trades = pd.read_parquet(out / "actual_close_fills.parquet")
    unresolved = pd.read_parquet(out / "unresolved_frozen_inventory.parquet")
    close = pd.read_parquet(ROOT / "data_processed/cb_evidence_v50/quotes/close_individual.parquet")
    vol = pd.read_parquet(ROOT / "data_processed/cb_evidence_v50/quotes/volume_individual.parquet")
    check("twelve_books", frame.asset_code.nunique() == 12)
    check("complete_replay_rows", len(frame) == 2127 * 12)
    check("no_duplicate_book_dates", not frame.duplicated(["date", "asset_code"]).any())
    check("signals_precede_returns_and_fills", (frame.signal_date < frame.date).all() and (trades.signal_date < trades.date).all())
    check("nonnegative_settlement_cash", frame.settlement_cash_yuan.ge(-1e-7).all())
    check("positive_provisional_nav", np.isfinite(frame.provisional_nav).all() and frame.provisional_nav.gt(0).all())
    check("five_bp_actual_two_way_notional", np.allclose(frame.transaction_cost_yuan, frame.traded_notional * .0005))
    check("unknown_current_valuation_invalidates_price_nav", frame.loc[frame.missing_held_count.gt(0), "observed_price_nav"].isna().all())
    check("no_fabricated_certified_total_return_nav", frame.certified_total_return_nav.isna().all())
    check("no_unexplained_observed_nav_holes", frame.loc[frame.missing_held_count.eq(0), "observed_price_nav"].notna().all())
    check("frozen_inventory_positive_quantity", unresolved.quantity.gt(0).all())
    keys = pd.MultiIndex.from_frame(trades[["date", "ts_code"]])
    recorded_prices = close.stack().reindex(keys).to_numpy()
    recorded_volume = vol.stack().reindex(keys).to_numpy()
    check("fills_at_exact_observed_close", np.isfinite(recorded_prices).all() and np.allclose(recorded_prices, trades.execution_close))
    check("all_fills_have_positive_volume_evidence", np.isfinite(recorded_volume).all() and (recorded_volume > 0).all())
    buys = trades.loc[trades.quantity_change.gt(0)]
    eligible = pd.read_parquet(ROOT / "data_processed/cb_masked_research_v51/signal_buy_eligibility.parquet")
    permission = eligible.stack().reindex(pd.MultiIndex.from_frame(buys[["signal_date", "ts_code"]]))
    check("every_buy_has_prior_signal_permission", permission.notna().all() and permission.all())
    check("formal_training_not_silently_enabled", state["training_started"] is False and state["ppo_started"] is False)
    check("gate_refuses_unresolved_execution", state["status"] == "blocked_before_retraining" and state["formal_promotion"] is False)
    for path, expected in state["source_sha256"].items():
        check("source_unchanged:" + path, sha(ROOT / path) == expected)
    pd.DataFrame(checks).to_csv(report / "artifact_checks.csv", index=False)
    result = dict(passed=all(c["passed"] for c in checks), checks=len(checks),
        replay_completed=True, training_allowed=False, formal_promotion=False,
        unresolved_book_security_days=len(unresolved), unresolved_unique_security_days=len(unresolved[["date", "ts_code"]].drop_duplicates()),
        unresolved_security_codes=unresolved.ts_code.nunique())
    save(report / "audit.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit("V52 artifact audit failed; inspect artifact_checks.csv")


if __name__ == "__main__":
    main()

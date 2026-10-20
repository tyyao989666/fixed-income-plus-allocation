"""Replay actual distribution schedules in explicit one-unit diagnostic books."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.v79_fund_entitlement_book import FundEntitlementBook
from src.v53_unified_data import merge_verified_levels

OUT = ROOT / "output/reports/fund_entitlements_v79"
CODE_TO_ASSET = {"511220.SH": "credit_lgfv_aaa", "510300.SH": "equity_broad", "159915.SZ": "equity_chinext",
    "159940.SZ": "equity_fin_property", "510170.SH": "equity_cyclical", "510880.SH": "equity_dividend_utility",
    "518880.SH": "alt_gold"}


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = {Path(__file__), ROOT / "src/v79_fund_entitlement_book.py", ROOT / "src/v53_unified_data.py"}

    def read(name):
        path = ROOT / name
        inputs.add(path)
        return pd.read_parquet(path) if path.suffix == ".parquet" else json.loads(path.read_text(encoding="utf-8"))

    for name, expected in read("output/reports/fund_cashflows_v76/source_hashes.json").items():
        path = ROOT / name
        inputs.add(path)
        assert sha(path) == expected, name
    events = read("output/reports/fund_cashflows_v76/announced_distribution_schedules.parquet")
    raw = read("data_processed/asset_returns_daily.parquet")
    raw.date = pd.to_datetime(raw.date)
    dates = read("data_processed/correct_credit_v74/research_returns_daily_38.parquet").index
    reits = read("data_processed/unified_v53/reits_nine_component_unadjusted_close.parquet")
    prices = {}
    for code, asset in CODE_TO_ASSET.items():
        old = raw.loc[raw.asset_code.eq(asset)].set_index("date").close.sort_index()
        fresh = read(f"data_raw/refresh_20261014_wind_verified/candidates/{asset}.parquet").set_index("date").close.sort_index()
        merged, _ = merge_verified_levels(old, fresh)
        prices[code] = merged.reindex(dates)
    prices.update({code: reits[code].reindex(dates) for code in reits})
    evidence = read("data_raw/corporate_action_v58/reviewed_split_evidence.json")
    assert evidence["verified_in_primary_source"] and evidence["new_units_per_old_unit"] == 10
    split = dict(ts_code="511220.SH", event_id="reviewed_511220_20230116", effective_date="2023-01-16",
        known_at=evidence["original_schedule_publication_date"], new_units_per_old_unit=10,
        primary_source_verified=True)
    paths, findings, blocked = [], [], []
    in_period = events.loc[events.eligible_exdate_research & events.ex_date.between(dates[0], dates[-1])]
    for event in in_period.itertuples(index=False):
        p = prices[event.ts_code]
        before = dates[dates < event.record_date]
        if not len(before):
            blocked.append(dict(event_id=event.event_id, code=event.ts_code, reason="no_opening_pre_record_quote"))
            continue
        start = before[-1]
        if not np.isfinite(p.at[start]) or p.at[start] <= 0:
            blocked.append(dict(event_id=event.event_id, code=event.ts_code, reason="opening_price_missing"))
            continue
        stop = min(pd.Timestamp(event.pay_date) + pd.Timedelta(days=3), dates[-1])
        scenario_days = pd.date_range(start + pd.Timedelta(days=1), stop, freq="D")
        # Daily calendar processing captures record/ex/payment dates even on closures.
        selected = events.loc[events.event_id.eq(event.event_id)].copy()
        relevant_splits = [split] if event.ts_code == "511220.SH" and start < pd.Timestamp(split["effective_date"]) <= stop else []
        book = FundEntitlementBook(event.ts_code, start, 1., p.at[start], selected, splits=relevant_splits)
        series = []
        try:
            for day in scenario_days:
                value = p.get(day, np.nan)
                result = book.step(day, value if np.isfinite(value) and value > 0 else None)
                result.update(event_id=event.event_id, code=event.ts_code, record_date=event.record_date,
                              ex_date=event.ex_date, scheduled_pay_date=event.pay_date,
                              source_schedule_synthetic=False)
                series.append(result)
            frame = pd.DataFrame(series)
            # Without a verified receipt, the planned payment date never creates cash.
            assert frame.cash.eq(0).all()
            after_ex = frame.loc[frame.date.ge(event.ex_date)]
            assert np.allclose(after_ex.receivable, event.cash_per_unit)
            findings.append(dict(event_id=event.event_id, code=event.ts_code, ex_date=event.ex_date,
                scheduled_pay_date=event.pay_date, cash_per_recorded_unit=event.cash_per_unit,
                source_schedule_is_real=True, opening_quantity_is_assumed=True,
                no_unverified_payment_credited=True, receivable_not_reinvested=True,
                missing_domestic_session_marks=int(frame.loc[frame.date.isin(dates), "missing_held_price"].sum()),
                fully_certified_portfolio=False))
            paths.extend(series)
        except ValueError as exc:
            blocked.append(dict(event_id=event.event_id, code=event.ts_code, reason=str(exc)))
    # Separate real split case, with no hypothetical distribution layered on top.
    p = prices["511220.SH"]
    book = FundEntitlementBook("511220.SH", "2023-01-13", 1., p.at[pd.Timestamp("2023-01-13")],
                              events.iloc[:0], splits=[split])
    split_rows = []
    for day in pd.date_range("2023-01-14", "2023-01-18"):
        price = p.get(day, np.nan)
        row = book.step(day, price if np.isfinite(price) else None)
        row["case"] = "verified_1_to_10_quantity_not_price_scaling"
        split_rows.append(row)
    split_frame = pd.DataFrame(split_rows)
    value_on_day = split_frame.loc[split_frame.date.eq(pd.Timestamp("2023-01-16"))].iloc[0]
    assert value_on_day.quantity == 10 and value_on_day.cash == 0
    price_return = value_on_day.observed_mark_nav / p.at[pd.Timestamp("2023-01-13")] - 1
    reference = read("data_processed/correct_credit_v74/research_returns_daily_38.parquet").at[pd.Timestamp("2023-01-16"), "credit_lgfv_aaa"]
    assert np.isclose(price_return, reference, atol=1e-8)
    pd.DataFrame(paths).to_parquet(OUT / "unit_position_event_paths.parquet", index=False)
    pd.DataFrame(findings).to_csv(OUT / "entitlement_conservation_checks.csv", index=False)
    pd.DataFrame(blocked, columns=["event_id", "code", "reason"]).to_csv(OUT / "blocked_cases.csv", index=False)
    split_frame.to_csv(OUT / "verified_split_quantity_path.csv", index=False)
    status = dict(status="unit_entitlement_replays_not_portfolio_certification", in_period_source_events=len(in_period),
        replayed_events=len(findings), blocked_events=len(blocked), actual_receipt_confirmations=0,
        unverified_payment_credit_count=0, verified_split_quantity_replay=True,
        split_value_return_matches_v74=True, assumed_unit_positions_not_actual_portfolio=True,
        raw_prices_and_returns_modified=False, formal_training_allowed=False, ppo_started=False, formal_promotion=False)
    (OUT / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    (OUT / "source_hashes.json").write_text(json.dumps({str(p.relative_to(ROOT)): sha(p) for p in sorted(inputs)}, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2), flush=True)


if __name__ == "__main__":
    main()

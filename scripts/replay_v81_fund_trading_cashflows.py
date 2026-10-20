"""Two-fund accounting rehearsal, not a replacement for the 38-asset strategy."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.v81_fund_portfolio import FundPortfolio
from src.v53_unified_data import merge_verified_levels

OUT = ROOT / "output/reports/fund_trading_v81"
CODES = {"510300.SH": "equity_broad", "510880.SH": "equity_dividend_utility"}


def replay(prices, volumes, sessions, events, *, truncate=None):
    stop = sessions[-1] if truncate is None else pd.Timestamp(truncate)
    days = pd.date_range(sessions[0], stop, freq="D")
    book = FundPortfolio(pd.Index(CODES), days[0], 1_000_000., events,
                         cost_rate=.0005, sessions=sessions)
    month_ends = set(pd.Series(sessions, index=sessions).groupby(sessions.to_period("M")).last())
    observations, trades = [], []
    for day in days[1:]:
        price = prices.loc[day] if day in sessions else pd.Series(np.nan, index=pd.Index(CODES))
        volume = volumes.loc[day] if day in sessions else pd.Series(0., index=pd.Index(CODES))
        target, signal, eligible = None, None, None
        if day in sessions:
            previous = sessions[sessions.get_loc(day) - 1]
            if previous == sessions[0] or previous in month_ends:
                # Predeclared fixed 50/50 target: solely a transaction/accounting test.
                target = pd.Series(.5, index=pd.Index(CODES))
                signal = previous
                eligible = prices.loc[previous].gt(0) & volumes.loc[previous].gt(0)
        result = book.step(day, price, volume, target=target, signal_date=signal, signal_buy_allowed=eligible)
        fills, quantities = result.pop("fills"), result.pop("quantities")
        if day in sessions:
            observations.append(result)
        if target is not None:
            for code in CODES:
                trades.append(dict(date=day, signal_date=signal, code=code, target=.5,
                    fill_quantity=float(fills[code]), end_quantity=float(quantities[code]),
                    execution_price=float(price[code]) if np.isfinite(price[code]) else np.nan,
                    cost=float(abs(fills[code]) * price[code] * .0005) if np.isfinite(price[code]) else 0.,
                    source_eligibility_known_at=signal, scenario_not_actual_portfolio=True))
    return pd.DataFrame(observations), pd.DataFrame(trades)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sources = {Path(__file__), ROOT / "src/v81_fund_portfolio.py", ROOT / "src/v51_cb_policy.py",
               ROOT / "src/v53_unified_data.py"}

    def read(name):
        path = ROOT / name
        sources.add(path)
        return pd.read_parquet(path) if path.suffix == ".parquet" else json.loads(path.read_text(encoding="utf-8"))

    for name, expected in read("output/reports/fund_cashflows_v76/source_hashes.json").items():
        path = ROOT / name
        sources.add(path)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, name
    raw = read("data_processed/asset_returns_daily.parquet")
    raw.date = pd.to_datetime(raw.date)
    sessions = read("data_processed/correct_credit_v74/research_returns_daily_38.parquet").index
    prices, volume = pd.DataFrame(index=sessions), pd.DataFrame(index=sessions)
    for code, asset in CODES.items():
        history = raw.loc[raw.asset_code.eq(asset)].set_index("date").sort_index()
        fresh = read(f"data_raw/refresh_20261014_wind_verified/candidates/{asset}.parquet").set_index("date").sort_index()
        levels, _ = merge_verified_levels(history.close, fresh.close)
        prices[code] = levels.reindex(sessions)
        volume[code] = pd.concat([history.vol.loc[history.index < fresh.index.min()], fresh.vol]).reindex(sessions)
    prices, volume = prices[list(CODES)], volume[list(CODES)]
    events = read("output/reports/fund_cashflows_v76/announced_distribution_schedules.parquet")
    events = events.loc[events.ts_code.isin(CODES) & events.eligible_exdate_research
                       & events.record_date.ge(sessions[0]) & events.ex_date.le(sessions[-1])].copy()
    events.to_parquet(OUT / "input_schedules.parquet", index=False)
    marked, trades = replay(prices, volume, sessions, events)
    control, _ = replay(prices, volume, sessions, events.iloc[:0])
    marked.to_csv(OUT / "strict_receivable_trade_path.csv", index=False)
    trades.to_csv(OUT / "t_plus_one_fills_and_costs.csv", index=False)
    control.to_csv(OUT / "price_only_accounting_control.csv", index=False)
    # Remove future event disclosures and daily observations at a fixed old cutoff.
    cutoff = pd.Timestamp("2022-12-30")
    prefix, prefix_trades = replay(prices.loc[:cutoff], volume.loc[:cutoff], sessions[sessions <= cutoff],
        events.loc[events.known_at <= cutoff], truncate=cutoff)
    past = marked.loc[marked.date <= cutoff].reset_index(drop=True)
    pd.testing.assert_frame_equal(prefix, past, check_exact=False, atol=1e-8, rtol=1e-12)
    pd.testing.assert_frame_equal(prefix_trades, trades.loc[trades.date <= cutoff].reset_index(drop=True),
                                  check_exact=False, atol=1e-8, rtol=1e-12)
    assert np.allclose(trades.cost.sum(), marked.cost.sum(), atol=1e-8)
    assert (trades.signal_date < trades.date).all()
    assert marked.spendable_cash.min() >= -1e-8
    assert marked.settled_event_count.eq(0).all()
    result = dict(status="fund_trade_entitlement_rehearsal_completed_not_formal_backtest",
        codes=list(CODES), predefined_target="50pct_each_for_accounting_test_only", initial_funding=1_000_000.,
        domestic_session_rows=len(marked), fill_instruction_rows=len(trades), distribution_schedules=len(events),
        total_fee_on_gross_traded_notional=float(marked.cost.sum()),
        future_truncation_cutoff="2022-12-30", future_truncation_passed=True,
        unverified_receipt_credit_count=0, actual_receipt_confirmation_count=0,
        no_receivable_financed_buys=True, original_38_asset_strategy_replaced=False,
        fully_certified_total_return=False, formal_training_allowed=False, ppo_started=False, formal_promotion=False,
        limitations=["Two-fund diagnostic, not three products or a selected investment strategy",
                     "All payments remain unconfirmed and receivable; not a faithful realized payout path",
                     "Fractional shares, no round lots, impact, spread or settlement financing model",
                     "No claim that all splits or source cashflows have been certified"])
    (OUT / "status.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (OUT / "source_hashes.json").write_text(json.dumps({str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(sources)}, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

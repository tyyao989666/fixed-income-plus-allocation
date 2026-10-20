"""Create comparable A0/A1 baselines for the 38-asset clean-calendar release.

A0 is hierarchical risk parity: strategic sleeve risk budgets, then inverse
volatility risk parity inside every sleeve.  A1 replays the existing formal
monthly Market View + EP/skfolio anchor with strict next-day execution.  Both
use the identical cleaned return panel, dates, no-cash rule and 5bp cost as the
final iTransformer/PPO portfolio.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.adaptive_daily_ppo import perf_stats
from src.v1_strategy import RISK_SLEEVES, solve_risk_budget
from src.portfolio_accounting import advance_holdings, calendar_decision_targets, accounting_return_panel


FINAL_DATE = pd.Timestamp("2026-07-15")
# First model signal is 2022-01-04; every layer starts earning on its t+1.
START_DATE = pd.Timestamp("2022-01-05")
COST_RATE = 0.0005
LOOKBACK_DAYS = 252
MIN_HISTORY_DAYS = 60
PROFILES = ("low_vol", "balanced", "high_vol")
TRADABLE_PANEL = None


def output(profile: str, name: str) -> Path:
    path = ROOT / f"output/reports/v44_accounting_fixed_{profile}_{name}"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def load_returns() -> tuple[pd.DataFrame, pd.DataFrame]:
    global TRADABLE_PANEL
    wide, modules, tradable, gaps = accounting_return_panel(
        ROOT, "data_processed/hierarchical_asset_itransformer_v43_clean_calendar_dataset.npz",
        "output/reports/hierarchical_asset_itransformer_v43_clean_calendar_metadata.json",
    )
    gaps.to_csv(ROOT / 'output/reports/v44_accounting_fixed_valuation_gaps.csv',index=False,encoding='utf-8-sig')
    TRADABLE_PANEL = tradable
    metadata = modules.rename_axis("asset_code").reset_index()
    return wide, metadata


def config_profiles() -> dict:
    with (ROOT / "config/v1_strategy_config.yaml").open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)["product"]["profiles"]


def formal_rate_split(profile: str) -> pd.Series:
    with (ROOT / "config/hierarchical_asset_itransformer_v43_clean_calendar.yaml").open("r", encoding="utf-8") as handle:
        split = yaml.safe_load(handle)["portfolio"]["formal_anchor_rate_split"][profile]
    values = pd.Series(split, dtype=float)
    if not np.isclose(float(values.sum()), 1.0):
        raise ValueError("formal_anchor_rate_split must sum to one")
    return values


def _sleeve_covariance(history: pd.DataFrame, module_map: pd.Series) -> pd.DataFrame:
    sleeve_returns: dict[str, pd.Series] = {}
    for sleeve in RISK_SLEEVES:
        assets = module_map.index[module_map.eq(sleeve)].tolist()
        sleeve_returns[sleeve] = history.reindex(columns=assets).mean(axis=1, skipna=True)
    values = pd.DataFrame(sleeve_returns)
    covariance = values.cov(min_periods=MIN_HISTORY_DAYS).reindex(index=RISK_SLEEVES, columns=RISK_SLEEVES)
    diagonal = np.nanmedian(np.diag(covariance.to_numpy(dtype=float)))
    floor = max(float(diagonal) if np.isfinite(diagonal) else 1e-6, 1e-8)
    covariance = covariance.fillna(0.0)
    for sleeve in RISK_SLEEVES:
        covariance.loc[sleeve, sleeve] = max(float(covariance.loc[sleeve, sleeve]), floor * 0.1)
    return covariance


def _within_sleeve_weights(history: pd.DataFrame, assets: list[str]) -> pd.Series:
    values = history.reindex(columns=assets)
    volatility = values.std(ddof=1).replace(0.0, np.nan)
    valid = volatility.notna() & values.notna().sum().ge(MIN_HISTORY_DAYS)
    if not bool(valid.any()):
        return pd.Series(1.0 / len(assets), index=assets)
    inverse = (1.0 / volatility[valid]).clip(upper=1e5)
    result = pd.Series(0.0, index=assets)
    result.loc[inverse.index] = inverse / inverse.sum()
    return result


def risk_parity_target(
    history: pd.DataFrame, module_map: pd.Series, profile_cfg: dict,
) -> pd.Series:
    covariance = _sleeve_covariance(history, module_map)
    target_rc = pd.Series(profile_cfg["target_risk_contribution"], dtype=float).reindex(RISK_SLEEVES)
    bounds = {
        sleeve: tuple(float(value) for value in profile_cfg["sleeve_bounds"][sleeve])
        for sleeve in RISK_SLEEVES
    }
    sleeve_weights = pd.Series(
        solve_risk_budget(covariance.to_numpy(dtype=float), target_rc.to_numpy(dtype=float), bounds, {"risk_budget": {"diagonal_vol_floor": 1e-4, "max_iterations": 500, "tolerance": 1e-8}}),
        index=RISK_SLEEVES,
    )
    target = pd.Series(0.0, index=module_map.index)
    for sleeve in RISK_SLEEVES:
        assets = module_map.index[module_map.eq(sleeve)].tolist()
        target.loc[assets] = _within_sleeve_weights(history, assets) * sleeve_weights[sleeve]
    return target / target.sum()


def simulate_monthly_risk_parity(
    returns: pd.DataFrame, history_returns: pd.DataFrame, module_map: pd.Series, profile_cfg: dict,
    tradable: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    weights = risk_parity_target(history_returns.loc[:returns.index[0] - pd.Timedelta(days=1)].tail(LOOKBACK_DAYS), module_map, profile_cfg)
    target_rows: list[dict] = []
    daily_rows: list[dict] = []
    month_end = returns.index.to_series().groupby(returns.index.to_period("M")).max()
    decision_dates = set(month_end.tolist())
    pending = None
    for date, row in returns.iterrows():
        step = advance_holdings(weights, weights if pending is None else pending,
                                0.0 if pending is None else 1.0, row.reindex(weights.index), COST_RATE,
                                tradable.loc[date] if tradable is not None else None)
        daily_rows.append({"date": date, **{key: step[key] for key in ["gross_return", "turnover", "cost", "return"]}})
        weights = pd.Series(step["end_weights"], index=weights.index)
        if not step['unfilled']:
            pending = None
        if date in decision_dates:
            history = history_returns.loc[:date].tail(LOOKBACK_DAYS)
            target = risk_parity_target(history, module_map, profile_cfg)
            pending = target
            for asset, value in target.items():
                target_rows.append({"decision_date": date, "effective_date": None, "asset_code": asset, "module": module_map[asset], "weight": value})
        # Rebalance occurs after the decision-date close. Cost is charged on
        # the next observed trading day via pending_turnover.
    daily = pd.DataFrame(daily_rows)
    daily["nav"] = (1.0 + daily["return"]).cumprod()
    targets = pd.DataFrame(target_rows)
    if not targets.empty:
        targets["effective_date"] = targets["decision_date"].map(
            {date: returns.index[position + 1] if position + 1 < len(returns.index) else pd.NaT for position, date in enumerate(returns.index)}
        )
    return daily, targets


def simulate_formal_anchor(returns: pd.DataFrame, profile: str, tradable: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = pd.read_parquet(ROOT / f"data_processed/v44_{profile}_target_weights_monthly.parquet")
    source["date"] = pd.to_datetime(source["date"], errors="coerce")
    source = source.loc[source["asset_code"].ne("cash")].copy()
    source["weight"] = pd.to_numeric(source["weight"], errors="coerce").fillna(0.0)
    module_map = source.drop_duplicates("asset_code", keep="last").set_index("asset_code")["module"].astype(str)
    wide = source.pivot_table(index="date", columns="asset_code", values="weight", aggfunc="last").sort_index().fillna(0.0)
    split = formal_rate_split(profile)
    old_rate_columns = [column for column in wide.columns if module_map.get(column) == "rates"]
    rate_total = wide[old_rate_columns].sum(axis=1)
    for asset, fraction in split.items():
        wide[asset] = rate_total * float(fraction)
        module_map.loc[asset] = "rates"
    wide = wide.reindex(columns=returns.columns, fill_value=0.0)
    target = calendar_decision_targets(wide, returns.index).ffill().fillna(0.0)
    target = target.div(target.sum(axis=1).replace(0.0, np.nan), axis=0).ffill().fillna(0.0)
    current = target.iloc[0].to_numpy(dtype=float)
    rows = []
    holdings = []
    pending = None
    for position, date in enumerate(returns.index):
        previous = max(position - 1, 0)
        decision = target.iloc[previous].to_numpy(dtype=float)
        changed = position > 1 and not np.allclose(decision, target.iloc[position - 2], atol=1e-12, rtol=0)
        if changed:
            pending = decision
        step = advance_holdings(current, decision if pending is None else pending, float(pending is not None), returns.iloc[position], COST_RATE,
                                tradable.loc[date] if tradable is not None else None)
        if not step['unfilled']:
            pending = None
        rows.append({"date": date, **{key: step[key] for key in ["gross_return", "turnover", "cost", "return"]}})
        holdings.append(step["weights"])
        current = step["end_weights"]
    daily = pd.DataFrame(rows)
    daily["nav"] = (1.0 + daily["return"]).cumprod()
    weights = pd.DataFrame(holdings, index=returns.index, columns=returns.columns).rename_axis(index="date", columns="asset_code").stack().rename("weight").reset_index()
    weights["module"] = weights["asset_code"].map(module_map)
    return daily, weights


def main() -> None:
    all_returns, metadata = load_returns()
    returns = all_returns.loc[START_DATE:FINAL_DATE].copy()
    module_map = metadata.drop_duplicates("asset_code").set_index("asset_code")["module"].reindex(all_returns.columns)
    if len(all_returns.columns) != 38 or module_map.isna().any():
        raise RuntimeError("The A0 baseline requires the cleaned 38-asset master universe.")
    profiles = config_profiles()
    summary: list[dict] = []
    for profile in PROFILES:
        rp_daily, rp_weights = simulate_monthly_risk_parity(returns, all_returns, module_map, profiles[profile], TRADABLE_PANEL)
        anchor_daily, anchor_weights = simulate_formal_anchor(returns, profile, TRADABLE_PANEL)
        for label, daily, weights in [("A0_hierarchical_risk_parity", rp_daily, rp_weights), ("A1_market_view_ep_skfolio", anchor_daily, anchor_weights)]:
            daily.to_csv(output(profile, f"{label}_daily.csv"), index=False, encoding="utf-8-sig")
            weights.to_parquet(output(profile, f"{label}_weights.parquet"), index=False)
            stats = perf_stats(daily["return"])
            summary.append({"profile": profile, "strategy": label, **stats, "mean_daily_turnover": float(daily["turnover"].mean()), "total_cost": float(daily["cost"].sum()), "start": str(daily["date"].min().date()), "end": str(daily["date"].max().date())})
    report = pd.DataFrame(summary)
    report.to_csv(ROOT / "output/reports/v44_accounting_fixed_control_variable_baselines.csv", index=False, encoding="utf-8-sig")
    (ROOT / "output/reports/v44_accounting_fixed_control_variable_baselines.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(report.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

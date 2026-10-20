"""Unify all 38 research tokens through cutoff without claiming tradable P&L."""
from pathlib import Path
import copy
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from src.v45_data import (MODULES, HORIZONS, local_features, observable_grid,
    label_reference_policy, reference_weights, relative_labels, annual_splits)
from src.v49_point_in_time import dividend_cash_per_unit
from src.v53_unified_data import (merge_verified_levels, curve_return, delayed_us_curve,
    validate_relative_dataset, independent_cb_relative_labels, harmonize_activity_units)
from scripts.run_v52_execution_preflight import save, sha
from scripts.acquire_v53_unified_sources import REITS


def main():
    raw = ROOT / "data_raw/unified_v53"
    out = ROOT / "data_processed/unified_v53"
    report = ROOT / "output/reports/unified_v53"
    out.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    save(report / "status.json", dict(status="running", stage="source_assembly"))
    meta_path = ROOT / "output/reports/v45_metadata.json"
    old_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assets, modules = old_meta["assets"], old_meta["asset_modules"]
    cb = pd.read_parquet(ROOT / "data_processed/cb_masked_research_v51/research_bucket_returns_not_executable_nav.parquet")
    dates = cb.index
    if len(assets) != 38 or dates[-1] != pd.Timestamp("2026-10-14"):
        raise ValueError("Unexpected context universe/calendar")
    source = ROOT / "data_processed/asset_returns_daily.parquet"
    old = pd.read_parquet(source)
    old["date"] = pd.to_datetime(old.date)
    if old.duplicated(["date", "asset_code"]).any():
        raise ValueError("Ambiguous historical asset bars")
    cache = ROOT / "data_raw/refresh_20261014_wind_verified"
    registered = pd.read_csv(cache / "coverage_status.csv").set_index("dataset")
    returns = pd.DataFrame(np.nan, index=dates, columns=assets)
    bars = {field: returns.copy() for field in ["open", "high", "low", "close", "vol", "amount"]}
    inputs = [source, meta_path, ROOT / "data_processed/cb_masked_research_v51/research_bucket_returns_not_executable_nav.parquet",
              cache / "coverage_status.csv", Path(__file__), ROOT / "src/v53_unified_data.py",
              ROOT / "src/v45_data.py", ROOT / "src/v49_point_in_time.py"]
    contracts, unit_audit, gap_repairs = [], [], []
    excluded = set(cb.columns) | {"alt_reits", "alt_us_bond", "rate_30y_curve_proxy"}
    for code in assets:
        if code in excluded:
            continue
        path = cache / "candidates" / f"{code}.parquet"
        if registered.at[code, "status"] != "validated_candidate" or not bool(registered.at[code, "passed"]):
            raise ValueError(f"Unverified refresh identity: {code}")
        fresh = pd.read_parquet(path).set_index("date").sort_index()
        history = old.loc[old.asset_code.eq(code)].set_index("date").sort_index()
        levels, audit = merge_verified_levels(history.close, fresh.close)
        if code in {"alt_commodity", "alt_chemical", "alt_nonferrous"}:
            probe_path = raw / f"nanhua_gap_{code}.parquet"
            probe = pd.read_parquet(probe_path)
            probe["date"] = pd.to_datetime(probe.trade_date, format="%Y%m%d")
            expected = {"alt_commodity":"NHCI.NH", "alt_chemical":"NHECI.NH", "alt_nonferrous":"NHNFI.NH"}[code]
            if not probe.ts_code.eq(expected).all() or probe.date.duplicated().any():
                raise ValueError("Nanhua gap evidence identity mismatch")
            quote = probe.set_index("date").close.sort_index()
            paired = pd.concat([levels.rename("old"), quote.rename("quote")], axis=1).dropna()
            if len(paired) < 5 or not np.allclose(paired.old, paired.quote, rtol=5e-5, atol=.0001):
                raise ValueError("Nanhua gap neighboring levels disagree")
            expanded = levels.reindex(levels.index.union(quote.index)).sort_index()
            missing_dates = quote.index[expanded.reindex(quote.index).isna() & quote.gt(0)]
            for missing_day in missing_dates:
                expanded.at[missing_day] = quote.at[missing_day]
                gap_repairs.append(dict(asset_code=code, date=missing_day, close=float(quote.at[missing_day]),
                    evidence_code=expected, neighbor_anchors=len(paired), method="missing_close_only_no_known_price_overwrite"))
            levels = expanded
            inputs += [probe_path, raw / "nanhua_identity.parquet"]
        aligned = levels.reindex(dates)
        returns[code] = aligned.pct_change(fill_method=None)
        for field in bars:
            # Credit/rate index pseudo OHLCV are NOT model inputs.
            if modules[assets.index(code)] in ["equity", "alternatives"]:
                historical_field, fresh_field = history[field], fresh[field]
                if field in ["vol", "amount"]:
                    historical_field, fresh_field, unit = harmonize_activity_units(historical_field, fresh_field, field)
                    unit_audit.append(dict(asset_code=code, **unit))
                merged = pd.concat([historical_field.loc[historical_field.index < fresh.index.min()], fresh_field]).sort_index()
                bars[field][code] = merged.reindex(dates).where(lambda x: x > 0)
        inputs.append(path)
        contracts.append(dict(asset_code=code, construction="same_identifier_overlap_checked_price_level_return",
            currency="CNY_or_index_points", portfolio_certified=False, **audit))
    for code in cb:
        returns[code] = cb[code]
        contracts.append(dict(asset_code=code, construction="complete_history_prior_signal_CB_bucket_price_return",
            currency="CNY", portfolio_certified=False, note="no coupon, no frozen-inventory substitution"))
    curve_path = ROOT / "data_processed/rate_curves_daily_v43.parquet"
    curve = pd.read_parquet(curve_path).set_index("date").treasury_y30
    fresh_curve_path = ROOT / "data_processed/v49_review_candidates/treasury30_source_verified.parquet"
    fresh_curve = pd.read_parquet(fresh_curve_path).set_index("date").yield_percent
    merged, audit = merge_verified_levels(curve.sort_index(), fresh_curve.sort_index(), tolerance=.001)
    # One domestic-session visibility lag: previous local source session only.
    visible_yield = merged.reindex(dates).shift(1)
    returns["rate_30y_curve_proxy"] = curve_return(visible_yield, 18.5)
    contracts.append(dict(asset_code="rate_30y_curve_proxy", construction="lagged_fixed_duration_curve", duration=18.5,
        currency="CNY_curve_proxy", visibility="one_domestic_observation_lag", portfolio_certified=False, **audit))
    inputs += [curve_path, fresh_curve_path]
    us_frames = []
    for year in range(2017, 2027):
        path = raw / f"us_curve_{year}.parquet"
        frame = pd.read_parquet(path)
        us_frames.append(pd.DataFrame(dict(source_date=pd.to_datetime(frame.date, format="%Y%m%d"), yield_percent=frame.y10)))
        inputs.append(path)
    visible_us = delayed_us_curve(pd.concat(us_frames, ignore_index=True), dates)
    visible_us.to_parquet(out / "us_curve_visibility_audit.parquet")
    usd_returns = curve_return(visible_us.yield_percent, 7.5)
    fx_frames = []
    for year in range(2017, 2027):
        path = raw / f"usdcny_parity_{year}.parquet"
        fx_frames.append(pd.read_parquet(path))
        inputs.append(path)
    fx = pd.concat(fx_frames).sort_values("source_date")
    if fx.source_date.duplicated().any() or not fx.cny_per_usd.gt(0).all():
        raise ValueError("Invalid central parity FX history")
    # CFETS central parity is an indicative valuation conversion, NOT a spot
    # execution price. An extra domestic-session lag is deliberately conservative.
    fx["available_date"] = fx.source_date + pd.Timedelta(days=1)
    visible_fx = pd.merge_asof(pd.DataFrame(dict(date=dates)), fx, left_on="date", right_on="available_date",
        direction="backward", tolerance=pd.Timedelta(days=7)).set_index("date")
    fx_return = visible_fx.cny_per_usd.pct_change(fill_method=None)
    returns["alt_us_bond"] = (1 + usd_returns) * (1 + fx_return) - 1
    visible_fx.to_parquet(out / "cny_parity_visibility_audit.parquet")
    contracts.append(dict(asset_code="alt_us_bond", construction="full_history_US_Treasury_10y_duration_7_5",
        currency="CNY_indicative_parity_converted_USD_proxy", visibility="US_two_calendar_days_FX_one_calendar_day",
        portfolio_certified=False, note="unhedged indicative CFETS parity conversion; not executable spot FX or a Treasury ETF"))
    reit_returns, reit_prices, distributions = {}, {}, {}
    for code in REITS:
        frames = []
        for year in range(2021, 2027):
            path = raw / f"reit_{code}_{year}.parquet"
            frames.append(pd.read_parquet(path))
            inputs.append(path)
        history = pd.concat(frames, ignore_index=True)
        if history.empty or not history.ts_code.eq(code).all() or history.trade_date.duplicated().any():
            raise ValueError(f"Invalid REIT identity/history: {code}")
        history["date"] = pd.to_datetime(history.trade_date, format="%Y%m%d")
        price = history.set_index("date").close.reindex(dates)
        path = raw / f"reit_div_{code}.parquet"
        cash = dividend_cash_per_unit(pd.read_parquet(path), dates, code)
        reit_prices[code], distributions[code] = price, cash
        reit_returns[code] = (price + cash) / price.shift(1) - 1
        inputs.append(path)
    reits = pd.DataFrame(reit_returns)
    returns["alt_reits"] = reits.mean(axis=1, skipna=False)
    reits.to_parquet(out / "reits_nine_component_exdate_research_returns.parquet")
    pd.DataFrame(distributions).to_parquet(out / "reits_deduplicated_announced_distributions.parquet")
    pd.DataFrame(reit_prices).to_parquet(out / "reits_nine_component_unadjusted_close.parquet")
    contracts.append(dict(asset_code="alt_reits", construction="first9_daily_equalweight_exdate_distribution_research_index",
        currency="CNY", portfolio_certified=False,
        note="exdate reinvestment research convention, not payment-date self-financing inventory; prelisting missing"))
    returns = returns.where(np.isfinite(returns) & returns.gt(-1))
    valid_ohlc = (bars["high"] >= bars["low"]) & (bars["open"] >= bars["low"]) & (bars["open"] <= bars["high"])
    valid_ohlc &= (bars["close"] >= bars["low"]) & (bars["close"] <= bars["high"])
    for field in ["open", "high", "low", "close"]:
        bars[field] = bars[field].where(valid_ohlc)
    returns.to_parquet(out / "research_returns_daily_38.parquet")
    for field in bars:
        bars[field].to_parquet(out / f"source_{field}.parquet")
    arrays = dict(dates=dates.asi8, returns=np.where(returns.notna(), returns, 0).astype(np.float32),
        return_masks=returns.notna().to_numpy(), module_index=np.asarray([MODULES.index(m) for m in modules]))
    arrays["local_values"], arrays["local_masks"], names = local_features(returns, bars)
    arrays["macro_values"], arrays["macro_masks"], arrays["regime_index"] = observable_grid(returns)
    preferred, policy = label_reference_policy(assets, modules)
    sw = reference_weights(arrays["returns"], arrays["return_masks"], preferred, arrays["module_index"])
    # Keep the SAME label benchmark policy as V45; no change to target for better scores.
    anchor_path = ROOT / "data_processed/v44_balanced_risk_budget_base_monthly.parquet"
    anchor = pd.read_parquet(anchor_path)
    anchor.date = pd.to_datetime(anchor.date)
    if anchor.date.duplicated().any():
        raise ValueError("Duplicate research benchmark anchor dates")
    aligned = pd.merge_asof(pd.DataFrame({"date": dates}), anchor.sort_values("date"), on="date", direction="backward")
    bw = aligned[MODULES].to_numpy(dtype=float)
    known = np.isfinite(bw).all(axis=1)
    bw[~known] = 0
    arrays["sleeve_reference_weights"], arrays["balanced_reference_weights"] = sw, bw
    refs = []
    for year in range(2017, 2027):
        path = raw / f"cb_reference_index_{year}.parquet"
        refs.append(pd.read_parquet(path))
        inputs.append(path)
    cb_index = pd.concat(refs, ignore_index=True)
    if not cb_index.ts_code.eq("000832.CSI").all() or cb_index.trade_date.duplicated().any():
        raise ValueError("Independent CB index identity/date mismatch")
    cb_index["date"] = pd.to_datetime(cb_index.trade_date, format="%Y%m%d")
    cb_index_return = (cb_index.set_index("date").close / cb_index.set_index("date").pre_close - 1).reindex(dates)
    cb_index_return.to_frame("independent_cb_reference_return").to_parquet(out / "cb_reference_returns.parquet")
    arrays.update(independent_cb_relative_labels(arrays["dates"], arrays["returns"], arrays["return_masks"],
        arrays["module_index"], sw, bw, cb_index_return.to_numpy()))
    arrays["sample_indices"] = np.arange(119, len(dates), dtype=np.int64)
    arrays["horizons_days"] = np.asarray(HORIZONS)
    validate_relative_dataset(arrays)
    np.savez_compressed(out / "dataset.npz", **arrays)
    inputs.append(anchor_path)
    metadata = copy.deepcopy(old_meta)
    metadata.update(version="v53_unified_research_NOT_execution_certified", rows=len(dates),
        start=str(dates[0].date()), end=str(dates[-1].date()), samples=len(arrays["sample_indices"]),
        local_features=names, return_policy="reconstructed research conventions per asset_contracts.csv; not certified portfolio total return",
        historical_CB_reconstruction_verified=False, nine_grid_preserved=True,
        source_manifest={str(p.relative_to(ROOT)): dict(path=str(p.relative_to(ROOT)), sha256=sha(p)) for p in inputs},
        array_schema={k: dict(shape=list(v.shape), dtype=str(v.dtype)) for k,v in arrays.items()})
    metadata["normalization"] = dict(window=252, min_periods=126, statistics_shift=1, method="median_IQR", iqr_floor=1e-8)
    metadata["label_policy"]["preferred_weights"] = policy
    metadata["label_policy"].update(version="v53_independent_CB_index_relative_reference",
        cb_reference="000832.CSI independent observed index; replaces ONLY the CB reference, not the 12 model assets",
        change_from_v45="Independent benchmark removes cross-bucket missing-label contagion without reweighting future survivors",
        holding_policy="other sleeve references fixed at signal; CB reference is the independently observed index path")
    metadata["grid_definition"]["source"] = "v53_reconstructed_research_returns"
    metadata["limitations"] = ["CB research price returns are not executable coupon-inclusive bucket returns",
        "REIT equalweight exdate research index is not a payment-date reinvestment ledger",
        "US token converted with lagged CFETS CNY central parity; executable spot/hedge costs not certified",
        "All prior research years have been examined before; not untouched OOS",
        "ETF tokens use unadjusted price returns; dividends are not implied",
        "Credit bucket names remain index/ETF proxies, not exact point-in-time rating universes",
        "Original macro-informed MarketSkill rules are not replaced by the price-only nine-grid"]
    save(report / "metadata.json", metadata)
    pd.DataFrame(contracts).to_csv(report / "asset_contracts.csv", index=False)
    pd.DataFrame(unit_audit).to_csv(report / "volume_amount_unit_audit.csv", index=False)
    pd.DataFrame(gap_repairs).to_csv(report / "nanhua_verified_missing_close_repairs.csv", index=False)
    coverage = []
    for a, code in enumerate(assets):
        observed = returns[code].dropna()
        coverage.append(dict(asset_code=code, calendar_start=str(dates[0].date()), calendar_end=str(dates[-1].date()),
            first_observation=str(observed.index.min().date()) if len(observed) else None,
            last_observation=str(observed.index.max().date()) if len(observed) else None,
            observed_days=len(observed), missing_days=int(returns[code].isna().sum()),
            latest_session_observed=bool(returns[code].notna().iloc[-1]),
            labels_5d=int(arrays["asset_target_mask"][:,0,a].sum()),
            labels_20d=int(arrays["asset_target_mask"][:,1,a].sum()), labels_60d=int(arrays["asset_target_mask"][:,2,a].sum())))
    pd.DataFrame(coverage).to_csv(report / "all_38_assets_coverage.csv", index=False)
    splits = []
    for year in range(2022, 2027):
        split = annual_splits(arrays, year)
        splits.append(dict(year=year, reason=split["reason"], **{key: len(split[key]) for key in ["train","validation","gate","test"]}))
    pd.DataFrame(splits).to_csv(report / "annual_split_readiness.csv", index=False)
    state = dict(status="unified_38_research_dataset_built_not_formal_execution_data", calendar_start=str(dates[0].date()),
        calendar_end=str(dates[-1].date()), rows=len(dates), assets=len(assets), nine_grid=True,
        latest_session_observed_assets=int(returns.iloc[-1].notna().sum()),
        masked_relative_labels_ready=True, models_retrained=False, portfolio_certified=False,
        formal_data_replaced=False, formal_training_allowed=False,
        dataset_sha256=sha(out / "dataset.npz"), metadata_sha256=sha(report / "metadata.json"),
        unresolved_portfolio_requirements=metadata["limitations"])
    save(report / "status.json", state)
    print(json.dumps(state, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

"""Causal, mask-preserving assembly of a common 38-token research calendar."""
import numpy as np
import pandas as pd


def harmonize_activity_units(old, fresh, field):
    """Accept only a stable standard denomination ratio, never fitted scaling.

    Output uses the refresh's units; insufficient identity evidence disables the
    feature instead of injecting a discontinuity or guessing its unit.
    """
    if field not in ["vol", "amount"]:
        raise ValueError("Activity unit conversion only")
    pair = pd.concat([old.rename("old"), fresh.rename("fresh")], axis=1).dropna()
    pair = pair.loc[np.isfinite(pair).all(axis=1) & pair.gt(0).all(axis=1)]
    if len(pair) < 5:
        return pd.Series(np.nan, index=old.index), pd.Series(np.nan, index=fresh.index), dict(
            field=field, available=False, reason="insufficient_overlap_for_activity_units")
    ratios = pair.fresh / pair.old
    factors = [1e-8, 1e-4, .001, .01, .1, 1., 10., 100., 1000., 10000., 1e8]
    valid = [f for f in factors if np.allclose(ratios, f, rtol=1e-5, atol=1e-8)]
    if len(valid) != 1:
        return pd.Series(np.nan, index=old.index), pd.Series(np.nan, index=fresh.index), dict(
            field=field, available=False, reason="nonconstant_or_nonstandard_activity_unit_ratio")
    factor = valid[0]
    return old * factor, fresh.copy(), dict(field=field, available=True,
        historical_to_refresh_unit_factor=factor, overlap_rows=len(pair),
        max_relative_unit_error=float((ratios / factor - 1).abs().max()))


def merge_verified_levels(old, fresh, tolerance=.001, minimum_overlap=5):
    """Splice raw levels only after same-code overlap; never splice CB returns."""
    for f in (old, fresh):
        if f.index.has_duplicates or not f.index.is_monotonic_increasing:
            raise ValueError("Price dates must be ordered and unique")
    pair = pd.concat([old.rename("old"), fresh.rename("fresh")], axis=1).dropna()
    if len(pair) < minimum_overlap or (pair <= 0).any().any():
        raise ValueError("Insufficient positive overlapping levels")
    error = (pair.fresh / pair.old - 1).abs().max()
    if not np.isfinite(error) or error > tolerance:
        raise ValueError("Price level/source basis mismatch")
    # The overlapping vendor refresh owns the interval, including explicit gaps.
    merged = pd.concat([old.loc[old.index < fresh.index.min()], fresh]).sort_index()
    return merged, dict(overlap_rows=len(pair), max_relative_level_error=float(error))


def curve_return(yield_percent, duration, sessions=252):
    """Fixed-duration first-order proxy, no clipping, no bridging missing dates."""
    if duration <= 0 or sessions <= 0:
        raise ValueError("Invalid curve proxy specification")
    y = yield_percent.where(np.isfinite(yield_percent))
    return y.shift(1) / 100 / sessions - duration * y.diff() / 100


def delayed_us_curve(source, domestic_dates, delay_days=2):
    """Deliberate two-calendar-day buffer, never same-date US closing evidence.

    A last published curve may remain observable across holidays; staleness over
    seven calendar days becomes missing. This is a USD curve feature/proxy, NOT
    an RMB-executable Treasury fund or a claim of source vintage certification.
    """
    if delay_days < 2:
        raise ValueError("US curve visibility buffer must be at least two days")
    f = source[["source_date", "yield_percent"]].dropna().sort_values("source_date")
    if f.source_date.duplicated().any():
        raise ValueError("Duplicate US curve dates")
    f["available_date"] = f.source_date + pd.Timedelta(days=delay_days)
    merged = pd.merge_asof(pd.DataFrame({"date": domestic_dates}), f,
        left_on="date", right_on="available_date", direction="backward", tolerance=pd.Timedelta(days=7))
    return merged.set_index("date")


def validate_relative_dataset(arrays, asset_count=38):
    from src.v49_masked_forecast import validate_dataset
    validate_dataset(arrays)
    if arrays["returns"].shape[1] != asset_count:
        raise ValueError("Context asset count changed")
    for key in ["asset_target", "sleeve_target"]:
        if not np.isfinite(arrays[key]).all():
            raise ValueError("Tensor labels must use finite placeholders under false masks")
        if (arrays[key][~arrays[key + "_mask"].astype(bool)] != 0).any():
            raise ValueError("Invalid-label storage placeholder is not zero")


def independent_cb_relative_labels(dates, returns, return_masks, module_index,
                                   sleeve_weights, balanced_weights, cb_reference_returns):
    """Replace ONLY the CB label reference by an independently observed index.

    No extra model token. No ex-post complete-case membership. Every asset still
    requires its OWN future path; other sleeves keep the old fixed-at-t benchmark.
    """
    from src.v45_data import relative_labels, HORIZONS
    result = relative_labels(dates, returns, return_masks, module_index, sleeve_weights, balanced_weights)
    r = np.asarray(returns, dtype=float)
    observed = np.asarray(return_masks, dtype=bool) & np.isfinite(r) & (r > -1)
    ref_r = np.asarray(cb_reference_returns, dtype=float)
    if ref_r.shape != (len(dates),):
        raise ValueError("Independent benchmark must match the full calendar")
    ref_ok = np.isfinite(ref_r) & (ref_r > -1)
    sw, bw = np.asarray(sleeve_weights), np.asarray(balanced_weights)
    modules = np.asarray(module_index)
    cb = modules == 2
    for t in range(len(dates)):
        for j, horizon in enumerate(HORIZONS):
            if t + horizon >= len(dates):
                continue
            future = observed[t+1:t+horizon+1].all(axis=0)
            growth = np.prod(1 + np.where(observed[t+1:t+horizon+1], r[t+1:t+horizon+1], 0), axis=0) - 1
            valid_refs = (sw[t].sum(axis=1) > 0) & ((sw[t] == 0) | future[None, :]).all(axis=1)
            refs = sw[t] @ growth
            valid_refs[2] = ref_ok[t] and ref_ok[t+1:t+horizon+1].all()
            refs[2] = np.prod(1 + np.where(ref_ok[t+1:t+horizon+1], ref_r[t+1:t+horizon+1], 0)) - 1
            valid_assets = observed[t] & future & valid_refs[modules]
            result["asset_target_mask"][t,j] = valid_assets
            result["asset_target"][t,j] = np.where(valid_assets, growth - refs[modules], 0)
            common = bw[t].sum() > 0 and np.all((bw[t] == 0) | valid_refs)
            valid_sleeves = valid_refs & common
            result["sleeve_target_mask"][t,j] = valid_sleeves
            result["sleeve_target"][t,j] = np.where(valid_sleeves, refs - bw[t] @ refs, 0)
    return result

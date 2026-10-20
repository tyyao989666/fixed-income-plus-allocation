"""Shared close-signal/next-session accounting for research portfolios."""
from __future__ import annotations

import numpy as np
import pandas as pd

ACCOUNTING_VERSION = "t1_drift_external_cov_standard_sharpe_v1"


def advance_holdings(current, target, fraction, asset_returns, cost_rate, tradable=None):
    current = np.asarray(current, dtype=float)
    target = np.asarray(target, dtype=float)
    values = np.asarray(asset_returns, dtype=float)
    if not 0 <= fraction <= 1 or cost_rate < 0:
        raise ValueError("Invalid execution fraction or cost rate")
    if not np.isfinite(current).all() or not np.isfinite(target).all():
        raise ValueError("Nonfinite portfolio weights")
    if np.min(current) < -1e-10 or np.min(target) < -1e-10:
        raise ValueError("Negative portfolio weights")
    if not np.isclose(current.sum(), 1) or not np.isclose(target.sum(), 1):
        raise ValueError("Portfolio weights must sum to one")
    traded = current + fraction * (target - current)
    unfilled = False
    if tradable is not None:
        available = np.asarray(tradable, dtype=bool)
        unfilled = bool(np.any((~available) & (np.abs(traded - current) > 1e-10)))
        traded[~available] = current[~available]
        budget = 1.0 - float(traded[~available].sum())
        if available.any():
            desired_sum = float(traded[available].sum())
            if desired_sum > 1e-12:
                traded[available] *= budget / desired_sum
            else:
                traded = current.copy()
    missing = ~np.isfinite(values)
    if np.any(missing & (traded > 1e-10)):
        raise ValueError("Missing execution return for a held asset")
    values = np.where(missing, 0.0, values)
    if np.any(values < -1.0):
        raise ValueError("Asset return below -100%")
    turnover = float(np.abs(traded - current).sum())
    cost = turnover * cost_rate
    gross = float(traded @ values)
    if 1 + gross <= 0 or 1 + gross - cost <= 0:
        raise ValueError("Portfolio exhausted by losses or costs")
    drifted = traded * (1 + values) / (1 + gross)
    return {"weights": traded, "end_weights": drifted, "gross_return": gross,
            "turnover": turnover, "cost": cost, "return": gross - cost, "unfilled": unfilled}


def calendar_decision_targets(source: pd.DataFrame, calendar: pd.DatetimeIndex):
    """Month-end labels refer to the last observed session of that month.

    The returned row at t is a decision at close t, for use on t+1.
    Pre-calendar targets are retained to initialize an already invested book.
    """
    source = source.sort_index().copy()
    positions = calendar.searchsorted(source.index, side="right") - 1
    mapped = [calendar[pos] if pos >= 0 and date == date + pd.offsets.MonthEnd(0) else date
              for date, pos in zip(source.index, positions)]
    source.index = pd.DatetimeIndex(mapped, name="date")
    source = source.groupby(level=0).last()
    left = pd.DataFrame({"date": calendar})
    return pd.merge_asof(left, source.reset_index(), on="date",
                         direction="backward", allow_exact_matches=True).set_index("date")


def frozen_return_panel(root, dataset_path, metadata_path):
    import json
    arrays = np.load(root / dataset_path)
    metadata = json.loads((root / metadata_path).read_text(encoding="utf-8"))
    values = arrays["returns"].astype(float)
    values[~arrays["return_masks"].astype(bool)] = np.nan
    frame = pd.DataFrame(values, index=pd.DatetimeIndex(pd.to_datetime(arrays["dates"]), name="date"),
                         columns=metadata["assets"])
    modules = pd.Series(metadata["asset_modules"], index=metadata["assets"], name="module")
    return frame, modules


def accounting_return_panel(root, dataset_path, metadata_path, max_stale_sessions=5):
    """Bounded last-known valuation, with execution blocked on unquoted days.

    Valid returns (including catch-up moves on the next quote) are unchanged.
    Pre-history and over-age gaps remain NaN and fail if actually held.
    Gap reasons are deliberately not inferred to be exchange holidays.
    """
    frame, modules = frozen_return_panel(root, dataset_path, metadata_path)
    tradable = frame.notna()
    rows = []
    for asset in frame:
        seen = False
        age = 0
        for date in frame.index:
            if pd.notna(frame.at[date,asset]):
                seen, age = True, 0
                continue
            age += 1
            carried = seen and age <= max_stale_sessions
            if carried:
                frame.at[date,asset] = 0.0
            rows.append({"date":date,"asset_code":asset,"stale_sessions":age,
                         "valuation_carried":carried,"tradable":False,
                         "reason":"unquoted_in_frozen_snapshot; holiday_or_gap_not_verified"})
    return frame, modules, tradable, pd.DataFrame(rows)

"""Paper-inspired valuation boundaries, with explicit availability and certification."""
import numpy as np
import pandas as pd


def monthly_boundary(monthly, history_months=120, smoothing_months=12):
    """Extrema are on smoothed observations, including the current known month.

    Monthly input must have dividend_yield and term_spread. Missing months and
    constant ranges remain unknown rather than being imputed as neutral states.
    """
    if history_months < 2 or smoothing_months < 1:
        raise ValueError('Invalid lookbacks')
    if not isinstance(monthly.index, pd.PeriodIndex) or monthly.index.freqstr != 'M':
        raise ValueError('Monthly PeriodIndex required')
    if monthly.index.has_duplicates or not monthly.index.is_monotonic_increasing:
        raise ValueError('Ordered unique months required')
    cols = ['dividend_yield', 'term_spread']
    if not len(monthly):
        raise ValueError('Empty monthly history')
    x = monthly[cols].astype(float).reindex(pd.period_range(monthly.index[0], monthly.index[-1], freq='M'))
    x = x.replace([np.inf, -np.inf], np.nan)
    smoothed = x.rolling(smoothing_months, min_periods=smoothing_months).mean()
    lo = smoothed.rolling(history_months, min_periods=history_months).min()
    hi = smoothed.rolling(history_months, min_periods=history_months).max()
    span = (hi-lo).where((hi-lo) > 1e-12)
    scaled = (2*(smoothed-lo)/span-1).clip(-1, 1)
    out = scaled.rename(columns={c: 'scaled_'+c for c in cols})
    out['boundary'] = scaled.pow(2).sum(axis=1, min_count=2)
    return out


def align_completed_months(monthly, trading_dates, certified=False):
    """Use previous complete calendar month, never the in-progress month.

    Retrospective source data alone do NOT establish historical availability.
    The caller's independent certification controls the strict model mask.
    """
    dates = pd.DatetimeIndex(trading_dates)
    if dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError('Ordered unique trading dates required')
    keys = dates.to_period('M')-1
    aligned = monthly.reindex(keys).copy()
    aligned.index = dates
    aligned['observation_month'] = keys.astype(str)
    finite = np.isfinite(aligned['boundary'].to_numpy())
    aligned['research_available'] = finite
    aligned['strict_available'] = finite & bool(certified)
    return aligned


def boundary_features(aligned, momentum, certified=False):
    """Dimensionless covariates, NOT expected returns, position sizes or rewards."""
    m = np.asarray(momentum, float)
    if m.shape != (len(aligned),):
        raise ValueError('Momentum/calendar mismatch')
    b = aligned['boundary'].to_numpy(float)
    legs = aligned[['scaled_dividend_yield', 'scaled_term_spread']].to_numpy(float)
    values = np.column_stack([legs, b, b*m, b*np.abs(m)])
    research_mask = np.isfinite(values)
    strict_mask = research_mask & bool(certified) & aligned['strict_available'].to_numpy(bool)[:, None]
    return values, research_mask, strict_mask

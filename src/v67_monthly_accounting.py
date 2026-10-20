"""Monthly research accounting: drift, risky-asset traded notional and gaps."""
import numpy as np
import pandas as pd


def monthly_ledger(returns, targets, cash, missing_return=0.0):
    """Targets at close t earn t+1; no same-month signal return.

    Missing held returns are flagged even in an explicit hypothetical scenario.
    Costs are proportional to risky-asset absolute traded notional. Cash is free.
    Drift uses gross returns; fees are charged to cash at the next boundary,
    whose tiny fee-induced drift is not modeled in this monthly diagnostic.
    """
    if not returns.index.equals(targets.index) or not returns.columns.equals(targets.columns):
        raise ValueError('Aligned return/target axes required')
    if not returns.index.equals(cash.index) or cash.isna().any():
        raise ValueError('Known aligned cash returns required')
    w = targets.to_numpy(float)
    if not np.isfinite(w).all() or (w < -1e-12).any() or (w.sum(1) > 1+1e-9).any():
        raise ValueError('Long-only fully funded targets required')
    if missing_return < -1 or not np.isfinite(missing_return):
        raise ValueError('Invalid scenario return')
    r, c = returns.to_numpy(float), cash.to_numpy(float)
    previous = np.zeros(w.shape[1])
    prior_drift = np.zeros_like(previous)
    rows = []
    for t in range(len(w)):
        if t == 0:
            traded = 0.0
        else:
            traded = np.abs(previous-prior_drift).sum()
        missing = (~np.isfinite(r[t])) & (previous > 1e-12)
        filled = np.where(np.isfinite(r[t]), r[t], missing_return)
        growth = previous*(1+filled)
        nav = growth.sum()+(1-previous.sum())*(1+c[t])
        if nav <= 0:
            raise ValueError('Nonpositive scenario NAV')
        prior_drift = growth/nav
        rows.append(dict(gross_return=nav-1, risky_traded_notional=traded,
            unknown_held_weight=float(previous[missing].sum()), unknown_held_names=int(missing.sum()),
            strict_month_known=not missing.any(), start_risky_weight=float(previous.sum())))
        previous = w[t].copy()
    return pd.DataFrame(rows, index=returns.index)

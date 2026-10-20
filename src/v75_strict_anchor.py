"""Causal research sleeve histories and auditable bounded risk budgeting."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize


def project_box(values, lo, hi):
    values, lo, hi = [np.asarray(x, dtype=float) for x in (values, lo, hi)]
    if (values.shape != lo.shape or lo.shape != hi.shape or not np.isfinite([values, lo, hi]).all()
            or (lo < 0).any() or (lo > hi).any() or lo.sum() > 1 + 1e-12 or hi.sum() < 1 - 1e-12):
        raise ValueError("Infeasible bounded simplex")
    left, right = float(np.min(values - hi) - 1), float(np.max(values - lo) + 1)
    for _ in range(100):
        mid = (left + right) / 2
        if np.clip(values - mid, lo, hi).sum() > 1:
            left = mid
        else:
            right = mid
    return np.clip(values - (left + right) / 2, lo, hi)


def risk_shares(weights, covariance):
    w, c = np.asarray(weights), np.asarray(covariance)
    variance = float(w @ c @ w)
    if not np.isfinite(variance) or variance <= 0:
        raise ValueError("Nonpositive portfolio variance")
    return w * (c @ w) / variance


def objective_and_gradient(w, c, target):
    marginal = c @ w
    variance = float(w @ marginal)
    numerator = w * marginal
    error = numerator / variance - target
    jac = (np.diag(marginal) + w[:, None] * c) / variance - np.outer(numerator, 2 * marginal) / variance**2
    return float(error @ error), 2 * jac.T @ error


def solve_budget(covariance, target, lo, hi, max_iterations=500):
    c, target, lo, hi = [np.asarray(x, dtype=float) for x in (covariance, target, lo, hi)]
    if (c.shape != (len(target), len(target)) or not np.isfinite(c).all()
            or not np.allclose(c, c.T, atol=1e-12) or np.linalg.eigvalsh(c).min() <= 0
            or (target <= 0).any() or not np.isclose(target.sum(), 1)):
        raise ValueError("Positive-definite covariance and positive risk budget required")
    c = c / np.trace(c)
    vol = np.sqrt(np.diag(c))
    seeds = [np.sqrt(target) / vol, 1 / vol, np.ones(len(target)), (lo + hi) / 2]
    attempts, candidates = [], []
    for seed in seeds:
        x0 = project_box(seed / seed.sum(), lo, hi)
        result = minimize(lambda w: objective_and_gradient(w, c, target), x0,
            jac=True, method="SLSQP", bounds=list(zip(lo, hi)),
            constraints=[dict(type="eq", fun=lambda w: w.sum() - 1, jac=lambda w: np.ones(len(w)))],
            options=dict(maxiter=max_iterations, ftol=1e-12))
        w = np.asarray(result.x)
        feasible = bool(np.isfinite(w).all() and abs(w.sum() - 1) <= 1e-8
                        and (w >= lo - 1e-8).all() and (w <= hi + 1e-8).all())
        f, gradient = objective_and_gradient(w, c, target)
        stationarity = float(np.linalg.norm(project_box(w - gradient, lo, hi) - w, ord=np.inf))
        accepted = bool(result.success and feasible and stationarity <= 1e-4)
        attempts.append(dict(success=bool(result.success), feasible=feasible, accepted=accepted,
            objective=f, iterations=int(result.nit), projected_gradient_residual=stationarity,
            message=str(result.message)))
        if accepted:
            candidates.append((f, w, stationarity))
    if not candidates:
        raise ValueError("No converged feasible stationary risk-budget solution: " + str(attempts))
    objective, weights, stationarity = min(candidates, key=lambda x: x[0])
    achieved = risk_shares(weights, c)
    return weights, dict(objective=float(objective), max_rc_error=float(np.max(np.abs(achieved - target))),
        projected_gradient_residual=stationarity, achieved_rc=achieved.tolist(), attempts=attempts,
        risk_target_exact=bool(np.max(np.abs(achieved - target)) <= 1e-6),
        solver="scipy_SLSQP_four_fixed_starts", global_optimum_proven=False)


def frozen_monthly_sleeves(daily, preference, cutoff):
    """Fix membership at previous domestic close; missing held paths stay missing.

    These are synthetic buy-and-hold references, not a transaction ledger.
    An unavailable reference remains unavailable through the entire month.
    """
    if not daily.index.is_monotonic_increasing or daily.index.has_duplicates:
        raise ValueError("Ordered unique daily calendar required")
    rows, memberships = [], []
    periods = daily.index.to_period("M")
    for month in periods.unique():
        if month.end_time.normalize() > pd.Timestamp(cutoff):
            continue
        positions = np.flatnonzero(periods == month)
        previous = positions[0] - 1
        for sleeve, configured in preference.items():
            if not set(configured).issubset(daily.columns):
                raise ValueError("Unknown preferred asset")
            w = pd.Series(configured, dtype=float)
            if (w < 0).any() or w.sum() <= 0:
                raise ValueError("Invalid preferred asset weights")
            if previous < 0:
                rows.append(dict(date=month.end_time.normalize(), sleeve=sleeve, return_1m=np.nan,
                                 status="no_previous_signal_snapshot"))
                continue
            signal = daily.index[previous]
            available = np.isfinite(daily.iloc[previous].reindex(w.index))
            w = w.where(available, 0)
            if w.sum() <= 0:
                rows.append(dict(date=month.end_time.normalize(), sleeve=sleeve, return_1m=np.nan,
                                 status="no_available_preferred_assets_at_signal"))
                continue
            w = w[w > 0] / w.sum()
            future = daily.iloc[positions][w.index]
            complete = bool(np.isfinite(future.to_numpy()).all() and future.gt(-1).all().all())
            value = float(((1 + future).prod() - 1) @ w) if complete else np.nan
            rows.append(dict(date=month.end_time.normalize(), sleeve=sleeve, return_1m=value,
                             status="complete_fixed_membership_path" if complete else "missing_held_path"))
            memberships.extend(dict(month=month.end_time.normalize(), signal_date=signal, sleeve=sleeve,
                                    asset=code, weight=float(weight)) for code, weight in w.items())
    return pd.DataFrame(rows), pd.DataFrame(memberships)


def covariance_complete(window, shrinkage=.2, floor=.002):
    complete = window.dropna(how="any")
    if len(complete) < 2:
        raise ValueError("Insufficient complete observations")
    covariance = complete.cov().to_numpy()
    diagonal = np.diag(np.maximum(np.diag(covariance), floor**2))
    covariance = (1 - shrinkage) * covariance + shrinkage * diagonal + np.eye(len(diagonal)) * floor**2
    return (covariance + covariance.T) / 2, len(complete)

"""Strict, isolated v45 portfolio solver. No silent constraint relaxation."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

import cvxpy as cp
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from skfolio.measures import RiskMeasure
from skfolio.optimization import MeanRisk, ObjectiveFunction
from skfolio.prior import EmpiricalPrior

from src.ep_skfolio_optimizer import RISK_SLEEVES, SuppliedCovariance


CONTRACT_VERSION = "v45_total_l1_strict_v1"
FEASIBILITY_TOL = 1e-6
BUDGET_TOL = 1e-8


class SolverContractError(ValueError):
    def __init__(self, status: str, detail: dict[str, Any]):
        self.status = status
        self.detail = {"status": status, **detail}
        super().__init__(f"{status}: {detail}")


@dataclass
class SolverResult:
    weights: pd.Series
    diagnostics: dict[str, Any]


def vector(value: pd.Series, names: list[str], label: str) -> np.ndarray:
    if not value.index.is_unique or set(value.index) != set(names):
        raise SolverContractError("invalid_input", {"field": label, "reason": "labels"})
    out = value.reindex(names).to_numpy(dtype=float)
    if not np.isfinite(out).all():
        raise SolverContractError("invalid_input", {"field": label, "reason": "nonfinite"})
    return out


def validate_box(lo: np.ndarray, hi: np.ndarray, budget: float = 1.0) -> None:
    if (not np.isfinite(lo).all() or not np.isfinite(hi).all()
            or np.any(lo < 0) or np.any(lo > hi) or not np.isfinite(budget)
            or budget < 0 or lo.sum() > budget + 1e-10 or hi.sum() < budget - 1e-10):
        raise SolverContractError("invalid_input", {
            "field": "bounds", "lower_sum": float(lo.sum()),
            "upper_sum": float(hi.sum()), "budget": float(budget)})


def bounded_projection(values, lo, hi, budget: float = 1.0) -> np.ndarray:
    values, lo, hi = [np.asarray(a, dtype=float) for a in (values, lo, hi)]
    if values.shape != lo.shape or lo.shape != hi.shape or not np.isfinite(values).all():
        raise SolverContractError("invalid_input", {"field": "projection"})
    validate_box(lo, hi, budget)
    left, right = float(np.min(values - hi) - 1), float(np.max(values - lo) + 1)
    for _ in range(100):
        mid = (left + right) / 2
        weights = np.clip(values - mid, lo, hi)
        if weights.sum() > budget:
            left = mid
        else:
            right = mid
    weights = np.clip(values - (left + right) / 2, lo, hi)
    if abs(weights.sum() - budget) > BUDGET_TOL:
        raise SolverContractError("blocked", {"reason": "projection_budget"})
    return weights


def residuals(w, base, previous, lo, hi, active_cap, turnover_cap):
    return {
        "budget_residual": float(abs(w.sum() - 1)),
        "bounds_residual": float(max(0, np.max(lo - w), np.max(w - hi))),
        "active_residual": float(max(0, np.abs(w - base).sum() - active_cap)),
        "turnover_residual": float(max(0, np.abs(w - previous).sum() - turnover_cap)),
    }


def valid(w, base, previous, lo, hi, active_cap, turnover_cap):
    if w is None or not np.isfinite(w).all():
        return False
    r = residuals(w, base, previous, lo, hi, active_cap, turnover_cap)
    return r["budget_residual"] <= BUDGET_TOL and all(
        v <= FEASIBILITY_TOL for k, v in r.items() if k != "budget_residual")


def minimum_turnover(base, previous, lo, hi, active_cap):
    """Exact minimum L1 change under all other hard constraints."""
    # Linear epigraph avoids conic accuracy warnings at exact box boundaries.
    n = len(base)
    eye, zero = np.eye(n), np.zeros((n,n))
    inequalities = np.vstack([
        np.hstack([eye,-eye,zero]), np.hstack([-eye,-eye,zero]),
        np.hstack([eye,zero,-eye]), np.hstack([-eye,zero,-eye]),
        np.r_[np.zeros(2*n),np.ones(n)][None,:]])
    rhs = np.r_[previous,-previous,base,-base,active_cap]
    result = linprog(np.r_[np.zeros(n),np.ones(n),np.zeros(n)],
        A_ub=inequalities,b_ub=rhs,A_eq=np.r_[np.ones(n),np.zeros(2*n)][None,:],
        b_eq=[1.],bounds=list(zip(lo,hi))+[(0,None)]*(2*n),method='highs',
        options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
    if not result.success:
        status = 'infeasible' if result.status == 2 else 'blocked'
        raise SolverContractError(status, {
            "reason": "box_and_anchor_conflict", "solver_status": result.message})
    candidate = np.asarray(result.x[:n])
    if not valid(candidate, base, previous, lo, hi, active_cap, np.inf):
        raise SolverContractError("blocked", {"reason": "invalid_feasibility_solution"})
    return float(np.abs(candidate - previous).sum()), candidate


def causal_anchor(raw: pd.Series, previous: pd.Series, bounds: dict,
                  active_cap: float, target_change_cap: float) -> SolverResult:
    """Closest feasible anchor using current/past inputs, never future returns.

    A jointly feasible target is an existence witness, not the utility solution.
    The regular solver subsequently optimizes under the unchanged hard caps.
    """
    names = RISK_SLEEVES
    b, p = vector(raw,names,'raw_anchor'), vector(previous,names,'previous_target')
    if min(b.min(),p.min()) < -1e-10 or max(abs(b.sum()-1),abs(p.sum()-1)) > BUDGET_TOL:
        raise SolverContractError('invalid_input',{'field':'anchor_budget'})
    if not np.isfinite([active_cap,target_change_cap]).all() or min(active_cap,target_change_cap) < 0:
        raise SolverContractError('invalid_input',{'field':'anchor_policy_limits'})
    lo=np.array([bounds[s][0] for s in names],dtype=float)
    hi=np.array([bounds[s][1] for s in names],dtype=float)
    validate_box(lo,hi)
    try:
        needed,_=minimum_turnover(b,p,lo,hi,active_cap)
    except SolverContractError as exc:
        if exc.status != 'infeasible':
            raise
        # New eligibility bounds can invalidate the raw anchor itself.
        needed=None
    if needed is not None and needed <= target_change_cap + 1e-8:
        return SolverResult(raw.reindex(names).copy(),{
            'anchor_adjusted':False,'anchor_adjustment_l1':0.,'raw_minimum_required_l1':needed})
    anchor, witness = cp.Variable(len(names)),cp.Variable(len(names))
    problem=cp.Problem(cp.Minimize(cp.sum_squares(anchor-b)),[
        cp.sum(anchor)==1,anchor>=lo,anchor<=hi,
        cp.sum(witness)==1,witness>=lo,witness<=hi,
        cp.norm1(witness-anchor)<=active_cap,
        cp.norm1(witness-p)<=target_change_cap])
    problem.solve(solver='CLARABEL',tol_gap_abs=1e-11,tol_feas=1e-11,tol_gap_rel=1e-11)
    if problem.status != 'optimal' or anchor.value is None:
        raise SolverContractError('infeasible',{'reason':'causal_anchor_policy_infeasible',
            'solver_status':problem.status})
    effective=np.asarray(anchor.value)
    if (not valid(np.asarray(witness.value),effective,p,lo,hi,active_cap,target_change_cap)
            or abs(effective.sum()-1)>BUDGET_TOL
            or np.any(effective<lo-1e-8) or np.any(effective>hi+1e-8)):
        raise SolverContractError('blocked',{'reason':'causal_anchor_invalid_solution'})
    repaired_needed,_=minimum_turnover(effective,p,lo,hi,active_cap)
    if repaired_needed > target_change_cap + FEASIBILITY_TOL:
        raise SolverContractError('blocked',{'reason':'causal_anchor_failed_exact_recheck'})
    return SolverResult(pd.Series(effective,index=names),{
        'anchor_adjusted':True,'anchor_adjustment_l1':float(np.abs(effective-b).sum()),
        'anchor_adjustment_l2':float(np.linalg.norm(effective-b)),
        'raw_minimum_required_l1':needed,'effective_minimum_required_l1':repaired_needed,
        'anchor_policy':'closest_feasible_anchor_joint_projection_v1'})


def solve(base: pd.Series, mu: pd.Series, covariance: pd.DataFrame,
          previous: pd.Series | None, bounds: dict, history: pd.DataFrame | None,
          options: dict, *, backend: str = "skfolio") -> SolverResult:
    names = RISK_SLEEVES
    b = vector(base, names, "anchor")
    p = b.copy() if previous is None else vector(previous, names, "previous_target")
    m = vector(mu, names, "mu")
    for label, a in [("anchor", b), ("previous_target", p)]:
        if np.any(a < -1e-10) or abs(a.sum() - 1) > BUDGET_TOL:
            raise SolverContractError("invalid_input", {"field": label, "reason": "budget"})
    if set(bounds) != set(names):
        raise SolverContractError("invalid_input", {"field": "bounds_labels"})
    lo = np.array([bounds[s][0] for s in names], dtype=float)
    hi = np.array([bounds[s][1] for s in names], dtype=float)
    validate_box(lo, hi)
    if (not covariance.index.is_unique or not covariance.columns.is_unique
            or set(covariance.index) != set(names) or set(covariance.columns) != set(names)):
        raise SolverContractError("invalid_input", {"field": "covariance_labels"})
    sigma = covariance.reindex(index=names, columns=names).to_numpy(dtype=float)
    if not np.isfinite(sigma).all() or not np.allclose(sigma, sigma.T, atol=1e-12, rtol=1e-8):
        raise SolverContractError("invalid_input", {"field": "covariance_values"})
    if np.linalg.eigvalsh(sigma).min() <= 0:
        raise SolverContractError("invalid_input", {"field": "covariance_not_pd"})
    # The caller owns PSD repair/jitter; never alter it differently by backend.
    a = float(options.get("max_active_deviation", .06))
    t = float(options.get("max_sleeve_turnover", .22))
    risk = float(options.get("risk_aversion", 3))
    bp = float(options.get("base_deviation_penalty", 6))
    tp = float(options.get("turnover_penalty", .45))
    if not np.isfinite([a, t, risk, bp, tp]).all() or min(a, t, risk, bp, tp) < 0:
        raise SolverContractError("invalid_input", {"field": "options"})
    required, feasible = minimum_turnover(b, p, lo, hi, a)
    if required > t + FEASIBILITY_TOL:
        raise SolverContractError("infeasible", {
            "reason": "total_target_change_conflict", "minimum_required_l1": required,
            "configured_l1_limit": t, "active_limit": a,
            "anchor_previous_l1": float(np.abs(b - p).sum()),
            "anchor": b.tolist(), "previous_target": p.tolist()})

    def objective(w):
        return float(risk * (w @ sigma @ w) - m @ w
                     + bp * np.square(w - b).sum() + tp * np.abs(w - p).sum())

    errors = []
    w_out, solver_name, actual_backend = None, None, None
    if backend not in {"skfolio", "cvxpy"}:
        raise SolverContractError("invalid_input", {"field": "backend"})
    if backend == "skfolio":
        if history is None or len(history) < 2 or not history.columns.is_unique or set(history.columns) != set(names):
            raise SolverContractError('invalid_input', {'field':'history_labels_or_length'})
        x = history.reindex(columns=names).astype(float)
        if not np.isfinite(x.to_numpy()).all():
            raise SolverContractError('invalid_input', {'field':'history_nonfinite'})
        try:
            model = MeanRisk(
                prior_estimator=EmpiricalPrior(covariance_estimator=SuppliedCovariance(sigma)),
                objective_function=ObjectiveFunction.MAXIMIZE_UTILITY,
                risk_measure=RiskMeasure.VARIANCE, risk_aversion=risk,
                min_weights=lo, max_weights=hi, budget=1,
                overwrite_expected_return=lambda w: m @ w,
                add_objective=lambda w: -bp * cp.sum_squares(w-b) - tp * cp.norm1(w-p),
                add_constraints=lambda w: [cp.norm1(w-b) <= a, cp.norm1(w-p) <= t],
                solver="CLARABEL", solver_params={"tol_gap_abs": 1e-10,
                    "tol_feas": 1e-10, "tol_gap_rel": 1e-10}, raise_on_failure=True)
            model.fit(x)
            candidate = np.asarray(model.weights_, dtype=float)
            if not valid(candidate, b, p, lo, hi, a, t):
                raise ValueError("skfolio final constraints failed")
            w_out, solver_name, actual_backend = candidate, "CLARABEL", "skfolio"
        except Exception as exc:
            errors.append(f"skfolio: {type(exc).__name__}: {exc}")

    if w_out is None:
        # Independent formulation, identical objective and feasible set.
        w = cp.Variable(len(names))
        problem = cp.Problem(cp.Minimize(risk * cp.quad_form(w, sigma) - m @ w
                + bp * cp.sum_squares(w-b) + tp * cp.norm1(w-p)),
                [cp.sum(w)==1, w>=lo, w<=hi, cp.norm1(w-b)<=a, cp.norm1(w-p)<=t])
        for solver in ["CLARABEL", "OSQP"]:
            if solver not in cp.installed_solvers():
                continue
            try:
                params = ({"tol_gap_abs": 1e-10, "tol_feas": 1e-10, "tol_gap_rel": 1e-10}
                          if solver == "CLARABEL" else {"eps_abs": 1e-9, "eps_rel": 1e-9, "max_iter": 100000})
                problem.solve(solver=solver, **params)
                if problem.status != "optimal" or not valid(w.value, b, p, lo, hi, a, t):
                    raise ValueError(f"Invalid solution: {problem.status}")
                w_out, solver_name, actual_backend = np.asarray(w.value), solver, "cvxpy"
                break
            except Exception as exc:
                errors.append(f"{solver}: {type(exc).__name__}: {exc}")
    status = "optimal_validated" if not errors else "fallback_validated"
    if w_out is None:
        if valid(p, b, p, lo, hi, a, t):
            w_out, status, actual_backend = p.copy(), "hold_validated", "hold"
        else:
            raise SolverContractError("blocked", {"reason": "all_solvers_failed", "errors": errors})
    return SolverResult(pd.Series(w_out, index=names), {
        "contract": CONTRACT_VERSION, "status": status, "backend": actual_backend,
        "solver": solver_name, "errors": errors, "objective": objective(w_out),
        "minimum_required_l1": required, "configured_l1_limit": t,
        "target_change_l1": float(np.abs(w_out-p).sum()),
        "active_l1": float(np.abs(w_out-b).sum()),
        "covariance_sha256": hashlib.sha256(sigma.tobytes()).hexdigest(),
        **residuals(w_out, b, p, lo, hi, a, t)})


def allocate_assets(base: pd.Series, score: pd.Series, uncertainty: pd.Series,
                    sleeve_weight: float, strength: float, max_share: float,
                    max_active: float) -> pd.Series:
    if not base.index.is_unique or not np.isfinite(base).all() or (base < 0).any():
        raise SolverContractError("invalid_input", {"field": "asset_anchor"})
    if not np.isfinite([sleeve_weight, strength, max_share, max_active]).all() or sleeve_weight < 0 or not 0 < max_share <= 1 or max_active < 0:
        raise SolverContractError("invalid_input", {"field": "asset_options"})
    if sleeve_weight == 0:
        return base * 0
    if base.sum() <= 0:
        raise SolverContractError("infeasible", {"reason": "empty_asset_sleeve"})
    s, u = score.reindex(base.index), uncertainty.reindex(base.index)
    if not np.isfinite(s).all() or not np.isfinite(u).all() or (u < 0).any():
        raise SolverContractError("invalid_input", {"field": "asset_views"})
    shares = base / base.sum()
    eligible = shares > 1e-12
    centered = s - np.average(s[eligible], weights=shares[eligible])
    tilt = shares * np.exp(np.clip(strength * centered / (1+u), -.75, .75))
    desired = tilt / tilt.sum() * sleeve_weight
    lo = np.maximum(shares.to_numpy()*sleeve_weight - max_active, 0)
    hi = np.minimum(shares.to_numpy()*sleeve_weight + max_active, max_share*sleeve_weight)
    hi[~eligible.to_numpy()] = 0
    out = bounded_projection(desired, lo, hi, sleeve_weight)
    return pd.Series(out, index=base.index)


def confidence_view_tilt(pairs: pd.DataFrame, scale: float) -> pd.Series:
    if not np.isfinite(scale) or scale < 0:
        raise SolverContractError("invalid_input", {"field": "view_scale"})
    out = pd.Series(0., index=RISK_SLEEVES)
    if pairs.empty:
        return out
    work = pairs[["source", "target", "confidence"]].copy()
    c = work.confidence.to_numpy(dtype=float)
    if not np.isfinite(c).all() or np.any(c < 0) or np.any(c > 1):
        raise SolverContractError("invalid_input", {"field": "view_confidence"})
    if not work.source.isin(out.index).all() or not work.target.isin(out.index).all():
        raise SolverContractError("invalid_input", {"field": "view_pair_labels"})
    work = work.loc[work.source != work.target].groupby(["source", "target"], as_index=False).confidence.max()
    if work.empty:
        return out
    for row in work.itertuples():
        out[row.source] += row.confidence
        out[row.target] -= row.confidence
    out -= out.mean()
    norm = out.abs().max()
    return out * 0 if norm <= 1e-12 else out / norm * scale * work.confidence.mean()

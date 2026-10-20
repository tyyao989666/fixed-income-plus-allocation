"""Positive-definite unconstrained risk-budget reference via a convex program."""
import cvxpy as cp
import numpy as np
from scipy.optimize import minimize

from src.v75_strict_anchor import risk_shares


def exact_positive_budget(covariance, budget):
    c, b = np.asarray(covariance, dtype=float), np.asarray(budget, dtype=float)
    if (c.shape != (len(b), len(b)) or not np.isfinite(c).all() or not np.isfinite(b).all()
            or not np.allclose(c, c.T, atol=1e-12) or np.linalg.eigvalsh(c).min() <= 0
            or (b <= 0).any() or not np.isclose(b.sum(), 1)):
        raise ValueError("SPD covariance and strictly positive unit risk budget required")
    scaled = c / np.trace(c)
    x = cp.Variable(len(b), pos=True)
    problem = cp.Problem(cp.Minimize(.5 * cp.quad_form(x, scaled) - cp.sum(cp.multiply(b, cp.log(x)))))
    problem.solve(solver="CLARABEL", tol_gap_abs=1e-11, tol_feas=1e-11, tol_gap_rel=1e-11, max_iter=500)
    primary_status = problem.status
    backend = "CVXPY_CLARABEL_convex_log_budget"
    if problem.status == "optimal" and x.value is not None:
        raw = np.asarray(x.value, dtype=float)
    else:
        # Solve the identical strictly convex objective, never accept 'inaccurate'.
        def objective(value):
            return (.5 * value @ scaled @ value - b @ np.log(value),
                    scaled @ value - b / value)
        alternative = minimize(objective, np.sqrt(b / np.diag(scaled)), jac=True,
            method="L-BFGS-B", bounds=[(1e-12, None)] * len(b),
            options=dict(maxiter=2000, ftol=1e-15, gtol=1e-11, maxls=50))
        if not alternative.success:
            raise ValueError("Both exact-budget solvers failed strict convergence")
        raw = np.asarray(alternative.x, dtype=float)
        backend = "scipy_LBFGSB_same_convex_objective_after_rejected_CLARABEL"
    if not np.isfinite(raw).all() or (raw <= 0).any():
        raise ValueError("Nonpositive convex solution")
    weights = raw / raw.sum()
    residual = float(np.max(np.abs(risk_shares(weights, c) - b)))
    if residual > 1e-6:
        raise ValueError(f"Convex risk-budget contribution residual too large: {residual}")
    return weights, dict(status="validated_convex_solution", primary_status=primary_status,
                         max_rc_error=residual, solver=backend, positive_budget_unique_solution=True)

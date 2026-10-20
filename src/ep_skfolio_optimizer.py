from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.rienet_covariance import estimate_covariance

try:
    import cvxpy as cp
    from skfolio.measures import RiskMeasure
    from skfolio.optimization import MeanRisk, ObjectiveFunction
    from skfolio.moments import BaseCovariance
    from skfolio.prior import EmpiricalPrior

    SKFOLIO_AVAILABLE = True
except Exception:
    cp = None
    MeanRisk = None
    ObjectiveFunction = None
    RiskMeasure = None
    SKFOLIO_AVAILABLE = False


if SKFOLIO_AVAILABLE:
    class SuppliedCovariance(BaseCovariance):
        """Use the point-in-time risk matrix in the optimizer's prior."""
        def __init__(self, covariance):
            super().__init__()
            self.covariance = covariance

        def fit(self, X, y=None, **fit_params):
            values = np.asarray(X, dtype=float)
            matrix = np.asarray(self.covariance, dtype=float)
            if matrix.shape != (values.shape[1], values.shape[1]):
                raise ValueError("Supplied covariance dimension mismatch")
            self.n_features_in_ = values.shape[1]
            self.location_ = np.mean(values, axis=0)
            self.covariance_ = matrix.copy()
            return self


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data_processed"
REPORTS = ROOT / "output" / "reports"
RISK_SLEEVES = ["rates", "credit", "convertible_bonds", "equity", "alternatives"]


@dataclass(frozen=True)
class EPSkfolioResult:
    weights: pd.Series
    posterior_mu: pd.Series
    diagnostics: dict[str, Any]


def project_simplex_bounds(values: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    w = np.clip(np.asarray(values, dtype=float), lo, hi)
    for _ in range(100):
        diff = 1.0 - float(w.sum())
        if abs(diff) < 1e-12:
            break
        if diff > 0:
            room = hi - w
            active = room > 1e-12
        else:
            room = w - lo
            active = room > 1e-12
        if not active.any():
            break
        w[active] += diff * room[active] / float(room[active].sum())
        w = np.clip(w, lo, hi)
    return w / float(w.sum())


def view_to_sleeve(asset: str, cfg: dict[str, Any]) -> str:
    return cfg.get("asset_mapping", {}).get(asset, {}).get("module", asset)


def effective_ep_config(cfg: dict[str, Any]) -> dict[str, Any]:
    ep_cfg = dict(cfg.get("ep_skfolio", {}))
    profile = str(cfg.get("product", {}).get("selected_profile", ""))
    profile_cfg = cfg.get("product", {}).get("profiles", {}).get(profile, {})
    overrides = profile_cfg.get("ep_skfolio", {})
    if overrides:
        ep_cfg.update(dict(overrides))
    return ep_cfg


def build_prior_mu(history: pd.DataFrame, cfg: dict[str, Any]) -> pd.Series:
    ep_cfg = effective_ep_config(cfg)
    lookback = int(ep_cfg.get("mu_lookback_months", 12))
    shrink = float(ep_cfg.get("prior_mu_shrinkage", 0.65))
    hist = history.tail(lookback).reindex(columns=RISK_SLEEVES).fillna(0.0)
    if hist.empty:
        return pd.Series(0.0, index=RISK_SLEEVES)
    realized = hist.mean()
    long_run = history.reindex(columns=RISK_SLEEVES).fillna(0.0).expanding().mean().iloc[-1]
    mu = (1.0 - shrink) * realized + shrink * long_run
    return mu.reindex(RISK_SLEEVES).fillna(0.0)


def posterior_from_views(
    prior_mu: pd.Series,
    views: pd.DataFrame,
    pred: pd.DataFrame,
    trend_scores: pd.Series | None,
    date: pd.Timestamp,
    cfg: dict[str, Any],
) -> tuple[pd.Series, list[dict[str, Any]]]:
    ep_cfg = effective_ep_config(cfg)
    mu = prior_mu.reindex(RISK_SLEEVES).astype(float).fillna(0.0).copy()
    rows: list[dict[str, Any]] = []
    view_scale = float(ep_cfg.get("view_mu_scale", 0.004))
    view_mode = str(ep_cfg.get("view_mode", "mean_rank")).lower()
    view_rank_center = bool(ep_cfg.get("view_rank_center", True))
    view_pair_scales = ep_cfg.get("view_pair_scales", {}) or {}
    pred_scale = float(ep_cfg.get("itransformer_mu_scale", 0.35))
    trend_scale = float(ep_cfg.get("trend_mu_scale", 0.18))
    max_abs_tilt = float(ep_cfg.get("max_abs_mu_tilt", 0.010))

    if views is not None and not views.empty:
        active = views[views["month"].eq(pd.Timestamp(date))].copy()
        if view_mode in {"mean_rank", "mean_views_rank", "rank"}:
            view_score = pd.Series(0.0, index=RISK_SLEEVES)
            raw_pairs: list[tuple[str, str, float]] = []
            for _, view in active.iterrows():
                source = view_to_sleeve(str(view["source_asset"]), cfg)
                target = view_to_sleeve(str(view["target_asset"]), cfg)
                if source not in view_score.index or target not in view_score.index or source == target:
                    continue
                confidence = max(float(view.get("confidence", 0.0)), 0.0)
                pair_key = f"{source}>{target}"
                confidence *= float(view_pair_scales.get(pair_key, 1.0))
                view_score[source] += confidence
                view_score[target] -= confidence
                raw_pairs.append((source, target, confidence))

            if raw_pairs:
                centered_score = view_score - float(view_score.mean()) if view_rank_center else view_score
                denom = float(centered_score.abs().max())
                rank_signal = centered_score / denom if denom > 1e-12 else centered_score * 0.0
                tilt = (rank_signal * view_scale).clip(lower=-max_abs_tilt, upper=max_abs_tilt)
                mu = mu.add(tilt, fill_value=0.0)
                for source, target, confidence in raw_pairs:
                    rows.append(
                        {
                            "date": date,
                            "signal_type": "market_skill_mean_view_raw_pair",
                            "source": source,
                            "target": target,
                            "confidence": confidence,
                            "mu_tilt": 0.0,
                            "view_mode": view_mode,
                        }
                    )
                for sleeve, value in tilt.items():
                    if abs(float(value)) > 1e-12:
                        rows.append(
                            {
                                "date": date,
                                "signal_type": "market_skill_mean_view_rank",
                                "source": sleeve,
                                "target": "",
                                "confidence": float(abs(rank_signal.loc[sleeve])),
                                "mu_tilt": float(value),
                                "view_score": float(view_score.loc[sleeve]),
                                "view_rank_signal": float(rank_signal.loc[sleeve]),
                                "view_mode": view_mode,
                            }
                        )
        elif view_mode in {"pairwise", "pairwise_mean_tilt"}:
            for _, view in active.iterrows():
                source = view_to_sleeve(str(view["source_asset"]), cfg)
                target = view_to_sleeve(str(view["target_asset"]), cfg)
                if source not in mu.index or target not in mu.index or source == target:
                    continue
                confidence = float(view.get("confidence", 0.0))
                amount = view_scale * confidence
                mu[source] += amount
                mu[target] -= amount
                rows.append(
                    {
                        "date": date,
                        "signal_type": "market_skill_pairwise_mean_view",
                        "source": source,
                        "target": target,
                        "confidence": confidence,
                        "mu_tilt": amount,
                        "view_mode": view_mode,
                    }
                )
        else:
            rows.append(
                {
                    "date": date,
                    "signal_type": "market_skill_view_skipped",
                    "source": "",
                    "target": "",
                    "confidence": 0.0,
                    "mu_tilt": 0.0,
                    "view_mode": view_mode,
                }
            )

    if trend_scores is not None and not trend_scores.empty:
        scores = trend_scores.reindex(RISK_SLEEVES).astype(float).fillna(0.0)
        scores = scores - float(scores.mean())
        tilt = (scores * trend_scale).clip(lower=-max_abs_tilt, upper=max_abs_tilt)
        mu = mu.add(tilt, fill_value=0.0)
        for sleeve, value in tilt.items():
            if abs(float(value)) > 1e-12:
                rows.append(
                    {
                        "date": date,
                        "signal_type": "trend_posterior",
                        "source": sleeve,
                        "target": "",
                        "confidence": 1.0,
                        "mu_tilt": float(value),
                    }
                )

    itransformer_overlay = cfg.get("signal_overlays", {}).get("itransformer_rank", {})
    if (
        pred is not None
        and not pred.empty
        and bool(itransformer_overlay.get("ep_mean_enabled", False))
    ):
        active_pred = pred[pred["as_of_date"].eq(pd.Timestamp(date))].dropna(subset=["pred_return"]).copy()
        if not active_pred.empty:
            centered = active_pred.set_index("sleeve")["pred_return"].reindex(RISK_SLEEVES).fillna(0.0)
            centered = centered - float(centered.mean())
            prob = active_pred.set_index("sleeve")["pred_prob"].reindex(RISK_SLEEVES).fillna(0.0)
            confidence = float(prob.max()) if prob.notna().any() else 0.0
            tilt = centered * pred_scale * max(confidence, 0.0)
            tilt = tilt.clip(lower=-max_abs_tilt, upper=max_abs_tilt)
            mu = mu.add(tilt, fill_value=0.0)
            for sleeve, value in tilt.items():
                if abs(float(value)) > 1e-12:
                    rows.append(
                        {
                            "date": date,
                            "signal_type": "itransformer_prediction",
                            "source": sleeve,
                            "target": "",
                            "confidence": confidence,
                            "mu_tilt": float(value),
                        }
                    )

    prior_delta = (mu - prior_mu).clip(lower=-max_abs_tilt, upper=max_abs_tilt)
    posterior = prior_mu + prior_delta
    return posterior.reindex(RISK_SLEEVES).fillna(0.0), rows


def trend_scores(history: pd.DataFrame, cfg: dict[str, Any]) -> pd.Series:
    ep_cfg = cfg.get("ep_skfolio", {})
    lookbacks = [int(x) for x in ep_cfg.get("trend_lookback_months", [3, 6, 12])]
    scores: dict[str, float] = {}
    for sleeve in RISK_SLEEVES:
        parts = []
        for lb in lookbacks:
            if len(history) >= lb:
                parts.append(float(history[sleeve].tail(lb).sum()))
        scores[sleeve] = float(np.mean(parts)) if parts else 0.0
    return pd.Series(scores).reindex(RISK_SLEEVES).fillna(0.0)


def optimize_sleeve_weights(
    base_weights: pd.Series,
    posterior_mu: pd.Series,
    cov: pd.DataFrame,
    returns_window: pd.DataFrame | None,
    prev_weights: pd.Series | None,
    bounds: dict[str, tuple[float, float]],
    cfg: dict[str, Any],
) -> EPSkfolioResult:
    opt_cfg = effective_ep_config(cfg)
    sleeves = RISK_SLEEVES
    lo = np.array([bounds[s][0] for s in sleeves], dtype=float)
    hi = np.array([bounds[s][1] for s in sleeves], dtype=float)
    base = base_weights.reindex(sleeves).astype(float).fillna(0.0).to_numpy()
    base = project_simplex_bounds(base, lo, hi)
    prev = base if prev_weights is None else prev_weights.reindex(sleeves).astype(float).fillna(0.0).to_numpy()
    prev = project_simplex_bounds(prev, lo, hi)
    mu = posterior_mu.reindex(sleeves).astype(float).fillna(0.0).to_numpy()
    sigma = cov.reindex(index=sleeves, columns=sleeves).fillna(0.0).to_numpy(dtype=float)
    sigma = (sigma + sigma.T) / 2.0
    sigma += np.eye(len(sleeves)) * float(opt_cfg.get("cov_jitter", 1e-7))
    if not np.isfinite(sigma).all() or np.linalg.eigvalsh(sigma).min() <= 0:
        raise ValueError("Covariance must be finite and positive definite")

    risk_aversion = float(opt_cfg.get("risk_aversion", 3.0))
    base_penalty = float(opt_cfg.get("base_deviation_penalty", 6.0))
    turnover_penalty = float(opt_cfg.get("turnover_penalty", 0.45))
    max_active_deviation = float(opt_cfg.get("max_active_deviation", 0.06))
    max_turnover = float(opt_cfg.get("max_sleeve_turnover", 0.22))
    min_feasible_turnover = max(0.0, float(np.abs(base - prev).sum()) - max_active_deviation)
    effective_max_turnover = max(max_turnover, min_feasible_turnover + 1e-8)

    backend = str(opt_cfg.get("backend", "skfolio_meanrisk_clarabel")).lower()
    if backend.startswith("skfolio") and SKFOLIO_AVAILABLE and returns_window is not None and not returns_window.empty:
        x = returns_window.reindex(columns=sleeves).astype(float).fillna(0.0)
        if len(x) >= 2:
            try:
                base_const = base.copy()
                prev_const = prev.copy()
                mu_const = mu.copy()

                def expected_return(w):
                    return mu_const @ w

                def custom_objective(w):
                    return -base_penalty * cp.sum_squares(w - base_const) - turnover_penalty * cp.norm1(w - prev_const)

                def custom_constraints(w):
                    return [cp.norm1(w - base_const) <= max_active_deviation]

                model = MeanRisk(
                    prior_estimator=EmpiricalPrior(covariance_estimator=SuppliedCovariance(sigma)),
                    objective_function=ObjectiveFunction.MAXIMIZE_UTILITY,
                    risk_measure=RiskMeasure.VARIANCE,
                    risk_aversion=risk_aversion,
                    min_weights=lo,
                    max_weights=hi,
                    budget=1.0,
                    previous_weights=prev,
                    max_turnover=effective_max_turnover,
                    overwrite_expected_return=expected_return,
                    add_objective=custom_objective,
                    add_constraints=custom_constraints,
                    solver=str(opt_cfg.get("solver", "CLARABEL")),
                    raise_on_failure=True,
                )
                model.fit(x)
                weights = project_simplex_bounds(np.asarray(model.weights_, dtype=float), lo, hi)
                out = pd.Series(weights, index=sleeves)
                diagnostics = {
                    "covariance_source": "supplied_point_in_time_matrix",
                    "optimizer_backend": "skfolio_meanrisk_clarabel",
                    "status": "ok",
                    "message": "skfolio MeanRisk solved",
                    "risk_aversion": risk_aversion,
                    "base_deviation_penalty": base_penalty,
                    "turnover_penalty": turnover_penalty,
                    "max_active_deviation_limit": max_active_deviation,
                    "view_mode": str(opt_cfg.get("view_mode", "mean_rank")),
                    "view_risk_adjustment_enabled": bool(opt_cfg.get("view_risk_adjustment_enabled", False)),
                    "objective": float(risk_aversion * (weights @ sigma @ weights) - mu @ weights
                                       + base_penalty * np.sum((weights - base)**2)
                                       + turnover_penalty * np.abs(weights - prev).sum()),
                    "active_deviation": float(np.abs(weights - base).sum()),
                    "sleeve_turnover": float(np.abs(weights - prev).sum()),
                    "effective_max_turnover": effective_max_turnover,
                    "expected_return": float(mu @ weights),
                    "portfolio_vol_monthly": float(np.sqrt(max(weights @ sigma @ weights, 0.0))),
                }
                return EPSkfolioResult(weights=out, posterior_mu=posterior_mu, diagnostics=diagnostics)
            except Exception as exc:
                skfolio_error = f"{type(exc).__name__}: {exc}"
        else:
            skfolio_error = "not_enough_rows_for_skfolio"
    else:
        skfolio_error = "skfolio_backend_disabled_or_unavailable"

    def objective(w: np.ndarray) -> float:
        active = w - base
        turnover = w - prev
        return float(
            risk_aversion * (w @ sigma @ w)
            - mu @ w
            + base_penalty * (active @ active)
            + turnover_penalty * np.abs(turnover).sum()
        )

    constraints = [{"type": "eq", "fun": lambda w: float(w.sum() - 1.0)}]
    constraints.append({"type": "ineq", "fun": lambda w: max_active_deviation - float(np.abs(w - base).sum())})
    constraints.append({"type": "ineq", "fun": lambda w: effective_max_turnover - float(np.abs(w - prev).sum())})

    res = minimize(
        objective,
        base,
        method="SLSQP",
        bounds=list(zip(lo, hi)),
        constraints=constraints,
        options={"maxiter": int(opt_cfg.get("max_iterations", 300)), "ftol": float(opt_cfg.get("tolerance", 1e-9)), "disp": False},
    )
    if res.success and np.all(np.isfinite(res.x)):
        weights = project_simplex_bounds(res.x, lo, hi)
        status = "ok"
        message = str(res.message)
    else:
        weights = base
        status = "fallback_base"
        message = str(getattr(res, "message", "optimizer_failed"))

    out = pd.Series(weights, index=sleeves)
    diagnostics = {
        "covariance_source": "supplied_point_in_time_matrix",
        "optimizer_backend": "scipy_slsqp_fallback",
        "status": status,
        "message": f"{message}; skfolio={skfolio_error}",
        "risk_aversion": risk_aversion,
        "base_deviation_penalty": base_penalty,
        "turnover_penalty": turnover_penalty,
        "max_active_deviation_limit": max_active_deviation,
        "view_mode": str(opt_cfg.get("view_mode", "mean_rank")),
        "view_risk_adjustment_enabled": bool(opt_cfg.get("view_risk_adjustment_enabled", False)),
        "objective": objective(weights),
        "active_deviation": float(np.abs(weights - base).sum()),
        "sleeve_turnover": float(np.abs(weights - prev).sum()),
        "effective_max_turnover": effective_max_turnover,
        "expected_return": float(mu @ weights),
        "portfolio_vol_monthly": float(np.sqrt(max(weights @ sigma @ weights, 0.0))),
    }
    return EPSkfolioResult(weights=out, posterior_mu=posterior_mu, diagnostics=diagnostics)


def optimize_path(
    base: pd.DataFrame,
    sleeve_ret: pd.DataFrame,
    views: pd.DataFrame,
    pred: pd.DataFrame,
    bounds: dict[str, tuple[float, float]],
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    opt_cfg = effective_ep_config(cfg)
    lookback = int(opt_cfg.get("cov_lookback_months", cfg.get("risk_budget", {}).get("lookback_months", 36)))
    min_history = int(opt_cfg.get("min_history_months", cfg.get("risk_budget", {}).get("min_history_months", 12)))
    cov_shrink = float(opt_cfg.get("covariance_shrinkage_to_diag", cfg.get("risk_budget", {}).get("covariance_shrinkage_to_diag", 0.2)))
    rows: list[dict[str, Any]] = []
    posterior_rows: list[dict[str, Any]] = []
    signal_rows: list[dict[str, Any]] = []
    prev_weights: pd.Series | None = None
    floor = float(cfg.get("risk_budget", {}).get("diagonal_vol_floor", 0.002)) ** 2

    for _, base_row in base.iterrows():
        date = pd.Timestamp(base_row["date"])
        hist = sleeve_ret.loc[:date].tail(lookback).copy()
        base_weights = pd.Series({s: float(base_row[s]) for s in RISK_SLEEVES})
        if len(hist) < min_history:
            weights = base_weights
            posterior_mu = pd.Series(0.0, index=RISK_SLEEVES)
            diagnostics = {"optimizer_backend": "not_enough_history", "status": "fallback_base"}
        else:
            prior_mu = build_prior_mu(hist, cfg)
            posterior_mu, sig_rows = posterior_from_views(prior_mu, views, pred, trend_scores(hist, cfg), date, cfg)
            signal_rows.extend(sig_rows)
            cov_estimate = estimate_covariance(
                hist,
                RISK_SLEEVES,
                opt_cfg,
                default_shrinkage=cov_shrink,
                floor=floor,
                jitter=float(opt_cfg.get("cov_jitter", 1e-7)),
            )
            cov = cov_estimate.covariance
            result = optimize_sleeve_weights(base_weights, posterior_mu, cov, hist, prev_weights, bounds, cfg)
            weights = result.weights
            posterior_mu = result.posterior_mu
            diagnostics = {**cov_estimate.diagnostics, **result.diagnostics}
        prev_weights = weights
        row = {"date": date, "product_profile": cfg["product"]["selected_profile"], "base_method": "ep_skfolio_optimizer"}
        row.update({s: float(weights.reindex(RISK_SLEEVES).fillna(0.0)[s]) for s in RISK_SLEEVES})
        row["cash"] = 0.0
        row.update({f"opt_{k}": v for k, v in diagnostics.items()})
        rows.append(row)
        posterior_rows.append({"date": date, **{f"posterior_mu_{s}": float(posterior_mu.get(s, 0.0)) for s in RISK_SLEEVES}})

    return pd.DataFrame(rows), pd.DataFrame(posterior_rows), pd.DataFrame(signal_rows)

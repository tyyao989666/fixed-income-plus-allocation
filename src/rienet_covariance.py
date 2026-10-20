from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CovarianceEstimate:
    covariance: pd.DataFrame
    diagnostics: dict[str, Any]


def _nearest_psd(matrix: np.ndarray, floor: float) -> tuple[np.ndarray, float]:
    sym = (matrix + matrix.T) / 2.0
    vals, vecs = np.linalg.eigh(sym)
    min_eig = float(vals.min()) if len(vals) else 0.0
    vals = np.maximum(vals, floor)
    repaired = (vecs * vals) @ vecs.T
    repaired = (repaired + repaired.T) / 2.0
    return repaired, min_eig


def _weighted_covariance(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    mean = np.sum(x * weights[:, None], axis=0)
    centered = x - mean
    denom = max(1.0 - float(np.sum(weights**2)), 1e-12)
    return (centered * weights[:, None]).T @ centered / denom


def _exp_weights(n: int, half_life: float) -> np.ndarray:
    if n <= 0:
        return np.array([], dtype=float)
    half_life = max(float(half_life), 1.0)
    age = np.arange(n - 1, -1, -1, dtype=float)
    weights = np.power(0.5, age / half_life)
    return weights / weights.sum()


def sample_shrink_covariance(
    history: pd.DataFrame,
    columns: list[str],
    shrinkage_to_diag: float,
    floor: float,
    jitter: float = 0.0,
) -> CovarianceEstimate:
    hist = history.reindex(columns=columns).astype(float).fillna(0.0)
    cov = hist.cov().reindex(index=columns, columns=columns).fillna(0.0)
    diag_values = np.maximum(np.diag(cov.to_numpy(dtype=float)), floor)
    diag = pd.DataFrame(np.diag(diag_values), index=columns, columns=columns)
    shrink = float(np.clip(shrinkage_to_diag, 0.0, 1.0))
    cleaned = (1.0 - shrink) * cov + shrink * diag
    arr, min_eig_before = _nearest_psd(cleaned.to_numpy(dtype=float), floor + float(jitter))
    out = pd.DataFrame(arr, index=columns, columns=columns)
    return CovarianceEstimate(
        covariance=out,
        diagnostics={
            "covariance_estimator": "sample_shrink",
            "cov_shrinkage_to_diag": shrink,
            "cov_min_eigenvalue_before_psd": min_eig_before,
            "cov_effective_observations": float(len(hist)),
        },
    )


def ewma_shrink_covariance(
    history: pd.DataFrame,
    columns: list[str],
    shrinkage_to_diag: float,
    floor: float,
    half_life: float,
    jitter: float = 0.0,
) -> CovarianceEstimate:
    hist = history.reindex(columns=columns).astype(float).fillna(0.0)
    weights = _exp_weights(len(hist), half_life)
    cov_arr = _weighted_covariance(hist.to_numpy(dtype=float), weights)
    diag_values = np.maximum(np.diag(cov_arr), floor)
    diag = np.diag(diag_values)
    shrink = float(np.clip(shrinkage_to_diag, 0.0, 1.0))
    cleaned = (1.0 - shrink) * cov_arr + shrink * diag
    arr, min_eig_before = _nearest_psd(cleaned, floor + float(jitter))
    out = pd.DataFrame(arr, index=columns, columns=columns)
    return CovarianceEstimate(
        covariance=out,
        diagnostics={
            "covariance_estimator": "ewma_shrink",
            "cov_shrinkage_to_diag": shrink,
            "cov_half_life": float(half_life),
            "cov_min_eigenvalue_before_psd": min_eig_before,
            "cov_effective_observations": float(1.0 / np.sum(weights**2)) if len(weights) else 0.0,
        },
    )


def rienet_inspired_covariance(
    history: pd.DataFrame,
    columns: list[str],
    cfg: dict[str, Any],
    floor: float,
    jitter: float = 0.0,
) -> CovarianceEstimate:
    """RIEnet-inspired covariance cleaner.

    This is intentionally not advertised as the full TensorFlow RIEnet model. It
    mirrors the paper's deployable idea for this repo: lag-weight returns,
    separate marginal vol from correlation, clean the correlation eigen-spectrum,
    then reconstruct a PSD covariance matrix for the existing optimizer.
    """

    hist = history.reindex(columns=columns).astype(float).fillna(0.0)
    x = hist.to_numpy(dtype=float)
    cleaner_cfg = cfg.get("rienet_cleaner", {}) or {}
    half_life = float(cleaner_cfg.get("half_life_months", max(len(hist) / 2.0, 1.0)))
    weights = _exp_weights(len(hist), half_life)
    if len(weights) == 0:
        return sample_shrink_covariance(hist, columns, 1.0, floor, jitter)

    winsor = cleaner_cfg.get("winsorize_z", 4.0)
    if winsor is not None:
        std = np.nanstd(x, axis=0, ddof=0)
        std = np.where(std > 1e-12, std, 1.0)
        mean = np.nanmean(x, axis=0)
        z = (x - mean) / std
        x = mean + np.clip(z, -float(winsor), float(winsor)) * std

    cov_arr = _weighted_covariance(x, weights)
    vol = np.sqrt(np.maximum(np.diag(cov_arr), floor))
    corr = cov_arr / np.outer(vol, vol)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 1.0)
    corr, corr_min_eig_before = _nearest_psd(corr, 1e-8)

    eigvals, eigvecs = np.linalg.eigh(corr)
    eigvals = np.maximum(eigvals, 1e-8)
    n_assets = len(columns)
    n_obs_eff = max(float(1.0 / np.sum(weights**2)), n_assets + 1.0)
    q = n_assets / n_obs_eff
    mp_upper = (1.0 + np.sqrt(q)) ** 2
    bulk_mean = float(np.mean(eigvals[eigvals <= mp_upper])) if np.any(eigvals <= mp_upper) else 1.0

    eigen_shrink = float(np.clip(cleaner_cfg.get("eigen_shrinkage", 0.45), 0.0, 1.0))
    max_condition = float(cleaner_cfg.get("max_condition_number", 25.0))
    cleaned_eig = eigvals.copy()
    bulk = cleaned_eig <= mp_upper
    cleaned_eig[bulk] = (1.0 - eigen_shrink) * cleaned_eig[bulk] + eigen_shrink * bulk_mean
    min_allowed = max(float(cleaner_cfg.get("min_eigenvalue", 1e-4)), cleaned_eig.max() / max_condition)
    cleaned_eig = np.maximum(cleaned_eig, min_allowed)
    cleaned_eig *= n_assets / cleaned_eig.sum()

    cleaned_corr = (eigvecs * cleaned_eig) @ eigvecs.T
    cleaned_corr = (cleaned_corr + cleaned_corr.T) / 2.0
    diag = np.sqrt(np.maximum(np.diag(cleaned_corr), 1e-12))
    cleaned_corr = cleaned_corr / np.outer(diag, diag)
    cleaned_corr, min_eig_before = _nearest_psd(cleaned_corr, 1e-8)
    np.fill_diagonal(cleaned_corr, 1.0)

    vol_shrink = float(np.clip(cleaner_cfg.get("vol_shrinkage_to_median", 0.20), 0.0, 1.0))
    median_vol = float(np.median(vol))
    cleaned_vol = (1.0 - vol_shrink) * vol + vol_shrink * median_vol
    cov_clean = cleaned_corr * np.outer(cleaned_vol, cleaned_vol)
    cov_clean, final_min_eig_before = _nearest_psd(cov_clean, floor + float(jitter))

    out = pd.DataFrame(cov_clean, index=columns, columns=columns)
    return CovarianceEstimate(
        covariance=out,
        diagnostics={
            "covariance_estimator": "rienet_inspired",
            "cov_half_life": half_life,
            "cov_effective_observations": n_obs_eff,
            "cov_mp_upper": float(mp_upper),
            "cov_bulk_mean_eigenvalue": bulk_mean,
            "cov_eigen_shrinkage": eigen_shrink,
            "cov_vol_shrinkage_to_median": vol_shrink,
            "cov_corr_min_eigenvalue_before_clean": corr_min_eig_before,
            "cov_cleaned_corr_min_eigenvalue_before_psd": min_eig_before,
            "cov_min_eigenvalue_before_psd": final_min_eig_before,
        },
    )


def estimate_covariance(
    history: pd.DataFrame,
    columns: list[str],
    cfg: dict[str, Any],
    default_shrinkage: float,
    floor: float,
    jitter: float = 0.0,
) -> CovarianceEstimate:
    cov_cfg = cfg.get("covariance_estimator", {}) or {}
    method = str(cov_cfg.get("method", "sample_shrink")).lower()
    shrinkage = float(cov_cfg.get("shrinkage_to_diag", default_shrinkage))
    if method in {"sample", "sample_shrink", "rolling_sample"}:
        return sample_shrink_covariance(history, columns, shrinkage, floor, jitter)
    if method in {"ewma", "ewma_shrink"}:
        return ewma_shrink_covariance(
            history,
            columns,
            shrinkage,
            floor,
            float(cov_cfg.get("half_life_months", 12.0)),
            jitter,
        )
    if method in {"rienet", "rienet_like", "rienet_inspired", "rienet_cleaner"}:
        return rienet_inspired_covariance(history, columns, cov_cfg, floor, jitter)
    raise ValueError(f"Unknown covariance estimator method: {method}")

from __future__ import annotations

import copy
import itertools
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from scipy.stats import rankdata, spearmanr
from torch.utils.data import DataLoader, TensorDataset

from src.itransformer_daily_signal import (
    DailyITransformerPaths,
    load_daily_config,
    load_daily_returns_with_proxy,
)
from src.itransformer_daily_timing_v2 import preferred_daily_sleeve_returns
from src.itransformer_signal import RISK_SLEEVES


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "market_state_vae.yaml"
CONCEPTS = [
    "trend_strength",
    "volatility_stress",
    "mean_reversion_pressure",
    "macro_commodity_regime",
]


def load_config(path: Path | None = None) -> dict[str, Any]:
    with (path or DEFAULT_CONFIG).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def output_path(cfg: dict[str, Any], key: str) -> Path:
    return ROOT / cfg["outputs"][key]


def _rolling_z(values: pd.Series, window: int, minimum: int) -> pd.Series:
    mean = values.rolling(window, min_periods=minimum).mean()
    std = values.rolling(window, min_periods=minimum).std(ddof=0)
    return (values - mean) / std.where(std.gt(1e-12))


def _compounded_return(values: pd.Series, window: int) -> pd.Series:
    log_return = np.log1p(values.clip(lower=-0.99))
    return np.expm1(log_return.rolling(window, min_periods=window).sum())


def _rolling_drawdown(values: np.ndarray) -> float:
    nav = np.cumprod(1.0 + np.nan_to_num(values, nan=0.0))
    peak = np.maximum.accumulate(nav)
    return float(np.min(nav / np.maximum(peak, 1e-12) - 1.0))


def _safe_mean(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    available = [column for column in columns if column in frame]
    if not available:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return frame[available].mean(axis=1)


def build_feature_frame(cfg: dict[str, Any]) -> tuple[pd.DataFrame, list[str]]:
    daily_cfg = load_daily_config(ROOT / cfg["data"]["daily_config"])
    sleeve_returns, _ = preferred_daily_sleeve_returns(daily_cfg)
    sleeve_returns.index = pd.to_datetime(sleeve_returns.index)
    sleeve_returns = sleeve_returns.sort_index()
    daily_assets = load_daily_returns_with_proxy(daily_cfg, DailyITransformerPaths())
    china_calendar = pd.DatetimeIndex(
        pd.to_datetime(
            daily_assets.loc[
                daily_assets["asset_code"].astype(str).eq("equity_broad"), "date"
            ],
            errors="coerce",
        ).dropna().unique()
    ).sort_values()
    sleeve_returns = sleeve_returns.reindex(china_calendar)
    start = pd.Timestamp(cfg["data"]["start_date"])
    end = pd.Timestamp(cfg["data"]["final_date"])
    sleeve_returns = sleeve_returns.loc[
        sleeve_returns.index.to_series().between(start, end).to_numpy()
    ]

    features = pd.DataFrame(index=sleeve_returns.index)
    windows = [int(value) for value in cfg["data"]["sleeve_windows_days"]]
    for sleeve in RISK_SLEEVES:
        returns = sleeve_returns[sleeve].astype(float).replace([np.inf, -np.inf], np.nan)
        features[f"sleeve_{sleeve}_ret_1d"] = returns
        nav = (1.0 + returns.fillna(0.0)).cumprod()
        for window in windows:
            features[f"sleeve_{sleeve}_mom_{window}d"] = _compounded_return(returns, window)
        features[f"sleeve_{sleeve}_vol_20d"] = (
            returns.rolling(20, min_periods=15).std(ddof=0) * math.sqrt(252.0)
        )
        features[f"sleeve_{sleeve}_vol_60d"] = (
            returns.rolling(60, min_periods=40).std(ddof=0) * math.sqrt(252.0)
        )
        features[f"sleeve_{sleeve}_drawdown_60d"] = returns.rolling(
            60, min_periods=40
        ).apply(_rolling_drawdown, raw=True)
        moving_average = nav.rolling(20, min_periods=15).mean()
        features[f"sleeve_{sleeve}_ma_gap_20d"] = nav / moving_average - 1.0

    panel = pd.read_parquet(DailyITransformerPaths().daily_features)
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce")
    panel = panel.dropna(subset=["date"]).sort_values("date").set_index("date")
    fill_limit = int(cfg["data"]["maximum_forward_fill_days"])
    for column in cfg["data"]["macro_features"]:
        if column not in panel:
            continue
        values = (
            pd.to_numeric(panel[column], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .reindex(features.index)
            .ffill(limit=fill_limit)
        )
        features[f"macro_{column}_z_3y"] = _rolling_z(values, 756, 126)
        features[f"macro_{column}_delta_20d_z_1y"] = _rolling_z(values.diff(20), 252, 60)

    folds = cfg.get("walk_forward", {}).get("folds", [])
    feature_fit_end = folds[0]["fit_end"] if folds else cfg["data"]["train_end"]
    train_end = pd.Timestamp(feature_fit_end)
    train = features.loc[features.index <= train_end]
    minimum_coverage = float(cfg["data"]["minimum_feature_train_coverage"])
    minimum_observations = int(cfg["data"]["minimum_train_observations"])
    feature_columns = [
        column
        for column in features
        if train[column].notna().sum() >= minimum_observations
        and train[column].notna().mean() >= minimum_coverage
        and train[column].var(skipna=True) > 1e-12
    ]
    features = features[feature_columns].copy()

    features["trend_strength"] = _safe_mean(
        features,
        [
            "sleeve_convertible_bonds_mom_20d",
            "sleeve_convertible_bonds_mom_60d",
            "sleeve_equity_mom_20d",
            "sleeve_equity_mom_60d",
            "sleeve_alternatives_mom_20d",
        ],
    )
    features["volatility_stress"] = _safe_mean(
        features,
        [f"sleeve_{sleeve}_vol_20d" for sleeve in RISK_SLEEVES]
        + [f"sleeve_{sleeve}_vol_60d" for sleeve in RISK_SLEEVES],
    ) + _safe_mean(
        features,
        [f"sleeve_{sleeve}_drawdown_60d" for sleeve in RISK_SLEEVES],
    ).abs()
    features["mean_reversion_pressure"] = _safe_mean(
        features,
        [
            "sleeve_convertible_bonds_ma_gap_20d",
            "sleeve_equity_ma_gap_20d",
            "sleeve_alternatives_ma_gap_20d",
        ],
    )
    commodity_support = _safe_mean(
        features,
        [
            "macro_alt_nanhua_commodity_z_3y",
            "macro_alt_nanhua_commodity_delta_20d_z_1y",
            "macro_alt_gold_au9999_z_3y",
            "macro_alt_gold_au9999_delta_20d_z_1y",
            "sleeve_alternatives_mom_20d",
            "sleeve_alternatives_mom_60d",
        ],
    )
    commodity_headwind = _safe_mean(
        features,
        [
            "macro_alt_usd_index_z_3y",
            "macro_alt_usd_index_delta_20d_z_1y",
            "macro_alt_real_rate_proxy_z_3y",
            "macro_alt_real_rate_proxy_delta_20d_z_1y",
        ],
    )
    features["macro_commodity_regime"] = commodity_support - commodity_headwind.fillna(0.0)

    cards = pd.read_parquet(ROOT / cfg["data"]["market_state_cards"])
    cards["date"] = pd.to_datetime(cards["date"], errors="coerce")
    card_columns = [
        "date", "stock_bond_regime", "bond_score", "equity_score",
        "cr_state", "r_state", "cb_state", "eq_state", "alt_state", "rb_state",
    ]
    cards = cards[[column for column in card_columns if column in cards]].drop_duplicates("date")
    frame = features.reset_index(names="date").merge(cards, on="date", how="left")
    frame = frame.sort_values("date").reset_index(drop=True)
    output_path(cfg, "feature_frame").parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path(cfg, "feature_frame"), index=False)
    return frame, feature_columns


@dataclass
class WindowData:
    values: np.ndarray
    dates: np.ndarray
    feature_columns: list[str]
    concepts: np.ndarray
    train_mask: np.ndarray
    validation_mask: np.ndarray
    oos_mask: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    source_rows: np.ndarray


def build_windows(
    frame: pd.DataFrame, feature_columns: list[str], cfg: dict[str, Any]
) -> WindowData:
    dates = pd.to_datetime(frame["date"])
    train_end = pd.Timestamp(cfg["data"]["train_end"])
    validation_end = pd.Timestamp(cfg["data"]["validation_end"])
    train_rows = dates <= train_end
    raw = frame[feature_columns].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
    train_raw = raw[train_rows.to_numpy()]
    lower_q = float(cfg["preprocessing"]["winsor_lower_quantile"])
    upper_q = float(cfg["preprocessing"]["winsor_upper_quantile"])
    lower = np.nanquantile(train_raw, lower_q, axis=0)
    upper = np.nanquantile(train_raw, upper_q, axis=0)
    median = np.nanmedian(train_raw, axis=0)
    clipped = np.clip(raw, lower, upper)
    clipped = np.where(np.isfinite(clipped), clipped, median)
    train_clipped = clipped[train_rows.to_numpy()]
    mean = train_clipped.mean(axis=0)
    std = train_clipped.std(axis=0)
    std = np.where(std >= float(cfg["preprocessing"]["minimum_std"]), std, 1.0)
    scaled = (clipped - mean) / std

    concept_raw = frame[CONCEPTS].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
    concept_train = concept_raw[train_rows.to_numpy()]
    concept_median = np.nanmedian(concept_train, axis=0)
    concept_values = np.where(np.isfinite(concept_raw), concept_raw, concept_median)
    concept_mean = concept_values[train_rows.to_numpy()].mean(axis=0)
    concept_std = concept_values[train_rows.to_numpy()].std(axis=0)
    concept_std = np.where(concept_std > 1e-12, concept_std, 1.0)
    concept_values = (concept_values - concept_mean) / concept_std

    window = int(cfg["data"]["window_size_days"])
    values, window_dates, concepts, source_rows = [], [], [], []
    for end in range(window - 1, len(frame)):
        values.append(scaled[end - window + 1 : end + 1].reshape(-1))
        window_dates.append(dates.iloc[end].to_datetime64())
        concepts.append(concept_values[end])
        source_rows.append(end)
    values_array = np.asarray(values, dtype=np.float32)
    date_array = np.asarray(window_dates, dtype="datetime64[ns]")
    concept_array = np.asarray(concepts, dtype=np.float32)
    train_mask = date_array <= np.datetime64(train_end)
    validation_mask = (date_array > np.datetime64(train_end)) & (
        date_array <= np.datetime64(validation_end)
    )
    oos_mask = date_array > np.datetime64(validation_end)
    np.savez_compressed(
        output_path(cfg, "windows_npz"),
        values=values_array,
        dates=date_array.astype(np.int64),
        concepts=concept_array,
        train_mask=train_mask,
        validation_mask=validation_mask,
        oos_mask=oos_mask,
        lower=lower,
        upper=upper,
        mean=mean,
        std=std,
        source_rows=np.asarray(source_rows, dtype=np.int64),
        feature_columns=np.asarray(feature_columns),
    )
    return WindowData(
        values_array, date_array, feature_columns, concept_array,
        train_mask, validation_mask, oos_mask, lower, upper, mean, std,
        np.asarray(source_rows, dtype=np.int64),
    )


class FinanceVAE(nn.Module):
    def __init__(self, input_dim: int, cfg: dict[str, Any]) -> None:
        super().__init__()
        hidden = int(cfg["hidden_dim"])
        latent = int(cfg["latent_dim"])
        dropout = float(cfg.get("dropout", 0.0))
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.GELU(),
        )
        self.fc_mu = nn.Linear(hidden // 2, latent)
        self.fc_logvar = nn.Linear(hidden // 2, latent)
        self.decoder = nn.Sequential(
            nn.Linear(latent, hidden // 2), nn.GELU(),
            nn.Linear(hidden // 2, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, input_dim),
        )

    def encode(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(values)
        return self.fc_mu(hidden), self.fc_logvar(hidden).clamp(-8.0, 6.0)

    def forward(
        self, values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(values)
        std = torch.exp(0.5 * logvar)
        latent = mu + torch.randn_like(std) * std
        return self.decoder(latent), mu, logvar


def _vae_loss(
    reconstructed: torch.Tensor,
    values: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    reconstruction = F.mse_loss(reconstructed, values, reduction="mean")
    kl = -0.5 * torch.mean(1.0 + logvar - mu.square() - logvar.exp())
    return reconstruction + beta * kl, reconstruction, kl


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _evaluate(model: FinanceVAE, values: torch.Tensor, beta: float) -> tuple[float, float, float]:
    model.eval()
    with torch.no_grad():
        mu, logvar = model.encode(values)
        reconstructed = model.decoder(mu)
        loss, reconstruction, kl = _vae_loss(reconstructed, values, mu, logvar, beta)
    return float(loss), float(reconstruction), float(kl)


def train_models(
    data: WindowData, cfg: dict[str, Any]
) -> tuple[FinanceVAE, pd.DataFrame, pd.DataFrame, int]:
    model_cfg = cfg["model"]
    train_tensor = torch.from_numpy(data.values[data.train_mask])
    validation_tensor = torch.from_numpy(data.values[data.validation_mask])
    logs: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    candidates: list[tuple[float, int, dict[str, torch.Tensor], int]] = []
    for seed in [int(value) for value in model_cfg["seeds"]]:
        _set_seed(seed)
        model = FinanceVAE(data.values.shape[1], model_cfg)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(model_cfg["learning_rate"]),
            weight_decay=float(model_cfg["weight_decay"]),
        )
        loader = DataLoader(
            TensorDataset(train_tensor),
            batch_size=int(model_cfg["batch_size"]),
            shuffle=True,
            generator=torch.Generator().manual_seed(seed),
        )
        best_loss = float("inf")
        best_epoch = 0
        best_state: dict[str, torch.Tensor] | None = None
        stale = 0
        for epoch in range(1, int(model_cfg["max_epochs"]) + 1):
            model.train()
            total_loss = total_reconstruction = total_kl = 0.0
            count = 0
            for (batch,) in loader:
                optimizer.zero_grad(set_to_none=True)
                reconstructed, mu, logvar = model(batch)
                loss, reconstruction, kl = _vae_loss(
                    reconstructed, batch, mu, logvar, float(model_cfg["beta_kl"])
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(model_cfg["gradient_clip"])
                )
                optimizer.step()
                size = len(batch)
                total_loss += float(loss.detach()) * size
                total_reconstruction += float(reconstruction.detach()) * size
                total_kl += float(kl.detach()) * size
                count += size
            validation_loss, validation_reconstruction, validation_kl = _evaluate(
                model, validation_tensor, float(model_cfg["beta_kl"])
            )
            row = {
                "seed": seed,
                "epoch": epoch,
                "train_loss": total_loss / max(count, 1),
                "train_reconstruction": total_reconstruction / max(count, 1),
                "train_kl": total_kl / max(count, 1),
                "validation_loss": validation_loss,
                "validation_reconstruction": validation_reconstruction,
                "validation_kl": validation_kl,
            }
            logs.append(row)
            if epoch >= int(model_cfg["minimum_epochs"]) and validation_loss < best_loss - 1e-5:
                best_loss = validation_loss
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                }
                stale = 0
            elif epoch >= int(model_cfg["minimum_epochs"]):
                stale += 1
            if stale >= int(model_cfg["patience"]):
                break
        if best_state is None:
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            best_epoch = epoch
            best_loss = validation_loss
        summaries.append(
            {
                "seed": seed,
                "epochs_run": epoch,
                "best_epoch": best_epoch,
                "best_validation_loss": best_loss,
            }
        )
        candidates.append((best_loss, seed, best_state, best_epoch))
    best_loss, selected_seed, selected_state, selected_epoch = min(candidates, key=lambda item: item[0])
    selected = FinanceVAE(data.values.shape[1], model_cfg)
    selected.load_state_dict(selected_state)
    selected.eval()
    return selected, pd.DataFrame(logs), pd.DataFrame(summaries), selected_seed


def deterministic_inference(
    model: FinanceVAE, values: np.ndarray, batch_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    latent_parts, logvar_parts, error_parts = [], [], []
    loader = DataLoader(TensorDataset(torch.from_numpy(values)), batch_size=batch_size)
    model.eval()
    with torch.no_grad():
        for (batch,) in loader:
            mu, logvar = model.encode(batch)
            reconstructed = model.decoder(mu)
            error = (reconstructed - batch).square().mean(dim=1)
            latent_parts.append(mu.numpy())
            logvar_parts.append(logvar.numpy())
            error_parts.append(error.numpy())
    return (
        np.concatenate(latent_parts),
        np.concatenate(logvar_parts),
        np.concatenate(error_parts),
    )


def _correlation(x: np.ndarray, y: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 20 or np.std(x[valid]) <= 1e-12 or np.std(y[valid]) <= 1e-12:
        return np.nan
    return float(spearmanr(x[valid], y[valid]).statistic)


def assign_axes(latent: np.ndarray, concepts: np.ndarray, train_mask: np.ndarray) -> dict[str, Any]:
    dimension_count = latent.shape[1]
    correlation = np.array(
        [
            [_correlation(latent[train_mask, dim], concepts[train_mask, concept]) for concept in range(4)]
            for dim in range(dimension_count)
        ]
    )
    best_score = -np.inf
    best_assignment: tuple[int, ...] | None = None
    for dimensions in itertools.permutations(range(dimension_count), 4):
        score = sum(
            abs(correlation[dimensions[concept], concept])
            if np.isfinite(correlation[dimensions[concept], concept]) else 0.0
            for concept in range(4)
        )
        if score > best_score:
            best_score = score
            best_assignment = dimensions
    if best_assignment is None:
        raise RuntimeError("Unable to assign latent dimensions")
    assignment: dict[str, Any] = {}
    for concept_position, concept in enumerate(CONCEPTS):
        dimension = int(best_assignment[concept_position])
        raw_value = correlation[dimension, concept_position]
        value = float(raw_value) if np.isfinite(raw_value) else 0.0
        assignment[concept] = {
            "dimension": dimension,
            "sign": 1.0 if value >= 0.0 else -1.0,
            "train_spearman": value,
        }
    return assignment


def _fit_kmeans(values: np.ndarray, clusters: int, restarts: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    best_centers = None
    best_inertia = np.inf
    for _ in range(restarts):
        centers = values[rng.choice(len(values), size=clusters, replace=False)].copy()
        for _ in range(100):
            distance = np.square(values[:, None, :] - centers[None, :, :]).sum(axis=2)
            labels = distance.argmin(axis=1)
            updated = centers.copy()
            for cluster in range(clusters):
                selected = values[labels == cluster]
                updated[cluster] = selected.mean(axis=0) if len(selected) else values[rng.integers(len(values))]
            if np.max(np.abs(updated - centers)) < 1e-6:
                centers = updated
                break
            centers = updated
        inertia = np.square(values - centers[labels]).sum()
        if inertia < best_inertia:
            best_inertia = inertia
            best_centers = centers
    if best_centers is None:
        raise RuntimeError("KMeans failed")
    return best_centers


def _empirical_percentile(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    reference = np.sort(reference[np.isfinite(reference)])
    return np.searchsorted(reference, values, side="right") / max(len(reference), 1)


def _auc(labels: np.ndarray, scores: np.ndarray) -> float:
    valid = np.isfinite(scores)
    labels = labels[valid].astype(bool)
    scores = scores[valid]
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return np.nan
    ranks = rankdata(scores)
    return float((ranks[labels].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def build_diagnostics(
    frame: pd.DataFrame,
    data: WindowData,
    latent: np.ndarray,
    logvar: np.ndarray,
    error: np.ndarray,
    assignment: dict[str, Any],
    cfg: dict[str, Any],
    selected_seed: int,
    threshold_mask: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    reference_mask = data.train_mask if threshold_mask is None else threshold_mask
    if int(reference_mask.sum()) < 20:
        raise ValueError("Anomaly threshold reference requires at least 20 observations")
    diagnostic = pd.DataFrame({"date": pd.to_datetime(data.dates)})
    for dimension in range(latent.shape[1]):
        diagnostic[f"latent_z{dimension + 1}"] = latent[:, dimension]
        diagnostic[f"latent_std_z{dimension + 1}"] = np.exp(0.5 * logvar[:, dimension])
    for concept, details in assignment.items():
        dimension = int(details["dimension"])
        oriented = float(details["sign"]) * latent[:, dimension]
        train_mean = oriented[data.train_mask].mean()
        train_std = oriented[data.train_mask].std()
        diagnostic[f"coord_{concept}"] = (oriented - train_mean) / max(train_std, 1e-12)
    diagnostic[CONCEPTS] = data.concepts
    diagnostic["reconstruction_error"] = error
    diagnostic["reconstruction_error_percentile"] = _empirical_percentile(
        error[reference_mask], error
    )
    drift_window = int(cfg["diagnostics"]["drift_window_days"])
    drift = np.full(len(latent), np.nan)
    drift[drift_window:] = np.linalg.norm(latent[drift_window:] - latent[:-drift_window], axis=1)
    diagnostic[f"latent_drift_{drift_window}d"] = drift
    valid_reference_drift = drift[reference_mask & np.isfinite(drift)]
    drift_fill = np.nanmedian(valid_reference_drift)
    drift_for_score = np.where(np.isfinite(drift), drift, drift_fill)
    diagnostic["latent_drift_percentile"] = _empirical_percentile(
        valid_reference_drift, drift_for_score
    )

    warning_q = float(cfg["diagnostics"]["warning_quantile"])
    anomaly_q = float(cfg["diagnostics"]["anomaly_quantile"])
    extreme_q = float(cfg["diagnostics"].get("extreme_anomaly_quantile", 0.99))
    reference_error = error[reference_mask]
    error_warning = float(np.quantile(reference_error, warning_q))
    error_anomaly = float(np.quantile(reference_error, anomaly_q))
    error_extreme = float(np.quantile(reference_error, extreme_q))
    drift_warning = float(np.quantile(valid_reference_drift, warning_q))
    drift_anomaly = float(np.quantile(valid_reference_drift, anomaly_q))
    familiarity = 1.0 - np.clip(
        (error - np.median(reference_error))
        / max(error_anomaly - np.median(reference_error), 1e-12),
        0.0,
        1.0,
    )
    stability = 1.0 - np.clip(
        (drift_for_score - np.median(valid_reference_drift))
        / max(drift_anomaly - np.median(valid_reference_drift), 1e-12),
        0.0,
        1.0,
    )
    confidence = 0.5 * familiarity + 0.5 * stability
    minimum = float(cfg["diagnostics"]["minimum_confidence_multiplier"])
    diagnostic["vae_familiarity_score"] = familiarity
    diagnostic["vae_latent_stability_score"] = stability
    diagnostic["vae_state_confidence"] = confidence
    diagnostic["vae_grid_confidence_multiplier"] = minimum + (1.0 - minimum) * confidence
    diagnostic["vae_warning"] = (error >= error_warning) | (drift_for_score >= drift_warning)
    joint_anomaly = (error >= error_anomaly) & (drift_for_score >= drift_anomaly)
    extreme_error = error >= error_extreme
    diagnostic["vae_abstain"] = extreme_error | joint_anomaly
    diagnostic["vae_abstain_reason"] = np.select(
        [extreme_error, joint_anomaly, diagnostic["vae_warning"]],
        ["extreme_reconstruction", "joint_error_and_drift", "warning_only"],
        default="normal",
    )
    diagnostic["vae_state_label"] = np.where(
        diagnostic["vae_abstain"], "ABSTAIN",
        np.where(diagnostic["vae_warning"], "WARNING", "FAMILIAR"),
    )
    diagnostic["selected_seed"] = selected_seed

    source = frame.iloc[data.source_rows].reset_index(drop=True)
    for column in [
        "stock_bond_regime", "bond_score", "equity_score",
        "cr_state", "r_state", "cb_state", "eq_state", "alt_state", "rb_state",
    ]:
        if column in source:
            diagnostic[column] = source[column].to_numpy()

    centers = _fit_kmeans(
        latent[data.train_mask],
        int(cfg["model"]["kmeans_clusters"]),
        int(cfg["model"]["kmeans_restarts"]),
        selected_seed,
    )
    diagnostic["latent_regime"] = np.square(
        latent[:, None, :] - centers[None, :, :]
    ).sum(axis=2).argmin(axis=1)

    correlation_rows: list[dict[str, Any]] = []
    split_masks = {
        "train": data.train_mask,
        "validation": data.validation_mask,
        "oos": data.oos_mask,
    }
    for split, mask in split_masks.items():
        for dimension in range(latent.shape[1]):
            for concept_position, concept in enumerate(CONCEPTS):
                correlation_rows.append(
                    {
                        "split": split,
                        "latent_dimension": f"z{dimension + 1}",
                        "concept": concept,
                        "spearman": _correlation(
                            latent[mask, dimension], data.concepts[mask, concept_position]
                        ),
                        "assigned": int(assignment[concept]["dimension"] == dimension),
                    }
                )
    correlations = pd.DataFrame(correlation_rows)

    cluster_summary = diagnostic.groupby("latent_regime", as_index=False).agg(
        samples=("date", "size"),
        start_date=("date", "min"),
        end_date=("date", "max"),
        trend_strength=("trend_strength", "mean"),
        volatility_stress=("volatility_stress", "mean"),
        mean_reversion_pressure=("mean_reversion_pressure", "mean"),
        macro_commodity_regime=("macro_commodity_regime", "mean"),
        reconstruction_error=("reconstruction_error", "mean"),
        abstain_rate=("vae_abstain", "mean"),
    )
    anomalies = diagnostic.sort_values("reconstruction_error", ascending=False).head(30).copy()

    transition_rows: list[dict[str, Any]] = []
    oos_start = pd.Timestamp(cfg["diagnostics"]["oos_start"])
    regimes = diagnostic["stock_bond_regime"].astype("string")
    for horizon in [int(value) for value in cfg["diagnostics"]["transition_horizons_days"]]:
        future = regimes.shift(-horizon)
        changed = regimes.ne(future) & regimes.notna() & future.notna()
        eligible = diagnostic["date"].ge(oos_start) & future.notna()
        for group, group_mask in {
            "all": pd.Series(True, index=diagnostic.index),
            "vae_abstain": diagnostic["vae_abstain"],
            "normal": ~diagnostic["vae_abstain"],
        }.items():
            selected = eligible & group_mask
            transition_rows.append(
                {
                    "horizon_days": horizon,
                    "group": group,
                    "samples": int(selected.sum()),
                    "grid_transition_rate": float(changed[selected].mean()) if selected.any() else np.nan,
                    "mean_reconstruction_percentile": float(
                        diagnostic.loc[selected, "reconstruction_error_percentile"].mean()
                    ) if selected.any() else np.nan,
                    "mean_grid_confidence_multiplier": float(
                        diagnostic.loc[selected, "vae_grid_confidence_multiplier"].mean()
                    ) if selected.any() else np.nan,
                    "reconstruction_auc_for_transition": _auc(
                        changed[eligible].to_numpy(),
                        diagnostic.loc[eligible, "reconstruction_error"].to_numpy(),
                    ),
                }
            )
    transition_audit = pd.DataFrame(transition_rows)

    thresholds = pd.DataFrame(
        [
            {
                "error_warning": error_warning,
                "error_anomaly": error_anomaly,
                "error_extreme": error_extreme,
                "drift_warning": drift_warning,
                "drift_anomaly": drift_anomaly,
            }
        ]
    )
    diagnostic.attrs["thresholds"] = thresholds.iloc[0].to_dict()
    diagnostic.attrs["axis_assignment"] = assignment
    diagnostic.attrs["cluster_centers"] = centers.tolist()
    return diagnostic, correlations, cluster_summary, anomalies, transition_audit


def save_plots(
    diagnostic: pd.DataFrame,
    training_log: pd.DataFrame,
    cfg: dict[str, Any],
) -> None:
    assigned = [f"coord_{CONCEPTS[0]}", f"coord_{CONCEPTS[3]}"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    colors = ["trend_strength", "volatility_stress", "reconstruction_error_percentile"]
    titles = ["Trend strength", "Volatility stress", "Reconstruction anomaly"]
    for axis, color, title in zip(axes, colors, titles):
        scatter = axis.scatter(
            diagnostic[assigned[0]], diagnostic[assigned[1]], c=diagnostic[color],
            s=8, alpha=0.65, cmap="viridis", linewidths=0,
        )
        axis.set_title(title)
        axis.set_xlabel("Latent trend coordinate")
        axis.set_ylabel("Latent macro/commodity coordinate")
        fig.colorbar(scatter, ax=axis, shrink=0.82)
    fig.savefig(output_path(cfg, "latent_plot"), dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    group_columns = ["seed"]
    if "model_year" in training_log:
        group_columns.insert(0, "model_year")
    for group_key, group in training_log.groupby(group_columns):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        label = "/".join(str(value) for value in group_key)
        axis.plot(group["epoch"], group["train_loss"], alpha=0.35, label=f"{label} train")
        axis.plot(group["epoch"], group["validation_loss"], label=f"{label} validation")
    axis.set_title("VAE training and validation loss")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.legend(ncol=2, fontsize=8)
    fig.savefig(output_path(cfg, "loss_plot"), dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(14, 4.8), constrained_layout=True)
    axis.plot(diagnostic["date"], diagnostic["reconstruction_error_percentile"], color="#174a7e", lw=1.1)
    axis.axhline(float(cfg["diagnostics"]["warning_quantile"]), color="#e6a700", ls="--", lw=1)
    axis.axhline(float(cfg["diagnostics"]["anomaly_quantile"]), color="#c43c35", ls="--", lw=1)
    axis.fill_between(
        diagnostic["date"], 0, 1,
        where=diagnostic["vae_abstain"].to_numpy(dtype=bool),
        color="#c43c35", alpha=0.10,
    )
    axis.set_ylim(0, 1.02)
    axis.set_title("Walk-forward market-state VAE anomaly percentile")
    axis.set_ylabel("Prior-year reference percentile")
    fig.savefig(output_path(cfg, "timeline_plot"), dpi=180)
    plt.close(fig)


def write_report(
    frame: pd.DataFrame,
    data: WindowData,
    diagnostic: pd.DataFrame,
    correlations: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    transition_audit: pd.DataFrame,
    seed_summary: pd.DataFrame,
    cfg: dict[str, Any],
    selected_seed: int,
) -> None:
    assignment = diagnostic.attrs["axis_assignment"]
    assignment_rows = []
    for concept, details in assignment.items():
        dimension = int(details["dimension"]) + 1
        train_corr = float(details["train_spearman"])
        oos_row = correlations[
            correlations["split"].eq("oos")
            & correlations["latent_dimension"].eq(f"z{dimension}")
            & correlations["concept"].eq(concept)
        ]
        assignment_rows.append(
            {
                "concept": concept,
                "latent_dimension": f"z{dimension}",
                "orientation": "+" if details["sign"] > 0 else "-",
                "train_spearman": train_corr,
                "oos_spearman": float(oos_row["spearman"].iloc[0]) if len(oos_row) else np.nan,
            }
        )
    assignment_table = pd.DataFrame(assignment_rows)
    threshold = diagnostic.attrs["thresholds"]
    oos = diagnostic[diagnostic["date"].ge(pd.Timestamp(cfg["diagnostics"]["oos_start"]))]
    lines = [
        "# 市场状态 VAE 表示层实验",
        "",
        "## 定位",
        "",
        "该模型是无监督市场表示与异常诊断层，不预测价格、不直接生成 Market Views，也不替代股债九宫格。九宫格仍是正式策略第一层；VAE 只输出可合并的置信度、异常和潜在状态坐标。",
        "",
        "## 数据与模型",
        "",
        f"- 数据范围：`{frame['date'].min().date()}` 至 `{frame['date'].max().date()}`。",
        f"- 有效市场特征：{len(data.feature_columns)} 个；窗口：{cfg['data']['window_size_days']} 日；输入维度：{data.values.shape[1]}。",
        f"- 训练窗口：{int(data.train_mask.sum())}；验证窗口：{int(data.validation_mask.sum())}；后续 OOS 窗口：{int(data.oos_mask.sum())}。",
        f"- 网络：`{data.values.shape[1]} -> {cfg['model']['hidden_dim']} -> {int(cfg['model']['hidden_dim']) // 2} -> {cfg['model']['latent_dim']}`，β={cfg['model']['beta_kl']}。",
        f"- 选择种子：{selected_seed}；最多 {cfg['model']['max_epochs']} epoch，使用验证重构损失早停。",
        "- Winsor、缺失值中位数、均值方差、聚类中心、潜在轴方向和异常阈值全部只在训练期拟合。",
        "",
        "## 潜在轴解释",
        "",
        assignment_table.to_markdown(index=False),
        "",
        "潜在维度的经济命名属于训练后解释，不是因果结论。若 OOS 相关性明显衰减或反向，该坐标只能保留原始 z 编号，不能继续使用经济名称。",
        "",
        "## 异常与九宫格置信度",
        "",
        f"- 训练期重构误差 warning/anomaly 阈值：{threshold['error_warning']:.6f}/{threshold['error_anomaly']:.6f}。",
        f"- OOS warning 比例：{oos['vae_warning'].mean():.2%}；abstain 比例：{oos['vae_abstain'].mean():.2%}。",
        "- `vae_grid_confidence_multiplier` 范围为 0.40 至 1.00，仅建议缩小观点或触发人工复核，不允许放大九宫格观点。",
        "- 本实验没有把该 multiplier 写入正式权重、EP/skfolio 或 PPO。",
        "",
        "## 九宫格状态迁移审计",
        "",
        transition_audit.to_markdown(index=False),
        "",
        "该表使用未来九宫格变化仅做 OOS 诊断，不参与 VAE 训练、模型选择或阈值拟合。AUC 接近 0.5 表示重构异常对状态切换没有区分能力。",
        "",
        "## 聚类描述",
        "",
        cluster_summary.to_markdown(index=False),
        "",
        "KMeans 只在训练期潜在坐标上拟合，编号没有天然经济顺序，不作为交易规则。",
        "",
        "## 种子与早停",
        "",
        seed_summary.to_markdown(index=False),
        "",
        "## 输出与接入边界",
        "",
        "- 日频诊断：`data_processed/market_state_vae_diagnostics_daily.parquet`。",
        "- 可用于状态卡的字段：`vae_state_confidence`、`vae_grid_confidence_multiplier`、`vae_warning`、`vae_abstain`、`latent_regime`。",
        "- 正式接入前需要在新时间段验证异常分数与九宫格切换/风险恶化的关系，并单独做交易成本后组合闸门。",
    ]
    output_path(cfg, "report").write_text("\n".join(lines), encoding="utf-8")


def _transition_audit_from_diagnostics(
    diagnostic: pd.DataFrame, cfg: dict[str, Any]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    regimes = diagnostic["stock_bond_regime"].astype("string")
    audit_start = pd.Timestamp(cfg["diagnostics"]["oos_start"])
    for horizon in [int(value) for value in cfg["diagnostics"]["transition_horizons_days"]]:
        future = regimes.shift(-horizon)
        changed = regimes.ne(future) & regimes.notna() & future.notna()
        eligible = diagnostic["date"].ge(audit_start) & future.notna()
        auc = _auc(
            changed[eligible].to_numpy(),
            diagnostic.loc[eligible, "reconstruction_error_percentile"].to_numpy(),
        )
        for group, mask in {
            "all": pd.Series(True, index=diagnostic.index),
            "vae_abstain": diagnostic["vae_abstain"],
            "normal": ~diagnostic["vae_abstain"],
        }.items():
            selected = eligible & mask
            rows.append(
                {
                    "horizon_days": horizon,
                    "group": group,
                    "samples": int(selected.sum()),
                    "grid_transition_rate": (
                        float(changed[selected].mean()) if selected.any() else np.nan
                    ),
                    "mean_reconstruction_percentile": (
                        float(diagnostic.loc[selected, "reconstruction_error_percentile"].mean())
                        if selected.any() else np.nan
                    ),
                    "mean_grid_confidence_multiplier": (
                        float(diagnostic.loc[selected, "vae_grid_confidence_multiplier"].mean())
                        if selected.any() else np.nan
                    ),
                    "anomaly_percentile_auc_for_transition": auc,
                }
            )
    return pd.DataFrame(rows)


def _checkpoint_payload(
    cfg: dict[str, Any],
    fold: dict[str, Any],
    model: FinanceVAE,
    data: WindowData,
    selected_seed: int,
    assignment: dict[str, Any],
    diagnostic: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "version": cfg["version"],
        "model_year": int(fold["model_year"]),
        "fit_end": str(fold["fit_end"]),
        "tune_end": str(fold["tune_end"]),
        "evaluation_end": str(fold["evaluation_end"]),
        "threshold_reference": "immediately preceding tune period",
        "state_dict": model.state_dict(),
        "input_dim": data.values.shape[1],
        "feature_columns": data.feature_columns,
        "window_size_days": int(cfg["data"]["window_size_days"]),
        "selected_seed": selected_seed,
        "preprocessing": {
            "lower": data.lower,
            "upper": data.upper,
            "mean": data.mean,
            "std": data.std,
        },
        "axis_assignment": assignment,
        "thresholds": diagnostic.attrs["thresholds"],
        "cluster_centers": diagnostic.attrs["cluster_centers"],
        "model_config": cfg["model"],
    }


def _write_walk_forward_report(
    frame: pd.DataFrame,
    feature_columns: list[str],
    diagnostics: pd.DataFrame,
    fold_summary: pd.DataFrame,
    correlations: pd.DataFrame,
    transition_audit: pd.DataFrame,
    cfg: dict[str, Any],
) -> None:
    retained_macro_sources = []
    omitted_macro_sources = []
    for source in cfg["data"]["macro_features"]:
        if any(column.startswith(f"macro_{source}_") for column in feature_columns):
            retained_macro_sources.append(source)
        else:
            omitted_macro_sources.append(source)
    assigned = correlations.loc[correlations["assigned"].eq(1)].copy()
    assigned = assigned[
        ["model_year", "split", "latent_dimension", "concept", "spearman"]
    ]
    oos = diagnostics[diagnostics["date"].ge(pd.Timestamp(cfg["diagnostics"]["oos_start"]))]
    lines = [
        "# 市场状态 VAE：逐年样本外实验与接入边界",
        "",
        "## 一句话定位",
        "",
        "风险平价底仓、股债九宫格和各资产状态卡仍是正式策略骨架。VAE 只把最近 20 个交易日的多资产行为压缩为潜在坐标，输出陌生度、稳定度和九宫格置信度折扣；它不预测涨跌、不生成 Market Views、不直接给仓位。",
        "",
        "## Point-in-Time 流程",
        "",
        "1. 每个模型年先冻结可见数据边界。`fit` 期只用于拟合预处理、VAE 参数、潜在轴解释和聚类中心。",
        "2. 紧邻模型年的上一完整年度是 `tune` 期，只用于早停、种子选择和异常阈值校准。",
        "3. 下一年度才是 evaluation；该年度任何数值都不参与该年度模型训练、模型选择或阈值拟合。",
        "4. 三个年度模型的 evaluation 输出按日期拼接，构成 2024 至最新日的逐年样本外诊断。",
        "",
        "## 输入与模型",
        "",
        f"- 原始日频范围：`{frame['date'].min().date()}` 至 `{frame['date'].max().date()}`。",
        f"- 输入特征：{len(feature_columns)} 个；滚动窗口：{cfg['data']['window_size_days']} 日；单样本输入维度：{len(feature_columns) * int(cfg['data']['window_size_days'])}。",
        "- 内容包括五个正式袖套的 1/5/20/60 日收益或动量、20/60 日波动、60 日回撤和 20 日均线偏离。",
        f"- 实际保留的宏观/状态源：{', '.join(retained_macro_sources)}。",
        f"- 因最早 fit 期覆盖率不足而未进入模型的候选源：{', '.join(omitted_macro_sources) if omitted_macro_sources else '无'}。",
        f"- 网络：`{len(feature_columns) * int(cfg['data']['window_size_days'])} -> {cfg['model']['hidden_dim']} -> {int(cfg['model']['hidden_dim']) // 2} -> {cfg['model']['latent_dim']}`；beta-VAE 的 `beta={cfg['model']['beta_kl']}`。",
        f"- 每折最多 {cfg['model']['max_epochs']} epoch，最少 {cfg['model']['minimum_epochs']} epoch，patience={cfg['model']['patience']}；多种子仅按当时 tune loss 选择。",
        "- Winsor、缺失值、均值方差均只在 fit 期拟合。宏观滚动指标只使用当日及过去值。",
        "",
        "## Walk-Forward 结果",
        "",
        fold_summary.to_markdown(index=False),
        "",
        "## 潜在轴稳定性",
        "",
        assigned.to_markdown(index=False),
        "",
        "潜在轴名称是训练后的相关性解释，不是因果关系。若 evaluation 相关性衰减或反向，应保留原始 z 编号并降低解释权重。",
        "",
        "## 九宫格切换诊断",
        "",
        transition_audit.to_markdown(index=False),
        "",
        "该审计中的未来九宫格变化只用于模型完成后的检验，从未进入训练或阈值校准。AUC 接近 0.5 代表 VAE 异常度不能预判九宫格切换。",
        "",
        "## 当前结论",
        "",
        f"- 2025 年以来 warning 比例：{oos['vae_warning'].mean():.2%}；abstain 比例：{oos['vae_abstain'].mean():.2%}。",
        "- `vae_grid_confidence_multiplier` 只能把九宫格观点缩小到 0.40–1.00，不能放大观点。",
        "- 只有当 evaluation 的潜在轴解释稳定、异常率不过度膨胀、且异常度对状态切换或尾部风险有稳定增量时，才考虑进入正式回测。当前版本仍为研究诊断层。",
        "",
        "## 输出",
        "",
        "- 日频逐年样本外诊断：`data_processed/market_state_vae_diagnostics_daily.parquet`。",
        "- 分年度模型：`data_processed/market_state_vae_models/`。",
        "- 可接入状态卡字段：`vae_state_confidence`、`vae_grid_confidence_multiplier`、`vae_warning`、`vae_abstain`、`latent_regime`。",
    ]
    output_path(cfg, "report").write_text("\n".join(lines), encoding="utf-8")


def _run_walk_forward(cfg: dict[str, Any]) -> dict[str, Any]:
    for key in [
        "diagnostics", "training_log", "seed_summary", "axis_correlations",
        "cluster_summary", "anomaly_dates", "transition_audit", "model",
        "latent_plot", "loss_plot", "timeline_plot", "report", "windows_npz",
    ]:
        output_path(cfg, key).parent.mkdir(parents=True, exist_ok=True)
    model_dir = output_path(cfg, "model_dir")
    model_dir.mkdir(parents=True, exist_ok=True)

    frame, feature_columns = build_feature_frame(cfg)
    diagnostics_parts: list[pd.DataFrame] = []
    training_parts: list[pd.DataFrame] = []
    seed_parts: list[pd.DataFrame] = []
    correlation_parts: list[pd.DataFrame] = []
    cluster_parts: list[pd.DataFrame] = []
    anomaly_parts: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    latest_payload: dict[str, Any] | None = None

    for fold in cfg["walk_forward"]["folds"]:
        fold_cfg = copy.deepcopy(cfg)
        fold_cfg["data"]["train_end"] = fold["fit_end"]
        fold_cfg["data"]["validation_end"] = fold["tune_end"]
        data = build_windows(frame, feature_columns, fold_cfg)
        evaluation_end = np.datetime64(pd.Timestamp(fold["evaluation_end"]))
        data.oos_mask = data.oos_mask & (data.dates <= evaluation_end)
        if not data.train_mask.any() or not data.validation_mask.any() or not data.oos_mask.any():
            raise ValueError(f"Empty walk-forward split for model year {fold['model_year']}")

        model, log, seed_summary, selected_seed = train_models(data, fold_cfg)
        latent, logvar, error = deterministic_inference(
            model, data.values, int(cfg["model"]["batch_size"])
        )
        assignment = assign_axes(latent, data.concepts, data.train_mask)
        diagnostic, correlations, _, _, _ = build_diagnostics(
            frame, data, latent, logvar, error, assignment, fold_cfg,
            selected_seed, threshold_mask=data.validation_mask,
        )
        evaluation_mask = data.oos_mask
        evaluation = diagnostic.loc[evaluation_mask].copy()
        evaluation["model_year"] = int(fold["model_year"])
        evaluation["fit_end"] = str(fold["fit_end"])
        evaluation["tune_end"] = str(fold["tune_end"])
        evaluation["evaluation_end"] = str(fold["evaluation_end"])
        evaluation["threshold_reference"] = "tune_period"
        diagnostics_parts.append(evaluation)

        log.insert(0, "model_year", int(fold["model_year"]))
        seed_summary.insert(0, "model_year", int(fold["model_year"]))
        correlations.insert(0, "model_year", int(fold["model_year"]))
        training_parts.append(log)
        seed_parts.append(seed_summary)
        correlation_parts.append(correlations)

        evaluation_clusters = evaluation.groupby("latent_regime", as_index=False).agg(
            samples=("date", "size"),
            start_date=("date", "min"),
            end_date=("date", "max"),
            trend_strength=("trend_strength", "mean"),
            volatility_stress=("volatility_stress", "mean"),
            reconstruction_error=("reconstruction_error", "mean"),
            abstain_rate=("vae_abstain", "mean"),
        )
        evaluation_clusters.insert(0, "model_year", int(fold["model_year"]))
        cluster_parts.append(evaluation_clusters)
        anomaly_parts.append(evaluation.nlargest(10, "reconstruction_error_percentile"))

        selected_row = seed_summary.loc[seed_summary["seed"].eq(selected_seed)].iloc[0]
        fold_rows.append(
            {
                "model_year": int(fold["model_year"]),
                "fit_end": fold["fit_end"],
                "tune_end": fold["tune_end"],
                "evaluation_end": fold["evaluation_end"],
                "fit_windows": int(data.train_mask.sum()),
                "tune_windows": int(data.validation_mask.sum()),
                "evaluation_windows": int(evaluation_mask.sum()),
                "selected_seed": selected_seed,
                "best_epoch": int(selected_row["best_epoch"]),
                "best_tune_loss": float(selected_row["best_validation_loss"]),
                "warning_rate": float(evaluation["vae_warning"].mean()),
                "abstain_rate": float(evaluation["vae_abstain"].mean()),
                "median_confidence": float(evaluation["vae_state_confidence"].median()),
            }
        )
        payload = _checkpoint_payload(
            cfg, fold, model, data, selected_seed, assignment, diagnostic
        )
        torch.save(payload, model_dir / f"market_state_vae_{fold['model_year']}.pt")
        latest_payload = payload

    diagnostics = pd.concat(diagnostics_parts, ignore_index=True).sort_values("date")
    if diagnostics["date"].duplicated().any():
        raise AssertionError("Walk-forward diagnostics contain duplicate dates")
    training_log = pd.concat(training_parts, ignore_index=True)
    seed_summary = pd.concat(seed_parts, ignore_index=True)
    correlations = pd.concat(correlation_parts, ignore_index=True)
    clusters = pd.concat(cluster_parts, ignore_index=True)
    anomalies = pd.concat(anomaly_parts, ignore_index=True).sort_values(
        ["model_year", "reconstruction_error_percentile"], ascending=[True, False]
    )
    fold_summary = pd.DataFrame(fold_rows)
    transition_audit = _transition_audit_from_diagnostics(diagnostics, cfg)

    diagnostics.to_parquet(output_path(cfg, "diagnostics"), index=False)
    training_log.to_csv(output_path(cfg, "training_log"), index=False, encoding="utf-8-sig")
    seed_summary.to_csv(output_path(cfg, "seed_summary"), index=False, encoding="utf-8-sig")
    correlations.to_csv(output_path(cfg, "axis_correlations"), index=False, encoding="utf-8-sig")
    clusters.to_csv(output_path(cfg, "cluster_summary"), index=False, encoding="utf-8-sig")
    anomalies.to_csv(output_path(cfg, "anomaly_dates"), index=False, encoding="utf-8-sig")
    transition_audit.to_csv(output_path(cfg, "transition_audit"), index=False, encoding="utf-8-sig")
    fold_summary.to_csv(
        output_path(cfg, "seed_summary").with_name("market_state_vae_fold_summary.csv"),
        index=False, encoding="utf-8-sig",
    )
    if latest_payload is None:
        raise RuntimeError("No walk-forward model was trained")
    torch.save(latest_payload, output_path(cfg, "model"))
    save_plots(diagnostics, training_log, cfg)
    _write_walk_forward_report(
        frame, feature_columns, diagnostics, fold_summary, correlations,
        transition_audit, cfg,
    )
    oos = diagnostics[diagnostics["date"].ge(pd.Timestamp(cfg["diagnostics"]["oos_start"]))]
    return {
        "mode": "annual_expanding_walk_forward",
        "model_years": fold_summary["model_year"].tolist(),
        "rows": len(diagnostics),
        "features": len(feature_columns),
        "input_dim": len(feature_columns) * int(cfg["data"]["window_size_days"]),
        "oos_warning_rate": float(oos["vae_warning"].mean()),
        "oos_abstain_rate": float(oos["vae_abstain"].mean()),
        "report": str(output_path(cfg, "report").relative_to(ROOT)),
    }


def run_pipeline(cfg: dict[str, Any]) -> dict[str, Any]:
    if cfg.get("walk_forward", {}).get("enabled", False):
        return _run_walk_forward(cfg)
    frame, feature_columns = build_feature_frame(cfg)
    data = build_windows(frame, feature_columns, cfg)
    model, training_log, seed_summary, selected_seed = train_models(data, cfg)
    latent, logvar, error = deterministic_inference(
        model, data.values, int(cfg["model"]["batch_size"])
    )
    assignment = assign_axes(latent, data.concepts, data.train_mask)
    diagnostic, correlations, cluster_summary, anomalies, transition_audit = build_diagnostics(
        frame, data, latent, logvar, error, assignment, cfg, selected_seed
    )
    for key in [
        "diagnostics", "training_log", "seed_summary", "axis_correlations",
        "cluster_summary", "anomaly_dates", "transition_audit", "model",
        "latent_plot", "loss_plot", "timeline_plot", "report",
    ]:
        output_path(cfg, key).parent.mkdir(parents=True, exist_ok=True)
    diagnostic.to_parquet(output_path(cfg, "diagnostics"), index=False)
    training_log.to_csv(output_path(cfg, "training_log"), index=False, encoding="utf-8-sig")
    seed_summary.to_csv(output_path(cfg, "seed_summary"), index=False, encoding="utf-8-sig")
    correlations.to_csv(output_path(cfg, "axis_correlations"), index=False, encoding="utf-8-sig")
    cluster_summary.to_csv(output_path(cfg, "cluster_summary"), index=False, encoding="utf-8-sig")
    anomalies.to_csv(output_path(cfg, "anomaly_dates"), index=False, encoding="utf-8-sig")
    transition_audit.to_csv(output_path(cfg, "transition_audit"), index=False, encoding="utf-8-sig")
    torch.save(
        {
            "version": cfg["version"],
            "state_dict": model.state_dict(),
            "input_dim": data.values.shape[1],
            "feature_columns": data.feature_columns,
            "window_size_days": int(cfg["data"]["window_size_days"]),
            "selected_seed": selected_seed,
            "preprocessing": {
                "lower": data.lower,
                "upper": data.upper,
                "mean": data.mean,
                "std": data.std,
            },
            "axis_assignment": assignment,
            "thresholds": diagnostic.attrs["thresholds"],
            "cluster_centers": diagnostic.attrs["cluster_centers"],
            "model_config": cfg["model"],
        },
        output_path(cfg, "model"),
    )
    save_plots(diagnostic, training_log, cfg)
    write_report(
        frame, data, diagnostic, correlations, cluster_summary,
        transition_audit, seed_summary, cfg, selected_seed,
    )
    return {
        "selected_seed": selected_seed,
        "rows": len(diagnostic),
        "features": len(feature_columns),
        "input_dim": data.values.shape[1],
        "oos_abstain_rate": float(
            diagnostic.loc[
                diagnostic["date"].ge(pd.Timestamp(cfg["diagnostics"]["oos_start"])),
                "vae_abstain",
            ].mean()
        ),
        "report": str(output_path(cfg, "report").relative_to(ROOT)),
    }

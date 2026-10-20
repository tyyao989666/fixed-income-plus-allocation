from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import Dataset

from src.itransformer_daily_signal import load_daily_config, load_daily_returns_with_proxy
from src.itransformer_daily_timing_v2 import preferred_daily_sleeve_returns
from src.itransformer_signal import RISK_SLEEVES
from src.point_in_time import _rolling_robust_series


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "hierarchical_asset_itransformer_v7.yaml"
STATE_MAPS = {
    "cr_state": {"CR1": 1.0, "CR2": 0.5, "CR3": -0.5, "CR4": -1.0, "CR5": -1.0},
    "r_state": {"R1": 1.0, "R2": 0.0, "R3": -0.3, "R4": -1.0, "R5": -1.0},
    "cb_state": {"CB1": 1.0, "CB2": 0.0, "CB3": -0.3, "CB4": -1.0, "CB5": -1.0},
    "eq_state": {"EQ1": 1.0, "EQ2": 0.0, "EQ3": -0.3, "EQ4": -1.0, "EQ5": -1.0},
    "alt_state": {"ALT1": 1.0, "ALT2": 0.0, "ALT3": -0.5, "ALT4": -1.0},
    "rb_state": {"RB1": 1.0, "RB2": 0.3, "RB3": -0.5, "RB4": -1.0},
}
STOCK_BOND_REGIMES = [
    f"bond_{bond}_equity_{equity}"
    for bond in ["strong", "neutral", "weak"]
    for equity in ["strong", "neutral", "weak"]
]
BASE_LOCAL_FEATURES = [
    "ret_1d",
    "mom_5d",
    "mom_20d",
    "mom_60d",
    "mom_120d",
    "vol_20d",
    "vol_60d",
    "dd_60d",
    "overnight_ret",
    "intraday_ret",
    "range_pct",
    "close_position",
    "log_vol",
    "log_amount",
    "card_score",
    "card_rank",
    "card_state",
    "card_overweight_eligible",
]
EXPLICIT_ELIGIBILITY_FEATURES = [
    "card_data_available",
    "card_holding_eligible",
]
PASSTHROUGH_LOCAL = {
    "card_rank",
    "card_state",
    "card_data_available",
    "card_holding_eligible",
    "card_overweight_eligible",
}


def _local_feature_names(cfg: dict[str, Any]) -> list[str]:
    features = list(BASE_LOCAL_FEATURES)
    for feature in cfg["data"].get("extra_local_features", []):
        if feature not in features:
            features.append(str(feature))
    if bool(cfg["data"].get("include_explicit_eligibility_features", False)):
        features.extend(EXPLICIT_ELIGIBILITY_FEATURES)
    return features


def _deep_merge_config(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    output = dict(base)
    for key, value in override.items():
        if (
            key in output
            and isinstance(output[key], dict)
            and isinstance(value, dict)
        ):
            output[key] = _deep_merge_config(output[key], value)
        else:
            output[key] = value
    return output


def load_config(path: Path | None = None) -> dict[str, Any]:
    source = (path or DEFAULT_CONFIG).resolve()
    with source.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    base_reference = config.pop("base_config", None)
    if base_reference is None:
        return config
    base_path = Path(base_reference)
    if not base_path.is_absolute():
        base_path = source.parent / base_path
    return _deep_merge_config(load_config(base_path), config)


def project_path(cfg: dict[str, Any], key: str) -> Path:
    path = ROOT / cfg["outputs"][key]
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _max_drawdown(values: np.ndarray) -> float:
    clean = np.nan_to_num(values, nan=0.0)
    nav = np.cumprod(1.0 + clean)
    if not len(nav):
        return np.nan
    return float(np.min(nav / np.maximum.accumulate(nav) - 1.0))


def _asset_state_frame(cards: pd.DataFrame, dates: pd.DatetimeIndex, assets: list[str]) -> dict[str, pd.DataFrame]:
    result = {
        "card_score": pd.DataFrame(np.nan, index=dates, columns=assets),
        "card_rank": pd.DataFrame(np.nan, index=dates, columns=assets),
        "card_state": pd.DataFrame(np.nan, index=dates, columns=assets),
        "card_data_available": pd.DataFrame(np.nan, index=dates, columns=assets),
        "card_holding_eligible": pd.DataFrame(np.nan, index=dates, columns=assets),
        "card_overweight_eligible": pd.DataFrame(np.nan, index=dates, columns=assets),
    }
    state_value = {"negative": -1.0, "neutral": 0.0, "positive": 1.0}
    cards = cards.set_index("date").reindex(dates)
    for date, raw in cards["asset_detail_states"].items():
        if not isinstance(raw, str) or not raw:
            continue
        try:
            payload = json.loads(raw).get("assets", {})
        except (json.JSONDecodeError, AttributeError):
            continue
        for asset, detail in payload.items():
            if asset not in result["card_score"].columns:
                continue
            result["card_score"].at[date, asset] = pd.to_numeric(detail.get("score"), errors="coerce")
            result["card_rank"].at[date, asset] = pd.to_numeric(detail.get("module_rank"), errors="coerce")
            result["card_state"].at[date, asset] = state_value.get(str(detail.get("state", "neutral")), 0.0)
            context = detail.get("domain_context", {}) or {}
            result["card_data_available"].at[date, asset] = float(bool(context.get("data_available", False)))
            result["card_holding_eligible"].at[date, asset] = float(bool(context.get("holding_eligible", False)))
            result["card_overweight_eligible"].at[date, asset] = float(bool(context.get("overweight_eligible", False)))
    return result


def _causal_scale_frame(frame: pd.DataFrame, window: int, min_periods: int, clip: float) -> pd.DataFrame:
    output = pd.DataFrame(index=frame.index, columns=frame.columns, dtype=float)
    for column in frame:
        output[column] = _rolling_robust_series(frame[column], window, min_periods, clip)
    return output


def _future_compounded(
    returns: pd.DataFrame, horizon: int, minimum_coverage: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    shifted = returns.shift(-1)
    future = (1.0 + shifted.fillna(0.0)).rolling(horizon, min_periods=horizon).apply(np.prod, raw=True).shift(-(horizon - 1)) - 1.0
    count = shifted.notna().rolling(horizon, min_periods=1).sum().shift(-(horizon - 1))
    return future, count.ge(math.ceil(horizon * minimum_coverage))


def _cross_section_demean(values: pd.DataFrame, groups: pd.Series | None = None) -> pd.DataFrame:
    output = values.copy()
    if groups is None:
        return output.sub(output.mean(axis=1), axis=0)
    for group in groups.dropna().unique():
        columns = groups.index[groups.eq(group)]
        output.loc[:, columns] = output.loc[:, columns].sub(output.loc[:, columns].mean(axis=1), axis=0)
    return output


def _rowwise_rank_score(values: pd.DataFrame) -> pd.DataFrame:
    """Convert a causal cross-sectional ordering to a stable [-1, 1] score."""
    ranks = values.rank(axis=1, method="average", pct=True)
    scores = (ranks - 0.5) * 2.0
    scores = scores.sub(scores.mean(axis=1), axis=0)
    inactive = values.abs().sum(axis=1).le(1e-12)
    scores.loc[inactive] = 0.0
    return scores.fillna(0.0)


def _group_rank_score(values: pd.DataFrame, groups: pd.Series | None) -> pd.DataFrame:
    """Rank across sleeves or within each sleeve while preserving the input layout."""
    if groups is None:
        return _rowwise_rank_score(values)
    output = pd.DataFrame(np.nan, index=values.index, columns=values.columns, dtype=float)
    for group in groups.dropna().unique():
        columns = groups.index[groups.eq(group)]
        output.loc[:, columns] = _rowwise_rank_score(values.loc[:, columns])
    return output


def _rowwise_relative_utility(
    values: pd.DataFrame,
    clip: float,
) -> pd.DataFrame:
    """Continuous cross-sectional relative score with stable small-universe scaling."""
    centered = values.sub(values.mean(axis=1), axis=0)
    scale = centered.pow(2).mean(axis=1).pow(0.5)
    fallback = centered.abs().mean(axis=1) * math.sqrt(math.pi / 2.0)
    scale = scale.where(scale.gt(1e-8), fallback).where(lambda x: x.gt(1e-8))
    utility = centered.div(scale, axis=0).clip(-float(clip), float(clip))
    return utility.sub(utility.mean(axis=1), axis=0)


def _group_relative_utility(
    values: pd.DataFrame,
    groups: pd.Series | None,
    clip: float,
) -> pd.DataFrame:
    if groups is None:
        return _rowwise_relative_utility(values, clip)
    output = pd.DataFrame(np.nan, index=values.index, columns=values.columns, dtype=float)
    for group in groups.dropna().unique():
        columns = groups.index[groups.eq(group)]
        output.loc[:, columns] = _rowwise_relative_utility(
            values.loc[:, columns], clip
        )
    return output


def _causal_momentum_utility_baseline(
    returns: pd.DataFrame,
    groups: pd.Series | None,
    windows: list[int],
    weights: list[float],
    clip: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(windows) != len(weights) or not windows:
        raise ValueError(
            "momentum baseline windows and weights must have the same non-zero length"
        )
    raw_weights = np.asarray(weights, dtype=float)
    if not np.isfinite(raw_weights).all() or raw_weights.sum() <= 0.0:
        raise ValueError(
            "momentum baseline weights must be finite and sum to a positive value"
        )
    raw_weights /= raw_weights.sum()
    weighted_score = pd.DataFrame(
        0.0, index=returns.index, columns=returns.columns
    )
    weighted_available = pd.DataFrame(
        0.0, index=returns.index, columns=returns.columns
    )
    for window, weight in zip(windows, raw_weights):
        minimum = max(3, int(math.ceil(int(window) * 0.50)))
        trailing = (1.0 + returns.fillna(0.0)).rolling(
            int(window), min_periods=minimum
        ).apply(np.prod, raw=True) - 1.0
        available = returns.rolling(int(window), min_periods=minimum).count().ge(
            minimum
        )
        score = _group_relative_utility(trailing, groups, clip)
        weighted_score += float(weight) * score.fillna(0.0) * available
        weighted_available += float(weight) * available.astype(float)
    blended = weighted_score.div(weighted_available.replace(0.0, np.nan))
    return (
        _group_relative_utility(blended, groups, clip),
        weighted_available.gt(0.0),
    )


def _causal_momentum_mean_baseline(
    returns: pd.DataFrame,
    groups: pd.Series | None,
    windows: list[int],
    weights: list[float],
    clip: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate an annualized relative mean without future cross-sectional scaling."""
    if len(windows) != len(weights) or not windows:
        raise ValueError(
            "momentum baseline windows and weights must have the same non-zero length"
        )
    raw_weights = np.asarray(weights, dtype=float)
    if not np.isfinite(raw_weights).all() or raw_weights.sum() <= 0.0:
        raise ValueError(
            "momentum baseline weights must be finite and sum to a positive value"
        )
    raw_weights /= raw_weights.sum()
    log_returns = np.log1p(returns.clip(lower=-0.99))
    weighted_score = pd.DataFrame(
        0.0, index=returns.index, columns=returns.columns
    )
    weighted_available = pd.DataFrame(
        0.0, index=returns.index, columns=returns.columns
    )
    for window, weight in zip(windows, raw_weights):
        minimum = max(3, int(math.ceil(int(window) * 0.50)))
        annualized = (
            log_returns.rolling(int(window), min_periods=minimum).mean() * 252.0
        )
        available = annualized.notna()
        relative = _cross_section_demean(annualized, groups).clip(
            -float(clip), float(clip)
        )
        weighted_score += float(weight) * relative.fillna(0.0) * available
        weighted_available += float(weight) * available.astype(float)
    blended = weighted_score.div(weighted_available.replace(0.0, np.nan))
    blended = _cross_section_demean(blended, groups).clip(
        -float(clip), float(clip)
    )
    return blended, weighted_available.gt(0.0)


def _causal_momentum_rank_baseline(
    returns: pd.DataFrame,
    groups: pd.Series | None,
    windows: list[int],
    weights: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(windows) != len(weights) or not windows:
        raise ValueError("momentum baseline windows and weights must have the same non-zero length")
    raw_weights = np.asarray(weights, dtype=float)
    if not np.isfinite(raw_weights).all() or raw_weights.sum() <= 0.0:
        raise ValueError("momentum baseline weights must be finite and sum to a positive value")
    raw_weights = raw_weights / raw_weights.sum()
    weighted_score = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)
    weighted_available = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)
    for window, weight in zip(windows, raw_weights):
        minimum = max(3, int(math.ceil(int(window) * 0.50)))
        trailing = returns.rolling(int(window), min_periods=minimum).sum()
        available = trailing.notna().astype(float)
        weighted_score += float(weight) * _group_rank_score(trailing, groups) * available
        weighted_available += float(weight) * available
    blended = weighted_score.div(weighted_available.replace(0.0, np.nan))
    return _group_rank_score(blended, groups), weighted_available.gt(0.0)


def _causal_rank_signal_experts(
    returns: pd.DataFrame,
    groups: pd.Series | None,
    windows: list[int],
    reversal_windows: list[int],
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Create point-in-time signal experts without using a future observation."""
    values: dict[str, pd.DataFrame] = {}
    masks: dict[str, pd.DataFrame] = {}
    momentum: dict[int, pd.DataFrame] = {}
    momentum_masks: dict[int, pd.DataFrame] = {}
    for window in windows:
        minimum = max(3, int(math.ceil(int(window) * 0.50)))
        trailing = returns.rolling(int(window), min_periods=minimum).sum()
        score = _group_rank_score(trailing, groups)
        available = trailing.notna()
        momentum[int(window)] = score
        momentum_masks[int(window)] = available
        values[f"momentum_{int(window)}d"] = score
        masks[f"momentum_{int(window)}d"] = available
    for window in reversal_windows:
        window = int(window)
        if window not in momentum:
            minimum = max(3, int(math.ceil(window * 0.50)))
            trailing = returns.rolling(window, min_periods=minimum).sum()
            momentum[window] = _group_rank_score(trailing, groups)
            momentum_masks[window] = trailing.notna()
        values[f"reversal_{window}d"] = -momentum[window]
        masks[f"reversal_{window}d"] = momentum_masks[window]
    return values, masks


def _causal_blended_signal_experts(
    returns: pd.DataFrame,
    groups: pd.Series | None,
    definitions: list[dict[str, Any]],
    target_mode: str,
    clip: float,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Build a small set of scale-compatible point-in-time mean-view experts.

    Each expert is a weighted blend of trailing compounded returns ending at
    the as-of date. The final cross-sectional normalization matches the label
    family, so routing is driven by ordering rather than arbitrary scale.
    """
    values: dict[str, pd.DataFrame] = {}
    masks: dict[str, pd.DataFrame] = {}
    for definition in definitions:
        name = str(definition.get("name", "")).strip()
        if not name or name == "fixed_causal_baseline":
            raise ValueError("Every signal expert needs a unique non-baseline name")
        if name in values:
            raise ValueError(f"Duplicate signal expert name: {name}")
        windows = [int(value) for value in definition.get("windows_days", [])]
        weights = [float(value) for value in definition.get("weights", [])]
        direction = float(definition.get("direction", 1.0))
        if direction not in {-1.0, 1.0}:
            raise ValueError(f"Expert {name} direction must be +1 or -1")
        if target_mode == "relative_return_utility":
            score, available = _causal_momentum_utility_baseline(
                returns, groups, windows, weights, float(clip)
            )
            score = _group_relative_utility(direction * score, groups, float(clip))
        elif target_mode == "relative_mean_return":
            score, available = _causal_momentum_mean_baseline(
                returns, groups, windows, weights, float(clip)
            )
            score = _cross_section_demean(direction * score, groups).clip(
                -float(clip), float(clip)
            )
        elif target_mode == "relative_return_rank":
            score, available = _causal_momentum_rank_baseline(
                returns, groups, windows, weights
            )
            score = _group_rank_score(direction * score, groups)
        else:
            raise ValueError(
                "Blended mean-view experts require a relative-return target mode"
            )
        values[name] = score.fillna(0.0)
        masks[name] = available.fillna(False)
    return values, masks


def _normalize_expert_frame(
    values: pd.DataFrame,
    groups: pd.Series | None,
    target_mode: str,
    clip: float,
) -> pd.DataFrame:
    if target_mode == "relative_return_utility":
        return _group_relative_utility(values, groups, float(clip)).fillna(0.0)
    if target_mode == "relative_mean_return":
        return _cross_section_demean(values, groups).clip(
            -float(clip), float(clip)
        ).fillna(0.0)
    if target_mode == "relative_return_rank":
        return _group_rank_score(values, groups).fillna(0.0)
    raise ValueError("Expert normalization requires a relative-return target mode")


def _single_date_group_rank_ic(
    prediction: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    groups: np.ndarray | None,
) -> float:
    values: list[float] = []
    group_values = [None] if groups is None else np.unique(groups).tolist()
    for group in group_values:
        selected = valid.astype(bool).copy()
        if group is not None:
            selected &= groups == group
        if selected.sum() < 2:
            continue
        prediction_rank = pd.Series(prediction[selected]).rank(
            method="average"
        ).to_numpy(dtype=float)
        target_rank = pd.Series(target[selected]).rank(
            method="average"
        ).to_numpy(dtype=float)
        if np.std(prediction_rank) <= 1e-12 or np.std(target_rank) <= 1e-12:
            continue
        values.append(float(np.corrcoef(prediction_rank, target_rank)[0, 1]))
    return float(np.mean(values)) if values else np.nan


def _causal_expert_reliability_features(
    experts: np.ndarray,
    expert_masks: np.ndarray,
    target: np.ndarray,
    target_mask: np.ndarray,
    label_end: np.ndarray,
    dates: pd.DatetimeIndex,
    groups: np.ndarray | None,
    expert_names: list[str],
    horizons: list[int],
    prefix: str,
    halflife_days: float,
    minimum_observations: int,
    full_confidence_observations: int,
    availability_lag_days: int,
) -> tuple[np.ndarray, list[str]]:
    """Expose only matured historical expert edges to the forecasting window."""
    if experts.shape[:3] != target.shape or experts.shape != expert_masks.shape:
        raise ValueError(f"{prefix} expert reliability tensor shapes do not align")
    date_ns = dates.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    label_ns = label_end.astype("datetime64[ns]").astype(np.int64)
    missing_time = np.iinfo(np.int64).min
    feature_values: list[np.ndarray] = []
    feature_names: list[str] = []
    lag = max(int(availability_lag_days), 1)
    for horizon_position, horizon in enumerate(horizons):
        for expert_position, expert_name in enumerate(expert_names[1:], start=1):
            edge = np.full(len(dates), np.nan, dtype=float)
            for position in range(len(dates)):
                common_valid = (
                    target_mask[position, horizon_position].astype(bool)
                    & expert_masks[position, horizon_position, :, 0].astype(bool)
                    & expert_masks[
                        position, horizon_position, :, expert_position
                    ].astype(bool)
                )
                baseline_ic = _single_date_group_rank_ic(
                    experts[position, horizon_position, :, 0],
                    target[position, horizon_position],
                    common_valid,
                    groups,
                )
                expert_ic = _single_date_group_rank_ic(
                    experts[position, horizon_position, :, expert_position],
                    target[position, horizon_position],
                    common_valid,
                    groups,
                )
                if np.isfinite(baseline_ic) and np.isfinite(expert_ic):
                    edge[position] = float(expert_ic - baseline_ic)
            matured = pd.Series(np.nan, index=dates, dtype=float)
            for issued_position, value in enumerate(edge):
                maturity = int(label_ns[issued_position, horizon_position])
                if not np.isfinite(value) or maturity == missing_time:
                    continue
                maturity_position = int(np.searchsorted(date_ns, maturity, side="left"))
                available_position = maturity_position + lag
                if available_position < len(dates):
                    matured.iloc[available_position] = float(value)
            history_count = matured.notna().cumsum().astype(float)
            reliability = matured.ewm(
                halflife=float(halflife_days),
                min_periods=int(minimum_observations),
                adjust=False,
                ignore_na=True,
            ).mean()
            reliability = reliability.ffill().fillna(0.0).clip(-2.0, 2.0)
            confidence = (
                history_count / max(int(full_confidence_observations), 1)
            ).clip(0.0, 1.0)
            stem = f"{prefix}_expert_edge_{expert_name}_h{int(horizon)}"
            feature_values.extend(
                [
                    reliability.to_numpy(dtype=np.float32),
                    confidence.to_numpy(dtype=np.float32),
                ]
            )
            feature_names.extend(
                [
                    f"{stem}_ewm{int(round(halflife_days))}",
                    f"{stem}_confidence",
                ]
            )
    if not feature_values:
        return np.empty((len(dates), 0), dtype=np.float32), []
    return np.column_stack(feature_values).astype(np.float32), feature_names


def _daily_expert_edges_by_group(
    experts: np.ndarray,
    expert_masks: np.ndarray,
    target: np.ndarray,
    target_mask: np.ndarray,
    groups: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Measure candidate-minus-baseline rank IC on identical cross-sections."""
    if experts.shape[:3] != target.shape or experts.shape != expert_masks.shape:
        raise ValueError("Expert skill tensors do not align with target tensors")
    if groups is None:
        group_values = np.array([0], dtype=np.int64)
        group_members = [np.arange(target.shape[2], dtype=np.int64)]
    else:
        group_values = np.unique(groups).astype(np.int64)
        group_members = [
            np.flatnonzero(groups == group).astype(np.int64)
            for group in group_values
        ]
    edges = np.zeros(
        (
            target.shape[0],
            target.shape[1],
            len(group_values),
            experts.shape[3],
        ),
        dtype=np.float32,
    )
    masks = np.zeros_like(edges, dtype=bool)
    for date_position in range(target.shape[0]):
        for horizon_position in range(target.shape[1]):
            for group_position, columns in enumerate(group_members):
                base_valid = (
                    target_mask[date_position, horizon_position, columns].astype(
                        bool
                    )
                    & expert_masks[
                        date_position, horizon_position, columns, 0
                    ].astype(bool)
                )
                if base_valid.sum() >= 2:
                    baseline_ic = _single_date_group_rank_ic(
                        experts[
                            date_position, horizon_position, columns, 0
                        ],
                        target[date_position, horizon_position, columns],
                        base_valid,
                        None,
                    )
                    if np.isfinite(baseline_ic):
                        masks[
                            date_position, horizon_position, group_position, 0
                        ] = True
                for expert_position in range(1, experts.shape[3]):
                    common_valid = (
                        base_valid
                        & expert_masks[
                            date_position,
                            horizon_position,
                            columns,
                            expert_position,
                        ].astype(bool)
                    )
                    if common_valid.sum() < 2:
                        continue
                    common_baseline_ic = _single_date_group_rank_ic(
                        experts[
                            date_position, horizon_position, columns, 0
                        ],
                        target[date_position, horizon_position, columns],
                        common_valid,
                        None,
                    )
                    expert_ic = _single_date_group_rank_ic(
                        experts[
                            date_position,
                            horizon_position,
                            columns,
                            expert_position,
                        ],
                        target[date_position, horizon_position, columns],
                        common_valid,
                        None,
                    )
                    if np.isfinite(common_baseline_ic) and np.isfinite(expert_ic):
                        edges[
                            date_position,
                            horizon_position,
                            group_position,
                            expert_position,
                        ] = float(expert_ic - common_baseline_ic)
                        masks[
                            date_position,
                            horizon_position,
                            group_position,
                            expert_position,
                        ] = True
    return edges, masks, group_values


def _forward_smoothed_expert_edge_targets(
    daily_edges: np.ndarray,
    daily_masks: np.ndarray,
    label_end: np.ndarray,
    window_days: int,
    minimum_observations: int,
    halflife_days: float,
    clip: float,
    uncertainty_penalty: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a lower-confidence expert-edge target with conservative maturity."""
    if daily_edges.shape != daily_masks.shape:
        raise ValueError("Daily expert edge values and masks do not align")
    length, horizons, _, _ = daily_edges.shape
    target = np.zeros_like(daily_edges, dtype=np.float32)
    mask = np.zeros_like(daily_masks, dtype=bool)
    target_label_end = np.full(
        (length, horizons), np.datetime64("NaT", "ns"), dtype="datetime64[ns]"
    )
    width = max(int(window_days), 1)
    minimum = max(min(int(minimum_observations), width), 1)
    offsets = np.arange(width, dtype=float)
    weights = np.exp(
        -np.log(2.0) * offsets / max(float(halflife_days), 1e-6)
    )
    label_values = label_end.astype("datetime64[ns]")
    for date_position in range(length):
        stop = min(date_position + width, length)
        local_weights = weights[: stop - date_position]
        for horizon_position in range(horizons):
            local_label_end = label_values[
                date_position:stop, horizon_position
            ]
            finite_label_end = ~np.isnat(local_label_end)
            if finite_label_end.any():
                target_label_end[date_position, horizon_position] = (
                    local_label_end[finite_label_end].max()
                )
            for group_position in range(daily_edges.shape[2]):
                for expert_position in range(daily_edges.shape[3]):
                    valid = daily_masks[
                        date_position:stop,
                        horizon_position,
                        group_position,
                        expert_position,
                    ].astype(bool)
                    if valid.sum() < minimum:
                        continue
                    selected_weights = local_weights[valid]
                    selected_values = daily_edges[
                        date_position:stop,
                        horizon_position,
                        group_position,
                        expert_position,
                    ][valid]
                    weighted_mean = float(
                        np.average(selected_values, weights=selected_weights)
                    )
                    centered = selected_values - weighted_mean
                    effective_observations = float(
                        selected_weights.sum() ** 2
                        / max(float(np.square(selected_weights).sum()), 1e-12)
                    )
                    weighted_variance = float(
                        np.average(np.square(centered), weights=selected_weights)
                    )
                    standard_error = math.sqrt(
                        max(weighted_variance, 0.0)
                        / max(effective_observations, 1.0)
                    )
                    target[
                        date_position,
                        horizon_position,
                        group_position,
                        expert_position,
                    ] = float(
                        weighted_mean
                        - abs(float(uncertainty_penalty)) * standard_error
                    )
                    mask[
                        date_position,
                        horizon_position,
                        group_position,
                        expert_position,
                    ] = True
    target = np.clip(target, -abs(float(clip)), abs(float(clip)))
    return target.astype(np.float32), mask, target_label_end


def _expand_group_expert_targets(
    group_target: np.ndarray,
    group_mask: np.ndarray,
    group_values: np.ndarray,
    groups: np.ndarray | None,
    column_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    if groups is None:
        return (
            np.repeat(group_target, column_count, axis=2),
            np.repeat(group_mask, column_count, axis=2),
        )
    expanded_target = np.zeros(
        (
            group_target.shape[0],
            group_target.shape[1],
            column_count,
            group_target.shape[3],
        ),
        dtype=np.float32,
    )
    expanded_mask = np.zeros_like(expanded_target, dtype=bool)
    for group_position, group in enumerate(group_values):
        columns = np.flatnonzero(groups == group)
        expanded_target[:, :, columns, :] = group_target[
            :, :, group_position : group_position + 1, :
        ]
        expanded_mask[:, :, columns, :] = group_mask[
            :, :, group_position : group_position + 1, :
        ]
    return expanded_target, expanded_mask


def _relative_return_rank_labels(
    returns: pd.DataFrame,
    horizons: list[int],
    minimum_future_coverage: float,
    groups: pd.Series | None,
    momentum_windows: list[int],
    momentum_weights: list[float],
    external_baseline: pd.DataFrame | None = None,
    momentum_baseline_weight: float = 1.0,
    external_baseline_weight: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build optimizer-aligned relative mean-return ranks and a causal baseline.

    Labels use only t+1..t+h returns. The baseline uses data through t and is
    deliberately kept separate from covariance/risk estimation downstream.
    """
    dates = returns.index
    momentum, momentum_available = _causal_momentum_rank_baseline(
        returns,
        groups,
        [int(value) for value in momentum_windows],
        [float(value) for value in momentum_weights],
    )
    baseline = float(momentum_baseline_weight) * momentum
    baseline_available = momentum_available.copy()
    if external_baseline is not None and float(external_baseline_weight) != 0.0:
        external = external_baseline.reindex(index=dates, columns=returns.columns)
        external_available = external.notna()
        external = _group_rank_score(external, groups)
        baseline = baseline + float(external_baseline_weight) * external
        if float(momentum_baseline_weight) == 0.0:
            baseline_available = external_available
        else:
            baseline_available = baseline_available | external_available
    baseline = _group_rank_score(baseline, groups)

    targets: list[np.ndarray] = []
    baselines: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    label_end = np.full((len(dates), len(horizons)), np.datetime64("NaT"), dtype="datetime64[ns]")
    date_values = dates.to_numpy(dtype="datetime64[ns]")
    for horizon_index, horizon in enumerate(horizons):
        future, complete = _future_compounded(returns, int(horizon), minimum_future_coverage)
        target = _group_rank_score(future, groups)
        horizon_available = pd.Series(False, index=dates)
        if horizon < len(dates):
            horizon_available.iloc[:-horizon] = True
        valid = complete & future.notna() & momentum_available
        valid = valid & horizon_available.to_numpy()[:, None]
        targets.append(target.fillna(0.0).to_numpy(dtype=np.float32))
        baselines.append(baseline.fillna(0.0).to_numpy(dtype=np.float32))
        masks.append(valid.to_numpy(dtype=bool))
        if horizon < len(dates):
            label_end[:-horizon, horizon_index] = date_values[horizon:]
    return (
        np.stack(targets, axis=1),
        np.stack(baselines, axis=1),
        np.stack(masks, axis=1),
        label_end,
        baseline_available.to_numpy(dtype=bool),
    )


def _relative_return_utility_labels(
    returns: pd.DataFrame,
    horizons: list[int],
    minimum_future_coverage: float,
    groups: pd.Series | None,
    momentum_windows: list[int],
    momentum_weights: list[float],
    clip: float,
    external_baseline: pd.DataFrame | None = None,
    momentum_baseline_weight: float = 1.0,
    external_baseline_weight: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Continuous optimizer-aligned mean-return utility and causal baseline."""
    dates = returns.index
    momentum, momentum_available = _causal_momentum_utility_baseline(
        returns,
        groups,
        [int(value) for value in momentum_windows],
        [float(value) for value in momentum_weights],
        float(clip),
    )
    baseline = float(momentum_baseline_weight) * momentum
    baseline_available = momentum_available.copy()
    if external_baseline is not None and float(external_baseline_weight) != 0.0:
        external = external_baseline.reindex(index=dates, columns=returns.columns)
        external_available = external.notna()
        external = _group_relative_utility(
            external, groups, float(clip)
        ).fillna(0.0)
        # A missing or cross-sectionally neutral external view is no view, not
        # an instruction to erase the valid momentum anchor.
        baseline = baseline + float(external_baseline_weight) * external
        if float(momentum_baseline_weight) == 0.0:
            baseline_available = external_available
        else:
            baseline_available = baseline_available | external_available
    baseline = _group_relative_utility(baseline, groups, float(clip))

    targets: list[np.ndarray] = []
    baselines: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    label_end = np.full(
        (len(dates), len(horizons)),
        np.datetime64("NaT"),
        dtype="datetime64[ns]",
    )
    date_values = dates.to_numpy(dtype="datetime64[ns]")
    for horizon_index, horizon in enumerate(horizons):
        future, complete = _future_compounded(
            returns, int(horizon), minimum_future_coverage
        )
        horizon_available = pd.Series(False, index=dates)
        if horizon < len(dates):
            horizon_available.iloc[:-horizon] = True
        future_available = complete & future.notna()
        future_available = future_available & horizon_available.to_numpy()[:, None]
        # An unavailable peer must not enter the same-date cross-sectional
        # centre or scale. Its filled zero return would otherwise distort the
        # valid assets' continuous utility target.
        target = _group_relative_utility(
            future.where(future_available), groups, float(clip)
        )
        valid = future_available & momentum_available & target.notna()
        targets.append(target.fillna(0.0).to_numpy(dtype=np.float32))
        baselines.append(baseline.fillna(0.0).to_numpy(dtype=np.float32))
        masks.append(valid.to_numpy(dtype=bool))
        if horizon < len(dates):
            label_end[:-horizon, horizon_index] = date_values[horizon:]
    return (
        np.stack(targets, axis=1),
        np.stack(baselines, axis=1),
        np.stack(masks, axis=1),
        label_end,
        baseline_available.to_numpy(dtype=bool),
    )


def _relative_mean_return_labels(
    returns: pd.DataFrame,
    horizons: list[int],
    minimum_future_coverage: float,
    groups: pd.Series | None,
    momentum_windows: list[int],
    momentum_weights: list[float],
    clip: float,
    external_baseline: pd.DataFrame | None = None,
    momentum_baseline_weight: float = 1.0,
    external_baseline_weight: float = 0.0,
    external_mean_return_scale: float = 0.05,
    require_baseline_for_target: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build annualized relative mean labels without same-date variance inflation.

    The future target uses only t+1..t+h observations. The scale is an
    annualized log mean, so a quiet cross-section remains quiet instead of being
    forced into a unit-variance rank target. Risk and covariance are deliberately
    left to EP/skfolio.
    """
    dates = returns.index
    momentum, momentum_available = _causal_momentum_mean_baseline(
        returns,
        groups,
        [int(value) for value in momentum_windows],
        [float(value) for value in momentum_weights],
        float(clip),
    )
    baseline = float(momentum_baseline_weight) * momentum
    baseline_available = momentum_available.copy()
    if external_baseline is not None and float(external_baseline_weight) != 0.0:
        external = external_baseline.reindex(index=dates, columns=returns.columns)
        external_available = external.notna()
        external = _cross_section_demean(external, groups).fillna(0.0)
        external = external.clip(-1.0, 1.0) * float(external_mean_return_scale)
        baseline = baseline + float(external_baseline_weight) * external
        if float(momentum_baseline_weight) == 0.0:
            baseline_available = external_available
        else:
            baseline_available = baseline_available | external_available
    baseline = _cross_section_demean(baseline, groups).clip(
        -float(clip), float(clip)
    )

    targets: list[np.ndarray] = []
    baselines: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    label_end = np.full(
        (len(dates), len(horizons)),
        np.datetime64("NaT"),
        dtype="datetime64[ns]",
    )
    date_values = dates.to_numpy(dtype="datetime64[ns]")
    for horizon_index, horizon in enumerate(horizons):
        future, complete = _future_compounded(
            returns, int(horizon), minimum_future_coverage
        )
        horizon_available = pd.Series(False, index=dates)
        if horizon < len(dates):
            horizon_available.iloc[:-horizon] = True
        future_available = complete & future.notna()
        future_available = (
            future_available & horizon_available.to_numpy()[:, None]
        )
        annualized_log_mean = (
            np.log1p(future.clip(lower=-0.99)) / float(horizon) * 252.0
        )
        target = _cross_section_demean(
            annualized_log_mean.where(future_available), groups
        ).clip(-float(clip), float(clip))
        valid = future_available & target.notna()
        if require_baseline_for_target:
            valid = valid & baseline_available
        targets.append(target.fillna(0.0).to_numpy(dtype=np.float32))
        baselines.append(baseline.fillna(0.0).to_numpy(dtype=np.float32))
        masks.append(valid.to_numpy(dtype=bool))
        if horizon < len(dates):
            label_end[:-horizon, horizon_index] = date_values[horizon:]
    return (
        np.stack(targets, axis=1),
        np.stack(baselines, axis=1),
        np.stack(masks, axis=1),
        label_end,
        baseline_available.to_numpy(dtype=bool),
    )


def _mechanical_sleeve_view_baseline(
    cfg: dict[str, Any], dates: pd.DatetimeIndex
) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = ROOT / cfg["data"]["mechanical_views"]
    views = pd.read_parquet(path)
    views["date"] = pd.to_datetime(views["date"], errors="coerce")
    views["confidence"] = pd.to_numeric(views["confidence"], errors="coerce")
    views = views.loc[
        views["date"].isin(dates)
        & views["validator_status"].astype(str).eq("valid")
        & views["confidence"].notna()
    ].copy()
    sleeve_map = {
        "rates": "rates",
        "rates_long_duration": "rates",
        "credit": "credit",
        "convertible_bonds": "convertible_bonds",
        "equity": "equity",
        "alternatives": "alternatives",
        "gold": "alternatives",
    }
    scores = pd.DataFrame(0.0, index=dates, columns=RISK_SLEEVES)
    accepted = 0
    for row in views.itertuples(index=False):
        source = sleeve_map.get(str(row.source_asset))
        target = sleeve_map.get(str(row.target_asset))
        if source is None or target is None or source == target:
            continue
        confidence = float(np.clip(row.confidence, 0.0, 1.0))
        relation = str(row.relation).lower()
        direction = -1.0 if "underweight" in relation else 1.0
        scores.at[row.date, source] += direction * confidence
        scores.at[row.date, target] -= direction * confidence
        accepted += 1
    ranked = _rowwise_rank_score(scores)
    centered = scores.sub(scores.mean(axis=1), axis=0)
    confidence_magnitude = np.tanh(centered)
    combined = 0.5 * ranked + 0.5 * confidence_magnitude
    scale = float(cfg["data"].get("mechanical_view_baseline_scale", 1.0))
    metadata = {
        "path": str(path.relative_to(ROOT)),
        "valid_input_rows": int(len(views)),
        "accepted_relative_rows": int(accepted),
        "active_dates": int(scores.abs().sum(axis=1).gt(0.0).sum()),
        "normalization": "50% same-day rank score plus 50% tanh of centered confidence sum",
        "scale": scale,
    }
    return combined * scale, metadata


def _risk_adjusted_labels(
    returns: pd.DataFrame,
    horizons: list[int],
    mean_window: int,
    mean_min: int,
    vol_window: int,
    vol_min: int,
    clip: float,
    minimum_future_coverage: float,
    groups: pd.Series | None = None,
    downside_penalty: float = 0.0,
    external_baseline: pd.DataFrame | None = None,
    momentum_baseline_weight: float = 1.0,
    external_baseline_weight: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dates = returns.index
    targets: list[np.ndarray] = []
    baselines: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    label_end = np.full((len(dates), len(horizons)), np.datetime64("NaT"), dtype="datetime64[ns]")
    expected_daily = returns.rolling(mean_window, min_periods=mean_min).mean()
    volatility = returns.rolling(vol_window, min_periods=vol_min).std(ddof=0)
    for horizon_index, horizon in enumerate(horizons):
        future, complete = _future_compounded(returns, horizon, minimum_future_coverage)
        horizon_available = pd.Series(False, index=dates)
        if horizon < len(dates):
            horizon_available.iloc[:-horizon] = True
        denominator = volatility * math.sqrt(horizon) + 1e-6
        target = (future - expected_daily * horizon).div(denominator).clip(-clip, clip)
        if downside_penalty > 0.0:
            shifted = returns.shift(-1).clip(upper=0.0)
            downside_min_periods = math.ceil(horizon * minimum_future_coverage)
            downside = (
                shifted.pow(2)
                .rolling(horizon, min_periods=downside_min_periods)
                .mean()
                .pow(0.5)
                .shift(-(horizon - 1))
                * math.sqrt(horizon)
            )
            target = target - float(downside_penalty) * downside.div(denominator)
        momentum = returns.rolling(horizon, min_periods=max(3, horizon // 2)).sum().div(denominator).clip(-clip, clip)
        target = _cross_section_demean(target, groups)
        momentum = _cross_section_demean(momentum, groups)
        if momentum_baseline_weight == 0.0:
            baseline = pd.DataFrame(0.0, index=dates, columns=returns.columns)
        else:
            baseline = momentum * float(momentum_baseline_weight)
        if external_baseline is not None and external_baseline_weight != 0.0:
            external = external_baseline.reindex(index=dates, columns=returns.columns)
            baseline = baseline + external * float(external_baseline_weight)
        baseline = _cross_section_demean(baseline, groups).clip(-clip, clip)
        valid = complete & target.notna() & baseline.notna()
        valid = valid & horizon_available.to_numpy()[:, None]
        targets.append(target.fillna(0.0).to_numpy(dtype=np.float32))
        baselines.append(baseline.fillna(0.0).to_numpy(dtype=np.float32))
        masks.append(valid.to_numpy(dtype=bool))
        end_values = np.full(len(dates), np.datetime64("NaT"), dtype="datetime64[ns]")
        if horizon < len(dates):
            end_values[:-horizon] = dates.to_numpy(dtype="datetime64[ns]")[horizon:]
        label_end[:, horizon_index] = end_values
    return (
        np.stack(targets, axis=1),
        np.stack(baselines, axis=1),
        np.stack(masks, axis=1),
        label_end,
    )


def _regime_path_targets(
    regime_index: np.ndarray,
    horizons: list[int],
    classes: int,
    minimum_coverage: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build future t+1..t+h state occupancy labels."""
    rows = len(regime_index)
    target = np.zeros((rows, len(horizons), classes), dtype=np.float32)
    mask = np.zeros((rows, len(horizons)), dtype=bool)
    one_hot = np.zeros((rows, classes), dtype=np.float32)
    known = (regime_index >= 0) & (regime_index < classes)
    one_hot[np.flatnonzero(known), regime_index[known].astype(int)] = 1.0
    cumulative = np.vstack(
        [np.zeros((1, classes), dtype=np.float32), np.cumsum(one_hot, axis=0)]
    )
    for horizon_position, horizon in enumerate(horizons):
        if horizon >= rows:
            continue
        starts = np.arange(rows - horizon)
        counts = cumulative[starts + horizon + 1] - cumulative[starts + 1]
        observed = counts.sum(axis=1)
        valid = observed >= float(horizon) * float(minimum_coverage)
        target[starts[valid], horizon_position] = (
            counts[valid] / observed[valid, None].clip(min=1.0)
        )
        mask[starts[valid], horizon_position] = True
    return target, mask


def _build_local_panel(
    daily: pd.DataFrame,
    cards: pd.DataFrame,
    dates: pd.DatetimeIndex,
    assets: list[str],
    cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    pivots: dict[str, pd.DataFrame] = {}
    for column in ["return_1d", "open", "high", "low", "close", "vol", "amount"]:
        pivots[column] = (
            daily.pivot_table(index="date", columns="asset_code", values=column, aggfunc="last")
            .reindex(index=dates, columns=assets)
        )
    returns = pivots["return_1d"]
    raw: dict[str, pd.DataFrame] = {"ret_1d": returns}
    for window in [int(value) for value in cfg["data"]["local_windows_days"]]:
        raw[f"mom_{window}d"] = returns.rolling(window, min_periods=max(3, window // 4)).sum()
        raw[f"vol_{window}d"] = returns.rolling(window, min_periods=max(3, window // 4)).std(ddof=0) * math.sqrt(252.0)
    raw["dd_60d"] = returns.rolling(60, min_periods=20).apply(_max_drawdown, raw=True)
    previous_close = pivots["close"].shift(1)
    raw["overnight_ret"] = pivots["open"].div(previous_close) - 1.0
    raw["intraday_ret"] = pivots["close"].div(pivots["open"]) - 1.0
    raw["range_pct"] = pivots["high"].div(pivots["low"]) - 1.0
    price_range = pivots["high"] - pivots["low"]
    raw["close_position"] = ((pivots["close"] - pivots["low"]).div(price_range) - 0.5).where(price_range.abs() > 1e-12)
    raw["log_vol"] = np.log1p(pivots["vol"].clip(lower=0.0))
    raw["log_amount"] = np.log1p(pivots["amount"].clip(lower=0.0))
    raw.update(_asset_state_frame(cards, dates, assets))

    clip = float(cfg["preprocessing"]["clip"])
    window = int(cfg["preprocessing"]["fast_window_days"])
    minimum = int(cfg["preprocessing"]["fast_min_days"])
    values: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    feature_names = _local_feature_names(cfg)
    for feature in feature_names:
        frame = raw[feature].reindex(index=dates, columns=assets)
        mask = frame.notna().to_numpy(dtype=np.float32)
        if feature in PASSTHROUGH_LOCAL:
            scaled = frame.fillna(0.0).clip(-1.0, 1.0)
        else:
            scaled = _causal_scale_frame(frame, window, minimum, clip)
        values.append(scaled.fillna(0.0).to_numpy(dtype=np.float32))
        masks.append(mask)
    return np.stack(values, axis=2), np.stack(masks, axis=2), feature_names


def _build_macro_panel(
    macro: pd.DataFrame,
    cards: pd.DataFrame,
    dates: pd.DatetimeIndex,
    cfg: dict[str, Any],
    mechanical_view_baseline: pd.DataFrame | None = None,
) -> tuple[np.ndarray, list[str]]:
    macro = macro.set_index("date").reindex(dates)
    cards = cards.set_index("date").reindex(dates)
    clip = float(cfg["preprocessing"]["clip"])
    level_window = int(cfg["preprocessing"]["level_window_days"])
    level_min = int(cfg["preprocessing"]["level_min_days"])
    delta_days = int(cfg["preprocessing"]["macro_delta_days"])
    fast_window = int(cfg["preprocessing"]["fast_window_days"])
    fast_min = int(cfg["preprocessing"]["fast_min_days"])
    output: dict[str, pd.Series] = {}
    for column in cfg["data"]["macro_features"]:
        values = pd.to_numeric(macro.get(column), errors="coerce")
        output[column] = _rolling_robust_series(values, level_window, level_min, clip)
        output[f"{column}__delta_{delta_days}"] = _rolling_robust_series(
            values.diff(delta_days), fast_window, fast_min, clip
        )
    for column in ["bond_score", "equity_score", "risk_budget_score"]:
        output[f"card_{column}"] = pd.to_numeric(cards[column], errors="coerce").fillna(0.5).clip(0.0, 1.0)
    output["card_liquidity_shock"] = cards["liquidity_shock"].fillna(False).astype(float)
    for column, mapping in STATE_MAPS.items():
        output[f"card_{column}"] = cards[column].map(mapping).fillna(0.0).astype(float)
    if bool(cfg["data"].get("include_stock_bond_grid_one_hot", False)):
        regime = cards["stock_bond_regime"].astype("string")
        for name in STOCK_BOND_REGIMES:
            output[f"grid_{name}"] = regime.eq(name).fillna(False).astype(float)
    if (
        bool(cfg["data"].get("include_mechanical_view_features", False))
        and mechanical_view_baseline is not None
    ):
        aligned = mechanical_view_baseline.reindex(index=dates, columns=RISK_SLEEVES).fillna(0.0)
        for sleeve in RISK_SLEEVES:
            output[f"mechanical_view_{sleeve}"] = aligned[sleeve]
    names = list(output)
    return pd.DataFrame(output, index=dates).fillna(0.0).to_numpy(dtype=np.float32), names


def build_dataset(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = cfg or load_config()
    daily_cfg = load_daily_config(ROOT / cfg["data"]["synthetic_rate_config"])
    daily = load_daily_returns_with_proxy(daily_cfg)
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce")
    daily = daily.dropna(subset=["date", "asset_code", "module"])
    daily = daily.loc[~daily["asset_code"].isin(cfg["data"].get("excluded_context_assets", []))].copy()
    macro = pd.read_parquet(ROOT / cfg["data"]["macro_panel"])
    macro["date"] = pd.to_datetime(macro["date"], errors="coerce")
    cards = pd.read_parquet(ROOT / cfg["data"]["state_cards"])
    cards["date"] = pd.to_datetime(cards["date"], errors="coerce")
    start = pd.Timestamp(cfg["data"]["start_date"])
    end = pd.Timestamp(cfg["data"]["final_date"])
    macro_dates = pd.DatetimeIndex(
        sorted(
            macro.loc[macro["date"].between(start, end), "date"]
            .dropna()
            .unique()
        )
    )
    calendar_mode = str(
        cfg["data"].get("trading_calendar_mode", "macro_dates")
    )
    calendar_anchor = str(
        cfg["data"].get("trading_calendar_anchor_asset", "")
    )
    if calendar_mode == "anchor_asset":
        if not calendar_anchor:
            raise ValueError(
                "data.trading_calendar_anchor_asset is required for anchor_asset mode"
            )
        dates = pd.DatetimeIndex(
            sorted(
                daily.loc[
                    daily["asset_code"].astype(str).eq(calendar_anchor)
                    & daily["date"].between(start, end),
                    "date",
                ]
                .dropna()
                .unique()
            )
        )
        if dates.empty:
            raise ValueError(
                f"Trading-calendar anchor {calendar_anchor!r} has no dates"
            )
        if bool((dates.dayofweek >= 5).any()):
            raise ValueError(
                f"Trading-calendar anchor {calendar_anchor!r} contains weekend dates"
            )
    elif calendar_mode == "macro_dates":
        dates = macro_dates
    else:
        raise ValueError(f"Unknown data.trading_calendar_mode={calendar_mode}")
    asset_info = (
        daily[["asset_code", "asset_name", "module", "sub_module"]]
        .drop_duplicates("asset_code", keep="last")
        .sort_values(["module", "asset_code"])
        .reset_index(drop=True)
    )
    assets = asset_info["asset_code"].astype(str).tolist()
    modules = asset_info.set_index("asset_code")["module"].astype(str).reindex(assets)
    module_index = np.array([RISK_SLEEVES.index(value) for value in modules], dtype=np.int64)

    sleeve_external_baseline = None
    sleeve_baseline_metadata: dict[str, Any] = {
        "mode": "same_horizon_causal_momentum",
        "momentum_weight": 1.0,
        "mechanical_view_weight": 0.0,
    }
    if str(cfg["data"].get("sleeve_baseline_mode", "momentum")) == "mechanical_view_blend":
        sleeve_external_baseline, view_metadata = _mechanical_sleeve_view_baseline(cfg, dates)
        sleeve_baseline_metadata = {
            "mode": "causal_mechanical_views_plus_momentum",
            "momentum_weight": float(cfg["data"].get("sleeve_momentum_baseline_weight", 0.0)),
            "mechanical_view_weight": float(cfg["data"].get("sleeve_mechanical_view_weight", 1.0)),
            "mechanical_views": view_metadata,
        }

    local_values, local_masks, local_feature_names = _build_local_panel(daily, cards, dates, assets, cfg)
    macro_values, macro_names = _build_macro_panel(
        macro, cards, dates, cfg, sleeve_external_baseline
    )
    returns = (
        daily.pivot_table(index="date", columns="asset_code", values="return_1d", aggfunc="last")
        .reindex(index=dates, columns=assets)
    )
    horizons = [int(value) for value in cfg["data"]["horizons_days"]]
    label_args = (
        int(cfg["data"]["expected_return_window_days"]),
        int(cfg["data"]["expected_return_min_days"]),
        int(cfg["data"]["volatility_window_days"]),
        int(cfg["data"]["volatility_min_days"]),
        float(cfg["data"]["target_clip"]),
        float(cfg["data"]["minimum_future_coverage"]),
    )
    sleeve_returns, _ = preferred_daily_sleeve_returns(daily_cfg)
    sleeve_returns = sleeve_returns.reindex(index=dates, columns=RISK_SLEEVES)
    target_mode = str(cfg["data"].get("target_mode", "risk_adjusted_residual"))
    forecast_baseline_mode = str(
        cfg["data"].get("forecast_baseline_mode", "causal_anchor")
    )
    if forecast_baseline_mode not in {"causal_anchor", "zero_active_view"}:
        raise ValueError(
            f"Unknown data.forecast_baseline_mode={forecast_baseline_mode}"
        )
    require_baseline_for_target = bool(
        cfg["data"].get(
            "require_baseline_for_target",
            forecast_baseline_mode != "zero_active_view",
        )
    )
    if target_mode in {
        "relative_return_rank",
        "relative_return_utility",
        "relative_mean_return",
    }:
        momentum_windows = [
            int(value)
            for value in cfg["data"].get("momentum_baseline_windows_days", horizons)
        ]
        momentum_weights = [
            float(value)
            for value in cfg["data"].get(
                "momentum_baseline_weights",
                np.repeat(1.0 / max(len(momentum_windows), 1), len(momentum_windows)),
            )
        ]
        if target_mode == "relative_return_rank":
            label_builder = _relative_return_rank_labels
            target_args: tuple[Any, ...] = ()
        elif target_mode == "relative_return_utility":
            label_builder = _relative_return_utility_labels
            target_args = (
                float(cfg["data"].get("relative_utility_clip", 3.0)),
            )
        else:
            label_builder = _relative_mean_return_labels
            target_args = (
                float(cfg["data"].get("relative_mean_return_clip", 0.60)),
            )
        (
            asset_target,
            asset_baseline,
            asset_target_mask,
            label_end,
            asset_baseline_available,
        ) = (
            label_builder(
                returns,
                horizons,
                float(cfg["data"]["minimum_future_coverage"]),
                modules,
                momentum_windows,
                momentum_weights,
                *target_args,
                **(
                    {
                        "require_baseline_for_target": require_baseline_for_target
                    }
                    if target_mode == "relative_mean_return"
                    else {}
                ),
            )
        )
        (
            sleeve_target,
            sleeve_baseline,
            sleeve_target_mask,
            sleeve_label_end,
            sleeve_baseline_available,
        ) = (
            label_builder(
                sleeve_returns,
                horizons,
                float(cfg["data"]["minimum_future_coverage"]),
                None,
                momentum_windows,
                momentum_weights,
                *target_args,
                external_baseline=sleeve_external_baseline,
                momentum_baseline_weight=float(sleeve_baseline_metadata["momentum_weight"]),
                external_baseline_weight=float(sleeve_baseline_metadata["mechanical_view_weight"]),
                **(
                    {
                        "external_mean_return_scale": float(
                            cfg["data"].get(
                                "mechanical_view_mean_return_scale", 0.05
                            )
                        ),
                        "require_baseline_for_target": require_baseline_for_target,
                    }
                    if target_mode == "relative_mean_return"
                    else {}
                ),
            )
        )
    elif target_mode == "risk_adjusted_residual":
        asset_target, asset_baseline, asset_target_mask, label_end = _risk_adjusted_labels(
            returns,
            horizons,
            *label_args,
            groups=modules,
            downside_penalty=float(cfg["data"].get("asset_downside_penalty", 0.0)),
        )
        sleeve_target, sleeve_baseline, sleeve_target_mask, sleeve_label_end = _risk_adjusted_labels(
            sleeve_returns,
            horizons,
            *label_args,
            groups=None,
            downside_penalty=float(cfg["data"].get("sleeve_downside_penalty", 0.0)),
            external_baseline=sleeve_external_baseline,
            momentum_baseline_weight=float(sleeve_baseline_metadata["momentum_weight"]),
            external_baseline_weight=float(sleeve_baseline_metadata["mechanical_view_weight"]),
        )
        asset_baseline_available = returns.notna().to_numpy(dtype=bool)
        sleeve_baseline_available = sleeve_returns.notna().to_numpy(dtype=bool)
    else:
        raise ValueError(f"Unknown data.target_mode={target_mode}")
    label_end = np.maximum(label_end, sleeve_label_end)

    # Momentum and mechanical Market Views remain causal model inputs. They are
    # not part of the learned model's deployment score when zero_active_view is
    # selected: alpha=0 must then be an exact zero incremental view.
    asset_causal_anchor = asset_baseline.copy()
    sleeve_causal_anchor = sleeve_baseline.copy()
    if forecast_baseline_mode == "zero_active_view":
        asset_baseline = np.zeros_like(asset_causal_anchor, dtype=np.float32)
        sleeve_baseline = np.zeros_like(sleeve_causal_anchor, dtype=np.float32)

    if forecast_baseline_mode == "zero_active_view":
        asset_expert_names = ["zero_incremental_view"]
        sleeve_expert_names = ["zero_incremental_view"]
    else:
        asset_expert_names = ["fixed_causal_baseline"]
        sleeve_expert_names = ["fixed_causal_baseline"]
    asset_experts = asset_baseline[..., None].astype(np.float32)
    sleeve_experts = sleeve_baseline[..., None].astype(np.float32)
    asset_expert_masks = np.repeat(
        asset_baseline_available[:, None, :, None],
        len(horizons),
        axis=1,
    ).astype(np.float32)
    sleeve_expert_masks = np.repeat(
        sleeve_baseline_available[:, None, :, None],
        len(horizons),
        axis=1,
    ).astype(np.float32)
    if str(cfg["model"].get("prediction_head_mode", "unconstrained")) == "adaptive_expert_mixture":
        expert_definitions = cfg["data"].get("signal_expert_blends")
        if expert_definitions:
            asset_signal_values, asset_signal_masks = _causal_blended_signal_experts(
                returns,
                modules,
                list(expert_definitions),
                target_mode,
                float(cfg["data"].get("relative_utility_clip", 3.0)),
            )
            sleeve_signal_values, sleeve_signal_masks = _causal_blended_signal_experts(
                sleeve_returns,
                None,
                list(expert_definitions),
                target_mode,
                float(cfg["data"].get("relative_utility_clip", 3.0)),
            )
        else:
            expert_windows = [
                int(value)
                for value in cfg["data"].get(
                    "signal_expert_windows_days", [5, 20, 60, 120, 252]
                )
            ]
            reversal_windows = [
                int(value)
                for value in cfg["data"].get(
                    "signal_expert_reversal_windows_days", [5, 20]
                )
            ]
            asset_signal_values, asset_signal_masks = _causal_rank_signal_experts(
                returns, modules, expert_windows, reversal_windows
            )
            sleeve_signal_values, sleeve_signal_masks = _causal_rank_signal_experts(
                sleeve_returns, None, expert_windows, reversal_windows
            )
        asset_value_blocks = [asset_experts]
        asset_mask_blocks = [asset_expert_masks]
        sleeve_value_blocks = [sleeve_experts]
        sleeve_mask_blocks = [sleeve_expert_masks]
        for name, frame in asset_signal_values.items():
            values = frame.reindex(index=dates, columns=assets).fillna(0.0).to_numpy(
                dtype=np.float32
            )
            masks = asset_signal_masks[name].reindex(
                index=dates, columns=assets
            ).fillna(False).to_numpy(dtype=np.float32)
            asset_value_blocks.append(
                np.repeat(values[:, None, :, None], len(horizons), axis=1)
            )
            asset_mask_blocks.append(
                np.repeat(masks[:, None, :, None], len(horizons), axis=1)
            )
            asset_expert_names.append(name)
        for name, frame in sleeve_signal_values.items():
            values = frame.reindex(index=dates, columns=RISK_SLEEVES).fillna(0.0).to_numpy(
                dtype=np.float32
            )
            masks = sleeve_signal_masks[name].reindex(
                index=dates, columns=RISK_SLEEVES
            ).fillna(False).to_numpy(dtype=np.float32)
            sleeve_value_blocks.append(
                np.repeat(values[:, None, :, None], len(horizons), axis=1)
            )
            sleeve_mask_blocks.append(
                np.repeat(masks[:, None, :, None], len(horizons), axis=1)
            )
            sleeve_expert_names.append(name)
        if bool(cfg["data"].get("include_card_rank_expert", True)):
            card_position = local_feature_names.index("card_rank")
            card_values = local_values[:, :, card_position]
            card_masks = local_masks[:, :, card_position]
            card_frame = _normalize_expert_frame(
                pd.DataFrame(card_values, index=dates, columns=assets),
                modules,
                target_mode,
                float(cfg["data"].get("relative_utility_clip", 3.0)),
            )
            asset_value_blocks.append(
                np.repeat(
                    card_frame.to_numpy(dtype=np.float32)[:, None, :, None],
                    len(horizons),
                    axis=1,
                )
            )
            asset_mask_blocks.append(
                np.repeat(card_masks[:, None, :, None], len(horizons), axis=1)
            )
            asset_expert_names.append("market_state_card_rank")
        if (
            bool(cfg["data"].get("include_mechanical_view_expert", True))
            and sleeve_external_baseline is not None
        ):
            mechanical = _normalize_expert_frame(
                sleeve_external_baseline.reindex(
                    index=dates, columns=RISK_SLEEVES
                ).fillna(0.0),
                None,
                target_mode,
                float(cfg["data"].get("relative_utility_clip", 3.0)),
            ).to_numpy(dtype=np.float32)
            sleeve_value_blocks.append(
                np.repeat(mechanical[:, None, :, None], len(horizons), axis=1)
            )
            sleeve_mask_blocks.append(
                np.ones(
                    (len(dates), len(horizons), len(RISK_SLEEVES), 1),
                    dtype=np.float32,
                )
            )
            sleeve_expert_names.append("mechanical_market_view")
        asset_experts = np.concatenate(asset_value_blocks, axis=3)
        asset_expert_masks = np.concatenate(asset_mask_blocks, axis=3)
        sleeve_experts = np.concatenate(sleeve_value_blocks, axis=3)
        sleeve_expert_masks = np.concatenate(sleeve_mask_blocks, axis=3)

    asset_expert_skill_target = np.zeros_like(asset_experts, dtype=np.float32)
    asset_expert_skill_mask = np.zeros_like(asset_expert_masks, dtype=bool)
    sleeve_expert_skill_target = np.zeros_like(
        sleeve_experts, dtype=np.float32
    )
    sleeve_expert_skill_mask = np.zeros_like(sleeve_expert_masks, dtype=bool)
    expert_skill_label_end = np.full(
        (len(dates), len(horizons)),
        np.datetime64("NaT", "ns"),
        dtype="datetime64[ns]",
    )
    forward_skill_configuration: dict[str, Any] = {}
    if bool(cfg["data"].get("include_forward_expert_skill_targets", False)):
        skill_cfg = cfg["data"].get("forward_expert_skill_target", {})
        forward_skill_configuration = {
            "window_days": int(skill_cfg.get("window_days", 20)),
            "minimum_observations": int(
                skill_cfg.get("minimum_observations", 10)
            ),
            "halflife_days": float(skill_cfg.get("halflife_days", 10)),
            "clip": float(skill_cfg.get("clip", 1.0)),
            "uncertainty_penalty": float(
                skill_cfg.get("uncertainty_penalty", 0.0)
            ),
        }
        asset_daily_edge, asset_daily_edge_mask, asset_group_values = (
            _daily_expert_edges_by_group(
                asset_experts,
                asset_expert_masks,
                asset_target,
                asset_target_mask,
                module_index,
            )
        )
        sleeve_daily_edge, sleeve_daily_edge_mask, sleeve_group_values = (
            _daily_expert_edges_by_group(
                sleeve_experts,
                sleeve_expert_masks,
                sleeve_target,
                sleeve_target_mask,
                None,
            )
        )
        (
            asset_group_skill,
            asset_group_skill_mask,
            asset_skill_label_end,
        ) = _forward_smoothed_expert_edge_targets(
            asset_daily_edge,
            asset_daily_edge_mask,
            label_end,
            **forward_skill_configuration,
        )
        (
            sleeve_group_skill,
            sleeve_group_skill_mask,
            sleeve_skill_label_end,
        ) = _forward_smoothed_expert_edge_targets(
            sleeve_daily_edge,
            sleeve_daily_edge_mask,
            label_end,
            **forward_skill_configuration,
        )
        asset_expert_skill_target, asset_expert_skill_mask = (
            _expand_group_expert_targets(
                asset_group_skill,
                asset_group_skill_mask,
                asset_group_values,
                module_index,
                len(assets),
            )
        )
        sleeve_expert_skill_target, sleeve_expert_skill_mask = (
            _expand_group_expert_targets(
                sleeve_group_skill,
                sleeve_group_skill_mask,
                sleeve_group_values,
                None,
                len(RISK_SLEEVES),
            )
        )
        expert_skill_label_end = np.maximum(
            asset_skill_label_end, sleeve_skill_label_end
        )

    reliability_feature_names: list[str] = []
    if bool(cfg["data"].get("include_expert_reliability_features", False)):
        reliability_cfg = cfg["data"].get("expert_reliability", {})
        reliability_arguments = {
            "halflife_days": float(reliability_cfg.get("halflife_days", 42)),
            "minimum_observations": int(
                reliability_cfg.get("minimum_observations", 20)
            ),
            "full_confidence_observations": int(
                reliability_cfg.get("full_confidence_observations", 60)
            ),
            "availability_lag_days": int(
                reliability_cfg.get("availability_lag_days", 1)
            ),
        }
        asset_reliability, asset_reliability_names = (
            _causal_expert_reliability_features(
                asset_experts,
                asset_expert_masks,
                asset_target,
                asset_target_mask,
                label_end,
                dates,
                module_index,
                asset_expert_names,
                horizons,
                "asset",
                **reliability_arguments,
            )
        )
        sleeve_reliability, sleeve_reliability_names = (
            _causal_expert_reliability_features(
                sleeve_experts,
                sleeve_expert_masks,
                sleeve_target,
                sleeve_target_mask,
                label_end,
                dates,
                None,
                sleeve_expert_names,
                horizons,
                "sleeve",
                **reliability_arguments,
            )
        )
        macro_values = np.concatenate(
            [macro_values, asset_reliability, sleeve_reliability], axis=1
        ).astype(np.float32)
        reliability_feature_names = [
            *asset_reliability_names,
            *sleeve_reliability_names,
        ]
        macro_names.extend(reliability_feature_names)

    if bool(cfg["data"].get("include_baseline_rank_features", False)):
        asset_baseline_features = np.transpose(
            asset_causal_anchor, (0, 2, 1)
        ).astype(np.float32)
        asset_baseline_feature_masks = np.repeat(
            asset_baseline_available[:, :, None],
            len(horizons),
            axis=2,
        ).astype(np.float32)
        local_values = np.concatenate([local_values, asset_baseline_features], axis=2)
        local_masks = np.concatenate(
            [local_masks, asset_baseline_feature_masks],
            axis=2,
        )
        baseline_feature_kind = (
            "utility"
            if target_mode == "relative_return_utility"
            else (
                "annualized_mean"
                if target_mode == "relative_mean_return"
                else "rank"
            )
        )
        local_feature_names.extend(
            [
                f"causal_baseline_{baseline_feature_kind}_h{horizon}"
                for horizon in horizons
            ]
        )
        sleeve_baseline_features = sleeve_causal_anchor.reshape(
            len(dates), -1
        ).astype(np.float32)
        macro_values = np.concatenate([macro_values, sleeve_baseline_features], axis=1)
        macro_names.extend(
            [
                f"causal_sleeve_baseline_h{horizon}_{sleeve}"
                for horizon in horizons
                for sleeve in RISK_SLEEVES
            ]
        )
    regime_names = list(STOCK_BOND_REGIMES)
    regime_map = {name: index for index, name in enumerate(regime_names)}
    regime_values = cards.set_index("date")["stock_bond_regime"].reindex(dates).astype("string")
    regime_index = regime_values.map(regime_map).fillna(len(regime_names)).to_numpy(dtype=np.int64)
    regime_prior_count = np.zeros(len(regime_index), dtype=np.float32)
    observed_regime_counts = np.zeros(len(regime_names) + 1, dtype=np.int64)
    for position, regime in enumerate(regime_index):
        regime_prior_count[position] = observed_regime_counts[regime]
        observed_regime_counts[regime] += 1
    regime_target = np.full((len(dates), len(horizons)), -1, dtype=np.int64)
    regime_target_mask = np.zeros((len(dates), len(horizons)), dtype=bool)
    regime_label_end = np.full(
        (len(dates), len(horizons)), np.iinfo(np.int64).min, dtype=np.int64
    )
    date_ns = dates.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    for horizon_position, horizon in enumerate(horizons):
        if horizon >= len(dates):
            continue
        future_regime = regime_index[horizon:]
        valid = future_regime < len(regime_names)
        regime_target[:-horizon, horizon_position] = future_regime
        regime_target_mask[:-horizon, horizon_position] = valid
        regime_label_end[:-horizon, horizon_position] = date_ns[horizon:]
    regime_occupancy_target, regime_occupancy_mask = _regime_path_targets(
        regime_index,
        horizons,
        len(regime_names),
        float(cfg["data"].get("minimum_regime_path_coverage", 0.90)),
    )
    seq_len = int(cfg["data"]["sequence_length_days"])
    sample_indices = np.arange(seq_len - 1, len(dates), int(cfg["data"]["prediction_stride_days"]), dtype=np.int64)

    np.savez_compressed(
        project_path(cfg, "dataset_npz"),
        dates=dates.to_numpy(dtype="datetime64[ns]").astype(np.int64),
        local_values=local_values,
        local_masks=local_masks,
        macro_values=macro_values,
        returns=returns.fillna(0.0).to_numpy(dtype=np.float32),
        return_masks=returns.notna().to_numpy(dtype=np.float32),
        asset_target=asset_target,
        asset_baseline=asset_baseline,
        asset_target_mask=asset_target_mask,
        sleeve_target=sleeve_target,
        sleeve_baseline=sleeve_baseline,
        sleeve_target_mask=sleeve_target_mask,
        asset_causal_anchor=asset_causal_anchor,
        sleeve_causal_anchor=sleeve_causal_anchor,
        asset_experts=asset_experts,
        asset_expert_masks=asset_expert_masks,
        sleeve_experts=sleeve_experts,
        sleeve_expert_masks=sleeve_expert_masks,
        asset_expert_skill_target=asset_expert_skill_target,
        asset_expert_skill_mask=asset_expert_skill_mask,
        sleeve_expert_skill_target=sleeve_expert_skill_target,
        sleeve_expert_skill_mask=sleeve_expert_skill_mask,
        expert_skill_label_end=expert_skill_label_end.astype(
            "datetime64[ns]"
        ).astype(np.int64),
        label_end=label_end.astype("datetime64[ns]").astype(np.int64),
        module_index=module_index,
        regime_index=regime_index,
        regime_prior_count=regime_prior_count,
        regime_target=regime_target,
        regime_target_mask=regime_target_mask,
        regime_label_end=regime_label_end,
        regime_occupancy_target=regime_occupancy_target,
        regime_occupancy_mask=regime_occupancy_mask,
        sample_indices=sample_indices,
    )
    metadata = {
        "version": cfg["version"],
        "rows": len(dates),
        "samples": len(sample_indices),
        "start": str(dates.min().date()),
        "end": str(dates.max().date()),
        "sequence_length_days": seq_len,
        "trading_calendar": {
            "mode": calendar_mode,
            "anchor_asset": calendar_anchor or None,
            "date_count": int(len(dates)),
            "macro_dates_excluded": int(len(macro_dates.difference(dates))),
            "weekend_date_count": int((dates.dayofweek >= 5).sum()),
        },
        "assets": assets,
        "asset_names": asset_info["asset_name"].astype(str).tolist(),
        "asset_modules": modules.tolist(),
        "investable_union_note": "All context assets are modeled; each product trades only assets present in its formal anchor.",
        "local_features": local_feature_names,
        "local_value_and_mask_channels": len(local_feature_names) * 2,
        "macro_features": macro_names,
        "horizons_days": horizons,
        "target_sleeves": RISK_SLEEVES,
        "target_mode": target_mode,
        "forecast_baseline_mode": forecast_baseline_mode,
        "target_definition": (
            "future t+1..t+h compounded mean-return rank within sleeve (assets) or across "
            "sleeves; no past mean is subtracted and covariance remains a separate EP input"
            if target_mode == "relative_return_rank"
            else (
                "future t+1..t+h compounded mean-return continuous cross-sectional utility "
                "within sleeve (assets) or across sleeves; risk remains a separate EP input"
                if target_mode == "relative_return_utility"
                else (
                    "future t+1..t+h annualized log mean return, cross-sectionally "
                    "demeaned without future variance normalization; covariance and "
                    "risk remain separate EP/skfolio inputs"
                    if target_mode == "relative_mean_return"
                    else "future excess return minus configured future downside-risk penalty, divided "
                    "by past-only volatility and cross-sectionally demeaned"
                )
            )
        ),
        "asset_downside_penalty": float(cfg["data"].get("asset_downside_penalty", 0.0)),
        "sleeve_downside_penalty": float(cfg["data"].get("sleeve_downside_penalty", 0.0)),
        "regime_names": regime_names + ["unknown"],
        "regime_definition": "point-in-time stock-bond 3x3 grid from the daily market state card",
        "asset_baseline_definition": (
            {
                "mode": "zero_incremental_model_view",
                "fallback": (
                    "When the model gate or blend alpha is zero, the learned "
                    "view is exactly zero; momentum and state rules remain "
                    "separate causal inputs."
                ),
            }
            if forecast_baseline_mode == "zero_active_view"
            else (
            {
                "mode": (
                    "causal_multi_window_momentum_utility"
                    if target_mode == "relative_return_utility"
                    else (
                        "causal_multi_window_annualized_relative_mean"
                        if target_mode == "relative_mean_return"
                        else "causal_multi_window_momentum_rank"
                    )
                ),
                "windows_days": cfg["data"].get("momentum_baseline_windows_days", horizons),
                "weights": cfg["data"].get("momentum_baseline_weights"),
            }
            if target_mode
            in {
                "relative_return_rank",
                "relative_return_utility",
                "relative_mean_return",
            }
            else "same-horizon causal momentum divided by past-only volatility"
            )
        ),
        "sleeve_baseline_definition": (
            {
                "mode": "zero_incremental_model_view",
                "fallback": "Exact zero learned sleeve view when gated off.",
            }
            if forecast_baseline_mode == "zero_active_view"
            else sleeve_baseline_metadata
        ),
        "causal_anchor_feature_definition": {
            "asset": {
                "mode": (
                    "causal_multi_window_annualized_relative_mean"
                    if target_mode == "relative_mean_return"
                    else "causal_multi_window_momentum"
                ),
                "windows_days": cfg["data"].get(
                    "momentum_baseline_windows_days", horizons
                ),
                "weights": cfg["data"].get("momentum_baseline_weights"),
            },
            "sleeve": sleeve_baseline_metadata,
            "role": (
                "Input feature and diagnostic only; never silently added to "
                "the learned deployment view under zero_active_view."
            ),
        },
        "asset_signal_experts": asset_expert_names,
        "sleeve_signal_experts": sleeve_expert_names,
        "signal_expert_rule": (
            (
                "The unconstrained direct-mean head uses an exact zero "
                "incremental fallback. Causal momentum and Market Views are "
                "input features, not deployment experts."
            )
            if forecast_baseline_mode == "zero_active_view"
            else (
                "Every expert uses information available on or before "
                "as_of_date. The fixed causal baseline is expert zero and is "
                "the learned-gate prior."
            )
        ),
        "expert_reliability_features": {
            "enabled": bool(reliability_feature_names),
            "features": reliability_feature_names,
            "rule": (
                "Each value is an exponentially weighted historical expert-minus-baseline "
                "rank-IC edge. A forecast enters this memory only after its complete label "
                "window has matured plus the configured trading-day lag."
            ),
            "configuration": cfg["data"].get("expert_reliability", {}),
        },
        "forward_expert_skill_target": {
            "enabled": bool(forward_skill_configuration),
            "rule": (
                "For each as_of_date, horizon and routing group, the target is "
                "the exponentially weighted candidate-minus-baseline "
                "cross-sectional rank-IC edge over the configured forward issue "
                "window. Candidate and baseline use the same available assets. "
                "Its maturity is the latest complete return-label end in that "
                "window, so it is never visible early."
            ),
            "configuration": forward_skill_configuration,
        },
        "point_in_time_rule": "all features end on as_of_date; labels start on the next trading day",
    }
    project_path(cfg, "metadata").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    if "label_diagnostics" in cfg.get("outputs", {}):
        diagnostic_rows: list[dict[str, Any]] = []
        split_masks = {
            "pre_prediction": dates < pd.Timestamp(cfg["data"]["prediction_start"]),
            "walkforward_2022_2023": (dates >= pd.Timestamp("2022-01-01"))
            & (dates < pd.Timestamp("2024-01-01")),
            "research_2024_onward": dates >= pd.Timestamp("2024-01-01"),
        }
        for split_name, split_mask in split_masks.items():
            selected = np.flatnonzero(np.asarray(split_mask))
            if not len(selected):
                continue
            for horizon_position, horizon in enumerate(horizons):
                for (
                    task,
                    deployment_baseline_values,
                    causal_anchor_values,
                    target_values,
                    target_masks,
                    groups,
                ) in [
                    (
                        "asset_within_sleeve",
                        asset_baseline,
                        asset_causal_anchor,
                        asset_target,
                        asset_target_mask,
                        module_index,
                    ),
                    (
                        "sleeve_cross_section",
                        sleeve_baseline,
                        sleeve_causal_anchor,
                        sleeve_target,
                        sleeve_target_mask,
                        None,
                    ),
                ]:
                    mask = target_masks[selected, horizon_position]
                    valid_values = target_values[selected, horizon_position][mask]
                    diagnostic_rows.append(
                        {
                            "split": split_name,
                            "task": task,
                            "horizon_days": int(horizon),
                            "rows": int(len(selected)),
                            "valid_cells": int(mask.sum()),
                            "valid_fraction": float(mask.mean()) if mask.size else np.nan,
                            "target_std": float(np.std(valid_values)) if valid_values.size else np.nan,
                            "deployment_baseline_rank_ic": rank_ic(
                                deployment_baseline_values[
                                    selected, horizon_position
                                ],
                                target_values[selected, horizon_position],
                                mask,
                                groups,
                            ),
                            "causal_anchor_rank_ic": rank_ic(
                                causal_anchor_values[
                                    selected, horizon_position
                                ],
                                target_values[selected, horizon_position],
                                mask,
                                groups,
                            ),
                        }
                    )
        pd.DataFrame(diagnostic_rows).to_csv(
            project_path(cfg, "label_diagnostics"),
            index=False,
            encoding="utf-8-sig",
        )
    return metadata


@dataclass
class DatasetArrays:
    dates: np.ndarray
    local_values: np.ndarray
    local_masks: np.ndarray
    macro_values: np.ndarray
    returns: np.ndarray
    return_masks: np.ndarray
    asset_target: np.ndarray
    asset_baseline: np.ndarray
    asset_target_mask: np.ndarray
    sleeve_target: np.ndarray
    sleeve_baseline: np.ndarray
    sleeve_target_mask: np.ndarray
    asset_experts: np.ndarray
    asset_expert_masks: np.ndarray
    sleeve_experts: np.ndarray
    sleeve_expert_masks: np.ndarray
    asset_expert_skill_target: np.ndarray
    asset_expert_skill_mask: np.ndarray
    sleeve_expert_skill_target: np.ndarray
    sleeve_expert_skill_mask: np.ndarray
    expert_skill_label_end: np.ndarray
    label_end: np.ndarray
    module_index: np.ndarray
    regime_index: np.ndarray
    regime_prior_count: np.ndarray
    regime_target: np.ndarray
    regime_target_mask: np.ndarray
    regime_label_end: np.ndarray
    regime_occupancy_target: np.ndarray
    regime_occupancy_mask: np.ndarray
    sample_indices: np.ndarray


def load_arrays(cfg: dict[str, Any] | None = None) -> tuple[DatasetArrays, dict[str, Any]]:
    cfg = cfg or load_config()
    raw = np.load(project_path(cfg, "dataset_npz"), allow_pickle=False)
    values = {name: raw[name] for name in DatasetArrays.__dataclass_fields__ if name in raw.files}
    if "asset_experts" not in values:
        values["asset_experts"] = raw["asset_baseline"][..., None].astype(np.float32)
        values["asset_expert_masks"] = np.ones_like(
            values["asset_experts"], dtype=np.float32
        )
    if "sleeve_experts" not in values:
        values["sleeve_experts"] = raw["sleeve_baseline"][..., None].astype(np.float32)
        values["sleeve_expert_masks"] = np.ones_like(
            values["sleeve_experts"], dtype=np.float32
        )
    if "asset_expert_skill_target" not in values:
        values["asset_expert_skill_target"] = np.zeros_like(
            values["asset_experts"], dtype=np.float32
        )
        values["asset_expert_skill_mask"] = np.zeros_like(
            values["asset_experts"], dtype=bool
        )
    if "sleeve_expert_skill_target" not in values:
        values["sleeve_expert_skill_target"] = np.zeros_like(
            values["sleeve_experts"], dtype=np.float32
        )
        values["sleeve_expert_skill_mask"] = np.zeros_like(
            values["sleeve_experts"], dtype=bool
        )
    if "expert_skill_label_end" not in values:
        values["expert_skill_label_end"] = np.full_like(
            raw["label_end"], np.iinfo(np.int64).min, dtype=np.int64
        )
    if "regime_index" not in values:
        values["regime_index"] = np.zeros(len(raw["dates"]), dtype=np.int64)
    if "regime_prior_count" not in values:
        regime_index = values["regime_index"].astype(np.int64)
        regime_prior_count = np.zeros(len(regime_index), dtype=np.float32)
        observed_regime_counts = np.zeros(int(regime_index.max(initial=0)) + 1, dtype=np.int64)
        for position, regime in enumerate(regime_index):
            regime_prior_count[position] = observed_regime_counts[regime]
            observed_regime_counts[regime] += 1
        values["regime_prior_count"] = regime_prior_count
    if any(
        name not in values
        for name in ["regime_target", "regime_target_mask", "regime_label_end"]
    ):
        dates = raw["dates"].astype(np.int64)
        regime_index = values["regime_index"].astype(np.int64)
        horizons = [int(value) for value in cfg["data"]["horizons_days"]]
        known_regime_count = int(cfg["model"].get("transition_classes", 9))
        regime_target = np.full((len(dates), len(horizons)), -1, dtype=np.int64)
        regime_target_mask = np.zeros((len(dates), len(horizons)), dtype=bool)
        regime_label_end = np.full(
            (len(dates), len(horizons)), np.iinfo(np.int64).min, dtype=np.int64
        )
        for horizon_position, horizon in enumerate(horizons):
            if horizon >= len(dates):
                continue
            future_regime = regime_index[horizon:]
            valid = future_regime < known_regime_count
            regime_target[:-horizon, horizon_position] = future_regime
            regime_target_mask[:-horizon, horizon_position] = valid
            regime_label_end[:-horizon, horizon_position] = dates[horizon:]
        values["regime_target"] = regime_target
        values["regime_target_mask"] = regime_target_mask
        values["regime_label_end"] = regime_label_end
    if any(
        name not in values
        for name in ["regime_occupancy_target", "regime_occupancy_mask"]
    ):
        horizons = [int(value) for value in cfg["data"]["horizons_days"]]
        known_regime_count = int(cfg["model"].get("transition_classes", 9))
        occupancy_target, occupancy_mask = _regime_path_targets(
            values["regime_index"].astype(np.int64),
            horizons,
            known_regime_count,
            float(cfg["data"].get("minimum_regime_path_coverage", 0.90)),
        )
        values["regime_occupancy_target"] = occupancy_target
        values["regime_occupancy_mask"] = occupancy_mask
    arrays = DatasetArrays(**values)
    metadata = json.loads(project_path(cfg, "metadata").read_text(encoding="utf-8"))
    return arrays, metadata


class WindowDataset(Dataset):
    def __init__(
        self,
        arrays: DatasetArrays,
        indices: np.ndarray,
        sequence_length: int,
        label_cutoff_ns: int | None = None,
    ) -> None:
        self.arrays = arrays
        self.indices = np.asarray(indices, dtype=np.int64)
        self.sequence_length = int(sequence_length)
        self.label_cutoff_ns = label_cutoff_ns

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        index = int(self.indices[item])
        start = index - self.sequence_length + 1
        asset_mask = self.arrays.asset_target_mask[index].copy()
        sleeve_mask = self.arrays.sleeve_target_mask[index].copy()
        asset_expert_skill_mask = (
            self.arrays.asset_expert_skill_mask[index].copy()
        )
        sleeve_expert_skill_mask = (
            self.arrays.sleeve_expert_skill_mask[index].copy()
        )
        regime_mask = self.arrays.regime_target_mask[index].copy()
        regime_occupancy_mask = self.arrays.regime_occupancy_mask[index].copy()
        if self.label_cutoff_ns is not None:
            matured = (self.arrays.label_end[index] > np.iinfo(np.int64).min) & (
                self.arrays.label_end[index] < self.label_cutoff_ns
            )
            asset_mask &= matured[:, None]
            sleeve_mask &= matured[:, None]
            expert_skill_matured = (
                (
                    self.arrays.expert_skill_label_end[index]
                    > np.iinfo(np.int64).min
                )
                & (
                    self.arrays.expert_skill_label_end[index]
                    < self.label_cutoff_ns
                )
            )
            asset_expert_skill_mask &= expert_skill_matured[:, None, None]
            sleeve_expert_skill_mask &= expert_skill_matured[:, None, None]
            regime_matured = (
                (self.arrays.regime_label_end[index] > np.iinfo(np.int64).min)
                & (self.arrays.regime_label_end[index] < self.label_cutoff_ns)
            )
            regime_mask &= regime_matured
            regime_occupancy_mask &= regime_matured
        return {
            "local_values": torch.from_numpy(self.arrays.local_values[start : index + 1]),
            "local_masks": torch.from_numpy(self.arrays.local_masks[start : index + 1]),
            "macro_values": torch.from_numpy(self.arrays.macro_values[start : index + 1]),
            "asset_target": torch.from_numpy(self.arrays.asset_target[index]),
            "asset_baseline": torch.from_numpy(self.arrays.asset_baseline[index]),
            "asset_mask": torch.from_numpy(asset_mask),
            "sleeve_target": torch.from_numpy(self.arrays.sleeve_target[index]),
            "sleeve_baseline": torch.from_numpy(self.arrays.sleeve_baseline[index]),
            "sleeve_mask": torch.from_numpy(sleeve_mask),
            "asset_experts": torch.from_numpy(self.arrays.asset_experts[index]),
            "asset_expert_masks": torch.from_numpy(
                self.arrays.asset_expert_masks[index]
            ),
            "sleeve_experts": torch.from_numpy(self.arrays.sleeve_experts[index]),
            "sleeve_expert_masks": torch.from_numpy(
                self.arrays.sleeve_expert_masks[index]
            ),
            "asset_expert_skill_target": torch.from_numpy(
                self.arrays.asset_expert_skill_target[index]
            ),
            "asset_expert_skill_mask": torch.from_numpy(
                asset_expert_skill_mask
            ),
            "sleeve_expert_skill_target": torch.from_numpy(
                self.arrays.sleeve_expert_skill_target[index]
            ),
            "sleeve_expert_skill_mask": torch.from_numpy(
                sleeve_expert_skill_mask
            ),
            "regime_index": torch.tensor(self.arrays.regime_index[index], dtype=torch.long),
            "regime_prior_count": torch.tensor(
                self.arrays.regime_prior_count[index], dtype=torch.float32
            ),
            "regime_target": torch.from_numpy(self.arrays.regime_target[index]),
            "regime_target_mask": torch.from_numpy(regime_mask),
            "regime_occupancy_target": torch.from_numpy(
                self.arrays.regime_occupancy_target[index]
            ),
            "regime_occupancy_mask": torch.from_numpy(regime_occupancy_mask),
            "index": torch.tensor(index, dtype=torch.long),
        }


def _share_expert_logits(
    logits: torch.Tensor,
    scope: str,
    groups: torch.Tensor | None = None,
) -> torch.Tensor:
    """Constrain routing to a cross-sectional decision, not one choice per asset."""
    if scope == "column":
        return logits
    if scope == "cross_section_shared":
        return logits.mean(dim=2, keepdim=True).expand_as(logits)
    if scope == "module_shared":
        if groups is None:
            raise ValueError("module_shared expert routing requires group labels")
        output = torch.empty_like(logits)
        for group in torch.unique(groups):
            columns = torch.where(groups.eq(group))[0]
            shared = logits[:, :, columns, :].mean(dim=2, keepdim=True)
            output[:, :, columns, :] = shared
        return output
    raise ValueError(f"Unknown expert gate scope: {scope}")


def _masked_shrunk_expert_gate(
    logits: torch.Tensor,
    masks: torch.Tensor | None,
    prior: torch.Tensor,
    adaptation_scale: float,
    temperature: float,
) -> torch.Tensor:
    """Return a masked convex gate while preserving a fixed baseline prior."""
    available = None if masks is None else masks.gt(0.0)
    masked_logits = (
        logits
        if available is None
        else logits.masked_fill(~available, -1e4)
    )
    adaptive = torch.softmax(
        masked_logits / max(float(temperature), 1e-3), dim=3
    )
    scale = float(np.clip(adaptation_scale, 0.0, 1.0))
    gate = (1.0 - scale) * prior[None, None, None, :] + scale * adaptive
    if available is None:
        return gate
    gate = gate * available.to(dtype=gate.dtype)
    normalizer = gate.sum(dim=3, keepdim=True)
    return torch.where(
        normalizer.gt(1e-12),
        gate / normalizer.clamp_min(1e-12),
        torch.zeros_like(gate),
    )


class HierarchicalAssetITransformer(nn.Module):
    def __init__(
        self,
        sequence_length: int,
        local_features: int,
        macro_features: int,
        module_index: np.ndarray,
        horizons: int,
        cfg: dict[str, Any],
        asset_experts: int = 1,
        sleeve_experts: int = 1,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        d_model = int(cfg["d_model"])
        self.sequence_length = int(sequence_length)
        self.asset_count = len(module_index)
        self.local_features = int(local_features)
        self.macro_features = int(macro_features)
        self.horizons = int(horizons)
        self.asset_expert_count = int(asset_experts)
        self.sleeve_expert_count = int(sleeve_experts)
        self.asset_module_ids = [int(value) for value in module_index.tolist()]
        self.register_buffer("module_index", torch.tensor(module_index, dtype=torch.long))
        self.local_token_count = local_features * 2
        self.local_projection = nn.Sequential(
            nn.Linear(sequence_length, d_model), nn.GELU(), nn.LayerNorm(d_model)
        )
        self.local_feature_embedding = nn.Embedding(self.local_token_count, d_model)
        local_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=int(cfg.get("local_n_heads", cfg["n_heads"])),
            dim_feedforward=int(cfg.get("local_dim_feedforward", cfg["dim_feedforward"])),
            dropout=float(cfg["dropout"]),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        local_layers = int(cfg.get("local_num_layers", 0))
        self.local_encoder = (
            nn.TransformerEncoder(local_layer, num_layers=local_layers, norm=nn.LayerNorm(d_model))
            if local_layers > 0
            else nn.Identity()
        )
        self.local_pool_score = nn.Linear(d_model, 1)
        self.macro_projection = nn.Sequential(nn.Linear(sequence_length, d_model), nn.GELU(), nn.LayerNorm(d_model))
        self.macro_feature_embedding = nn.Embedding(macro_features, d_model)
        self.macro_pool_score = nn.Linear(d_model, 1)
        self.asset_embedding = nn.Embedding(self.asset_count, d_model)
        self.module_embedding = nn.Embedding(len(RISK_SLEEVES), d_model)
        self.token_type = nn.Embedding(2, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=int(cfg["n_heads"]),
            dim_feedforward=int(cfg["dim_feedforward"]),
            dropout=float(cfg["dropout"]),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(cfg["num_layers"]), norm=nn.LayerNorm(d_model))
        self.asset_head_scope = str(cfg.get("asset_head_scope", "shared"))
        if self.asset_head_scope == "module":
            self.asset_mean_by_module = nn.ModuleList(
                [nn.Linear(d_model, horizons) for _ in RISK_SLEEVES]
            )
            self.asset_logvar_by_module = nn.ModuleList(
                [nn.Linear(d_model, horizons) for _ in RISK_SLEEVES]
            )
        elif self.asset_head_scope == "shared":
            self.asset_mean = nn.Linear(d_model, horizons)
            self.asset_logvar = nn.Linear(d_model, horizons)
        else:
            raise ValueError(f"Unknown model.asset_head_scope={self.asset_head_scope}")
        self.sleeve_head = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.GELU(), nn.LayerNorm(d_model))
        self.sleeve_mean = nn.Linear(d_model, horizons)
        self.sleeve_logvar = nn.Linear(d_model, horizons)
        self.prediction_head_mode = str(
            cfg.get("prediction_head_mode", "unconstrained")
        )
        if self.prediction_head_mode == "adaptive_expert_mixture":
            if self.asset_expert_count < 1 or self.sleeve_expert_count < 1:
                raise ValueError("adaptive_expert_mixture requires at least one signal expert")
            self.asset_expert_gate = nn.Linear(
                d_model, horizons * self.asset_expert_count
            )
            self.sleeve_expert_gate = nn.Linear(
                d_model, horizons * self.sleeve_expert_count
            )
            initial_probability = float(
                cfg.get("expert_initial_baseline_probability", 0.96)
            )
            initial_probability = float(np.clip(initial_probability, 0.50, 0.999))

            def initial_logits(count: int) -> torch.Tensor:
                if count == 1:
                    return torch.zeros(1, dtype=torch.float32)
                remainder = (1.0 - initial_probability) / (count - 1)
                probabilities = torch.full(
                    (count,), remainder, dtype=torch.float32
                )
                probabilities[0] = initial_probability
                return probabilities.log()

            asset_logits = initial_logits(self.asset_expert_count)
            sleeve_logits = initial_logits(self.sleeve_expert_count)
            nn.init.zeros_(self.asset_expert_gate.weight)
            nn.init.zeros_(self.sleeve_expert_gate.weight)
            with torch.no_grad():
                self.asset_expert_gate.bias.copy_(asset_logits.repeat(horizons))
                self.sleeve_expert_gate.bias.copy_(sleeve_logits.repeat(horizons))
            asset_prior = torch.zeros(self.asset_expert_count, dtype=torch.float32)
            sleeve_prior = torch.zeros(self.sleeve_expert_count, dtype=torch.float32)
            asset_prior[0] = 1.0
            sleeve_prior[0] = 1.0
            self.register_buffer("asset_expert_prior", asset_prior)
            self.register_buffer("sleeve_expert_prior", sleeve_prior)
        elif self.prediction_head_mode != "unconstrained":
            raise ValueError(f"Unknown prediction_head_mode={self.prediction_head_mode}")
        self.sleeve_architecture = str(cfg.get("sleeve_architecture", "shared_regression"))
        if self.sleeve_architecture == "regime_moe_pairwise":
            self.sleeve_expert_count = int(cfg.get("sleeve_expert_count", 3))
            self.regime_count = int(cfg.get("regime_count", 10))
            self.regime_embedding = nn.Embedding(self.regime_count, d_model)
            self.sleeve_gate = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.GELU(),
                nn.LayerNorm(d_model),
                nn.Linear(d_model, self.sleeve_expert_count),
            )
            self.sleeve_rank_experts = nn.ModuleList(
                [nn.Linear(d_model, horizons) for _ in range(self.sleeve_expert_count)]
            )
        elif self.sleeve_architecture == "hierarchical_regime_pairwise":
            self.regime_count = int(cfg.get("regime_count", 10))
            self.regime_embedding = nn.Embedding(self.regime_count, d_model)
            self.bond_regime_embedding = nn.Embedding(4, d_model)
            self.equity_regime_embedding = nn.Embedding(4, d_model)
            self.regime_condition_norm = nn.LayerNorm(d_model)
            self.sleeve_rank_global = nn.Linear(d_model, horizons)
            adapter_dim = int(cfg.get("regime_adapter_dim", max(d_model // 2, 8)))
            self.sleeve_rank_regime = nn.Sequential(
                nn.Linear(d_model * 2, adapter_dim),
                nn.GELU(),
                nn.Dropout(float(cfg["dropout"])),
                nn.Linear(adapter_dim, horizons),
            )
        elif self.sleeve_architecture == "regime_transition_hybrid":
            self.regime_count = int(cfg.get("regime_count", 10))
            self.transition_classes = int(cfg.get("transition_classes", 9))
            self.regime_embedding = nn.Embedding(self.regime_count, d_model)
            self.bond_regime_embedding = nn.Embedding(4, d_model)
            self.equity_regime_embedding = nn.Embedding(4, d_model)
            self.regime_condition_norm = nn.LayerNorm(d_model)
            self.transition_head = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.GELU(),
                nn.Dropout(float(cfg["dropout"])),
                nn.Linear(d_model, horizons * self.transition_classes),
            )
            sleeve_count = len(RISK_SLEEVES)
            self.state_payoff_global = nn.Parameter(torch.zeros(horizons, sleeve_count))
            self.state_payoff_bond = nn.Parameter(torch.zeros(horizons, 3, sleeve_count))
            self.state_payoff_equity = nn.Parameter(torch.zeros(horizons, 3, sleeve_count))
            self.state_payoff_interaction = nn.Parameter(
                torch.zeros(horizons, self.transition_classes, sleeve_count)
            )
            nn.init.normal_(self.state_payoff_bond, std=0.02)
            nn.init.normal_(self.state_payoff_equity, std=0.02)
            nn.init.normal_(self.state_payoff_interaction, std=0.01)
        elif self.sleeve_architecture == "factorized_regime_transition":
            self.regime_count = int(cfg.get("regime_count", 10))
            self.transition_classes = 9
            self.regime_embedding = nn.Embedding(self.regime_count, d_model)
            self.bond_regime_embedding = nn.Embedding(4, d_model)
            self.equity_regime_embedding = nn.Embedding(4, d_model)
            self.regime_condition_norm = nn.LayerNorm(d_model)
            transition_hidden = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.GELU(),
                nn.Dropout(float(cfg["dropout"])),
            )
            self.factorized_transition_hidden = transition_hidden
            self.bond_transition_head = nn.Linear(d_model, horizons * 3)
            self.equity_transition_head = nn.Linear(d_model, horizons * 3)
            self.grid_interaction_head = nn.Linear(d_model, horizons * 9)
            axis = np.array([1.0, 0.0, -1.0])
            compatibility = np.zeros((9, len(RISK_SLEEVES)), dtype=np.float32)
            for state in range(9):
                bond = axis[state // 3]
                equity = axis[state % 3]
                defensive = float(equity < 0.0)
                compatibility[state] = [
                    bond,
                    0.70 * bond,
                    0.65 * equity + 0.35 * bond,
                    equity,
                    0.25 * bond - 0.20 * equity + 0.35 * defensive,
                ]
                compatibility[state] -= compatibility[state].mean()
            self.register_buffer(
                "grid_compatibility", torch.tensor(compatibility, dtype=torch.float32)
            )
        elif self.sleeve_architecture != "shared_regression":
            raise ValueError(f"Unknown sleeve_architecture={self.sleeve_architecture}")

    def encode_context(
        self,
        local_values: torch.Tensor,
        local_masks: torch.Tensor,
        macro_values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode one strictly point-in-time market window.

        This method is shared by supervised forecasting and masked-history
        pretraining. Exposing the representations avoids maintaining a second,
        subtly different encoder for the self-supervised stage.
        """
        batch = local_values.shape[0]
        # Inverted-token hierarchy: each local feature contributes its complete
        # 120-day history, then a shared within-asset encoder pools those feature
        # tokens before cross-asset attention. This avoids a large flattening layer.
        asset_input = torch.cat([local_values, local_masks], dim=-1).permute(0, 2, 3, 1)
        local = asset_input.reshape(batch * self.asset_count, self.local_token_count, self.sequence_length)
        local = self.local_projection(local)
        local_ids = torch.arange(self.local_token_count, device=local_values.device)
        local = local + self.local_feature_embedding(local_ids)[None, :, :]
        local = self.local_encoder(local)
        local_weights = torch.softmax(self.local_pool_score(local).squeeze(-1), dim=1)
        asset_tokens = torch.sum(local * local_weights.unsqueeze(-1), dim=1).reshape(batch, self.asset_count, -1)
        asset_ids = torch.arange(self.asset_count, device=local_values.device)
        asset_tokens = asset_tokens + self.asset_embedding(asset_ids)[None, :, :]
        asset_tokens = asset_tokens + self.module_embedding(self.module_index)[None, :, :]
        asset_tokens = asset_tokens + self.token_type(torch.zeros(self.asset_count, dtype=torch.long, device=local_values.device))[None, :, :]

        macro_input = macro_values.permute(0, 2, 1)
        macro_tokens = self.macro_projection(macro_input)
        macro_ids = torch.arange(self.macro_features, device=local_values.device)
        macro_tokens = macro_tokens + self.macro_feature_embedding(macro_ids)[None, :, :]
        macro_tokens = macro_tokens + self.token_type(
            torch.ones(self.macro_features, dtype=torch.long, device=local_values.device)
        )[None, :, :]
        encoded = self.encoder(torch.cat([asset_tokens, macro_tokens], dim=1))
        encoded_assets = encoded[:, : self.asset_count]
        encoded_macro = encoded[:, self.asset_count :]
        macro_weights = torch.softmax(self.macro_pool_score(encoded_macro).squeeze(-1), dim=1)
        macro_pool = torch.sum(encoded_macro * macro_weights.unsqueeze(-1), dim=1)

        sleeve_tokens: list[torch.Tensor] = []
        for module in range(len(RISK_SLEEVES)):
            pooled = encoded_assets[:, self.module_index.eq(module)].mean(dim=1)
            sleeve_tokens.append(self.sleeve_head(torch.cat([pooled, macro_pool], dim=1)))
        sleeves = torch.stack(sleeve_tokens, dim=1)
        return encoded_assets, encoded_macro, macro_pool, sleeves

    def forward(
        self,
        local_values: torch.Tensor,
        local_masks: torch.Tensor,
        macro_values: torch.Tensor,
        regime_index: torch.Tensor | None = None,
        regime_prior_count: torch.Tensor | None = None,
        asset_experts: torch.Tensor | None = None,
        asset_expert_masks: torch.Tensor | None = None,
        sleeve_experts: torch.Tensor | None = None,
        sleeve_expert_masks: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        batch = local_values.shape[0]
        encoded_assets, encoded_macro, macro_pool, sleeves = self.encode_context(
            local_values, local_masks, macro_values
        )
        if self.asset_head_scope == "module":
            asset_mean = torch.stack(
                [
                    self.asset_mean_by_module[module](
                        encoded_assets[:, asset_position]
                    )
                    for asset_position, module in enumerate(
                        self.asset_module_ids
                    )
                ],
                dim=1,
            ).permute(0, 2, 1)
            asset_logvar = torch.stack(
                [
                    self.asset_logvar_by_module[module](
                        encoded_assets[:, asset_position]
                    )
                    for asset_position, module in enumerate(
                        self.asset_module_ids
                    )
                ],
                dim=1,
            ).permute(0, 2, 1).clamp(-4.0, 3.0)
        else:
            asset_mean = self.asset_mean(encoded_assets).permute(0, 2, 1)
            asset_logvar = self.asset_logvar(encoded_assets).permute(
                0, 2, 1
            ).clamp(-4.0, 3.0)
        sleeve_mean = self.sleeve_mean(sleeves).permute(0, 2, 1)
        sleeve_logvar = self.sleeve_logvar(sleeves).permute(0, 2, 1).clamp(-4.0, 3.0)
        if (
            self.prediction_head_mode == "unconstrained"
            and str(self.cfg.get("prediction_output_activation", "linear"))
            == "tanh"
        ):
            asset_mean = torch.tanh(asset_mean) * float(
                self.cfg.get("asset_prediction_output_scale", 1.0)
            )
            sleeve_mean = torch.tanh(sleeve_mean) * float(
                self.cfg.get("sleeve_prediction_output_scale", 1.0)
            )
        if self.prediction_head_mode == "adaptive_expert_mixture":
            if self.sleeve_architecture != "shared_regression":
                raise ValueError(
                    "adaptive_expert_mixture currently requires sleeve_architecture=shared_regression"
                )
            if asset_experts is None or sleeve_experts is None:
                raise ValueError("Signal expert tensors are required by the adaptive mixture head")
            asset_logits = self.asset_expert_gate(encoded_assets).reshape(
                batch, self.asset_count, self.horizons, self.asset_expert_count
            ).permute(0, 2, 1, 3)
            sleeve_logits = self.sleeve_expert_gate(sleeves).reshape(
                batch,
                len(RISK_SLEEVES),
                self.horizons,
                self.sleeve_expert_count,
            ).permute(0, 2, 1, 3)
            asset_logits = _share_expert_logits(
                asset_logits,
                str(self.cfg.get("asset_expert_gate_scope", "column")),
                self.module_index,
            )
            sleeve_logits = _share_expert_logits(
                sleeve_logits,
                str(self.cfg.get("sleeve_expert_gate_scope", "column")),
            )
            temperature = float(self.cfg.get("expert_gate_temperature", 1.0))
            adaptation_scale = float(
                self.cfg.get("expert_adaptation_scale", 0.45)
            )
            asset_gate = _masked_shrunk_expert_gate(
                asset_logits,
                asset_expert_masks,
                self.asset_expert_prior,
                float(
                    self.cfg.get(
                        "asset_expert_adaptation_scale", adaptation_scale
                    )
                ),
                temperature,
            )
            sleeve_gate = _masked_shrunk_expert_gate(
                sleeve_logits,
                sleeve_expert_masks,
                self.sleeve_expert_prior,
                float(
                    self.cfg.get(
                        "sleeve_expert_adaptation_scale", adaptation_scale
                    )
                ),
                temperature,
            )
            asset_mean = torch.sum(asset_gate * asset_experts, dim=3)
            sleeve_mean = torch.sum(sleeve_gate * sleeve_experts, dim=3)
            module_membership = F.one_hot(
                self.module_index, num_classes=len(RISK_SLEEVES)
            ).to(dtype=asset_mean.dtype)
            module_count = module_membership.sum(dim=0).clamp_min(1.0)
            module_mean = torch.einsum(
                "bha,as->bhs", asset_mean, module_membership
            ) / module_count[None, None, :]
            asset_mean = asset_mean - torch.einsum(
                "bhs,as->bha", module_mean, module_membership
            )
            sleeve_mean = sleeve_mean - sleeve_mean.mean(dim=2, keepdim=True)
            return (
                asset_mean,
                asset_logvar,
                sleeve_mean,
                sleeve_logvar,
                asset_gate,
                sleeve_gate,
            )
        if self.sleeve_architecture == "regime_moe_pairwise":
            if regime_index is None:
                regime_index = torch.zeros(batch, dtype=torch.long, device=local_values.device)
            regime_index = regime_index.clamp(0, self.regime_count - 1)
            regime_token = self.regime_embedding(regime_index)
            temperature = max(float(self.cfg.get("sleeve_gate_temperature", 1.0)), 1e-3)
            gate = torch.softmax(
                self.sleeve_gate(torch.cat([macro_pool, regime_token], dim=1)) / temperature,
                dim=1,
            )
            expert_scores = torch.stack(
                [expert(sleeves).permute(0, 2, 1) for expert in self.sleeve_rank_experts],
                dim=1,
            )
            sleeve_rank_strength = torch.sum(
                expert_scores * gate[:, :, None, None], dim=1
            )
            sleeve_rank_strength = torch.tanh(sleeve_rank_strength)
            sleeve_rank_strength = sleeve_rank_strength - sleeve_rank_strength.mean(dim=2, keepdim=True)
            return (
                asset_mean,
                asset_logvar,
                sleeve_mean,
                sleeve_logvar,
                sleeve_rank_strength,
                gate,
            )
        if self.sleeve_architecture == "hierarchical_regime_pairwise":
            if regime_index is None:
                regime_index = torch.full(
                    (batch,), self.regime_count - 1, dtype=torch.long, device=local_values.device
                )
            regime_index = regime_index.clamp(0, self.regime_count - 1)
            known_regime = regime_index.lt(9)
            bond_index = torch.where(known_regime, regime_index // 3, torch.full_like(regime_index, 3))
            equity_index = torch.where(known_regime, regime_index % 3, torch.full_like(regime_index, 3))
            regime_condition = self.regime_condition_norm(
                self.regime_embedding(regime_index)
                + self.bond_regime_embedding(bond_index)
                + self.equity_regime_embedding(equity_index)
            )
            global_strength = self.sleeve_rank_global(sleeves).permute(0, 2, 1)
            repeated_condition = regime_condition[:, None, :].expand(-1, sleeves.shape[1], -1)
            regime_residual = self.sleeve_rank_regime(
                torch.cat([sleeves, repeated_condition], dim=2)
            ).permute(0, 2, 1)
            regime_residual = regime_residual - regime_residual.mean(dim=2, keepdim=True)
            if regime_prior_count is None:
                regime_prior_count = torch.zeros(batch, dtype=torch.float32, device=local_values.device)
            prior = max(float(self.cfg.get("regime_shrinkage_prior_count", 60.0)), 1e-6)
            adjustment_weight = torch.sqrt(
                regime_prior_count.float().clamp_min(0.0)
                / (regime_prior_count.float().clamp_min(0.0) + prior)
            )
            adjustment_weight = adjustment_weight * known_regime.float()
            adjustment_weight = (
                adjustment_weight * float(self.cfg.get("regime_adjustment_scale", 0.35))
            ).clamp(0.0, 1.0)
            sleeve_rank_strength = global_strength + adjustment_weight[:, None, None] * regime_residual
            sleeve_rank_strength = torch.tanh(sleeve_rank_strength)
            sleeve_rank_strength = sleeve_rank_strength - sleeve_rank_strength.mean(dim=2, keepdim=True)
            return (
                asset_mean,
                asset_logvar,
                sleeve_mean,
                sleeve_logvar,
                sleeve_rank_strength,
                adjustment_weight[:, None],
                regime_residual,
            )
        if self.sleeve_architecture == "regime_transition_hybrid":
            if regime_index is None:
                regime_index = torch.full(
                    (batch,), self.regime_count - 1, dtype=torch.long, device=local_values.device
                )
            regime_index = regime_index.clamp(0, self.regime_count - 1)
            known_regime = regime_index.lt(9)
            bond_index = torch.where(known_regime, regime_index // 3, torch.full_like(regime_index, 3))
            equity_index = torch.where(known_regime, regime_index % 3, torch.full_like(regime_index, 3))
            regime_condition = self.regime_condition_norm(
                self.regime_embedding(regime_index)
                + self.bond_regime_embedding(bond_index)
                + self.equity_regime_embedding(equity_index)
            )
            transition_logits = self.transition_head(
                torch.cat([macro_pool, regime_condition], dim=1)
            ).reshape(batch, self.horizons, self.transition_classes)
            temperature = max(float(self.cfg.get("transition_temperature", 1.0)), 1e-3)
            transition_probabilities = torch.softmax(transition_logits / temperature, dim=2)
            state_ids = torch.arange(self.transition_classes, device=local_values.device)
            state_bond = torch.div(state_ids, 3, rounding_mode="floor").clamp_max(2)
            state_equity = torch.remainder(state_ids, 3).clamp_max(2)
            interaction_scale = float(self.cfg.get("transition_interaction_scale", 0.25))
            state_payoff = (
                self.state_payoff_global[:, None, :]
                + self.state_payoff_bond[:, state_bond, :]
                + self.state_payoff_equity[:, state_equity, :]
                + interaction_scale * self.state_payoff_interaction
            )
            state_payoff = state_payoff - state_payoff.mean(dim=2, keepdim=True)
            sleeve_rank_strength = torch.einsum(
                "bhk,hks->bhs", transition_probabilities, state_payoff
            )
            sleeve_rank_strength = torch.tanh(sleeve_rank_strength)
            sleeve_rank_strength = sleeve_rank_strength - sleeve_rank_strength.mean(dim=2, keepdim=True)
            return (
                asset_mean,
                asset_logvar,
                sleeve_mean,
                sleeve_logvar,
                sleeve_rank_strength,
                transition_logits,
                transition_probabilities,
                self.state_payoff_interaction,
            )
        if self.sleeve_architecture == "factorized_regime_transition":
            if regime_index is None:
                regime_index = torch.full(
                    (batch,), self.regime_count - 1, dtype=torch.long, device=local_values.device
                )
            regime_index = regime_index.clamp(0, self.regime_count - 1)
            known_regime = regime_index.lt(9)
            bond_index = torch.where(known_regime, regime_index // 3, torch.full_like(regime_index, 3))
            equity_index = torch.where(known_regime, regime_index % 3, torch.full_like(regime_index, 3))
            regime_condition = self.regime_condition_norm(
                self.regime_embedding(regime_index)
                + self.bond_regime_embedding(bond_index)
                + self.equity_regime_embedding(equity_index)
            )
            hidden = self.factorized_transition_hidden(
                torch.cat([macro_pool, regime_condition], dim=1)
            )
            bond_logits = self.bond_transition_head(hidden).reshape(batch, self.horizons, 3)
            equity_logits = self.equity_transition_head(hidden).reshape(batch, self.horizons, 3)
            interaction_logits = self.grid_interaction_head(hidden).reshape(batch, self.horizons, 9)
            joint_logits = (
                bond_logits[:, :, :, None] + equity_logits[:, :, None, :]
            ).reshape(batch, self.horizons, 9)
            joint_logits = joint_logits + float(
                self.cfg.get("transition_interaction_logit_scale", 0.10)
            ) * interaction_logits
            temperature = max(float(self.cfg.get("transition_temperature", 1.0)), 1e-3)
            joint_probabilities = torch.softmax(joint_logits / temperature, dim=2)
            sleeve_rank_strength = torch.einsum(
                "bhk,ks->bhs", joint_probabilities, self.grid_compatibility
            )
            sleeve_rank_strength = sleeve_rank_strength - sleeve_rank_strength.mean(dim=2, keepdim=True)
            return (
                asset_mean,
                asset_logvar,
                sleeve_mean,
                sleeve_logvar,
                sleeve_rank_strength,
                bond_logits,
                equity_logits,
                joint_logits,
                joint_probabilities,
                interaction_logits,
            )
        return asset_mean, asset_logvar, sleeve_mean, sleeve_logvar


def _horizon_weights(
    cfg: dict[str, Any],
    count: int,
    device: torch.device,
    task: str | None = None,
) -> torch.Tensor:
    raw = (
        cfg.get(f"{task}_horizon_loss_weights")
        if task is not None
        else None
    )
    if raw is None:
        raw = cfg.get("horizon_loss_weights")
    if raw is None:
        return torch.full((count,), 1.0 / count, dtype=torch.float32, device=device)
    weights = torch.tensor(raw, dtype=torch.float32, device=device)
    if len(weights) != count or bool((weights < 0).any()) or float(weights.sum()) <= 0.0:
        prefix = f"{task}_" if task is not None else ""
        raise ValueError(
            f"{prefix}horizon_loss_weights must contain {count} non-negative values"
        )
    return weights / weights.sum()


def _asset_module_horizon_weights(
    cfg: dict[str, Any],
    count: int,
    device: torch.device,
) -> torch.Tensor | None:
    """Return independently normalized horizon weights for each asset sleeve."""
    raw = cfg.get("asset_module_horizon_loss_weights")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(
            "asset_module_horizon_loss_weights must map sleeve names to weights"
        )
    fallback = _horizon_weights(cfg, count, device, "asset")
    rows: list[torch.Tensor] = []
    unknown = sorted(set(raw) - set(RISK_SLEEVES))
    if unknown:
        raise ValueError(
            "Unknown asset_module_horizon_loss_weights sleeves: "
            + ", ".join(unknown)
        )
    for module in RISK_SLEEVES:
        values = raw.get(module)
        if values is None:
            rows.append(fallback)
            continue
        weights = torch.tensor(values, dtype=torch.float32, device=device)
        if (
            len(weights) != count
            or bool((weights < 0).any())
            or float(weights.sum()) <= 0.0
        ):
            raise ValueError(
                "asset_module_horizon_loss_weights."
                f"{module} must contain {count} non-negative values "
                "with a positive sum"
            )
        rows.append(weights / weights.sum())
    return torch.stack(rows)


def _asset_module_aggregation_weights(
    cfg: dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    """Weight module losses without letting larger universes dominate."""
    raw = cfg.get("asset_module_loss_weights")
    if raw is None:
        return torch.full(
            (len(RISK_SLEEVES),),
            1.0 / len(RISK_SLEEVES),
            dtype=torch.float32,
            device=device,
        )
    if not isinstance(raw, dict):
        raise ValueError("asset_module_loss_weights must map sleeve names to weights")
    unknown = sorted(set(raw) - set(RISK_SLEEVES))
    if unknown:
        raise ValueError(
            "Unknown asset_module_loss_weights sleeves: " + ", ".join(unknown)
        )
    weights = torch.tensor(
        [float(raw.get(module, 1.0)) for module in RISK_SLEEVES],
        dtype=torch.float32,
        device=device,
    )
    if (
        not torch.isfinite(weights).all()
        or bool((weights < 0).any())
        or float(weights.sum()) <= 0.0
    ):
        raise ValueError(
            "asset_module_loss_weights must contain finite non-negative values "
            "with a positive sum"
        )
    return weights / weights.sum()


def _horizon_value(
    raw: Any,
    horizon: int,
    count: int,
    default: float,
) -> float:
    if raw is None:
        return float(default)
    if isinstance(raw, (list, tuple, np.ndarray)):
        if len(raw) != count:
            raise ValueError(
                f"horizon parameter must contain {count} values, got {len(raw)}"
            )
        return float(raw[horizon])
    return float(raw)


def _masked_regression_loss(
    pred: torch.Tensor,
    logvar: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    horizon_weights: torch.Tensor,
    mode: str = "heteroscedastic_nll",
    huber_beta: float = 1.0,
    uncertainty_nll_weight: float = 0.25,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    valid_weights: list[torch.Tensor] = []
    for horizon in range(pred.shape[1]):
        if float(horizon_weights[horizon]) <= 0.0:
            continue
        valid = mask[:, horizon].bool()
        if not valid.any():
            continue
        error = pred[:, horizon] - target[:, horizon]
        robust = F.smooth_l1_loss(
            pred[:, horizon][valid],
            target[:, horizon][valid],
            beta=max(float(huber_beta), 1e-6),
        )
        if mode == "heteroscedastic_nll":
            nll = 0.5 * (
                torch.exp(-logvar[:, horizon]) * error.square()
                + logvar[:, horizon]
            )
            losses.append(nll[valid].mean() + 0.25 * robust)
        elif mode == "robust_with_detached_uncertainty":
            # The uncertainty head is calibrated to the detached error. It
            # cannot reduce the gradient on the mean prediction by inflating
            # variance on hard observations.
            detached_error = error.detach().square()
            uncertainty_nll = 0.5 * (
                torch.exp(-logvar[:, horizon]) * detached_error
                + logvar[:, horizon]
            )
            losses.append(
                robust
                + float(uncertainty_nll_weight)
                * uncertainty_nll[valid].mean()
            )
        else:
            raise ValueError(f"Unknown regression_loss_mode={mode}")
        valid_weights.append(horizon_weights[horizon])
    if not losses:
        return pred.sum() * 0.0
    weights = torch.stack(valid_weights)
    weights = weights / weights.sum()
    return torch.sum(torch.stack(losses) * weights)


def _fold_training_loss_scale(
    raw: Any,
    pred: torch.Tensor,
    groups: torch.Tensor | None,
    name: str,
) -> torch.Tensor:
    """Broadcast a scale estimated only from the current fold's training labels."""
    if raw is None:
        return torch.ones(
            (1, pred.shape[1], pred.shape[2]),
            dtype=pred.dtype,
            device=pred.device,
        )
    scale = torch.as_tensor(raw, dtype=pred.dtype, device=pred.device)
    if scale.ndim != 2 or scale.shape[0] != pred.shape[1]:
        raise ValueError(
            f"{name} must have shape [horizons, groups_or_columns], got "
            f"{tuple(scale.shape)}"
        )
    if groups is None:
        if scale.shape[1] != pred.shape[2]:
            raise ValueError(
                f"{name} has {scale.shape[1]} columns, expected {pred.shape[2]}"
            )
        expanded = scale
    else:
        if groups.ndim != 1 or len(groups) != pred.shape[2]:
            raise ValueError("groups must map every prediction column")
        if int(groups.min()) < 0 or int(groups.max()) >= scale.shape[1]:
            raise ValueError(f"{name} does not cover every group index")
        expanded = scale[:, groups.long()]
    if not torch.isfinite(expanded).all() or bool((expanded <= 0).any()):
        raise ValueError(f"{name} must contain finite positive values")
    return expanded.unsqueeze(0)


def _pairwise_rank_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    groups: torch.Tensor | None,
    horizon_weights: torch.Tensor,
    minimum_margin: Any = 1e-6,
    importance_scale: Any = None,
    importance_power: float = 0.0,
    maximum_importance: float = 1.0,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    loss_weights: list[torch.Tensor] = []
    for horizon in range(pred.shape[1]):
        if float(horizon_weights[horizon]) <= 0.0:
            continue
        horizon_losses: list[torch.Tensor] = []
        group_values = [None] if groups is None else torch.unique(groups).tolist()
        for group in group_values:
            columns = torch.arange(pred.shape[-1], device=pred.device) if group is None else torch.where(groups.eq(group))[0]
            if len(columns) < 2:
                continue
            p = pred[:, horizon, columns]
            y = target[:, horizon, columns]
            m = mask[:, horizon, columns].bool()
            pdiff = p.unsqueeze(-1) - p.unsqueeze(-2)
            ydiff = y.unsqueeze(-1) - y.unsqueeze(-2)
            margin = _horizon_value(
                minimum_margin, horizon, pred.shape[1], 1e-6
            )
            pair_mask = (
                m.unsqueeze(-1)
                & m.unsqueeze(-2)
                & ydiff.abs().gt(margin)
            )
            upper = torch.triu(torch.ones(len(columns), len(columns), dtype=torch.bool, device=pred.device), diagonal=1)
            pair_mask = pair_mask & upper
            if pair_mask.any():
                pair_loss = F.softplus(
                    -torch.sign(ydiff[pair_mask]) * pdiff[pair_mask]
                )
                if float(importance_power) > 0.0:
                    scale = max(
                        _horizon_value(
                            importance_scale,
                            horizon,
                            pred.shape[1],
                            1.0,
                        ),
                        1e-8,
                    )
                    importance = (
                        ydiff[pair_mask].abs() / scale
                    ).clamp(
                        min=1e-3,
                        max=max(float(maximum_importance), 1e-3),
                    ).pow(float(importance_power))
                    pair_loss = (
                        pair_loss * importance
                    ).sum() / importance.sum().clamp_min(1e-8)
                else:
                    pair_loss = pair_loss.mean()
                horizon_losses.append(pair_loss)
        if horizon_losses:
            losses.append(torch.stack(horizon_losses).mean())
            loss_weights.append(horizon_weights[horizon])
    if not losses:
        return pred.sum() * 0.0
    weights = torch.stack(loss_weights)
    weights = weights / weights.sum()
    return torch.sum(torch.stack(losses) * weights)


def _decision_aligned_mean_loss(
    pred: torch.Tensor,
    baseline: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    groups: torch.Tensor | None,
    horizon_weights: torch.Tensor,
    horizon_days: list[int],
    temperature: Any,
    transaction_cost_bps: Any,
    utility_scale: Any,
) -> torch.Tensor:
    """Optimize a bounded mean view for downstream value after implementation cost.

    This is a differentiable training surrogate only. It does not estimate
    covariance or replace EP/skfolio. Predicted and baseline scores are mapped
    to comparable long-only shadow allocations, and the model is rewarded only
    for future active mean return that clears one-way transaction cost.
    """
    if len(horizon_days) != pred.shape[1]:
        raise ValueError("decision_horizons_days must match model horizons")
    losses: list[torch.Tensor] = []
    loss_weights: list[torch.Tensor] = []
    group_values = [None] if groups is None else torch.unique(groups).tolist()
    for horizon in range(pred.shape[1]):
        if float(horizon_weights[horizon]) <= 0.0:
            continue
        horizon_losses: list[torch.Tensor] = []
        tau = max(
            _horizon_value(temperature, horizon, pred.shape[1], 0.10),
            1e-4,
        )
        scale = max(
            _horizon_value(utility_scale, horizon, pred.shape[1], 0.02),
            1e-6,
        )
        cost_rate = (
            _horizon_value(
                transaction_cost_bps, horizon, pred.shape[1], 5.0
            )
            * 1e-4
            * 252.0
            / max(float(horizon_days[horizon]), 1.0)
        )
        for group in group_values:
            columns = (
                torch.arange(pred.shape[-1], device=pred.device)
                if group is None
                else torch.where(groups.eq(group))[0]
            )
            if len(columns) < 2:
                continue
            selected_mask = mask[:, horizon, columns].bool()
            valid_rows = selected_mask.sum(dim=1).ge(2)
            if not valid_rows.any():
                continue
            selected_mask = selected_mask[valid_rows]
            selected_pred = pred[valid_rows, horizon][:, columns]
            selected_baseline = baseline[valid_rows, horizon][:, columns]
            selected_target = target[valid_rows, horizon][:, columns]
            floor = torch.finfo(selected_pred.dtype).min
            pred_weights = torch.softmax(
                (selected_pred / tau).masked_fill(~selected_mask, floor),
                dim=1,
            )
            baseline_weights = torch.softmax(
                (selected_baseline / tau).masked_fill(~selected_mask, floor),
                dim=1,
            )
            active = pred_weights - baseline_weights
            gross_active_mean = (active * selected_target).sum(dim=1)
            turnover = 0.5 * active.abs().sum(dim=1)
            net_active_mean = gross_active_mean - cost_rate * turnover
            horizon_losses.append(
                (F.softplus(-net_active_mean / scale) * scale).mean()
            )
        if horizon_losses:
            losses.append(torch.stack(horizon_losses).mean())
            loss_weights.append(horizon_weights[horizon])
    if not losses:
        return pred.sum() * 0.0
    weights = torch.stack(loss_weights)
    weights = weights / weights.sum()
    return torch.sum(torch.stack(losses) * weights)


def _listwise_rank_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    groups: torch.Tensor | None,
    horizon_weights: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Symmetric top/bottom listwise cross-entropy for long-only active tilts."""
    losses: list[torch.Tensor] = []
    loss_weights: list[torch.Tensor] = []
    tau = max(float(temperature), 1e-3)
    for horizon in range(pred.shape[1]):
        if float(horizon_weights[horizon]) <= 0.0:
            continue
        horizon_losses: list[torch.Tensor] = []
        group_values = [None] if groups is None else torch.unique(groups).tolist()
        for group in group_values:
            columns = (
                torch.arange(pred.shape[-1], device=pred.device)
                if group is None
                else torch.where(groups.eq(group))[0]
            )
            if len(columns) < 2:
                continue
            p = pred[:, horizon, columns]
            y = target[:, horizon, columns]
            m = mask[:, horizon, columns].bool()
            valid_rows = m.sum(dim=1).ge(2)
            if not valid_rows.any():
                continue
            p = p[valid_rows] / tau
            y = y[valid_rows] / tau
            m = m[valid_rows]
            negative_infinity = torch.finfo(p.dtype).min
            masked_pred = p.masked_fill(~m, negative_infinity)
            masked_target = y.masked_fill(~m, negative_infinity)
            masked_bottom_target = (-y).masked_fill(~m, negative_infinity)
            top_distribution = torch.softmax(masked_target.detach(), dim=1)
            bottom_distribution = torch.softmax(masked_bottom_target.detach(), dim=1)
            top = -(top_distribution * F.log_softmax(masked_pred, dim=1)).sum(dim=1)
            bottom = -(
                bottom_distribution
                * F.log_softmax((-p).masked_fill(~m, negative_infinity), dim=1)
            ).sum(dim=1)
            horizon_losses.append(0.5 * (top + bottom))
        if horizon_losses:
            losses.append(torch.cat(horizon_losses).mean())
            loss_weights.append(horizon_weights[horizon])
    if not losses:
        return pred.sum() * 0.0
    weights = torch.stack(loss_weights)
    weights = weights / weights.sum()
    return torch.sum(torch.stack(losses) * weights)


def _cross_sectional_expert_skill(
    expert_values: torch.Tensor,
    target_values: torch.Tensor,
    valid: torch.Tensor,
    metric: str,
    minimum_assets: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score each expert date by date without letting signal scale choose it."""
    counts = valid.sum(dim=1)
    available = counts.ge(int(minimum_assets))
    weights = valid.to(dtype=expert_values.dtype)
    denominator = counts.clamp_min(1).to(dtype=expert_values.dtype)
    if metric == "mse":
        squared_error = (expert_values - target_values.unsqueeze(-1)).square()
        error = (squared_error * weights).sum(dim=1) / denominator
        return -error.masked_fill(~available, 1e4), available
    if metric != "cross_sectional_correlation":
        raise ValueError(f"Unknown expert oracle metric: {metric}")
    target_expanded = target_values.unsqueeze(-1)
    target_mean = (target_expanded * weights).sum(dim=1) / denominator
    expert_mean = (expert_values * weights).sum(dim=1) / denominator
    centered_target = (target_expanded - target_mean.unsqueeze(1)) * weights
    centered_expert = (expert_values - expert_mean.unsqueeze(1)) * weights
    covariance = (centered_target * centered_expert).sum(dim=1)
    target_energy = centered_target.square().sum(dim=1)
    expert_energy = centered_expert.square().sum(dim=1)
    correlation = covariance / torch.sqrt(
        (target_energy * expert_energy).clamp_min(1e-12)
    )
    finite = torch.isfinite(correlation) & target_energy.gt(1e-12) & expert_energy.gt(
        1e-12
    )
    available = available & finite
    return correlation.masked_fill(~available, -1e4), available


def _expert_oracle_distribution(
    expert_skill: torch.Tensor,
    expert_available: torch.Tensor,
    temperature: float,
    mode: str,
    minimum_edge: float,
    label_smoothing: float,
    activation_edge_scale: float = 0.25,
    maximum_nonbaseline_weight: float = 1.0,
) -> torch.Tensor:
    tau = max(float(temperature), 1e-3)
    distribution: torch.Tensor
    if mode == "soft":
        masked_skill = expert_skill.detach().masked_fill(
            ~expert_available, -1e4
        )
        distribution = torch.softmax(masked_skill / tau, dim=1)
    elif mode == "abstaining_soft":
        masked_skill = expert_skill.detach().masked_fill(
            ~expert_available, -1e4
        )
        baseline_available = expert_available[:, 0]
        baseline_skill = masked_skill[:, :1]
        advantages = masked_skill - baseline_skill
        eligible = expert_available.clone()
        if expert_skill.shape[1] > 1:
            eligible[:, 1:] &= advantages[:, 1:].ge(float(minimum_edge))
        eligible[:, 0] &= baseline_available
        has_eligible = eligible.any(dim=1)
        fallback = expert_available & ~has_eligible[:, None]
        eligible = eligible | fallback
        logits = torch.full_like(masked_skill, -1e4)
        logits[:, 0] = torch.where(
            eligible[:, 0],
            torch.zeros_like(masked_skill[:, 0]),
            logits[:, 0],
        )
        if expert_skill.shape[1] > 1:
            candidate_logits = (
                advantages[:, 1:] - float(minimum_edge)
            ) / tau
            logits[:, 1:] = torch.where(
                eligible[:, 1:],
                candidate_logits,
                logits[:, 1:],
            )
        distribution = torch.softmax(logits, dim=1)
    elif mode == "continuous_abstaining":
        masked_skill = expert_skill.detach().masked_fill(
            ~expert_available, -1e4
        )
        baseline_available = expert_available[:, 0]
        if expert_skill.shape[1] == 1:
            distribution = baseline_available.to(
                dtype=expert_skill.dtype
            )[:, None]
        else:
            baseline_skill = torch.where(
                baseline_available,
                masked_skill[:, 0],
                torch.zeros_like(masked_skill[:, 0]),
            )
            advantages = masked_skill[:, 1:] - baseline_skill[:, None]
            candidate_available = expert_available[:, 1:]
            candidate_logits = advantages.masked_fill(
                ~candidate_available, -1e4
            )
            candidate_weights = torch.softmax(
                candidate_logits / tau, dim=1
            )
            candidate_weights = candidate_weights * candidate_available.to(
                dtype=candidate_weights.dtype
            )
            candidate_weights = candidate_weights / candidate_weights.sum(
                dim=1, keepdim=True
            ).clamp_min(1e-8)
            best_advantage = candidate_logits.max(dim=1).values
            has_candidate = candidate_available.any(dim=1)
            edge_scale = max(float(activation_edge_scale), 1e-6)
            activation = (
                (best_advantage - float(minimum_edge)) / edge_scale
            ).clamp(
                0.0,
                float(np.clip(maximum_nonbaseline_weight, 0.0, 1.0)),
            )
            activation = torch.where(
                has_candidate,
                activation,
                torch.zeros_like(activation),
            )
            activation = torch.where(
                baseline_available,
                activation,
                has_candidate.to(dtype=activation.dtype),
            )
            distribution = torch.cat(
                [
                    (1.0 - activation)[:, None]
                    * baseline_available.to(dtype=activation.dtype)[:, None],
                    activation[:, None] * candidate_weights,
                ],
                dim=1,
            )
            distribution = distribution.masked_fill(
                ~expert_available.any(dim=1)[:, None], 0.0
            )
    elif mode == "abstaining_best":
        masked_skill = expert_skill.detach().masked_fill(
            ~expert_available, -1e4
        )
        best_all = masked_skill.argmax(dim=1)
        chosen = best_all.clone()
        if expert_skill.shape[1] > 1:
            nonbaseline_skill, nonbaseline_position = masked_skill[:, 1:].max(
                dim=1
            )
            baseline_available = expert_available[:, 0]
            nonbaseline_available = expert_available[:, 1:].any(dim=1)
            edge = nonbaseline_skill - masked_skill[:, 0]
            use_nonbaseline = (
                nonbaseline_available
                & (~baseline_available | edge.ge(float(minimum_edge)))
            )
            chosen = torch.where(
                use_nonbaseline,
                nonbaseline_position + 1,
                torch.zeros_like(chosen),
            )
            chosen = torch.where(
                baseline_available | nonbaseline_available, chosen, best_all
            )
        distribution = F.one_hot(
            chosen, num_classes=expert_skill.shape[1]
        ).to(dtype=expert_skill.dtype)
        distribution = distribution * expert_available.to(
            dtype=distribution.dtype
        )
    else:
        raise ValueError(f"Unknown expert oracle mode: {mode}")
    smoothing = float(np.clip(label_smoothing, 0.0, 0.25))
    if smoothing <= 0.0:
        return distribution
    uniform = expert_available.to(dtype=distribution.dtype)
    uniform = uniform / uniform.sum(dim=1, keepdim=True).clamp_min(1.0)
    return (1.0 - smoothing) * distribution + smoothing * uniform


def _expert_routing_loss(
    gate: torch.Tensor,
    experts: torch.Tensor,
    expert_masks: torch.Tensor,
    target: torch.Tensor,
    target_mask: torch.Tensor,
    groups: torch.Tensor | None,
    horizon_weights: torch.Tensor,
    temperature: float,
    metric: str = "mse",
    minimum_assets: int = 2,
    nonbaseline_margin: float = 0.0,
    oracle_mode: str = "soft",
    minimum_edge: float = 0.0,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Teach the router which causal expert best matches each training label."""
    losses: list[torch.Tensor] = []
    weights: list[torch.Tensor] = []
    tau = max(float(temperature), 1e-3)
    group_values = [None] if groups is None else torch.unique(groups).tolist()
    for horizon in range(gate.shape[1]):
        if float(horizon_weights[horizon]) <= 0.0:
            continue
        horizon_losses: list[torch.Tensor] = []
        for group in group_values:
            columns = (
                torch.arange(gate.shape[2], device=gate.device)
                if group is None
                else torch.where(groups.eq(group))[0]
            )
            if len(columns) < 2:
                continue
            target_values = target[:, horizon, columns]
            expert_values = experts[:, horizon, columns, :]
            valid = (
                target_mask[:, horizon, columns, None].bool()
                & expert_masks[:, horizon, columns, :].bool()
            )
            expert_skill, expert_available = _cross_sectional_expert_skill(
                expert_values,
                target_values,
                valid,
                metric,
                minimum_assets,
            )
            usable = expert_available.any(dim=1)
            if not usable.any():
                continue
            if float(nonbaseline_margin) != 0.0 and expert_skill.shape[1] > 1:
                expert_skill = expert_skill.clone()
                expert_skill[:, 1:] -= float(nonbaseline_margin)
            oracle_distribution = _expert_oracle_distribution(
                expert_skill,
                expert_available,
                tau,
                oracle_mode,
                minimum_edge,
                label_smoothing,
            )
            selected_gate = gate[:, horizon, columns, :]
            usable_columns = expert_masks[
                :, horizon, columns, :
            ].bool().any(dim=2)
            group_gate = (
                selected_gate
                * usable_columns.to(dtype=selected_gate.dtype).unsqueeze(-1)
            ).sum(dim=1) / usable_columns.sum(dim=1, keepdim=True).clamp_min(1)
            group_gate = group_gate / group_gate.sum(
                dim=1, keepdim=True
            ).clamp_min(1e-8)
            group_gate = group_gate.clamp_min(1e-8)
            cross_entropy = -(
                oracle_distribution * group_gate.log()
            ).sum(dim=1)
            horizon_losses.append(cross_entropy[usable].mean())
        if horizon_losses:
            losses.append(torch.stack(horizon_losses).mean())
            weights.append(horizon_weights[horizon])
    if not losses:
        return gate.sum() * 0.0
    normalized = torch.stack(weights)
    normalized = normalized / normalized.sum()
    return torch.sum(torch.stack(losses) * normalized)


def _precomputed_expert_routing_loss(
    gate: torch.Tensor,
    current_expert_masks: torch.Tensor,
    skill_target: torch.Tensor,
    skill_mask: torch.Tensor,
    groups: torch.Tensor | None,
    horizon_weights: torch.Tensor,
    temperature: float,
    minimum_edge: float,
    label_smoothing: float,
    oracle_mode: str = "abstaining_soft",
    activation_edge_scale: float = 0.25,
    maximum_nonbaseline_weight: float = 1.0,
) -> torch.Tensor:
    """Route from a matured, forward-smoothed expert edge target."""
    losses: list[torch.Tensor] = []
    weights: list[torch.Tensor] = []
    group_values = [None] if groups is None else torch.unique(groups).tolist()
    for horizon in range(gate.shape[1]):
        if float(horizon_weights[horizon]) <= 0.0:
            continue
        horizon_losses: list[torch.Tensor] = []
        for group in group_values:
            columns = (
                torch.arange(gate.shape[2], device=gate.device)
                if group is None
                else torch.where(groups.eq(group))[0]
            )
            if len(columns) < 2:
                continue
            selected_skill = skill_target[:, horizon, columns, :]
            selected_skill_mask = skill_mask[:, horizon, columns, :].bool()
            counts = selected_skill_mask.sum(dim=1)
            expert_skill = (
                selected_skill
                * selected_skill_mask.to(dtype=selected_skill.dtype)
            ).sum(dim=1) / counts.clamp_min(1).to(dtype=selected_skill.dtype)
            current_available = current_expert_masks[
                :, horizon, columns, :
            ].bool().any(dim=1)
            expert_available = counts.gt(0) & current_available
            usable = expert_available.any(dim=1)
            if not usable.any():
                continue
            oracle_distribution = _expert_oracle_distribution(
                expert_skill,
                expert_available,
                temperature,
                oracle_mode,
                minimum_edge,
                label_smoothing,
                activation_edge_scale,
                maximum_nonbaseline_weight,
            )
            selected_gate = gate[:, horizon, columns, :]
            usable_columns = current_expert_masks[
                :, horizon, columns, :
            ].bool().any(dim=2)
            group_gate = (
                selected_gate
                * usable_columns.to(dtype=selected_gate.dtype).unsqueeze(-1)
            ).sum(dim=1) / usable_columns.sum(
                dim=1, keepdim=True
            ).clamp_min(1)
            group_gate = group_gate / group_gate.sum(
                dim=1, keepdim=True
            ).clamp_min(1e-8)
            cross_entropy = -(
                oracle_distribution * group_gate.clamp_min(1e-8).log()
            ).sum(dim=1)
            horizon_losses.append(cross_entropy[usable].mean())
        if horizon_losses:
            losses.append(torch.stack(horizon_losses).mean())
            weights.append(horizon_weights[horizon])
    if not losses:
        return gate.sum() * 0.0
    normalized = torch.stack(weights)
    normalized = normalized / normalized.sum()
    return torch.sum(torch.stack(losses) * normalized)


def _orthogonalize_to_baseline(
    values: torch.Tensor,
    baseline: torch.Tensor,
    mask: torch.Tensor,
    groups: torch.Tensor | None,
) -> torch.Tensor:
    """Remove the contemporaneous baseline projection cross-section by cross-section.

    A naive target-minus-baseline residual contains a mechanical ``-baseline``
    component. Without this projection a network can reduce regression loss by
    learning to cancel momentum, even when it has learned no future information.
    """
    output = torch.zeros_like(values)
    if groups is None:
        column_groups = [
            torch.arange(values.shape[2], device=values.device, dtype=torch.long)
        ]
    else:
        column_groups = [
            torch.nonzero(groups.eq(group), as_tuple=False).flatten()
            for group in torch.unique(groups)
        ]
    for columns in column_groups:
        if columns.numel() == 0:
            continue
        valid = mask[:, :, columns].bool()
        weights = valid.to(dtype=values.dtype)
        counts = weights.sum(dim=2, keepdim=True).clamp_min(1.0)
        selected_values = values[:, :, columns]
        selected_baseline = baseline[:, :, columns]
        value_centered = selected_values - (
            selected_values * weights
        ).sum(dim=2, keepdim=True) / counts
        baseline_centered = selected_baseline - (
            selected_baseline * weights
        ).sum(dim=2, keepdim=True) / counts
        covariance = (
            value_centered * baseline_centered * weights
        ).sum(dim=2, keepdim=True)
        baseline_variance = (
            baseline_centered.square() * weights
        ).sum(dim=2, keepdim=True)
        projection = covariance / baseline_variance.clamp_min(1e-6)
        residual = value_centered - projection * baseline_centered
        residual = residual - (
            residual * weights
        ).sum(dim=2, keepdim=True) / counts
        output[:, :, columns] = torch.where(valid, residual, 0.0)
    return output


def multitask_loss(
    outputs: tuple[torch.Tensor, ...],
    batch: dict[str, torch.Tensor],
    module_index: torch.Tensor,
    cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    prediction_head_mode = str(cfg.get("prediction_head_mode", "unconstrained"))
    asset_expert_gate = None
    sleeve_expert_gate = None
    if len(outputs) == 6 and prediction_head_mode == "adaptive_expert_mixture":
        (
            asset_pred,
            asset_logvar,
            sleeve_pred,
            sleeve_logvar,
            asset_expert_gate,
            sleeve_expert_gate,
        ) = outputs
        sleeve_rank_pred = None
        gate = None
        regime_residual = None
        regime_adjustment_weight = None
        transition_logits = None
        bond_transition_logits = None
        equity_transition_logits = None
        state_payoff_interaction = None
    elif len(outputs) == 4:
        asset_pred, asset_logvar, sleeve_pred, sleeve_logvar = outputs
        sleeve_rank_pred = None
        gate = None
        regime_residual = None
        regime_adjustment_weight = None
        transition_logits = None
        bond_transition_logits = None
        equity_transition_logits = None
        state_payoff_interaction = None
    elif len(outputs) == 6:
        (
            asset_pred,
            asset_logvar,
            sleeve_pred,
            sleeve_logvar,
            sleeve_rank_pred,
            gate,
        ) = outputs
        regime_residual = None
        regime_adjustment_weight = None
        transition_logits = None
        bond_transition_logits = None
        equity_transition_logits = None
        state_payoff_interaction = None
    elif len(outputs) == 7:
        (
            asset_pred,
            asset_logvar,
            sleeve_pred,
            sleeve_logvar,
            sleeve_rank_pred,
            regime_adjustment_weight,
            regime_residual,
        ) = outputs
        gate = None
        transition_logits = None
        bond_transition_logits = None
        equity_transition_logits = None
        state_payoff_interaction = None
    elif len(outputs) == 8:
        (
            asset_pred,
            asset_logvar,
            sleeve_pred,
            sleeve_logvar,
            sleeve_rank_pred,
            transition_logits,
            _,
            state_payoff_interaction,
        ) = outputs
        gate = None
        regime_residual = None
        regime_adjustment_weight = None
        bond_transition_logits = None
        equity_transition_logits = None
    elif len(outputs) == 10:
        (
            asset_pred,
            asset_logvar,
            sleeve_pred,
            sleeve_logvar,
            sleeve_rank_pred,
            bond_transition_logits,
            equity_transition_logits,
            transition_logits,
            _,
            state_payoff_interaction,
        ) = outputs
        gate = None
        regime_residual = None
        regime_adjustment_weight = None
    else:
        raise ValueError(f"Unexpected model output count: {len(outputs)}")
    asset_horizon_weights = _horizon_weights(
        cfg, asset_pred.shape[1], asset_pred.device, "asset"
    )
    sleeve_horizon_weights = _horizon_weights(
        cfg, sleeve_pred.shape[1], sleeve_pred.device, "sleeve"
    )
    asset_module_horizon_weights = _asset_module_horizon_weights(
        cfg, asset_pred.shape[1], asset_pred.device
    )
    asset_module_loss_weights = _asset_module_aggregation_weights(
        cfg, asset_pred.device
    )
    prediction_target_mode = str(cfg.get("prediction_target_mode", "residual_rank"))
    residual_penalty = asset_pred.sum() * 0.0
    if prediction_target_mode in {"direct_rank", "direct_mean"}:
        asset_regression_target = batch["asset_target"]
        sleeve_regression_target = batch["sleeve_target"]
        asset_regression_prediction = asset_pred
        sleeve_regression_prediction = sleeve_pred
        asset_rank_prediction = asset_pred
        sleeve_default_rank_prediction = sleeve_pred
    elif prediction_target_mode == "residual_rank":
        if bool(cfg.get("orthogonal_residual", False)):
            asset_regression_target = _orthogonalize_to_baseline(
                batch["asset_target"],
                batch["asset_baseline"],
                batch["asset_mask"],
                module_index,
            )
            sleeve_regression_target = _orthogonalize_to_baseline(
                batch["sleeve_target"],
                batch["sleeve_baseline"],
                batch["sleeve_mask"],
                None,
            )
            asset_pred = _orthogonalize_to_baseline(
                asset_pred,
                batch["asset_baseline"],
                batch["asset_mask"],
                module_index,
            )
            sleeve_pred = _orthogonalize_to_baseline(
                sleeve_pred,
                batch["sleeve_baseline"],
                batch["sleeve_mask"],
                None,
            )
        else:
            asset_regression_target = batch["asset_target"] - batch["asset_baseline"]
            sleeve_regression_target = batch["sleeve_target"] - batch["sleeve_baseline"]
        asset_regression_prediction = asset_pred
        sleeve_regression_prediction = sleeve_pred
        asset_rank_prediction = batch["asset_baseline"] + asset_pred
        sleeve_default_rank_prediction = batch["sleeve_baseline"] + sleeve_pred
    elif prediction_target_mode == "anchored_rank_boost":
        if bool(cfg.get("orthogonal_residual", False)):
            asset_pred = _orthogonalize_to_baseline(
                asset_pred,
                batch["asset_baseline"],
                batch["asset_mask"],
                module_index,
            )
            sleeve_pred = _orthogonalize_to_baseline(
                sleeve_pred,
                batch["sleeve_baseline"],
                batch["sleeve_mask"],
                None,
            )
        correction_scale = float(cfg.get("training_correction_scale", 1.0))
        asset_rank_prediction = (
            batch["asset_baseline"] + correction_scale * asset_pred
        )
        sleeve_default_rank_prediction = (
            batch["sleeve_baseline"] + correction_scale * sleeve_pred
        )
        asset_regression_prediction = asset_rank_prediction
        sleeve_regression_prediction = sleeve_default_rank_prediction
        asset_regression_target = batch["asset_target"]
        sleeve_regression_target = batch["sleeve_target"]
    else:
        raise ValueError(f"Unknown prediction_target_mode={prediction_target_mode}")
    asset_loss_scale = _fold_training_loss_scale(
        cfg.get("asset_regression_loss_scale_by_module"),
        asset_regression_prediction,
        module_index,
        "asset_regression_loss_scale_by_module",
    )
    sleeve_loss_scale = _fold_training_loss_scale(
        cfg.get("sleeve_regression_loss_scale_by_sleeve"),
        sleeve_regression_prediction,
        None,
        "sleeve_regression_loss_scale_by_sleeve",
    )
    if prediction_target_mode in {
        "direct_mean",
        "residual_rank",
        "anchored_rank_boost",
    }:
        asset_weights = batch["asset_mask"].to(dtype=asset_pred.dtype)
        sleeve_weights = batch["sleeve_mask"].to(dtype=sleeve_pred.dtype)
        asset_energy = (
            asset_pred.square() * asset_weights
        ).sum() / asset_weights.sum().clamp_min(1.0)
        sleeve_energy = (
            sleeve_pred.square() * sleeve_weights
        ).sum() / sleeve_weights.sum().clamp_min(1.0)
        residual_penalty = 0.5 * (asset_energy + sleeve_energy)
    if asset_module_horizon_weights is None:
        asset_reg = _masked_regression_loss(
            asset_regression_prediction / asset_loss_scale,
            asset_logvar,
            asset_regression_target / asset_loss_scale,
            batch["asset_mask"],
            asset_horizon_weights,
            str(cfg.get("regression_loss_mode", "heteroscedastic_nll")),
            float(cfg.get("regression_huber_beta", 1.0)),
            float(cfg.get("uncertainty_nll_weight", 0.25)),
        )
    else:
        module_losses: list[torch.Tensor] = []
        module_weights: list[torch.Tensor] = []
        for module_position in range(len(RISK_SLEEVES)):
            columns = torch.where(module_index.eq(module_position))[0]
            if len(columns) == 0 or float(
                asset_module_loss_weights[module_position]
            ) <= 0.0:
                continue
            module_losses.append(
                _masked_regression_loss(
                    (asset_regression_prediction / asset_loss_scale)[
                        :, :, columns
                    ],
                    asset_logvar[:, :, columns],
                    (asset_regression_target / asset_loss_scale)[:, :, columns],
                    batch["asset_mask"][:, :, columns],
                    asset_module_horizon_weights[module_position],
                    str(
                        cfg.get(
                            "regression_loss_mode", "heteroscedastic_nll"
                        )
                    ),
                    float(cfg.get("regression_huber_beta", 1.0)),
                    float(cfg.get("uncertainty_nll_weight", 0.25)),
                )
            )
            module_weights.append(asset_module_loss_weights[module_position])
        if not module_losses:
            asset_reg = asset_pred.sum() * 0.0
        else:
            normalized = torch.stack(module_weights)
            normalized = normalized / normalized.sum()
            asset_reg = torch.sum(torch.stack(module_losses) * normalized)
    sleeve_reg = _masked_regression_loss(
        sleeve_regression_prediction / sleeve_loss_scale,
        sleeve_logvar,
        sleeve_regression_target / sleeve_loss_scale,
        batch["sleeve_mask"],
        sleeve_horizon_weights,
        str(cfg.get("regression_loss_mode", "heteroscedastic_nll")),
        float(cfg.get("regression_huber_beta", 1.0)),
        float(cfg.get("uncertainty_nll_weight", 0.25)),
    )
    if asset_module_horizon_weights is None:
        asset_rank = _pairwise_rank_loss(
            asset_rank_prediction,
            batch["asset_target"],
            batch["asset_mask"],
            module_index,
            asset_horizon_weights,
            cfg.get(
                "asset_pairwise_minimum_margin",
                cfg.get("asset_pairwise_margin", 1e-6),
            ),
            cfg.get("asset_pairwise_importance_scale"),
            float(cfg.get("pairwise_importance_power", 0.0)),
            float(cfg.get("pairwise_maximum_importance", 1.0)),
        )
    else:
        module_losses = []
        module_weights = []
        for module_position in range(len(RISK_SLEEVES)):
            columns = torch.where(module_index.eq(module_position))[0]
            if len(columns) < 2 or float(
                asset_module_loss_weights[module_position]
            ) <= 0.0:
                continue
            module_losses.append(
                _pairwise_rank_loss(
                    asset_rank_prediction[:, :, columns],
                    batch["asset_target"][:, :, columns],
                    batch["asset_mask"][:, :, columns],
                    None,
                    asset_module_horizon_weights[module_position],
                    cfg.get(
                        "asset_pairwise_minimum_margin",
                        cfg.get("asset_pairwise_margin", 1e-6),
                    ),
                    cfg.get("asset_pairwise_importance_scale"),
                    float(cfg.get("pairwise_importance_power", 0.0)),
                    float(cfg.get("pairwise_maximum_importance", 1.0)),
                )
            )
            module_weights.append(asset_module_loss_weights[module_position])
        if not module_losses:
            asset_rank = asset_pred.sum() * 0.0
        else:
            normalized = torch.stack(module_weights)
            normalized = normalized / normalized.sum()
            asset_rank = torch.sum(torch.stack(module_losses) * normalized)
    if sleeve_rank_pred is None:
        sleeve_rank = _pairwise_rank_loss(
            sleeve_default_rank_prediction,
            batch["sleeve_target"],
            batch["sleeve_mask"],
            None,
            sleeve_horizon_weights,
            cfg.get(
                "sleeve_pairwise_minimum_margin",
                cfg.get("sleeve_pairwise_margin", 1e-6),
            ),
            cfg.get("sleeve_pairwise_importance_scale"),
            float(cfg.get("pairwise_importance_power", 0.0)),
            float(cfg.get("pairwise_maximum_importance", 1.0)),
        )
        rank = 0.5 * (asset_rank + sleeve_rank)
        rank_term = (
            float(cfg.get("asset_pairwise_loss_weight", cfg["rank_loss_weight"])) * asset_rank
            + float(cfg.get("sleeve_pairwise_loss_weight", cfg["rank_loss_weight"])) * sleeve_rank
        )
    else:
        sleeve_rank = _pairwise_rank_loss(
            sleeve_rank_pred,
            batch["sleeve_target"],
            batch["sleeve_mask"],
            None,
            sleeve_horizon_weights,
            cfg.get(
                "sleeve_pairwise_minimum_margin",
                cfg.get("sleeve_pairwise_margin", 0.10),
            ),
            cfg.get("sleeve_pairwise_importance_scale"),
            float(cfg.get("pairwise_importance_power", 0.0)),
            float(cfg.get("pairwise_maximum_importance", 1.0)),
        )
        rank = asset_rank
        rank_term = (
            float(cfg["rank_loss_weight"]) * asset_rank
            + float(cfg.get("sleeve_pairwise_loss_weight", 1.0)) * sleeve_rank
        )
    if asset_module_horizon_weights is None:
        asset_listwise = _listwise_rank_loss(
            asset_rank_prediction,
            batch["asset_target"],
            batch["asset_mask"],
            module_index,
            asset_horizon_weights,
            float(cfg.get("listwise_temperature", 0.35)),
        )
    else:
        module_losses = []
        module_weights = []
        for module_position in range(len(RISK_SLEEVES)):
            columns = torch.where(module_index.eq(module_position))[0]
            if len(columns) < 2 or float(
                asset_module_loss_weights[module_position]
            ) <= 0.0:
                continue
            module_losses.append(
                _listwise_rank_loss(
                    asset_rank_prediction[:, :, columns],
                    batch["asset_target"][:, :, columns],
                    batch["asset_mask"][:, :, columns],
                    None,
                    asset_module_horizon_weights[module_position],
                    float(cfg.get("listwise_temperature", 0.35)),
                )
            )
            module_weights.append(asset_module_loss_weights[module_position])
        if not module_losses:
            asset_listwise = asset_pred.sum() * 0.0
        else:
            normalized = torch.stack(module_weights)
            normalized = normalized / normalized.sum()
            asset_listwise = torch.sum(torch.stack(module_losses) * normalized)
    sleeve_listwise_pred = (
        sleeve_default_rank_prediction
        if sleeve_rank_pred is None
        else sleeve_rank_pred
    )
    sleeve_listwise = _listwise_rank_loss(
        sleeve_listwise_pred,
        batch["sleeve_target"],
        batch["sleeve_mask"],
        None,
        sleeve_horizon_weights,
        float(cfg.get("listwise_temperature", 0.35)),
    )
    listwise = 0.5 * (asset_listwise + sleeve_listwise)
    decision_horizons = [
        int(value)
        for value in cfg.get(
            "decision_horizons_days",
            list(range(1, asset_rank_prediction.shape[1] + 1)),
        )
    ]
    if asset_module_horizon_weights is None:
        asset_decision_utility = _decision_aligned_mean_loss(
            asset_rank_prediction,
            batch["asset_baseline"],
            batch["asset_target"],
            batch["asset_mask"],
            module_index,
            asset_horizon_weights,
            decision_horizons,
            cfg.get("asset_decision_temperature", 0.10),
            cfg.get("decision_transaction_cost_bps", 5.0),
            cfg.get("decision_utility_scale", 0.02),
        )
    else:
        module_losses = []
        module_weights = []
        for module_position in range(len(RISK_SLEEVES)):
            columns = torch.where(module_index.eq(module_position))[0]
            if len(columns) < 2 or float(
                asset_module_loss_weights[module_position]
            ) <= 0.0:
                continue
            module_losses.append(
                _decision_aligned_mean_loss(
                    asset_rank_prediction[:, :, columns],
                    batch["asset_baseline"][:, :, columns],
                    batch["asset_target"][:, :, columns],
                    batch["asset_mask"][:, :, columns],
                    None,
                    asset_module_horizon_weights[module_position],
                    decision_horizons,
                    cfg.get("asset_decision_temperature", 0.10),
                    cfg.get("decision_transaction_cost_bps", 5.0),
                    cfg.get("decision_utility_scale", 0.02),
                )
            )
            module_weights.append(asset_module_loss_weights[module_position])
        if not module_losses:
            asset_decision_utility = asset_pred.sum() * 0.0
        else:
            normalized = torch.stack(module_weights)
            normalized = normalized / normalized.sum()
            asset_decision_utility = torch.sum(
                torch.stack(module_losses) * normalized
            )
    sleeve_decision_utility = _decision_aligned_mean_loss(
        sleeve_default_rank_prediction,
        batch["sleeve_baseline"],
        batch["sleeve_target"],
        batch["sleeve_mask"],
        None,
        sleeve_horizon_weights,
        decision_horizons,
        cfg.get("sleeve_decision_temperature", 0.10),
        cfg.get("decision_transaction_cost_bps", 5.0),
        cfg.get("decision_utility_scale", 0.02),
    )
    uncertainty = 0.5 * (asset_logvar.square().mean() + sleeve_logvar.square().mean())
    if gate is None:
        moe_balance = uncertainty * 0.0
        moe_entropy = uncertainty * 0.0
    else:
        expert_count = gate.shape[1]
        average_load = gate.mean(dim=0)
        moe_balance = expert_count * average_load.square().sum() - 1.0
        moe_entropy = -(gate * gate.clamp_min(1e-8).log()).sum(dim=1).mean()
    if regime_residual is None:
        regime_residual_penalty = uncertainty * 0.0
        mean_regime_adjustment = uncertainty * 0.0
    else:
        regime_residual_penalty = regime_residual.square().mean()
        mean_regime_adjustment = regime_adjustment_weight.mean()
    if transition_logits is None:
        transition_loss = uncertainty * 0.0
        transition_accuracy = uncertainty * 0.0
        bond_transition_accuracy = uncertainty * 0.0
        equity_transition_accuracy = uncertainty * 0.0
        state_payoff_interaction_penalty = uncertainty * 0.0
    else:
        transition_losses: list[torch.Tensor] = []
        transition_accuracies: list[torch.Tensor] = []
        bond_transition_accuracies: list[torch.Tensor] = []
        equity_transition_accuracies: list[torch.Tensor] = []
        transition_weights: list[torch.Tensor] = []
        use_path_occupancy = cfg.get("transition_target", "terminal_state") == "path_occupancy"
        for horizon in range(transition_logits.shape[1]):
            if float(sleeve_horizon_weights[horizon]) <= 0.0:
                continue
            valid = (
                batch["regime_occupancy_mask"][:, horizon].bool()
                if use_path_occupancy
                else batch["regime_target_mask"][:, horizon].bool()
            )
            if not valid.any():
                continue
            logits = transition_logits[:, horizon][valid]
            if use_path_occupancy:
                target_distribution = batch["regime_occupancy_target"][:, horizon][valid]
                smoothing = float(cfg.get("transition_label_smoothing", 0.05))
                target_distribution = (
                    (1.0 - smoothing) * target_distribution
                    + smoothing / target_distribution.shape[1]
                )
                target = target_distribution.argmax(dim=1)
                joint_loss = -(
                    target_distribution * F.log_softmax(logits, dim=1)
                ).sum(dim=1).mean()
            else:
                target = batch["regime_target"][:, horizon][valid].long()
                target_distribution = None
                joint_loss = F.cross_entropy(
                    logits,
                    target,
                    label_smoothing=float(cfg.get("transition_label_smoothing", 0.05)),
                )
            if bond_transition_logits is None or equity_transition_logits is None:
                transition_losses.append(joint_loss)
            else:
                bond_logits = bond_transition_logits[:, horizon][valid]
                equity_logits = equity_transition_logits[:, horizon][valid]
                if target_distribution is None:
                    bond_target = torch.div(target, 3, rounding_mode="floor")
                    equity_target = torch.remainder(target, 3)
                    bond_loss = F.cross_entropy(
                        bond_logits,
                        bond_target,
                        label_smoothing=float(cfg.get("transition_label_smoothing", 0.05)),
                    )
                    equity_loss = F.cross_entropy(
                        equity_logits,
                        equity_target,
                        label_smoothing=float(cfg.get("transition_label_smoothing", 0.05)),
                    )
                else:
                    path_grid = target_distribution.reshape(-1, 3, 3)
                    bond_distribution = path_grid.sum(dim=2)
                    equity_distribution = path_grid.sum(dim=1)
                    bond_target = bond_distribution.argmax(dim=1)
                    equity_target = equity_distribution.argmax(dim=1)
                    bond_loss = -(
                        bond_distribution * F.log_softmax(bond_logits, dim=1)
                    ).sum(dim=1).mean()
                    equity_loss = -(
                        equity_distribution * F.log_softmax(equity_logits, dim=1)
                    ).sum(dim=1).mean()
                transition_losses.append(
                    float(cfg.get("transition_joint_loss_weight", 0.35)) * joint_loss
                    + float(cfg.get("transition_axis_loss_weight", 0.65))
                    * 0.5
                    * (bond_loss + equity_loss)
                )
                bond_transition_accuracies.append(
                    bond_logits.argmax(dim=1).eq(bond_target).float().mean()
                )
                equity_transition_accuracies.append(
                    equity_logits.argmax(dim=1).eq(equity_target).float().mean()
                )
            transition_accuracies.append(logits.argmax(dim=1).eq(target).float().mean())
            transition_weights.append(sleeve_horizon_weights[horizon])
        if transition_losses:
            weights = torch.stack(transition_weights)
            weights = weights / weights.sum()
            transition_loss = torch.sum(torch.stack(transition_losses) * weights)
            transition_accuracy = torch.sum(torch.stack(transition_accuracies) * weights)
            if bond_transition_accuracies:
                bond_transition_accuracy = torch.sum(
                    torch.stack(bond_transition_accuracies) * weights
                )
                equity_transition_accuracy = torch.sum(
                    torch.stack(equity_transition_accuracies) * weights
                )
            else:
                bond_transition_accuracy = transition_accuracy * 0.0
                equity_transition_accuracy = transition_accuracy * 0.0
        else:
            transition_loss = transition_logits.sum() * 0.0
            transition_accuracy = transition_logits.sum() * 0.0
            bond_transition_accuracy = transition_logits.sum() * 0.0
            equity_transition_accuracy = transition_logits.sum() * 0.0
        state_payoff_interaction_penalty = state_payoff_interaction.square().mean()
    if asset_expert_gate is None or sleeve_expert_gate is None:
        expert_gate_deviation = uncertainty * 0.0
        asset_nonbaseline_expert_weight = uncertainty * 0.0
        sleeve_nonbaseline_expert_weight = uncertainty * 0.0
        asset_expert_routing_loss = uncertainty * 0.0
        sleeve_expert_routing_loss = uncertainty * 0.0
    else:
        asset_nonbaseline = 1.0 - asset_expert_gate[..., 0]
        sleeve_nonbaseline = 1.0 - sleeve_expert_gate[..., 0]
        asset_gate_valid = batch["asset_expert_masks"][..., 0].gt(0.0)
        sleeve_gate_valid = batch["sleeve_expert_masks"][..., 0].gt(0.0)
        asset_gate_count = asset_gate_valid.sum().clamp_min(1).to(
            dtype=asset_nonbaseline.dtype
        )
        sleeve_gate_count = sleeve_gate_valid.sum().clamp_min(1).to(
            dtype=sleeve_nonbaseline.dtype
        )
        asset_gate_deviation = (
            asset_nonbaseline.square()
            * asset_gate_valid.to(dtype=asset_nonbaseline.dtype)
        ).sum() / asset_gate_count
        sleeve_gate_deviation = (
            sleeve_nonbaseline.square()
            * sleeve_gate_valid.to(dtype=sleeve_nonbaseline.dtype)
        ).sum() / sleeve_gate_count
        expert_gate_deviation = 0.5 * (
            asset_gate_deviation + sleeve_gate_deviation
        )
        asset_nonbaseline_expert_weight = (
            asset_nonbaseline
            * asset_gate_valid.to(dtype=asset_nonbaseline.dtype)
        ).sum() / asset_gate_count
        sleeve_nonbaseline_expert_weight = (
            sleeve_nonbaseline
            * sleeve_gate_valid.to(dtype=sleeve_nonbaseline.dtype)
        ).sum() / sleeve_gate_count
        routing_temperature = float(cfg.get("expert_oracle_temperature", 0.15))
        routing_metric = str(cfg.get("expert_oracle_metric", "mse"))
        routing_minimum_assets = int(
            cfg.get("expert_oracle_minimum_assets", 2)
        )
        routing_nonbaseline_margin = float(
            cfg.get("expert_oracle_nonbaseline_margin", 0.0)
        )
        routing_oracle_mode = str(
            cfg.get("expert_oracle_mode", "soft")
        )
        routing_minimum_edge = float(
            cfg.get("expert_oracle_minimum_edge", 0.0)
        )
        routing_label_smoothing = float(
            cfg.get("expert_oracle_label_smoothing", 0.0)
        )
        routing_activation_edge_scale = float(
            cfg.get("expert_oracle_activation_edge_scale", 0.25)
        )
        routing_maximum_nonbaseline_weight = float(
            cfg.get("expert_oracle_maximum_nonbaseline_weight", 1.0)
        )
        routing_source = str(
            cfg.get("expert_oracle_source", "same_date_realized")
        )
        if routing_source == "forward_smoothed_edge":
            asset_expert_routing_loss = (
                _precomputed_expert_routing_loss(
                    asset_expert_gate,
                    batch["asset_expert_masks"],
                    batch["asset_expert_skill_target"],
                    batch["asset_expert_skill_mask"],
                    module_index,
                    asset_horizon_weights,
                    routing_temperature,
                    routing_minimum_edge,
                    routing_label_smoothing,
                    routing_oracle_mode,
                    routing_activation_edge_scale,
                    routing_maximum_nonbaseline_weight,
                )
            )
            sleeve_expert_routing_loss = (
                _precomputed_expert_routing_loss(
                    sleeve_expert_gate,
                    batch["sleeve_expert_masks"],
                    batch["sleeve_expert_skill_target"],
                    batch["sleeve_expert_skill_mask"],
                    None,
                    sleeve_horizon_weights,
                    routing_temperature,
                    routing_minimum_edge,
                    routing_label_smoothing,
                    routing_oracle_mode,
                    routing_activation_edge_scale,
                    routing_maximum_nonbaseline_weight,
                )
            )
        elif routing_source == "same_date_realized":
            asset_expert_routing_loss = _expert_routing_loss(
                asset_expert_gate,
                batch["asset_experts"],
                batch["asset_expert_masks"],
                batch["asset_target"],
                batch["asset_mask"],
                module_index,
                asset_horizon_weights,
                routing_temperature,
                routing_metric,
                routing_minimum_assets,
                routing_nonbaseline_margin,
                routing_oracle_mode,
                routing_minimum_edge,
                routing_label_smoothing,
            )
            sleeve_expert_routing_loss = _expert_routing_loss(
                sleeve_expert_gate,
                batch["sleeve_experts"],
                batch["sleeve_expert_masks"],
                batch["sleeve_target"],
                batch["sleeve_mask"],
                None,
                sleeve_horizon_weights,
                routing_temperature,
                routing_metric,
                routing_minimum_assets,
                routing_nonbaseline_margin,
                routing_oracle_mode,
                routing_minimum_edge,
                routing_label_smoothing,
            )
        else:
            raise ValueError(
                f"Unknown expert_oracle_source={routing_source}"
            )
    expert_routing_loss = 0.5 * (
        asset_expert_routing_loss + sleeve_expert_routing_loss
    )
    legacy_routing_weight = float(
        cfg.get("expert_routing_loss_weight", 0.0)
    )
    asset_routing_weight = float(
        cfg.get(
            "asset_expert_routing_loss_weight",
            legacy_routing_weight * 0.5,
        )
    )
    sleeve_routing_weight = float(
        cfg.get(
            "sleeve_expert_routing_loss_weight",
            legacy_routing_weight * 0.5,
        )
    )
    loss = (
        float(cfg["asset_loss_weight"]) * asset_reg
        + float(cfg["sleeve_loss_weight"]) * sleeve_reg
        + rank_term
        + float(cfg["uncertainty_weight"]) * uncertainty
        + float(cfg.get("listwise_rank_loss_weight", 0.0)) * listwise
        + float(cfg.get("asset_decision_loss_weight", 0.0))
        * asset_decision_utility
        + float(cfg.get("sleeve_decision_loss_weight", 0.0))
        * sleeve_decision_utility
        + float(cfg.get("moe_load_balance_weight", 0.0)) * moe_balance
        - float(cfg.get("moe_entropy_weight", 0.0)) * moe_entropy
        + float(cfg.get("regime_residual_penalty_weight", 0.0)) * regime_residual_penalty
        + float(cfg.get("transition_loss_weight", 0.0)) * transition_loss
        + float(cfg.get("state_payoff_interaction_penalty_weight", 0.0))
        * state_payoff_interaction_penalty
        + float(cfg.get("expert_gate_deviation_weight", 0.0))
        * expert_gate_deviation
        + asset_routing_weight * asset_expert_routing_loss
        + sleeve_routing_weight * sleeve_expert_routing_loss
        + float(cfg.get("residual_shrinkage_weight", 0.0)) * residual_penalty
    )
    return loss, {
        "asset_reg": float(asset_reg.detach()),
        "sleeve_reg": float(sleeve_reg.detach()),
        "rank": float(rank.detach()),
        "sleeve_pairwise": float(sleeve_rank.detach()),
        "listwise": float(listwise.detach()),
        "asset_decision_utility_loss": float(
            asset_decision_utility.detach()
        ),
        "sleeve_decision_utility_loss": float(
            sleeve_decision_utility.detach()
        ),
        "moe_balance": float(moe_balance.detach()),
        "moe_entropy": float(moe_entropy.detach()),
        "regime_residual_penalty": float(regime_residual_penalty.detach()),
        "mean_regime_adjustment": float(mean_regime_adjustment.detach()),
        "transition_loss": float(transition_loss.detach()),
        "transition_accuracy": float(transition_accuracy.detach()),
        "bond_transition_accuracy": float(bond_transition_accuracy.detach()),
        "equity_transition_accuracy": float(equity_transition_accuracy.detach()),
        "state_payoff_interaction_penalty": float(state_payoff_interaction_penalty.detach()),
        "expert_gate_deviation": float(expert_gate_deviation.detach()),
        "asset_nonbaseline_expert_weight": float(
            asset_nonbaseline_expert_weight.detach()
        ),
        "sleeve_nonbaseline_expert_weight": float(
            sleeve_nonbaseline_expert_weight.detach()
        ),
        "expert_routing_loss": float(expert_routing_loss.detach()),
        "asset_expert_routing_loss": float(
            asset_expert_routing_loss.detach()
        ),
        "sleeve_expert_routing_loss": float(
            sleeve_expert_routing_loss.detach()
        ),
        "residual_penalty": float(residual_penalty.detach()),
        "uncertainty": float(uncertainty.detach()),
    }


def rank_ic(pred: np.ndarray, target: np.ndarray, mask: np.ndarray, groups: np.ndarray | None = None) -> float:
    values: list[float] = []
    for row in range(pred.shape[0]):
        group_values = [None] if groups is None else np.unique(groups)
        for group in group_values:
            valid = mask[row].astype(bool)
            if group is not None:
                valid &= groups == group
            if valid.sum() < 2:
                continue
            p = pd.Series(pred[row, valid]).rank(method="average").to_numpy(dtype=float)
            y = pd.Series(target[row, valid]).rank(method="average").to_numpy(dtype=float)
            target_has_cross_section = np.std(y) > 1e-12
            if np.std(p) > 1e-12 and target_has_cross_section:
                values.append(float(np.corrcoef(p, y)[0, 1]))
            elif target_has_cross_section:
                values.append(0.0)
    return float(np.nanmean(values)) if values else np.nan

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.itransformer_daily_signal import (
    DailyITransformerPaths,
    build_daily_feature_frame,
    build_daily_sleeve_features,
    load_daily_config,
    load_daily_returns_with_proxy,
)
from src.itransformer_daily_timing import select_timing_features
from src.itransformer_signal import RISK_SLEEVES
from src.point_in_time import transform_point_in_time_features
from src.v1_strategy import load_config as load_v1_config


ROOT = Path(__file__).resolve().parents[1]


def v2_path(cfg: dict[str, Any], key: str) -> Path:
    return ROOT / cfg["outputs"][key]


def _preferred_asset_weights(v1_cfg: dict[str, Any], profile: str, sleeve: str) -> dict[str, float]:
    profile_cfg = v1_cfg["product"]["profiles"][profile]
    raw = profile_cfg.get("asset_overrides", {}).get(sleeve, {}).get("preferred_asset_weights")
    if not raw:
        for item in v1_cfg.get("asset_mapping", {}).values():
            if item.get("module") == sleeve and item.get("preferred_asset_weights"):
                raw = item["preferred_asset_weights"]
                break
    weights = {str(key): max(float(value), 0.0) for key, value in dict(raw or {}).items()}
    total = sum(weights.values())
    if total <= 0:
        raise ValueError(f"No preferred asset weights for {profile}/{sleeve}")
    return {key: value / total for key, value in weights.items()}


def preferred_daily_sleeve_returns(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    timing_cfg = cfg["daily_timing_v2"]
    profile = str(timing_cfg["label_profile"])
    v1_cfg = load_v1_config()
    daily = load_daily_returns_with_proxy(cfg, DailyITransformerPaths())
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce")
    daily["return_1d"] = pd.to_numeric(daily["return_1d"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return_parts: list[pd.DataFrame] = []
    coverage_parts: list[pd.DataFrame] = []
    for sleeve in RISK_SLEEVES:
        weights = _preferred_asset_weights(v1_cfg, profile, sleeve)
        part = daily[
            daily["module"].eq(sleeve) & daily["asset_code"].astype(str).isin(weights)
        ].copy()
        part["anchor_weight"] = part["asset_code"].astype(str).map(weights)
        part = part.dropna(subset=["date", "return_1d", "anchor_weight"])
        part["weighted_return"] = part["return_1d"] * part["anchor_weight"]
        grouped = part.groupby("date", as_index=False).agg(
            weighted_return=("weighted_return", "sum"),
            available_weight=("anchor_weight", "sum"),
        )
        grouped[sleeve] = grouped["weighted_return"] / grouped["available_weight"].where(
            grouped["available_weight"].gt(0)
        )
        return_parts.append(grouped.set_index("date")[[sleeve]])
        coverage_parts.append(
            grouped.assign(sleeve=sleeve)[["date", "sleeve", "available_weight"]]
        )
    returns = pd.concat(return_parts, axis=1).sort_index()
    coverage = pd.concat(coverage_parts, ignore_index=True).sort_values(["date", "sleeve"])
    return returns, coverage


def _compounded_returns(returns: pd.DataFrame, horizon: int, forward: bool) -> pd.DataFrame:
    log_returns = np.log1p(returns.clip(lower=-0.99))
    compounded = np.expm1(log_returns.rolling(horizon, min_periods=horizon).sum())
    return compounded.shift(-horizon) if forward else compounded


def _month_end_mask(dates: pd.Series) -> np.ndarray:
    month_last = dates.groupby(dates.dt.to_period("M")).transform("max")
    return dates.eq(month_last).to_numpy()


def build_v2_dataset(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = cfg or load_daily_config()
    timing_cfg = cfg["daily_timing_v2"]
    features, feature_cols = build_daily_feature_frame(cfg, DailyITransformerPaths())
    sleeve_path = build_daily_sleeve_features(cfg, DailyITransformerPaths())
    missing_path_cols = [
        column for column in sleeve_path.columns if column != "date" and column not in features.columns
    ]
    if missing_path_cols:
        features = features.merge(sleeve_path[["date"] + missing_path_cols], on="date", how="left")
        feature_cols += missing_path_cols
    scaled, transformed_cols = transform_point_in_time_features(
        features,
        feature_cols,
        cfg.get("data", {}).get("preprocessing", {}),
    )
    scaled.insert(0, "date", pd.to_datetime(features["date"], errors="coerce").to_numpy())
    selection_cfg = dict(cfg)
    selection_cfg["daily_timing"] = {
        **cfg["daily_timing"],
        "max_features": int(timing_cfg["max_features"]),
        "train_end": str(timing_cfg["folds"][0]["fit_end"]),
        "required_ohlcv_suffixes": list(
            timing_cfg.get("required_ohlcv_suffixes", [])
        ),
        "required_ohlcv_maximum_missing_rate": float(
            timing_cfg.get("required_ohlcv_maximum_missing_rate", 0.40)
        ),
    }
    selected = select_timing_features(scaled, transformed_cols, selection_cfg)
    scaled = (
        scaled[["date"] + selected]
        .dropna(subset=["date"])
        .sort_values("date")
        .drop_duplicates("date")
        .reset_index(drop=True)
    )
    # Use the investable China trading calendar. The outer feature panel also
    # contains dates where only US assets traded, which are not rebalance days.
    daily_calendar_source = load_daily_returns_with_proxy(cfg, DailyITransformerPaths())
    china_calendar = pd.DatetimeIndex(
        pd.to_datetime(
            daily_calendar_source.loc[
                daily_calendar_source["asset_code"].astype(str).eq("equity_broad"), "date"
            ],
            errors="coerce",
        ).dropna().unique()
    )
    scaled = scaled[scaled["date"].isin(china_calendar)].reset_index(drop=True)
    scaled[selected] = scaled[selected].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    sleeve_returns, coverage = preferred_daily_sleeve_returns(cfg)
    sleeve_returns = sleeve_returns.reindex(pd.DatetimeIndex(scaled["date"])).fillna(0.0)
    horizons = [int(value) for value in timing_cfg["horizons_days"]]
    mean_window = int(timing_cfg["expected_return_window_days"])
    mean_min = int(timing_cfg["expected_return_min_days"])
    vol_window = int(timing_cfg["volatility_window_days"])
    vol_min = int(timing_cfg["volatility_min_days"])
    daily_mean = sleeve_returns.rolling(mean_window, min_periods=mean_min).mean()
    daily_vol = sleeve_returns.rolling(vol_window, min_periods=vol_min).std()

    targets: dict[int, pd.DataFrame] = {}
    raw_future_returns: dict[int, pd.DataFrame] = {}
    baselines: dict[int, pd.DataFrame] = {}
    for horizon in horizons:
        scale_floor = 0.002 * np.sqrt(horizon / 20.0)
        scale = (daily_vol * np.sqrt(horizon)).clip(lower=scale_floor)
        expected = daily_mean * horizon
        raw_future = _compounded_returns(sleeve_returns, horizon, True)
        target = (raw_future - expected) / scale
        baseline = (_compounded_returns(sleeve_returns, horizon, False) - expected) / scale
        target = target.sub(target.mean(axis=1), axis=0).clip(
            -float(timing_cfg["target_clip"]), float(timing_cfg["target_clip"])
        )
        baseline = baseline.sub(baseline.mean(axis=1), axis=0).clip(
            -float(timing_cfg["target_clip"]), float(timing_cfg["target_clip"])
        )
        targets[horizon] = target
        raw_future_returns[horizon] = raw_future
        baselines[horizon] = baseline

    dates = pd.Series(pd.DatetimeIndex(scaled["date"]), index=np.arange(len(scaled)))
    seq_len = int(cfg["data"]["sequence_length_days"])
    max_horizon = max(horizons)
    eval_stride = int(timing_cfg["evaluation_stride_days"])
    month_end = _month_end_mask(dates)
    feature_values = scaled[selected].to_numpy(dtype=np.float32)
    rows: list[dict[str, Any]] = []
    sequences: list[np.ndarray] = []
    target_arrays: list[np.ndarray] = []
    baseline_arrays: list[np.ndarray] = []
    raw_future_arrays: list[np.ndarray] = []
    for idx in range(seq_len - 1, len(scaled)):
        as_of_date = pd.Timestamp(dates.iloc[idx])
        target_values = np.stack(
            [targets[horizon].iloc[idx].to_numpy(dtype=float) for horizon in horizons], axis=0
        )
        baseline_values = np.stack(
            [baselines[horizon].iloc[idx].to_numpy(dtype=float) for horizon in horizons], axis=0
        )
        raw_future_values = np.stack(
            [raw_future_returns[horizon].iloc[idx].to_numpy(dtype=float) for horizon in horizons],
            axis=0,
        )
        has_label = bool(np.isfinite(target_values).all())
        label_end_date = (
            pd.Timestamp(dates.iloc[idx + max_horizon]) if idx + max_horizon < len(dates) else pd.NaT
        )
        rows.append(
            {
                "as_of_date": as_of_date,
                "label_end_date": label_end_date,
                "has_label": has_label,
                "train_anchor": bool(idx % int(timing_cfg["train_stride_days"]) == 0),
                "evaluation_anchor": bool(
                    idx % eval_stride == 0
                    or (bool(timing_cfg["include_month_ends"]) and month_end[idx])
                ),
            }
        )
        sequences.append(feature_values[idx - seq_len + 1 : idx + 1])
        target_arrays.append(target_values.astype(np.float32))
        baseline_arrays.append(baseline_values.astype(np.float32))
        raw_future_arrays.append(raw_future_values.astype(np.float32))

    frame = pd.DataFrame(rows)
    x = np.stack(sequences).astype(np.float32)
    y = np.stack(target_arrays).astype(np.float32)
    baseline = np.stack(baseline_arrays).astype(np.float32)
    raw_future = np.stack(raw_future_arrays).astype(np.float32)
    v2_path(cfg, "timing_v2_frame").parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(v2_path(cfg, "timing_v2_frame"), index=False)
    np.savez_compressed(
        v2_path(cfg, "timing_v2_arrays"),
        x=x,
        y=y,
        baseline=baseline,
        raw_future=raw_future,
    )
    coverage_summary = (
        coverage.groupby("sleeve")["available_weight"]
        .agg(["min", "median", "mean"])
        .reset_index()
        .to_dict(orient="records")
    )
    metadata = {
        "version": "daily_timing_v2_identified_multihorizon",
        "rows": len(frame),
        "labelled_rows": int(frame["has_label"].sum()),
        "start": str(pd.Timestamp(frame["as_of_date"].min()).date()),
        "end": str(pd.Timestamp(frame["as_of_date"].max()).date()),
        "sequence_length_days": seq_len,
        "feature_count": len(selected),
        "feature_columns": selected,
        "target_sleeves": RISK_SLEEVES,
        "horizons_days": horizons,
        "label_profile": str(timing_cfg["label_profile"]),
        "coverage_summary": coverage_summary,
        "target_definition": "future compounded return minus past-only expected return, divided by past-only volatility, then cross-sectionally demeaned",
        "no_future_data_rule": "features, expected returns, volatility and baseline use observations on or before as_of_date; targets start on the next trading day",
    }
    v2_path(cfg, "timing_v2_metadata").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metadata


def load_v2_arrays(
    cfg: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    cfg = cfg or load_daily_config()
    frame = pd.read_parquet(v2_path(cfg, "timing_v2_frame"))
    arrays = np.load(v2_path(cfg, "timing_v2_arrays"))
    metadata = json.loads(v2_path(cfg, "timing_v2_metadata").read_text(encoding="utf-8"))
    return frame, arrays["x"], arrays["y"], arrays["baseline"], arrays["raw_future"], metadata


def nested_fold_masks(
    frame: pd.DataFrame,
    fit_end: str,
    tune_end: str,
    evaluation_end: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    as_of = pd.to_datetime(frame["as_of_date"], errors="coerce")
    label_end = pd.to_datetime(frame["label_end_date"], errors="coerce")
    labelled = frame["has_label"].astype(bool).to_numpy()
    train_anchor = frame["train_anchor"].astype(bool).to_numpy()
    evaluation_anchor = frame["evaluation_anchor"].astype(bool).to_numpy()
    fit_end_ts = pd.Timestamp(fit_end)
    tune_end_ts = pd.Timestamp(tune_end)
    evaluation_end_ts = pd.Timestamp(evaluation_end)
    fit = labelled & train_anchor & label_end.le(fit_end_ts).to_numpy()
    tune = (
        labelled
        & evaluation_anchor
        & as_of.gt(fit_end_ts).to_numpy()
        & label_end.le(tune_end_ts).to_numpy()
    )
    # Evaluation is assigned by decision date; labels may mature after year-end
    # because they are never used to fit or tune the model for this fold.
    evaluation = (
        labelled
        & evaluation_anchor
        & as_of.gt(tune_end_ts).to_numpy()
        & as_of.le(evaluation_end_ts).to_numpy()
    )
    return fit, tune, evaluation


class IdentifiedInvertedTransformer(nn.Module):
    """iTransformer encoder with variable identity and sleeve/horizon query tokens."""

    def __init__(
        self,
        seq_len: int,
        n_features: int,
        n_targets: int,
        d_model: int,
        n_heads: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.n_targets = n_targets
        self.value_embedding = nn.Linear(seq_len, d_model)
        self.feature_identity = nn.Embedding(n_features, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.target_queries = nn.Parameter(torch.randn(1, n_targets, d_model) * 0.02)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.value_embedding(x.transpose(1, 2))
        feature_ids = torch.arange(self.n_features, device=x.device)
        tokens = tokens + self.feature_identity(feature_ids).unsqueeze(0)
        encoded = self.encoder(tokens)
        queries = self.target_queries.expand(x.shape[0], -1, -1)
        attended, _ = self.cross_attention(queries, encoded, encoded, need_weights=False)
        return self.head(self.norm(attended + queries)).squeeze(-1)

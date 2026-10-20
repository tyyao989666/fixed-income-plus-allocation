from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.itransformer_daily_signal import (
    DailyITransformerPaths,
    build_daily_feature_frame,
    build_daily_sleeve_features,
    load_daily_config,
    load_daily_returns_with_proxy,
)
from src.itransformer_signal import RISK_SLEEVES
from src.point_in_time import transform_point_in_time_features


ROOT = Path(__file__).resolve().parents[1]


def timing_path(cfg: dict[str, Any], key: str) -> Path:
    return ROOT / cfg["outputs"][key]


def _product_return(values: np.ndarray) -> float:
    clean = np.nan_to_num(values.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
    return float(np.prod(1.0 + clean) - 1.0)


def daily_sleeve_returns(cfg: dict[str, Any]) -> pd.DataFrame:
    daily = load_daily_returns_with_proxy(cfg, DailyITransformerPaths())
    preferred_rates = {"rate_30y_curve_proxy", "rate_7_10y"}
    parts: list[pd.DataFrame] = []
    for sleeve in RISK_SLEEVES:
        sub = daily[daily["module"].eq(sleeve)].copy()
        if sleeve == "rates":
            sub = sub[sub["asset_code"].astype(str).isin(preferred_rates)]
        part = sub.groupby("date", as_index=False)["return_1d"].mean()
        part["sleeve"] = sleeve
        parts.append(part)
    long = pd.concat(parts, ignore_index=True)
    return (
        long.pivot(index="date", columns="sleeve", values="return_1d")
        .reindex(columns=RISK_SLEEVES)
        .sort_index()
        .fillna(0.0)
    )


def _anchor_mask(dates: pd.Series, stride: int, include_month_ends: bool) -> np.ndarray:
    base = np.arange(len(dates)) % max(stride, 1) == 0
    if not include_month_ends:
        return base
    month_last = dates.groupby(dates.dt.to_period("M")).transform("max")
    return base | dates.eq(month_last).to_numpy()


def select_timing_features(
    scaled: pd.DataFrame,
    transformed_cols: list[str],
    cfg: dict[str, Any],
) -> list[str]:
    timing = cfg["daily_timing"]
    limit = int(timing["max_features"])
    key_bases = [str(value) for value in cfg["data"].get("required_panel_features", [])]
    state_required = [
        column
        for column in transformed_cols
        if any(column == base or column == f"{base}__delta_20" for base in key_bases)
    ]
    path_patterns = (
        "_ret_1d",
        "_mom_20d",
        "_mom_60d",
        "_vol_20d",
        "_vol_60d",
        "_dd_60d",
    )
    path_required = [
        column
        for column in transformed_cols
        if column.startswith(tuple(f"{sleeve}_" for sleeve in RISK_SLEEVES))
        and column.endswith(path_patterns)
        and not column.endswith("__available")
    ]
    train = scaled[
        pd.to_datetime(scaled["date"], errors="coerce").le(
            pd.Timestamp(timing["train_end"])
        )
    ]
    ohlcv_suffixes = tuple(
        str(value) for value in timing.get("required_ohlcv_suffixes", [])
    )
    ohlcv_required: list[str] = []
    if ohlcv_suffixes:
        maximum_missing = float(
            timing.get("required_ohlcv_maximum_missing_rate", 0.40)
        )
        for column in transformed_cols:
            if not column.endswith(ohlcv_suffixes) or column.endswith("__available"):
                continue
            values = pd.to_numeric(train[column], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
            missing = float(values.isna().mean())
            variance = (
                float(values.var(skipna=True))
                if values.notna().sum() > 60
                else 0.0
            )
            if missing <= maximum_missing and variance > 1e-12:
                ohlcv_required.append(column)
    required = list(
        dict.fromkeys(state_required + path_required + ohlcv_required)
    )

    quality: list[tuple[float, float, str]] = []
    for column in transformed_cols:
        if column in required:
            continue
        values = pd.to_numeric(train[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        missing = float(values.isna().mean())
        variance = float(values.var(skipna=True)) if values.notna().sum() > 60 else 0.0
        if missing <= 0.40 and variance > 1e-12:
            quality.append((missing, -variance, column))
    ranked = required + [column for _, _, column in sorted(quality)]
    return ranked[:limit]


def build_timing_dataset(cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    cfg = cfg or load_daily_config()
    timing = cfg["daily_timing"]
    features, feature_cols = build_daily_feature_frame(cfg, DailyITransformerPaths())
    sleeve_path = build_daily_sleeve_features(cfg, DailyITransformerPaths())
    missing_path_cols = [column for column in sleeve_path.columns if column != "date" and column not in features.columns]
    if missing_path_cols:
        features = features.merge(sleeve_path[["date"] + missing_path_cols], on="date", how="left")
        feature_cols = feature_cols + missing_path_cols
    scaled, transformed_cols = transform_point_in_time_features(
        features,
        feature_cols,
        cfg.get("data", {}).get("preprocessing", {}),
    )
    scaled.insert(0, "date", pd.to_datetime(features["date"], errors="coerce").to_numpy())
    selected = select_timing_features(scaled, transformed_cols, cfg)
    scaled = scaled[["date"] + selected].dropna(subset=["date"]).sort_values("date").drop_duplicates("date")

    sleeve = daily_sleeve_returns(cfg).reindex(pd.DatetimeIndex(scaled["date"])).fillna(0.0)
    dates = pd.Series(pd.DatetimeIndex(scaled["date"]), index=np.arange(len(scaled)))
    seq_len = int(cfg["data"]["sequence_length_days"])
    horizon = int(timing["horizon_days"])
    anchors = _anchor_mask(dates, int(timing["anchor_stride_days"]), bool(timing["include_month_ends"]))
    label_end_limit = pd.Timestamp(timing["label_end"])

    rows: list[dict[str, Any]] = []
    values = scaled[selected].to_numpy(dtype=np.float32)
    sleeve_values = sleeve.to_numpy(dtype=float)
    for idx in range(seq_len - 1, len(scaled) - horizon):
        if not anchors[idx]:
            continue
        as_of = pd.Timestamp(dates.iloc[idx])
        label_end = pd.Timestamp(dates.iloc[idx + horizon])
        if label_end > label_end_limit:
            continue
        row: dict[str, Any] = {"as_of_date": as_of, "label_end_date": label_end}
        window = values[idx - seq_len + 1 : idx + 1]
        for step in range(seq_len):
            for feature_idx, column in enumerate(selected):
                row[f"x_t{step:03d}_{column}"] = float(window[step, feature_idx])
        forward = sleeve_values[idx + 1 : idx + 1 + horizon]
        targets = np.array([_product_return(forward[:, j]) for j in range(len(RISK_SLEEVES))], dtype=float)
        for j, sleeve_name in enumerate(RISK_SLEEVES):
            row[f"target_{sleeve_name}"] = float(targets[j])
        row["winner"] = RISK_SLEEVES[int(np.argmax(targets))]
        rows.append(row)

    out = pd.DataFrame(rows)
    path = timing_path(cfg, "timing_dataset")
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    metadata = {
        "version": f"{cfg['version']}_daily_20d_timing",
        "rows": len(out),
        "start": str(pd.Timestamp(out["as_of_date"].min()).date()) if not out.empty else "",
        "end": str(pd.Timestamp(out["as_of_date"].max()).date()) if not out.empty else "",
        "sequence_length_days": seq_len,
        "horizon_days": horizon,
        "anchor_stride_days": int(timing["anchor_stride_days"]),
        "include_month_ends": bool(timing["include_month_ends"]),
        "feature_count": len(selected),
        "feature_columns": selected,
        "target_sleeves": RISK_SLEEVES,
        "split_rule": "train/validation membership is based on label_end_date; test starts after val_end",
        "no_future_data_rule": "every x_t observation is on or before as_of_date; targets begin on the next trading day",
    }
    meta_path = timing_path(cfg, "timing_metadata")
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def load_timing_arrays(cfg: dict[str, Any] | None = None) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    cfg = cfg or load_daily_config()
    frame = pd.read_parquet(timing_path(cfg, "timing_dataset"))
    metadata = json.loads(timing_path(cfg, "timing_metadata").read_text(encoding="utf-8"))
    features = list(metadata["feature_columns"])
    seq_len = int(metadata["sequence_length_days"])
    x_cols = [f"x_t{step:03d}_{column}" for step in range(seq_len) for column in features]
    y_cols = [f"target_{sleeve}" for sleeve in RISK_SLEEVES]
    x = frame[x_cols].to_numpy(dtype=np.float32).reshape(len(frame), seq_len, len(features))
    y = frame[y_cols].to_numpy(dtype=np.float32)
    return frame, x, y, features


def timing_split_masks(frame: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, np.ndarray]:
    as_of = pd.to_datetime(frame["as_of_date"])
    label_end = pd.to_datetime(frame["label_end_date"])
    train_end = pd.Timestamp(cfg["daily_timing"]["train_end"])
    val_end = pd.Timestamp(cfg["daily_timing"]["val_end"])
    return {
        "train": label_end.le(train_end).to_numpy(),
        "val": as_of.gt(train_end).to_numpy() & label_end.le(val_end).to_numpy(),
        "test": as_of.gt(val_end).to_numpy(),
    }

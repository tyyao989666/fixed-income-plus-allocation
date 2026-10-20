from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd


AVAILABILITY_SUFFIX = "__available"


def _rolling_robust_series(
    values: pd.Series,
    window: int,
    min_periods: int,
    clip: float,
) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    rolling = numeric.rolling(window=window, min_periods=min_periods)
    median = rolling.median()
    q25 = rolling.quantile(0.25)
    q75 = rolling.quantile(0.75)

    # The expanding fallback is causal and avoids fitting a scaler on future rows.
    expanding = numeric.expanding(min_periods=2)
    median = median.combine_first(expanding.median())
    q25 = q25.combine_first(expanding.quantile(0.25))
    q75 = q75.combine_first(expanding.quantile(0.75))
    scale = ((q75 - q25) / 1.349).where(lambda x: x.abs() >= 1e-8, 1.0)
    return ((numeric - median) / scale).clip(-clip, clip).fillna(0.0)


def causal_fill_features(
    frame: pd.DataFrame,
    feature_cols: list[str],
    cfg: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Forward-fill known observations and optionally add point-in-time availability flags."""
    cfg = cfg or {}
    add_flags = bool(cfg.get("include_availability_flags", True))
    min_missing_rate = float(cfg.get("availability_flag_min_missing_rate", 0.01))
    out = frame.copy()
    ordered_cols = list(dict.fromkeys(feature_cols))
    result_cols: list[str] = []
    updates: dict[str, pd.Series] = {}

    for col in ordered_cols:
        raw = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        filled = raw.ffill()
        updates[col] = filled
        if col not in result_cols:
            result_cols.append(col)
        if col.endswith(AVAILABILITY_SUFFIX):
            continue
        if add_flags and float(raw.isna().mean()) >= min_missing_rate:
            flag = f"{col}{AVAILABILITY_SUFFIX}"
            if flag in out.columns:
                updates[flag] = pd.to_numeric(out[flag], errors="coerce").ffill().fillna(0.0).clip(0.0, 1.0)
            else:
                updates[flag] = filled.notna().astype(float)
            if flag not in result_cols:
                result_cols.append(flag)

    untouched = out.drop(columns=list(updates), errors="ignore")
    out = pd.concat([untouched, pd.DataFrame(updates, index=out.index)], axis=1)
    return out, result_cols


def rolling_robust_standardize(
    frame: pd.DataFrame,
    feature_cols: list[str],
    cfg: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Causally standardize each row using only the trailing window ending at that row."""
    cfg = cfg or {}
    window = int(cfg.get("window", 36))
    min_periods = int(cfg.get("min_periods", max(3, window // 4)))
    clip = float(cfg.get("clip", 5.0))
    scaled_columns: dict[str, pd.Series] = {}

    for col in feature_cols:
        values = pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if col.endswith(AVAILABILITY_SUFFIX):
            scaled_columns[col] = values.fillna(0.0).clip(0.0, 1.0)
            continue

        scaled_columns[col] = _rolling_robust_series(values, window, min_periods, clip)

    return pd.DataFrame(scaled_columns, index=frame.index)[feature_cols]


def causal_rolling_quantile_clip(
    values: pd.Series,
    window: int,
    min_periods: int,
    lower_quantile: float,
) -> pd.Series:
    """Winsorize a series with trailing quantiles; early rows remain unmodified."""
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    rolling = numeric.rolling(window=window, min_periods=min_periods)
    lower = rolling.quantile(lower_quantile)
    upper = rolling.quantile(1.0 - lower_quantile)
    clipped = numeric.where(lower.isna(), numeric.clip(lower=lower, upper=upper))
    return clipped


def causal_rolling_percentile(
    values: pd.Series,
    window: int,
    min_periods: int,
) -> pd.Series:
    """Return the current observation's trailing percentile without future rows."""
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)

    def current_rank(x: np.ndarray) -> float:
        clean = x[~np.isnan(x)]
        if not len(clean):
            return np.nan
        return float(np.mean(clean <= clean[-1]))

    rolling = numeric.rolling(window=window, min_periods=min_periods).apply(current_rank, raw=True)
    expanding = numeric.expanding(min_periods=min_periods).apply(current_rank, raw=True)
    return rolling.combine_first(expanding).clip(0.0, 1.0)


def _matches_feature_group(column: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, column, flags=re.IGNORECASE) is not None for pattern in patterns)


def _feature_group(column: str, cfg: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    groups = cfg.get("groups", {})
    priority = cfg.get("group_priority", list(groups))
    for name in priority:
        group_cfg = groups.get(name, {})
        if _matches_feature_group(column, list(group_cfg.get("patterns", []))):
            return str(name), group_cfg
    return "default", cfg.get("default_group", {})


def causal_multiscale_transform(
    frame: pd.DataFrame,
    feature_cols: list[str],
    cfg: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Apply feature-specific causal scaling and retain slow-state anchors."""
    cfg = cfg or {}
    clip = float(cfg.get("clip", 5.0))
    transformed: dict[str, pd.Series] = {}
    output_cols: list[str] = []

    for col in list(dict.fromkeys(feature_cols)):
        values = pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        group_name, group_cfg = _feature_group(col, cfg)
        if col.endswith(AVAILABILITY_SUFFIX) or bool(group_cfg.get("passthrough", False)):
            lower = float(group_cfg.get("lower", 0.0))
            upper = float(group_cfg.get("upper", 1.0))
            transformed[col] = values.fillna(0.0).clip(lower, upper)
        else:
            window = int(group_cfg.get("window", cfg.get("window", 756)))
            min_periods = int(group_cfg.get("min_periods", cfg.get("min_periods", 60)))
            transformed[col] = _rolling_robust_series(values, window, min_periods, clip)
        output_cols.append(col)

        for delta_window in group_cfg.get("delta_windows", []):
            delta_window = int(delta_window)
            delta = values.diff(delta_window)
            delta_scale_window = int(group_cfg.get("delta_scale_window", 252))
            delta_min_periods = int(group_cfg.get("delta_min_periods", max(20, delta_scale_window // 4)))
            name = f"{col}__delta_{delta_window}"
            transformed[name] = _rolling_robust_series(delta, delta_scale_window, delta_min_periods, clip)
            output_cols.append(name)

        for percentile_window in group_cfg.get("percentile_windows", []):
            percentile_window = int(percentile_window)
            percentile_min_periods = int(group_cfg.get("percentile_min_periods", max(20, percentile_window // 5)))
            name = f"{col}__pct_{percentile_window}"
            transformed[name] = causal_rolling_percentile(
                values,
                window=percentile_window,
                min_periods=percentile_min_periods,
            ).fillna(0.5)
            output_cols.append(name)

        if group_name in set(cfg.get("history_flag_groups", [])):
            candidate_windows = list(group_cfg.get("percentile_windows", []))
            history_window = int(group_cfg.get("history_window", max(candidate_windows or [1])))
            name = f"{col}__history_fraction"
            transformed[name] = values.notna().rolling(history_window, min_periods=1).sum().div(history_window).clip(0.0, 1.0)
            output_cols.append(name)

    return pd.DataFrame(transformed, index=frame.index)[output_cols], output_cols


def transform_point_in_time_features(
    frame: pd.DataFrame,
    feature_cols: list[str],
    cfg: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    cfg = cfg or {}
    if str(cfg.get("method", "causal_rolling_robust")) == "causal_multiscale_robust":
        return causal_multiscale_transform(frame, feature_cols, cfg)
    return rolling_robust_standardize(frame, feature_cols, cfg), list(feature_cols)


def normalization_metadata(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = cfg or {}
    method = str(cfg.get("method", "causal_rolling_robust"))
    if method == "causal_multiscale_robust":
        groups = {}
        for name, group_cfg in cfg.get("groups", {}).items():
            groups[name] = {
                key: value
                for key, value in group_cfg.items()
                if key in {"window", "min_periods", "delta_windows", "delta_scale_window", "percentile_windows", "percentile_min_periods", "passthrough"}
            }
        return {
            "method": method,
            "groups": groups,
            "future_fill": False,
            "availability_flags": bool(cfg.get("include_availability_flags", True)),
        }
    return {
        "method": method,
        "window": int(cfg.get("window", 36)),
        "min_periods": int(cfg.get("min_periods", 12)),
        "clip": float(cfg.get("clip", 5.0)),
        "future_fill": False,
        "availability_flags": bool(cfg.get("include_availability_flags", True)),
    }

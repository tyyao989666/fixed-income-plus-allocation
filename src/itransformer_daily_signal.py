from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.point_in_time import (
    causal_fill_features,
    causal_rolling_quantile_clip,
    normalization_metadata,
    transform_point_in_time_features,
)
from src.itransformer_signal import RISK_SLEEVES


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data_processed"
REPORTS = ROOT / "output" / "reports"


@dataclass(frozen=True)
class DailyITransformerPaths:
    config: Path = ROOT / "config" / "itransformer_daily_config.yaml"
    daily_returns: Path = PROCESSED / "asset_returns_daily.parquet"
    monthly_returns: Path = PROCESSED / "asset_returns_monthly.parquet"
    daily_features: Path = PROCESSED / "feature_panel_daily_v43.parquet"
    market_state: Path = PROCESSED / "market_state_cards_daily_v1.parquet"
    rate_curves: Path = PROCESSED / "rate_curves_daily_v43.parquet"


def load_daily_config(path: Path | None = None) -> dict[str, Any]:
    with (path or DailyITransformerPaths().config).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def daily_out_path(cfg: dict[str, Any], key: str) -> Path:
    return ROOT / cfg["outputs"][key]


def build_rate_30y_proxy_daily(cfg: dict[str, Any], paths: DailyITransformerPaths | None = None) -> pd.DataFrame:
    paths = paths or DailyITransformerPaths()
    proxy_cfg = cfg.get("synthetic_assets", {}).get("rate_30y_curve_proxy", {})
    if not proxy_cfg.get("enabled", False):
        return pd.DataFrame()
    curves = pd.read_parquet(paths.rate_curves)
    curves["date"] = pd.to_datetime(curves["date"], errors="coerce")
    y_col = str(proxy_cfg.get("yield_column", "treasury_y30"))
    if y_col not in curves.columns:
        return pd.DataFrame()
    daily = curves[["date", y_col]].dropna(subset=["date", y_col]).sort_values("date").copy()
    daily[y_col] = pd.to_numeric(daily[y_col], errors="coerce")
    duration = float(proxy_cfg.get("modified_duration", 18.5))
    carry = daily[y_col].shift(1) / 100.0 / 252.0
    dy = daily[y_col].diff() / 100.0
    ret = (carry - duration * dy).replace([np.inf, -np.inf], np.nan).clip(-0.20, 0.20)
    close = (1.0 + ret.fillna(0.0)).cumprod() * float(proxy_cfg.get("base_close", 100.0))
    return pd.DataFrame(
        {
            "date": daily["date"],
            "asset_code": str(proxy_cfg.get("asset_code", "rate_30y_curve_proxy")),
            "asset_name": str(proxy_cfg.get("asset_name", "30Y国债曲线代理")),
            "module": "rates",
            "close": close.to_numpy(dtype=float),
            "return_1d": ret.to_numpy(dtype=float),
            "data_source": f"synthetic_daily:{y_col}:duration_{duration:g}",
        }
    )


def load_daily_returns_with_proxy(cfg: dict[str, Any], paths: DailyITransformerPaths | None = None) -> pd.DataFrame:
    paths = paths or DailyITransformerPaths()
    daily = pd.read_parquet(paths.daily_returns)
    proxy = build_rate_30y_proxy_daily(cfg, paths)
    if not proxy.empty:
        proxy_code = proxy["asset_code"].iloc[0]
        daily = daily[daily["asset_code"].astype(str).ne(proxy_code)].copy()
        daily = pd.concat([daily, proxy], ignore_index=True, sort=False)
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce")
    daily["return_1d"] = pd.to_numeric(daily["return_1d"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    daily = daily.dropna(subset=["date", "asset_code", "module"]).sort_values(["date", "asset_code"])
    return daily


def load_monthly_sleeve_targets(cfg: dict[str, Any], paths: DailyITransformerPaths | None = None) -> pd.DataFrame:
    paths = paths or DailyITransformerPaths()
    monthly = pd.read_parquet(paths.monthly_returns)
    monthly["date"] = pd.to_datetime(monthly["date"], errors="coerce")
    label_end = cfg.get("data", {}).get("label_end")
    if label_end:
        monthly = monthly[monthly["date"].le(pd.Timestamp(label_end))].copy()
    monthly["return_1m"] = pd.to_numeric(monthly["return_1m"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    monthly.loc[monthly["return_1m"].abs() > 0.80, "return_1m"] = np.nan
    proxy_daily = build_rate_30y_proxy_daily(cfg, paths)
    if not proxy_daily.empty:
        proxy_code = proxy_daily["asset_code"].iloc[0]
        month_last = proxy_daily.set_index("date")["close"].resample("ME").last().dropna().to_frame("close")
        proxy_monthly = pd.DataFrame(
            {
                "date": month_last.index,
                "asset_code": proxy_code,
                "module": "rates",
                "return_1m": month_last["close"].pct_change().to_numpy(dtype=float),
            }
        )
        if label_end:
            proxy_monthly = proxy_monthly[proxy_monthly["date"].le(pd.Timestamp(label_end))].copy()
        monthly = monthly[monthly["asset_code"].astype(str).ne(proxy_code)].copy()
        monthly = pd.concat([monthly, proxy_monthly], ignore_index=True, sort=False)

    preferred_rates = {"rate_30y_curve_proxy", "rate_7_10y"}
    parts: list[pd.DataFrame] = []
    for sleeve in RISK_SLEEVES:
        sub = monthly[monthly["module"].eq(sleeve)].copy()
        if sleeve == "rates":
            sub = sub[sub["asset_code"].astype(str).isin(preferred_rates)]
        part = sub.groupby("date", as_index=False)["return_1m"].mean()
        part["sleeve"] = sleeve
        parts.append(part)
    long = pd.concat(parts, ignore_index=True)
    wide = long.pivot(index="date", columns="sleeve", values="return_1m").reindex(columns=RISK_SLEEVES).sort_index()
    return wide


def selected_daily_panel_columns(features: pd.DataFrame, cfg: dict[str, Any]) -> list[str]:
    prefixes = tuple(cfg["data"].get("feature_prefixes", []))
    excluded = {"date", "month"}
    numeric_cols = [c for c in features.columns if c not in excluded and pd.api.types.is_numeric_dtype(features[c])]
    candidates = [c for c in numeric_cols if not prefixes or c.startswith(prefixes)]
    train = features[pd.to_datetime(features["date"], errors="coerce").le(pd.Timestamp(cfg["data"]["train_end"]))]
    required_cfg = [str(c) for c in cfg["data"].get("required_panel_features", [])]
    required: list[str] = []
    quality = []
    for col in candidates:
        s = pd.to_numeric(train[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        miss = float(s.isna().mean())
        var = float(s.var(skipna=True)) if s.notna().sum() > 20 else 0.0
        if col in required_cfg and var > 1e-12:
            required.append(col)
            continue
        if miss <= 0.40 and var > 1e-12:
            quality.append((miss, -var, col))
    quality = sorted(quality)
    limit = int(cfg["data"].get("max_panel_features", 64))
    ranked = required + [c for _, _, c in quality if c not in required]
    return ranked[:limit]


def select_final_feature_columns(frame: pd.DataFrame, candidate_cols: list[str], cfg: dict[str, Any]) -> list[str]:
    train = frame[pd.to_datetime(frame["date"], errors="coerce").le(pd.Timestamp(cfg["data"]["train_end"]))]
    required_patterns = [str(p) for p in cfg["data"].get("required_transformed_patterns", [])]
    required = [c for c in candidate_cols if any(re.search(pattern, c, flags=re.IGNORECASE) for pattern in required_patterns)]
    quality = []
    for col in candidate_cols:
        if col in required:
            continue
        s = pd.to_numeric(train[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        miss = float(s.isna().mean())
        var = float(s.var(skipna=True)) if s.notna().sum() > 20 else 0.0
        if miss <= 0.50 and var > 1e-12:
            quality.append((miss, -var, col))
    quality = sorted(quality)
    max_total = int(cfg["data"].get("max_base_features", cfg["data"].get("max_total_features", len(quality))))
    ranked = required + [c for _, _, c in quality]
    return ranked[:max_total]


def _max_drawdown(x: pd.Series) -> float:
    nav = (1.0 + x.fillna(0.0)).cumprod()
    dd = nav / nav.cummax() - 1.0
    return float(dd.min()) if len(dd) else 0.0


def build_daily_sleeve_features(cfg: dict[str, Any], paths: DailyITransformerPaths | None = None) -> pd.DataFrame:
    daily = load_daily_returns_with_proxy(cfg, paths)
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
    wide = long.pivot(index="date", columns="sleeve", values="return_1d").reindex(columns=RISK_SLEEVES).sort_index()
    windows = [int(x) for x in cfg["data"].get("daily_windows", [5, 20, 60, 120])]
    out = pd.DataFrame(index=wide.index)
    for sleeve in RISK_SLEEVES:
        ret = wide[sleeve].astype(float)
        out[f"{sleeve}_ret_1d"] = ret
        for w in windows:
            out[f"{sleeve}_mom_{w}d"] = ret.rolling(w, min_periods=max(3, w // 4)).sum()
            out[f"{sleeve}_vol_{w}d"] = ret.rolling(w, min_periods=max(3, w // 4)).std() * np.sqrt(252)
            out[f"{sleeve}_dd_{w}d"] = ret.rolling(w, min_periods=max(3, w // 4)).apply(_max_drawdown, raw=False)
    out = out.reset_index().rename(columns={"index": "date"})
    return out


def build_ohlcv_sleeve_features(cfg: dict[str, Any], paths: DailyITransformerPaths | None = None) -> pd.DataFrame:
    """Build scale-free OHLCV features and aggregate them robustly by sleeve."""
    paths = paths or DailyITransformerPaths()
    feature_cfg = cfg.get("data", {}).get("ohlcv_features", {})
    if not feature_cfg.get("enabled", False):
        return pd.DataFrame(columns=["date"])

    daily = pd.read_parquet(paths.daily_returns)
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce")
    required = ["open", "high", "low", "close", "vol", "amount", "return_1d"]
    for col in required:
        if col not in daily.columns:
            daily[col] = np.nan
        daily[col] = pd.to_numeric(daily[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    daily = daily.dropna(subset=["date", "asset_code", "module"]).sort_values(["asset_code", "date"])

    group = daily.groupby("asset_code", group_keys=False)
    prev_close = group["close"].shift(1)
    daily["overnight_ret"] = daily["open"] / prev_close - 1.0
    daily["intraday_ret"] = daily["close"] / daily["open"] - 1.0
    daily["range_pct"] = daily["high"] / daily["low"] - 1.0
    price_range = daily["high"] - daily["low"]
    daily["close_position"] = ((daily["close"] - daily["low"]) / price_range - 0.5).where(price_range.abs() > 1e-12)

    window = int(feature_cfg.get("normalization_window", 20))
    min_periods = max(5, window // 4)
    log_vol = np.log1p(daily["vol"].clip(lower=0.0))
    log_amount = np.log1p(daily["amount"].clip(lower=0.0))
    vol_mean = log_vol.groupby(daily["asset_code"]).transform(lambda s: s.rolling(window, min_periods=min_periods).mean())
    amount_mean = log_amount.groupby(daily["asset_code"]).transform(lambda s: s.rolling(window, min_periods=min_periods).mean())
    daily["volume_surprise"] = log_vol - vol_mean
    daily["amount_surprise"] = log_amount - amount_mean
    daily["amihud_log"] = np.log1p(daily["return_1d"].abs() / daily["amount"].replace(0.0, np.nan) * 1e8)
    daily["price_volume_interaction"] = daily["return_1d"] * daily["amount_surprise"]

    engineered = [
        "overnight_ret",
        "intraday_ret",
        "range_pct",
        "close_position",
        "volume_surprise",
        "amount_surprise",
        "amihud_log",
        "price_volume_interaction",
    ]
    daily[engineered] = daily[engineered].replace([np.inf, -np.inf], np.nan)
    clip_quantile = float(feature_cfg.get("winsor_quantile", 0.01))
    if 0.0 < clip_quantile < 0.5:
        winsor_window = int(feature_cfg.get("winsor_window", 252))
        winsor_min_periods = int(feature_cfg.get("winsor_min_periods", 20))
        for col in engineered:
            daily[col] = daily.groupby("asset_code", group_keys=False)[col].transform(
                lambda s: causal_rolling_quantile_clip(
                    s,
                    window=winsor_window,
                    min_periods=winsor_min_periods,
                    lower_quantile=clip_quantile,
                )
            )

    median = daily.groupby(["date", "module"])[engineered].median().unstack("module")
    median.columns = [f"{module}_ohlcv_{feature}_median" for feature, module in median.columns]
    out = median.reset_index()
    if feature_cfg.get("include_dispersion", True):
        dispersion = daily.groupby(["date", "module"])[engineered].std(ddof=0).unstack("module")
        dispersion.columns = [f"{module}_ohlcv_{feature}_dispersion" for feature, module in dispersion.columns]
        out = out.merge(dispersion.reset_index(), on="date", how="outer")
    return out.sort_values("date").reset_index(drop=True)


def build_market_state_features(cfg: dict[str, Any], paths: DailyITransformerPaths | None = None) -> pd.DataFrame:
    paths = paths or DailyITransformerPaths()
    if not cfg["data"].get("include_market_state", True) or not paths.market_state.exists():
        return pd.DataFrame(columns=["date"])
    cards = pd.read_parquet(paths.market_state)
    cards["date"] = pd.to_datetime(cards["date"], errors="coerce")
    numeric_cols = [
        c
        for c in cards.columns
        if c != "date" and pd.api.types.is_numeric_dtype(cards[c])
    ]
    state_cols = [c for c in ["cr_state", "r_state", "cb_state", "eq_state", "alt_state", "rb_state"] if c in cards.columns]
    out = cards[["date"] + numeric_cols + state_cols].copy()
    if state_cols:
        out = pd.get_dummies(out, columns=state_cols, prefix=state_cols, dtype=float)
    for c in out.columns:
        if c != "date":
            out[c] = pd.to_numeric(out[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return out


def build_daily_feature_frame(cfg: dict[str, Any] | None = None, paths: DailyITransformerPaths | None = None) -> tuple[pd.DataFrame, list[str]]:
    cfg = cfg or load_daily_config()
    paths = paths or DailyITransformerPaths()
    panel = pd.read_parquet(paths.daily_features)
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce")
    panel = panel.dropna(subset=["date"]).sort_values("date").drop_duplicates("date")
    panel = panel[panel["date"].ge(pd.Timestamp(cfg["data"]["start_date"]))].copy()
    panel_cols = selected_daily_panel_columns(panel, cfg)
    frame = panel[["date"] + panel_cols].copy()

    sleeve = build_daily_sleeve_features(cfg, paths)
    ohlcv = build_ohlcv_sleeve_features(cfg, paths)
    state = build_market_state_features(cfg, paths)
    frame = frame.merge(sleeve, on="date", how="left")
    if not ohlcv.empty:
        frame = frame.merge(ohlcv, on="date", how="left")
    if not state.empty:
        state_cols = ["date"] + [c for c in state.columns if c != "date" and c not in frame.columns]
        frame = frame.merge(state[state_cols], on="date", how="left")
    frame = frame.sort_values("date").drop_duplicates("date")

    candidate_cols = [c for c in frame.columns if c != "date"]
    frame, candidate_cols = causal_fill_features(
        frame,
        candidate_cols,
        cfg.get("data", {}).get("preprocessing", {}),
    )
    feature_cols = select_final_feature_columns(frame, candidate_cols, cfg)
    frame = frame[["date"] + feature_cols].copy()
    for c in feature_cols:
        frame[c] = pd.to_numeric(frame[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return frame[["date"] + feature_cols], feature_cols


def build_daily_path_dataset(cfg: dict[str, Any] | None = None, paths: DailyITransformerPaths | None = None) -> pd.DataFrame:
    cfg = cfg or load_daily_config()
    paths = paths or DailyITransformerPaths()
    features, feature_cols = build_daily_feature_frame(cfg, paths)
    targets = load_monthly_sleeve_targets(cfg, paths)
    horizon = int(cfg["data"]["horizon_months"])
    seq_len = int(cfg["data"]["sequence_length_days"])

    month_ends = pd.Index(targets.index).sort_values()
    label_targets = targets.shift(-horizon)
    preprocessing = cfg.get("data", {}).get("preprocessing", {})
    scaled, transformed_cols = transform_point_in_time_features(features, feature_cols, preprocessing)
    scaled.insert(0, "date", features["date"].to_numpy())
    transformed_cfg = {
        **cfg,
        "data": {
            **cfg["data"],
            "max_base_features": cfg["data"].get("max_total_features", len(transformed_cols)),
        },
    }
    feature_cols = select_final_feature_columns(scaled, transformed_cols, transformed_cfg)
    scaled = scaled[["date"] + feature_cols]

    rows: list[dict[str, Any]] = []
    for as_of in month_ends:
        as_of = pd.Timestamp(as_of)
        target = label_targets.loc[as_of] if as_of in label_targets.index else None
        if target is None or target.isna().any():
            continue
        hist = scaled[scaled["date"].le(as_of)].tail(seq_len)
        if len(hist) < seq_len:
            continue
        row: dict[str, Any] = {"as_of_date": as_of, "label_date": as_of + pd.offsets.MonthEnd(horizon)}
        window = hist[feature_cols].to_numpy(dtype=float)
        for t in range(seq_len):
            for j, col in enumerate(feature_cols):
                row[f"x_t{t:03d}_{col}"] = float(window[t, j])
        for sleeve in RISK_SLEEVES:
            row[f"target_{sleeve}"] = float(target[sleeve])
        row["winner"] = RISK_SLEEVES[int(np.argmax(target.reindex(RISK_SLEEVES).to_numpy(dtype=float)))]
        rows.append(row)

    out = pd.DataFrame(rows)
    path = daily_out_path(cfg, "dataset")
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    meta = {
        "version": cfg["version"],
        "input_frequency": "daily_path",
        "output_frequency": "monthly",
        "seq_len": seq_len,
        "horizon_months": horizon,
        "feature_count": len(feature_cols),
        "feature_columns": feature_cols,
        "target_sleeves": RISK_SLEEVES,
        "rows": len(out),
        "start": str(out["as_of_date"].min().date()) if not out.empty else "",
        "end": str(out["as_of_date"].max().date()) if not out.empty else "",
        "train_end": cfg["data"]["train_end"],
        "val_end": cfg["data"]["val_end"],
        "normalization": normalization_metadata(preprocessing),
    }
    meta_path = daily_out_path(cfg, "metadata")
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def load_daily_dataset_arrays(cfg: dict[str, Any] | None = None) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    cfg = cfg or load_daily_config()
    df = pd.read_parquet(daily_out_path(cfg, "dataset"))
    meta = json.loads(daily_out_path(cfg, "metadata").read_text(encoding="utf-8"))
    feature_cols = list(meta["feature_columns"])
    seq_len = int(meta["seq_len"])
    x_cols = [f"x_t{t:03d}_{col}" for t in range(seq_len) for col in feature_cols]
    y_cols = [f"target_{s}" for s in RISK_SLEEVES]
    x = df[x_cols].to_numpy(dtype=np.float32).reshape(len(df), seq_len, len(feature_cols))
    y = df[y_cols].to_numpy(dtype=np.float32)
    return df, x, y, feature_cols


def split_daily_masks(df: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, np.ndarray]:
    dates = pd.to_datetime(df["as_of_date"])
    train_end = pd.Timestamp(cfg["data"]["train_end"])
    val_end = pd.Timestamp(cfg["data"]["val_end"])
    pred_start = pd.Timestamp(cfg["data"]["prediction_start"])
    return {
        "train": dates.le(train_end).to_numpy(),
        "val": dates.gt(train_end).to_numpy() & dates.le(val_end).to_numpy(),
        "test": dates.gt(val_end).to_numpy(),
        "prediction": dates.ge(pred_start).to_numpy(),
    }

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.itransformer_daily_signal import (
    DailyITransformerPaths,
    build_daily_feature_frame,
    load_daily_config,
    load_daily_returns_with_proxy,
)
from src.itransformer_signal import RISK_SLEEVES, robust_scale


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "daily_rl_overlay_config.yaml"


def load_overlay_config(path: Path | None = None) -> dict[str, Any]:
    with (path or DEFAULT_CONFIG).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def overlay_out_path(cfg: dict[str, Any], key: str) -> Path:
    return ROOT / cfg["outputs"][key]


def _product_return(values: np.ndarray) -> float:
    clean = np.nan_to_num(values.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
    return float(np.prod(1.0 + clean) - 1.0)


def _period_max_drawdown(values: np.ndarray) -> float:
    clean = np.nan_to_num(values.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
    nav = np.cumprod(1.0 + clean)
    peak = np.maximum.accumulate(nav)
    dd = nav / np.where(peak == 0.0, 1.0, peak) - 1.0
    return float(dd.min()) if len(dd) else 0.0


def build_daily_sleeve_return_matrix(cfg: dict[str, Any]) -> pd.DataFrame:
    feature_cfg = load_daily_config(ROOT / cfg["data"]["feature_source_config"])
    daily = load_daily_returns_with_proxy(feature_cfg, DailyITransformerPaths())
    daily["return_1d"] = pd.to_numeric(daily["return_1d"], errors="coerce").replace([np.inf, -np.inf], np.nan)
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
    wide = wide.fillna(0.0)
    start = pd.Timestamp(cfg["data"]["start_date"])
    end = pd.Timestamp(cfg["data"]["label_end"])
    return wide[(wide.index >= start) & (wide.index <= end)].copy()


def future_label_frame(sleeve_returns: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    labels = pd.DataFrame(index=sleeve_returns.index)
    for sleeve in RISK_SLEEVES:
        ret = sleeve_returns[sleeve].astype(float)
        values = ret.to_numpy(dtype=float)
        fut_ret = np.full(len(values), np.nan)
        fut_vol = np.full(len(values), np.nan)
        fut_dd = np.full(len(values), np.nan)
        for i in range(len(values)):
            block = values[i + 1 : i + 1 + horizon_days]
            if len(block) < horizon_days:
                continue
            fut_ret[i] = _product_return(block)
            fut_vol[i] = float(np.nanstd(block, ddof=0) * np.sqrt(252))
            fut_dd[i] = _period_max_drawdown(block)
        labels[f"label_next_{horizon_days}d_return_{sleeve}"] = fut_ret
        labels[f"label_next_{horizon_days}d_vol_{sleeve}"] = fut_vol
        labels[f"label_next_{horizon_days}d_maxdd_{sleeve}"] = fut_dd
    return labels


def decision_dates(dates: pd.Series, frequency: str) -> pd.Series:
    unique = pd.Series(pd.to_datetime(dates).dropna().sort_values().unique())
    if frequency == "daily":
        return unique
    if frequency != "month_end":
        raise ValueError(f"Unsupported decision_frequency: {frequency}")
    grouped = unique.groupby(unique.dt.to_period("M")).max()
    return pd.Series(grouped.to_numpy())


def load_sleeve_weights(cfg: dict[str, Any], decision_index: pd.Series) -> pd.DataFrame:
    profile = str(cfg["data"].get("profile", "balanced"))
    version = str(cfg["data"].get("strategy_version", "no_agent"))
    pattern_key = "agentic_pattern" if version == "agentic" else "no_agent_pattern"
    rel = cfg["weights"][pattern_key].format(profile=profile)
    path = ROOT / rel
    if not path.exists():
        equal = pd.DataFrame({"date": decision_index})
        for sleeve in RISK_SLEEVES:
            equal[f"current_weight_{sleeve}"] = 1.0 / len(RISK_SLEEVES)
        return equal
    weights = pd.read_parquet(path)
    weights["date"] = pd.to_datetime(weights["date"], errors="coerce")
    cols = ["date"] + [s for s in RISK_SLEEVES if s in weights.columns]
    weights = weights[cols].dropna(subset=["date"]).sort_values("date")
    weights["date"] = weights["date"].astype("datetime64[ns]")
    weights = weights.rename(columns={s: f"current_weight_{s}" for s in RISK_SLEEVES})
    frame = pd.DataFrame({"date": pd.to_datetime(decision_index).astype("datetime64[ns]")})
    merged = pd.merge_asof(frame.sort_values("date"), weights.sort_values("date"), on="date", direction="backward")
    for sleeve in RISK_SLEEVES:
        col = f"current_weight_{sleeve}"
        if col not in merged.columns:
            merged[col] = np.nan
    weight_cols = [f"current_weight_{s}" for s in RISK_SLEEVES]
    merged[weight_cols] = merged[weight_cols].ffill().fillna(1.0 / len(RISK_SLEEVES))
    total = merged[weight_cols].sum(axis=1).replace(0.0, np.nan)
    merged[weight_cols] = merged[weight_cols].div(total, axis=0).fillna(1.0 / len(RISK_SLEEVES))
    return merged


def add_baseline_labels(df: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    out = df.copy()
    ret_cols = [f"label_next_{horizon_days}d_return_{s}" for s in RISK_SLEEVES]
    vol_cols = [f"label_next_{horizon_days}d_vol_{s}" for s in RISK_SLEEVES]
    dd_cols = [f"label_next_{horizon_days}d_maxdd_{s}" for s in RISK_SLEEVES]
    weight_cols = [f"current_weight_{s}" for s in RISK_SLEEVES]
    w = out[weight_cols].to_numpy(dtype=float)
    out[f"label_next_{horizon_days}d_baseline_return"] = np.sum(w * out[ret_cols].to_numpy(dtype=float), axis=1)
    out[f"label_next_{horizon_days}d_baseline_vol_proxy"] = np.sum(w * out[vol_cols].to_numpy(dtype=float), axis=1)
    out[f"label_next_{horizon_days}d_baseline_maxdd_proxy"] = np.sum(w * out[dd_cols].to_numpy(dtype=float), axis=1)
    sleeve_rets = out[ret_cols].to_numpy(dtype=float)
    out["label_best_sleeve"] = [RISK_SLEEVES[int(np.nanargmax(row))] for row in sleeve_rets]
    risky = out[
        [
            f"label_next_{horizon_days}d_return_convertible_bonds",
            f"label_next_{horizon_days}d_return_equity",
            f"label_next_{horizon_days}d_return_alternatives",
        ]
    ].mean(axis=1)
    defensive = out[[f"label_next_{horizon_days}d_return_rates", f"label_next_{horizon_days}d_return_credit"]].mean(axis=1)
    out["label_risk_on_minus_defensive"] = risky - defensive
    return out


def build_daily_rl_overlay_dataset(cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    cfg = cfg or load_overlay_config()
    feature_cfg = load_daily_config(ROOT / cfg["data"]["feature_source_config"])
    features, feature_cols = build_daily_feature_frame(feature_cfg, DailyITransformerPaths())
    max_features = int(cfg["data"].get("max_feature_count", len(feature_cols)))
    feature_cols = feature_cols[:max_features]
    features = features[["date"] + feature_cols].copy()
    features["date"] = pd.to_datetime(features["date"], errors="coerce")
    features = features[features["date"].ge(pd.Timestamp(cfg["data"]["start_date"]))].copy()

    train_mask = features["date"].le(pd.Timestamp(cfg["data"]["train_end"]))
    scaled_values, med, mad = robust_scale(
        features.loc[train_mask, feature_cols].to_numpy(dtype=float),
        features[feature_cols].to_numpy(dtype=float),
    )
    scaled = pd.DataFrame(scaled_values, columns=[f"state_{c}" for c in feature_cols])
    scaled.insert(0, "date", features["date"].to_numpy())

    sleeve_returns = build_daily_sleeve_return_matrix(cfg)
    horizon_days = int(cfg["data"]["horizon_days"])
    labels = future_label_frame(sleeve_returns, horizon_days).reset_index().rename(columns={"index": "date"})
    decisions = decision_dates(scaled["date"], str(cfg["data"].get("decision_frequency", "month_end")))
    decisions = decisions[decisions.ge(pd.Timestamp(cfg["data"]["start_date"])) & decisions.le(pd.Timestamp(cfg["data"]["label_end"]))]

    data = pd.DataFrame({"date": decisions})
    data = data.merge(scaled, on="date", how="left")
    data = data.merge(labels, on="date", how="left")
    data = data.merge(load_sleeve_weights(cfg, decisions), on="date", how="left")
    required_labels = [f"label_next_{horizon_days}d_return_{s}" for s in RISK_SLEEVES]
    data = data.dropna(subset=required_labels).sort_values("date").reset_index(drop=True)
    data = add_baseline_labels(data, horizon_days)
    data.insert(1, "profile", str(cfg["data"].get("profile", "balanced")))
    data.insert(2, "strategy_version", str(cfg["data"].get("strategy_version", "no_agent")))

    out_path = overlay_out_path(cfg, "dataset")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(out_path, index=False)

    meta = {
        "version": cfg["version"],
        "rows": len(data),
        "start": str(data["date"].min().date()) if not data.empty else "",
        "end": str(data["date"].max().date()) if not data.empty else "",
        "input_frequency": "daily_features",
        "decision_frequency": cfg["data"].get("decision_frequency", "month_end"),
        "label_frequency": f"next_{horizon_days}_trading_days",
        "profile": cfg["data"].get("profile", "balanced"),
        "strategy_version": cfg["data"].get("strategy_version", "no_agent"),
        "feature_count": len(feature_cols),
        "feature_columns": feature_cols,
        "state_columns": [f"state_{c}" for c in feature_cols],
        "current_weight_columns": [f"current_weight_{s}" for s in RISK_SLEEVES],
        "label_columns": [c for c in data.columns if c.startswith("label_")],
        "scaler_median": med.tolist(),
        "scaler_mad": mad.tolist(),
        "no_future_data_rule": "state_* and current_weight_* use information available at or before date; label_* fields are future training labels only.",
    }
    meta_path = overlay_out_path(cfg, "metadata")
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    report_path = overlay_out_path(cfg, "report_md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "\n".join(
            [
                "# Daily iTransformer + PPO Overlay Dataset",
                "",
                f"- Rows: {meta['rows']}",
                f"- Date range: {meta['start']} to {meta['end']}",
                f"- Input: daily v4.3 panel, daily sleeve returns, OHLCV-style sleeve features, market state card fields.",
                f"- Current policy context: `{meta['profile']}` / `{meta['strategy_version']}` sleeve weights.",
                f"- Labels: next {horizon_days} trading-day sleeve returns, volatility proxy, drawdown proxy, baseline return.",
                "- Leakage rule: model state columns never include label columns; future labels are used only for supervised/RL training rewards.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return data

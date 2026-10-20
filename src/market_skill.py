from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data_processed"
REPORTS = ROOT / "output" / "reports"


@dataclass(frozen=True)
class MarketSkillPaths:
    feature_panel: Path = PROCESSED / "feature_panel_daily_v43.parquet"
    asset_returns: Path = PROCESSED / "asset_returns_daily.parquet"
    rate_curves: Path = PROCESSED / "rate_curves_daily_v43.parquet"
    credit_turnover: Path = PROCESSED / "credit_turnover_daily.parquet"
    credit_tail_risk: Path = PROCESSED / "credit_tail_risk_monthly_v42.parquet"
    alt_features: Path = PROCESSED / "alt_features_daily_v43.parquet"
    cards: Path = PROCESSED / "market_state_cards_daily_v1.parquet"
    views: Path = PROCESSED / "market_skill_views_daily_v1.parquet"
    mechanical_views: Path = PROCESSED / "market_skill_views_mechanical_daily_v1.parquet"
    llm_views: Path = PROCESSED / "market_skill_views_llm_daily_v1.parquet"
    ep_views: Path = PROCESSED / "ep_views_daily_v1.parquet"
    latest_json: Path = REPORTS / "market_state_card_latest.json"
    latest_md: Path = REPORTS / "market_state_card_latest.md"
    view_report: Path = REPORTS / "market_skill_views_report.md"
    llm_view_report: Path = REPORTS / "market_skill_llm_views_report.md"


ALLOWED_VIEW_ASSETS = {
    "rates_long_duration",
    "credit",
    "convertible_bonds",
    "equity",
    "gold",
    "alternatives",
    "cash",
}

BROAD_VIEW_ASSET_MODULE = {
    "rates_long_duration": "rates",
    "credit": "credit",
    "convertible_bonds": "convertible_bonds",
    "equity": "equity",
    "gold": "alternatives",
    "alternatives": "alternatives",
    "cash": "cash",
}

VIEW_COLUMNS = [
    "date",
    "source_asset",
    "target_asset",
    "relation",
    "confidence",
    "raw_confidence",
    "confidence_calibration",
    "reason_tags",
    "view_type",
    "generator",
    "validator_status",
    "validator_reason",
]


def load_config(path: Path | None = None) -> dict[str, Any]:
    cfg_path = path or ROOT / "config" / "market_skill_config.yaml"
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_local_env(path: Path | None = None) -> None:
    env_path = path or ROOT / ".env.local"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _num(row: pd.Series, col: str, default: float = np.nan) -> float:
    val = row.get(col, default)
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _tag(condition: bool, tag: str) -> list[str]:
    return [tag] if condition else []


def _clip01(value: float, default: float = 0.0) -> float:
    if not np.isfinite(value):
        return default
    return min(1.0, max(0.0, float(value)))


def _safe_round(value: Any, ndigits: int = 3) -> float | None:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(val):
        return None
    return round(val, ndigits)


def _pressure_from_change(value: float, high_bp: float) -> float:
    if not np.isfinite(value) or high_bp <= 0:
        return 0.0
    return _clip01(value / high_bp)


def _bucket3(score: float, strong: float, weak: float, labels: tuple[str, str, str]) -> str:
    if score >= strong:
        return labels[0]
    if score <= weak:
        return labels[2]
    return labels[1]


def _jsonable_metrics(items: dict[str, Any], ndigits: int = 4) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in items.items():
        if isinstance(value, (str, bool)) or value is None:
            out[key] = value
        else:
            out[key] = _safe_round(value, ndigits)
    return out


def _load_credit_turnover_features(paths: MarketSkillPaths) -> pd.DataFrame:
    if not paths.credit_turnover.exists():
        return pd.DataFrame(columns=["date"])
    df = pd.read_parquet(paths.credit_turnover)
    if "date" not in df.columns:
        return pd.DataFrame(columns=["date"])
    keep = [
        c
        for c in [
            "date",
            "credit_turnover_amount",
            "credit_turnover_ma3",
            "credit_turnover_ma20",
            "credit_turnover_ratio_3_20",
            "credit_turnover_stress",
        ]
        if c in df.columns
    ]
    out = df[keep].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    return out.dropna(subset=["date"]).sort_values("date")


def _load_credit_tail_features(paths: MarketSkillPaths) -> pd.DataFrame:
    if not paths.credit_tail_risk.exists():
        return pd.DataFrame(columns=["date"])
    df = pd.read_parquet(paths.credit_tail_risk)
    visible_col = "effective_date_conservative" if "effective_date_conservative" in df.columns else "month"
    if visible_col not in df.columns:
        return pd.DataFrame(columns=["date"])
    work = df.copy()
    work["date"] = pd.to_datetime(work[visible_col], errors="coerce")
    numeric_cols = [
        c
        for c in [
            "credit_tail_risk_score_proxy",
            "rating_downgrade_pressure_proxy",
            "proxy_credit_default_count",
            "proxy_credit_default_amount",
            "credit_default_count",
            "credit_default_amount",
            "credit_net_financing_preferred",
            "summary_credit_bond_net_financing",
            "credit_bond_net_financing",
            "proxy_cancel_issuance_count",
            "proxy_cancel_issuance_amount",
        ]
        if c in work.columns
    ]
    if not numeric_cols:
        return pd.DataFrame(columns=["date"])
    tail = work.dropna(subset=["date"]).sort_values("date")
    tail = tail.groupby("date", as_index=False)[numeric_cols].max(numeric_only=True)
    tail = tail.rename(
        columns={
            "credit_tail_risk_score_proxy": "credit_tail_risk_score",
            "rating_downgrade_pressure_proxy": "rating_downgrade_pressure",
        }
    )
    return tail.sort_values("date")


def enrich_market_features(
    df: pd.DataFrame,
    paths: MarketSkillPaths,
    cfg: dict[str, Any] | None = None,
) -> pd.DataFrame:
    cfg = cfg or load_config()
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").astype("datetime64[ns]")
    out = out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    turnover = _load_credit_turnover_features(paths)
    if not turnover.empty:
        turnover["date"] = pd.to_datetime(turnover["date"], errors="coerce").astype("datetime64[ns]")
        out = pd.merge_asof(out, turnover, on="date", direction="backward")

    tail = _load_credit_tail_features(paths)
    if not tail.empty:
        tail["date"] = pd.to_datetime(tail["date"], errors="coerce").astype("datetime64[ns]")
        out = pd.merge_asof(out, tail, on="date", direction="backward")

    alt = _load_alt_detail_features(paths)
    if not alt.empty:
        return_columns = [
            column for column in alt.columns
            if column == "date" or column.endswith("_ret_1d")
        ]
        level_columns = [
            column for column in alt.columns
            if column == "date" or not column.endswith("_ret_1d")
        ]
        out = _merge_asof_fill(out, alt[level_columns], on="date")
        out = _merge_exact_fill(out, alt[return_columns], on="date")

    alt_asset_returns = _load_alt_asset_return_features(paths)
    if not alt_asset_returns.empty:
        out = _merge_exact_fill(out, alt_asset_returns, on="date")

    asset_states = _asset_state_feature_frame(paths, cfg)
    if not asset_states.empty:
        asset_states["date"] = pd.to_datetime(asset_states["date"], errors="coerce").astype("datetime64[ns]")
        out = pd.merge_asof(out, asset_states.sort_values("date"), on="date", direction="backward")

    alt_return_cols = [
        "alt_gold_au9999_ret_1d",
        "alt_nanhua_commodity_ret_1d",
        "alt_nanhua_nonferrous_ret_1d",
        "alt_nanhua_chemical_ret_1d",
        "alt_us_bond_curve_proxy_ret_1d",
    ]
    for col in alt_return_cols:
        if col in out.columns:
            out[col.replace("_ret_1d", "_ret_20d")] = _compound_trailing(
                pd.to_numeric(out[col], errors="coerce"), 20, 10
            )

    return out


def _merge_asof_fill(left: pd.DataFrame, right: pd.DataFrame, on: str = "date") -> pd.DataFrame:
    fill_cols = [c for c in right.columns if c != on]
    existing = [c for c in fill_cols if c in left.columns]
    renamed = right.rename(columns={c: f"{c}__fill" for c in existing})
    merged = pd.merge_asof(left, renamed.sort_values(on), on=on, direction="backward")
    for col in existing:
        fill_col = f"{col}__fill"
        merged[col] = merged[col].where(merged[col].notna(), merged[fill_col])
        merged = merged.drop(columns=[fill_col])
    return merged


def _merge_exact_fill(left: pd.DataFrame, right: pd.DataFrame, on: str = "date") -> pd.DataFrame:
    """Fill point observations only on their exact observation date."""
    fill_cols = [column for column in right.columns if column != on]
    existing = [column for column in fill_cols if column in left.columns]
    renamed = right.rename(columns={column: f"{column}__exact" for column in existing})
    merged = left.merge(renamed, on=on, how="left", sort=False)
    for column in existing:
        exact = f"{column}__exact"
        merged[column] = merged[column].where(merged[column].notna(), merged[exact])
        merged = merged.drop(columns=[exact])
    return merged


def _load_alt_detail_features(paths: MarketSkillPaths) -> pd.DataFrame:
    if not paths.alt_features.exists():
        return pd.DataFrame(columns=["date"])
    df = pd.read_parquet(paths.alt_features)
    if "date" not in df.columns:
        return pd.DataFrame(columns=["date"])
    rename = {
        "nanhua_commodity": "alt_nanhua_commodity",
        "nanhua_nonferrous": "alt_nanhua_nonferrous",
        "nanhua_chemical": "alt_nanhua_chemical",
        "nanhua_commodity_ret_1d": "alt_nanhua_commodity_ret_1d",
        "nanhua_nonferrous_ret_1d": "alt_nanhua_nonferrous_ret_1d",
        "nanhua_chemical_ret_1d": "alt_nanhua_chemical_ret_1d",
        "gold_au9999": "alt_gold_au9999",
        "gold_au9999_ret_1d": "alt_gold_au9999_ret_1d",
        "usd_index": "alt_usd_index",
        "usd_index_ret_1d": "alt_usd_index_ret_1d",
        "us_10y_yield": "alt_us_10y_yield",
        "us_bond_curve_proxy_ret_1d": "alt_us_bond_curve_proxy_ret_1d",
        "reits_distribution_yield": "alt_reits_distribution_yield",
    }
    keep = ["date"] + [c for c in rename if c in df.columns]
    out = df[keep].rename(columns=rename).copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    return out.dropna(subset=["date"]).sort_values("date")


def _load_alt_asset_return_features(paths: MarketSkillPaths) -> pd.DataFrame:
    """Load canonical ALT returns added after the original v4.3 ALT panel was built."""
    if not paths.asset_returns.exists():
        return pd.DataFrame(columns=["date"])
    source = pd.read_parquet(
        paths.asset_returns,
        columns=["date", "asset_code", "return_1d"],
    )
    mapping = {
        "alt_nonferrous": "alt_nanhua_nonferrous_ret_1d",
        "alt_chemical": "alt_nanhua_chemical_ret_1d",
        "alt_us_bond": "alt_us_bond_curve_proxy_ret_1d",
    }
    source = source[source["asset_code"].isin(mapping)].copy()
    if source.empty:
        return pd.DataFrame(columns=["date"])
    source["date"] = pd.to_datetime(source["date"], errors="coerce")
    source["return_1d"] = pd.to_numeric(source["return_1d"], errors="coerce")
    source = source.dropna(subset=["date", "asset_code"]).drop_duplicates(
        ["date", "asset_code"], keep="last"
    )
    wide = source.pivot(index="date", columns="asset_code", values="return_1d")
    wide = wide.rename(columns=mapping).reset_index()
    wide.columns.name = None
    return wide.sort_values("date")


def _compound_trailing(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").clip(lower=-0.99)
    return np.expm1(np.log1p(values).rolling(window, min_periods=min_periods).sum())


def _last_percentile(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if len(finite) < 2:
        return np.nan
    return float((finite <= finite[-1]).mean())


def _replace_rate_30y_with_curve_proxy(
    source: pd.DataFrame,
    paths: MarketSkillPaths,
    cfg: dict[str, Any],
) -> pd.DataFrame:
    """Use the same 30Y duration proxy in state cards as in portfolio construction.

    The raw ``rate_30y`` series is a 10Y+ wealth-index proxy.  V1 portfolio
    weights and the daily iTransformer instead use a 30Y curve change plus a
    fixed modified duration.  Keeping both labels under the same asset name
    created a silent source mismatch, so the raw state-card series is replaced
    here without altering the underlying raw return file.
    """
    proxy_cfg = cfg.get("canonical_assets", {}).get("rate_30y", {})
    if not bool(proxy_cfg.get("enabled", True)) or not paths.rate_curves.exists():
        return source

    raw_code = str(proxy_cfg.get("raw_code_to_replace", "rate_30y"))
    canonical_code = str(proxy_cfg.get("canonical_code", "rate_30y_curve_proxy"))
    yield_column = str(proxy_cfg.get("yield_column", "treasury_y30"))
    duration = float(proxy_cfg.get("modified_duration", 18.5))
    base_close = float(proxy_cfg.get("base_close", 100.0))
    curves = pd.read_parquet(paths.rate_curves)
    if not {"date", yield_column}.issubset(curves.columns):
        return source

    yields = (
        curves[["date", yield_column]]
        .copy()
        .assign(date=lambda frame: pd.to_datetime(frame["date"], errors="coerce"))
        .dropna(subset=["date"])
        .sort_values("date")
        .drop_duplicates("date", keep="last")
    )
    yields[yield_column] = pd.to_numeric(yields[yield_column], errors="coerce")
    # Curve values are stored in percentage points (for example 2.25, not
    # 0.0225). Match the monthly V1 proxy convention: daily carry plus a
    # duration return from the percentage-point yield change.
    delta_yield = yields[yield_column].diff() / 100.0
    daily_carry = yields[yield_column].shift(1) / 100.0 / 252.0
    proxy_return = (daily_carry - duration * delta_yield).clip(-0.15, 0.15)
    proxy = pd.DataFrame(
        {
            "date": yields["date"],
            "asset_code": canonical_code,
            "asset_name": str(proxy_cfg.get("asset_name", "30Y国债曲线代理")),
            "module": "rates",
            "sub_module": canonical_code,
            "return_1d": proxy_return,
            "data_source": f"synthetic_curve:{yield_column}:duration_{duration:g}",
        }
    )
    proxy = proxy.dropna(subset=["return_1d"])
    if proxy.empty:
        return source
    return pd.concat(
        [source.loc[source["asset_code"].ne(raw_code)].copy(), proxy],
        ignore_index=True,
        sort=False,
    )


def _asset_state_feature_frame(paths: MarketSkillPaths, cfg: dict[str, Any]) -> pd.DataFrame:
    if not paths.asset_returns.exists():
        return pd.DataFrame(columns=["date", "asset_detail_states"])
    source = pd.read_parquet(paths.asset_returns)
    source = _replace_rate_30y_with_curve_proxy(source, paths, cfg)
    required = {"date", "asset_code", "module", "return_1d"}
    if not required.issubset(source.columns):
        return pd.DataFrame(columns=["date", "asset_detail_states"])
    keep = [
        col
        for col in ["date", "asset_code", "asset_name", "module", "sub_module", "return_1d", "data_source"]
        if col in source.columns
    ]
    source = source[keep].copy()
    source["date"] = pd.to_datetime(source["date"], errors="coerce")
    source["return_1d"] = pd.to_numeric(source["return_1d"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    source = source.dropna(subset=["date", "asset_code", "module"]).sort_values(["asset_code", "date"])

    state_cfg = cfg.get("asset_state_cards", {})
    scales = state_cfg.get("return_scales", {})
    default_scales = {"ret_5d": 0.02, "ret_20d": 0.06, "ret_60d": 0.12, "drawdown_60d": 0.12}
    positive = float(state_cfg.get("positive_score", 0.62))
    negative = float(state_cfg.get("negative_score", 0.38))
    min_coverage = float(state_cfg.get("min_coverage_60d", 0.60))
    min_history = int(state_cfg.get("min_history_days", 20))
    max_staleness_days = int(state_cfg.get("max_staleness_days", 5))
    calendar = pd.Index(sorted(source["date"].dropna().unique()), name="date")
    parts: list[pd.DataFrame] = []

    for asset_code, group in source.groupby("asset_code", sort=False):
        observed = group.sort_values("date").drop_duplicates("date", keep="last").set_index("date")
        metadata = {
            col: observed[col].dropna().iloc[-1] if observed[col].notna().any() else ""
            for col in ["asset_code", "asset_name", "module", "sub_module", "data_source"]
            if col in observed.columns
        }
        work = observed.reindex(calendar).reset_index()
        for col, value in metadata.items():
            work[col] = work[col].fillna(value)
        work["last_observation_date"] = work["date"].where(work["return_1d"].notna()).ffill()
        work["observation_age_days"] = (work["date"] - work["last_observation_date"]).dt.days
        returns = work["return_1d"]
        work["ret_5d"] = _compound_trailing(returns, 5, 3)
        work["ret_20d"] = _compound_trailing(returns, 20, 10)
        work["ret_60d"] = _compound_trailing(returns, 60, 30)
        work["vol_20d"] = returns.rolling(20, min_periods=10).std(ddof=0) * np.sqrt(252.0)
        wealth = (1.0 + returns.fillna(0.0).clip(lower=-0.99)).cumprod()
        work["drawdown_60d"] = wealth / wealth.rolling(60, min_periods=20).max() - 1.0
        work["coverage_60d"] = returns.notna().astype(float).rolling(60, min_periods=1).mean()
        work["history_days"] = returns.notna().cumsum()
        work["momentum_percentile_252d"] = work["ret_20d"].rolling(252, min_periods=126).apply(
            _last_percentile, raw=True
        )

        module = str(work["module"].iloc[-1])
        module_scales = {**default_scales, **dict(scales.get(module, {}))}
        score = pd.Series(0.50, index=work.index, dtype=float)
        score += 0.12 * (work["ret_5d"] / max(float(module_scales["ret_5d"]), 1e-6)).clip(-1.0, 1.0)
        score += 0.22 * (work["ret_20d"] / max(float(module_scales["ret_20d"]), 1e-6)).clip(-1.0, 1.0)
        score += 0.12 * (work["ret_60d"] / max(float(module_scales["ret_60d"]), 1e-6)).clip(-1.0, 1.0)
        score += 0.08 * ((work["momentum_percentile_252d"] - 0.50) * 2.0).fillna(0.0)
        score -= 0.12 * (
            (-work["drawdown_60d"]) / max(float(module_scales["drawdown_60d"]), 1e-6)
        ).clip(0.0, 1.0)
        available = (
            work["coverage_60d"].ge(min_coverage)
            & work["history_days"].ge(min_history)
            & work["observation_age_days"].le(max_staleness_days)
        )
        work["raw_score"] = score.clip(0.0, 1.0).where(available, 0.50)
        work["available"] = available
        parts.append(work)

    detail = pd.concat(parts, ignore_index=True)
    detail["rank_score"] = detail["raw_score"].where(detail["available"])
    detail["module_rank"] = detail.groupby(["date", "module"])["rank_score"].rank(pct=True, method="average")
    detail["module_rank"] = detail["module_rank"].fillna(0.50)
    detail["score"] = (
        detail["raw_score"] + 0.10 * ((detail["module_rank"] - 0.50) * 2.0).fillna(0.0)
    ).clip(0.0, 1.0)
    detail["state"] = np.select(
        [~detail["available"], detail["score"].ge(positive), detail["score"].le(negative)],
        ["unavailable", "positive", "negative"],
        default="neutral",
    )

    def make_card(group: pd.DataFrame) -> dict[str, Any]:
        assets: dict[str, Any] = {}
        for row in group.itertuples(index=False):
            tags: list[str] = []
            if row.state == "positive":
                tags.append(f"{row.asset_code}_trend_positive")
            elif row.state == "negative":
                tags.append(f"{row.asset_code}_trend_negative")
            elif row.state == "unavailable":
                tags.append(f"{row.asset_code}_insufficient_history")
            if np.isfinite(row.observation_age_days) and row.observation_age_days > 0:
                tags.append(f"{row.asset_code}_stale_{int(row.observation_age_days)}d")
            if np.isfinite(row.drawdown_60d) and row.drawdown_60d <= -0.08:
                tags.append(f"{row.asset_code}_drawdown_pressure")
            assets[str(row.asset_code)] = {
                "asset_code": str(row.asset_code),
                "label": str(getattr(row, "asset_name", row.asset_code) or row.asset_code),
                "module": str(row.module),
                "sub_module": str(getattr(row, "sub_module", "") or ""),
                "state": str(row.state),
                "score": _safe_round(row.score, 4),
                "module_rank": _safe_round(row.module_rank, 4),
                "metrics": {
                    "ret_5d": _safe_round(row.ret_5d, 6),
                    "ret_20d": _safe_round(row.ret_20d, 6),
                    "ret_60d": _safe_round(row.ret_60d, 6),
                    "vol_20d": _safe_round(row.vol_20d, 6),
                    "drawdown_60d": _safe_round(row.drawdown_60d, 6),
                    "momentum_percentile_252d": _safe_round(row.momentum_percentile_252d, 4),
                    "coverage_60d": _safe_round(row.coverage_60d, 4),
                    "history_days": int(row.history_days),
                    "observation_age_days": _safe_round(row.observation_age_days, 0),
                },
                "tags": tags,
                "data_quality": {
                    "source": str(getattr(row, "data_source", "") or ""),
                    "available": bool(row.available),
                },
            }
        modules: dict[str, Any] = {}
        for module, subset in group.groupby("module"):
            modules[str(module)] = {
                "positive_assets": sorted(subset.loc[subset["state"].eq("positive"), "asset_code"].astype(str)),
                "negative_assets": sorted(subset.loc[subset["state"].eq("negative"), "asset_code"].astype(str)),
                "unavailable_assets": sorted(subset.loc[subset["state"].eq("unavailable"), "asset_code"].astype(str)),
                "mean_score": _safe_round(subset["score"].mean(), 4),
                "dispersion": _safe_round(subset["score"].std(ddof=0), 4),
            }
        return {"assets": assets, "modules": modules, "version": "asset_state_cards_v1_point_in_time"}

    rows = [
        {"date": date, "asset_detail_states": make_card(group)}
        for date, group in detail.groupby("date", sort=True)
    ]
    return pd.DataFrame(rows).sort_values("date")


def _credit_v43(row: pd.Series, cfg: dict[str, Any]) -> dict[str, Any]:
    th = cfg["state_thresholds"]["credit"]
    spread_pct = _num(row, "credit_spread_aaa_3y_proxy_pct_2018_to_t")
    spread_change_20d = _num(row, "credit_spread_aaa_3y_proxy_change_20d", 0.0)
    aap_pct = _num(row, "credit_spread_aap_3y_proxy_pct_2018_to_t", spread_pct)
    aa_pct = _num(row, "credit_spread_aa_3y_proxy_pct_2018_to_t", spread_pct)
    aa_aaa_spread = _num(row, "credit_spread_aa_aaa_proxy", 0.0)
    funding_stress = _num(row, "funding_funding_stress_score_proxy", 0.0)
    turnover_stress = _num(row, "credit_turnover_stress", 0.0)
    turnover_ratio = _num(row, "credit_turnover_ratio_3_20", np.nan)
    tail_risk = _clip01(_num(row, "credit_tail_risk_score", 0.0), 0.0)
    rating_downgrade = _clip01(_num(row, "rating_downgrade_pressure", 0.0), 0.0)
    default_amount = max(_num(row, "proxy_credit_default_amount", _num(row, "credit_default_amount", 0.0)), 0.0)
    net_financing = _num(
        row,
        "credit_net_financing_preferred",
        _num(row, "summary_credit_bond_net_financing", _num(row, "credit_bond_net_financing", np.nan)),
    )

    spread_values = [x for x in [spread_pct, aap_pct, aa_pct] if np.isfinite(x)]
    spread_score = float(np.mean(spread_values)) if spread_values else np.nan
    if not np.isfinite(spread_score):
        spread_score = spread_pct
    widening_pressure = _pressure_from_change(spread_change_20d, th["widening_20d_bp"])
    credit_turnover_stress = max(_clip01(turnover_stress), _clip01(1.0 - turnover_ratio if np.isfinite(turnover_ratio) else 0.0))
    regional_default_pressure = _clip01(0.65 * tail_risk + 0.25 * rating_downgrade + 0.10 * min(default_amount / 100.0, 1.0))
    tail_risk_score = _clip01(0.65 * tail_risk + 0.20 * rating_downgrade + 0.15 * regional_default_pressure)
    pressure_score = _clip01(
        0.30 * widening_pressure
        + 0.25 * funding_stress
        + 0.20 * credit_turnover_stress
        + 0.15 * regional_default_pressure
        + 0.10 * tail_risk_score
    )
    positive_score = _clip01(0.70 * spread_score + 0.30 * (1.0 - pressure_score))
    negative_score = _clip01(0.45 * pressure_score + 0.25 * (1.0 - spread_score) + 0.20 * tail_risk_score + 0.10 * funding_stress)

    if pressure_score >= max(th["spread_widening_pressure_high"], th["funding_stress_high"]):
        cr_state = "CR4"
    elif negative_score >= th["credit_negative_threshold"] and spread_score <= th["thin_pct"]:
        cr_state = "CR3"
    elif positive_score >= th["credit_positive_threshold"] and pressure_score < th["cr2_caution_threshold"]:
        cr_state = "CR1"
    elif spread_score <= th["thin_pct"]:
        cr_state = "CR3"
    else:
        cr_state = "CR2"

    caution = bool(
        cr_state == "CR2"
        and (
            pressure_score >= th["cr2_caution_threshold"]
            or credit_turnover_stress >= th["turnover_stress_high"]
            or tail_risk_score >= th["tail_risk_high"]
        )
    )
    transition = "neutral"
    if pressure_score >= th["cr2_caution_threshold"] or widening_pressure >= 0.5:
        transition = "deteriorating"
    elif positive_score >= th["credit_positive_threshold"] and pressure_score < 0.35:
        transition = "improving"

    tags: list[str] = []
    tags += _tag(spread_score >= th["cheap_pct"], "credit_spread_compensation_high")
    tags += _tag(spread_score <= th["thin_pct"], "credit_spread_compensation_thin")
    tags += _tag(widening_pressure >= 0.5, "credit_spread_widening")
    tags += _tag(funding_stress >= th["funding_stress_high"], "funding_pressure_high")
    tags += _tag(credit_turnover_stress >= th["turnover_stress_high"], "credit_turnover_stress")
    tags += _tag(regional_default_pressure >= th["regional_default_pressure_high"], "regional_default_pressure")
    tags += _tag(tail_risk_score >= th["tail_risk_high"], "credit_tail_risk_high")
    tags += _tag(caution, "CR2_caution_overlay")

    buckets = {
        "spread_compensation": _bucket3(spread_score, th["cheap_pct"], th["thin_pct"], ("high", "neutral", "thin")),
        "spread_momentum": _bucket3(1.0 - widening_pressure, 0.67, 0.40, ("stable", "neutral", "widening")),
        "turnover": _bucket3(1.0 - credit_turnover_stress, 0.67, 0.40, ("healthy", "neutral", "stressed")),
        "funding": _bucket3(1.0 - funding_stress, 0.67, 0.40, ("easy", "neutral", "tight")),
        "tail_risk": _bucket3(1.0 - tail_risk_score, 0.67, 0.40, ("low", "neutral", "high")),
    }

    return {
        "cr_state": cr_state,
        "credit_tags": tags,
        "liquidity_shock": bool(funding_stress >= th["funding_stress_high"] or tail_risk_score >= th["tail_risk_high"]),
        "credit_spread_score": _clip01(spread_score),
        "spread_widening_pressure": widening_pressure,
        "credit_turnover_stress": credit_turnover_stress,
        "funding_stress_score": _clip01(funding_stress),
        "regional_default_pressure": regional_default_pressure,
        "credit_tail_risk_score": tail_risk_score,
        "CR2_caution_overlay": caution,
        "cr_state_transition": transition,
        "credit_final_buckets": buckets,
        "credit_spread_pct": spread_pct,
        "credit_spread_change_20d": spread_change_20d,
        "credit_aa_aaa_spread": aa_aaa_spread,
        "credit_net_financing": net_financing,
        "credit_data_quality": "v43_proxy_daily_plus_monthly_tail_asof",
    }


def _cross_asset_regime(
    r_state: str,
    eq_state: str,
    rb_state: str,
    rate_change_20d: float,
    breadth_5d: float,
    valuation_pct: float,
    risk_budget_score: float,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    th = cfg["state_thresholds"]
    bond_score = 0.50
    if r_state == "R1":
        bond_score += 0.28
    elif r_state == "R4":
        bond_score -= 0.28
    bond_score += _clip01(max(-rate_change_20d, 0.0) / abs(th["rates"]["bull_20d_bp"]), 0.0) * 0.12
    bond_score -= _clip01(max(rate_change_20d, 0.0) / th["rates"]["bear_20d_bp"], 0.0) * 0.12
    bond_score = _clip01(bond_score)

    equity_score = 0.50
    if eq_state == "EQ1":
        equity_score += 0.25
    elif eq_state == "EQ4":
        equity_score -= 0.25
    if np.isfinite(breadth_5d):
        equity_score += (breadth_5d - 0.5) * 0.35
    if np.isfinite(valuation_pct):
        equity_score -= max(valuation_pct - th["equity"]["valuation_high_pct"], 0.0) * 0.25
        equity_score += max(th["equity"]["valuation_low_pct"] - valuation_pct, 0.0) * 0.15
    if rb_state in {"RB3", "RB4"}:
        equity_score -= 0.10
    equity_score = _clip01(equity_score)

    bond_bucket = _bucket3(
        bond_score,
        th["rates"].get("bond_strong_threshold", 0.58),
        th["rates"].get("bond_weak_threshold", 0.42),
        ("bond_strong", "bond_neutral", "bond_weak"),
    )
    equity_bucket = _bucket3(
        equity_score,
        th["equity"].get("equity_strong_threshold", 0.58),
        th["equity"].get("equity_weak_threshold", 0.42),
        ("equity_strong", "equity_neutral", "equity_weak"),
    )
    regime = f"{bond_bucket}_{equity_bucket}"
    if rb_state in {"RB3", "RB4"} and equity_bucket == "equity_strong":
        hint = "risk_budget_limited_keep_equity_selective"
    elif bond_bucket == "bond_strong" and equity_bucket == "equity_weak":
        hint = "prefer_duration_and_defensive_assets"
    elif bond_bucket == "bond_strong" and equity_bucket == "equity_strong":
        hint = "barbell_duration_with_selective_risk_assets"
    elif bond_bucket == "bond_weak" and equity_bucket == "equity_strong":
        hint = "reduce_duration_keep_equity_cb_beta"
    elif bond_bucket == "bond_weak" and equity_bucket == "equity_weak":
        hint = "de_risk_prefer_cash_like_credit_quality_and_gold"
    else:
        hint = "stay_near_risk_budget_anchor"
    return {
        "bond_score": bond_score,
        "equity_score": equity_score,
        "bond_bucket": bond_bucket,
        "equity_bucket": equity_bucket,
        "stock_bond_regime": regime,
        "policy_hint": hint,
    }


def _alt_asset_state(
    row: pd.Series,
    asset_code: str,
    label: str,
    ret_1d_col: str | None = None,
    ret_20d_col: str | None = None,
    extra_metrics: dict[str, Any] | None = None,
    strong: float = 0.03,
    weak: float = -0.03,
) -> dict[str, Any]:
    ret_1d = _num(row, ret_1d_col, np.nan) if ret_1d_col else np.nan
    ret_20d = _num(row, ret_20d_col, np.nan) if ret_20d_col else np.nan
    score = 0.50
    tags: list[str] = []
    if np.isfinite(ret_20d):
        score += np.clip(ret_20d / 0.08, -0.35, 0.35)
        if ret_20d >= strong:
            tags.append(f"{asset_code}_20d_momentum_positive")
        elif ret_20d <= weak:
            tags.append(f"{asset_code}_20d_momentum_negative")
    if np.isfinite(ret_1d):
        score += np.clip(ret_1d / 0.04, -0.10, 0.10)
    score = _clip01(score, 0.50)
    if score >= 0.62:
        state = "positive"
    elif score <= 0.38:
        state = "negative"
    else:
        state = "neutral"
    metrics = {"ret_1d": _safe_round(ret_1d, 5), "ret_20d": _safe_round(ret_20d, 5)}
    if extra_metrics:
        metrics.update(_jsonable_metrics(extra_metrics, ndigits=5))
    return {
        "asset_code": asset_code,
        "label": label,
        "state": state,
        "score": _safe_round(score, 4),
        "tags": tags,
        "metrics": metrics,
        "data_quality": "daily_proxy_or_index",
    }


def _alt_detail_states(row: pd.Series, eq_state: str, cfg: dict[str, Any]) -> dict[str, Any]:
    th = cfg["state_thresholds"].get("alt", {})
    strong = float(th.get("momentum_20d_strong", 0.03))
    weak = float(th.get("momentum_20d_weak", -0.03))

    gold_ret = _num(row, "alt_gold_au9999_ret_1d", 0.0)
    usd_ret = _num(row, "alt_usd_index_ret_1d", 0.0)
    reits_yield = _num(row, "alt_reits_distribution_yield")

    assets = {
        "alt_gold": _alt_asset_state(
            row,
            "alt_gold",
            "黄金",
            "alt_gold_au9999_ret_1d",
            "alt_gold_au9999_ret_20d",
            {"usd_index_ret_1d": usd_ret},
            strong,
            weak,
        ),
        "alt_commodity": _alt_asset_state(
            row,
            "alt_commodity",
            "南华商品",
            "alt_nanhua_commodity_ret_1d",
            "alt_nanhua_commodity_ret_20d",
            None,
            strong,
            weak,
        ),
        "alt_nonferrous": _alt_asset_state(
            row,
            "alt_nonferrous",
            "南华有色",
            "alt_nanhua_nonferrous_ret_1d",
            "alt_nanhua_nonferrous_ret_20d",
            None,
            strong,
            weak,
        ),
        "alt_chemical": _alt_asset_state(
            row,
            "alt_chemical",
            "南华能化",
            "alt_nanhua_chemical_ret_1d",
            "alt_nanhua_chemical_ret_20d",
            None,
            strong,
            weak,
        ),
        "alt_us_bond": _alt_asset_state(
            row,
            "alt_us_bond",
            "美债10Y曲线代理",
            "alt_us_bond_curve_proxy_ret_1d",
            "alt_us_bond_curve_proxy_ret_20d",
            {"us_10y_yield": _num(row, "alt_us_10y_yield", np.nan)},
            strong,
            weak,
        ),
    }

    reits_score = 0.50
    reits_tags: list[str] = []
    if np.isfinite(reits_yield):
        if reits_yield >= th.get("reits_yield_high", 5.0):
            reits_score = 0.65
            reits_tags.append("reits_yield_compensation_high")
        elif reits_yield <= th.get("reits_yield_low", 3.5):
            reits_score = 0.40
            reits_tags.append("reits_yield_compensation_low")
    assets["alt_reits"] = {
        "asset_code": "alt_reits",
        "label": "公募REITs等权",
        "state": "positive" if reits_score >= 0.62 else "negative" if reits_score <= 0.38 else "neutral",
        "score": _safe_round(reits_score, 4),
        "tags": reits_tags,
        "metrics": {"distribution_yield": _safe_round(reits_yield, 4)},
        "data_quality": "distribution_yield_proxy",
    }

    if gold_ret > 0 and usd_ret > 0:
        assets["alt_gold"]["tags"].append("safe_haven_bid")
        assets["alt_gold"]["score"] = _safe_round(max(float(assets["alt_gold"]["score"] or 0.5), 0.66), 4)
        assets["alt_gold"]["state"] = "positive"

    positive = [k for k, v in assets.items() if v["state"] == "positive"]
    negative = [k for k, v in assets.items() if v["state"] == "negative"]
    commodity_complex = ["alt_commodity", "alt_nonferrous", "alt_chemical"]
    commodity_positive = [k for k in commodity_complex if assets[k]["state"] == "positive"]
    defensive_positive = [k for k in ["alt_gold", "alt_us_bond"] if assets[k]["state"] == "positive"]

    if defensive_positive and (eq_state == "EQ4" or "safe_haven_bid" in assets["alt_gold"]["tags"]):
        alt_state = "ALT1"
        summary = "defensive_hedge"
    elif len(commodity_positive) >= 2:
        alt_state = "ALT3"
        summary = "commodity_inflation_beta"
    elif len(negative) >= 3:
        alt_state = "ALT4"
        summary = "broad_alt_weakness"
    else:
        alt_state = "ALT2"
        summary = "neutral_mixed"

    tags = sorted({tag for v in assets.values() for tag in v.get("tags", [])})
    return {
        "alt_state": alt_state,
        "summary": summary,
        "positive_assets": positive,
        "negative_assets": negative,
        "assets": assets,
        "tags": tags,
    }


def _rates_detail_state(row: pd.Series, cfg: dict[str, Any]) -> dict[str, Any]:
    th = cfg["state_thresholds"].get("rates", {})
    y10_chg = _num(row, "rate_treasury_10y_change_20d", 0.0)
    y5_chg = _num(row, "rate_treasury_10y_change_5d", np.nan)
    curve_30y_10y = _num(row, "rate_curve_30y_10y", np.nan)
    curve_10y_1y = _num(row, "rate_curve_10y_1y", np.nan)
    bull = float(th.get("bull_20d_bp", -0.08))
    bear = float(th.get("bear_20d_bp", 0.08))
    score = 0.50
    if np.isfinite(y10_chg):
        score -= np.clip(y10_chg / max(abs(bear), 1e-6), -0.35, 0.35)
    if np.isfinite(curve_30y_10y) and curve_30y_10y <= th.get("curve_flat_30y_10y", 0.35):
        score += 0.08
    score = _clip01(score)
    tags: list[str] = []
    tags += _tag(np.isfinite(y10_chg) and y10_chg <= bull, "rates_10y_rally_20d")
    tags += _tag(np.isfinite(y10_chg) and y10_chg >= bear, "rates_10y_selloff_20d")
    tags += _tag(np.isfinite(curve_30y_10y) and curve_30y_10y <= th.get("curve_flat_30y_10y", 0.35), "rates_30y_10y_curve_flat")
    long_duration = "positive" if score >= 0.60 else "negative" if score <= 0.40 else "neutral"
    return {
        "summary": long_duration,
        "long_duration": {
            "state": long_duration,
            "score": _safe_round(score, 4),
            "metrics": {
                "treasury_10y_change_5d": _safe_round(y5_chg, 5),
                "treasury_10y_change_20d": _safe_round(y10_chg, 5),
                "curve_30y_10y": _safe_round(curve_30y_10y, 5),
                "curve_10y_1y": _safe_round(curve_10y_1y, 5),
            },
            "tags": tags,
        },
    }


def _cb_detail_state(row: pd.Series, cfg: dict[str, Any]) -> dict[str, Any]:
    th = cfg["state_thresholds"].get("cb", {})
    price = _num(row, "cb_cb_price_median", np.nan)
    low_110 = _num(row, "cb_low_price_ratio_110", _num(row, "cb_low_price_ratio_100_proxy", np.nan))
    high_ratio = _num(row, "cb_high_price_ratio_proxy", np.nan)
    premium_raw = _num(row, "cb_cb_premium_median", np.nan)
    premium_unit = str(th.get("premium_input_unit", "percent")).lower()
    premium_ratio = premium_raw / 100.0 if premium_unit == "percent" else premium_raw
    ytm = _num(row, "cb_cb_ytm_median", np.nan)
    callable_pressure = _num(row, "cb_callable_pressure_ratio", 0.0)
    score = 0.50
    if np.isfinite(price):
        if price <= th.get("cheap_price", 115):
            score += 0.22
        elif price >= th.get("expensive_price", 135):
            score -= 0.25
    if np.isfinite(low_110):
        score += np.clip((low_110 - th.get("low_price_ratio_high", 0.25)) * 0.45, -0.10, 0.18)
    if np.isfinite(callable_pressure):
        score -= np.clip(callable_pressure * 0.35, 0.0, 0.15)
    if np.isfinite(premium_ratio):
        premium_start = float(th.get("premium_penalty_start_ratio", 0.45))
        premium_scale = float(th.get("premium_penalty_scale", 0.20))
        premium_cap = float(th.get("premium_penalty_cap", 0.08))
        score -= np.clip(max(premium_ratio - premium_start, 0.0) * premium_scale, 0.0, premium_cap)
    score = _clip01(score)
    tags: list[str] = []
    tags += _tag(np.isfinite(price) and price <= th.get("cheap_price", 115), "cb_price_low")
    tags += _tag(np.isfinite(price) and price >= th.get("expensive_price", 135), "cb_price_expensive")
    tags += _tag(np.isfinite(low_110) and low_110 >= th.get("low_price_ratio_high", 0.25), "cb_low_price_pool_large")
    tags += _tag(callable_pressure >= th.get("callable_pressure_high", 0.20), "cb_callable_pressure_high")
    valuation_state = "positive" if score >= 0.60 else "negative" if score <= 0.40 else "neutral"
    return {
        "summary": valuation_state,
        "valuation": {
            "state": valuation_state,
            "score": _safe_round(score, 4),
            "metrics": {
                "price_median": _safe_round(price, 4),
                "low_price_ratio_110": _safe_round(low_110, 4),
                "high_price_ratio": _safe_round(high_ratio, 4),
                "premium_median": _safe_round(premium_raw, 4),
                "premium_median_pct": _safe_round(premium_raw, 4),
                "premium_median_ratio": _safe_round(premium_ratio, 6),
                "premium_input_unit": premium_unit,
                "ytm_median": _safe_round(ytm, 4),
                "callable_pressure": _safe_round(callable_pressure, 4),
            },
            "tags": tags,
        },
    }


def _equity_detail_state(row: pd.Series, cfg: dict[str, Any]) -> dict[str, Any]:
    th = cfg["state_thresholds"].get("equity", {})
    breadth_1d = _num(row, "equity_sw_l1_positive_1d_ratio", np.nan)
    breadth_5d = _num(row, "equity_sw_l1_positive_5d_ratio", np.nan)
    breadth_20d = _num(row, "equity_sw_l1_positive_20d_ratio", np.nan)
    median_1d = _num(row, "equity_sw_l1_median_1d_ret", np.nan)
    valuation_pct = _num(row, "equity_csi1000_pe_pct_2018_to_t", np.nan)
    margin_change = _num(row, "equity_margin_balance_change_20d", np.nan)
    score = 0.50
    if np.isfinite(breadth_5d):
        score += np.clip((breadth_5d - 0.5) * 0.45, -0.20, 0.20)
    if np.isfinite(breadth_20d):
        score += np.clip((breadth_20d - 0.5) * 0.25, -0.12, 0.12)
    if np.isfinite(valuation_pct):
        score -= np.clip(max(valuation_pct - th.get("valuation_high_pct", 0.75), 0.0) * 0.35, 0.0, 0.12)
        score += np.clip(max(th.get("valuation_low_pct", 0.25) - valuation_pct, 0.0) * 0.20, 0.0, 0.08)
    if np.isfinite(margin_change):
        score += np.clip(margin_change / 1e11, -0.05, 0.05)
    score = _clip01(score)
    tags: list[str] = []
    tags += _tag(np.isfinite(breadth_5d) and breadth_5d >= th.get("breadth_strong", 0.60), "equity_breadth_strong")
    tags += _tag(np.isfinite(breadth_5d) and breadth_5d <= th.get("breadth_weak", 0.35), "equity_breadth_weak")
    tags += _tag(np.isfinite(valuation_pct) and valuation_pct >= th.get("valuation_high_pct", 0.75), "equity_valuation_high")
    tags += _tag(np.isfinite(valuation_pct) and valuation_pct <= th.get("valuation_low_pct", 0.25), "equity_valuation_low")
    state = "positive" if score >= 0.60 else "negative" if score <= 0.40 else "neutral"
    return {
        "summary": state,
        "breadth_valuation": {
            "state": state,
            "score": _safe_round(score, 4),
            "metrics": {
                "breadth_1d": _safe_round(breadth_1d, 4),
                "breadth_5d": _safe_round(breadth_5d, 4),
                "breadth_20d": _safe_round(breadth_20d, 4),
                "median_1d_ret": _safe_round(median_1d, 5),
                "csi1000_pe_pct": _safe_round(valuation_pct, 4),
                "margin_balance_change_20d": _safe_round(margin_change, 2),
            },
            "tags": tags,
        },
    }


def _contextualize_asset_detail(
    asset_detail: dict[str, Any],
    *,
    cr_state: str,
    r_state: str,
    cb_state: str,
    eq_state: str,
    alt_state: str,
    credit: dict[str, Any],
    rates_detail: dict[str, Any],
    cb_detail: dict[str, Any],
    equity_detail: dict[str, Any],
    alt_detail: dict[str, Any],
    filters: dict[str, bool],
) -> dict[str, Any]:
    if not asset_detail:
        return {}
    result = json.loads(json.dumps(asset_detail, ensure_ascii=False))
    broad_states = {
        "rates": r_state,
        "credit": cr_state,
        "convertible_bonds": cb_state,
        "equity": eq_state,
        "alternatives": alt_state,
    }
    broad_details = {
        "rates": rates_detail,
        "credit": {
            "credit_final_buckets": credit.get("credit_final_buckets", {}),
            "CR2_caution_overlay": bool(credit.get("CR2_caution_overlay", False)),
            "liquidity_shock": bool(credit.get("liquidity_shock", False)),
        },
        "convertible_bonds": cb_detail,
        "equity": equity_detail,
        "alternatives": alt_detail,
    }

    for asset_code, item in result.get("assets", {}).items():
        module = item.get("module")
        base_state = item.get("state", "unavailable")
        data_available = bool(
            base_state != "unavailable"
            and item.get("data_quality", {}).get("available", False)
        )
        holding_blocked = False
        holding_block_reason = ""
        overweight_blocked = False
        overweight_block_reason = ""
        if module == "credit" and filters.get("block_credit_overweight", False):
            overweight_blocked = True
            overweight_block_reason = "credit_overweight_blocked"
        if (
            module == "credit"
            and (asset_code.endswith("_aa") or asset_code.endswith("_aa_plus"))
            and filters.get("block_aa_credit", False)
        ):
            holding_blocked = True
            holding_block_reason = "lower_rating_credit_holding_blocked"
            overweight_blocked = True
            overweight_block_reason = "lower_rating_credit_overweight_blocked"
        if module == "rates" and asset_code in {"rate_7_10y", "rate_30y_curve_proxy"} and filters.get("block_long_rate_overweight", False):
            overweight_blocked = True
            overweight_block_reason = "long_duration_overweight_blocked"
        if module == "convertible_bonds" and filters.get("block_cb_overweight", False):
            overweight_blocked = True
            overweight_block_reason = "cb_overweight_blocked"
        if module == "equity" and filters.get("block_equity_overweight", False):
            overweight_blocked = True
            overweight_block_reason = "equity_overweight_blocked"

        holding_eligible = bool(data_available and not holding_blocked)
        overweight_eligible = bool(holding_eligible and not overweight_blocked)

        domain_signal: dict[str, Any] = {}
        if module == "alternatives":
            domain_signal = alt_detail.get("assets", {}).get(asset_code, {})
        elif module == "rates":
            domain_signal = rates_detail.get("long_duration", {}) if asset_code in {"rate_7_10y", "rate_30y_curve_proxy"} else {}
        elif module == "convertible_bonds":
            domain_signal = cb_detail.get("valuation", {})
        elif module == "equity":
            domain_signal = equity_detail.get("breadth_valuation", {})

        domain_state = domain_signal.get("state")
        if base_state == "unavailable":
            decision_state = "unavailable"
        elif holding_blocked:
            decision_state = "holding_blocked"
        elif overweight_blocked and base_state == "positive":
            decision_state = "positive_but_overweight_blocked"
        elif overweight_blocked:
            decision_state = "holding_only_no_overweight"
        elif domain_state == base_state and base_state in {"positive", "negative"}:
            decision_state = f"{base_state}_confirmed"
        elif domain_state in {"positive", "negative"} and base_state in {"positive", "negative"}:
            decision_state = "signal_conflict"
        else:
            decision_state = base_state

        item["domain_context"] = {
            "broad_state": broad_states.get(module),
            "domain_signal_state": domain_state,
            "domain_signal_score": domain_signal.get("score"),
            "data_available": data_available,
            "holding_eligible": holding_eligible,
            "overweight_eligible": overweight_eligible,
            "holding_block_reason": holding_block_reason or None,
            "overweight_block_reason": overweight_block_reason or None,
            "block_reason": holding_block_reason or overweight_block_reason or None,
            "decision_state": decision_state,
        }
        item["domain_evidence"] = broad_details.get(module, {})

    result["version"] = "asset_state_cards_v3_explicit_holding_and_overweight_eligibility"
    return result


def _state_row(row: pd.Series, cfg: dict[str, Any]) -> dict[str, Any]:
    th = cfg["state_thresholds"]

    credit = _credit_v43(row, cfg)
    cr_state = credit["cr_state"]
    credit_tags = credit["credit_tags"]
    funding_stress = credit["funding_stress_score"]

    rate_change_20d = _num(row, "rate_treasury_10y_change_20d", 0.0)
    curve_30y_10y = _num(row, "rate_curve_30y_10y", np.nan)
    rate_tags: list[str] = []
    rate_tags += _tag(rate_change_20d <= th["rates"]["bull_20d_bp"], "rates_rally_20d")
    rate_tags += _tag(rate_change_20d >= th["rates"]["bear_20d_bp"], "rates_selloff_20d")
    rate_tags += _tag(curve_30y_10y <= th["rates"]["curve_flat_30y_10y"], "long_end_curve_flat")
    if rate_change_20d <= th["rates"]["bull_20d_bp"]:
        r_state = "R1"
    elif rate_change_20d >= th["rates"]["bear_20d_bp"]:
        r_state = "R4"
    else:
        r_state = "R2"
    rates_detail = _rates_detail_state(row, cfg)

    cb_price = _num(row, "cb_cb_price_median")
    low_ratio = _num(row, "cb_low_price_ratio_110", _num(row, "cb_low_price_ratio_100_proxy", 0.0))
    callable_pressure = _num(row, "cb_callable_pressure_ratio", 0.0)
    cb_tags: list[str] = []
    cb_tags += _tag(cb_price <= th["cb"]["cheap_price"], "cb_price_low")
    cb_tags += _tag(low_ratio >= th["cb"]["low_price_ratio_high"], "cb_low_price_pool_large")
    cb_tags += _tag(cb_price >= th["cb"]["expensive_price"], "cb_price_expensive")
    cb_tags += _tag(callable_pressure >= th["cb"]["callable_pressure_high"], "cb_callable_pressure_high")
    if cb_price <= th["cb"]["cheap_price"] or low_ratio >= th["cb"]["low_price_ratio_high"]:
        cb_state = "CB1"
    elif cb_price >= th["cb"]["expensive_price"] or callable_pressure >= th["cb"]["callable_pressure_high"]:
        cb_state = "CB4"
    else:
        cb_state = "CB2"
    cb_detail = _cb_detail_state(row, cfg)

    breadth_5d = _num(row, "equity_sw_l1_positive_5d_ratio")
    csi1000_pe = _num(row, "equity_csi1000_pe")
    valuation_pct = _num(row, "equity_csi1000_pe_pct_2018_to_t")
    equity_tags: list[str] = []
    equity_tags += _tag(breadth_5d >= th["equity"]["breadth_strong"], "equity_breadth_strong")
    equity_tags += _tag(breadth_5d <= th["equity"]["breadth_weak"], "equity_breadth_weak")
    equity_tags += _tag(valuation_pct >= th["equity"]["valuation_high_pct"], "equity_valuation_high")
    equity_tags += _tag(valuation_pct <= th["equity"]["valuation_low_pct"], "equity_valuation_low")
    if breadth_5d >= th["equity"]["breadth_strong"] and not valuation_pct >= th["equity"]["valuation_high_pct"]:
        eq_state = "EQ1"
    elif breadth_5d <= th["equity"]["breadth_weak"] or valuation_pct >= th["equity"]["valuation_high_pct"]:
        eq_state = "EQ4"
    else:
        eq_state = "EQ2"
    equity_detail = _equity_detail_state(row, cfg)

    alt_detail = _alt_detail_states(row, eq_state, cfg)
    alt_state = alt_detail["alt_state"]
    alt_tags = list(alt_detail["tags"])
    asset_detail = row.get("asset_detail_states", {})
    if not isinstance(asset_detail, dict):
        asset_detail = {}

    liquidity_shock = bool(credit["liquidity_shock"])
    rb_score = min(1.0, max(0.0, 0.55 * funding_stress + 0.25 * max(rate_change_20d, 0.0) / 0.15 + 0.20 * max(-breadth_5d + 0.5, 0.0)))
    if rb_score >= 0.75:
        rb_state = "RB4"
    elif rb_score >= 0.55:
        rb_state = "RB3"
    else:
        rb_state = "RB1"

    cross_asset = _cross_asset_regime(r_state, eq_state, rb_state, rate_change_20d, breadth_5d, valuation_pct, rb_score, cfg)
    state_tags = sorted(set(credit_tags + rate_tags + cb_tags + equity_tags + alt_tags))
    eligible = eligible_universe(
        cr_state,
        cb_state,
        r_state,
        eq_state,
        liquidity_shock,
        bool(credit["CR2_caution_overlay"]),
    )
    filters = view_filters(cr_state, cb_state, r_state, eq_state, liquidity_shock, bool(credit["CR2_caution_overlay"]))
    asset_detail = _contextualize_asset_detail(
        asset_detail,
        cr_state=cr_state,
        r_state=r_state,
        cb_state=cb_state,
        eq_state=eq_state,
        alt_state=alt_state,
        credit=credit,
        rates_detail=rates_detail,
        cb_detail=cb_detail,
        equity_detail=equity_detail,
        alt_detail=alt_detail,
        filters=filters,
    )
    rule_output = _rule_output_object(
        cr_state=cr_state,
        r_state=r_state,
        cb_state=cb_state,
        eq_state=eq_state,
        alt_state=alt_state,
        rb_state=rb_state,
        credit=credit,
        rates_detail=rates_detail,
        cb_detail=cb_detail,
        equity_detail=equity_detail,
        alt_detail=alt_detail,
        cross_asset=cross_asset,
        state_tags=state_tags,
        eligible=eligible,
        filters=filters,
        liquidity_shock=liquidity_shock,
    )
    return {
        "date": row["date"],
        "cr_state": cr_state,
        "r_state": r_state,
        "cb_state": cb_state,
        "eq_state": eq_state,
        "alt_state": alt_state,
        "rb_state": rb_state,
        "liquidity_shock": liquidity_shock,
        "risk_budget_score": rb_score,
        "credit_spread_pct": credit["credit_spread_pct"],
        "credit_spread_change_20d": credit["credit_spread_change_20d"],
        "credit_spread_score": credit["credit_spread_score"],
        "spread_widening_pressure": credit["spread_widening_pressure"],
        "credit_turnover_stress": credit["credit_turnover_stress"],
        "funding_stress_score": credit["funding_stress_score"],
        "regional_default_pressure": credit["regional_default_pressure"],
        "credit_tail_risk_score": credit["credit_tail_risk_score"],
        "CR2_caution_overlay": credit["CR2_caution_overlay"],
        "cr_state_transition": credit["cr_state_transition"],
        "credit_final_buckets": credit["credit_final_buckets"],
        "credit_aa_aaa_spread": credit["credit_aa_aaa_spread"],
        "credit_net_financing": credit["credit_net_financing"],
        "credit_data_quality": credit["credit_data_quality"],
        "rate_10y_change_20d": rate_change_20d,
        "cb_price_median": cb_price,
        "equity_breadth_5d": breadth_5d,
        "equity_csi1000_pe": csi1000_pe,
        "cross_asset_regime": cross_asset,
        "bond_score": cross_asset["bond_score"],
        "equity_score": cross_asset["equity_score"],
        "stock_bond_regime": cross_asset["stock_bond_regime"],
        "stock_bond_policy_hint": cross_asset["policy_hint"],
        "rates_detail_states": rates_detail,
        "cb_detail_states": cb_detail,
        "equity_detail_states": equity_detail,
        "alt_detail_states": alt_detail,
        "asset_detail_states": asset_detail,
        "state_tags": state_tags,
        "skill_layers": select_skill_layers(cr_state, r_state, cb_state, eq_state, alt_state, rb_state, liquidity_shock, cfg),
        "eligible_universe": eligible,
        "view_filters": filters,
        "rule_output_object": rule_output,
        "quality_flag": "deterministic_market_skill_v43",
    }


def select_skill_layers(
    cr_state: str,
    r_state: str,
    cb_state: str,
    eq_state: str,
    alt_state: str,
    rb_state: str,
    liquidity_shock: bool,
    cfg: dict[str, Any],
) -> dict[str, list[str]]:
    layers = cfg.get("skill_layers", {})
    selected = {
        "P1_always": layers.get("P1_always", []),
        "P2_baseline": layers.get("P2_baseline", []),
        "P3_stable_principles": layers.get("P3_stable_principles", []),
        "P4_context_specific": [],
        "P5_stress_overrides": [],
    }
    p4 = layers.get("P4_context_specific", {})
    for state in [cr_state, r_state, cb_state, eq_state, alt_state, rb_state]:
        if state in p4:
            selected["P4_context_specific"].append(p4[state])
    if liquidity_shock or rb_state in {"RB3", "RB4"}:
        selected["P5_stress_overrides"] = layers.get("P5_stress_overrides", [])
    return selected


def eligible_universe(
    cr_state: str,
    cb_state: str,
    r_state: str,
    eq_state: str,
    liquidity_shock: bool,
    cr2_caution_overlay: bool = False,
) -> dict[str, bool]:
    filters = view_filters(
        cr_state,
        cb_state,
        r_state,
        eq_state,
        liquidity_shock,
        cr2_caution_overlay,
    )
    return {
        "allow_aa_credit": not filters["block_aa_credit"],
        "allow_credit_overweight": not filters["block_credit_overweight"],
        "allow_strong_cb_overweight": not filters["block_cb_overweight"],
        "allow_long_duration_overweight": not filters["block_long_rate_overweight"],
        "allow_equity_overweight": not filters["block_equity_overweight"],
    }


def view_filters(cr_state: str, cb_state: str, r_state: str, eq_state: str, liquidity_shock: bool, cr2_caution_overlay: bool = False) -> dict[str, bool]:
    long_duration_blocked = r_state in {"R4", "R5"} or (liquidity_shock and r_state not in {"R1"})
    return {
        "block_credit_overweight": cr_state in {"CR3", "CR4", "CR5"} or liquidity_shock or cr2_caution_overlay,
        "block_aa_credit": cr_state in {"CR3", "CR4", "CR5"} or liquidity_shock or cr2_caution_overlay,
        "block_cb_overweight": cb_state in {"CB4", "CB5"} or liquidity_shock,
        "block_long_rate_overweight": long_duration_blocked,
        "block_equity_overweight": eq_state in {"EQ4", "EQ5"} or liquidity_shock,
    }


def _rule_bound_level(level: str, expression: str, reason: str) -> dict[str, Any]:
    return {"level": level, "expression": expression, "reason": reason}


def _rule_output_object(
    *,
    cr_state: str,
    r_state: str,
    cb_state: str,
    eq_state: str,
    alt_state: str,
    rb_state: str,
    credit: dict[str, Any],
    rates_detail: dict[str, Any],
    cb_detail: dict[str, Any],
    equity_detail: dict[str, Any],
    alt_detail: dict[str, Any],
    cross_asset: dict[str, Any],
    state_tags: list[str],
    eligible: dict[str, bool],
    filters: dict[str, bool],
    liquidity_shock: bool,
) -> dict[str, Any]:
    cr2_caution = bool(credit.get("CR2_caution_overlay", False))

    credit_rating_allowed = ["AAA"]
    if eligible.get("allow_aa_credit"):
        credit_rating_allowed.append("AA+")
    if cr_state == "CR1" and not liquidity_shock and not cr2_caution:
        credit_rating_allowed.append("AA")

    credit_duration_allowed = ["1-3Y"]
    if cr_state in {"CR1", "CR2"} and not cr2_caution and not liquidity_shock:
        credit_duration_allowed.append("3-5Y")

    if cr_state == "CR1":
        credit_upper = _rule_bound_level("high", "min(product_credit_cap, w_base_credit + credit_overweight_room)", "利差补偿高且尾部风险可控，允许信用票息增强")
        aa_credit_upper = _rule_bound_level("medium", "small_product_aa_cap", "CR1下可少量使用优质AA")
        credit_downshift_penalty = "low_to_medium"
    elif cr_state == "CR2" and not cr2_caution:
        credit_upper = _rule_bound_level("neutral", "min(product_credit_cap, w_base_credit + credit_overweight_room_low)", "信用中性，允许温和票息仓")
        aa_credit_upper = _rule_bound_level("low", "aa_plus_only_or_small_aa_cap", "CR2只允许有限下沉")
        credit_downshift_penalty = "medium"
    else:
        credit_upper = _rule_bound_level("low", "min(product_credit_cap, w_base_credit)", "信用补偿薄、尾部风险或CR2 caution，禁止主动超配信用")
        aa_credit_upper = _rule_bound_level("blocked", "0", "禁止AA信用下沉")
        credit_downshift_penalty = "high"

    if r_state == "R1":
        rate_upper = _rule_bound_level("high", "min(product_rate_cap, w_base_rate + duration_attack_room)", "利率下行确认，允许久期进攻")
        thirty_year_upper = _rule_bound_level("medium_high", "min(product_30y_cap, crowding_adjusted_30y_cap)", "30Y可作为久期弹性，但受拥挤度约束")
        rate_duration_allowed = ["3-5Y", "7-10Y", "30Y"]
        duration_penalty = "low_if_not_crowded"
    elif r_state == "R4":
        rate_upper = _rule_bound_level("low", "w_base_rate_or_lower", "利率上行压力，降低久期")
        thirty_year_upper = _rule_bound_level("blocked", "0", "R4下不做30Y进攻")
        rate_duration_allowed = ["1-3Y", "3-5Y"]
        duration_penalty = "high"
    else:
        rate_upper = _rule_bound_level("neutral", "product_rate_cap", "利率中性，久期围绕基准")
        thirty_year_upper = _rule_bound_level("neutral", "min(product_30y_cap, neutral_30y_cap)", "30Y中性配置")
        rate_duration_allowed = ["1-3Y", "3-5Y", "7-10Y"]
        duration_penalty = "medium"

    if cb_state == "CB1" and eq_state in {"EQ1", "EQ2"}:
        cb_upper = _rule_bound_level("high", "min(product_cb_cap, w_base_cb + cb_overweight_room)", "转债便宜且权益环境不差，允许选择性增强")
        cb_style_allowed = ["偏债", "平衡", "股性低溢价"]
        cb_overweight_penalty = "low_to_medium"
        high_price_high_premium_allowed = False
    elif cb_state == "CB4":
        cb_upper = _rule_bound_level("low", "min(product_cb_cap, w_base_cb + cb_overweight_room_low)", "转债估值贵或赎回压力高，压低强超配")
        cb_style_allowed = ["偏债", "平衡"]
        cb_overweight_penalty = "high"
        high_price_high_premium_allowed = False
    else:
        cb_upper = _rule_bound_level("neutral", "product_cb_cap", "转债中性，跟随产品预算")
        cb_style_allowed = ["偏债", "平衡"]
        cb_overweight_penalty = "medium"
        high_price_high_premium_allowed = False

    if eq_state == "EQ1":
        equity_upper = _rule_bound_level("high", "product_equity_cap", "权益宽度支持且估值不过热，允许权益相对正向view")
        broad_beta_overweight_allowed = True
        sector_rotation_allowed = True
        equity_beta_penalty = "low_to_medium"
    elif eq_state == "EQ4":
        equity_upper = _rule_bound_level("low", "min(product_equity_cap, w_base_equity)", "权益宽度弱或估值高，禁止主动超配")
        broad_beta_overweight_allowed = False
        sector_rotation_allowed = False
        equity_beta_penalty = "high"
    else:
        equity_upper = _rule_bound_level("neutral", "product_equity_cap", "权益中性，只做结构性机会")
        broad_beta_overweight_allowed = False
        sector_rotation_allowed = True
        equity_beta_penalty = "medium"

    if alt_state == "ALT1":
        alt_upper = _rule_bound_level("medium_high", "min(product_alt_cap, w_base_alt + defensive_alt_room)", "黄金/美债防御价值较高")
        alt_bucket_allowed = ["gold", "us_bond", "reits_selective"]
        commodity_beta_allowed = False
        alt_overweight_penalty = "low_for_defensive"
    elif alt_state == "ALT3":
        alt_upper = _rule_bound_level("medium_high", "min(product_alt_cap, w_base_alt + commodity_alt_room)", "商品通胀或顺周期beta较强")
        alt_bucket_allowed = ["commodity", "nonferrous", "chemical", "gold_partial"]
        commodity_beta_allowed = True
        alt_overweight_penalty = "medium"
    elif alt_state == "ALT4":
        alt_upper = _rule_bound_level("low", "min(product_alt_cap, w_base_alt)", "另类整体动量偏弱，避免主动超配")
        alt_bucket_allowed = ["reits_selective", "us_bond_neutral", "gold_small"]
        commodity_beta_allowed = False
        alt_overweight_penalty = "high"
    else:
        alt_upper = _rule_bound_level("neutral", "product_alt_cap", "另类中性或分化")
        alt_bucket_allowed = ["gold", "commodity", "nonferrous", "chemical", "us_bond", "reits"]
        commodity_beta_allowed = False
        alt_overweight_penalty = "medium"

    if rb_state in {"RB3", "RB4"} or liquidity_shock:
        adjust_speed = "slow_30_to_45pct"
        risk_asset_new_buy_penalty = "high"
        max_active_deviation_hint = "tighten"
    else:
        adjust_speed = "normal_60pct"
        risk_asset_new_buy_penalty = "normal"
        max_active_deviation_hint = "normal"

    hard_constraints = {
        "equity_upper": equity_upper,
        "cb_upper": cb_upper,
        "rate_upper": rate_upper,
        "thirty_year_upper": thirty_year_upper,
        "credit_upper": credit_upper,
        "aa_credit_upper": aa_credit_upper,
        "alt_upper": alt_upper,
        "cash_policy": _rule_bound_level("blocked", "0", "当前V1禁用现金，防御通过久期/信用质量/黄金表达"),
    }

    bounds = {
        "rate_weight": ["product_rate_lower", "rate_upper"],
        "credit_weight": ["product_credit_lower", "credit_upper"],
        "cb_weight": ["product_cb_lower", "cb_upper"],
        "equity_weight": ["product_equity_lower", "equity_upper"],
        "alt_weight": ["product_alt_lower", "alt_upper"],
        "cash_weight": [0, 0],
    }

    penalties = {
        "cb_overweight_penalty": cb_overweight_penalty,
        "duration_penalty": duration_penalty,
        "credit_downshift_penalty": credit_downshift_penalty,
        "equity_beta_penalty": equity_beta_penalty,
        "alt_overweight_penalty": alt_overweight_penalty,
        "risk_asset_new_buy_penalty": risk_asset_new_buy_penalty,
    }

    return {
        "state_tags": {
            "cr_state": cr_state,
            "r_state": r_state,
            "cb_state": cb_state,
            "eq_state": eq_state,
            "alt_state": alt_state,
            "rb_state": rb_state,
            "E_state": cross_asset.get("equity_bucket"),
            "B_state": cross_asset.get("bond_bucket"),
            "stock_bond_regime": cross_asset.get("stock_bond_regime"),
            "policy_hint": cross_asset.get("policy_hint"),
            "raw_tags": state_tags,
        },
        "hard_constraints": hard_constraints,
        "bounds": bounds,
        "penalties": penalties,
        "eligible_universe": {
            **eligible,
            "credit_rating_allowed": credit_rating_allowed,
            "credit_duration_allowed": credit_duration_allowed,
            "rate_duration_allowed": rate_duration_allowed,
            "cb_style_allowed": cb_style_allowed,
            "high_price_high_premium_cb_allowed": high_price_high_premium_allowed,
            "broad_beta_overweight_allowed": broad_beta_overweight_allowed,
            "sector_rotation_allowed": sector_rotation_allowed,
            "strong_sector_universe": "行业动量和分化度筛选；当前由资产级iTransformer/权益资产池近似表达",
            "alt_bucket_allowed": alt_bucket_allowed,
            "commodity_beta_allowed": commodity_beta_allowed,
        },
        "view_filters": {
            **filters,
            "allow_strong_cb_overweight_view": not filters.get("block_cb_overweight", False),
            "allow_credit_downshift_view": eligible.get("allow_aa_credit", False),
            "allow_duration_attack_view": not filters.get("block_long_rate_overweight", False),
            "allow_equity_beta_view": not filters.get("block_equity_overweight", False),
            "allow_commodity_beta_view": commodity_beta_allowed,
        },
        "execution_guidance": {
            "adjust_speed": adjust_speed,
            "max_active_deviation_hint": max_active_deviation_hint,
            "optimizer_role": "规则只给边界、惩罚和relative views；最终仓位由EP/skfolio在约束内求解",
        },
        "module_evidence": {
            "credit_v43": {
                "credit_spread_score": credit.get("credit_spread_score"),
                "spread_widening_pressure": credit.get("spread_widening_pressure"),
                "credit_turnover_stress": credit.get("credit_turnover_stress"),
                "funding_stress_score": credit.get("funding_stress_score"),
                "credit_tail_risk_score": credit.get("credit_tail_risk_score"),
                "CR2_caution_overlay": cr2_caution,
            },
            "rates_detail": rates_detail,
            "cb_detail": cb_detail,
            "equity_detail": equity_detail,
            "alt_detail": alt_detail,
        },
    }


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _json_loads(value: Any) -> Any:
    if isinstance(value, str):
        return json.loads(value)
    return value


def _view_asset_allowed(asset: str, cfg: dict[str, Any], allow_cash: bool | None = None) -> bool:
    allowed = set(cfg.get("view_generation", {}).get("allowed_assets", ALLOWED_VIEW_ASSETS))
    if allow_cash is None:
        allow_cash = bool(cfg.get("view_generation", {}).get("allow_cash_views", False))
    if not allow_cash:
        allowed.discard("cash")
    return asset in allowed


def _views_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    out = pd.DataFrame(rows)
    for col in VIEW_COLUMNS:
        if col not in out.columns:
            out[col] = pd.Series(dtype="object")
    return out[VIEW_COLUMNS + [c for c in out.columns if c not in VIEW_COLUMNS]]


def build_market_state_cards(cfg: dict[str, Any] | None = None, paths: MarketSkillPaths | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    paths = paths or MarketSkillPaths()
    df = pd.read_parquet(paths.feature_panel)
    df = enrich_market_features(df, paths, cfg)
    release_as_of = cfg.get("release", {}).get("daily_as_of_date")
    if release_as_of:
        df = df.loc[df["date"] <= pd.Timestamp(release_as_of)].copy()
    if "equity_csi1000_pe" in df.columns:
        df["equity_csi1000_pe_pct_2018_to_t"] = df["equity_csi1000_pe"].expanding(
            min_periods=cfg["windows"]["percentile_min_periods"]
        ).rank(pct=True)
    rows = [_state_row(row, cfg) for _, row in df.iterrows()]
    cards = pd.DataFrame(rows)
    for col in [
        "state_tags",
        "skill_layers",
        "eligible_universe",
        "view_filters",
        "credit_final_buckets",
        "cross_asset_regime",
        "rates_detail_states",
        "cb_detail_states",
        "equity_detail_states",
        "alt_detail_states",
        "asset_detail_states",
        "rule_output_object",
    ]:
        cards[col] = cards[col].map(_json_dumps)
    paths.cards.parent.mkdir(parents=True, exist_ok=True)
    cards.to_parquet(paths.cards, index=False)
    write_latest_card_report(cards, paths)
    return cards


def write_latest_card_report(cards: pd.DataFrame, paths: MarketSkillPaths | None = None) -> None:
    paths = paths or MarketSkillPaths()
    REPORTS.mkdir(parents=True, exist_ok=True)
    latest = cards.sort_values("date").iloc[-1].to_dict()
    for col in [
        "state_tags",
        "skill_layers",
        "eligible_universe",
        "view_filters",
        "credit_final_buckets",
        "cross_asset_regime",
        "rates_detail_states",
        "cb_detail_states",
        "equity_detail_states",
        "alt_detail_states",
        "asset_detail_states",
        "rule_output_object",
    ]:
        latest[col] = _json_loads(latest[col])
    latest["date"] = str(pd.Timestamp(latest["date"]).date())
    paths.latest_json.write_text(json.dumps(latest, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Market State Card Latest",
        "",
        f"- date: {latest['date']}",
        f"- states: CR={latest['cr_state']}, R={latest['r_state']}, CB={latest['cb_state']}, EQ={latest['eq_state']}, ALT={latest['alt_state']}, RB={latest['rb_state']}",
        f"- stock_bond_regime: {latest['stock_bond_regime']} ({latest['stock_bond_policy_hint']})",
        f"- liquidity_shock: {latest['liquidity_shock']}",
        f"- risk_budget_score: {latest['risk_budget_score']:.3f}",
        f"- credit_v43: spread_score={latest['credit_spread_score']:.3f}, widening_pressure={latest['spread_widening_pressure']:.3f}, turnover_stress={latest['credit_turnover_stress']:.3f}, tail_risk={latest['credit_tail_risk_score']:.3f}, CR2_caution={latest['CR2_caution_overlay']}",
        f"- rates_detail: {latest['rates_detail_states'].get('summary')}",
        f"- cb_detail: {latest['cb_detail_states'].get('summary')}",
        f"- equity_detail: {latest['equity_detail_states'].get('summary')}",
        f"- alt_detail: {latest['alt_detail_states'].get('summary')} | positive={', '.join(latest['alt_detail_states'].get('positive_assets', [])) or 'none'} | negative={', '.join(latest['alt_detail_states'].get('negative_assets', [])) or 'none'}",
        f"- asset_cards: {len(latest['asset_detail_states'].get('assets', {}))} assets across {len(latest['asset_detail_states'].get('modules', {}))} modules",
        f"- state_tags: {', '.join(latest['state_tags']) if latest['state_tags'] else 'none'}",
        "",
        "## View Filters",
    ]
    for key, value in latest["view_filters"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Rule Output Object"])
    rule = latest.get("rule_output_object", {})
    lines.append("### Hard Constraints")
    for key, value in rule.get("hard_constraints", {}).items():
        lines.append(f"- {key}: {value}")
    lines.append("### Penalties")
    for key, value in rule.get("penalties", {}).items():
        lines.append(f"- {key}: {value}")
    lines.append("### Eligible Universe")
    for key, value in rule.get("eligible_universe", {}).items():
        lines.append(f"- {key}: {value}")
    lines.append("### Execution Guidance")
    for key, value in rule.get("execution_guidance", {}).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Active Skill Fragments"])
    for layer, fragments in latest["skill_layers"].items():
        if fragments:
            lines.append(f"- {layer}: " + " | ".join(fragments))
    lines.extend(["", "## Detail States"])
    lines.append(f"- rates_long_duration: {latest['rates_detail_states'].get('long_duration')}")
    lines.append(f"- convertible_bonds: {latest['cb_detail_states'].get('valuation')}")
    lines.append(f"- equity: {latest['equity_detail_states'].get('breadth_valuation')}")
    lines.extend(["", "## ALT Detail States"])
    for asset_code, state in latest["alt_detail_states"].get("assets", {}).items():
        tags = ", ".join(state.get("tags", [])) or "none"
        metrics = state.get("metrics", {})
        lines.append(
            f"- {asset_code}: state={state.get('state')}, score={state.get('score')}, tags={tags}, metrics={metrics}"
        )
    lines.extend(["", "## All Asset Detail States"])
    all_assets = latest["asset_detail_states"].get("assets", {})
    for module in ["rates", "credit", "convertible_bonds", "equity", "alternatives"]:
        module_assets = [(code, item) for code, item in all_assets.items() if item.get("module") == module]
        if not module_assets:
            continue
        lines.append(f"### {module}")
        for asset_code, state in sorted(module_assets, key=lambda item: item[1].get("module_rank") or 0.0, reverse=True):
            metrics = state.get("metrics", {})
            context = state.get("domain_context", {})
            lines.append(
                f"- {asset_code} ({state.get('label')}): state={state.get('state')}, score={state.get('score')}, "
                f"rank={state.get('module_rank')}, decision={context.get('decision_state')}, "
                f"data_available={context.get('data_available')}, holding_eligible={context.get('holding_eligible')}, "
                f"overweight_eligible={context.get('overweight_eligible')}, ret20={metrics.get('ret_20d')}, ret60={metrics.get('ret_60d')}, "
                f"drawdown60={metrics.get('drawdown_60d')}, available={state.get('data_quality', {}).get('available')}"
            )
    paths.latest_md.write_text("\n".join(lines), encoding="utf-8")


def _base_view(date: Any, source: str, target: str, relation: str, confidence: float, tags: list[str]) -> dict[str, Any]:
    confidence = round(float(confidence), 3)
    return {
        "date": date,
        "source_asset": source,
        "target_asset": target,
        "relation": relation,
        "confidence": confidence,
        "raw_confidence": confidence,
        "confidence_calibration": "static_rule",
        "reason_tags": tags,
        "view_type": "relative",
        "generator": "mechanical_market_skill_v1",
    }


def _add_unique_view(
    views: list[dict[str, Any]],
    date: Any,
    source: str,
    target: str,
    confidence: float,
    tags: list[str],
) -> None:
    if source == target:
        return
    key = (source, target, "overweight")
    for view in views:
        if (view["source_asset"], view["target_asset"], view["relation"]) == key:
            view["confidence"] = round(max(float(view["confidence"]), float(confidence)), 3)
            view["reason_tags"] = sorted(set(list(view.get("reason_tags", [])) + tags))
            return
    views.append(_base_view(date, source, target, "overweight", confidence, sorted(set(tags))))


def _state_code(card: pd.Series, key: str, default: str = "") -> str:
    val = card.get(key, default)
    return str(val) if val is not None else default


def _nine_grid_tags(card: pd.Series) -> list[str]:
    regime = str(card.get("stock_bond_regime", "unknown_regime"))
    policy = str(card.get("stock_bond_policy_hint", "unknown_policy"))
    tags = [regime, policy]
    bond_score = card.get("bond_score")
    equity_score = card.get("equity_score")
    if pd.notna(bond_score):
        tags.append(f"bond_score={float(bond_score):.2f}")
    if pd.notna(equity_score):
        tags.append(f"equity_score={float(equity_score):.2f}")
    return tags


def _detail_score(card: pd.Series, field: str, nested_key: str | None = None, default: float = 0.5) -> float:
    obj = _json_loads(card.get(field, "{}"))
    if nested_key:
        obj = obj.get(nested_key, {}) if isinstance(obj, dict) else {}
    try:
        value = float(obj.get("score", default)) if isinstance(obj, dict) else default
    except (TypeError, ValueError):
        value = default
    return float(np.clip(value, 0.0, 1.0))


def _asset_state_score(card: pd.Series, field: str, asset_code: str, default: float = 0.5) -> float:
    obj = _json_loads(card.get(field, "{}"))
    try:
        value = float(obj.get("assets", {}).get(asset_code, {}).get("score", default))
    except (AttributeError, TypeError, ValueError):
        value = default
    return float(np.clip(value, 0.0, 1.0))


def _source_strength(card: pd.Series, asset: str) -> float:
    if asset == "rates_long_duration":
        return _detail_score(card, "rates_detail_states", "long_duration")
    if asset == "convertible_bonds":
        return _detail_score(card, "cb_detail_states", "valuation")
    if asset == "equity":
        return _detail_score(card, "equity_detail_states", "breadth_valuation")
    if asset == "gold":
        return _asset_state_score(card, "alt_detail_states", "alt_gold")
    if asset == "alternatives":
        alt = _json_loads(card.get("alt_detail_states", "{}"))
        scores = []
        if isinstance(alt, dict):
            for state in alt.get("assets", {}).values():
                try:
                    scores.append(float(state.get("score", np.nan)))
                except (TypeError, ValueError, AttributeError):
                    continue
        return float(np.nanmean(scores)) if scores else 0.5
    if asset == "credit":
        try:
            spread_score = float(card.get("credit_spread_score", 0.5))
            pressure = float(card.get("spread_widening_pressure", 0.0)) * 0.35 + float(card.get("funding_stress_score", 0.0)) * 0.25 + float(card.get("credit_turnover_stress", 0.0)) * 0.20 + float(card.get("credit_tail_risk_score", 0.0)) * 0.20
        except (TypeError, ValueError):
            return 0.5
        return float(np.clip(0.65 * spread_score + 0.35 * (1.0 - pressure), 0.0, 1.0))
    return 0.5


def _target_weakness(card: pd.Series, asset: str) -> float:
    return 1.0 - _source_strength(card, asset)


def _calibrate_mechanical_view_confidence(view: dict[str, Any], card: pd.Series) -> dict[str, Any]:
    raw = float(view.get("confidence", 0.0))
    source = str(view.get("source_asset", ""))
    target = str(view.get("target_asset", ""))
    source_strength = _source_strength(card, source)
    target_weakness = _target_weakness(card, target)
    adjustment = 0.08 * (source_strength - 0.5) + 0.06 * (target_weakness - 0.5)

    try:
        bond_score = float(card.get("bond_score", np.nan))
        equity_score = float(card.get("equity_score", np.nan))
    except (TypeError, ValueError):
        bond_score = equity_score = np.nan
    if np.isfinite(bond_score) and np.isfinite(equity_score):
        spread = bond_score - equity_score
        if source == "rates_long_duration" and target in {"equity", "convertible_bonds"}:
            adjustment += float(np.clip(spread, -0.25, 0.25)) * 0.10
        if source == "equity" and target == "rates_long_duration":
            adjustment -= float(np.clip(spread, -0.25, 0.25)) * 0.10

    tags = view.get("reason_tags", []) or []
    if isinstance(tags, str):
        tags = _json_loads(tags)
    tag_set = {str(x) for x in tags}
    if "CR2_caution_overlay" in tag_set or "credit_tail_risk_high" in tag_set:
        adjustment += 0.025 if source == "rates_long_duration" else -0.025
    if "equity_valuation_high" in tag_set and target == "equity":
        adjustment += 0.020
    if "cb_price_expensive" in tag_set and target == "convertible_bonds":
        adjustment += 0.020
    if "alt_broad_weakness" in tag_set and target == "alternatives":
        adjustment += 0.018

    calibrated = float(np.clip(raw + adjustment, 0.45, 0.78))
    view["raw_confidence"] = round(raw, 3)
    view["confidence"] = round(calibrated, 3)
    view["confidence_calibration"] = _json_dumps(
        {
            "method": "mechanical_evidence_strength_v1",
            "source_strength": round(float(source_strength), 4),
            "target_weakness": round(float(target_weakness), 4),
            "adjustment": round(float(adjustment), 4),
        }
    )
    return view


def generate_candidate_views(card: pd.Series) -> list[dict[str, Any]]:
    date = card["date"]
    tags = _json_loads(card["state_tags"])
    views: list[dict[str, Any]] = []
    if card["cr_state"] == "CR1":
        views.append(_base_view(date, "credit", "cash", "overweight", 0.62, ["CR1"] + tags))
    if card["cr_state"] in {"CR3", "CR4"}:
        views.append(_base_view(date, "cash", "credit", "overweight", 0.68, [card["cr_state"]] + tags))
    if card["r_state"] == "R1":
        views.append(_base_view(date, "rates_long_duration", "cash", "overweight", 0.60, ["R1"] + tags))
    if card["r_state"] == "R4":
        views.append(_base_view(date, "cash", "rates_long_duration", "overweight", 0.65, ["R4"] + tags))
    if card["cb_state"] == "CB1" and card["eq_state"] in {"EQ1", "EQ2"}:
        views.append(_base_view(date, "convertible_bonds", "cash", "overweight", 0.58, ["CB1", card["eq_state"]] + tags))
    if card["cb_state"] == "CB4":
        views.append(_base_view(date, "cash", "convertible_bonds", "overweight", 0.64, ["CB4"] + tags))
    if card["eq_state"] == "EQ1":
        views.append(_base_view(date, "equity", "cash", "overweight", 0.56, ["EQ1"] + tags))
    if card["eq_state"] == "EQ4":
        views.append(_base_view(date, "cash", "equity", "overweight", 0.62, ["EQ4"] + tags))
    if card["alt_state"] == "ALT1":
        views.append(_base_view(date, "gold", "equity", "overweight", 0.54, ["ALT1"] + tags))
    if card["alt_state"] == "ALT3":
        views.append(_base_view(date, "alternatives", "credit", "overweight", 0.54, ["ALT3"] + tags))
    if card["alt_state"] == "ALT4":
        views.append(_base_view(date, "rates_long_duration", "alternatives", "overweight", 0.54, ["ALT4"] + tags))
    return views


def generate_executable_mechanical_views(card: pd.Series, cfg: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    cfg = cfg or load_config()
    date = card["date"]
    tags = _json_loads(card["state_tags"])
    state_tags = [
        _state_code(card, "cr_state"),
        _state_code(card, "r_state"),
        _state_code(card, "cb_state"),
        _state_code(card, "eq_state"),
        _state_code(card, "alt_state"),
        _state_code(card, "rb_state"),
    ] + _nine_grid_tags(card) + tags
    regime = str(card.get("stock_bond_regime", ""))
    policy_hint = str(card.get("stock_bond_policy_hint", ""))
    bond_score = float(card.get("bond_score", np.nan))
    equity_score = float(card.get("equity_score", np.nan))
    cr_state = _state_code(card, "cr_state")
    r_state = _state_code(card, "r_state")
    cb_state = _state_code(card, "cb_state")
    eq_state = _state_code(card, "eq_state")
    alt_state = _state_code(card, "alt_state")
    rb_state = _state_code(card, "rb_state")
    cr2_caution = bool(card.get("CR2_caution_overlay", False))
    views: list[dict[str, Any]] = []

    # v4.3 direction boundary: use the E/B nine-grid as the first layer,
    # then let module states add relative views within that boundary.
    if "bond_strong_equity_weak" in regime or policy_hint == "prefer_duration_and_defensive_assets":
        _add_unique_view(views, date, "rates_long_duration", "equity", 0.68, state_tags + ["nine_grid_duration_over_equity"])
        _add_unique_view(views, date, "rates_long_duration", "convertible_bonds", 0.66, state_tags + ["nine_grid_duration_over_cb"])
    elif "bond_strong_equity_strong" in regime:
        _add_unique_view(views, date, "rates_long_duration", "credit", 0.56, state_tags + ["nine_grid_barbell_duration"])
        if cb_state in {"CB1", "CB2"}:
            _add_unique_view(views, date, "convertible_bonds", "credit", 0.52, state_tags + ["nine_grid_selective_cb"])
    elif "bond_weak_equity_strong" in regime:
        _add_unique_view(views, date, "equity", "rates_long_duration", 0.58, state_tags + ["nine_grid_equity_over_duration"])
        if cb_state in {"CB1", "CB2"}:
            _add_unique_view(views, date, "convertible_bonds", "rates_long_duration", 0.54, state_tags + ["nine_grid_cb_beta_allowed"])
    elif "bond_weak_equity_weak" in regime:
        _add_unique_view(views, date, "credit", "equity", 0.58, state_tags + ["nine_grid_de_risk_credit_quality"])
        _add_unique_view(views, date, "gold", "equity", 0.56, state_tags + ["nine_grid_defensive_gold"])
    else:
        if np.isfinite(bond_score) and np.isfinite(equity_score):
            spread = bond_score - equity_score
            if spread >= 0.16:
                _add_unique_view(views, date, "rates_long_duration", "equity", 0.58, state_tags + ["nine_grid_bond_score_advantage"])
            elif spread <= -0.16:
                _add_unique_view(views, date, "equity", "rates_long_duration", 0.54, state_tags + ["nine_grid_equity_score_advantage"])

    if cr_state == "CR1":
        _add_unique_view(views, date, "credit", "rates_long_duration", 0.58, state_tags + ["credit_spread_compensation_high"])
        if cb_state in {"CB3", "CB4", "CB5"} or eq_state in {"EQ3", "EQ4", "EQ5"}:
            _add_unique_view(views, date, "credit", "convertible_bonds", 0.56, state_tags + ["credit_carry_over_cb"])
    elif cr_state == "CR2" and not cr2_caution:
        if cb_state == "CB4":
            _add_unique_view(views, date, "credit", "convertible_bonds", 0.60, state_tags + ["credit_neutral_cb_expensive"])
        if eq_state == "EQ4":
            _add_unique_view(views, date, "credit", "equity", 0.54, state_tags + ["credit_quality_over_weak_equity"])
    elif cr_state in {"CR3", "CR4", "CR5"} or cr2_caution:
        _add_unique_view(views, date, "rates_long_duration", "credit", 0.64, state_tags + ["credit_compensation_thin_or_caution"])

    if r_state == "R1":
        _add_unique_view(views, date, "rates_long_duration", "equity", 0.62, state_tags + ["duration_attack_allowed"])
        if cb_state in {"CB3", "CB4", "CB5"}:
            _add_unique_view(views, date, "rates_long_duration", "convertible_bonds", 0.64, state_tags + ["duration_over_expensive_cb"])
    elif r_state == "R4":
        _add_unique_view(views, date, "credit", "rates_long_duration", 0.62, state_tags + ["duration_attack_blocked"])

    if cb_state == "CB1" and eq_state in {"EQ1", "EQ2"}:
        _add_unique_view(views, date, "convertible_bonds", "credit", 0.56, state_tags + ["cb_cheap_with_equity_confirmation"])
    elif cb_state == "CB4":
        _add_unique_view(views, date, "credit", "convertible_bonds", 0.62, state_tags + ["cb_expensive_reduce_beta"])
        if r_state in {"R1", "R2"}:
            _add_unique_view(views, date, "rates_long_duration", "convertible_bonds", 0.62, state_tags + ["rates_over_expensive_cb"])

    if eq_state == "EQ1":
        _add_unique_view(views, date, "equity", "credit", 0.54, state_tags + ["equity_environment_positive"])
    elif eq_state == "EQ4":
        _add_unique_view(views, date, "rates_long_duration", "equity", 0.62, state_tags + ["equity_breadth_or_valuation_block"])
        _add_unique_view(views, date, "gold", "equity", 0.54, state_tags + ["defensive_alt_over_weak_equity"])

    if alt_state == "ALT1":
        _add_unique_view(views, date, "gold", "equity", 0.56, state_tags + ["alt_defensive_hedge"])
    elif alt_state == "ALT3":
        _add_unique_view(views, date, "alternatives", "credit", 0.54, state_tags + ["alt_commodity_inflation_beta"])
    elif alt_state == "ALT4":
        _add_unique_view(views, date, "rates_long_duration", "alternatives", 0.54, state_tags + ["alt_broad_weakness"])

    if rb_state in {"RB3", "RB4"}:
        _add_unique_view(views, date, "rates_long_duration", "equity", 0.60, state_tags + ["risk_budget_tight_de_risk"])
        _add_unique_view(views, date, "credit", "convertible_bonds", 0.56, state_tags + ["risk_budget_tight_reduce_cb_beta"])

    out = []
    for view in views:
        if _view_asset_allowed(view["source_asset"], cfg) and _view_asset_allowed(view["target_asset"], cfg):
            view = _calibrate_mechanical_view_confidence(view, card)
            view["generator"] = "mechanical_market_skill_executable_v43"
            out.append(view)
    return out


def _view_asset_module(asset: str) -> str:
    if asset in BROAD_VIEW_ASSET_MODULE:
        return BROAD_VIEW_ASSET_MODULE[asset]
    if asset.startswith("rate_"):
        return "rates"
    if asset.startswith("credit_"):
        return "credit"
    if asset.startswith("cb_"):
        return "convertible_bonds"
    if asset.startswith("equity_"):
        return "equity"
    if asset.startswith("alt_"):
        return "alternatives"
    return asset


def validate_view(view: dict[str, Any], filters: dict[str, bool], cfg: dict[str, Any] | None = None) -> tuple[str, str]:
    source = view["source_asset"]
    target = view["target_asset"]
    allowed = set((cfg or {}).get("view_generation", {}).get("allowed_assets", ALLOWED_VIEW_ASSETS))
    if source not in allowed or target not in allowed:
        return "blocked", "unknown_view_asset"
    if source == target:
        return "blocked", "same_source_target"
    if view.get("relation") != "overweight":
        return "blocked", "unsupported_relation"
    source_module = _view_asset_module(str(source))
    if source_module == "credit" and filters.get("block_credit_overweight"):
        return "blocked", "credit_overweight_blocked_by_state"
    if source_module == "convertible_bonds" and filters.get("block_cb_overweight"):
        return "blocked", "cb_overweight_blocked_by_state"
    if source_module == "rates" and filters.get("block_long_rate_overweight"):
        return "blocked", "long_duration_overweight_blocked_by_state"
    if source_module == "equity" and filters.get("block_equity_overweight"):
        return "blocked", "equity_overweight_blocked_by_state"
    return "valid", ""


def build_views(
    cards: pd.DataFrame | None = None,
    paths: MarketSkillPaths | None = None,
    cfg: dict[str, Any] | None = None,
    executable: bool | None = None,
    write_active: bool = True,
) -> pd.DataFrame:
    paths = paths or MarketSkillPaths()
    cfg = cfg or load_config()
    if cards is None:
        cards = pd.read_parquet(paths.cards)
    if executable is None:
        executable = str(cfg.get("view_generation", {}).get("mechanical_mode", "executable")) == "executable"
    views: list[dict[str, Any]] = []
    for _, card in cards.iterrows():
        filters = _json_loads(card["view_filters"])
        candidates = generate_executable_mechanical_views(card, cfg) if executable else generate_candidate_views(card)
        for view in candidates:
            status, reason = validate_view(view, filters, cfg)
            view["validator_status"] = status
            view["validator_reason"] = reason
            view["reason_tags"] = _json_dumps(sorted(set(view["reason_tags"])))
            views.append(view)
    out = _views_dataframe(views)
    if not out.empty:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out = out.sort_values(["date", "validator_status", "source_asset", "target_asset"])
    out.to_parquet(paths.mechanical_views, index=False)
    if write_active:
        out.to_parquet(paths.views, index=False)
        write_views_report(out, paths)
    return out


def _month_end_cards(cards: pd.DataFrame) -> pd.DataFrame:
    work = cards.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work = work.dropna(subset=["date"]).sort_values("date")
    month = work["date"] + pd.offsets.MonthEnd(0)
    idx = work.groupby(month)["date"].idxmax()
    return work.loc[idx].sort_values("date").reset_index(drop=True)


def _latest_cards(cards: pd.DataFrame) -> pd.DataFrame:
    if cards.empty:
        return cards
    work = cards.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    return work.dropna(subset=["date"]).sort_values("date").tail(1).reset_index(drop=True)


def cards_for_llm_frequency(cards: pd.DataFrame, frequency: str) -> pd.DataFrame:
    if frequency == "all":
        return cards.copy()
    if frequency == "latest":
        return _latest_cards(cards)
    if frequency == "month_end":
        return _month_end_cards(cards)
    raise ValueError(f"Unsupported LLM view frequency: {frequency}")


def _compact_card_for_prompt(card: pd.Series, decision_id: str | None = None) -> dict[str, Any]:
    decision_key = decision_id or str(card.get("decision_id", "decision_hidden_date"))
    credit_final_buckets = _json_loads(card.get("credit_final_buckets", "{}"))
    cross_asset_regime = _json_loads(card.get("cross_asset_regime", "{}"))
    rates_detail_states = _json_loads(card.get("rates_detail_states", "{}"))
    cb_detail_states = _json_loads(card.get("cb_detail_states", "{}"))
    equity_detail_states = _json_loads(card.get("equity_detail_states", "{}"))
    alt_detail_states = _json_loads(card.get("alt_detail_states", "{}"))
    asset_detail_states = _json_loads(card.get("asset_detail_states", "{}"))
    rule_output_object = _json_loads(card.get("rule_output_object", "{}"))
    compact_assets: dict[str, Any] = {}
    for asset_code, item in asset_detail_states.get("assets", {}).items():
        metrics = item.get("metrics", {})
        compact_assets[asset_code] = {
            "module": item.get("module"),
            "state": item.get("state"),
            "score": item.get("score"),
            "module_rank": item.get("module_rank"),
            "ret_20d": metrics.get("ret_20d"),
            "ret_60d": metrics.get("ret_60d"),
            "drawdown_60d": metrics.get("drawdown_60d"),
            "available": item.get("data_quality", {}).get("available"),
            "domain_context": item.get("domain_context", {}),
        }
    return {
        "decision_id": decision_key,
        "data_scope": {
            "as_of_date": decision_key,
            "visible_card_count": 1,
            "visible_unit": "single_market_state_card",
            "future_data_allowed": False,
            "history_table_allowed": False,
            "calendar_date_hidden": True,
        },
        "states": {
            "credit": card["cr_state"],
            "rates": card["r_state"],
            "convertible_bonds": card["cb_state"],
            "equity": card["eq_state"],
            "alternatives": card["alt_state"],
            "risk_budget": card["rb_state"],
        },
        "liquidity_shock": bool(card["liquidity_shock"]),
        "risk_budget_score": round(float(card["risk_budget_score"]), 3),
        "credit_v43": {
            "cr_state": card["cr_state"],
            "cr_overlay": "CR2_caution_overlay" if bool(card.get("CR2_caution_overlay", False)) else "none",
            "credit_spread_score": _safe_round(card.get("credit_spread_score"), 3),
            "spread_widening_pressure": _safe_round(card.get("spread_widening_pressure"), 3),
            "credit_turnover_stress": _safe_round(card.get("credit_turnover_stress"), 3),
            "funding_stress_score": _safe_round(card.get("funding_stress_score"), 3),
            "regional_default_pressure": _safe_round(card.get("regional_default_pressure"), 3),
            "credit_tail_risk_score": _safe_round(card.get("credit_tail_risk_score"), 3),
            "cr_state_transition": card.get("cr_state_transition"),
            "credit_final_buckets": credit_final_buckets,
            "credit_data_quality": card.get("credit_data_quality"),
        },
        "cross_asset_regime": {
            "bond_score": _safe_round(card.get("bond_score"), 3),
            "equity_score": _safe_round(card.get("equity_score"), 3),
            "stock_bond_regime": card.get("stock_bond_regime"),
            "policy_hint": card.get("stock_bond_policy_hint"),
            "detail": cross_asset_regime,
        },
        "rates_detail_states": rates_detail_states,
        "cb_detail_states": cb_detail_states,
        "equity_detail_states": equity_detail_states,
        "alt_detail_states": alt_detail_states,
        "asset_detail_states": {
            "assets": compact_assets,
            "modules": asset_detail_states.get("modules", {}),
            "version": asset_detail_states.get("version"),
        },
        "rule_output_object": rule_output_object,
        "key_metrics": {
            "credit_spread_pct": round(float(card.get("credit_spread_pct", np.nan)), 3),
            "credit_spread_change_20d": round(float(card.get("credit_spread_change_20d", np.nan)), 4),
            "credit_aa_aaa_spread": _safe_round(card.get("credit_aa_aaa_spread"), 4),
            "credit_net_financing": _safe_round(card.get("credit_net_financing"), 2),
            "funding_stress_score": round(float(card.get("funding_stress_score", np.nan)), 3),
            "rate_10y_change_20d": round(float(card.get("rate_10y_change_20d", np.nan)), 4),
            "cb_price_median": round(float(card.get("cb_price_median", np.nan)), 2),
            "equity_breadth_5d": round(float(card.get("equity_breadth_5d", np.nan)), 3),
            "equity_csi1000_pe": round(float(card.get("equity_csi1000_pe", np.nan)), 2),
            "alt_gold_20d_return": alt_detail_states.get("assets", {}).get("alt_gold", {}).get("metrics", {}).get("ret_20d"),
            "alt_commodity_20d_return": alt_detail_states.get("assets", {}).get("alt_commodity", {}).get("metrics", {}).get("ret_20d"),
            "alt_nonferrous_20d_return": alt_detail_states.get("assets", {}).get("alt_nonferrous", {}).get("metrics", {}).get("ret_20d"),
            "alt_chemical_20d_return": alt_detail_states.get("assets", {}).get("alt_chemical", {}).get("metrics", {}).get("ret_20d"),
            "alt_us_bond_20d_return": alt_detail_states.get("assets", {}).get("alt_us_bond", {}).get("metrics", {}).get("ret_20d"),
            "alt_reits_distribution_yield": alt_detail_states.get("assets", {}).get("alt_reits", {}).get("metrics", {}).get("distribution_yield"),
        },
        "state_tags": _json_loads(card["state_tags"]),
        "view_filters": _json_loads(card["view_filters"]),
        "eligible_universe": _json_loads(card["eligible_universe"]),
        "skill_layers": _json_loads(card["skill_layers"]),
    }


def _extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()
    # Some OpenAI-compatible providers occasionally return literal control
    # characters inside JSON strings under long prompts. Collapsing them keeps
    # the structural braces intact while avoiding brittle parse failures.
    text = re.sub(r"[\x00-\x1f]+", " ", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def _glm_chat_completion(messages: list[dict[str, str]], cfg: dict[str, Any]) -> str:
    load_local_env()
    llm_cfg = cfg.get("llm_views", {})
    api_key_env = str(llm_cfg.get("api_key_env", "BIGMODEL_API_KEY"))
    api_key = os.environ.get(api_key_env) or os.environ.get("GLM_API_KEY") or os.environ.get("ZAI_API_KEY")
    if not api_key:
        raise RuntimeError(f"Missing GLM API key. Set {api_key_env} or GLM_API_KEY, or put it in .env.local.")

    base_url = str(llm_cfg.get("base_url", "https://open.bigmodel.cn/api/paas/v4")).rstrip("/")
    payload = {
        "model": str(llm_cfg.get("model", "glm-5.1")),
        "messages": messages,
        "temperature": float(llm_cfg.get("temperature", 0.1)),
        "max_tokens": int(llm_cfg.get("max_tokens", 1200)),
        "stream": False,
    }
    if bool(llm_cfg.get("response_format_json", False)):
        payload["response_format"] = {"type": "json_object"}
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=data,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    timeout = int(llm_cfg.get("timeout_seconds", 90))
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return str(body["choices"][0]["message"]["content"])


def _llm_prompt(card_payload: dict[str, Any], cfg: dict[str, Any]) -> list[dict[str, str]]:
    allowed_assets = [x for x in cfg.get("view_generation", {}).get("allowed_assets", []) if x != "cash"]
    if not allowed_assets:
        allowed_assets = ["rates_long_duration", "credit", "convertible_bonds", "equity", "gold", "alternatives"]
    system = (
        "你是固收+多资产配置的MarketSkill view生成器。"
        "你只能输出relative views，不能输出最终组合权重，不能突破给定view_filters和eligible_universe。"
        "输出必须是严格JSON，不要Markdown，不要解释。"
    )
    user = {
        "task": "根据Market State Card生成0到3条可执行relative views。",
        "rules": [
            "view表示source_asset相对target_asset应超配，relation固定为overweight。",
            "source_asset和target_asset只能来自allowed_assets。",
            "不要使用cash；现金不是当前产品的可投风险袖套。",
            "如果状态冲突或没有高置信观点，可以返回空views。",
            "confidence范围0到1，保守使用0.45到0.75。",
            "reason_tags必须引用状态码或state_tags，不能编造未提供的数据。",
        ],
        "allowed_assets": allowed_assets,
        "required_json_schema": {
            "views": [
                {
                    "source_asset": "rates_long_duration",
                    "target_asset": "equity",
                    "relation": "overweight",
                    "confidence": 0.6,
                    "reason_tags": ["R1", "EQ4"],
                }
            ]
        },
        "market_state_card": card_payload,
    }
    return [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(user, ensure_ascii=False)}]


def _parse_llm_views(text: str, card: pd.Series, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    obj = _extract_json_object(text)
    raw_views = obj.get("views", [])
    if not isinstance(raw_views, list):
        raw_views = []
    out: list[dict[str, Any]] = []
    for raw in raw_views[: int(cfg.get("llm_views", {}).get("max_views_per_date", 3))]:
        if not isinstance(raw, dict):
            continue
        source = str(raw.get("source_asset", "")).strip()
        target = str(raw.get("target_asset", "")).strip()
        relation = str(raw.get("relation", "overweight")).strip()
        try:
            confidence = float(raw.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        tags = raw.get("reason_tags", [])
        if isinstance(tags, str):
            tags = [tags]
        if not isinstance(tags, list):
            tags = []
        view = _base_view(card["date"], source, target, relation, np.clip(confidence, 0.0, 1.0), [str(x) for x in tags])
        view["generator"] = "glm_5_1_market_skill_v1"
        out.append(view)
    return out


def generate_llm_views_for_card(card: pd.Series, cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    payload = _compact_card_for_prompt(card)
    messages = _llm_prompt(payload, cfg)
    text = _glm_chat_completion(messages, cfg)
    return _parse_llm_views(text, card, cfg), text


def build_llm_views(
    cards: pd.DataFrame | None = None,
    paths: MarketSkillPaths | None = None,
    cfg: dict[str, Any] | None = None,
    frequency: str | None = None,
    write_active: bool = False,
) -> pd.DataFrame:
    paths = paths or MarketSkillPaths()
    cfg = cfg or load_config()
    if cards is None:
        cards = pd.read_parquet(paths.cards)
    frequency = frequency or str(cfg.get("llm_views", {}).get("frequency", "month_end"))
    selected = cards_for_llm_frequency(cards, frequency)
    delay = float(cfg.get("llm_views", {}).get("request_delay_seconds", 0.2))
    views: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    for _, card in selected.iterrows():
        filters = _json_loads(card["view_filters"])
        try:
            candidates, raw_text = generate_llm_views_for_card(card, cfg)
            raw_rows.append({"date": card["date"], "status": "ok", "raw_response": raw_text})
        except Exception as exc:
            raw_rows.append({"date": card["date"], "status": "error", "raw_response": str(exc)})
            candidates = []
        for view in candidates:
            if not _view_asset_allowed(view["source_asset"], cfg, allow_cash=False) or not _view_asset_allowed(view["target_asset"], cfg, allow_cash=False):
                status, reason = "blocked", "asset_not_allowed_for_llm"
            else:
                status, reason = validate_view(view, filters, cfg)
            view["validator_status"] = status
            view["validator_reason"] = reason
            view["reason_tags"] = _json_dumps(sorted(set(view["reason_tags"])))
            views.append(view)
        if delay > 0:
            time.sleep(delay)
    out = _views_dataframe(views)
    if not out.empty:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out = out.sort_values(["date", "validator_status", "source_asset", "target_asset"])
    out.to_parquet(paths.llm_views, index=False)
    raw = pd.DataFrame(raw_rows)
    raw_path = REPORTS / "market_skill_llm_raw_responses.csv"
    raw.to_csv(raw_path, index=False, encoding="utf-8-sig")
    write_views_report(out, paths, report_path=paths.llm_view_report, title="MarketSkill LLM Views")
    if write_active:
        out.to_parquet(paths.views, index=False)
        write_views_report(out, paths)
    return out


def build_ep_views(views: pd.DataFrame | None = None, paths: MarketSkillPaths | None = None, source: str = "active") -> pd.DataFrame:
    paths = paths or MarketSkillPaths()
    if views is None:
        source_path = {
            "active": paths.views,
            "mechanical": paths.mechanical_views,
            "llm": paths.llm_views,
        }.get(source)
        if source_path is None:
            raise ValueError(f"Unknown EP view source: {source}")
        views = pd.read_parquet(source_path)
    if views.empty:
        ep = views
    else:
        ep = views[views["validator_status"].eq("valid")].copy()
        ep["ep_view_id"] = ep.groupby("date").cumcount() + 1
        ep["view_weight"] = ep["confidence"].clip(0.0, 1.0)
    ep.to_parquet(paths.ep_views, index=False)
    return ep


def write_views_report(
    views: pd.DataFrame,
    paths: MarketSkillPaths | None = None,
    report_path: Path | None = None,
    title: str = "MarketSkill Views",
) -> None:
    paths = paths or MarketSkillPaths()
    report_path = report_path or paths.view_report
    if views.empty:
        report_path.write_text(f"# {title}\n\nNo views generated.\n", encoding="utf-8")
        return
    latest_date = views["date"].max()
    latest = views[views["date"].eq(latest_date)].copy()
    lines = [f"# {title}", "", f"latest_date: {latest_date.date()}", ""]
    for _, row in latest.iterrows():
        lines.append(
            f"- {row['validator_status']}: {row['source_asset']} {row['relation']} vs {row['target_asset']} "
            f"(confidence={row['confidence']}, reason={row['validator_reason'] or 'ok'})"
        )
    valid_ratio = float(views["validator_status"].eq("valid").mean())
    lines.extend(["", f"valid_ratio_all_history: {valid_ratio:.3f}", f"rows: {len(views)}"])
    report_path.write_text("\n".join(lines), encoding="utf-8")

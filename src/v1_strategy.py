from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.ep_skfolio_optimizer import optimize_path
from src.rienet_covariance import estimate_covariance


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data_processed"


@dataclass(frozen=True)
class V1Paths:
    returns: Path = PROCESSED / "asset_returns_monthly.parquet"
    views: Path = PROCESSED / "ep_views_daily_v1.parquet"
    market_state_cards: Path = PROCESSED / "market_state_cards_daily_v1.parquet"
    rate_curves: Path = PROCESSED / "rate_curves_daily_v43.parquet"
    itransformer_predictions: Path = PROCESSED / "itransformer_predictions_monthly.parquet"
    asset_itransformer_predictions: Path = PROCESSED / "asset_itransformer_predictions_monthly.parquet"
    config: Path = ROOT / "config" / "v1_strategy_config.yaml"


RISK_SLEEVES = ["rates", "credit", "convertible_bonds", "equity", "alternatives"]


def load_config(path: Path | None = None) -> dict[str, Any]:
    with (path or V1Paths().config).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _output_path(cfg: dict[str, Any], key: str) -> Path:
    return ROOT / cfg["outputs"][key]


def write_data_quality_report(cfg: dict[str, Any] | None = None, paths: V1Paths | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    paths = paths or V1Paths()
    raw = pd.read_parquet(paths.returns)
    close = pd.to_numeric(raw.get("close"), errors="coerce")
    ret = pd.to_numeric(raw.get("return_1m"), errors="coerce")
    max_abs_return = float(cfg.get("data_cleaning", {}).get("max_abs_monthly_return", 0.80))
    rows = [
        {"check": "rows", "value": len(raw)},
        {"check": "nonfinite_close_rows", "value": int(np.isinf(close).sum())},
        {"check": "nonfinite_return_rows", "value": int(np.isinf(ret).sum())},
        {"check": "abs_return_above_cap_rows", "value": int((ret.abs() > max_abs_return).sum())},
        {"check": "max_abs_monthly_return_cap", "value": max_abs_return},
    ]
    out = pd.DataFrame(rows)
    if "data_quality" in cfg.get("outputs", {}):
        path = _output_path(cfg, "data_quality")
        path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(path, index=False, encoding="utf-8-sig")
    return out


def selected_profile(cfg: dict[str, Any]) -> dict[str, Any]:
    name = cfg["product"]["selected_profile"]
    return cfg["product"]["profiles"][name]


def apply_selected_profile_overrides(cfg: dict[str, Any]) -> dict[str, Any]:
    profile = selected_profile(cfg)
    ep_overrides = profile.get("ep_skfolio_overrides", {})
    if ep_overrides:
        cfg.setdefault("ep_skfolio", {}).update(ep_overrides)
    overlay_overrides = profile.get("signal_overlay_overrides", {})
    for overlay_name, values in overlay_overrides.items():
        if isinstance(values, dict):
            cfg.setdefault("signal_overlays", {}).setdefault(overlay_name, {}).update(values)
    version = str(cfg.get("strategy_version", "no_agent"))
    version_overrides = cfg.get("strategy_version_overrides", {}).get(version, {})
    if isinstance(version_overrides, dict):
        ep_version_overrides = version_overrides.get("ep_skfolio", {})
        if isinstance(ep_version_overrides, dict):
            cfg.setdefault("ep_skfolio", {}).update(ep_version_overrides)
        overlay_version_overrides = version_overrides.get("signal_overlays", {})
        if isinstance(overlay_version_overrides, dict):
            for overlay_name, values in overlay_version_overrides.items():
                if isinstance(values, dict):
                    cfg.setdefault("signal_overlays", {}).setdefault(overlay_name, {}).update(values)
        view_scale_overrides = version_overrides.get("view_tilt_profile_scale", {})
        if isinstance(view_scale_overrides, dict):
            cfg.setdefault("view_tilt_profile_scale", {}).update(view_scale_overrides)
        profile_version_overrides = version_overrides.get("profiles", {}).get(cfg["product"]["selected_profile"], {})
        if isinstance(profile_version_overrides, dict):
            ep_profile_version_overrides = profile_version_overrides.get("ep_skfolio", {})
            if isinstance(ep_profile_version_overrides, dict):
                cfg.setdefault("ep_skfolio", {}).update(ep_profile_version_overrides)
            overlay_profile_version_overrides = profile_version_overrides.get("signal_overlays", {})
            if isinstance(overlay_profile_version_overrides, dict):
                for overlay_name, values in overlay_profile_version_overrides.items():
                    if isinstance(values, dict):
                        cfg.setdefault("signal_overlays", {}).setdefault(overlay_name, {}).update(values)
            view_profile_scale_overrides = profile_version_overrides.get("view_tilt_profile_scale", {})
            if isinstance(view_profile_scale_overrides, dict):
                cfg.setdefault("view_tilt_profile_scale", {}).update(view_profile_scale_overrides)
    return cfg


def sleeve_bounds(cfg: dict[str, Any]) -> dict[str, tuple[float, float]]:
    profile = selected_profile(cfg)
    return {k: tuple(map(float, v)) for k, v in profile["sleeve_bounds"].items()}


def target_risk_contribution(cfg: dict[str, Any]) -> pd.Series:
    profile = selected_profile(cfg)
    target = pd.Series(profile["target_risk_contribution"], dtype=float).reindex(RISK_SLEEVES)
    return target / target.sum()


def build_rate_curve_proxy_returns(cfg: dict[str, Any], paths: V1Paths | None = None) -> pd.DataFrame:
    paths = paths or V1Paths()
    proxy_cfg = cfg.get("synthetic_assets", {}).get("rate_30y_curve_proxy", {})
    if not proxy_cfg.get("enabled", False):
        return pd.DataFrame()
    if not paths.rate_curves.exists():
        raise FileNotFoundError(f"Missing rate curve file: {paths.rate_curves}")

    curves = pd.read_parquet(paths.rate_curves)
    curves["date"] = pd.to_datetime(curves["date"], errors="coerce")
    y_col = str(proxy_cfg.get("yield_column", "treasury_y30"))
    if y_col not in curves.columns:
        raise KeyError(f"Missing yield column for rate curve proxy: {y_col}")

    daily = curves[["date", y_col]].dropna(subset=["date"]).sort_values("date").copy()
    daily[y_col] = pd.to_numeric(daily[y_col], errors="coerce")
    daily = daily.dropna(subset=[y_col])
    monthly = daily.set_index("date")[y_col].resample("ME").last().dropna().to_frame("yield_pct")
    if monthly.empty:
        return pd.DataFrame()

    duration = float(proxy_cfg.get("modified_duration", 18.5))
    carry = monthly["yield_pct"].shift(1) / 100.0 / 12.0
    dy = monthly["yield_pct"].diff() / 100.0
    ret = carry - duration * dy
    max_abs_return = float(cfg.get("data_cleaning", {}).get("max_abs_monthly_return", 0.80))
    ret = ret.where(ret.abs().le(max_abs_return))
    close = (1.0 + ret.fillna(0.0)).cumprod() * float(proxy_cfg.get("base_close", 100.0))

    out = pd.DataFrame(
        {
            "date": monthly.index,
            "asset_code": str(proxy_cfg.get("asset_code", "rate_30y_curve_proxy")),
            "code": str(proxy_cfg.get("asset_code", "rate_30y_curve_proxy")),
            "asset_name": str(proxy_cfg.get("asset_name", "30Y国债曲线代理")),
            "asset_class": "rates",
            "module": "rates",
            "sub_module": "rate_30y_curve_proxy",
            "close": close.to_numpy(dtype=float),
            "return_1m": ret.to_numpy(dtype=float),
            "data_source": f"synthetic:{y_col}:duration_{duration:g}",
            "fetch_time": "generated_in_v1_strategy",
            "is_proxy": True,
        }
    )
    start = pd.Timestamp(cfg["rebalance"]["start_date"])
    end = pd.Timestamp(cfg["rebalance"]["end_date"])
    return out[out["date"].between(start, end)].reset_index(drop=True)


def write_synthetic_assets(cfg: dict[str, Any] | None = None, paths: V1Paths | None = None) -> dict[str, str]:
    cfg = cfg or load_config()
    paths = paths or V1Paths()
    written: dict[str, str] = {}
    proxy = build_rate_curve_proxy_returns(cfg, paths)
    if not proxy.empty:
        out_path = PROCESSED / "rate_30y_curve_proxy_monthly.parquet"
        proxy.to_parquet(out_path, index=False)
        written["rate_30y_curve_proxy"] = str(out_path.relative_to(ROOT))
    return written


def load_monthly_returns(paths: V1Paths | None = None, cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    paths = paths or V1Paths()
    cfg = cfg or load_config()
    df = pd.read_parquet(paths.returns)
    proxy = build_rate_curve_proxy_returns(cfg, paths)
    if not proxy.empty:
        proxy_code = str(cfg["synthetic_assets"]["rate_30y_curve_proxy"].get("asset_code", "rate_30y_curve_proxy"))
        df = df[df["asset_code"].astype(str).ne(proxy_code)].copy()
        df = pd.concat([df, proxy], ignore_index=True, sort=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    df["return_1m"] = pd.to_numeric(df["return_1m"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    max_abs_return = float(cfg.get("data_cleaning", {}).get("max_abs_monthly_return", 0.80))
    df.loc[df["return_1m"].abs() > max_abs_return, "return_1m"] = np.nan
    df = df.dropna(subset=["date", "asset_code", "module", "close"]).copy()
    return df.sort_values(["date", "asset_code"]).reset_index(drop=True)


def build_sleeve_returns(paths: V1Paths | None = None, cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    returns = load_monthly_returns(paths, cfg)
    work = returns[returns["module"].isin(RISK_SLEEVES)].copy()
    method = str(cfg.get("risk_budget", {}).get("sleeve_return_aggregation", "equal_assets"))
    if method == "equal_assets":
        long = work.groupby(["date", "module"], as_index=False)["return_1m"].mean()
    elif method in {"preferred_assets_equal", "preferred_asset_weights"}:
        parts: list[pd.DataFrame] = []
        for sleeve in RISK_SLEEVES:
            part = work[work["module"].eq(sleeve)].copy()
            preferred = _preferred_asset_weight_map(sleeve, cfg)
            if preferred:
                part = part[part["asset_code"].astype(str).isin(preferred)].copy()
            if part.empty:
                continue
            if method == "preferred_asset_weights" and preferred:
                part["aggregation_weight"] = part["asset_code"].astype(str).map(preferred).fillna(0.0)
            else:
                part["aggregation_weight"] = 1.0
            part["return_1m"] = pd.to_numeric(part["return_1m"], errors="coerce")
            part = part.dropna(subset=["return_1m"])
            part["weighted_return"] = part["return_1m"] * part["aggregation_weight"]
            grouped = part.groupby("date", as_index=False).agg(
                weighted_return=("weighted_return", "sum"),
                weight_sum=("aggregation_weight", "sum"),
            )
            grouped["return_1m"] = grouped["weighted_return"] / grouped["weight_sum"].replace(0.0, np.nan)
            grouped["module"] = sleeve
            parts.append(grouped[["date", "module", "return_1m"]])
        long = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["date", "module", "return_1m"])
    else:
        raise ValueError(f"Unknown risk_budget.sleeve_return_aggregation={method}")
    sleeve_ret = (
        long.pivot(index="date", columns="module", values="return_1m")
        .reindex(columns=RISK_SLEEVES)
        .sort_index()
    )
    return sleeve_ret.fillna(0.0)


def _tail_view_dates(views: pd.DataFrame, count: int) -> pd.DataFrame:
    count = max(int(count), 1)
    keep = (
        views[["month", "date"]]
        .drop_duplicates()
        .sort_values(["month", "date"])
        .groupby("month", group_keys=False)
        .tail(count)
        .assign(_keep=True)
    )
    return views.merge(keep, on=["month", "date"], how="inner").drop(columns="_keep")


def _aggregate_view_pairs(views: pd.DataFrame, persistence_power: float = 0.0) -> pd.DataFrame:
    if views.empty:
        return views
    keys = ["month", "source_asset", "target_asset"]
    latest_idx = views.sort_values("date").groupby(keys)["date"].idxmax()
    out = views.loc[latest_idx].copy().set_index(keys)
    grouped = views.groupby(keys, dropna=False)
    out["confidence"] = grouped["confidence"].mean().astype(float)
    observations = grouped["date"].nunique().astype(float)
    available_dates = views.groupby("month")["date"].nunique().astype(float)
    persistence = observations / observations.index.get_level_values("month").map(available_dates)
    persistence = persistence.clip(lower=0.0, upper=1.0)
    if persistence_power > 0:
        out["confidence"] *= persistence.pow(float(persistence_power))
    out["view_observations"] = observations.astype(int)
    out["view_persistence"] = persistence.astype(float)
    return out.reset_index().sort_values(["month", "source_asset", "target_asset"]).reset_index(drop=True)


def month_end_views(paths: V1Paths | None = None, cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    paths = paths or V1Paths()
    views = pd.read_parquet(paths.views)
    if views.empty:
        return views
    views["date"] = pd.to_datetime(views["date"], errors="coerce")
    views = views.dropna(subset=["date"]).copy()
    views["month"] = views["date"] + pd.offsets.MonthEnd(0)
    views = views[views["validator_status"].eq("valid")].copy()
    preprocessing = (cfg or {}).get("signal_overlays", {}).get("view_preprocessing", {})
    mode = str(preprocessing.get("mode", "raw_month")).lower()
    recent_days = max(int(preprocessing.get("recent_days", 5)), 1)
    persistence_power = float(preprocessing.get("persistence_power", 0.5))
    if mode == "raw_month":
        return views
    if mode == "last_day":
        return _tail_view_dates(views, 1)
    if mode == "recent_days_raw":
        return _tail_view_dates(views, recent_days)
    if mode == "monthly_pair_mean":
        return _aggregate_view_pairs(views, persistence_power=0.0)
    if mode == "monthly_pair_persistence":
        return _aggregate_view_pairs(views, persistence_power=persistence_power)
    if mode in {"recent_pair_mean", "recent_pair_persistence"}:
        recent = _tail_view_dates(views, recent_days)
        power = persistence_power if mode == "recent_pair_persistence" else 0.0
        return _aggregate_view_pairs(recent, persistence_power=power)
    raise ValueError(f"Unsupported signal_overlays.view_preprocessing.mode={mode!r}")


def month_end_rule_cards(cfg: dict[str, Any], paths: V1Paths | None = None) -> pd.DataFrame:
    paths = paths or V1Paths()
    overlay = cfg.get("signal_overlays", {}).get("rule_output", {})
    if not overlay.get("enabled", False):
        return pd.DataFrame()
    card_path = ROOT / str(overlay.get("state_cards_path", paths.market_state_cards.relative_to(ROOT)))
    if not card_path.exists():
        return pd.DataFrame()
    cards = pd.read_parquet(card_path)
    if cards.empty:
        return cards
    cards["date"] = pd.to_datetime(cards["date"], errors="coerce")
    cards = cards.dropna(subset=["date"]).sort_values("date")
    cards["month"] = cards["date"] + pd.offsets.MonthEnd(0)
    idx = cards.groupby("month")["date"].idxmax()
    return cards.loc[idx].sort_values("month").reset_index(drop=True)


def _project_weights(w: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    w = np.clip(np.asarray(w, dtype=float), lo, hi)
    for _ in range(100):
        diff = 1.0 - w.sum()
        if abs(diff) < 1e-12:
            break
        if diff > 0:
            room = hi - w
            active = room > 1e-12
            if not active.any():
                break
            w[active] += diff * room[active] / room[active].sum()
        else:
            room = w - lo
            active = room > 1e-12
            if not active.any():
                break
            w[active] += diff * room[active] / room[active].sum()
        w = np.clip(w, lo, hi)
    return w / w.sum()


def _clean_cov(window: pd.DataFrame, cfg: dict[str, Any]) -> np.ndarray:
    rb = cfg["risk_budget"]
    cov = window.cov().reindex(index=RISK_SLEEVES, columns=RISK_SLEEVES).fillna(0.0).to_numpy(dtype=float)
    floor = float(rb["diagonal_vol_floor"]) ** 2
    diag = np.diag(np.maximum(np.diag(cov), floor))
    shrink = float(rb["covariance_shrinkage_to_diag"])
    cov = (1.0 - shrink) * cov + shrink * diag
    cov = (cov + cov.T) / 2.0
    cov += np.eye(len(RISK_SLEEVES)) * floor
    return cov


def solve_risk_budget(cov: np.ndarray, target_rc: np.ndarray, bounds: dict[str, tuple[float, float]], cfg: dict[str, Any]) -> np.ndarray:
    lo = np.array([bounds[s][0] for s in RISK_SLEEVES], dtype=float)
    hi = np.array([bounds[s][1] for s in RISK_SLEEVES], dtype=float)
    vol = np.sqrt(np.maximum(np.diag(cov), float(cfg["risk_budget"]["diagonal_vol_floor"]) ** 2))
    w = target_rc / vol
    w = _project_weights(w / w.sum(), lo, hi)
    for _ in range(int(cfg["risk_budget"]["max_iterations"])):
        mrc = cov @ w
        rc = np.maximum(w * mrc, 1e-12)
        rc_share = rc / rc.sum()
        err = rc_share - target_rc
        if np.max(np.abs(err)) < float(cfg["risk_budget"]["tolerance"]):
            break
        w *= np.sqrt(np.maximum(target_rc, 1e-12) / np.maximum(rc_share, 1e-12))
        w = _project_weights(w, lo, hi)
    return w


def build_risk_budget_base(cfg: dict[str, Any] | None = None, paths: V1Paths | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    cfg = apply_selected_profile_overrides(cfg)
    paths = paths or V1Paths()
    write_data_quality_report(cfg, paths)
    sleeve_ret = build_sleeve_returns(paths, cfg)
    target = target_risk_contribution(cfg)
    bounds = sleeve_bounds(cfg)
    start = pd.Timestamp(cfg["rebalance"]["start_date"])
    end = pd.Timestamp(cfg["rebalance"]["end_date"])
    dates = [d for d in sleeve_ret.index if start <= d <= end]
    rows: list[dict[str, Any]] = []
    lookback = int(cfg["risk_budget"]["lookback_months"])
    min_history = int(cfg["risk_budget"]["min_history_months"])
    for date in dates:
        hist = sleeve_ret.loc[:date].tail(lookback)
        if len(hist) < min_history:
            weights = target.to_numpy(dtype=float)
            weights = _project_weights(weights, np.array([bounds[s][0] for s in RISK_SLEEVES]), np.array([bounds[s][1] for s in RISK_SLEEVES]))
            method = "target_rc_until_min_history"
        else:
            cov = _clean_cov(hist, cfg)
            weights = solve_risk_budget(cov, target.to_numpy(dtype=float), bounds, cfg)
            method = "rolling_risk_budget"
        row = {"date": date, "method": method, "product_profile": cfg["product"]["selected_profile"]}
        row.update({s: float(weights[i]) for i, s in enumerate(RISK_SLEEVES)})
        row["cash"] = 0.0
        rows.append(row)
    out = pd.DataFrame(rows)
    out_path = _output_path(cfg, "risk_budget_base")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    return out


def _clip_sleeve_dict(weights: dict[str, float], cfg: dict[str, Any]) -> dict[str, float]:
    bounds = sleeve_bounds(cfg)
    all_sleeves = RISK_SLEEVES + ["cash"]
    lo = np.array([bounds[s][0] for s in all_sleeves])
    hi = np.array([bounds[s][1] for s in all_sleeves])
    w = np.array([weights.get(s, 0.0) for s in all_sleeves])
    w = _project_weights(w, lo, hi)
    return {s: float(w[i]) for i, s in enumerate(all_sleeves)}


def _json_loads_safe(value: Any, default: Any | None = None) -> Any:
    if default is None:
        default = {}
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    return value


def _rule_by_month(rule_cards: pd.DataFrame) -> dict[pd.Timestamp, pd.Series]:
    if rule_cards.empty:
        return {}
    work = rule_cards.copy()
    if "month" not in work.columns:
        work["month"] = pd.to_datetime(work["date"], errors="coerce") + pd.offsets.MonthEnd(0)
    return {pd.Timestamp(row["month"]): row for _, row in work.iterrows()}


def _rule_output_for_row(row: pd.Series | None) -> dict[str, Any]:
    if row is None:
        return {}
    return _json_loads_safe(row.get("rule_output_object", "{}"), {})


def _adjust_speed_from_rule(rule: dict[str, Any], cfg: dict[str, Any]) -> float:
    overlay = cfg.get("signal_overlays", {}).get("rule_output", {})
    if not overlay.get("enabled", False) or not overlay.get("apply_adjust_speed", False):
        return 1.0
    guidance = rule.get("execution_guidance", {}) if isinstance(rule, dict) else {}
    key = str(guidance.get("adjust_speed", "normal_60pct"))
    speed_map = overlay.get("adjust_speed_map", {})
    return float(np.clip(float(speed_map.get(key, 1.0)), 0.0, 1.0))


def _apply_rule_adjust_speed(weights: pd.DataFrame, rule_cards: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    overlay = cfg.get("signal_overlays", {}).get("rule_output", {})
    if not overlay.get("enabled", False) or not overlay.get("apply_adjust_speed", False) or rule_cards.empty:
        return weights
    rule_map = _rule_by_month(rule_cards)
    rows: list[dict[str, Any]] = []
    prev: dict[str, float] | None = None
    for _, row in weights.sort_values("date").iterrows():
        raw = {s: float(row[s]) for s in RISK_SLEEVES + ["cash"]}
        rule = _rule_output_for_row(rule_map.get(pd.Timestamp(row["date"])))
        speed = _adjust_speed_from_rule(rule, cfg)
        if prev is None or speed >= 0.999:
            adjusted = raw
        else:
            adjusted = {s: prev.get(s, 0.0) + speed * (raw.get(s, 0.0) - prev.get(s, 0.0)) for s in RISK_SLEEVES + ["cash"]}
            adjusted = _clip_sleeve_dict(adjusted, cfg)
        out = row.to_dict()
        out.update(adjusted)
        out["rule_adjust_speed"] = speed
        rows.append(out)
        prev = {s: float(out[s]) for s in RISK_SLEEVES + ["cash"]}
    return pd.DataFrame(rows)


def _view_to_sleeve(asset: str, cfg: dict[str, Any]) -> str:
    return cfg.get("asset_mapping", {}).get(asset, {}).get("module", asset)


def calibrate_market_view_confidence(
    views: pd.DataFrame,
    sleeve_ret: pd.DataFrame,
    cfg: dict[str, Any],
) -> pd.DataFrame:
    overlay = cfg.get("signal_overlays", {}).get("view_reliability", {})
    if not overlay.get("enabled", False) or views.empty or sleeve_ret.empty:
        return views
    if str(overlay.get("group_by", "sleeve_direction")) != "sleeve_direction":
        raise ValueError("view_reliability currently supports group_by=sleeve_direction only")

    out = views.copy().sort_values(["month", "date"]).reset_index(drop=True)
    out["source_sleeve"] = out["source_asset"].astype(str).map(lambda value: _view_to_sleeve(value, cfg))
    out["target_sleeve"] = out["target_asset"].astype(str).map(lambda value: _view_to_sleeve(value, cfg))
    valid = (
        out["source_sleeve"].isin(RISK_SLEEVES)
        & out["target_sleeve"].isin(RISK_SLEEVES)
        & out["source_sleeve"].ne(out["target_sleeve"])
    )
    skill = (
        out.loc[valid]
        .groupby(["month", "source_sleeve", "target_sleeve"], as_index=False)
        .agg(historical_confidence=("confidence", "mean"))
        .sort_values("month")
    )
    if skill.empty:
        return views

    return_dates = pd.DatetimeIndex(pd.to_datetime(sleeve_ret.index)).sort_values().unique()
    next_month: dict[pd.Timestamp, pd.Timestamp] = {}
    for month in pd.DatetimeIndex(skill["month"].unique()).sort_values():
        future = return_dates[return_dates > month]
        if len(future):
            next_month[pd.Timestamp(month)] = pd.Timestamp(future[0])
    skill["outcome_month"] = skill["month"].map(next_month)

    def realized_spread(row: pd.Series) -> float:
        outcome_month = row["outcome_month"]
        if pd.isna(outcome_month):
            return np.nan
        source = str(row["source_sleeve"])
        target = str(row["target_sleeve"])
        if source not in sleeve_ret.columns or target not in sleeve_ret.columns:
            return np.nan
        return float(sleeve_ret.loc[pd.Timestamp(outcome_month), source] - sleeve_ret.loc[pd.Timestamp(outcome_month), target])

    skill["realized_spread"] = skill.apply(realized_spread, axis=1)
    skill["hit"] = skill["realized_spread"].gt(0.0).astype(float)

    lookback_months = max(int(overlay.get("lookback_months", 36)), 1)
    min_observations = max(int(overlay.get("min_observations", 8)), 1)
    prior_strength = max(float(overlay.get("prior_strength", 8.0)), 0.0)
    sensitivity = max(float(overlay.get("sensitivity", 1.0)), 0.0)
    minimum_factor = float(overlay.get("minimum_factor", 0.50))
    maximum_factor = float(overlay.get("maximum_factor", 1.35))
    confidence_weighted = bool(overlay.get("confidence_weighted", True))
    only_downweight = bool(overlay.get("only_downweight", False))
    confidence_cap = float(overlay.get("confidence_cap", 1.0))

    factors: list[float] = []
    posterior_rates: list[float] = []
    observation_counts: list[int] = []
    for _, row in out.iterrows():
        if row["source_sleeve"] == row["target_sleeve"] or row["source_sleeve"] not in RISK_SLEEVES or row["target_sleeve"] not in RISK_SLEEVES:
            factors.append(1.0)
            posterior_rates.append(0.5)
            observation_counts.append(0)
            continue
        decision_month = pd.Timestamp(row["month"])
        history = skill[
            skill["source_sleeve"].eq(row["source_sleeve"])
            & skill["target_sleeve"].eq(row["target_sleeve"])
            & skill["outcome_month"].notna()
            & skill["outcome_month"].le(decision_month)
            & skill["realized_spread"].notna()
        ].sort_values("outcome_month").tail(lookback_months)
        count = int(len(history))
        observation_counts.append(count)
        if count < min_observations:
            factors.append(1.0)
            posterior_rates.append(0.5)
            continue
        weights = history["historical_confidence"].clip(lower=0.0).astype(float) if confidence_weighted else pd.Series(1.0, index=history.index)
        effective_n = float(weights.sum())
        posterior = (0.5 * prior_strength + float((weights * history["hit"]).sum())) / max(prior_strength + effective_n, 1e-12)
        factor = 1.0 + sensitivity * 2.0 * (posterior - 0.5)
        if only_downweight:
            factor = min(factor, 1.0)
        factor = float(np.clip(factor, minimum_factor, maximum_factor))
        factors.append(factor)
        posterior_rates.append(float(posterior))

    out["raw_confidence_before_reliability"] = pd.to_numeric(out["confidence"], errors="coerce").fillna(0.0)
    out["view_reliability_factor"] = factors
    out["view_reliability_hit_rate"] = posterior_rates
    out["view_reliability_observations"] = observation_counts
    out["confidence"] = (out["raw_confidence_before_reliability"] * out["view_reliability_factor"]).clip(lower=0.0, upper=confidence_cap)
    return out.drop(columns=["source_sleeve", "target_sleeve"])


def _apply_view(weights: dict[str, float], source: str, target: str, confidence: float, cfg: dict[str, Any]) -> None:
    source_sleeve = _view_to_sleeve(source, cfg)
    target_sleeve = _view_to_sleeve(target, cfg)
    if "cash" in {source_sleeve, target_sleeve}:
        return
    if source_sleeve not in weights or target_sleeve not in weights or source_sleeve == target_sleeve:
        return
    tilt_key = source if source in cfg["view_tilts"] else source_sleeve
    profile = cfg["product"]["selected_profile"]
    profile_scale = float(cfg.get("view_tilt_profile_scale", {}).get(profile, 1.0))
    application_scale = float(cfg.get("signal_overlays", {}).get("view_application", {}).get("direct_tilt_scale", 1.0))
    tilt = float(cfg["view_tilts"].get(tilt_key, 0.02)) * float(confidence) * profile_scale * application_scale
    weights[source_sleeve] += tilt
    weights[target_sleeve] -= tilt


def _trend_signal_scores(sleeve_ret: pd.DataFrame, date: pd.Timestamp, cfg: dict[str, Any]) -> dict[str, float]:
    overlay = cfg.get("signal_overlays", {}).get("trend_following", {})
    if not overlay.get("enabled", False):
        return {}
    lookbacks = [int(x) for x in overlay.get("lookback_months", [3, 6, 12])]
    hist = sleeve_ret.loc[:date].copy()
    scores: dict[str, float] = {}
    for sleeve in RISK_SLEEVES:
        parts = []
        for lb in lookbacks:
            if len(hist) < lb:
                continue
            value = float(hist[sleeve].tail(lb).sum())
            parts.append(value)
        if parts:
            scores[sleeve] = float(np.mean(parts))
    return scores


def _transfer_weight(weights: dict[str, float], source: str, receivers: list[str], amount: float) -> None:
    if amount <= 0 or source not in weights:
        return
    amount = min(amount, max(weights.get(source, 0.0), 0.0))
    active = [s for s in receivers if s in weights and s != source]
    if not active:
        return
    weights[source] -= amount
    add = amount / len(active)
    for sleeve in active:
        weights[sleeve] += add


def _apply_trend_overlay(weights: dict[str, float], sleeve_ret: pd.DataFrame, date: pd.Timestamp, cfg: dict[str, Any]) -> None:
    overlay = cfg.get("signal_overlays", {}).get("trend_following", {})
    if not overlay.get("enabled", False):
        return
    enabled_profiles = overlay.get("enabled_profiles")
    if enabled_profiles and cfg["product"]["selected_profile"] not in set(enabled_profiles):
        return
    scores = _trend_signal_scores(sleeve_ret, date, cfg)
    if not scores:
        return
    source_sleeves = list(overlay.get("source_sleeves", ["convertible_bonds", "equity"]))
    receiver_sleeves = list(overlay.get("receiver_sleeves", ["rates", "credit", "alternatives"]))
    positive_threshold = float(overlay.get("positive_threshold", 0.0))
    negative_threshold = float(overlay.get("negative_threshold", 0.0))
    tilt_per_score = float(overlay.get("tilt_per_score", 0.015))
    max_sleeve_tilt = float(overlay.get("max_sleeve_tilt", 0.05))

    positive_receivers = [s for s in receiver_sleeves if scores.get(s, 0.0) > positive_threshold]
    if not positive_receivers:
        positive_receivers = sorted(receiver_sleeves, key=lambda s: scores.get(s, -np.inf), reverse=True)[:2]

    for source in source_sleeves:
        score = scores.get(source)
        if score is None or score >= negative_threshold:
            continue
        amount = min(max_sleeve_tilt, tilt_per_score * abs(score) / 0.03)
        _transfer_weight(weights, source, positive_receivers, amount)


def _apply_risk_on_rotation(weights: dict[str, float], sleeve_ret: pd.DataFrame, date: pd.Timestamp, cfg: dict[str, Any]) -> None:
    overlay = cfg.get("signal_overlays", {}).get("risk_on_rotation", {})
    if not overlay.get("enabled", False):
        return
    profile = str(cfg.get("product", {}).get("selected_profile", ""))
    enabled_profiles = set(overlay.get("enabled_profiles", []))
    if enabled_profiles and profile not in enabled_profiles:
        return

    lookbacks = sorted({max(1, int(x)) for x in overlay.get("lookback_months", [6, 12])})
    hist = sleeve_ret.loc[:date].sort_index()
    if len(hist) < max(lookbacks):
        return
    threshold = float(overlay.get("positive_threshold", 0.0))
    require_all = bool(overlay.get("require_all_lookbacks_positive", True))
    candidates = [s for s in overlay.get("receiver_sleeves", ["equity", "alternatives"]) if s in RISK_SLEEVES]
    scores: dict[str, float] = {}
    for sleeve in candidates:
        series = pd.to_numeric(hist[sleeve], errors="coerce").dropna()
        if len(series) < max(lookbacks):
            continue
        parts = [float((1.0 + series.tail(lb)).prod() - 1.0) for lb in lookbacks]
        if require_all and any(value <= threshold for value in parts):
            continue
        score = float(np.mean(parts))
        if score > threshold:
            scores[sleeve] = score
    if not scores:
        return

    top_n = max(1, int(overlay.get("top_n", 1)))
    selected = sorted(scores, key=scores.get, reverse=True)[:top_n]
    max_tilt_map = overlay.get("profile_max_tilt", {})
    max_tilt = float(max_tilt_map.get(profile, overlay.get("max_tilt", 0.0)))
    momentum_scale = max(float(overlay.get("momentum_scale", 0.10)), 1e-8)
    strength = float(np.clip(max(scores[s] for s in selected) / momentum_scale, 0.0, 1.0))
    requested = max_tilt * strength
    if requested <= 0:
        return

    bounds = sleeve_bounds(cfg)
    donors = [s for s in overlay.get("donor_sleeves", ["rates", "credit"]) if s in RISK_SLEEVES and s not in selected]
    donor_room = pd.Series({s: max(weights.get(s, 0.0) - bounds[s][0], 0.0) for s in donors}, dtype=float)
    receiver_room = pd.Series({s: max(bounds[s][1] - weights.get(s, 0.0), 0.0) for s in selected}, dtype=float)
    amount = min(requested, float(donor_room.sum()), float(receiver_room.sum()))
    if amount <= 1e-12:
        return
    donor_cut = donor_room / donor_room.sum() * amount
    receiver_score = pd.Series({s: max(scores[s] - threshold, 1e-12) for s in selected}, dtype=float)
    receiver_add = receiver_score / receiver_score.sum() * amount
    for sleeve, value in donor_cut.items():
        weights[sleeve] -= float(value)
    for sleeve, value in receiver_add.items():
        weights[sleeve] += float(value)


def load_itransformer_predictions(cfg: dict[str, Any], paths: V1Paths | None = None) -> pd.DataFrame:
    paths = paths or V1Paths()
    overlay = cfg.get("signal_overlays", {}).get("itransformer_rank", {})
    if not overlay.get("enabled", False):
        return pd.DataFrame()
    if overlay.get("require_live_gate", False):
        gate_path = ROOT / str(overlay.get("gate_report_path", ""))
        if not gate_path.exists():
            raise FileNotFoundError(f"Missing iTransformer gate report: {gate_path}")
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        if not bool(gate.get("enabled_for_formal_strategy", False)):
            raise RuntimeError(f"iTransformer formal gate is closed: {gate_path}")
    pred_path = ROOT / str(overlay.get("predictions_path", paths.itransformer_predictions.relative_to(ROOT)))
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing iTransformer prediction file: {pred_path}")
    pred = pd.read_parquet(pred_path)
    pred["as_of_date"] = pd.to_datetime(pred["as_of_date"], errors="coerce")
    pred["sleeve"] = pred["sleeve"].astype(str)
    pred["pred_return"] = pd.to_numeric(pred["pred_return"], errors="coerce")
    pred["pred_prob"] = pd.to_numeric(pred.get("pred_prob"), errors="coerce")
    pred = pred[pred["sleeve"].isin(RISK_SLEEVES)].dropna(subset=["as_of_date", "pred_return"])
    return pred.sort_values(["as_of_date", "pred_return"], ascending=[True, False]).reset_index(drop=True)


def _apply_itransformer_overlay(weights: dict[str, float], pred: pd.DataFrame, date: pd.Timestamp, cfg: dict[str, Any]) -> None:
    overlay = cfg.get("signal_overlays", {}).get("itransformer_rank", {})
    if not overlay.get("enabled", False) or not overlay.get("direct_tilt_enabled", True):
        return
    enabled_profiles = overlay.get("enabled_profiles")
    if enabled_profiles and cfg["product"]["selected_profile"] not in set(enabled_profiles):
        return
    if pred.empty:
        return
    active = pred[pred["as_of_date"].eq(pd.Timestamp(date))].dropna(subset=["pred_return"]).copy()
    if active.empty:
        return
    active = active.sort_values("pred_return", ascending=False)
    top_n = max(1, int(overlay.get("top_n", 1)))
    bottom_n = max(1, int(overlay.get("bottom_n", 1)))
    top = [s for s in active.head(top_n)["sleeve"].tolist() if s in weights]
    bottom = [s for s in active.tail(bottom_n)["sleeve"].tolist() if s in weights and s not in top]
    if not top or not bottom:
        return
    min_top_prob = float(overlay.get("min_top_prob", 0.0))
    top_prob = float(active.head(top_n)["pred_prob"].fillna(0.0).max())
    if top_prob < min_top_prob:
        return
    total_tilt = float(overlay.get("tilt", 0.02))
    amount = total_tilt / len(bottom)
    for source in bottom:
        _transfer_weight(weights, source, top, amount)


def _build_overlay_sleeve_weights(
    base: pd.DataFrame,
    views: pd.DataFrame,
    sleeve_ret: pd.DataFrame,
    itransformer_pred: pd.DataFrame,
    cfg: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    view_application = cfg.get("signal_overlays", {}).get("view_application", {})
    direct_tilt_enabled = bool(view_application.get("direct_tilt_enabled", True))
    for _, base_row in base.iterrows():
        weights = {s: float(base_row[s]) for s in RISK_SLEEVES + ["cash"]}
        date = base_row["date"]
        if direct_tilt_enabled and not views.empty:
            active = views[views["month"].eq(date)]
            for _, view in active.iterrows():
                _apply_view(weights, str(view["source_asset"]), str(view["target_asset"]), float(view["confidence"]), cfg)
        _apply_trend_overlay(weights, sleeve_ret, pd.Timestamp(date), cfg)
        _apply_risk_on_rotation(weights, sleeve_ret, pd.Timestamp(date), cfg)
        _apply_itransformer_overlay(weights, itransformer_pred, pd.Timestamp(date), cfg)
        weights = _clip_sleeve_dict(weights, cfg)
        row = {"date": date, "product_profile": cfg["product"]["selected_profile"], "base_method": base_row["method"]}
        row.update(weights)
        rows.append(row)
    return pd.DataFrame(rows)


def build_sleeve_weights(cfg: dict[str, Any] | None = None, paths: V1Paths | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    cfg = apply_selected_profile_overrides(cfg)
    paths = paths or V1Paths()
    base = build_risk_budget_base(cfg, paths)
    sleeve_ret = build_sleeve_returns(paths, cfg)
    views = calibrate_market_view_confidence(month_end_views(paths, cfg), sleeve_ret, cfg)
    rule_cards = month_end_rule_cards(cfg, paths)
    itransformer_pred = load_itransformer_predictions(cfg, paths)
    overlay_base = _build_overlay_sleeve_weights(base, views, sleeve_ret, itransformer_pred, cfg)
    if cfg.get("ep_skfolio", {}).get("enabled", False):
        view_application = cfg.get("signal_overlays", {}).get("view_application", {})
        optimizer_views = views if bool(view_application.get("ep_mean_enabled", True)) else views.iloc[0:0].copy()
        out, posterior, signals = optimize_path(overlay_base, sleeve_ret, optimizer_views, itransformer_pred, sleeve_bounds(cfg), cfg)
        out = _apply_rule_adjust_speed(out, rule_cards, cfg)
        out_path = _output_path(cfg, "sleeve_weights")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(out_path, index=False)
        if "ep_posterior" in cfg.get("outputs", {}):
            posterior_path = _output_path(cfg, "ep_posterior")
            posterior_path.parent.mkdir(parents=True, exist_ok=True)
            posterior.to_parquet(posterior_path, index=False)
        if "ep_signals" in cfg.get("outputs", {}):
            signals_path = _output_path(cfg, "ep_signals")
            signals_path.parent.mkdir(parents=True, exist_ok=True)
            signals.to_csv(signals_path, index=False, encoding="utf-8-sig")
        diagnostics_cols = [c for c in out.columns if c.startswith("opt_")]
        if "optimizer_diagnostics" in cfg.get("outputs", {}) and diagnostics_cols:
            diag_path = _output_path(cfg, "optimizer_diagnostics")
            diag_path.parent.mkdir(parents=True, exist_ok=True)
            out[["date"] + diagnostics_cols].to_csv(diag_path, index=False, encoding="utf-8-sig")
        return out
    out = overlay_base
    out = _apply_rule_adjust_speed(out, rule_cards, cfg)
    out_path = _output_path(cfg, "sleeve_weights")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    return out


def _preferred_assets_for_sleeve(sleeve: str, cfg: dict[str, Any]) -> list[str] | None:
    for info in cfg.get("asset_mapping", {}).values():
        if info.get("module") == sleeve and "preferred_assets" in info:
            return list(info["preferred_assets"])
    return None


def _preferred_asset_weight_map(sleeve: str, cfg: dict[str, Any]) -> dict[str, float]:
    profile_cfg = selected_profile(cfg)
    sleeve_override = profile_cfg.get("asset_overrides", {}).get(sleeve, {})
    raw_override = sleeve_override.get("preferred_asset_weights")
    if raw_override:
        weights = {str(k): float(v) for k, v in dict(raw_override).items()}
        total = sum(max(v, 0.0) for v in weights.values())
        if total > 0:
            return {k: max(v, 0.0) / total for k, v in weights.items()}
    for info in cfg.get("asset_mapping", {}).values():
        if info.get("module") != sleeve:
            continue
        raw = info.get("preferred_asset_weights")
        if not raw:
            return {}
        weights = {str(k): float(v) for k, v in dict(raw).items()}
        total = sum(max(v, 0.0) for v in weights.values())
        if total <= 0:
            return {}
        return {k: max(v, 0.0) / total for k, v in weights.items()}
    return {}


def load_asset_itransformer_predictions(cfg: dict[str, Any], paths: V1Paths | None = None) -> pd.DataFrame:
    paths = paths or V1Paths()
    overlay = cfg.get("signal_overlays", {}).get("asset_itransformer", {})
    if not overlay.get("enabled", False):
        return pd.DataFrame()
    pred_path = ROOT / str(overlay.get("predictions_path", paths.asset_itransformer_predictions.relative_to(ROOT)))
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing asset iTransformer prediction file: {pred_path}")
    pred = pd.read_parquet(pred_path)
    pred["as_of_date"] = pd.to_datetime(pred["as_of_date"], errors="coerce")
    pred["asset_code"] = pred["asset_code"].astype(str)
    pred["module"] = pred.get("module", "").astype(str)
    pred["pred_return"] = pd.to_numeric(pred["pred_return"], errors="coerce")
    pred["pred_prob"] = pd.to_numeric(pred.get("pred_prob"), errors="coerce")
    pred = pred.dropna(subset=["as_of_date", "asset_code", "pred_return"])
    return pred.sort_values(["as_of_date", "module", "pred_return"], ascending=[True, True, False]).reset_index(drop=True)


def _cap_and_normalize_share(values: pd.Series, max_share: float) -> pd.Series:
    share = values.astype(float).clip(lower=0.0)
    if share.sum() <= 0:
        return pd.Series(1.0 / len(share), index=share.index)
    share = share / share.sum()
    max_share = float(max(max_share, 1.0 / len(share)))
    for _ in range(20):
        capped = share.clip(upper=max_share)
        excess = float(share.sum() - capped.sum())
        share = capped
        if excess <= 1e-12:
            break
        room = max_share - share
        active = room > 1e-12
        if not active.any():
            break
        share.loc[active] += excess * room.loc[active] / room.loc[active].sum()
    return share / share.sum()


def _asset_itransformer_shares(
    assets: pd.DataFrame,
    sleeve: str,
    date: pd.Timestamp,
    pred: pd.DataFrame,
    cfg: dict[str, Any],
) -> pd.Series:
    equal = pd.Series(1.0 / len(assets), index=assets["asset_code"].astype(str).to_list())
    overlay = cfg.get("signal_overlays", {}).get("asset_itransformer", {})
    if not overlay.get("enabled", False) or pred.empty:
        return equal
    enabled_modules = set(overlay.get("enabled_modules", RISK_SLEEVES))
    if sleeve not in enabled_modules:
        return equal
    active_codes = equal.index.tolist()
    active = pred[pred["as_of_date"].eq(pd.Timestamp(date)) & pred["asset_code"].isin(active_codes)].copy()
    min_assets = int(overlay.get("min_assets", 2))
    if len(active) < min_assets:
        return equal

    temperature = max(float(overlay.get("temperature", 0.035)), 1e-8)
    blend = float(np.clip(float(overlay.get("blend", 0.50)), 0.0, 1.0))
    score = active.set_index("asset_code")["pred_return"].reindex(active_codes).dropna()
    z = score / temperature
    z = z - z.max()
    signal = np.exp(z)
    signal = pd.Series(signal / signal.sum(), index=score.index)
    signal = signal.reindex(active_codes).fillna(0.0)
    signal = _cap_and_normalize_share(signal, float(overlay.get("max_asset_share", 0.75)))
    share = (1.0 - blend) * equal + blend * signal
    return _cap_and_normalize_share(share, float(overlay.get("max_asset_share", 0.75)))


def _fixed_preferred_asset_shares(assets: pd.DataFrame, sleeve: str, cfg: dict[str, Any]) -> pd.Series:
    codes = assets["asset_code"].astype(str).to_list()
    equal = pd.Series(1.0 / len(codes), index=codes)
    weight_map = _preferred_asset_weight_map(sleeve, cfg)
    if not weight_map:
        return equal
    raw = pd.Series({code: weight_map.get(code, 0.0) for code in codes}, dtype=float)
    if raw.sum() <= 0:
        return equal
    return raw / raw.sum()


def _apply_asset_momentum_overlay(
    shares: pd.Series,
    assets: pd.DataFrame,
    sleeve: str,
    date: pd.Timestamp,
    returns: pd.DataFrame,
    cfg: dict[str, Any],
) -> pd.Series:
    overlay = cfg.get("signal_overlays", {}).get("asset_momentum", {})
    if not overlay.get("enabled", False):
        return shares
    profile = str(cfg.get("product", {}).get("selected_profile", ""))
    enabled_profiles = set(overlay.get("enabled_profiles", []))
    if enabled_profiles and profile not in enabled_profiles:
        return shares
    enabled_modules = set(overlay.get("enabled_modules", []))
    if enabled_modules and sleeve not in enabled_modules:
        return shares

    codes = assets["asset_code"].astype(str).tolist()
    min_assets = int(overlay.get("min_assets", 2))
    if len(codes) < min_assets:
        return shares
    lookbacks = sorted({max(1, int(x)) for x in overlay.get("lookback_months", [6, 12])})
    min_history = max(int(overlay.get("min_history_months", max(lookbacks))), max(lookbacks))
    hist = (
        returns[
            returns["asset_code"].astype(str).isin(codes)
            & returns["date"].le(pd.Timestamp(date))
        ]
        .pivot_table(index="date", columns="asset_code", values="return_1m", aggfunc="last")
        .sort_index()
        .reindex(columns=codes)
        .tail(max(min_history, max(lookbacks)))
    )
    scores: dict[str, float] = {}
    raw_momentum: dict[str, float] = {}
    score_method = str(overlay.get("score_method", "raw_momentum"))
    vol_floor = float(overlay.get("volatility_floor", 0.02))
    for code in codes:
        series = pd.to_numeric(hist[code], errors="coerce").dropna()
        if len(series) < min_history:
            continue
        parts = [float((1.0 + series.tail(lb)).prod() - 1.0) for lb in lookbacks]
        momentum = float(np.mean(parts))
        raw_momentum[code] = momentum
        if score_method == "risk_adjusted_momentum":
            annual_vol = float(series.tail(max(lookbacks)).std(ddof=1) * np.sqrt(12.0))
            scores[code] = momentum / max(annual_vol, vol_floor)
        else:
            scores[code] = momentum
    if len(scores) < min_assets:
        return shares

    base = shares.reindex(codes).fillna(0.0).astype(float)
    if base.sum() <= 0:
        base = pd.Series(1.0 / len(codes), index=codes, dtype=float)
    else:
        base = base / base.sum()
    score = pd.Series(scores, dtype=float)
    rank = score.rank(method="average", pct=True) - 0.5
    concentration = float(overlay.get("rank_concentration", 2.0))
    target = base.copy()
    target.loc[rank.index] = target.loc[rank.index] * np.exp(concentration * rank)
    negative_multiplier = float(np.clip(float(overlay.get("negative_momentum_multiplier", 0.50)), 0.0, 1.0))
    negative_threshold = float(overlay.get("negative_momentum_threshold", 0.0))
    for code, momentum in raw_momentum.items():
        if momentum < negative_threshold:
            target.loc[code] *= negative_multiplier
    if target.sum() <= 0:
        return base
    target = target / target.sum()
    blend_map = overlay.get("profile_blend", {})
    blend = float(np.clip(float(blend_map.get(profile, overlay.get("blend", 0.20))), 0.0, 1.0))
    out = (1.0 - blend) * base + blend * target
    return _cap_and_normalize_share(out, float(overlay.get("max_asset_share", 0.75)))


def _apply_alt_internal_overlay(
    shares: pd.Series,
    assets: pd.DataFrame,
    rule: dict[str, Any],
    cfg: dict[str, Any],
) -> pd.Series:
    overlay = cfg.get("signal_overlays", {}).get("rule_output", {})
    if not overlay.get("enabled", False) or not overlay.get("apply_alt_internal_overlay", False):
        return shares
    state_tags = rule.get("state_tags", {}) if isinstance(rule, dict) else {}
    alt_state = str(state_tags.get("alt_state", ""))
    state_weights = overlay.get("alt_state_weights", {}).get(alt_state)
    if not state_weights:
        return shares
    codes = assets["asset_code"].astype(str).to_list()
    target = pd.Series({code: float(state_weights.get(code, 0.0)) for code in codes}, dtype=float)
    if target.sum() <= 0:
        return shares
    target = target / target.sum()
    strength = float(np.clip(float(overlay.get("alt_overlay_strength", 0.50)), 0.0, 1.0))
    blended = (1.0 - strength) * shares.reindex(codes).fillna(0.0) + strength * target
    if blended.sum() <= 0:
        return shares
    return blended / blended.sum()


def _asset_code_from_view_asset(asset: str, codes: list[str], cfg: dict[str, Any]) -> str | None:
    if asset in codes:
        return asset
    mapping = cfg.get("asset_mapping", {}).get(asset, {})
    preferred = mapping.get("preferred_assets", [])
    if len(preferred) == 1 and preferred[0] in codes:
        return str(preferred[0])
    if asset == "gold" and "alt_gold" in codes:
        return "alt_gold"
    return None


def _apply_asset_view_internal_overlay(
    shares: pd.Series,
    assets: pd.DataFrame,
    sleeve: str,
    date: pd.Timestamp,
    views: pd.DataFrame,
    cfg: dict[str, Any],
) -> pd.Series:
    overlay = cfg.get("signal_overlays", {}).get("agentic_asset_views", {})
    if not overlay.get("enabled", False) or views.empty:
        return shares
    enabled_profiles = set(overlay.get("enabled_profiles", []))
    profile = str(cfg.get("product", {}).get("selected_profile", ""))
    if enabled_profiles and profile not in enabled_profiles:
        return shares
    enabled_modules = set(overlay.get("enabled_modules", RISK_SLEEVES))
    if sleeve not in enabled_modules:
        return shares

    codes = assets["asset_code"].astype(str).to_list()
    if len(codes) < 2:
        return shares
    out = shares.reindex(codes).fillna(0.0).astype(float)
    if out.sum() <= 0:
        out = pd.Series(1.0 / len(codes), index=codes)
    else:
        out = out / out.sum()

    active = views[views["month"].eq(pd.Timestamp(date))].copy()
    if active.empty:
        return out

    base_tilt = float(overlay.get("tilt_per_confidence", 0.10))
    max_pair_transfer = float(overlay.get("max_pair_transfer", 0.10))
    max_asset_share = float(overlay.get("max_asset_share", 0.75))
    min_confidence = float(overlay.get("min_confidence", 0.50))
    for _, view in active.iterrows():
        source = _asset_code_from_view_asset(str(view.get("source_asset", "")), codes, cfg)
        target = _asset_code_from_view_asset(str(view.get("target_asset", "")), codes, cfg)
        if source is None or target is None or source == target:
            continue
        confidence = float(view.get("confidence", 0.0))
        if confidence < min_confidence:
            continue
        amount = min(max_pair_transfer, base_tilt * confidence)
        transfer = min(amount, float(out.loc[target]))
        out.loc[source] += transfer
        out.loc[target] -= transfer
        out = _cap_and_normalize_share(out, max_asset_share)
    return out / out.sum()


def _gmv_share_from_cov(cov: pd.DataFrame, max_share: float) -> pd.Series:
    codes = cov.index.astype(str).tolist()
    arr = cov.reindex(index=codes, columns=codes).fillna(0.0).to_numpy(dtype=float)
    arr = (arr + arr.T) / 2.0
    arr += np.eye(len(codes)) * max(float(np.trace(arr)) / max(len(codes), 1) * 1e-6, 1e-10)
    ones = np.ones(len(codes))
    try:
        raw = np.linalg.solve(arr, ones)
    except np.linalg.LinAlgError:
        raw = np.linalg.pinv(arr) @ ones
    if not np.all(np.isfinite(raw)) or raw.sum() <= 0 or (raw < 0).any():
        vol = np.sqrt(np.maximum(np.diag(arr), 1e-10))
        raw = 1.0 / vol
    share = pd.Series(np.maximum(raw, 0.0), index=codes, dtype=float)
    if share.sum() <= 0:
        share = pd.Series(1.0 / len(codes), index=codes, dtype=float)
    else:
        share = share / share.sum()
    return _cap_and_normalize_share(share, max_share)


def _apply_asset_covariance_overlay(
    shares: pd.Series,
    assets: pd.DataFrame,
    sleeve: str,
    date: pd.Timestamp,
    returns: pd.DataFrame,
    cfg: dict[str, Any],
) -> pd.Series:
    overlay = cfg.get("signal_overlays", {}).get("asset_covariance_cleaner", {})
    if not overlay.get("enabled", False):
        return shares
    profile = str(cfg.get("product", {}).get("selected_profile", ""))
    enabled_profiles = set(overlay.get("enabled_profiles", []))
    if enabled_profiles and profile not in enabled_profiles:
        return shares
    enabled_modules = set(overlay.get("enabled_modules", RISK_SLEEVES))
    if sleeve not in enabled_modules:
        return shares

    codes = assets["asset_code"].astype(str).tolist()
    min_assets = int(overlay.get("min_assets", 2))
    if len(codes) < min_assets:
        return shares
    lookback = int(overlay.get("lookback_months", 36))
    min_history = int(overlay.get("min_history_months", 12))
    hist = (
        returns[
            returns["asset_code"].astype(str).isin(codes)
            & returns["date"].le(pd.Timestamp(date))
        ]
        .pivot_table(index="date", columns="asset_code", values="return_1m", aggfunc="last")
        .sort_index()
        .tail(lookback)
    )
    available_cols = [code for code in codes if code in hist.columns and hist[code].notna().sum() >= min_history]
    if len(available_cols) < min_assets:
        return shares
    hist = hist.reindex(columns=available_cols).fillna(0.0)

    cov_cfg = {
        "covariance_estimator": overlay.get("covariance_estimator", {"method": "rienet_inspired"}),
    }
    cov_est = estimate_covariance(
        hist,
        available_cols,
        cov_cfg,
        default_shrinkage=float(overlay.get("shrinkage_to_diag", 0.20)),
        floor=float(overlay.get("diagonal_vol_floor", 0.002)) ** 2,
        jitter=float(overlay.get("cov_jitter", 1e-8)),
    )
    model_share = _gmv_share_from_cov(cov_est.covariance, float(overlay.get("max_asset_share", 0.75)))
    base = shares.reindex(codes).fillna(0.0).astype(float)
    if base.sum() <= 0:
        base = pd.Series(1.0 / len(codes), index=codes, dtype=float)
    else:
        base = base / base.sum()
    model_full = pd.Series(0.0, index=codes, dtype=float)
    model_full.loc[model_share.index] = model_share
    if model_full.sum() <= 0:
        return base
    model_full = model_full / model_full.sum()
    blend = float(np.clip(float(overlay.get("blend", 0.35)), 0.0, 1.0))
    out = (1.0 - blend) * base + blend * model_full
    return _cap_and_normalize_share(out, float(overlay.get("max_asset_share", 0.75)))


def _available_assets_for_sleeve(available: pd.DataFrame, sleeve: str, cfg: dict[str, Any]) -> pd.DataFrame:
    assets = available[available["module"].eq(sleeve)].drop_duplicates("asset_code").copy()
    preferred = _preferred_assets_for_sleeve(sleeve, cfg)
    if preferred:
        preferred_assets = assets[assets["asset_code"].isin(preferred)]
        if not preferred_assets.empty:
            assets = preferred_assets
    return assets


def _redistribute_unavailable_sleeves(
    weights: dict[str, float],
    available_by_sleeve: dict[str, pd.DataFrame],
) -> dict[str, float]:
    adjusted = weights.copy()
    unavailable = [
        sleeve
        for sleeve in RISK_SLEEVES
        if adjusted.get(sleeve, 0.0) > 0 and available_by_sleeve.get(sleeve, pd.DataFrame()).empty
    ]
    if not unavailable:
        return adjusted
    blocked_weight = sum(adjusted.get(sleeve, 0.0) for sleeve in unavailable)
    for sleeve in unavailable:
        adjusted[sleeve] = 0.0
    receivers = [
        sleeve
        for sleeve in RISK_SLEEVES
        if not available_by_sleeve.get(sleeve, pd.DataFrame()).empty and adjusted.get(sleeve, 0.0) > 0
    ]
    if not receivers:
        receivers = [sleeve for sleeve in RISK_SLEEVES if not available_by_sleeve.get(sleeve, pd.DataFrame()).empty]
    if not receivers:
        raise ValueError("No available non-cash assets for this rebalance date; cash is disabled.")
    base = pd.Series({sleeve: max(adjusted.get(sleeve, 0.0), 0.0) for sleeve in receivers}, dtype=float)
    if base.sum() <= 0:
        base = pd.Series(1.0 / len(receivers), index=receivers)
    else:
        base = base / base.sum()
    for sleeve, share in base.items():
        adjusted[sleeve] += blocked_weight * float(share)
    adjusted["cash"] = 0.0
    return adjusted


def build_asset_weights(cfg: dict[str, Any] | None = None, paths: V1Paths | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    cfg = apply_selected_profile_overrides(cfg)
    paths = paths or V1Paths()
    returns = load_monthly_returns(paths, cfg)
    sleeves = build_sleeve_weights(cfg, paths)
    bounds = sleeve_bounds(cfg)
    asset_pred = load_asset_itransformer_predictions(cfg, paths)
    views = month_end_views(paths, cfg)
    rule_map = _rule_by_month(month_end_rule_cards(cfg, paths))
    rows: list[dict[str, Any]] = []
    for _, sleeve_row in sleeves.iterrows():
        date = sleeve_row["date"]
        rule = _rule_output_for_row(rule_map.get(pd.Timestamp(date)))
        available = returns[returns["date"].eq(date)].copy()
        available_by_sleeve = {sleeve: _available_assets_for_sleeve(available, sleeve, cfg) for sleeve in RISK_SLEEVES}
        sleeve_weights = {sleeve: float(sleeve_row[sleeve]) for sleeve in RISK_SLEEVES + ["cash"]}
        sleeve_weights = _redistribute_unavailable_sleeves(sleeve_weights, available_by_sleeve)
        for sleeve in RISK_SLEEVES + ["cash"]:
            sleeve_weight = float(sleeve_weights.get(sleeve, 0.0))
            if sleeve == "cash":
                rows.append({"date": date, "asset_code": "cash", "asset_name": "现金", "module": "cash", "weight": sleeve_weight})
                continue
            assets = available_by_sleeve[sleeve]
            if assets.empty or sleeve_weight <= 0:
                if sleeve != "cash" and sleeve_weight > 0 and bounds.get("cash", (0.0, 0.0))[1] <= 0:
                    raise ValueError(f"No available assets for sleeve={sleeve} date={date:%Y-%m-%d}; cash is disabled.")
                rows.append({"date": date, "asset_code": "cash", "asset_name": "现金", "module": "cash", "weight": sleeve_weight})
                continue
            shares = _fixed_preferred_asset_shares(assets, sleeve, cfg)
            model_shares = _asset_itransformer_shares(assets, sleeve, pd.Timestamp(date), asset_pred, cfg)
            if not model_shares.equals(pd.Series(1.0 / len(assets), index=assets["asset_code"].astype(str).to_list())):
                shares = model_shares
            shares = _apply_asset_momentum_overlay(shares, assets, sleeve, pd.Timestamp(date), returns, cfg)
            if sleeve == "alternatives":
                shares = _apply_alt_internal_overlay(shares, assets, rule, cfg)
            shares = _apply_asset_view_internal_overlay(shares, assets, sleeve, pd.Timestamp(date), views, cfg)
            shares = _apply_asset_covariance_overlay(shares, assets, sleeve, pd.Timestamp(date), returns, cfg)
            for _, asset in assets.iterrows():
                code = str(asset["asset_code"])
                rows.append({"date": date, "asset_code": code, "asset_name": asset["asset_name"], "module": sleeve, "weight": sleeve_weight * float(shares.get(code, 0.0))})
    out = pd.DataFrame(rows).groupby(["date", "asset_code", "asset_name", "module"], as_index=False)["weight"].sum()
    out_path = _output_path(cfg, "weights")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    return out


def run_backtest(cfg: dict[str, Any] | None = None, paths: V1Paths | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = cfg or load_config()
    cfg = apply_selected_profile_overrides(cfg)
    paths = paths or V1Paths()
    returns = load_monthly_returns(paths, cfg)
    weights = build_asset_weights(cfg, paths)
    ret = returns.pivot_table(index="date", columns="asset_code", values="return_1m", aggfunc="last").sort_index()
    ret["cash"] = 0.0
    target_w = weights.pivot_table(index="date", columns="asset_code", values="weight", aggfunc="sum").sort_index().fillna(0.0)
    cols = sorted(set(ret.columns).union(target_w.columns))
    ret = ret.reindex(columns=cols)
    target_w = target_w.reindex(columns=cols).fillna(0.0)
    applied_w = target_w.shift(int(cfg["rebalance"]["signal_lag_periods"])).dropna(how="all")
    dates = ret.index.intersection(applied_w.index)
    applied_w = applied_w.loc[dates]
    realized_raw = ret.loc[dates]
    missing_return_weight = applied_w.where(realized_raw.isna(), 0.0).abs().sum(axis=1)
    realized = realized_raw.fillna(0.0)
    gross = (applied_w * realized).sum(axis=1)
    turnover = (applied_w - applied_w.shift(1).fillna(0.0)).abs().sum(axis=1)
    cost = turnover * float(cfg["costs"]["transaction_cost_bps"]) / 10000.0
    net = gross - cost
    nav = (1.0 + net).cumprod()
    out = pd.DataFrame(
        {
            "date": dates,
            "gross_return": gross.values,
            "turnover": turnover.values,
            "transaction_cost": cost.values,
            "missing_return_weight": missing_return_weight.values,
            "net_return": net.values,
            "nav": nav.values,
        }
    )
    metrics = calc_metrics(out)
    metrics = pd.concat(
        [
            metrics,
            pd.DataFrame(
                [
                    {"metric": "max_missing_return_weight", "value": float(missing_return_weight.max())},
                    {"metric": "months_with_missing_return_exposure", "value": float((missing_return_weight > 1e-12).sum())},
                ]
            ),
        ],
        ignore_index=True,
    )
    _output_path(cfg, "nav").parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(_output_path(cfg, "nav"), index=False, encoding="utf-8-sig")
    metrics.to_csv(_output_path(cfg, "metrics"), index=False, encoding="utf-8-sig")
    return out, metrics


def calc_metrics(nav: pd.DataFrame) -> pd.DataFrame:
    if nav.empty:
        return pd.DataFrame()
    r = nav["net_return"].astype(float)
    n = len(r)
    total_return = float(nav["nav"].iloc[-1] - 1.0)
    annual_return = float(nav["nav"].iloc[-1] ** (12 / n) - 1.0)
    annual_vol = float(r.std(ddof=0) * np.sqrt(12))
    sharpe = annual_return / annual_vol if annual_vol > 0 else np.nan
    drawdown = nav["nav"] / nav["nav"].cummax() - 1.0
    max_drawdown = float(drawdown.min())
    calmar = annual_return / abs(max_drawdown) if max_drawdown < 0 else np.nan
    return pd.DataFrame(
        [
            {"metric": "months", "value": n},
            {"metric": "total_return", "value": total_return},
            {"metric": "annual_return", "value": annual_return},
            {"metric": "annual_vol", "value": annual_vol},
            {"metric": "sharpe", "value": sharpe},
            {"metric": "max_drawdown", "value": max_drawdown},
            {"metric": "calmar", "value": calmar},
            {"metric": "avg_turnover", "value": float(nav["turnover"].mean())},
        ]
    )


def run_attribution(cfg: dict[str, Any] | None = None, paths: V1Paths | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    cfg = apply_selected_profile_overrides(cfg)
    paths = paths or V1Paths()
    returns = load_monthly_returns(paths, cfg)
    weights = build_asset_weights(cfg, paths)
    merged = weights.merge(returns[["date", "asset_code", "return_1m"]], on=["date", "asset_code"], how="left")
    merged["return_1m"] = merged["return_1m"].fillna(0.0)
    merged = merged.sort_values(["asset_code", "date"])
    merged["applied_weight"] = merged.groupby("asset_code")["weight"].shift(int(cfg["rebalance"]["signal_lag_periods"]))
    merged["contribution"] = merged["applied_weight"].fillna(0.0) * merged["return_1m"]
    out = merged.groupby("module", as_index=False).agg(total_contribution=("contribution", "sum"), avg_applied_weight=("applied_weight", "mean"))
    out = out.sort_values("total_contribution", ascending=False)
    _output_path(cfg, "attribution").parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(_output_path(cfg, "attribution"), index=False, encoding="utf-8-sig")
    return out


def run_robustness(cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    cfg = apply_selected_profile_overrides(cfg)
    nav_path = _output_path(cfg, "nav")
    if not nav_path.exists():
        out = pd.DataFrame([{"check": "nav_exists", "status": "missing", "detail": str(nav_path)}])
    else:
        nav = pd.read_csv(nav_path)
        nav["date"] = pd.to_datetime(nav["date"], errors="coerce")
        rows = []
        for start in ["2019-01-31", "2020-01-31", "2021-01-31", "2022-01-31"]:
            sub = nav[nav["date"].ge(pd.Timestamp(start))].copy()
            if sub.empty:
                continue
            metrics = calc_metrics(sub)
            rows.append(
                {
                    "check": f"start_{start}",
                    "status": "ok",
                    "annual_return": metrics.loc[metrics["metric"].eq("annual_return"), "value"].iloc[0],
                    "max_drawdown": metrics.loc[metrics["metric"].eq("max_drawdown"), "value"].iloc[0],
                }
            )
        out = pd.DataFrame(rows)
    _output_path(cfg, "robustness").parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(_output_path(cfg, "robustness"), index=False, encoding="utf-8-sig")
    return out

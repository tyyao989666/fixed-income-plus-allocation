from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.itransformer_signal import RISK_SLEEVES
from src.point_in_time import causal_rolling_percentile


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "daily_timing_card_config.yaml"


def load_timing_card_config(path: Path | None = None) -> dict[str, Any]:
    with (path or DEFAULT_CONFIG).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def out_path(cfg: dict[str, Any], key: str) -> Path:
    return ROOT / cfg["outputs"][key]


def _clip01(x: pd.Series | np.ndarray | float) -> pd.Series | np.ndarray | float:
    return np.clip(x, 0.0, 1.0)


def _safe_num(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)


def _rolling_pct(s: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    return causal_rolling_percentile(s, window, min_periods or max(20, window // 4))


def _history_weight(length: int, window: int) -> pd.Series:
    return pd.Series(np.minimum(np.arange(1, length + 1) / max(window, 1), 1.0), dtype=float)


def _confidence_blend(value: pd.Series, confidence: pd.Series, neutral: float = 0.5) -> pd.Series:
    return neutral + confidence * (value.fillna(neutral) - neutral)


def load_base_inputs(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    market = pd.read_parquet(ROOT / cfg["data"]["market_state_path"])
    features = pd.read_parquet(ROOT / cfg["data"]["feature_panel_path"])
    returns = pd.read_parquet(ROOT / cfg["data"]["asset_returns_path"])
    for df in [market, features, returns]:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    start = pd.Timestamp(cfg["data"]["start_date"])
    end = pd.Timestamp(cfg["data"]["end_date"])
    market = market[market["date"].between(start, end)].sort_values("date").drop_duplicates("date")
    features = features[features["date"].between(start, end)].sort_values("date").drop_duplicates("date")
    returns = returns[returns["date"].between(start, end)].sort_values(["date", "asset_code"])
    return market, features, returns


def build_return_features(returns: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    ret = returns.copy()
    ret["return_1d"] = pd.to_numeric(ret["return_1d"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    sleeve = ret.groupby(["date", "module"], as_index=False)["return_1d"].mean()
    wide = sleeve.pivot(index="date", columns="module", values="return_1d").sort_index()
    windows = cfg["windows"]
    out = pd.DataFrame(index=wide.index)
    for module in RISK_SLEEVES:
        s = wide[module].astype(float) if module in wide.columns else pd.Series(np.nan, index=wide.index)
        out[f"{module}_ret_1d"] = s
        for label, w in [
            ("5d", int(windows["short"])),
            ("20d", int(windows["medium"])),
            ("60d", int(windows["long"])),
            ("120d", int(windows["path"])),
        ]:
            out[f"{module}_mom_{label}"] = s.rolling(w, min_periods=max(3, w // 4)).sum()
            out[f"{module}_vol_{label}"] = s.rolling(w, min_periods=max(3, w // 4)).std() * np.sqrt(252)

    asset_codes = ["alt_gold", "alt_commodity", "alt_nonferrous", "alt_chemical", "alt_us_bond", "alt_reits"]
    asset = ret[ret["asset_code"].astype(str).isin(asset_codes)]
    if not asset.empty:
        awide = asset.pivot_table(index="date", columns="asset_code", values="return_1d", aggfunc="mean").sort_index()
        for code in asset_codes:
            s = awide[code].reindex(wide.index).astype(float) if code in awide.columns else pd.Series(np.nan, index=wide.index)
            for label, w in [("20d", int(windows["medium"])), ("60d", int(windows["long"])), ("120d", int(windows["path"]))]:
                out[f"{code}_mom_{label}"] = s.rolling(w, min_periods=max(5, w // 4)).sum()
                out[f"{code}_vol_{label}"] = s.rolling(w, min_periods=max(5, w // 4)).std() * np.sqrt(252)
    return out.reset_index().rename(columns={"index": "date"})


def current_weights_for_profile(profile: str, dates: pd.Series, cfg: dict[str, Any]) -> pd.DataFrame:
    path = ROOT / cfg["data"]["weights_pattern"].format(profile=profile)
    frame = pd.DataFrame({"date": pd.to_datetime(dates).astype("datetime64[ns]")})
    if not path.exists():
        for sleeve in RISK_SLEEVES:
            frame[f"current_weight_{sleeve}"] = 1.0 / len(RISK_SLEEVES)
        return frame
    weights = pd.read_parquet(path)
    weights["date"] = pd.to_datetime(weights["date"], errors="coerce").astype("datetime64[ns]")
    cols = ["date"] + [s for s in RISK_SLEEVES if s in weights.columns]
    weights = weights[cols].dropna(subset=["date"]).sort_values("date")
    weights = weights.rename(columns={s: f"current_weight_{s}" for s in RISK_SLEEVES})
    merged = pd.merge_asof(frame.sort_values("date"), weights.sort_values("date"), on="date", direction="backward")
    for sleeve in RISK_SLEEVES:
        col = f"current_weight_{sleeve}"
        if col not in merged.columns:
            merged[col] = np.nan
    wcols = [f"current_weight_{s}" for s in RISK_SLEEVES]
    merged[wcols] = merged[wcols].ffill().fillna(1.0 / len(RISK_SLEEVES))
    total = merged[wcols].sum(axis=1).replace(0.0, np.nan)
    merged[wcols] = merged[wcols].div(total, axis=0).fillna(1.0 / len(RISK_SLEEVES))
    return merged


def build_base_daily_scores(cfg: dict[str, Any]) -> pd.DataFrame:
    market, features, returns = load_base_inputs(cfg)
    ret_feat = build_return_features(returns, cfg)
    df = market.merge(features, on="date", how="left", suffixes=("", "_feature")).merge(ret_feat, on="date", how="left")
    df = df.sort_values("date").reset_index(drop=True)

    windows = cfg["windows"]
    annual = int(windows["annual"])
    regime = int(windows["regime"])
    structural = int(windows["structural"])
    min_cfg = windows.get("percentile_min_periods", {})
    df["history_confidence_3y"] = _history_weight(len(df), regime).to_numpy()
    df["history_confidence_5y"] = _history_weight(len(df), structural).to_numpy()

    anchor_specs = {
        "rate_10y": "rate_treasury_y10",
        "credit_spread": "credit_spread_aaa_3y_proxy",
        "cb_price": "cb_cb_price_median",
        "equity_pe": "equity_csi1000_pe",
        "real_rate": "alt_real_rate_proxy",
        "gold_price": "alt_gold_au9999",
        "commodity_price": "alt_nanhua_commodity",
    }
    for name, column in anchor_specs.items():
        values = _safe_num(df, column, np.nan)
        df[f"{name}_pct_1y"] = _rolling_pct(values, annual, int(min_cfg.get("annual", 60)))
        df[f"{name}_pct_3y"] = _rolling_pct(values, regime, int(min_cfg.get("regime", 126)))
        df[f"{name}_pct_5y"] = _rolling_pct(values, structural, int(min_cfg.get("structural", 252)))

    rate_anchor = _confidence_blend(df["rate_10y_pct_5y"], df["history_confidence_5y"])
    credit_anchor = _confidence_blend(df["credit_spread_pct_5y"], df["history_confidence_5y"])
    cb_anchor = _confidence_blend(df["cb_price_pct_5y"], df["history_confidence_5y"])
    equity_value_anchor = _confidence_blend(1.0 - df["equity_pe_pct_5y"], df["history_confidence_5y"])
    real_rate_anchor = _confidence_blend(df["real_rate_pct_5y"], df["history_confidence_5y"])

    credit_pressure = (
        0.25 * _safe_num(df, "spread_widening_pressure")
        + 0.25 * _safe_num(df, "credit_turnover_stress")
        + 0.25 * _safe_num(df, "funding_stress_score")
        + 0.20 * _safe_num(df, "credit_tail_risk_score")
        + 0.05 * _safe_num(df, "regional_default_pressure")
    )
    cr_state_penalty = df.get("cr_state", pd.Series("", index=df.index)).astype(str).map({"CR3": 0.12, "CR4": 0.22}).fillna(0.0)
    cr2_overlay = df.get("CR2_caution_overlay", pd.Series(False, index=df.index)).astype(bool).astype(float) * 0.12
    thin_spread_caution = 1.0 - credit_anchor
    df["credit_caution_score"] = _clip01(0.90 * credit_pressure + 0.10 * thin_spread_caution + cr_state_penalty + cr2_overlay)

    rate_change_20d = _safe_num(df, "rate_10y_change_20d", np.nan)
    rate_bull_raw = _clip01(0.50 - rate_change_20d * 3.0)
    rate_state_bonus = df.get("r_state", pd.Series("", index=df.index)).astype(str).map({"R1": 0.20, "R2": 0.05, "R4": -0.15}).fillna(0.0)
    df["rate_bull_score"] = _clip01(
        rate_bull_raw.fillna(0.5)
        + 0.12 * (rate_anchor - 0.5)
        + rate_state_bonus
        + (_safe_num(df, "bond_score", 0.5) - 0.5) * 0.5
    )

    cb_price = _safe_num(df, "cb_price_median", 120.0)
    cb_low_price_ratio = _safe_num(df, "cb_low_price_ratio_110", 0.0)
    cb_state_bonus = df.get("cb_state", pd.Series("", index=df.index)).astype(str).map({"CB1": 0.18, "CB2": 0.06, "CB4": -0.20}).fillna(0.0)
    df["cb_opportunity_score"] = _clip01(
        0.75 * ((125.0 - cb_price) / 35.0)
        + 0.25 * (1.0 - cb_anchor)
        + 0.35 * cb_low_price_ratio
        + cb_state_bonus
    )

    breadth = _safe_num(df, "equity_breadth_5d", _safe_num(df, "equity_sw_l1_positive_5d_ratio", 0.5))
    eq_mom = _safe_num(df, "equity_mom_20d", 0.0)
    eq_mom_60 = _safe_num(df, "equity_mom_60d", 0.0)
    eq_mom_120 = _safe_num(df, "equity_mom_120d", 0.0)
    pe = _safe_num(df, "equity_csi1000_pe", np.nan)
    pe_pct = _rolling_pct(pe, int(cfg["windows"]["percentile_window"]))
    eq_state_bonus = df.get("eq_state", pd.Series("", index=df.index)).astype(str).map({"EQ1": 0.18, "EQ2": 0.06, "EQ4": -0.18}).fillna(0.0)
    df["equity_trend_score"] = _clip01(
        0.20
        + 0.40 * breadth
        + 1.8 * eq_mom
        + 0.8 * eq_mom_60
        + 0.35 * eq_mom_120
        + 0.12 * equity_value_anchor
        - 0.12 * pe_pct.fillna(0.5)
        + eq_state_bonus
    )

    gold_mom = _safe_num(df, "alt_gold_mom_20d", _safe_num(df, "alt_gold_au9999_ret_1d", 0.0) * 20)
    gold_mom_60 = _safe_num(df, "alt_gold_mom_60d", 0.0)
    gold_mom_120 = _safe_num(df, "alt_gold_mom_120d", 0.0)
    real_rate = _safe_num(df, "alt_real_rate_proxy", np.nan)
    real_rate_down = -real_rate.diff(20).fillna(0.0)
    alt_state_bonus = df.get("alt_state", pd.Series("", index=df.index)).astype(str).map({"ALT1": 0.16, "ALT3": 0.10, "ALT4": 0.08}).fillna(0.0)
    df["alt_hedge_score"] = _clip01(
        0.38
        + 1.8 * gold_mom.fillna(0.0)
        + 0.7 * gold_mom_60
        + 0.25 * gold_mom_120
        + 0.10 * real_rate_down
        + 0.08 * (1.0 - real_rate_anchor)
        + alt_state_bonus
    )

    commodity_mom = _safe_num(df, "alt_commodity_mom_20d", _safe_num(df, "alt_nanhua_commodity_ret_1d", 0.0) * 20)
    commodity_mom_60 = _safe_num(df, "alt_commodity_mom_60d", 0.0)
    chemical_mom = _safe_num(df, "alt_chemical_mom_20d", 0.0)
    nonferrous_mom = _safe_num(df, "alt_nonferrous_mom_20d", 0.0)
    commodity_anchor = _confidence_blend(df["commodity_price_pct_5y"], df["history_confidence_5y"])
    df["commodity_inflation_score"] = _clip01(
        0.38 + 1.3 * commodity_mom + 0.6 * commodity_mom_60 + 0.9 * chemical_mom + 0.9 * nonferrous_mom + 0.10 * commodity_anchor
    )

    liquidity_shock = df.get("liquidity_shock", pd.Series(False, index=df.index)).astype(bool).astype(float)
    df["risk_off_score"] = _clip01(
        0.15
        + 0.35 * df["credit_caution_score"]
        + 0.20 * (1.0 - df["equity_trend_score"])
        + 0.15 * liquidity_shock
        + 0.10 * df["alt_hedge_score"]
    )
    df["risk_on_score"] = _clip01(
        0.20
        + 0.30 * df["equity_trend_score"]
        + 0.20 * df["cb_opportunity_score"]
        + 0.15 * (1.0 - df["credit_caution_score"])
        + 0.10 * df["commodity_inflation_score"]
        + 0.05 * _safe_num(df, "risk_budget_score", 0.0)
    )
    return df


def classify_gate(row: pd.Series, profile: str, cfg: dict[str, Any]) -> tuple[str, str, bool, float]:
    th = cfg["thresholds"]
    scale = float(cfg["profile_gate"][profile]["risk_on_scale"])
    max_action = float(cfg["profile_gate"][profile]["max_action_hint"])
    risk_on = float(row["risk_on_score"]) * scale
    risk_off = float(row["risk_off_score"])
    credit_caution = float(row["credit_caution_score"])
    strong_module = max(
        float(row["rate_bull_score"]),
        float(row["cb_opportunity_score"]),
        float(row["equity_trend_score"]) * scale,
        float(row["alt_hedge_score"]),
        float(row["commodity_inflation_score"]) * scale,
    )
    if bool(row.get("liquidity_shock", False)) or credit_caution >= float(th["credit_caution"]) or risk_off >= float(th["defensive_tilt"]):
        return "allow_defensive_tilt", "risk_off_or_credit_caution", True, max_action
    if risk_on >= float(th["risk_on_allow"]) and risk_off < float(th["risk_off_block"]):
        gate = "allow_risk_on_tilt" if profile != "low_vol" else "allow_small_tilt"
        return gate, "risk_on_confirmed_by_daily_card", True, max_action
    if strong_module >= float(th["strong_module_signal"]) and risk_off < float(th["risk_off_block"]):
        return "allow_small_tilt", "single_module_signal_without_risk_off", True, max_action * 0.5
    return "no_trade", "signal_not_strong_enough_after_cost_gate", False, 0.0


def build_daily_timing_cards(cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    cfg = cfg or load_timing_card_config()
    base = build_base_daily_scores(cfg)
    rows = []
    for profile in cfg["data"]["profiles"]:
        weights = current_weights_for_profile(profile, base["date"], cfg)
        prof = base.merge(weights, on="date", how="left").copy()
        gates = prof.apply(lambda r: classify_gate(r, profile, cfg), axis=1)
        prof["profile"] = profile
        prof["action_gate"] = [g[0] for g in gates]
        prof["gate_reason"] = [g[1] for g in gates]
        prof["allow_model_timing"] = [g[2] for g in gates]
        prof["max_action_hint"] = [g[3] for g in gates]
        prof["risk_regime"] = np.select(
            [
                prof["risk_off_score"].ge(cfg["thresholds"]["risk_off_block"]),
                prof["risk_on_score"].mul(float(cfg["profile_gate"][profile]["risk_on_scale"])).ge(cfg["thresholds"]["risk_on_allow"]),
            ],
            ["risk_off", "risk_on"],
            default="neutral",
        )
        prof["timing_signal_strength"] = prof[
            [
                "risk_on_score",
                "risk_off_score",
                "rate_bull_score",
                "credit_caution_score",
                "cb_opportunity_score",
                "equity_trend_score",
                "alt_hedge_score",
                "commodity_inflation_score",
            ]
        ].max(axis=1)
        keep_cols = [
            "date",
            "profile",
            "risk_regime",
            "action_gate",
            "gate_reason",
            "allow_model_timing",
            "max_action_hint",
            "risk_on_score",
            "risk_off_score",
            "rate_bull_score",
            "credit_caution_score",
            "cb_opportunity_score",
            "equity_trend_score",
            "alt_hedge_score",
            "commodity_inflation_score",
            "timing_signal_strength",
            "history_confidence_3y",
            "history_confidence_5y",
            "rate_10y_pct_1y",
            "rate_10y_pct_3y",
            "rate_10y_pct_5y",
            "credit_spread_pct_1y",
            "credit_spread_pct_3y",
            "credit_spread_pct_5y",
            "cb_price_pct_1y",
            "cb_price_pct_3y",
            "cb_price_pct_5y",
            "equity_pe_pct_1y",
            "equity_pe_pct_3y",
            "equity_pe_pct_5y",
            "real_rate_pct_1y",
            "real_rate_pct_3y",
            "real_rate_pct_5y",
            "gold_price_pct_1y",
            "gold_price_pct_3y",
            "gold_price_pct_5y",
            "commodity_price_pct_1y",
            "commodity_price_pct_3y",
            "commodity_price_pct_5y",
            "cr_state",
            "r_state",
            "cb_state",
            "eq_state",
            "alt_state",
            "rb_state",
            "stock_bond_regime",
            "stock_bond_policy_hint",
            "view_filters",
            "eligible_universe",
            "current_weight_rates",
            "current_weight_credit",
            "current_weight_convertible_bonds",
            "current_weight_equity",
            "current_weight_alternatives",
        ]
        rows.append(prof[[c for c in keep_cols if c in prof.columns]])
    cards = pd.concat(rows, ignore_index=True).sort_values(["date", "profile"]).reset_index(drop=True)
    for key in ["cards"]:
        path = out_path(cfg, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        cards.to_parquet(path, index=False)
    latest = cards[cards["date"].eq(cards["date"].max())].copy()
    latest_path = out_path(cfg, "latest_csv")
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    latest.to_csv(latest_path, index=False, encoding="utf-8-sig")
    summary = (
        cards.groupby(["profile", "risk_regime", "action_gate"], dropna=False)
        .size()
        .reset_index(name="days")
        .sort_values(["profile", "days"], ascending=[True, False])
    )
    summary_path = out_path(cfg, "summary_csv")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    notes = {
        "version": cfg["version"],
        "rows": int(len(cards)),
        "dates": int(cards["date"].nunique()),
        "profiles": cfg["data"]["profiles"],
        "start": str(cards["date"].min().date()) if not cards.empty else "",
        "end": str(cards["date"].max().date()) if not cards.empty else "",
        "no_future_data_rule": "Daily timing cards use only same-day/as-of market state, feature panel, realized trailing returns, and latest prior monthly v1 weights. No future returns are used.",
    }
    notes_path = out_path(cfg, "notes_md")
    notes_path.write_text(
        "# Daily Timing State Cards\n\n"
        + "\n".join([f"- {k}: {v}" for k, v in notes.items()])
        + "\n\nThe card is a timing permission layer, not a final portfolio-weight generator.\n",
        encoding="utf-8",
    )
    return cards

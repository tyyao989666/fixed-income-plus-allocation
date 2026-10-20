from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.adaptive_daily_ppo import ActorCritic, perf_stats
from src.event_driven_rebalance_ppo import EventDrivenEnvironment, load_config as load_event_config
from src.hierarchical_asset_itransformer import ROOT, load_config as load_model_config, project_path
from src.integrated_daily_view_ppo import MODULES, build_state_frame, load_config as load_integrated_config
from src.itransformer_daily_signal import load_daily_config, load_daily_returns_with_proxy
from src.v1_strategy import apply_selected_profile_overrides, load_config as load_v1_config
from src.portfolio_accounting import ACCOUNTING_VERSION, advance_holdings, calendar_decision_targets, accounting_return_panel


PROFILES = ["low_vol", "balanced", "high_vol"]


def _output_path(cfg: dict[str, Any], key: str, profile: str | None = None) -> Path:
    value = str(cfg["outputs"][key])
    if profile is not None:
        value = value.format(profile=profile)
    path = ROOT / value
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _strict_monthly_anchor(
    profile: str, calendar: pd.DatetimeIndex, cfg: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, str]]:
    anchor_path = cfg.get("portfolio", {}).get("formal_anchor_pattern", "data_processed/v1_{profile}_target_weights_monthly.parquet")
    source = pd.read_parquet(ROOT / anchor_path.format(profile=profile))
    source["date"] = pd.to_datetime(source["date"], errors="coerce")
    source = source.loc[source["asset_code"].ne("cash")].copy()
    source["weight"] = pd.to_numeric(source["weight"], errors="coerce").fillna(0.0)
    module_map = source.drop_duplicates("asset_code", keep="last").set_index("asset_code")["module"].astype(str).to_dict()
    wide = source.pivot_table(index="date", columns="asset_code", values="weight", aggfunc="last").fillna(0.0)
    # The technology token remains available to the forecasting model, but it is
    # deliberately excluded from trading because its formal anchor is always zero.
    investable = [column for column in wide if float(wide[column].abs().max()) > 1e-12]
    wide = wide.reindex(columns=investable, fill_value=0.0).sort_index()
    merged = calendar_decision_targets(wide, calendar)
    merged = merged.ffill().dropna(how="all").reindex(calendar).ffill().fillna(0.0)
    rate_split = cfg.get("portfolio", {}).get("formal_anchor_rate_split", {}).get(profile)
    if rate_split:
        split = pd.Series(rate_split, dtype=float)
        expected = {"rate_1_3y", "rate_3_5y", "rate_7_10y", "rate_30y_curve_proxy"}
        if set(split.index) != expected or not np.isclose(float(split.sum()), 1.0):
            raise ValueError("formal_anchor_rate_split must contain the four rate curve proxies and sum to one")
        current_rate_assets = [asset for asset, module in module_map.items() if module == "rates"]
        rate_total = merged.reindex(columns=current_rate_assets, fill_value=0.0).sum(axis=1)
        for asset, fraction in split.items():
            merged[asset] = rate_total * float(fraction)
            module_map[asset] = "rates"
    merged = merged.div(merged.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
    # ``merged`` may include fixed strategic curve-tenor columns added above;
    # return the mapping for the actual portfolio columns rather than only the
    # historical monthly-anchor columns.
    return merged, {asset: module_map[asset] for asset in merged.columns}


def _asset_score_horizon_weights(cfg: dict[str, Any]) -> dict[int, float]:
    """Return the predeclared, non-performance-selected horizon blend.

    A horizon has economic meaning (short tactical / intermediate / strategic),
    not a rebalance clock.  Compatibility with older configurations that use a
    single ``asset_score_horizon_days`` is retained for historical reruns.
    """
    portfolio_cfg = cfg["portfolio"]
    configured = portfolio_cfg.get("asset_score_horizon_weights")
    if configured is None:
        return {int(portfolio_cfg["asset_score_horizon_days"]): 1.0}
    weights = {int(horizon): float(weight) for horizon, weight in configured.items()}
    weights = {horizon: weight for horizon, weight in weights.items() if weight > 0.0}
    if not weights:
        raise ValueError("portfolio.asset_score_horizon_weights must contain a positive weight")
    return weights


def _asset_prediction_panels(cfg: dict[str, Any], dates: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    predictions = pd.read_parquet(project_path(cfg, "asset_predictions"))
    predictions["as_of_date"] = pd.to_datetime(predictions["as_of_date"], errors="coerce")
    horizon_weights = _asset_score_horizon_weights(cfg)
    predictions = predictions.loc[predictions["horizon_days"].isin(horizon_weights)].copy()
    if predictions.empty:
        empty = pd.DataFrame(index=dates)
        return empty, empty.copy(), empty.copy()

    # ``walkforward_relative_score`` is exactly zero when a component has not
    # passed the causal forecast gate. ``residual_alpha`` identifies whether a
    # component is live, so a neutral-but-live score is not mistaken for a
    # fallback. No raw prediction can enter this aggregation.
    predictions["configured_horizon_weight"] = predictions["horizon_days"].map(horizon_weights).astype(float)
    predictions["active_horizon_weight"] = predictions["configured_horizon_weight"] * predictions[
        "residual_alpha"
    ].gt(0.0).astype(float)
    key_columns = ["as_of_date", "asset_code"]
    # This is intentionally vectorised: three products reuse a panel with more
    # than one hundred thousand horizon rows, so a Python group loop materially
    # delays an otherwise deterministic backtest.
    weights = predictions["active_horizon_weight"]
    predictions["weighted_score"] = weights * predictions["walkforward_relative_score"]
    predictions["weighted_residual"] = weights * predictions["model_residual_score"]
    predictions["weighted_uncertainty_square"] = weights * np.square(
        predictions["prediction_uncertainty"].fillna(1.0)
    )
    blended = predictions.groupby(key_columns, as_index=False).agg(
        total_weight=("active_horizon_weight", "sum"),
        weighted_score=("weighted_score", "sum"),
        weighted_residual=("weighted_residual", "sum"),
        weighted_uncertainty_square=("weighted_uncertainty_square", "sum"),
    )
    active_total = blended["total_weight"].gt(0.0)
    denominator = blended["total_weight"].where(active_total, 1.0)
    blended["score"] = (blended["weighted_score"] / denominator).where(active_total, 0.0)
    blended["residual"] = (blended["weighted_residual"] / denominator).where(active_total, 0.0)
    # RMS preserves the penalty from any noisy active horizon.
    blended["uncertainty"] = np.sqrt(
        blended["weighted_uncertainty_square"] / denominator
    ).where(active_total, 1.0)
    score = blended.pivot(index="as_of_date", columns="asset_code", values="score").reindex(dates).fillna(0.0)
    residual = blended.pivot(index="as_of_date", columns="asset_code", values="residual").reindex(dates).fillna(0.0)
    uncertainty = blended.pivot(index="as_of_date", columns="asset_code", values="uncertainty").reindex(dates).fillna(1.0)
    return score, residual, uncertainty


def _bounded_within_sleeve(
    base: pd.Series,
    score: pd.Series,
    uncertainty: pd.Series,
    sleeve_weight: float,
    strength: float,
    max_share: float,
    max_active: float,
) -> pd.Series:
    base = base.clip(lower=0.0)
    if float(base.sum()) <= 1e-12 or sleeve_weight <= 1e-12:
        return base * 0.0
    shares = base / float(base.sum())
    values = pd.to_numeric(score.reindex(base.index), errors="coerce").fillna(0.0)
    risk = pd.to_numeric(uncertainty.reindex(base.index), errors="coerce").fillna(1.0).clip(lower=0.0)
    valid = shares.gt(1e-12)
    centered = values - float(np.average(values[valid], weights=shares[valid]))
    confidence = 1.0 / (1.0 + risk)
    multiplier = np.exp(np.clip(strength * centered * confidence, -0.75, 0.75))
    tilted = shares * multiplier
    tilted = tilted / float(tilted.sum())
    tilted = tilted.clip(upper=max_share)
    tilted = tilted / float(tilted.sum())
    raw = tilted * sleeve_weight
    lower = (shares * sleeve_weight - max_active).clip(lower=0.0)
    upper = shares * sleeve_weight + max_active
    constrained = raw.clip(lower=lower, upper=upper)
    # A small bounded-simplex projection preserves the exact sleeve total.
    for _ in range(20):
        gap = sleeve_weight - float(constrained.sum())
        if abs(gap) <= 1e-12:
            break
        room = (upper - constrained) if gap > 0 else (constrained - lower)
        eligible = room.gt(1e-12)
        if not bool(eligible.any()):
            break
        allocation = room[eligible] / float(room[eligible].sum())
        constrained.loc[eligible] += np.sign(gap) * np.minimum(abs(gap) * allocation, room[eligible])
    return constrained * (sleeve_weight / max(float(constrained.sum()), 1e-12))


def build_daily_targets(profile: str, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    integrated_cfg = load_integrated_config(ROOT / cfg["portfolio"]["integrated_base_config"])
    integrated_cfg = copy.deepcopy(integrated_cfg)
    integrated_cfg["data"]["predictions"] = str(cfg["outputs"]["sleeve_predictions"])
    integrated_cfg["data"]["start_date"] = str(cfg["data"]["prediction_start"])
    for key, portfolio_key in [("risk_budget_pattern", "risk_budget_pattern"),
                               ("formal_sleeve_pattern", "formal_sleeve_pattern"),
                               ("formal_asset_pattern", "formal_anchor_pattern")]:
        if portfolio_key in cfg["portfolio"]:
            integrated_cfg["data"][key] = cfg["portfolio"][portfolio_key]
    integrated_cfg["data"]["rule_views"] = "data_processed/market_skill_views_mechanical_daily_v1.parquet"
    if "frozen_asset_returns" in cfg["portfolio"]:
        integrated_cfg["data"]["asset_returns"] = cfg["portfolio"]["frozen_asset_returns"]
    for key,portfolio_key in [('state_cards','frozen_state_cards'),('rule_views','frozen_rule_views')]:
        if portfolio_key in cfg['portfolio']:
            integrated_cfg['data'][key] = cfg['portfolio'][portfolio_key]
    state = build_state_frame(integrated_cfg)

    event_cfg, _ = load_event_config()
    event_cfg = copy.deepcopy(event_cfg)
    event_cfg["event"]["action_names"] = ["hold", "formal_anchor", "rules_only", "adaptive_fused"]
    event_cfg["event"]["allowed_families"][profile] = event_cfg["event"]["action_names"]
    event_cfg["event"]["prediction_horizons_days"] = list(cfg["data"]["horizons_days"])
    env = EventDrivenEnvironment(profile, state, event_cfg, integrated_cfg)

    # Strict point-in-time treatment for both formal and risk-budget anchors.
    calendar = pd.DatetimeIndex(env.calendar)
    formal_assets, module_map = _strict_monthly_anchor(profile, calendar, cfg)
    risk_source = pd.read_parquet(ROOT / integrated_cfg["data"]["risk_budget_pattern"].format(profile=profile))
    risk_source["date"] = pd.to_datetime(risk_source["date"], errors="coerce")
    strict_risk = calendar_decision_targets(risk_source.set_index("date")[MODULES], calendar).ffill()
    if strict_risk.notna().all(axis=1).any():
        env.base.risk_budget_values = np.vstack(
            [
                env.base.risk_budget_values[i] if row.isna().any() else row.to_numpy(dtype=float)
                for i, (_, row) in enumerate(strict_risk[MODULES].iterrows())
            ]
        )

    dates = pd.DatetimeIndex(env.state["date"])
    score, residual, uncertainty = _asset_prediction_panels(cfg, dates)
    action = env.action_index["adaptive_fused"]
    target_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    asset_columns = list(formal_assets.columns)
    for index, date in enumerate(dates):
        if index % 200 == 0:
            print(f"[{profile}] building repaired targets {index}/{len(dates)}", flush=True)
        calendar_position = int(env.calendar_positions[index])
        sleeve_target, diagnostics = env.candidate_weights(index, action)
        base = formal_assets.iloc[calendar_position].reindex(asset_columns).fillna(0.0)
        output = pd.Series(0.0, index=asset_columns)
        for sleeve_position, sleeve in enumerate(MODULES):
            selected = [asset for asset in asset_columns if module_map.get(asset) == sleeve]
            if not selected:
                continue
            output.loc[selected] = _bounded_within_sleeve(
                base.loc[selected],
                score.loc[date].reindex(selected),
                uncertainty.loc[date].reindex(selected),
                float(sleeve_target[sleeve_position]),
                float(cfg["portfolio"]["asset_tilt_strength"][profile]),
                float(cfg["portfolio"]["max_asset_share_of_sleeve"]),
                float(cfg["portfolio"]["max_asset_active_weight"][profile]),
            )
        output /= max(float(output.sum()), 1e-12)
        gated_scores: dict[str, float] = {}
        gated_uncertainties: dict[str, float] = {}
        for sleeve in MODULES:
            selected = [asset for asset in asset_columns if module_map.get(asset) == sleeve]
            if not selected:
                gated_scores[sleeve] = 0.0
                gated_uncertainties[sleeve] = 1.0
                continue
            sleeve_base = base.loc[selected].clip(lower=0.0)
            if float(sleeve_base.sum()) <= 1e-12:
                sleeve_weights = np.full(len(selected), 1.0 / len(selected))
            else:
                sleeve_weights = (sleeve_base / float(sleeve_base.sum())).to_numpy(dtype=float)
            sleeve_score = score.loc[date].reindex(selected).fillna(0.0).to_numpy(dtype=float)
            sleeve_uncertainty = uncertainty.loc[date].reindex(selected).fillna(1.0).to_numpy(dtype=float)
            gated_scores[sleeve] = float(np.dot(sleeve_weights, sleeve_score))
            gated_uncertainties[sleeve] = float(
                np.sqrt(np.dot(sleeve_weights, np.square(sleeve_uncertainty)))
            )
        for asset in asset_columns:
            target_rows.append(
                {
                    "date": date,
                    "profile": profile,
                    "asset_code": asset,
                    "module": module_map[asset],
                    "weight": float(output[asset]),
                    "formal_anchor_weight": float(base[asset]),
                    "asset_score": float(score.at[date, asset]) if asset in score and pd.notna(score.at[date, asset]) else 0.0,
                    "asset_model_residual": float(residual.at[date, asset]) if asset in residual and pd.notna(residual.at[date, asset]) else 0.0,
                    "asset_uncertainty": float(uncertainty.at[date, asset]) if asset in uncertainty and pd.notna(uncertainty.at[date, asset]) else 1.0,
                }
            )
        diagnostic_rows.append(
            {
                "date": date,
                "profile": profile,
                "forecast_confidence": float(diagnostics.get("forecast_confidence", 0.0)),
                "effective_model_level": float(diagnostics.get("effective_model_level", 0.0)),
                "optimizer_backend": diagnostics.get("optimizer_backend", "unknown"),
                "horizon_weights": json.dumps(diagnostics.get("horizon_weights", {}), sort_keys=True),
                "target_distance_from_formal": float(np.abs(output - base).sum()),
                "mean_prediction_uncertainty": float(uncertainty.loc[date].reindex(asset_columns).mean()),
                "max_horizon_end_date": pd.Timestamp(env.state.loc[index, "max_horizon_end_date"]),
                **{f"target_{sleeve}": float(sleeve_target[i]) for i, sleeve in enumerate(MODULES)},
                **{f"gated_score_{sleeve}": gated_scores[sleeve] for sleeve in MODULES},
                **{f"gated_uncertainty_{sleeve}": gated_uncertainties[sleeve] for sleeve in MODULES},
            }
        )
    targets = pd.DataFrame(target_rows)
    diagnostics = pd.DataFrame(diagnostic_rows)
    targets.to_parquet(_output_path(cfg, "target_weights_pattern", profile), index=False)
    return targets, diagnostics, env.state.copy(), formal_assets


@dataclass
class ControllerContext:
    age: int = 0
    last_action: int = 0
    nav: float = 1.0
    formal_nav: float = 1.0
    peak: float = 1.0
    formal_peak: float = 1.0
    since_trade_nav: float = 1.0
    since_trade_formal_nav: float = 1.0
    ewma_variance: float = 0.0
    formal_ewma_variance: float = 0.0
    formal_weights: np.ndarray | None = None
    formal_pending: np.ndarray | None = None


class AdaptiveFractionEnvironment:
    def __init__(
        self,
        profile: str,
        cfg: dict[str, Any],
        targets: pd.DataFrame,
        diagnostics: pd.DataFrame,
        state: pd.DataFrame,
        formal_assets: pd.DataFrame,
    ) -> None:
        self.profile = profile
        self.cfg = cfg
        self.actions = np.asarray(cfg["controller"]["action_fractions"], dtype=float)
        self.assets = sorted(targets["asset_code"].unique())
        self.module_map = targets.drop_duplicates("asset_code").set_index("asset_code")["module"].to_dict()
        self.target = targets.pivot_table(index="date", columns="asset_code", values="weight", aggfunc="last").reindex(columns=self.assets)
        self.diagnostics = diagnostics.set_index("date").sort_index()
        self.state = state.set_index("date").sort_index()
        returns, _, tradable, gaps = accounting_return_panel(ROOT, cfg["outputs"]["dataset_npz"], cfg["outputs"]["metadata"])
        common_dates = self.target.index.intersection(returns.index)
        self.dates = pd.DatetimeIndex(common_dates).sort_values()
        self.target = self.target.reindex(self.dates).fillna(0.0)
        self.returns = returns.reindex(index=self.dates, columns=self.assets)
        self.tradable = tradable.reindex(index=self.dates, columns=self.assets).fillna(False)
        self.formal = formal_assets.reindex(index=self.dates, columns=self.assets).ffill().fillna(0.0)
        self.diagnostics = self.diagnostics.reindex(self.dates).ffill()
        self.state = self.state.reindex(self.dates).ffill()
        preferred = [
            "state_bond_score", "state_equity_score", "state_risk_budget_score", "state_liquidity_shock",
            "state_card_cr_state", "state_card_r_state", "state_card_cb_state", "state_card_eq_state",
            "state_card_alt_state", "state_card_rb_state",
        ]
        preferred.extend(f"state_card_{module}_mean" for module in MODULES)
        preferred.extend(f"state_card_{module}_dispersion" for module in MODULES)
        for horizon in cfg["data"]["horizons_days"]:
            preferred.extend(
                [
                    f"state_it_h{horizon}_model_reliability",
                    f"state_it_h{horizon}_baseline_reliability",
                    f"state_it_h{horizon}_spread",
                ]
            )
            preferred.extend(
                f"state_it_uncertainty_h{horizon}_{module}" for module in MODULES
            )
        self.state_cols = [column for column in preferred if column in self.state.columns]
        self.cost_rate = float(cfg["portfolio"]["transaction_cost_bps"]) / 10000.0

    def initial_weights(self, index: int) -> np.ndarray:
        values = self.formal.iloc[index].to_numpy(dtype=float)
        return values / max(float(values.sum()), 1e-12)

    @staticmethod
    def initial_context() -> ControllerContext:
        return ControllerContext()

    def eligible(self, cutoff: pd.Timestamp) -> np.ndarray:
        execution = self.dates.to_series().shift(-1).to_numpy()
        return np.flatnonzero(
            (self.dates < cutoff) & (pd.DatetimeIndex(pd.to_datetime(execution)) < cutoff)
        )

    def _sleeves(self, weights: np.ndarray) -> np.ndarray:
        return np.array([
            float(sum(weights[i] for i, asset in enumerate(self.assets) if self.module_map.get(asset) == module))
            for module in MODULES
        ])

    def observation(self, index: int, current: np.ndarray, context: ControllerContext, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        static = self.state.iloc[index][self.state_cols].to_numpy(dtype=float)
        static = np.nan_to_num((static - mean) / std, nan=0.0, posinf=6.0, neginf=-6.0)
        target = self.target.iloc[index].to_numpy(dtype=float)
        current_sleeve = self._sleeves(current)
        target_sleeve = self._sleeves(target)
        formal_sleeve = self._sleeves(self.formal.iloc[index].to_numpy(dtype=float))
        diag = self.diagnostics.iloc[index]
        drawdown = context.nav / max(context.peak, 1e-12) - 1.0
        relative_since = context.since_trade_nav / max(context.since_trade_formal_nav, 1e-12) - 1.0
        dynamic = np.array(
            [
                *current_sleeve,
                *target_sleeve,
                *(target_sleeve - formal_sleeve),
                *[float(diag.get(f"gated_score_{module}", 0.0)) for module in MODULES],
                *[float(diag.get(f"gated_uncertainty_{module}", 1.0)) for module in MODULES],
                float(np.abs(target - current).sum()),
                float(diag.get("forecast_confidence", 0.0)),
                float(diag.get("effective_model_level", 0.0)),
                float(diag.get("mean_prediction_uncertainty", 1.0)),
                math.log1p(context.age) / math.log(253.0),
                float(np.clip(relative_since / 0.05, -3.0, 3.0)),
                float(np.clip(drawdown / 0.10, -3.0, 0.0)),
                float(np.clip(math.sqrt(max(context.ewma_variance, 0.0) * 252.0) / 0.20, 0.0, 3.0)),
                *[float(context.last_action == action) for action in range(len(self.actions))],
            ],
            dtype=float,
        )
        return np.concatenate([np.clip(static, -6.0, 6.0), dynamic]).astype(np.float32)

    def mechanical_action(
        self, index: int, current: np.ndarray, context: ControllerContext | None = None
    ) -> int:
        target = self.target.iloc[index].to_numpy(dtype=float)
        distance = float(np.abs(target - current).sum())
        threshold = float(self.cfg["controller"]["mechanical_no_trade_l1"][self.profile])
        if distance < threshold:
            return 0
        minimum_hold = int(self.cfg["controller"].get("minimum_hold_days", {}).get(self.profile, 0))
        liquidity_shock = bool(self.state.iloc[index].get("state_liquidity_shock", 0.0))
        if context is not None and context.age < minimum_hold and not liquidity_shock:
            return 0
        confidence = float(self.diagnostics.iloc[index].get("forecast_confidence", 0.0))
        uncertainty = float(self.diagnostics.iloc[index].get("mean_prediction_uncertainty", 1.0))
        effective = confidence / (1.0 + float(self.cfg["controller"]["uncertainty_penalty"]) * max(uncertainty, 0.0))
        if distance < 2.0 * threshold or effective < 0.05:
            return 1
        if distance < 4.0 * threshold or effective < 0.15:
            return 2
        return len(self.actions) - 1

    def step(
        self, index: int, action: int, current: np.ndarray, context: ControllerContext
    ) -> tuple[float, int, np.ndarray, ControllerContext, dict[str, Any]]:
        if index + 1 >= len(self.dates):
            return 0.0, len(self.dates), current, context, {}
        target = self.target.iloc[index].to_numpy(dtype=float)
        fraction = float(self.actions[action])
        next_returns = self.returns.iloc[index + 1].to_numpy(dtype=float)
        quotes = self.tradable.iloc[index + 1].to_numpy(dtype=bool)
        execution = advance_holdings(current, target, fraction, next_returns, self.cost_rate, quotes)
        traded, drifted = execution["weights"], execution["end_weights"]
        turnover, cost = execution["turnover"], execution["cost"]
        strategy_return = execution["return"]

        formal_target = self.formal.iloc[index].to_numpy(dtype=float)
        formal_prev = self.formal.iloc[max(index - 1, 0)].to_numpy(dtype=float)
        formal_current = formal_target if context.formal_weights is None else context.formal_weights
        formal_changed = not np.allclose(formal_target, formal_prev, atol=1e-12, rtol=0)
        pending = formal_target if formal_changed else context.formal_pending
        formal_execution = advance_holdings(formal_current, formal_target if pending is None else pending, float(pending is not None), next_returns, self.cost_rate, quotes)
        formal_return = formal_execution["return"]
        beta = 0.94
        strategy_var = beta * context.ewma_variance + (1.0 - beta) * strategy_return**2
        formal_var = beta * context.formal_ewma_variance + (1.0 - beta) * formal_return**2
        nav = context.nav * (1.0 + strategy_return)
        formal_nav = context.formal_nav * (1.0 + formal_return)
        peak = max(context.peak, nav)
        formal_peak = max(context.formal_peak, formal_nav)
        dd = nav / max(peak, 1e-12) - 1.0
        formal_dd = formal_nav / max(formal_peak, 1e-12) - 1.0
        downside = max(0.0, -strategy_return) - max(0.0, -formal_return)
        vol_worsening = max(0.0, math.sqrt(strategy_var * 252.0) - math.sqrt(formal_var * 252.0)) * (1.0 - beta)
        dd_worsening = max(0.0, abs(dd) - abs(formal_dd))
        active = float(np.abs(traded - formal_target).sum())
        reward = (
            strategy_return - formal_return
            - 0.35 * max(downside, 0.0)
            - {"low_vol": 0.75, "balanced": 0.50, "high_vol": 0.20}[self.profile] * vol_worsening
            - {"low_vol": 0.50, "balanced": 0.35, "high_vol": 0.15}[self.profile] * dd_worsening
            - float(self.cfg["controller"]["active_weight_penalty"]) * active
        ) * 100.0
        rebalanced = turnover > 1e-8
        new_context = ControllerContext(
            age=0 if rebalanced else context.age + 1,
            last_action=action if rebalanced else context.last_action,
            nav=nav,
            formal_nav=formal_nav,
            peak=peak,
            formal_peak=formal_peak,
            since_trade_nav=(1.0 + strategy_return) if rebalanced else context.since_trade_nav * (1.0 + strategy_return),
            since_trade_formal_nav=(1.0 + formal_return) if rebalanced else context.since_trade_formal_nav * (1.0 + formal_return),
            ewma_variance=strategy_var,
            formal_ewma_variance=formal_var,
            formal_weights=formal_execution["end_weights"],
            formal_pending=pending if formal_execution["unfilled"] else None,
        )
        info = {
            "date": self.dates[index + 1],
            "signal_date": self.dates[index],
            "action_fraction": fraction,
            "strategy_return": strategy_return,
            "formal_return": formal_return,
            "turnover": turnover,
            "transaction_cost": cost,
            "active_l1": active,
            "formal_turnover": formal_execution["turnover"],
            "formal_cost": formal_execution["cost"],
            "weights": traded.copy(),
            "end_weights": drifted.copy(),
        }
        return float(reward), index + 1, drifted, new_context, info

    def teacher_action(
        self, index: int, current: np.ndarray, horizon_days: int, context: ControllerContext | None = None
    ) -> int:
        """Cost-aware historical execution label used only inside a train split.

        Each candidate action is executed once, then held for a short realised
        horizon and compared with the contemporaneous formal anchor.  These
        outcomes are labels for dates whose entire holding window is before the
        controller training cutoff; they are never present in an online state.
        """
        if horizon_days < 1 or index + horizon_days >= len(self.dates):
            raise IndexError("Teacher horizon extends beyond available returns")
        target = self.target.iloc[index].to_numpy(dtype=float)
        formal_penalty = {
            "low_vol": 0.50,
            "balanced": 0.35,
            "high_vol": 0.15,
        }[self.profile]
        utilities: list[float] = []
        formal_held = self.formal.iloc[index].to_numpy(dtype=float) if context is None or context.formal_weights is None else context.formal_weights.copy()
        formal_returns = []
        for offset in range(1, horizon_days + 1):
            position = index + offset - 1
            formal_target = self.formal.iloc[position].to_numpy(dtype=float)
            formal_prev = self.formal.iloc[max(position - 1, 0)].to_numpy(dtype=float)
            changed = not np.allclose(formal_target, formal_prev, atol=1e-12, rtol=0)
            if offset == 1:
                teacher_pending = None if context is None else context.formal_pending
            if changed:
                teacher_pending = formal_target
            execution = advance_holdings(formal_held, formal_target if teacher_pending is None else teacher_pending, float(teacher_pending is not None), self.returns.iloc[position + 1], self.cost_rate, self.tradable.iloc[position + 1])
            if not execution["unfilled"]:
                teacher_pending = None
            formal_returns.append(execution["return"])
            formal_held = execution["end_weights"]
        for action, fraction in enumerate(self.actions):
            held = current.copy()
            utility = 0.0
            for offset in range(1, horizon_days + 1):
                returns = self.returns.iloc[index + offset].to_numpy(dtype=float)
                execution = advance_holdings(held, target, float(fraction) if offset == 1 else 0.0, returns, self.cost_rate, self.tradable.iloc[index + offset])
                strategy_return = execution["return"]
                formal_return = formal_returns[offset - 1]
                utility += strategy_return - formal_return
                utility -= 0.35 * max(0.0, max(0.0, -strategy_return) - max(0.0, -formal_return))
                # Hold the candidate after the first execution; the label is a
                # timing decision, not a clairvoyant sequence of later trades.
                held = execution["end_weights"]
            active = float(np.abs(held - self.formal.iloc[index + horizon_days - 1].to_numpy(dtype=float)).sum())
            utility -= formal_penalty * float(self.cfg["controller"]["active_weight_penalty"]) * active
            utilities.append(utility)
        return int(np.argmax(np.asarray(utilities, dtype=float)))


def _simulate(
    env: AdaptiveFractionEnvironment,
    start: int,
    stop: int,
    model: ActorCritic | None,
    mean: np.ndarray,
    std: np.ndarray,
) -> tuple[float, list[dict[str, Any]]]:
    current = env.initial_weights(start)
    context = env.initial_context()
    total = 0.0
    rows: list[dict[str, Any]] = []
    for index in range(start, min(stop, len(env.dates) - 1)):
        if model is None:
            action = env.mechanical_action(index, current, context)
        else:
            with torch.no_grad():
                obs = torch.tensor(env.observation(index, current, context, mean, std))[None, :]
                logits, _ = model(obs)
                action = int(logits.argmax(dim=1).item())
        reward, _, current, context, info = env.step(index, action, current, context)
        total += reward
        if info:
            info["action"] = action
            rows.append(info)
    return total, rows


def _controller_gate_diagnostics(
    env: AdaptiveFractionEnvironment,
    start: int,
    stop: int,
    model: ActorCritic,
    mean: np.ndarray,
    std: np.ndarray,
    blocks: int,
) -> dict[str, Any]:
    boundaries = np.linspace(start, stop, max(int(blocks), 1) + 1, dtype=int)
    edges: list[float] = []
    model_rewards: list[float] = []
    fallback_rewards: list[float] = []
    for left, right in zip(boundaries[:-1], boundaries[1:], strict=True):
        if right - left < 5:
            continue
        model_reward, _ = _simulate(env, int(left), int(right), model, mean, std)
        fallback_reward, _ = _simulate(
            env, int(left), int(right), None, mean, std
        )
        days = max(int(right - left), 1)
        model_rewards.append(float(model_reward) / days)
        fallback_rewards.append(float(fallback_reward) / days)
        edges.append(float(model_reward - fallback_reward) / days)
    edge_array = np.asarray(edges, dtype=float)
    return {
        "deployment_gate_model_reward": float(np.mean(model_rewards))
        if model_rewards
        else -np.inf,
        "deployment_gate_fallback_reward": float(np.mean(fallback_rewards))
        if fallback_rewards
        else -np.inf,
        "deployment_gate_mean_edge": float(np.mean(edge_array))
        if edge_array.size
        else -np.inf,
        "deployment_gate_median_edge": float(np.median(edge_array))
        if edge_array.size
        else -np.inf,
        "deployment_gate_worst_edge": float(np.min(edge_array))
        if edge_array.size
        else -np.inf,
        "deployment_gate_positive_fraction": float(np.mean(edge_array > 0.0))
        if edge_array.size
        else 0.0,
        "deployment_gate_blocks": int(edge_array.size),
        "deployment_gate_edges": edge_array.tolist(),
    }


def controller_signature(env: AdaptiveFractionEnvironment, year: int) -> str:
    digest = hashlib.sha256()
    digest.update(ACCOUNTING_VERSION.encode())
    for filename in [__file__, ROOT / "src/portfolio_accounting.py", ROOT / "src/ep_skfolio_optimizer.py"]:
        digest.update(Path(filename).read_bytes())
    digest.update(json.dumps({"controller": env.cfg["controller"], "portfolio": env.cfg["portfolio"], "profile": env.profile, "year": year}, sort_keys=True).encode())
    cutoff = pd.Timestamp(year, 1, 1)
    for frame in [env.returns, env.tradable, env.target, env.formal, env.state[env.state_cols]]:
        history = frame.loc[frame.index < cutoff]
        digest.update(pd.util.hash_pandas_object(history, index=True).values.tobytes())
    return digest.hexdigest()


def load_controller_models(env: AdaptiveFractionEnvironment, year: int, path: Path):
    loaded = torch.load(path, map_location="cpu", weights_only=False) if path.exists() else []
    expected = controller_signature(env, year)
    if any(model.get("accounting_signature") != expected for model in loaded):
        raise RuntimeError(f"Stale PPO checkpoint for {env.profile}/{year}: retrain under repaired accounting")
    return loaded


def train_controller_seed(
    env: AdaptiveFractionEnvironment, eligible: np.ndarray, year: int, seed: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    cfg = env.cfg["controller"]
    selection_fraction = float(
        cfg.get("selection_validation_fraction", cfg["validation_fraction"] / 2.0)
    )
    deployment_fraction = float(
        cfg.get("deployment_gate_fraction", cfg["validation_fraction"] / 2.0)
    )
    n_selection = max(30, int(math.ceil(len(eligible) * selection_fraction)))
    n_deployment = max(30, int(math.ceil(len(eligible) * deployment_fraction)))
    purge = max(int(cfg.get("validation_purge_days", 20)), 0)
    deployment_start = len(eligible) - n_deployment
    selection_start = deployment_start - n_selection
    train_stop_position = selection_start - purge
    if train_stop_position < 60:
        raise ValueError(
            f"Not enough controller rows after nested validation split: {len(eligible)}"
        )
    train_rows = eligible[:train_stop_position]
    selection_rows = eligible[selection_start:deployment_start]
    deployment_rows = eligible[deployment_start:]
    static = env.state.iloc[train_rows][env.state_cols].to_numpy(dtype=float)
    mean = np.nanmean(static, axis=0)
    std = np.nanstd(static, axis=0)
    std = np.where(std > 1e-8, std, 1.0)
    probe = env.observation(int(train_rows[0]), env.initial_weights(int(train_rows[0])), env.initial_context(), mean, std)
    model = ActorCritic(len(probe), len(env.actions), int(cfg["hidden_dim"]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=1e-4)
    train_start, train_stop = int(train_rows[0]), int(train_rows[-1]) + 1
    val_start, val_stop = int(selection_rows[0]), int(selection_rows[-1]) + 1
    deployment_start_index = int(deployment_rows[0])
    deployment_stop_index = int(deployment_rows[-1]) + 1
    fallback_val, _ = _simulate(env, val_start, val_stop, None, mean, std)
    best_val = -np.inf
    best_state = copy.deepcopy(model.state_dict())
    wait = 0
    logs: list[dict[str, Any]] = []
    teacher_cfg = cfg.get("teacher_guidance", {})
    teacher_enabled = bool(teacher_cfg.get("enabled", False))
    if teacher_enabled:
        teacher_horizon = int(teacher_cfg.get("horizon_days", 5))
        teacher_indices = train_rows[: max(len(train_rows) - teacher_horizon, 0)]
        teacher_current = env.initial_weights(train_start)
        teacher_context = env.initial_context()
        teacher_observations: list[np.ndarray] = []
        teacher_actions: list[int] = []
        for index in teacher_indices:
            teacher_observations.append(
                env.observation(int(index), teacher_current, teacher_context, mean, std)
            )
            teacher_actions.append(
                env.teacher_action(int(index), teacher_current, teacher_horizon, teacher_context)
            )
            mechanical = env.mechanical_action(int(index), teacher_current, teacher_context)
            _, _, teacher_current, teacher_context, _ = env.step(
                int(index), mechanical, teacher_current, teacher_context
            )
        if not teacher_observations:
            raise ValueError("Teacher guidance has no fully matured training rows")
        teacher_obs_tensor = torch.tensor(np.vstack(teacher_observations), dtype=torch.float32)
        teacher_action_tensor = torch.tensor(teacher_actions, dtype=torch.long)
        for _ in range(int(teacher_cfg.get("pretrain_epochs", 0))):
            teacher_logits, _ = model(teacher_obs_tensor)
            teacher_loss = F.cross_entropy(teacher_logits, teacher_action_tensor)
            optimizer.zero_grad()
            teacher_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    else:
        teacher_obs_tensor = None
        teacher_action_tensor = None
    for epoch in range(1, int(cfg["max_epochs"]) + 1):
        current = env.initial_weights(train_start)
        context = env.initial_context()
        observations: list[np.ndarray] = []
        actions: list[int] = []
        rewards: list[float] = []
        old_logp: list[float] = []
        values: list[float] = []
        for index in range(train_start, train_stop):
            obs_array = env.observation(index, current, context, mean, std)
            obs = torch.tensor(obs_array)[None, :]
            with torch.no_grad():
                logits, value = model(obs)
                distribution = torch.distributions.Categorical(logits=logits)
                action = int(distribution.sample().item())
                logp = float(distribution.log_prob(torch.tensor([action])).item())
            reward, _, current, context, _ = env.step(index, action, current, context)
            observations.append(obs_array)
            actions.append(action)
            rewards.append(reward)
            old_logp.append(logp)
            values.append(float(value.item()))
        rewards_array = np.asarray(rewards)
        values_array = np.asarray(values)
        advantage = np.zeros(len(rewards_array))
        last = 0.0
        next_value = 0.0
        for position in reversed(range(len(rewards_array))):
            delta = rewards_array[position] + float(cfg["gamma"]) * next_value - values_array[position]
            last = delta + float(cfg["gamma"]) * float(cfg["gae_lambda"]) * last
            advantage[position] = last
            next_value = values_array[position]
        target_value = advantage + values_array
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        obs_tensor = torch.tensor(np.vstack(observations), dtype=torch.float32)
        action_tensor = torch.tensor(actions, dtype=torch.long)
        old_logp_tensor = torch.tensor(old_logp, dtype=torch.float32)
        advantage_tensor = torch.tensor(advantage, dtype=torch.float32)
        target_tensor = torch.tensor(target_value, dtype=torch.float32)
        for _ in range(int(cfg["updates_per_epoch"])):
            logits, value = model(obs_tensor)
            distribution = torch.distributions.Categorical(logits=logits)
            logp = distribution.log_prob(action_tensor)
            ratio = torch.exp(logp - old_logp_tensor)
            clipped = torch.clamp(ratio, 1.0 - float(cfg["clip_ratio"]), 1.0 + float(cfg["clip_ratio"]))
            policy_loss = -torch.minimum(ratio * advantage_tensor, clipped * advantage_tensor).mean()
            value_loss = torch.mean((value - target_tensor) ** 2)
            entropy = distribution.entropy().mean()
            teacher_loss = torch.tensor(0.0)
            if teacher_obs_tensor is not None and teacher_action_tensor is not None:
                teacher_logits, _ = model(teacher_obs_tensor)
                teacher_loss = F.cross_entropy(teacher_logits, teacher_action_tensor)
            loss = (
                policy_loss
                + float(cfg["value_loss_weight"]) * value_loss
                - float(cfg["entropy_weight"]) * entropy
                + float(teacher_cfg.get("policy_loss_weight", 0.0)) * teacher_loss
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        if epoch == 1 or epoch % 10 == 0:
            validation, _ = _simulate(env, val_start, val_stop, model, mean, std)
            logs.append(
                {
                    "profile": env.profile, "year": year, "seed": seed, "epoch": epoch,
                    "rollout_reward": float(rewards_array.sum()), "validation_reward": validation,
                    "mechanical_validation_reward": fallback_val, "entropy": float(entropy.detach()),
                    "teacher_loss": float(teacher_loss.detach()),
                }
            )
            print(
                f"[{env.profile}][{year}][seed={seed}] controller epoch={epoch} "
                f"validation={validation:.6f} fallback={fallback_val:.6f}", flush=True
            )
            if validation > best_val + 1e-4:
                best_val = validation
                best_state = copy.deepcopy(model.state_dict())
                wait = 0
            else:
                wait += 1
                if wait >= int(cfg["patience"]):
                    break
    model.load_state_dict(best_state)
    gate_diagnostics = _controller_gate_diagnostics(
        env,
        deployment_start_index,
        deployment_stop_index,
        model,
        mean,
        std,
        int(cfg.get("deployment_gate_blocks", 3)),
    )
    logs.append(
        {
            "profile": env.profile,
            "year": year,
            "seed": seed,
            "epoch": "deployment_gate",
            "validation_reward": best_val,
            "mechanical_validation_reward": fallback_val,
            **gate_diagnostics,
        }
    )
    return {
        "accounting_signature": controller_signature(env, year),
        "accounting_version": ACCOUNTING_VERSION,
        "seed": seed,
        "evaluation_year": year,
        "state_dict": best_state,
        "mean": mean,
        "std": std,
        "obs_dim": len(probe),
        "hidden_dim": int(cfg["hidden_dim"]),
        "action_dim": len(env.actions),
        "training_cutoff": pd.Timestamp(year, 1, 1),
        "validation_reward": best_val,
        "fallback_validation_reward": fallback_val,
        **gate_diagnostics,
    }, logs


def _choose_controller_action(
    env: AdaptiveFractionEnvironment,
    index: int,
    current: np.ndarray,
    context: ControllerContext,
    models: list[dict[str, Any]],
) -> tuple[int, str, int]:
    fallback = env.mechanical_action(index, current, context)
    minimum_edge = float(env.cfg["controller"]["minimum_validation_edge"])
    minimum_positive_fraction = float(
        env.cfg["controller"].get("minimum_deployment_positive_fraction", 2.0 / 3.0)
    )
    minimum_blocks = int(
        env.cfg["controller"].get("minimum_deployment_gate_blocks", 2)
    )
    worst_edge_tolerance = float(
        env.cfg["controller"].get("minimum_deployment_worst_edge", -0.0002)
    )
    # A deployment split with three blocks reports 2 / 3 as the binary float
    # 0.666666666..., while the configuration is written as decimal 0.6666667.
    # Treat mathematically equal fractions as equal; this is numerical hygiene,
    # not a relaxation of the economic gate.
    fraction_tolerance = 1e-7
    qualified = [
        model for model in models
        if float(model.get("deployment_gate_mean_edge", -np.inf)) >= minimum_edge
        and float(model.get("deployment_gate_positive_fraction", 0.0))
        >= minimum_positive_fraction - fraction_tolerance
        and int(model.get("deployment_gate_blocks", 0)) >= minimum_blocks
        and float(model.get("deployment_gate_worst_edge", -np.inf))
        >= worst_edge_tolerance
    ]
    if len(qualified) < int(env.cfg["controller"]["minimum_qualified_models"]):
        return fallback, "mechanical_fallback", len(qualified)
    votes: list[int] = []
    for checkpoint in qualified:
        model = ActorCritic(checkpoint["obs_dim"], checkpoint["action_dim"], checkpoint["hidden_dim"])
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        obs = env.observation(index, current, context, checkpoint["mean"], checkpoint["std"])
        with torch.no_grad():
            logits, _ = model(torch.tensor(obs)[None, :])
        votes.append(int(logits.argmax(dim=1).item()))
    counts = pd.Series(votes).value_counts()
    if int(counts.iloc[0]) < int(env.cfg["controller"]["minimum_qualified_models"]):
        return fallback, "ensemble_disagreement", len(qualified)
    return int(counts.index[0]), "ppo", len(qualified)


def run_profile(profile: str, cfg: dict[str, Any], train_models: bool = True) -> dict[str, Any]:
    targets, diagnostics, state, formal_assets = build_daily_targets(profile, cfg)
    env = AdaptiveFractionEnvironment(profile, cfg, targets, diagnostics, state, formal_assets)
    years = sorted(pd.DatetimeIndex(env.dates).year.unique())
    controller_model_key = (
        "controller_model_dir"
        if "controller_model_dir" in cfg.get("outputs", {})
        else "model_dir"
    )
    model_dir = (
        project_path(cfg, controller_model_key) / f"controller_{profile}"
    )
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoints: dict[int, list[dict[str, Any]]] = {}
    training_logs: list[dict[str, Any]] = []
    for year in years:
        eligible = env.eligible(pd.Timestamp(int(year), 1, 1))
        if len(eligible) < 300:
            checkpoints[int(year)] = []
            continue
        path = model_dir / f"policy_{int(year)}.pt"
        if not train_models:
            checkpoints[int(year)] = load_controller_models(env, int(year), path)
            continue
        annual: list[dict[str, Any]] = []
        for seed in cfg["controller"]["seeds"]:
            started = time.perf_counter()
            checkpoint, logs = train_controller_seed(env, eligible, int(year), int(seed))
            annual.append(checkpoint)
            training_logs.extend(logs)
            print(
                f"[{profile}][{year}][seed={seed}] controller done in {time.perf_counter()-started:.1f}s",
                flush=True,
            )
        checkpoints[int(year)] = annual
        torch.save(annual, path)

    current = env.initial_weights(0)
    context = env.initial_context()
    daily_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    holding_rows: list[dict[str, Any]] = []
    for index in range(len(env.dates) - 1):
        date = env.dates[index]
        action, source, qualified = _choose_controller_action(
            env, index, current, context, checkpoints.get(int(date.year), [])
        )
        reward, _, current, context, info = env.step(index, action, current, context)
        info.update({"profile": profile, "policy_source": source, "reward": reward})
        daily_rows.append({key: value for key, value in info.items() if key not in {"weights", "end_weights"}})
        decision_rows.append(
            {
                "date": date,
                "execution_date": info["date"],
                "profile": profile,
                "action": action,
                "action_fraction": info["action_fraction"],
                "policy_source": source,
                "qualified_models": qualified,
                "turnover": info["turnover"],
                "transaction_cost": info["transaction_cost"],
                "forecast_confidence": float(env.diagnostics.iloc[index].get("forecast_confidence", 0.0)),
                "max_horizon_end_date": env.diagnostics.iloc[index].get("max_horizon_end_date"),
                "training_cutoff": pd.Timestamp(int(date.year), 1, 1) if checkpoints.get(int(date.year)) else pd.NaT,
            }
        )
        for asset_position, asset in enumerate(env.assets):
            holding_rows.append(
                {
                    "date": info["date"], "signal_date": date, "profile": profile,
                    "asset_code": asset, "module": env.module_map[asset], "weight": float(info["weights"][asset_position]),
                }
            )
    daily = pd.DataFrame(daily_rows).sort_values("date")
    decisions = pd.DataFrame(decision_rows).sort_values("date")
    holdings = pd.DataFrame(holding_rows)
    daily["strategy_nav"] = (1.0 + daily["strategy_return"]).cumprod()
    daily["formal_nav"] = (1.0 + daily["formal_return"]).cumprod()
    mechanical_reward, mechanical_rows = _simulate(env, 0, len(env.dates) - 1, None, np.zeros(len(env.state_cols)), np.ones(len(env.state_cols)))
    mechanical = pd.DataFrame(mechanical_rows)[["date", "strategy_return"]].rename(columns={"strategy_return": "mechanical_return"})
    daily = daily.merge(mechanical, on="date", how="left")
    daily["mechanical_nav"] = (1.0 + daily["mechanical_return"]).cumprod()
    metric_rows: list[dict[str, Any]] = []
    for split, mask in {
        "full_oos": pd.Series(True, index=daily.index),
        "oos_2024_onward": daily["date"].ge(pd.Timestamp("2024-01-01")),
    }.items():
        row: dict[str, Any] = {"profile": profile, "split": split}
        for name in ["formal", "mechanical", "strategy"]:
            stats = perf_stats(daily.loc[mask, f"{name}_return"])
            row.update({f"{name}_{key}": value for key, value in stats.items()})
        row["delta_sharpe_vs_formal"] = row.get("strategy_sharpe", np.nan) - row.get("formal_sharpe", np.nan)
        row["delta_return_vs_formal"] = row.get("strategy_annual_return", np.nan) - row.get("formal_annual_return", np.nan)
        row["mean_daily_turnover"] = float(daily.loc[mask, "turnover"].mean())
        row["annualized_turnover"] = float(daily.loc[mask, "turnover"].mean() * 252.0)
        metric_rows.append(row)
    metrics = pd.DataFrame(metric_rows)
    daily.to_csv(_output_path(cfg, "daily_pattern", profile), index=False, encoding="utf-8-sig")
    decisions.to_parquet(_output_path(cfg, "decisions_pattern", profile), index=False)
    holdings.to_parquet(_output_path(cfg, "holdings_pattern", profile), index=False)
    metrics.to_csv(_output_path(cfg, "metrics_pattern", profile), index=False, encoding="utf-8-sig")
    # Replay mode loads causally trained checkpoints. Preserve the original
    # optimisation logs instead of replacing them with an empty CSV.
    if train_models or training_logs:
        pd.DataFrame(training_logs).to_csv(
            _output_path(cfg, "controller_log_pattern", profile), index=False, encoding="utf-8-sig"
        )
    return {
        "profile": profile,
        "daily_rows": len(daily),
        "ppo_days": int(daily["policy_source"].eq("ppo").sum()),
        "rebalances": int(daily["turnover"].gt(1e-8).sum()),
        "mechanical_reward": mechanical_reward,
    }


def _block_bootstrap_comparison(
    strategy: pd.Series,
    formal: pd.Series,
    draws: int,
    block_days: int,
    seed: int,
) -> dict[str, float]:
    """Circular block bootstrap for a same-date strategy versus formal anchor."""
    paired = pd.concat([strategy.rename("strategy"), formal.rename("formal")], axis=1).dropna()
    values = paired.to_numpy(dtype=float)
    n_obs = len(values)
    if n_obs < max(2 * block_days, 40):
        raise ValueError("Insufficient observations for block-bootstrap robustness gate")
    rng = np.random.default_rng(seed)
    annual_edges = np.empty(draws, dtype=float)
    sharpe_edges = np.empty(draws, dtype=float)
    n_blocks = int(math.ceil(n_obs / block_days))
    for draw in range(draws):
        starts = rng.integers(0, n_obs, size=n_blocks)
        indices = np.concatenate(
            [(np.arange(start, start + block_days) % n_obs) for start in starts]
        )[:n_obs]
        sample = values[indices]
        strategy_sample, formal_sample = sample[:, 0], sample[:, 1]
        annual_edges[draw] = (np.prod(1.0 + strategy_sample) ** (252.0 / n_obs)
                              - np.prod(1.0 + formal_sample) ** (252.0 / n_obs))
        strategy_sharpe = strategy_sample.mean() / max(strategy_sample.std(ddof=1), 1e-12) * math.sqrt(252.0)
        formal_sharpe = formal_sample.mean() / max(formal_sample.std(ddof=1), 1e-12) * math.sqrt(252.0)
        sharpe_edges[draw] = strategy_sharpe - formal_sharpe
    return {
        "bootstrap_draws": int(draws),
        "bootstrap_block_days": int(block_days),
        "bootstrap_return_positive_probability": float(np.mean(annual_edges > 0.0)),
        "bootstrap_sharpe_positive_probability": float(np.mean(sharpe_edges > 0.0)),
        "bootstrap_return_edge_q05": float(np.quantile(annual_edges, 0.05)),
        "bootstrap_return_edge_q95": float(np.quantile(annual_edges, 0.95)),
        "bootstrap_sharpe_edge_q05": float(np.quantile(sharpe_edges, 0.05)),
        "bootstrap_sharpe_edge_q95": float(np.quantile(sharpe_edges, 0.95)),
    }


def audit_and_report(cfg: dict[str, Any], profiles: list[str]) -> dict[str, Any]:
    summaries: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    for profile in profiles:
        metrics = pd.read_csv(_output_path(cfg, "metrics_pattern", profile))
        summaries.append(metrics)
        decisions = pd.read_parquet(_output_path(cfg, "decisions_pattern", profile))
        holdings = pd.read_parquet(_output_path(cfg, "holdings_pattern", profile))
        targets = pd.read_parquet(_output_path(cfg, "target_weights_pattern", profile))
        sums = holdings.groupby("date")["weight"].sum()
        target_sleeves = targets.groupby(["date", "module"])["weight"].sum().unstack(fill_value=0.0)
        v1cfg = load_v1_config()
        v1cfg["product"]["selected_profile"] = profile
        v1cfg = apply_selected_profile_overrides(v1cfg)
        bounds = v1cfg["product"]["profiles"][profile]["sleeve_bounds"]
        sleeve_bounds_ok = all(
            (target_sleeves[sleeve] >= float(bound[0]) - 1e-6).all()
            and (target_sleeves[sleeve] <= float(bound[1]) + 1e-6).all()
            for sleeve, bound in bounds.items()
            if sleeve in target_sleeves
        )
        audit_rows.extend(
            [
                {"profile": profile, "check": "weight_sum", "passed": bool(np.allclose(sums, 1.0, atol=1e-6)), "value": float((sums - 1.0).abs().max())},
                {"profile": profile, "check": "no_cash", "passed": bool(~holdings["asset_code"].eq("cash").any()), "value": int(holdings["asset_code"].eq("cash").sum())},
                {"profile": profile, "check": "signal_before_execution", "passed": bool(pd.to_datetime(decisions["date"]).lt(pd.to_datetime(decisions["execution_date"])).all()), "value": int((pd.to_datetime(decisions["date"]) >= pd.to_datetime(decisions["execution_date"])).sum())},
                {"profile": profile, "check": "controller_train_before_decision", "passed": bool((pd.to_datetime(decisions["training_cutoff"]).isna() | pd.to_datetime(decisions["training_cutoff"]).le(pd.to_datetime(decisions["date"]))).all()), "value": 0},
                {"profile": profile, "check": "teacher_label_horizon_within_controller_purge", "passed": bool(int(cfg["controller"].get("teacher_guidance", {}).get("horizon_days", 0)) <= int(cfg["controller"].get("validation_purge_days", 0))), "value": int(cfg["controller"].get("teacher_guidance", {}).get("horizon_days", 0))},
                {"profile": profile, "check": "target_sleeve_bounds", "passed": bool(sleeve_bounds_ok), "value": 0},
                {"profile": profile, "check": "nonnegative_holdings", "passed": bool(holdings["weight"].ge(-1e-12).all()), "value": float(holdings["weight"].min())},
                {"profile": profile, "check": "context_only_tech_not_traded", "passed": bool(~holdings["asset_code"].eq("equity_tech_growth").any()), "value": int(holdings["asset_code"].eq("equity_tech_growth").sum())},
            ]
        )
        v7 = pd.read_csv(_output_path(cfg, "daily_pattern", profile), parse_dates=["date"])
        bootstrap_cfg = cfg["portfolio"]["bootstrap"]
        oos_v7 = v7.loc[v7["date"].ge(pd.Timestamp("2024-01-01"))]
        bootstrap_rows.append(
            {
                "profile": profile,
                "start": oos_v7["date"].min(),
                "end": oos_v7["date"].max(),
                "days": int(len(oos_v7)),
                **_block_bootstrap_comparison(
                    oos_v7["strategy_return"],
                    oos_v7["formal_return"],
                    int(bootstrap_cfg["draws"]),
                    int(bootstrap_cfg["block_days"]),
                    int(bootstrap_cfg["seed"]) + len(bootstrap_rows),
                ),
            }
        )
        v6_path = ROOT / f"output/reports/integrated_daily_view_ppo_{profile}_daily.csv"
        if v6_path.exists():
            v6 = pd.read_csv(v6_path, parse_dates=["date"])
            common = v7[["date", "strategy_return"]].merge(
                v6[["date", "strategy_return"]], on="date", suffixes=("_v7", "_v6")
            )
            common = common.loc[common["date"].ge(pd.Timestamp("2024-01-01"))]
            v7_stats = perf_stats(common["strategy_return_v7"])
            v6_stats = perf_stats(common["strategy_return_v6"])
            comparison_rows.append(
                {
                    "profile": profile,
                    "start": common["date"].min(),
                    "end": common["date"].max(),
                    "days": len(common),
                    **{f"v7_{key}": value for key, value in v7_stats.items()},
                    **{f"v6_{key}": value for key, value in v6_stats.items()},
                    "delta_annual_return_vs_v6": v7_stats.get("annual_return", np.nan) - v6_stats.get("annual_return", np.nan),
                    "delta_sharpe_vs_v6": v7_stats.get("sharpe", np.nan) - v6_stats.get("sharpe", np.nan),
                    "drawdown_worsening_vs_v6": abs(v7_stats.get("max_drawdown", np.nan)) - abs(v6_stats.get("max_drawdown", np.nan)),
                }
            )
    summary = pd.concat(summaries, ignore_index=True)
    audit = pd.DataFrame(audit_rows)
    comparison = pd.DataFrame(comparison_rows)
    bootstrap = pd.DataFrame(bootstrap_rows)
    summary.to_csv(_output_path(cfg, "portfolio_summary"), index=False, encoding="utf-8-sig")
    audit.to_csv(_output_path(cfg, "audit"), index=False, encoding="utf-8-sig")
    comparison.to_csv(_output_path(cfg, "v6_comparison"), index=False, encoding="utf-8-sig")
    bootstrap.to_csv(_output_path(cfg, "bootstrap_report"), index=False, encoding="utf-8-sig")
    gate_rows = summary.loc[summary["split"].eq("oos_2024_onward")]
    complete_gate_rows = gate_rows.dropna(subset=["delta_sharpe_vs_formal", "delta_return_vs_formal"])
    has_gate_period = len(complete_gate_rows) == len(profiles)
    return_bootstrap_pass = bool(
        len(bootstrap) == len(profiles)
        and bootstrap["bootstrap_return_positive_probability"].ge(
            float(cfg["portfolio"]["bootstrap"]["minimum_return_positive_probability"])
        ).all()
    )
    sharpe_bootstrap_pass = bool(
        len(bootstrap) == len(profiles)
        and bootstrap["bootstrap_sharpe_positive_probability"].ge(
            float(cfg["portfolio"]["bootstrap"]["minimum_sharpe_positive_probability"])
        ).all()
    )
    v6_profile_gate: dict[str, bool] = {}
    drawdown_limits = {"low_vol": 0.003, "balanced": 0.005, "high_vol": 0.010}
    for _, row in comparison.iterrows():
        v6_profile_gate[str(row["profile"])] = bool(
            row["delta_annual_return_vs_v6"] >= 0.0
            and row["delta_sharpe_vs_v6"] >= 0.0
            and row["drawdown_worsening_vs_v6"] <= drawdown_limits[str(row["profile"])]
        )
    forecast_gate = json.loads(project_path(cfg, "gate_report").read_text(encoding="utf-8"))
    forecast_formal = bool(forecast_gate.get("enabled_for_formal_strategy", False))
    required_seeds = set(cfg["controller"].get("required_seeds", cfg["model"]["seeds"]))
    checkpoint_complete = all(
        set(source.get("seed") for source in torch.load(path, map_location="cpu", weights_only=False)) == required_seeds
        and all(source.get("accounting_version") == ACCOUNTING_VERSION
                for source in torch.load(path, map_location="cpu", weights_only=False))
        for profile in profiles
        for path in (project_path(cfg, "controller_model_dir") / f"controller_{profile}").glob("policy_*.pt")
    )
    has_checkpoints = all(any((project_path(cfg, "controller_model_dir") / f"controller_{profile}").glob("policy_*.pt")) for profile in profiles)
    gate = {
        "enabled_as_formal": bool(
            audit["passed"].all()
            and has_gate_period
            and (complete_gate_rows["delta_sharpe_vs_formal"] >= 0.0).all()
            and (complete_gate_rows["delta_return_vs_formal"] >= 0.0).all()
            and return_bootstrap_pass
            and sharpe_bootstrap_pass
            and forecast_formal
            and has_checkpoints
            and checkpoint_complete
        ),
        "checks": {
            "forecast_formal_gate": forecast_formal,
            "repaired_controller_checkpoints_present": has_checkpoints and checkpoint_complete,
            "point_in_time_audit": bool(audit["passed"].all()),
            "complete_2024_plus_period": bool(has_gate_period),
            "all_profiles_sharpe_not_worse": bool(has_gate_period and (complete_gate_rows["delta_sharpe_vs_formal"] >= 0.0).all()),
            "all_profiles_return_not_worse": bool(has_gate_period and (complete_gate_rows["delta_return_vs_formal"] >= 0.0).all()),
            "bootstrap_return_probability": return_bootstrap_pass,
            "bootstrap_sharpe_probability": sharpe_bootstrap_pass,
        },
        "v6_replacement_gate_by_profile": v6_profile_gate,
        "note": "A challenger remains shadow-only unless every formal gate passes.",
    }
    _output_path(cfg, "portfolio_gate").write_text(json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Hierarchical asset iTransformer + adaptive PPO v7",
        "",
        "- The 5/20/60-day outputs are forecast tasks, not fixed rebalancing clocks.",
        "- Risk budget is the anchor; rule views and model views enter EP/skfolio at sleeve level.",
        "- Asset forecasts only alter within-sleeve composition subject to formal-universe and active limits.",
        "- PPO chooses daily progress toward the target from 0%, 25%, 50%, or 100%.",
        "- PPO is initialized from five-day, cost-aware historical labels whose full outcome windows are inside each training split, then fine-tuned on-policy; online decisions never read those labels or future returns.",
        "- Every signal executes on the next trading day; 5bp turnover cost and weight drift are included.",
        "- Formal promotion also requires 20-day block-bootstrap evidence over the 2024-onward period; return and Sharpe probabilities are reported separately.",
        "",
        "## Formal gate",
        "",
        f"`enabled_as_formal = {gate['enabled_as_formal']}`",
        "",
        "## Results",
        "",
        summary.to_markdown(index=False),
        "",
        "## Same-date comparison with v6",
        "",
        comparison.to_markdown(index=False),
        "",
        "## Block-bootstrap robustness versus formal anchor",
        "",
        bootstrap.to_markdown(index=False),
        "",
        "## Point-in-time audit",
        "",
        audit.to_markdown(index=False),
    ]
    _output_path(cfg, "portfolio_report").write_text("\n".join(lines), encoding="utf-8")
    return gate


def run_pipeline(
    config_path: Path | None = None,
    profiles: list[str] | None = None,
    train_models: bool = True,
    controller_epochs: int | None = None,
    controller_seeds: list[int] | None = None,
) -> dict[str, Any]:
    cfg = load_model_config(config_path)
    torch.set_num_threads(int(cfg["controller"].get("torch_threads", 8)))
    cfg["controller"]["required_seeds"] = list(cfg["controller"]["seeds"])
    if bool(cfg.get("controller", {}).get("enabled_only_after_forecast_gate", False)):
        forecast_gate_path = project_path(cfg, "gate_report")
        if not forecast_gate_path.exists():
            raise RuntimeError(
                "Forecast gate report is missing; refusing to build model targets or train PPO."
            )
        forecast_gate = json.loads(forecast_gate_path.read_text(encoding="utf-8"))
        if not bool(forecast_gate.get("enabled_for_portfolio_stage", False)):
            summary = forecast_gate.get("summary", {})
            raise RuntimeError(
                "Forecast gate failed; model residuals and PPO are blocked. "
                f"asset_rank_ic={summary.get('asset_rank_ic')}, "
                f"asset_incremental_rank_ic={summary.get('asset_incremental_rank_ic')}, "
                f"sleeve_rank_ic={summary.get('sleeve_rank_ic')}, "
                f"sleeve_incremental_rank_ic={summary.get('sleeve_incremental_rank_ic')}"
            )
    if controller_epochs is not None:
        cfg["controller"]["max_epochs"] = int(controller_epochs)
    if controller_seeds is not None:
        cfg["controller"]["seeds"] = [int(value) for value in controller_seeds]
        cfg["controller"]["minimum_qualified_models"] = min(
            int(cfg["controller"]["minimum_qualified_models"]), len(controller_seeds)
        )
    selected = profiles or PROFILES
    results = [run_profile(profile, cfg, train_models=train_models) for profile in selected]
    gate = audit_and_report(cfg, selected)
    return {"profiles": results, "gate": gate}

from __future__ import annotations

import copy
import json
import math
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

from src.adaptive_daily_ppo import perf_stats
from src.ep_skfolio_optimizer import optimize_sleeve_weights
from src.integrated_daily_view_ppo import (
    MODULES,
    ROOT,
    build_environment,
    build_state_frame,
    load_config as load_integrated_config,
    _materialize_asset_weights,
)


DEFAULT_CONFIG = ROOT / "config" / "event_driven_rebalance_ppo_config.yaml"
ACTION_NAMES = ["hold", "formal_anchor", "rules_only", "baseline_fused", "model_fused"]


def load_config(path: Path | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    config_path = path or DEFAULT_CONFIG
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    base_path = ROOT / cfg.get("base_config", "config/integrated_daily_view_ppo_config.yaml")
    base_cfg = load_integrated_config(base_path)
    return cfg, base_cfg


def output_path(cfg: dict[str, Any], key: str, profile: str | None = None) -> Path:
    value = str(cfg["outputs"][key])
    if profile is not None:
        value = value.format(profile=profile)
    path = ROOT / value
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


@dataclass(frozen=True)
class HoldingContext:
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
    drawdown_gap: float = 0.0


class HierarchicalActorCritic(nn.Module):
    """Separate option termination from the conditional rebalance destination."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        if action_dim < 2:
            raise ValueError("hierarchical policy requires hold plus at least one rebalance action")
        self.action_dim = action_dim
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.termination_head = nn.Linear(hidden_dim, 2)
        self.family_head = nn.Linear(hidden_dim, action_dim - 1) if action_dim > 2 else None
        self.value_head = nn.Linear(hidden_dim, 1)

    def heads(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded = self.encoder(obs.float())
        family = (
            self.family_head(encoded)
            if self.family_head is not None
            else torch.zeros((len(encoded), 1), dtype=encoded.dtype, device=encoded.device)
        )
        return self.termination_head(encoded), family, self.value_head(encoded).squeeze(-1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        termination, family, value = self.heads(obs)
        termination_logp = torch.log_softmax(termination, dim=-1)
        if self.action_dim == 2:
            return termination_logp, value
        family_logp = torch.log_softmax(family, dim=-1)
        combined_logp = torch.cat(
            [termination_logp[:, :1], termination_logp[:, 1:2] + family_logp], dim=-1
        )
        return combined_logp, value


class EventDrivenEnvironment:
    """Daily optimal-stopping layer over the existing constrained candidate generator."""

    def __init__(self, profile: str, state: pd.DataFrame, cfg: dict[str, Any], base_cfg: dict[str, Any]) -> None:
        self.profile = profile
        self.cfg = cfg
        self.base_cfg = copy.deepcopy(base_cfg)
        self.base = build_environment(profile, state, self.base_cfg)
        self.state = self.base.state
        self.state_cols = self.base.state_cols
        self.calendar = self.base.calendar
        self.calendar_positions = self.base.calendar_positions
        self.action_names = list(cfg["event"].get("action_names", ACTION_NAMES))
        if not self.action_names or self.action_names[0] != "hold":
            raise ValueError("the first event action must be hold")
        self.action_index = {name: index for index, name in enumerate(self.action_names)}
        self.candidate_cache: dict[tuple[int, int], tuple[np.ndarray, dict[str, Any]]] = {}
        self.transaction_cost = float(base_cfg["reward"]["transaction_cost_bps"]) / 10000.0

    def initial_weights(self, index: int) -> np.ndarray:
        return self.base.initial_weights(index)

    @staticmethod
    def initial_context() -> HoldingContext:
        return HoldingContext()

    def eligible_rows(self, cutoff: pd.Timestamp) -> np.ndarray:
        return self.base.eligible_rows(cutoff)

    def _allowed(self) -> set[str]:
        return set(str(value) for value in self.cfg["event"]["allowed_families"][self.profile])

    def _horizon_available(self, index: int, horizon: int, family: str) -> bool:
        row = self.state.loc[index]
        model = row[[f"state_it_model_h{horizon}_{sleeve}" for sleeve in MODULES]].to_numpy(dtype=float)
        baseline = row[[f"state_it_baseline_h{horizon}_{sleeve}" for sleeve in MODULES]].to_numpy(dtype=float)
        model_reliability = float(row[f"state_it_h{horizon}_model_reliability"])
        baseline_reliability = float(row[f"state_it_h{horizon}_baseline_reliability"])
        if family == "model_fused":
            incremental = model_reliability - baseline_reliability
            forecast = model
            reliable = bool(
                incremental >= float(self.base_cfg["actions"]["minimum_incremental_reliability"])
                and model_reliability >= float(self.base_cfg["actions"]["minimum_model_reliability"])
            )
        else:
            forecast = baseline
            reliable = baseline_reliability >= float(self.base_cfg["actions"]["minimum_model_reliability"])
        return bool(
            reliable
            and np.isfinite(forecast).all()
            and float(np.ptp(forecast)) >= float(self.base_cfg["actions"]["minimum_score_spread"])
        )

    def action_mask(self, index: int) -> np.ndarray:
        allowed = self._allowed()
        mask = np.array([name in allowed for name in self.action_names], dtype=bool)
        horizons = [int(value) for value in self.cfg["event"]["prediction_horizons_days"]]
        for family in ["baseline_fused", "model_fused"]:
            if family in self.action_index:
                action = self.action_index[family]
                mask[action] = bool(mask[action] and any(self._horizon_available(index, h, family) for h in horizons))
        if "adaptive_fused" in self.action_index:
            action = self.action_index["adaptive_fused"]
            mask[action] = bool(
                mask[action]
                and any(
                    self._horizon_available(index, h, "model_fused")
                    or self._horizon_available(index, h, "baseline_fused")
                    for h in horizons
                )
            )
        mask[self.action_index["hold"]] = True
        return mask

    def _dynamic_predictive_view(self, index: int, family: str) -> tuple[np.ndarray, dict[str, Any]]:
        component_rows: list[tuple[int, np.ndarray, float, dict[str, float]]] = []
        for horizon in [int(value) for value in self.cfg["event"]["prediction_horizons_days"]]:
            if not self._horizon_available(index, horizon, family):
                continue
            base_family = "model" if family == "model_fused" else "baseline_adaptive"
            forecast, confidence, diagnostics = self.base._predictive_view(index, horizon, base_family)
            spread = float(np.ptp(forecast))
            alignment_multiplier = 0.50 + 0.50 * max(float(diagnostics["grid_alignment"]), 0.0)
            quality = max(confidence, 0.0) * max(spread, 0.0) * alignment_multiplier
            if quality >= float(self.cfg["event"]["minimum_horizon_quality"]):
                component_rows.append((horizon, forecast, quality, diagnostics))
        if not component_rows:
            return np.zeros(len(MODULES)), {
                "horizon_weights": {},
                "model_reliability": 0.0,
                "baseline_reliability": 0.0,
                "incremental_reliability": 0.0,
                "itransformer_weight": 0.0,
                "grid_alignment": 0.0,
                "forecast_confidence": 0.0,
            }
        raw_quality = np.array([row[2] for row in component_rows], dtype=float)
        temperature = max(float(self.cfg["event"].get("reliability_temperature", 0.05)), 1e-6)
        logits = raw_quality / temperature
        weights = np.exp(logits - float(logits.max()))
        weights /= weights.sum()
        forecast = np.sum(
            np.vstack([row[1] for row in component_rows]) * weights[:, None], axis=0
        )
        diagnostics = {
            key: float(np.dot(weights, [row[3][key] for row in component_rows]))
            for key in [
                "model_reliability",
                "baseline_reliability",
                "incremental_reliability",
                "itransformer_weight",
                "grid_alignment",
            ]
        }
        diagnostics["horizon_weights"] = {
            str(row[0]): float(weight) for row, weight in zip(component_rows, weights)
        }
        diagnostics["forecast_confidence"] = float(np.dot(weights, raw_quality))
        return forecast, diagnostics

    def _adaptive_predictive_view(self, index: int) -> tuple[np.ndarray, dict[str, Any]]:
        components: list[tuple[int, np.ndarray, float, dict[str, float], str]] = []
        for horizon in [int(value) for value in self.cfg["event"]["prediction_horizons_days"]]:
            candidates: list[tuple[str, str]] = []
            if self._horizon_available(index, horizon, "model_fused"):
                candidates.append(("model", "model"))
            if self._horizon_available(index, horizon, "baseline_fused"):
                candidates.append(("baseline", "baseline_adaptive"))
            for source, family in candidates:
                forecast, confidence, diagnostics = self.base._predictive_view(
                    index, horizon, family
                )
                spread = float(np.ptp(forecast))
                alignment_multiplier = 0.50 + 0.50 * max(
                    float(diagnostics["grid_alignment"]), 0.0
                )
                quality = (
                    max(confidence, 0.0)
                    * max(spread, 0.0)
                    * alignment_multiplier
                )
                if quality >= float(self.cfg["event"]["minimum_horizon_quality"]):
                    components.append(
                        (horizon, forecast, quality, diagnostics, source)
                    )
                    break
        if not components:
            return np.zeros(len(MODULES)), {
                "horizon_weights": {},
                "horizon_sources": {},
                "model_horizon_share": 0.0,
                "model_reliability": 0.0,
                "baseline_reliability": 0.0,
                "incremental_reliability": 0.0,
                "itransformer_weight": 0.0,
                "grid_alignment": 0.0,
                "forecast_confidence": 0.0,
            }
        quality = np.array([row[2] for row in components], dtype=float)
        temperature = max(float(self.cfg["event"].get("reliability_temperature", 0.05)), 1e-6)
        logits = quality / temperature
        weights = np.exp(logits - float(logits.max()))
        weights /= weights.sum()
        forecast = np.sum(np.vstack([row[1] for row in components]) * weights[:, None], axis=0)
        diagnostics = {
            key: float(np.dot(weights, [row[3][key] for row in components]))
            for key in [
                "model_reliability",
                "baseline_reliability",
                "incremental_reliability",
                "itransformer_weight",
                "grid_alignment",
            ]
        }
        diagnostics.update(
            {
                "horizon_weights": {
                    str(row[0]): float(weight) for row, weight in zip(components, weights)
                },
                "horizon_sources": {str(row[0]): row[4] for row in components},
                "model_horizon_share": float(
                    sum(weight for row, weight in zip(components, weights) if row[4] == "model")
                ),
                "forecast_confidence": float(np.dot(weights, quality)),
            }
        )
        return forecast, diagnostics

    def candidate_weights(self, index: int, action: int) -> tuple[np.ndarray, dict[str, Any]]:
        key = (int(index), int(action))
        if key in self.candidate_cache:
            weights, diagnostics = self.candidate_cache[key]
            return weights.copy(), copy.deepcopy(diagnostics)
        name = self.action_names[action]
        calendar_position = int(self.calendar_positions[index])
        risk_base = self.base.risk_budget_values[calendar_position]
        if name == "hold":
            raise ValueError("hold target depends on current weights and is not cacheable")
        if name == "formal_anchor":
            target = self.base._apply_guardrails(
                index, self.base.formal_values[calendar_position], risk_base
            )
            diagnostics = {
                "action_name": name,
                "optimizer_backend": "formal_pipeline_anchor",
                "horizon_weights": {},
                "forecast_confidence": 0.0,
                "effective_model_level": 0.0,
                **{f"fused_view_{sleeve}": 0.0 for sleeve in MODULES},
            }
            self.candidate_cache[key] = (target.copy(), copy.deepcopy(diagnostics))
            return target, diagnostics

        row = self.state.loc[index]
        rule = row[[f"state_rule_view_{sleeve}" for sleeve in MODULES]].to_numpy(dtype=float)
        if name in {"baseline_fused", "model_fused", "adaptive_fused"}:
            if name == "adaptive_fused":
                predictive, forecast_diagnostics = self._adaptive_predictive_view(index)
            else:
                predictive, forecast_diagnostics = self._dynamic_predictive_view(index, name)
            forecast_confidence = float(forecast_diagnostics["forecast_confidence"])
        else:
            predictive = np.zeros(len(MODULES))
            forecast_confidence = 0.0
            forecast_diagnostics = {
                "horizon_weights": {},
                "model_reliability": 0.0,
                "baseline_reliability": 0.0,
                "incremental_reliability": 0.0,
                "itransformer_weight": 0.0,
                "grid_alignment": 0.0,
            }
        fused = float(self.base_cfg["views"]["rule_weight"]) * rule
        fused += float(self.base_cfg["views"]["model_weight"]) * forecast_confidence * predictive
        max_abs = float(np.max(np.abs(fused)))
        if max_abs > 1e-12:
            fused /= max_abs
        posterior_mu = pd.Series(
            fused * float(self.base_cfg["views"]["daily_mu_scale"][self.profile]),
            index=MODULES,
        )
        lookback = int(self.base_cfg["views"]["covariance_lookback_days"])
        minimum = int(self.base_cfg["views"]["minimum_covariance_days"])
        history = (
            self.base.sleeve_returns.iloc[: calendar_position + 1]
            .tail(lookback)
            .reindex(columns=MODULES)
            .fillna(0.0)
        )
        if len(history) < minimum:
            target = risk_base.copy()
            optimizer_diagnostics = {
                "optimizer_backend": "risk_budget_fallback",
                "status": "insufficient_history",
            }
            covariance_diagnostics = {}
        else:
            covariance_estimate = self.base.covariance_estimate(history)
            result = optimize_sleeve_weights(
                base_weights=pd.Series(risk_base, index=MODULES),
                posterior_mu=posterior_mu,
                cov=covariance_estimate.covariance,
                returns_window=history,
                prev_weights=pd.Series(risk_base, index=MODULES),
                bounds=self.base.bounds,
                cfg=self.base.v1cfg,
            )
            target = result.weights.reindex(MODULES).to_numpy(dtype=float)
            optimizer_diagnostics = result.diagnostics
            covariance_diagnostics = covariance_estimate.diagnostics
        target = self.base._apply_guardrails(index, target, risk_base)
        diagnostics = {
            "action_name": name,
            "optimizer_backend": optimizer_diagnostics.get("optimizer_backend", "unknown"),
            "forecast_confidence": forecast_confidence,
            "effective_model_level": (
                forecast_confidence * float(forecast_diagnostics.get("model_horizon_share", 1.0))
                if name in {"model_fused", "adaptive_fused"}
                else 0.0
            ),
            **forecast_diagnostics,
            **covariance_diagnostics,
            **{f"fused_view_{sleeve}": float(fused[i]) for i, sleeve in enumerate(MODULES)},
        }
        self.candidate_cache[key] = (target.copy(), copy.deepcopy(diagnostics))
        return target, diagnostics

    def mechanical_action(self, index: int, current_weights: np.ndarray) -> int:
        family = str(self.cfg["event"]["base_family"][self.profile])
        action = self.action_index[family]
        if not self.action_mask(index)[action]:
            if "formal_anchor" in self.action_index and self.action_mask(index)[
                self.action_index["formal_anchor"]
            ]:
                action = self.action_index["formal_anchor"]
            else:
                return self.action_index["hold"]
        target, _ = self.candidate_weights(index, action)
        distance = float(np.abs(target - np.asarray(current_weights, dtype=float)).sum())
        threshold = float(self.cfg["event"]["mechanical_no_trade_l1"][self.profile])
        return action if distance >= threshold else self.action_index["hold"]

    def observation(
        self,
        index: int,
        current_weights: np.ndarray,
        context: HoldingContext,
        mean: np.ndarray,
        std: np.ndarray,
    ) -> np.ndarray:
        static = self.state.loc[index, self.state_cols].to_numpy(dtype=float)
        scaled = np.clip((static - mean) / std, -6.0, 6.0)
        calendar_position = int(self.calendar_positions[index])
        risk_base = self.base.risk_budget_values[calendar_position]
        active = np.clip((np.asarray(current_weights) - risk_base) / 0.05, -3.0, 3.0)
        strategy_drawdown = context.nav / max(context.peak, 1e-12) - 1.0
        since_return = context.since_trade_nav - 1.0
        since_formal = context.since_trade_formal_nav - 1.0
        dynamic = [
            math.log1p(context.age) / math.log(253.0),
            float(np.clip(since_return / 0.05, -3.0, 3.0)),
            float(np.clip((since_return - since_formal) / 0.05, -3.0, 3.0)),
            float(np.clip(strategy_drawdown / 0.10, -3.0, 0.0)),
            float(np.clip(math.sqrt(max(context.ewma_variance, 0.0) * 252.0) / 0.20, 0.0, 3.0)),
            float(np.clip(math.sqrt(max(context.formal_ewma_variance, 0.0) * 252.0) / 0.20, 0.0, 3.0)),
        ]
        dynamic.extend(float(context.last_action == action) for action in range(len(self.action_names)))
        mask = self.action_mask(index)
        for action in range(1, len(self.action_names)):
            if mask[action]:
                target, _ = self.candidate_weights(index, action)
                distance = float(np.abs(target - np.asarray(current_weights)).sum())
            else:
                distance = 0.0
            dynamic.append(float(np.clip(distance / 0.05, 0.0, 3.0)))
        dynamic.extend(float(value) for value in mask[1:])
        return np.concatenate([scaled, active, np.asarray(dynamic, dtype=float)]).astype(np.float32)

    def step(
        self,
        index: int,
        action: int,
        previous_weights: np.ndarray,
        context: HoldingContext,
    ) -> tuple[float, int, np.ndarray, HoldingContext, dict[str, Any]]:
        signal_position = int(self.calendar_positions[index])
        start = signal_position + 1
        if start >= len(self.calendar):
            return 0.0, len(self.state), previous_weights.copy(), context, {"daily": []}
        name = self.action_names[action]
        risk_base = self.base.risk_budget_values[signal_position]
        if name == "hold":
            if bool(self.cfg["event"].get("force_guardrails_on_hold", True)):
                target = self.base._apply_guardrails(index, np.asarray(previous_weights), risk_base)
            else:
                target = np.asarray(previous_weights, dtype=float).copy()
            diagnostics: dict[str, Any] = {
                "action_name": name,
                "optimizer_backend": "hold_no_trade_region",
                "horizon_weights": {},
                "forecast_confidence": 0.0,
                "effective_model_level": 0.0,
                **{f"fused_view_{sleeve}": 0.0 for sleeve in MODULES},
            }
        else:
            target, diagnostics = self.candidate_weights(index, action)

        turnover = float(np.abs(target - np.asarray(previous_weights, dtype=float)).sum())
        cost = turnover * self.transaction_cost
        returns = self.base.return_values[start]
        strategy_return = float(np.dot(target, returns) - cost)
        formal_weights = self.base.formal_values[start]
        formal_return = float(np.dot(formal_weights, returns) - self.base.formal_cost[start])

        beta = float(self.cfg["reward"]["ewma_beta"])
        new_variance = beta * context.ewma_variance + (1.0 - beta) * strategy_return**2
        new_formal_variance = beta * context.formal_ewma_variance + (1.0 - beta) * formal_return**2
        new_nav = context.nav * (1.0 + strategy_return)
        new_formal_nav = context.formal_nav * (1.0 + formal_return)
        new_peak = max(context.peak, new_nav)
        new_formal_peak = max(context.formal_peak, new_formal_nav)
        new_drawdown = new_nav / max(new_peak, 1e-12) - 1.0
        new_formal_drawdown = new_formal_nav / max(new_formal_peak, 1e-12) - 1.0
        new_gap = max(0.0, abs(new_drawdown) - abs(new_formal_drawdown))
        drawdown_worsening = max(0.0, new_gap - context.drawdown_gap)
        volatility_worsening = max(
            0.0,
            math.sqrt(max(new_variance, 0.0) * 252.0)
            - math.sqrt(max(new_formal_variance, 0.0) * 252.0),
        ) * (1.0 - beta)
        downside_worsening = max(0.0, -strategy_return) - max(0.0, -formal_return)
        active_deviation = float(np.abs(target - formal_weights).sum())
        reward = (
            strategy_return
            - formal_return
            - float(self.cfg["reward"]["downside_penalty"]) * max(0.0, downside_worsening)
            - float(self.cfg["reward"]["relative_volatility_penalty"][self.profile]) * volatility_worsening
            - float(self.cfg["reward"]["relative_drawdown_penalty"][self.profile]) * drawdown_worsening
            - float(self.cfg["reward"]["active_deviation_penalty"]) * active_deviation
        ) * float(self.cfg["reward"]["reward_scale"])

        rebalanced = turnover > float(self.cfg["event"]["turnover_epsilon"])
        if rebalanced:
            since_nav = 1.0 + strategy_return
            since_formal_nav = 1.0 + formal_return
            age = 0
            last_action = action
        else:
            since_nav = context.since_trade_nav * (1.0 + strategy_return)
            since_formal_nav = context.since_trade_formal_nav * (1.0 + formal_return)
            age = context.age + 1
            last_action = context.last_action
        new_context = HoldingContext(
            age=age,
            last_action=last_action,
            nav=new_nav,
            formal_nav=new_formal_nav,
            peak=new_peak,
            formal_peak=new_formal_peak,
            since_trade_nav=since_nav,
            since_trade_formal_nav=since_formal_nav,
            ewma_variance=new_variance,
            formal_ewma_variance=new_formal_variance,
            drawdown_gap=new_gap,
        )
        next_index = int(np.searchsorted(self.calendar_positions, start, side="left"))
        next_index = max(index + 1, next_index)
        daily = {
            "date": self.calendar[start],
            "signal_date": pd.Timestamp(self.state.loc[index, "date"]),
            "formal_return": formal_return,
            "strategy_return": strategy_return,
            "strategy_turnover": turnover,
            "strategy_cost": cost,
            **{f"weight_{sleeve}": float(target[i]) for i, sleeve in enumerate(MODULES)},
        }
        info = {
            "daily": [daily],
            "action_name": name,
            "execution_start": self.calendar[start],
            "execution_end": self.calendar[start],
            "turnover": turnover,
            "cost": cost,
            "rebalanced": rebalanced,
            "forced_guardrail": bool(name == "hold" and rebalanced),
            "hold_age_before": context.age,
            "hold_age_after": age,
            "active_deviation": active_deviation,
            "volatility_worsening": volatility_worsening,
            "drawdown_worsening": drawdown_worsening,
            **diagnostics,
        }
        return float(reward), next_index, target.copy(), new_context, info


def _simulate_policy(
    env: EventDrivenEnvironment,
    start: int,
    stop: int,
    model: HierarchicalActorCritic,
    mean: np.ndarray,
    std: np.ndarray,
) -> float:
    index = start
    current = env.initial_weights(index)
    context = env.initial_context()
    total = 0.0
    with torch.no_grad():
        while index < stop:
            obs = torch.tensor(env.observation(index, current, context, mean, std))[None, :]
            logits, _ = model(obs)
            mask = torch.tensor(env.action_mask(index), dtype=torch.bool)[None, :]
            action = int(logits.masked_fill(~mask, -1e9).argmax(dim=1).item())
            reward, next_index, current, context, _ = env.step(index, action, current, context)
            total += reward
            index = max(index + 1, next_index)
    return float(total)


def _simulate_mechanical_reward(env: EventDrivenEnvironment, start: int, stop: int) -> float:
    index = start
    current = env.initial_weights(index)
    context = env.initial_context()
    total = 0.0
    while index < stop:
        action = env.mechanical_action(index, current)
        reward, next_index, current, context, _ = env.step(index, action, current, context)
        total += reward
        index = max(index + 1, next_index)
    return float(total)


def train_seed(
    env: EventDrivenEnvironment,
    eligible: np.ndarray,
    year: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    cfg = env.cfg["ppo"]
    n_val = max(30, int(math.ceil(len(eligible) * float(cfg["validation_fraction"]))))
    split = max(30, len(eligible) - n_val)
    train_rows = eligible[:split]
    val_rows = eligible[split:]
    static = env.state.loc[train_rows, env.state_cols].to_numpy(dtype=float)
    mean = np.nanmean(static, axis=0)
    std = np.nanstd(static, axis=0)
    std = np.where(std > 1e-8, std, 1.0)

    current = env.initial_weights(int(train_rows[0]))
    context = env.initial_context()
    observations: list[np.ndarray] = []
    teacher_actions: list[int] = []
    teacher_masks: list[np.ndarray] = []
    stride = max(int(cfg.get("teacher_stride", 1)), 1)
    for position, index in enumerate(train_rows):
        index = int(index)
        mask = env.action_mask(index)
        if position % stride == 0:
            observation = env.observation(index, current, context, mean, std)
            scores = np.full(len(env.action_names), -np.inf, dtype=float)
            for action in np.flatnonzero(mask):
                score, _, _, _, _ = env.step(index, int(action), current, context)
                scores[action] = score
            hold_score = float(scores[env.action_index["hold"]])
            best_action = int(np.argmax(scores))
            if best_action != env.action_index["hold"] and scores[best_action] - hold_score < float(
                cfg["minimum_teacher_edge"]
            ):
                best_action = env.action_index["hold"]
            observations.append(observation)
            teacher_actions.append(best_action)
            teacher_masks.append(mask)
            selected = best_action
        else:
            selected = env.mechanical_action(index, current)
        _, _, current, context, _ = env.step(index, selected, current, context)

    obs_dim = len(observations[0])
    model = HierarchicalActorCritic(obs_dim, len(env.action_names), int(cfg["hidden_dim"]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=1e-4)
    bc_obs = torch.tensor(np.vstack(observations), dtype=torch.float32)
    bc_mask = torch.tensor(np.vstack(teacher_masks), dtype=torch.bool)
    counts = np.bincount(teacher_actions, minlength=len(env.action_names)).astype(float)
    power = float(cfg["behavior_clone_class_balance_power"])

    def balanced_weights(targets: np.ndarray, classes: int) -> torch.Tensor:
        local_counts = np.bincount(targets, minlength=classes).astype(float)
        present = local_counts > 0
        result = np.zeros(classes, dtype=float)
        result[present] = np.power(local_counts[present].sum() / local_counts[present], power)
        result[present] /= result[present].mean()
        return torch.tensor(np.clip(result, 0.0, 6.0), dtype=torch.float32)

    termination_target_np = (np.asarray(teacher_actions) != env.action_index["hold"]).astype(int)
    termination_target = torch.tensor(termination_target_np, dtype=torch.long)
    termination_weights = balanced_weights(termination_target_np, 2)
    rebalance_rows = np.flatnonzero(termination_target_np)
    if len(rebalance_rows) and len(env.action_names) > 2:
        family_target_np = np.asarray(teacher_actions, dtype=int)[rebalance_rows] - 1
        family_target = torch.tensor(family_target_np, dtype=torch.long)
        family_weights = balanced_weights(family_target_np, len(env.action_names) - 1)
    teacher_distribution = {
        env.action_names[action]: int(count) for action, count in enumerate(counts) if count > 0
    }
    print(
        f"[{env.profile}][{year}][seed={seed}] teacher_distribution="
        f"{json.dumps(teacher_distribution, sort_keys=True)}",
        flush=True,
    )
    for _ in range(int(cfg["behavior_clone_epochs"])):
        termination_logits, family_logits, values = model.heads(bc_obs)
        termination_loss = nn.functional.cross_entropy(
            termination_logits, termination_target, weight=termination_weights
        )
        if len(rebalance_rows) and len(env.action_names) > 2:
            selected_family_logits = family_logits[rebalance_rows].masked_fill(
                ~bc_mask[rebalance_rows, 1:], -1e9
            )
            family_loss = nn.functional.cross_entropy(
                selected_family_logits, family_target, weight=family_weights
            )
        else:
            family_loss = torch.tensor(0.0)
        loss = termination_loss + 0.75 * family_loss + 0.02 * torch.mean(values**2)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    train_start, train_stop = int(train_rows[0]), int(train_rows[-1]) + 1
    val_start, val_stop = int(val_rows[0]), int(val_rows[-1]) + 1
    best_state = copy.deepcopy(model.state_dict())
    best_val = _simulate_policy(env, val_start, val_stop, model, mean, std)
    fallback_val = _simulate_mechanical_reward(env, val_start, val_stop)
    wait = 0
    logs: list[dict[str, Any]] = []
    for epoch in range(1, int(cfg["max_epochs"]) + 1):
        index = train_start
        current = env.initial_weights(index)
        context = env.initial_context()
        obs_rows: list[np.ndarray] = []
        actions: list[int] = []
        rewards: list[float] = []
        old_logp: list[float] = []
        values: list[float] = []
        masks: list[np.ndarray] = []
        while index < train_stop:
            obs_array = env.observation(index, current, context, mean, std)
            obs = torch.tensor(obs_array)[None, :]
            mask_array = env.action_mask(index)
            with torch.no_grad():
                logits, value = model(obs)
                mask = torch.tensor(mask_array, dtype=torch.bool)[None, :]
                distribution = torch.distributions.Categorical(logits=logits.masked_fill(~mask, -1e9))
                action = int(distribution.sample().item())
                logp = float(distribution.log_prob(torch.tensor([action])).item())
            reward, next_index, current, context, _ = env.step(index, action, current, context)
            obs_rows.append(obs_array)
            actions.append(action)
            rewards.append(reward)
            old_logp.append(logp)
            values.append(float(value.item()))
            masks.append(mask_array)
            index = max(index + 1, next_index)

        obs_tensor = torch.tensor(np.vstack(obs_rows), dtype=torch.float32)
        action_tensor = torch.tensor(actions, dtype=torch.long)
        old_logp_tensor = torch.tensor(old_logp, dtype=torch.float32)
        mask_tensor = torch.tensor(np.vstack(masks), dtype=torch.bool)
        rewards_array = np.asarray(rewards, dtype=float)
        values_array = np.asarray(values, dtype=float)
        advantage = np.zeros(len(rewards), dtype=float)
        last = 0.0
        next_value = 0.0
        gamma = float(cfg["gamma"])
        for position in reversed(range(len(rewards))):
            delta = rewards_array[position] + gamma * next_value - values_array[position]
            last = delta + gamma * float(cfg["gae_lambda"]) * last
            advantage[position] = last
            next_value = values_array[position]
        target = advantage + values_array
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        advantage_tensor = torch.tensor(advantage, dtype=torch.float32)
        target_tensor = torch.tensor(target, dtype=torch.float32)
        for _ in range(int(cfg["updates_per_epoch"])):
            logits, value = model(obs_tensor)
            distribution = torch.distributions.Categorical(logits=logits.masked_fill(~mask_tensor, -1e9))
            logp = distribution.log_prob(action_tensor)
            ratio = torch.exp(logp - old_logp_tensor)
            clipped = torch.clamp(ratio, 1.0 - float(cfg["clip_ratio"]), 1.0 + float(cfg["clip_ratio"]))
            policy_loss = -torch.minimum(ratio * advantage_tensor, clipped * advantage_tensor).mean()
            value_loss = torch.mean((value - target_tensor) ** 2)
            entropy = distribution.entropy().mean()
            loss = policy_loss + float(cfg["value_loss_weight"]) * value_loss - float(cfg["entropy_weight"]) * entropy
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        if epoch == 1 or epoch % int(cfg["evaluation_every"]) == 0:
            validation = _simulate_policy(env, val_start, val_stop, model, mean, std)
            logs.append(
                {
                    "profile": env.profile,
                    "evaluation_year": year,
                    "seed": seed,
                    "epoch": epoch,
                    "rollout_steps": len(rewards),
                    "rollout_reward": float(rewards_array.sum()),
                    "validation_reward": validation,
                    "fallback_validation_reward": fallback_val,
                    "entropy": float(entropy.detach()),
                }
            )
            print(
                f"[{env.profile}][{year}][seed={seed}] epoch={epoch} "
                f"rollout={rewards_array.sum():.6f} validation={validation:.6f} fallback={fallback_val:.6f}",
                flush=True,
            )
            if validation > best_val + float(cfg["early_stopping_min_delta"]):
                best_val = validation
                best_state = copy.deepcopy(model.state_dict())
                wait = 0
            else:
                wait += 1
                if wait >= int(cfg["early_stop_patience"]):
                    break
    max_label_end = pd.to_datetime(env.state.loc[eligible, "max_horizon_end_date"], errors="coerce").max()
    return {
        "state_dict": best_state,
        "mean": mean,
        "std": std,
        "obs_dim": obs_dim,
        "action_dim": len(env.action_names),
        "hidden_dim": int(cfg["hidden_dim"]),
        "training_cutoff": pd.Timestamp(year, 1, 1),
        "training_max_label_end": max_label_end,
        "validation_reward": best_val,
        "fallback_validation_reward": fallback_val,
        "teacher_distribution": teacher_distribution,
    }, logs


def choose_action(
    models: list[dict[str, Any]],
    env: EventDrivenEnvironment,
    index: int,
    current: np.ndarray,
    context: HoldingContext,
) -> tuple[int, int, str]:
    fallback = env.mechanical_action(index, current)
    cfg = env.cfg["ppo"]
    qualified: list[dict[str, Any]] = []
    for checkpoint in models:
        required = float(checkpoint["fallback_validation_reward"]) + float(
            cfg["minimum_validation_edge_vs_fallback"]
        )
        absolute_floor = cfg.get("minimum_policy_validation_reward")
        if absolute_floor is not None:
            required = max(required, float(absolute_floor))
        if float(checkpoint["validation_reward"]) >= required:
            qualified.append(checkpoint)
    if len(qualified) < int(cfg["minimum_qualified_models"]):
        return fallback, len(qualified), "mechanical_fallback"
    mask = env.action_mask(index)
    probabilities: list[np.ndarray] = []
    votes: list[int] = []
    for checkpoint in qualified:
        model = HierarchicalActorCritic(
            checkpoint["obs_dim"], checkpoint["action_dim"], checkpoint["hidden_dim"]
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        obs = env.observation(index, current, context, checkpoint["mean"], checkpoint["std"])
        with torch.no_grad():
            logits, _ = model(torch.tensor(obs)[None, :])
            probs = torch.softmax(
                logits.masked_fill(~torch.tensor(mask, dtype=torch.bool)[None, :], -1e9), dim=1
            ).squeeze(0).numpy()
        probabilities.append(probs)
        votes.append(int(np.argmax(probs)))
    counts = pd.Series(votes).value_counts()
    if int(counts.iloc[0]) < int(cfg["ensemble_min_agreement"]):
        return fallback, len(qualified), "ensemble_disagreement"
    selected = int(counts.index[0])
    mean_probability = np.mean(np.vstack(probabilities), axis=0)
    competitors = np.delete(mean_probability, selected)
    if (
        float(mean_probability[selected]) < float(cfg["minimum_action_probability"])
        or float(mean_probability[selected]) - float(competitors.max())
        < float(cfg["minimum_probability_margin"])
    ):
        return fallback, len(qualified), "probability_fallback"
    return selected, len(qualified), "event_ppo"


def _simulate_mechanical_daily(env: EventDrivenEnvironment) -> pd.DataFrame:
    index = 0
    current = env.initial_weights(index)
    context = env.initial_context()
    rows: list[dict[str, Any]] = []
    while index < len(env.state):
        action = env.mechanical_action(index, current)
        _, next_index, current, context, info = env.step(index, action, current, context)
        rows.extend(info.get("daily", []))
        index = max(index + 1, next_index)
    daily = pd.DataFrame(rows).drop_duplicates("date", keep="last")
    return daily[["date", "strategy_return", "strategy_turnover"]].rename(
        columns={
            "strategy_return": "mechanical_return",
            "strategy_turnover": "mechanical_turnover",
        }
    )


def run_profile(
    profile: str,
    state: pd.DataFrame,
    cfg: dict[str, Any],
    base_cfg: dict[str, Any],
    train_models: bool = True,
) -> dict[str, Any]:
    env = EventDrivenEnvironment(profile, state, cfg, base_cfg)
    model_dir = output_path(cfg, "model_dir") / profile
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoints: dict[int, list[dict[str, Any]]] = {}
    training_logs: list[dict[str, Any]] = []
    years = sorted(pd.to_datetime(env.state["date"]).dt.year.unique())
    for year in years:
        cutoff = pd.Timestamp(int(year), 1, 1)
        eligible = env.eligible_rows(cutoff)
        if len(eligible) < int(cfg["ppo"]["min_train_rows"]):
            checkpoints[int(year)] = []
            continue
        checkpoint_path = model_dir / f"policy_{int(year)}.pt"
        if not train_models:
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"missing event-driven checkpoint: {checkpoint_path}")
            checkpoints[int(year)] = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
            continue
        annual_models: list[dict[str, Any]] = []
        for seed in cfg["ppo"]["seeds"]:
            started = time.perf_counter()
            print(
                f"[{profile}][{int(year)}][seed={int(seed)}] event_training_start "
                f"eligible_rows={len(eligible)}",
                flush=True,
            )
            checkpoint, logs = train_seed(env, eligible, int(year), int(seed))
            annual_models.append(checkpoint)
            training_logs.extend(logs)
            print(
                f"[{profile}][{int(year)}][seed={int(seed)}] event_training_done "
                f"validation={checkpoint['validation_reward']:.6f} elapsed_seconds={time.perf_counter()-started:.1f}",
                flush=True,
            )
        checkpoints[int(year)] = annual_models
        torch.save(annual_models, checkpoint_path)

    index = 0
    current = env.initial_weights(index)
    context = env.initial_context()
    daily_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    while index < len(env.state):
        date = pd.Timestamp(env.state.loc[index, "date"])
        models = checkpoints.get(int(date.year), [])
        action, qualified, source = choose_action(models, env, index, current, context)
        reward, next_index, new_weights, new_context, info = env.step(index, action, current, context)
        if not info.get("daily"):
            break
        checkpoint_cutoffs = [pd.Timestamp(model["training_cutoff"]) for model in models]
        training_label_ends = [pd.Timestamp(model["training_max_label_end"]) for model in models]
        decision = {
            "date": date,
            "profile": profile,
            "stock_bond_regime": env.state.loc[index, "stock_bond_regime"],
            "action_index": action,
            "action_name": env.action_names[action],
            "policy_source": source,
            "qualified_model_count": qualified,
            "policy_training_cutoff": max(checkpoint_cutoffs) if checkpoint_cutoffs else pd.NaT,
            "training_max_label_end": max(training_label_ends) if training_label_ends else pd.NaT,
            "itransformer_information_cutoff": env.state.loc[index, "itransformer_information_cutoff"],
            "execution_start": info["execution_start"],
            "execution_end": info["execution_end"],
            "reward": reward,
            "turnover": info["turnover"],
            "cost": info["cost"],
            "rebalanced": info["rebalanced"],
            "forced_guardrail": info["forced_guardrail"],
            "hold_age_before": info["hold_age_before"],
            "hold_age_after": info["hold_age_after"],
            "horizon_weights": json.dumps(info.get("horizon_weights", {}), sort_keys=True),
            "forecast_confidence": info.get("forecast_confidence", 0.0),
            "effective_model_level": info.get("effective_model_level", 0.0),
            "optimizer_backend": info.get("optimizer_backend", "unknown"),
        }
        for position, sleeve in enumerate(MODULES):
            decision[f"target_{sleeve}"] = float(new_weights[position])
            decision[f"fused_view_{sleeve}"] = float(info.get(f"fused_view_{sleeve}", 0.0))
        decision_rows.append(decision)
        daily_rows.extend(info["daily"])
        current, context = new_weights, new_context
        index = max(index + 1, next_index)

    daily = pd.DataFrame(daily_rows).drop_duplicates("date", keep="last").sort_values("date")
    decisions = pd.DataFrame(decision_rows).sort_values("date")
    mechanical = _simulate_mechanical_daily(env)
    v6 = pd.read_csv(
        ROOT / f"output/reports/integrated_daily_view_ppo_{profile}_daily.csv",
        parse_dates=["date"],
    )[["date", "rule_only_return", "strategy_return"]].rename(columns={"strategy_return": "v6_return"})
    daily["date"] = pd.to_datetime(daily["date"])
    mechanical["date"] = pd.to_datetime(mechanical["date"])
    daily = daily.merge(mechanical, on="date", how="inner").merge(v6, on="date", how="inner")
    for name in ["formal", "rule_only", "v6", "mechanical", "strategy"]:
        daily[f"{name}_nav"] = (1.0 + daily[f"{name}_return"]).cumprod()

    metrics_rows: list[dict[str, Any]] = []
    for split_name, mask in {
        "full_oos": pd.Series(True, index=daily.index),
        "oos_2024_onward": daily["date"].ge(pd.Timestamp(cfg["gate"]["evaluation_start"])),
    }.items():
        stats = {
            name: perf_stats(daily.loc[mask, f"{name}_return"])
            for name in ["formal", "rule_only", "v6", "mechanical", "strategy"]
        }
        row: dict[str, Any] = {"profile": profile, "split": split_name}
        for name, values in stats.items():
            row.update({f"{name}_{key}": value for key, value in values.items()})
        row.update(
            {
                "delta_annual_return_vs_v6": stats["strategy"]["annual_return"] - stats["v6"]["annual_return"],
                "delta_sharpe_vs_v6": stats["strategy"]["sharpe"] - stats["v6"]["sharpe"],
                "max_drawdown_worsening_vs_v6": abs(stats["strategy"]["max_drawdown"])
                - abs(stats["v6"]["max_drawdown"]),
                "delta_sharpe_vs_mechanical": stats["strategy"]["sharpe"] - stats["mechanical"]["sharpe"],
                "mean_strategy_turnover": float(daily.loc[mask, "strategy_turnover"].mean()),
            }
        )
        metrics_rows.append(row)
    metrics = pd.DataFrame(metrics_rows)
    daily.to_csv(output_path(cfg, "daily_pattern", profile), index=False, encoding="utf-8-sig")
    decisions.to_parquet(output_path(cfg, "decisions_pattern", profile), index=False)
    metrics.to_csv(output_path(cfg, "metrics_pattern", profile), index=False, encoding="utf-8-sig")
    if train_models:
        pd.DataFrame(training_logs).to_csv(
            output_path(cfg, "training_log_pattern", profile), index=False, encoding="utf-8-sig"
        )
    holdings = _materialize_asset_weights(profile, daily, base_cfg)
    holdings.to_parquet(output_path(cfg, "holdings_pattern", profile), index=False)
    return {
        "profile": profile,
        "daily_rows": len(daily),
        "decision_rows": len(decisions),
        "rebalances": int(decisions["rebalanced"].sum()),
        "ppo_actions": int(decisions["policy_source"].eq("event_ppo").sum()),
    }


def audit_and_report(cfg: dict[str, Any], base_cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    summaries: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    intervals: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    for profile in cfg["profiles"]:
        metrics = pd.read_csv(output_path(cfg, "metrics_pattern", profile))
        summaries.append(metrics)
        daily = pd.read_csv(output_path(cfg, "daily_pattern", profile), parse_dates=["date", "signal_date"])
        decisions = pd.read_parquet(output_path(cfg, "decisions_pattern", profile))
        for column in [
            "date",
            "execution_start",
            "policy_training_cutoff",
            "training_max_label_end",
            "itransformer_information_cutoff",
        ]:
            decisions[column] = pd.to_datetime(decisions[column], errors="coerce")
        weight_columns = [f"weight_{sleeve}" for sleeve in MODULES]
        checks = {
            "daily_dates_unique": bool(daily["date"].is_unique),
            "signal_precedes_execution": bool((decisions["execution_start"] > decisions["date"]).all()),
            "execution_is_one_day": bool((decisions["execution_start"] == decisions["execution_end"]).all()),
            "weights_sum_to_one": bool(np.allclose(daily[weight_columns].sum(axis=1), 1.0, atol=1e-8)),
            "returns_finite": bool(np.isfinite(daily[["formal_return", "rule_only_return", "v6_return", "strategy_return"]]).all().all()),
            "training_cutoff_precedes_decision": bool(
                (
                    decisions["policy_training_cutoff"].isna()
                    | (decisions["policy_training_cutoff"] <= decisions["date"])
                ).all()
            ),
            "training_labels_precede_cutoff": bool(
                (
                    decisions["training_max_label_end"].isna()
                    | (decisions["training_max_label_end"] < decisions["policy_training_cutoff"])
                ).all()
            ),
            "information_cutoff_not_future": bool(
                (decisions["itransformer_information_cutoff"] <= decisions["date"]).all()
            ),
            "same_daily_sample_as_v6": bool(len(daily) == 1035),
        }
        audits.extend(
            {"profile": profile, "check": check, "passed": passed}
            for check, passed in checks.items()
        )
        rebalanced = decisions.index[decisions["rebalanced"].astype(bool)].to_numpy(dtype=int)
        gaps = np.diff(rebalanced) if len(rebalanced) > 1 else np.array([], dtype=int)
        intervals.append(
            {
                "profile": profile,
                "decision_days": len(decisions),
                "rebalance_count": len(rebalanced),
                "rebalance_rate": float(len(rebalanced) / max(len(decisions), 1)),
                "mean_interval_trading_days": float(gaps.mean()) if len(gaps) else np.nan,
                "median_interval_trading_days": float(np.median(gaps)) if len(gaps) else np.nan,
                "min_interval_trading_days": int(gaps.min()) if len(gaps) else np.nan,
                "max_interval_trading_days": int(gaps.max()) if len(gaps) else np.nan,
                "unique_interval_count": int(len(np.unique(gaps))) if len(gaps) else 0,
            }
        )
        for action, count in decisions["action_name"].value_counts().items():
            action_rows.append({"profile": profile, "action": action, "count": int(count)})

    summary = pd.concat(summaries, ignore_index=True)
    audit = pd.DataFrame(audits)
    interval_summary = pd.DataFrame(intervals)
    evaluation = summary.loc[summary["split"].eq("oos_2024_onward")].set_index("profile")
    profile_gates: dict[str, Any] = {}
    for profile, row in evaluation.iterrows():
        profile_gates[profile] = {
            "annual_return_vs_v6": bool(row["delta_annual_return_vs_v6"] >= float(cfg["gate"]["min_annual_return_delta_vs_v6"])),
            "sharpe_vs_v6": bool(row["delta_sharpe_vs_v6"] >= float(cfg["gate"]["min_sharpe_delta_vs_v6"])),
            "drawdown_vs_v6": bool(
                row["max_drawdown_worsening_vs_v6"] <= float(cfg["gate"]["max_drawdown_worsening_vs_v6"])
            ),
            "variable_intervals": bool(
                interval_summary.set_index("profile").loc[profile, "unique_interval_count"] > 1
            ),
        }
    gate = {
        "enabled_for_formal_strategy": bool(
            audit["passed"].all()
            and all(all(values.values()) for values in profile_gates.values())
        ),
        "profiles": profile_gates,
        "caveat": "2024+ has already been observed; this is a research comparison, not a new untouched holdout.",
    }
    summary.to_csv(output_path(cfg, "summary"), index=False, encoding="utf-8-sig")
    interval_summary.to_csv(output_path(cfg, "interval_summary"), index=False, encoding="utf-8-sig")
    audit.to_csv(output_path(cfg, "audit_csv"), index=False, encoding="utf-8-sig")
    output_path(cfg, "gate_report").write_text(json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8")

    display_columns = [
        "profile",
        "strategy_annual_return",
        "strategy_annual_vol",
        "strategy_sharpe",
        "strategy_max_drawdown",
        "delta_annual_return_vs_v6",
        "delta_sharpe_vs_v6",
        "delta_sharpe_vs_mechanical",
    ]
    display = evaluation.reset_index()[display_columns]
    lines = [
        "# 事件驱动自适应调仓 iTransformer + PPO 验证报告",
        "",
        "## 1. 核心变化",
        "",
        f"本版本不再让动作携带 1/5/20/60 日固定持有期。系统每天观察持仓路径并在 `{', '.join(cfg['event']['action_names'])}` 中决策；一旦选择 hold，仓位持续到下一次由状态触发的重配。",
        "",
        "iTransformer 的 5/20/60 日输出仍作为多尺度预测证据，但由当日历史可靠度、横截面分散度和九宫格一致性动态融合。它们不再控制下一次调仓日期。统一目标版本中，EP/skfolio 每天先生成唯一的 iTransformer 增强目标，PPO 只学习是否终止旧仓。",
        "",
        "每日状态额外包含持有年龄、持有期累计收益及相对正式收益、当前回撤、EWMA 波动、上次动作、四个候选目标距离和候选可用性。交易成本直接进入每日奖励，使策略可以学习无交易区间。",
        "",
        "理论依据是 options/SMDP 的可中断时间抽象与交易成本下的 no-trade region；本实现仍保留风险预算、九宫格、EP/skfolio、产品上下限、无现金和 T+1 执行。",
        "",
        "## 2. 2024+ 结果",
        "",
        display.to_markdown(index=False),
        "",
        "## 3. 自适应调仓间隔",
        "",
        interval_summary.to_markdown(index=False),
        "",
        "动作没有最大持有期；间隔完全由每日 hold/rebalance 决策产生。`unique_interval_count > 1` 只是验证结果确实不是固定周期。",
        "",
        "## 4. 动作分布",
        "",
        pd.DataFrame(action_rows).to_markdown(index=False),
        "",
        "## 5. Point-in-Time 审计",
        "",
        audit.to_markdown(index=False),
        "",
        "## 6. 正式门禁",
        "",
        f"- 是否替换 v6：`{gate['enabled_for_formal_strategy']}`",
        f"- 详细门禁：`{cfg['outputs']['gate_report']}`",
        "- 2024+ 已被用于既有研发，结果只能作为结构验证；即使全面改善，也需要新增日期纸面跟踪后才能正式替换。",
        "",
        "## 7. 运行",
        "",
        "```powershell",
        "python scripts\\132_run_event_driven_rebalance_ppo.py",
        "```",
    ]
    output_path(cfg, "report_md").write_text("\n".join(lines), encoding="utf-8")

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    for axis, profile in zip(axes, cfg["profiles"]):
        daily = pd.read_csv(output_path(cfg, "daily_pattern", profile), parse_dates=["date"])
        for name, label in [
            ("formal_nav", "Formal"),
            ("rule_only_nav", "Rules"),
            ("v6_nav", "V6 fixed-duration"),
            ("strategy_nav", "Event-driven"),
        ]:
            axis.plot(daily["date"], daily[name], label=label, linewidth=1.4)
        axis.set_title(profile)
        axis.grid(alpha=0.2)
        axis.legend(ncol=4, fontsize=8)
    fig.suptitle("Event-driven adaptive rebalancing vs controls")
    fig.tight_layout()
    fig.savefig(output_path(cfg, "visual_png"), dpi=160, bbox_inches="tight")
    plt.close(fig)
    return summary, audit, gate


def run_pipeline(
    path: Path | None, profiles: list[str], train_models: bool = True
) -> dict[str, Any]:
    cfg, base_cfg = load_config(path)
    cfg["profiles"] = profiles
    state = build_state_frame(base_cfg)
    print(
        json.dumps(
            {
                "stage": "event_state_ready",
                "rows": len(state),
                "state_features": len([column for column in state if column.startswith("state_")]),
                "start": str(pd.Timestamp(state["date"].min()).date()),
                "end": str(pd.Timestamp(state["date"].max()).date()),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    results = []
    for position, profile in enumerate(profiles, 1):
        print(f"[{position}/{len(profiles)}] event-driven PPO {profile}", flush=True)
        results.append(run_profile(profile, state, cfg, base_cfg, train_models=train_models))
    summary, audit, gate = audit_and_report(cfg, base_cfg)
    return {
        "results": results,
        "summary_rows": len(summary),
        "audit_failures": int((~audit["passed"]).sum()),
        "formal_gate": gate["enabled_for_formal_strategy"],
        "report": str(output_path(cfg, "report_md")),
    }

from __future__ import annotations

import copy
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

from src.adaptive_daily_ppo import perf_stats
from src.integrated_daily_view_ppo import (
    MODULES,
    ROOT,
    build_environment,
    load_card_states,
    load_config as load_base_config,
    load_prediction_states,
    load_rule_view_states,
)


DEFAULT_CONFIG = ROOT / "config" / "robust_execution_ppo_v1_config.yaml"
FORBIDDEN_STATE_TOKENS = ("realized", "label", "forward", "future", "next_return")


def load_config(path: Path | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    config_path = path or DEFAULT_CONFIG
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    base_path = ROOT / cfg.get("base_config", "config/integrated_daily_view_ppo_config.yaml")
    base_cfg = copy.deepcopy(load_base_config(base_path))
    base_cfg["data"]["predictions"] = cfg["data"]["predictions"]
    base_cfg["data"]["start_date"] = cfg["data"]["start_date"]
    base_cfg["reward"]["transaction_cost_bps"] = cfg["reward"]["transaction_cost_bps"]
    return cfg, base_cfg


def output_path(cfg: dict[str, Any], key: str, profile: str | None = None) -> Path:
    value = str(cfg["outputs"][key])
    if profile is not None:
        value = value.format(profile=profile)
    path = ROOT / value
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _safe_cross_section(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).all():
        return np.zeros_like(array)
    centered = array - float(array.mean())
    scale = float(centered.std(ddof=0))
    return np.clip(centered / scale, -3.0, 3.0) if scale > 1e-8 else np.zeros_like(array)


def build_state_frame(
    cfg: dict[str, Any],
    base_cfg: dict[str, Any],
    write_output: bool = True,
) -> pd.DataFrame:
    cards = load_card_states(base_cfg)
    rules = load_rule_view_states(base_cfg)
    predictions = load_prediction_states(base_cfg)
    state = cards.merge(rules, on="date", how="left").merge(predictions, on="date", how="left")
    state = state.loc[pd.to_datetime(state["date"]).ge(pd.Timestamp(cfg["data"]["start_date"]))].copy()
    complete_as_of = cfg["data"].get("complete_data_as_of")
    if complete_as_of:
        state = state.loc[pd.to_datetime(state["date"]).le(pd.Timestamp(complete_as_of))].copy()

    prediction_columns = [
        column for column in predictions.columns if column.startswith("state_it_")
    ]
    state["state_it_available"] = state[prediction_columns].notna().any(axis=1).astype(float)
    state[prediction_columns] = state[prediction_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    rule_columns = [column for column in state if column.startswith("state_rule_")]
    state[rule_columns] = state[rule_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    other_state_columns = [
        column
        for column in state
        if column.startswith("state_") and column not in prediction_columns and column not in rule_columns
    ]
    state[other_state_columns] = (
        state[other_state_columns].replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
    )
    state["stock_bond_regime"] = state["stock_bond_regime"].ffill().fillna("unknown")

    derived_columns: dict[str, np.ndarray] = {}
    for horizon in [int(value) for value in base_cfg["actions"]["horizons_days"]]:
        model_columns = [f"state_it_model_h{horizon}_{sleeve}" for sleeve in MODULES]
        if not set(model_columns).issubset(state.columns):
            continue
        model = state[model_columns].to_numpy(dtype=float)
        grid = state[[f"state_grid_compat_{sleeve}" for sleeve in MODULES]].to_numpy(dtype=float)
        denominator = np.linalg.norm(model, axis=1) * np.linalg.norm(grid, axis=1)
        derived_columns[f"state_it_grid_alignment_h{horizon}"] = np.divide(
            np.sum(model * grid, axis=1),
            denominator,
            out=np.zeros(len(state)),
            where=denominator > 1e-8,
        )
        derived_columns[f"state_it_incremental_reliability_h{horizon}"] = (
            state[f"state_it_h{horizon}_model_reliability"]
            - state[f"state_it_h{horizon}_baseline_reliability"]
        )
        derived_columns[f"state_it_cross_section_h{horizon}"] = np.ptp(model, axis=1)

    if derived_columns:
        state = pd.concat([state, pd.DataFrame(derived_columns, index=state.index)], axis=1)

    state = state.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    if write_output:
        output = output_path(cfg, "state")
        state.to_parquet(output, index=False)
    return state


def _selected_static_columns(state: pd.DataFrame, base_cfg: dict[str, Any]) -> list[str]:
    columns = [
        "state_grid_bond",
        "state_grid_equity",
        "state_grid_duration",
        "state_grid_regime_changed",
        "state_risk_budget_score",
        "state_liquidity_shock",
        "state_card_cr_state",
        "state_card_r_state",
        "state_card_cb_state",
        "state_card_eq_state",
        "state_card_alt_state",
        "state_card_rb_state",
        "state_rule_view_count",
        "state_rule_view_mean_confidence",
        "state_it_available",
        "state_it_global_gate",
    ]
    for sleeve in MODULES:
        columns.extend(
            [
                f"state_card_{sleeve}_mean",
                f"state_card_{sleeve}_dispersion",
                f"state_rule_view_{sleeve}",
            ]
        )
    for horizon in [int(value) for value in base_cfg["actions"]["horizons_days"]]:
        columns.extend(
            [
                f"state_it_incremental_reliability_h{horizon}",
                f"state_it_grid_alignment_h{horizon}",
                f"state_it_cross_section_h{horizon}",
            ]
        )
    missing = [column for column in columns if column not in state]
    if missing:
        raise ValueError(f"robust PPO state is missing columns: {missing}")
    forbidden = [
        column
        for column in columns
        if any(token in column.lower() for token in FORBIDDEN_STATE_TOKENS)
    ]
    if forbidden:
        raise ValueError(f"future-looking columns are forbidden in PPO state: {forbidden}")
    return columns


@dataclass(frozen=True)
class ExecutionContext:
    age: int = 0
    last_fraction: float = 0.0
    last_action: int = 0
    nav: float = 1.0
    formal_nav: float = 1.0
    peak: float = 1.0
    formal_peak: float = 1.0
    ewma_variance: float = 0.0
    formal_ewma_variance: float = 0.0
    ewma_excess: float = 0.0
    drawdown_gap: float = 0.0


class ExecutionActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.value = nn.Linear(hidden_dim, 1)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.constant_(self.actor.bias, 0.0)
        nn.init.orthogonal_(self.value.weight, gain=1.0)
        nn.init.constant_(self.value.bias, 0.0)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(obs.float())
        return self.actor(encoded), self.value(encoded).squeeze(-1)


class RobustExecutionEnvironment:
    def __init__(
        self,
        profile: str,
        state: pd.DataFrame,
        cfg: dict[str, Any],
        base_cfg: dict[str, Any],
        transaction_cost_bps: float | None = None,
    ) -> None:
        self.profile = profile
        self.cfg = cfg
        self.base_cfg = copy.deepcopy(base_cfg)
        self.base = build_environment(profile, state, self.base_cfg)
        valid_positions = np.isfinite(self.base.formal_values).all(axis=1)
        valid_positions &= np.isfinite(self.base.risk_budget_values).all(axis=1)
        valid_state = valid_positions[self.base.calendar_positions]
        self.base.state = self.base.state.loc[valid_state].reset_index(drop=True)
        self.base.calendar_positions = self.base.calendar_positions[valid_state]
        if len(self.base.state) == 0:
            raise ValueError(f"{profile} has no finite formal/risk-budget rows")
        self.state = self.base.state
        self.calendar = self.base.calendar
        self.calendar_positions = self.base.calendar_positions
        self.static_cols = _selected_static_columns(self.state, self.base_cfg)
        self.fractions = np.asarray(cfg["target"]["action_fractions"], dtype=float)
        if not np.all(np.diff(self.fractions) >= 0) or self.fractions[0] != 0.0 or self.fractions[-1] != 1.0:
            raise ValueError("action_fractions must be sorted and span 0.0 to 1.0")
        configured_names = cfg["target"].get("action_names")
        self.action_names = (
            [str(value) for value in configured_names]
            if configured_names is not None
            else [f"progress_{fraction:g}" for fraction in self.fractions]
        )
        if len(self.action_names) != len(self.fractions):
            raise ValueError("action_names and action_fractions must have equal length")
        self.destination_mode = configured_names is not None
        self.transaction_cost = float(
            transaction_cost_bps if transaction_cost_bps is not None else cfg["reward"]["transaction_cost_bps"]
        ) / 10000.0
        family = str(cfg["target"]["family"])
        if family == "rules_only":
            self.target_action = self.base.rules_20_action
        elif family == "formal_anchor":
            self.target_action = self.base.formal_anchor_action
        elif family == "itransformer_ep_target":
            self.target_action = None
        else:
            raise ValueError(f"unsupported robust PPO target family: {family}")
        self.target_cache: dict[int, tuple[np.ndarray, dict[str, Any]]] = {}

    @staticmethod
    def initial_context() -> ExecutionContext:
        return ExecutionContext()

    def initial_weights(self, index: int) -> np.ndarray:
        return self.base.initial_weights(index)

    def full_target(self, index: int) -> tuple[np.ndarray, dict[str, Any]]:
        if index not in self.target_cache:
            if self.target_action is None:
                target, diagnostics = self.base.canonical_view_target_weights(index)
            else:
                target, diagnostics = self.base.candidate_weights(index, self.target_action)
            self.target_cache[index] = (target.copy(), dict(diagnostics))
        target, diagnostics = self.target_cache[index]
        return target.copy(), dict(diagnostics)

    def execution_date(self, index: int) -> pd.Timestamp | None:
        position = int(self.calendar_positions[index]) + 1
        return pd.Timestamp(self.calendar[position]) if position < len(self.calendar) else None

    def eligible_rows(self, cutoff: pd.Timestamp) -> np.ndarray:
        eligible: list[int] = []
        for index in range(len(self.state)):
            execution = self.execution_date(index)
            if execution is not None and execution < cutoff:
                eligible.append(index)
        return np.asarray(eligible, dtype=int)

    def observation(
        self,
        index: int,
        current_weights: np.ndarray,
        context: ExecutionContext,
        mean: np.ndarray,
        std: np.ndarray,
    ) -> np.ndarray:
        static = self.state.loc[index, self.static_cols].to_numpy(dtype=float)
        scaled = np.clip((static - mean) / std, -5.0, 5.0)
        target, _ = self.full_target(index)
        previous_target = self.full_target(max(index - 1, 0))[0]
        calendar_position = int(self.calendar_positions[index])
        risk_base = self.base.risk_budget_values[calendar_position]
        current_active = np.clip((np.asarray(current_weights) - risk_base) / 0.05, -3.0, 3.0)
        target_gap = np.clip((target - np.asarray(current_weights)) / 0.05, -3.0, 3.0)
        target_change = np.clip((target - previous_target) / 0.03, -3.0, 3.0)
        strategy_drawdown = context.nav / max(context.peak, 1e-12) - 1.0
        formal_drawdown = context.formal_nav / max(context.formal_peak, 1e-12) - 1.0
        last_action = [float(context.last_action == action) for action in range(len(self.action_names))]
        dynamic = np.asarray(
            [
                math.log1p(context.age) / math.log(253.0),
                float(np.abs(target - np.asarray(current_weights)).sum() / 0.10),
                float(np.abs(target - previous_target).sum() / 0.10),
                float(np.clip(context.ewma_excess / 0.002, -3.0, 3.0)),
                float(np.clip(strategy_drawdown / 0.10, -3.0, 0.0)),
                float(np.clip(formal_drawdown / 0.10, -3.0, 0.0)),
                float(np.clip(math.sqrt(max(context.ewma_variance, 0.0) * 252.0) / 0.20, 0.0, 3.0)),
                float(np.clip(math.sqrt(max(context.formal_ewma_variance, 0.0) * 252.0) / 0.20, 0.0, 3.0)),
                *last_action,
            ],
            dtype=float,
        )
        return np.concatenate([scaled, current_active, target_gap, target_change, dynamic]).astype(np.float32)

    def threshold_action(self, index: int, current_weights: np.ndarray) -> int:
        target, _ = self.full_target(index)
        distance = float(np.abs(target - np.asarray(current_weights)).sum())
        threshold = float(self.cfg["target"]["no_trade_l1"][self.profile])
        destination = (
            self.action_names.index("rules_only")
            if self.destination_mode
            else len(self.fractions) - 1
        )
        return destination if distance >= threshold else 0

    def step(
        self,
        index: int,
        action: int,
        previous_weights: np.ndarray,
        context: ExecutionContext,
        action_fraction_override: float | None = None,
    ) -> tuple[float, int, np.ndarray, ExecutionContext, dict[str, Any]]:
        signal_position = int(self.calendar_positions[index])
        execution_position = signal_position + 1
        if execution_position >= len(self.calendar):
            return 0.0, len(self.state), previous_weights.copy(), context, {"daily": []}
        full_target, diagnostics = self.full_target(index)
        fraction = float(
            np.clip(action_fraction_override, 0.0, 1.0)
            if action_fraction_override is not None
            else self.fractions[int(action)]
        )
        previous = np.asarray(previous_weights, dtype=float)
        action_name = (
            "ppo_shielded_blend"
            if action_fraction_override is not None
            else self.action_names[int(action)]
        )
        if self.destination_mode:
            if action_name == "hold":
                target = previous.copy()
                diagnostics = {"optimizer_backend": "hold"}
            elif action_name == "formal_anchor":
                target, diagnostics = self.base.candidate_weights(
                    index, self.base.formal_anchor_action
                )
            elif action_name == "rules_only":
                target = full_target.copy()
            else:
                raise ValueError(f"unsupported PPO destination action: {action_name}")
        else:
            target = previous + fraction * (full_target - previous)
        risk_base = self.base.risk_budget_values[signal_position]
        target = self.base._apply_guardrails(index, target, risk_base)
        turnover = float(np.abs(target - previous).sum())
        linear_cost = turnover * self.transaction_cost
        quadratic_cfg = self.cfg["reward"].get("quadratic_impact", {})
        quadratic_rate = (
            float(quadratic_cfg.get(self.profile, 0.0))
            if isinstance(quadratic_cfg, dict)
            else float(quadratic_cfg)
        )
        impact_cost = quadratic_rate * turnover**2
        cost = linear_cost + impact_cost
        returns = self.base.return_values[execution_position]
        strategy_return = float(np.dot(target, returns) - cost)
        formal_weights = self.base.formal_values[execution_position]
        formal_return = float(
            np.dot(formal_weights, returns) - self.base.formal_cost[execution_position]
        )

        beta = float(self.cfg["reward"]["ewma_beta"])
        new_variance = beta * context.ewma_variance + (1.0 - beta) * strategy_return**2
        new_formal_variance = beta * context.formal_ewma_variance + (1.0 - beta) * formal_return**2
        new_nav = context.nav * (1.0 + strategy_return)
        new_formal_nav = context.formal_nav * (1.0 + formal_return)
        new_peak = max(context.peak, new_nav)
        new_formal_peak = max(context.formal_peak, new_formal_nav)
        new_drawdown = new_nav / max(new_peak, 1e-12) - 1.0
        new_formal_drawdown = new_formal_nav / max(new_formal_peak, 1e-12) - 1.0
        new_drawdown_gap = max(0.0, abs(new_drawdown) - abs(new_formal_drawdown))
        drawdown_worsening = max(0.0, new_drawdown_gap - context.drawdown_gap)
        volatility_worsening = max(
            0.0,
            math.sqrt(max(new_variance, 0.0) * 252.0)
            - math.sqrt(max(new_formal_variance, 0.0) * 252.0),
        ) * (1.0 - beta)
        downside_worsening = max(0.0, -strategy_return) - max(0.0, -formal_return)
        remaining_gap = float(np.abs(full_target - target).sum())
        reward = (
            strategy_return
            - formal_return
            - float(self.cfg["reward"]["downside_penalty"][self.profile])
            * max(0.0, downside_worsening)
            - float(self.cfg["reward"]["relative_volatility_penalty"][self.profile])
            * volatility_worsening
            - float(self.cfg["reward"]["relative_drawdown_penalty"][self.profile])
            * drawdown_worsening
            - float(self.cfg["reward"]["target_gap_penalty"][self.profile]) * remaining_gap
        ) * float(self.cfg["reward"]["reward_scale"])

        rebalanced = turnover > 1e-10
        new_context = ExecutionContext(
            age=0 if rebalanced else context.age + 1,
            last_fraction=fraction if rebalanced else context.last_fraction,
            last_action=int(action) if rebalanced else context.last_action,
            nav=new_nav,
            formal_nav=new_formal_nav,
            peak=new_peak,
            formal_peak=new_formal_peak,
            ewma_variance=new_variance,
            formal_ewma_variance=new_formal_variance,
            ewma_excess=beta * context.ewma_excess + (1.0 - beta) * (strategy_return - formal_return),
            drawdown_gap=new_drawdown_gap,
        )
        next_index = int(np.searchsorted(self.calendar_positions, execution_position, side="left"))
        next_index = max(index + 1, next_index)
        daily = {
            "date": self.calendar[execution_position],
            "signal_date": pd.Timestamp(self.state.loc[index, "date"]),
            "formal_return": formal_return,
            "strategy_return": strategy_return,
            "strategy_turnover": turnover,
            "strategy_cost": cost,
            **{f"weight_{sleeve}": float(target[i]) for i, sleeve in enumerate(MODULES)},
        }
        info = {
            "daily": [daily],
            "action_fraction": fraction,
            "action_name": action_name,
            "turnover": turnover,
            "cost": cost,
            "linear_cost": linear_cost,
            "impact_cost": impact_cost,
            "remaining_target_gap": remaining_gap,
            "full_target_distance": float(np.abs(full_target - previous).sum()),
            "execution_start": self.calendar[execution_position],
            "optimizer_backend": diagnostics.get("optimizer_backend", "unknown"),
            "itransformer_active": bool(diagnostics.get("itransformer_active", False)),
            "itransformer_horizon_count": int(
                diagnostics.get("itransformer_horizon_count", 0)
            ),
            "itransformer_disagreement": float(
                diagnostics.get("itransformer_disagreement", 0.0)
            ),
            "itransformer_uncertainty_multiplier": float(
                diagnostics.get("itransformer_uncertainty_multiplier", 0.0)
            ),
        }
        return float(reward), next_index, target.copy(), new_context, info


def _normalizer(env: RobustExecutionEnvironment, train_rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = env.state.loc[train_rows, env.static_cols].to_numpy(dtype=float)
    mean = np.nanmean(values, axis=0)
    std = np.nanstd(values, axis=0)
    std = np.where(std > 1e-8, std, 1.0)
    return mean, std


def _policy_action(
    env: RobustExecutionEnvironment,
    model: ExecutionActorCritic,
    index: int,
    current: np.ndarray,
    context: ExecutionContext,
    mean: np.ndarray,
    std: np.ndarray,
) -> tuple[int, np.ndarray]:
    observation = env.observation(index, current, context, mean, std)
    with torch.no_grad():
        logits, _ = model(torch.tensor(observation)[None, :])
        probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    return int(np.argmax(probabilities)), probabilities


def _simulate_segment(
    env: RobustExecutionEnvironment,
    rows: np.ndarray,
    selector: Callable[[int, np.ndarray, ExecutionContext], int],
    include_daily: bool = False,
) -> dict[str, Any]:
    if len(rows) == 0:
        return {"mean_reward": float("nan"), "sum_reward": float("nan"), "daily": [], "actions": []}
    allowed = set(int(value) for value in rows)
    index = int(rows[0])
    stop = int(rows[-1]) + 1
    current = env.initial_weights(index)
    context = env.initial_context()
    rewards: list[float] = []
    daily_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    while index < stop and index in allowed:
        action = int(selector(index, current, context))
        reward, next_index, current, context, info = env.step(index, action, current, context)
        if not info.get("daily"):
            break
        rewards.append(float(reward))
        if include_daily:
            daily_rows.extend(info["daily"])
            action_rows.append(
                {
                    "signal_date": pd.Timestamp(env.state.loc[index, "date"]),
                    "action": action,
                    "action_fraction": float(env.fractions[action]),
                    "reward": float(reward),
                    **{key: value for key, value in info.items() if key != "daily"},
                }
            )
        index = max(index + 1, next_index)
    return {
        "mean_reward": float(np.mean(rewards)) if rewards else float("nan"),
        "sum_reward": float(np.sum(rewards)) if rewards else float("nan"),
        "daily": daily_rows,
        "actions": action_rows,
    }


def _baseline_selectors(env: RobustExecutionEnvironment) -> dict[str, Callable[[int, np.ndarray, ExecutionContext], int]]:
    selectors = {
        "hold": lambda index, current, context: 0,
        "full": lambda index, current, context: (
            env.action_names.index("rules_only")
            if env.destination_mode
            else len(env.fractions) - 1
        ),
        "threshold": lambda index, current, context: env.threshold_action(index, current),
    }
    if env.destination_mode and "formal_anchor" in env.action_names:
        selectors["formal_action"] = lambda index, current, context: env.action_names.index(
            "formal_anchor"
        )
    return selectors


def _validation_blocks(rows: np.ndarray, count: int) -> list[np.ndarray]:
    return [block.astype(int) for block in np.array_split(rows, count) if len(block) >= 20]


def _passes_validation_gate(
    median_edge: float,
    positive_block_fraction: float,
    full_cfg: dict[str, Any],
) -> bool:
    deployment = full_cfg["deployment"]
    return bool(
        positive_block_fraction
        >= float(deployment["minimum_positive_validation_block_fraction"])
        and median_edge
        >= float(deployment["minimum_median_edge_vs_best_baseline"])
    )


def _evaluate_validation(
    env: RobustExecutionEnvironment,
    blocks: list[np.ndarray],
    model: ExecutionActorCritic,
    mean: np.ndarray,
    std: np.ndarray,
) -> tuple[float, list[dict[str, Any]]]:
    selectors = _baseline_selectors(env)
    block_results: list[dict[str, Any]] = []
    for block in blocks:
        policy = _simulate_segment(
            env,
            block,
            lambda index, current, context: _policy_action(
                env, model, index, current, context, mean, std
            )[0],
        )
        baseline_scores = {
            name: _simulate_segment(env, block, selector)["mean_reward"]
            for name, selector in selectors.items()
        }
        block_results.append(
            {"block": block, "policy": policy["mean_reward"], "baselines": baseline_scores}
        )
    baseline_medians = {
        name: float(np.median([result["baselines"][name] for result in block_results]))
        for name in selectors
    }
    best_name = max(baseline_medians, key=baseline_medians.get)
    rows: list[dict[str, Any]] = []
    for block_number, result in enumerate(block_results, start=1):
        block = result["block"]
        baseline_scores = result["baselines"]
        edge = float(result["policy"] - baseline_scores[best_name])
        rows.append(
            {
                "block": block_number,
                "start_date": pd.Timestamp(env.state.loc[int(block[0]), "date"]),
                "end_date": pd.Timestamp(env.state.loc[int(block[-1]), "date"]),
                "policy_mean_reward": result["policy"],
                "best_baseline": best_name,
                "best_baseline_mean_reward": baseline_scores[best_name],
                "best_baseline_validation_median": baseline_medians[best_name],
                "edge_vs_best_baseline": edge,
                **{f"{name}_mean_reward": value for name, value in baseline_scores.items()},
            }
        )
    edges = np.asarray([row["edge_vs_best_baseline"] for row in rows], dtype=float)
    score = float(np.median(edges) + 0.25 * np.min(edges)) if len(edges) else -np.inf
    return score, rows


def _validation_confidence_floor(
    env: RobustExecutionEnvironment,
    blocks: list[np.ndarray],
    model: ExecutionActorCritic,
    mean: np.ndarray,
    std: np.ndarray,
) -> tuple[float, float]:
    top_probabilities: list[float] = []
    margins: list[float] = []
    for block in blocks:
        current = env.initial_weights(int(block[0]))
        context = env.initial_context()
        for index in block:
            action, probabilities = _policy_action(
                env, model, int(index), current, context, mean, std
            )
            top_probabilities.append(float(probabilities[action]))
            competitor = float(np.delete(probabilities, action).max())
            margins.append(float(probabilities[action] - competitor))
            _, _, current, context, info = env.step(int(index), action, current, context)
            if not info.get("daily"):
                break
    if not top_probabilities:
        return 1.0, 1.0
    return (
        float(np.quantile(top_probabilities, 0.10)),
        float(np.quantile(margins, 0.10)),
    )


def _collect_rollouts(
    env: RobustExecutionEnvironment,
    model: ExecutionActorCritic,
    train_rows: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, torch.Tensor]:
    cfg = env.cfg["ppo"]
    horizon = min(int(cfg["rollout_days"]), len(train_rows))
    starts = rng.integers(0, max(len(train_rows) - horizon + 1, 1), size=int(cfg["episodes_per_epoch"]))
    observations: list[np.ndarray] = []
    actions: list[int] = []
    old_logp: list[float] = []
    rewards: list[float] = []
    values: list[float] = []
    dones: list[float] = []
    for start_position in starts:
        episode_rows = train_rows[int(start_position) : int(start_position) + horizon]
        allowed = set(int(value) for value in episode_rows)
        index = int(episode_rows[0])
        stop = int(episode_rows[-1]) + 1
        current = env.initial_weights(index)
        context = env.initial_context()
        episode_transition_positions: list[int] = []
        while index < stop and index in allowed:
            observation = env.observation(index, current, context, mean, std)
            obs_tensor = torch.tensor(observation)[None, :]
            with torch.no_grad():
                logits, value = model(obs_tensor)
                distribution = torch.distributions.Categorical(logits=logits)
                action = int(distribution.sample().item())
                log_probability = float(distribution.log_prob(torch.tensor([action])).item())
            reward, next_index, current, context, info = env.step(index, action, current, context)
            if not info.get("daily"):
                break
            observations.append(observation)
            actions.append(action)
            old_logp.append(log_probability)
            rewards.append(float(reward))
            values.append(float(value.item()))
            dones.append(0.0)
            episode_transition_positions.append(len(dones) - 1)
            index = max(index + 1, next_index)
        if episode_transition_positions:
            dones[episode_transition_positions[-1]] = 1.0
    if not observations:
        raise ValueError("PPO rollout contains no transitions")
    return {
        "obs": torch.tensor(np.vstack(observations), dtype=torch.float32),
        "action": torch.tensor(actions, dtype=torch.long),
        "old_logp": torch.tensor(old_logp, dtype=torch.float32),
        "reward": torch.tensor(rewards, dtype=torch.float32),
        "old_value": torch.tensor(values, dtype=torch.float32),
        "done": torch.tensor(dones, dtype=torch.float32),
    }


def _purged_behavior_pretrain(
    env: RobustExecutionEnvironment,
    model: ExecutionActorCritic,
    optimizer: torch.optim.Optimizer,
    train_rows: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> dict[str, Any]:
    epochs = int(env.cfg["ppo"].get("causal_behavior_clone_epochs", 0))
    if epochs <= 0:
        return {"causal_teacher_rows": 0, "causal_teacher_accuracy": float("nan")}
    teacher_mode = str(
        env.cfg["ppo"].get("behavior_clone_teacher", "threshold")
    )
    if teacher_mode not in {"threshold", "one_step_oracle"}:
        raise ValueError(f"unsupported behavior_clone_teacher: {teacher_mode}")
    oracle_minimum_edge = float(
        env.cfg["ppo"].get("oracle_teacher_minimum_reward_edge", 0.0)
    )
    stride = max(int(env.cfg["ppo"].get("causal_teacher_stride", 2)), 1)
    current = env.initial_weights(int(train_rows[0]))
    context = env.initial_context()
    observations: list[np.ndarray] = []
    targets: list[int] = []
    oracle_positive_edges = 0
    for position, index in enumerate(train_rows):
        index = int(index)
        baseline_action = env.threshold_action(index, current)
        action = baseline_action
        if teacher_mode == "one_step_oracle":
            rewards = np.asarray(
                [
                    env.step(index, candidate, current, context)[0]
                    for candidate in range(len(env.fractions))
                ],
                dtype=float,
            )
            best_action = int(np.argmax(rewards))
            edge = float(rewards[best_action] - rewards[baseline_action])
            if edge > oracle_minimum_edge:
                action = best_action
                oracle_positive_edges += 1
        if position % stride == 0:
            observations.append(env.observation(index, current, context, mean, std))
            targets.append(action)
        # Observations stay on the causal threshold path. The one-step oracle
        # is only a training label and never determines validation/deployment
        # holdings.
        _, _, current, context, info = env.step(
            index, baseline_action, current, context
        )
        if not info.get("daily"):
            break
    observation_tensor = torch.tensor(np.vstack(observations), dtype=torch.float32)
    target_tensor = torch.tensor(targets, dtype=torch.long)
    counts = np.bincount(targets, minlength=len(env.fractions)).astype(float)
    present = counts > 0
    class_weights = np.zeros(len(env.fractions), dtype=float)
    power = float(env.cfg["ppo"].get("causal_class_balance_power", 0.5))
    class_weights[present] = np.power(counts[present].sum() / counts[present], power)
    class_weights[present] /= class_weights[present].mean()
    weight_tensor = torch.tensor(np.clip(class_weights, 0.0, 4.0), dtype=torch.float32)
    for _ in range(epochs):
        logits, values = model(observation_tensor)
        loss = nn.functional.cross_entropy(logits, target_tensor, weight=weight_tensor)
        loss += 0.01 * torch.mean(values**2)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), float(env.cfg["ppo"]["max_grad_norm"]))
        optimizer.step()
    with torch.no_grad():
        logits, _ = model(observation_tensor)
        accuracy = float((logits.argmax(dim=1) == target_tensor).float().mean())
    return {
        "causal_teacher_rows": len(targets),
        "causal_teacher_accuracy": accuracy,
        "behavior_teacher_mode": teacher_mode,
        "oracle_label_positive_edge_fraction": (
            float(oracle_positive_edges / max(len(train_rows), 1))
            if teacher_mode == "one_step_oracle"
            else 0.0
        ),
        "causal_teacher_distribution": {
            env.action_names[action]: int(count)
            for action, count in enumerate(counts)
            if count > 0
        },
    }


def _advantages(
    reward: torch.Tensor,
    value: torch.Tensor,
    done: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    advantage = torch.zeros_like(reward)
    last = torch.tensor(0.0)
    next_value = torch.tensor(0.0)
    for position in reversed(range(len(reward))):
        non_terminal = 1.0 - done[position]
        delta = reward[position] + gamma * next_value * non_terminal - value[position]
        last = delta + gamma * gae_lambda * non_terminal * last
        advantage[position] = last
        next_value = value[position]
    return advantage, advantage + value


def train_seed(
    env: RobustExecutionEnvironment,
    eligible: np.ndarray,
    year: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    cfg = env.cfg["ppo"]
    n_validation = max(90, int(math.ceil(len(eligible) * float(cfg["validation_fraction"]))))
    validation_rows = eligible[-n_validation:]
    train_stop = len(eligible) - n_validation - int(cfg["purge_days"])
    if train_stop < int(cfg["min_train_rows"]):
        raise ValueError(f"insufficient purged training rows for {year}: {train_stop}")
    train_rows = eligible[:train_stop]
    blocks = _validation_blocks(validation_rows, int(cfg["validation_blocks"]))
    mean, std = _normalizer(env, train_rows)
    initial_context = env.initial_context()
    obs_dim = len(env.observation(int(train_rows[0]), env.initial_weights(int(train_rows[0])), initial_context, mean, std))
    model = ExecutionActorCritic(obs_dim, len(env.action_names), int(cfg["hidden_dim"]))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=1e-4
    )
    pretrain = _purged_behavior_pretrain(
        env, model, optimizer, train_rows, mean, std
    )
    best_state = copy.deepcopy(model.state_dict())
    best_score, best_validation = _evaluate_validation(env, blocks, model, mean, std)
    best_epoch = 0
    initial_edges = np.asarray(
        [row["edge_vs_best_baseline"] for row in best_validation], dtype=float
    )
    best_passes_gate = _passes_validation_gate(
        float(np.median(initial_edges)),
        float(np.mean(initial_edges > 0.0)),
        env.cfg,
    )
    wait = 0
    logs: list[dict[str, Any]] = []
    max_epochs = int(cfg["max_epochs"])
    for epoch in range(1, max_epochs + 1):
        batch = _collect_rollouts(env, model, train_rows, mean, std, rng)
        advantage, returns = _advantages(
            batch["reward"],
            batch["old_value"],
            batch["done"],
            float(cfg["gamma"]),
            float(cfg["gae_lambda"]),
        )
        advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8)
        sample_count = len(batch["action"])
        approx_kl = 0.0
        entropy_value = 0.0
        value_loss_value = 0.0
        policy_loss_value = 0.0
        stop_updates = False
        for _ in range(int(cfg["update_epochs"])):
            permutation = torch.randperm(sample_count)
            for start in range(0, sample_count, int(cfg["minibatch_size"])):
                indices = permutation[start : start + int(cfg["minibatch_size"])]
                logits, value = model(batch["obs"][indices])
                distribution = torch.distributions.Categorical(logits=logits)
                logp = distribution.log_prob(batch["action"][indices])
                log_ratio = logp - batch["old_logp"][indices]
                ratio = torch.exp(log_ratio)
                clipped = torch.clamp(
                    ratio, 1.0 - float(cfg["clip_ratio"]), 1.0 + float(cfg["clip_ratio"])
                )
                policy_loss = -torch.minimum(
                    ratio * advantage[indices], clipped * advantage[indices]
                ).mean()
                value_clipped = batch["old_value"][indices] + torch.clamp(
                    value - batch["old_value"][indices],
                    -float(cfg["value_clip"]),
                    float(cfg["value_clip"]),
                )
                value_loss = 0.5 * torch.maximum(
                    (value - returns[indices]) ** 2,
                    (value_clipped - returns[indices]) ** 2,
                ).mean()
                entropy = distribution.entropy().mean()
                loss = (
                    policy_loss
                    + float(cfg["value_loss_weight"]) * value_loss
                    - float(cfg["entropy_weight"]) * entropy
                )
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), float(cfg["max_grad_norm"]))
                optimizer.step()
                approx_kl = float(((ratio - 1.0) - log_ratio).mean().detach())
                entropy_value = float(entropy.detach())
                value_loss_value = float(value_loss.detach())
                policy_loss_value = float(policy_loss.detach())
                if approx_kl > float(cfg["target_kl"]):
                    stop_updates = True
                    break
            if stop_updates:
                break

        if epoch == 1 or epoch % int(cfg["evaluation_every"]) == 0:
            score, validation = _evaluate_validation(env, blocks, model, mean, std)
            edges = np.asarray([row["edge_vs_best_baseline"] for row in validation])
            log = {
                "profile": env.profile,
                "evaluation_year": year,
                "seed": seed,
                "epoch": epoch,
                "rollout_steps": sample_count,
                "rollout_mean_reward": float(batch["reward"].mean()),
                "validation_selection_score": score,
                "validation_median_edge": float(np.median(edges)),
                "validation_worst_edge": float(np.min(edges)),
                "validation_positive_block_fraction": float(np.mean(edges > 0.0)),
                "validation_candidate_qualified": _passes_validation_gate(
                    float(np.median(edges)),
                    float(np.mean(edges > 0.0)),
                    env.cfg,
                ),
                "policy_loss": policy_loss_value,
                "value_loss": value_loss_value,
                "entropy": entropy_value,
                "approx_kl": approx_kl,
                "causal_teacher_rows": pretrain["causal_teacher_rows"],
                "causal_teacher_accuracy": pretrain["causal_teacher_accuracy"],
            }
            logs.append(log)
            print(
                f"[{env.profile}][{year}][seed={seed}] epoch={epoch} "
                f"rollout={log['rollout_mean_reward']:.6f} val={score:.6f} "
                f"median_edge={log['validation_median_edge']:.6f} "
                f"positive_blocks={log['validation_positive_block_fraction']:.2f} kl={approx_kl:.6f}",
                flush=True,
            )
            candidate_passes_gate = bool(log["validation_candidate_qualified"])
            candidate_is_better = bool(
                (candidate_passes_gate and not best_passes_gate)
                or (
                    candidate_passes_gate == best_passes_gate
                    and score
                    > best_score + float(cfg["early_stopping_min_delta"])
                )
            )
            if candidate_is_better:
                best_score = score
                best_state = copy.deepcopy(model.state_dict())
                best_validation = validation
                best_epoch = epoch
                best_passes_gate = candidate_passes_gate
                wait = 0
            else:
                wait += 1
                if wait >= int(cfg["early_stop_patience"]):
                    break

    edges = np.asarray([row["edge_vs_best_baseline"] for row in best_validation], dtype=float)
    model.load_state_dict(best_state)
    model.eval()
    probability_floor, margin_floor = _validation_confidence_floor(
        env, blocks, model, mean, std
    )
    checkpoint = {
        "state_dict": best_state,
        "mean": mean,
        "std": std,
        "obs_dim": obs_dim,
        "action_dim": len(env.action_names),
        "hidden_dim": int(cfg["hidden_dim"]),
        "training_cutoff": pd.Timestamp(year, 1, 1),
        "training_max_execution_date": env.execution_date(int(train_rows[-1])),
        "validation_start_date": pd.Timestamp(env.state.loc[int(validation_rows[0]), "date"]),
        "validation_end_date": pd.Timestamp(env.state.loc[int(validation_rows[-1]), "date"]),
        "best_epoch": best_epoch,
        "validation_selection_score": best_score,
        "validation_median_edge": float(np.median(edges)),
        "validation_worst_edge": float(np.min(edges)),
        "validation_positive_block_fraction": float(np.mean(edges > 0.0)),
        "validation_probability_floor": probability_floor,
        "validation_margin_floor": margin_floor,
        **pretrain,
    }
    validation_rows_output = [
        {"profile": env.profile, "evaluation_year": year, "seed": seed, **row}
        for row in best_validation
    ]
    return checkpoint, logs, validation_rows_output


def _qualified(checkpoint: dict[str, Any], cfg: dict[str, Any]) -> bool:
    deployment = cfg["deployment"]
    return bool(
        int(checkpoint["best_epoch"]) >= int(deployment.get("minimum_best_epoch", 1))
        and _passes_validation_gate(
            float(checkpoint["validation_median_edge"]),
            float(checkpoint["validation_positive_block_fraction"]),
            cfg,
        )
    )


def choose_action(
    env: RobustExecutionEnvironment,
    checkpoints: list[dict[str, Any]],
    index: int,
    current: np.ndarray,
    context: ExecutionContext,
) -> tuple[int, str, int, float, float, float | None]:
    fallback = env.threshold_action(index, current)
    qualified = [checkpoint for checkpoint in checkpoints if _qualified(checkpoint, env.cfg)]
    if len(qualified) < int(env.cfg["deployment"]["minimum_qualified_models"]):
        if (
            len(qualified) == 1
            and not env.destination_mode
            and bool(env.cfg["deployment"].get("allow_fraction_blending", True))
        ):
            checkpoint = qualified[0]
            model = ExecutionActorCritic(
                checkpoint["obs_dim"], checkpoint["action_dim"], checkpoint["hidden_dim"]
            )
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()
            _, probabilities = _policy_action(
                env,
                model,
                index,
                current,
                context,
                checkpoint["mean"],
                checkpoint["std"],
            )
            proposed_fraction = float(np.dot(probabilities, env.fractions))
            fallback_fraction = float(env.fractions[fallback])
            blend = float(env.cfg["deployment"].get("single_qualified_model_blend", 0.15))
            executed_fraction = fallback_fraction + blend * (
                proposed_fraction - fallback_fraction
            )
            top = int(np.argmax(probabilities))
            margin = float(probabilities[top] - np.delete(probabilities, top).max())
            return (
                fallback,
                "ppo_shielded_single_model",
                1,
                float(probabilities[top]),
                margin,
                executed_fraction,
            )
        return fallback, "threshold_fallback", len(qualified), 0.0, 0.0, None
    probabilities: list[np.ndarray] = []
    votes: list[int] = []
    for checkpoint in qualified:
        model = ExecutionActorCritic(
            checkpoint["obs_dim"], checkpoint["action_dim"], checkpoint["hidden_dim"]
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        action, probability = _policy_action(
            env,
            model,
            index,
            current,
            context,
            checkpoint["mean"],
            checkpoint["std"],
        )
        votes.append(action)
        probabilities.append(probability)
    counts = pd.Series(votes).value_counts()
    if int(counts.iloc[0]) < int(env.cfg["deployment"]["ensemble_min_agreement"]):
        if (
            not env.destination_mode
            and bool(env.cfg["deployment"].get("allow_fraction_blending", True))
        ):
            mean_probability = np.mean(np.vstack(probabilities), axis=0)
            proposed_fraction = float(np.dot(mean_probability, env.fractions))
            fallback_fraction = float(env.fractions[fallback])
            blend = float(env.cfg["deployment"].get("uncertain_ensemble_blend", 0.25))
            return (
                fallback,
                "ppo_shielded_disagreement",
                len(qualified),
                float(mean_probability.max()),
                0.0,
                fallback_fraction + blend * (proposed_fraction - fallback_fraction),
            )
        return fallback, "ensemble_fallback", len(qualified), 0.0, 0.0, None
    selected = int(counts.index[0])
    mean_probability = np.mean(np.vstack(probabilities), axis=0)
    competitors = np.delete(mean_probability, selected)
    probability = float(mean_probability[selected])
    margin = probability - float(competitors.max())
    probability_floor = max(
        float(env.cfg["deployment"]["minimum_action_probability"]),
        float(np.median([checkpoint.get("validation_probability_floor", 0.0) for checkpoint in qualified])),
    )
    margin_floor = max(
        float(env.cfg["deployment"]["minimum_probability_margin"]),
        float(np.median([checkpoint.get("validation_margin_floor", 0.0) for checkpoint in qualified])),
    )
    if (
        probability < probability_floor
        or margin < margin_floor
    ):
        if (
            not env.destination_mode
            and bool(env.cfg["deployment"].get("allow_fraction_blending", True))
        ):
            proposed_fraction = float(np.dot(mean_probability, env.fractions))
            fallback_fraction = float(env.fractions[fallback])
            blend = float(env.cfg["deployment"].get("uncertain_ensemble_blend", 0.25))
            return (
                fallback,
                "ppo_shielded_probability",
                len(qualified),
                probability,
                margin,
                fallback_fraction + blend * (proposed_fraction - fallback_fraction),
            )
        return fallback, "probability_fallback", len(qualified), probability, margin, None
    return selected, "robust_ppo", len(qualified), probability, margin, None


def _load_checkpoints(model_dir: Path, years: list[int]) -> dict[int, list[dict[str, Any]]]:
    checkpoints: dict[int, list[dict[str, Any]]] = {}
    for year in years:
        path = model_dir / f"policy_{year}.pt"
        checkpoints[year] = (
            torch.load(path, map_location="cpu", weights_only=False) if path.exists() else []
        )
    return checkpoints


def _run_strategy_path(
    env: RobustExecutionEnvironment,
    checkpoints: dict[int, list[dict[str, Any]]],
    deployment_start: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    index = 0
    current = env.initial_weights(index)
    context = env.initial_context()
    daily_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    while index < len(env.state):
        signal_date = pd.Timestamp(env.state.loc[index, "date"])
        annual = checkpoints.get(signal_date.year, [])
        if signal_date < deployment_start or not annual:
            action = env.threshold_action(index, current)
            source = "predeployment_threshold"
            qualified, probability, margin = 0, 0.0, 0.0
        else:
            action, source, qualified, probability, margin, fraction_override = choose_action(
                env, annual, index, current, context
            )
        if signal_date < deployment_start or not annual:
            fraction_override = None
        reward, next_index, current, context, info = env.step(
            index, action, current, context, action_fraction_override=fraction_override
        )
        if not info.get("daily"):
            break
        training_cutoffs = [pd.Timestamp(checkpoint["training_cutoff"]) for checkpoint in annual]
        max_training_dates = [
            pd.Timestamp(checkpoint["training_max_execution_date"]) for checkpoint in annual
        ]
        decision_rows.append(
            {
                "date": signal_date,
                "profile": env.profile,
                "stock_bond_regime": env.state.loc[index, "stock_bond_regime"],
                "action_index": action,
                "action_fraction": float(info["action_fraction"]),
                "action_name": env.action_names[action],
                "policy_source": source,
                "qualified_model_count": qualified,
                "selected_probability": probability,
                "probability_margin": margin,
                "policy_training_cutoff": max(training_cutoffs) if training_cutoffs else pd.NaT,
                "training_max_execution_date": max(max_training_dates) if max_training_dates else pd.NaT,
                "execution_start": info["execution_start"],
                "reward": reward,
                "turnover": info["turnover"],
                "cost": info["cost"],
                "linear_cost": info["linear_cost"],
                "impact_cost": info["impact_cost"],
                "remaining_target_gap": info["remaining_target_gap"],
                "full_target_distance": info["full_target_distance"],
                "optimizer_backend": info["optimizer_backend"],
                "itransformer_available": float(
                    env.state.loc[index, "state_it_available"]
                ),
                "itransformer_active": info["itransformer_active"],
                "itransformer_horizon_count": info["itransformer_horizon_count"],
                "itransformer_disagreement": info["itransformer_disagreement"],
                "itransformer_uncertainty_multiplier": info[
                    "itransformer_uncertainty_multiplier"
                ],
                "itransformer_information_cutoff": pd.to_datetime(
                    env.state.loc[index, "itransformer_information_cutoff"],
                    errors="coerce",
                ),
                **{
                    field: value
                    for field, value in env.full_target(index)[1].items()
                    if str(field).startswith("it_h")
                },
                **{f"weight_{sleeve}": float(current[i]) for i, sleeve in enumerate(MODULES)},
            }
        )
        daily_rows.extend(info["daily"])
        index = max(index + 1, next_index)
    return pd.DataFrame(daily_rows), pd.DataFrame(decision_rows)


def _run_comparator(
    env: RobustExecutionEnvironment,
    name: str,
) -> pd.DataFrame:
    selectors = _baseline_selectors(env)
    rows = np.arange(len(env.state), dtype=int)
    result = _simulate_segment(env, rows, selectors[name], include_daily=True)
    daily = pd.DataFrame(result["daily"])
    return daily[["date", "strategy_return", "strategy_turnover", "strategy_cost"]].rename(
        columns={
            "strategy_return": f"{name}_return",
            "strategy_turnover": f"{name}_turnover",
            "strategy_cost": f"{name}_cost",
        }
    )


def _metrics(
    profile: str,
    daily: pd.DataFrame,
    cfg: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    deployment_start = pd.Timestamp(cfg["data"]["deployment_start"])
    evaluation_start = pd.Timestamp(cfg["data"]["evaluation_start"])
    splits = {
        "full_history": pd.Series(True, index=daily.index),
        "walk_forward": daily["date"].ge(deployment_start),
        "oos_2024_onward": daily["date"].ge(evaluation_start),
    }
    strategies = ["formal", "hold", "full", "threshold", "strategy"]
    for split, mask in splits.items():
        row: dict[str, Any] = {"profile": profile, "split": split, "days": int(mask.sum())}
        stats: dict[str, dict[str, float]] = {}
        for strategy in strategies:
            stats[strategy] = perf_stats(daily.loc[mask, f"{strategy}_return"])
            row.update({f"{strategy}_{key}": value for key, value in stats[strategy].items()})
        row["delta_return_vs_threshold"] = (
            stats["strategy"]["annual_return"] - stats["threshold"]["annual_return"]
        )
        row["delta_sharpe_vs_threshold"] = stats["strategy"]["sharpe"] - stats["threshold"]["sharpe"]
        row["delta_return_vs_formal"] = stats["strategy"]["annual_return"] - stats["formal"]["annual_return"]
        row["delta_sharpe_vs_formal"] = stats["strategy"]["sharpe"] - stats["formal"]["sharpe"]
        row["mean_strategy_turnover"] = float(daily.loc[mask, "strategy_turnover"].mean())
        row["mean_threshold_turnover"] = float(daily.loc[mask, "threshold_turnover"].mean())
        for cost_bps in cfg["stress"]["cost_bps"]:
            extra = daily.loc[mask, "strategy_turnover"] * (
                float(cost_bps) - float(cfg["reward"]["transaction_cost_bps"])
            ) / 10000.0
            stressed = daily.loc[mask, "strategy_return"] - extra
            stressed_stats = perf_stats(stressed)
            row[f"strategy_sharpe_cost_{float(cost_bps):g}bp"] = stressed_stats["sharpe"]
            row[f"strategy_return_cost_{float(cost_bps):g}bp"] = stressed_stats["annual_return"]
        rows.append(row)
    return pd.DataFrame(rows)


def _paired_block_bootstrap(
    profile: str,
    daily: pd.DataFrame,
    cfg: dict[str, Any],
) -> pd.DataFrame:
    """Paired moving-block bootstrap of PPO versus the mechanical threshold policy."""
    deployment_start = pd.Timestamp(cfg["data"]["deployment_start"])
    evaluation_start = pd.Timestamp(cfg["data"]["evaluation_start"])
    splits = {
        "walk_forward": daily["date"].ge(deployment_start),
        "oos_2024_onward": daily["date"].ge(evaluation_start),
    }
    draws = int(cfg["stress"].get("bootstrap_draws", 1000))
    block_days = max(2, int(cfg["stress"].get("bootstrap_block_days", 20)))
    profile_seed = {"low_vol": 1103, "balanced": 2203, "high_vol": 3301}.get(profile, 4409)
    rows: list[dict[str, Any]] = []
    for split_index, (split, mask) in enumerate(splits.items()):
        sample = daily.loc[mask, ["strategy_return", "threshold_return"]].dropna()
        strategy = sample["strategy_return"].to_numpy(dtype=float)
        threshold = sample["threshold_return"].to_numpy(dtype=float)
        n_rows = len(sample)
        if n_rows < block_days or draws <= 0:
            continue
        rng = np.random.default_rng(profile_seed + split_index)
        annual_edges = np.empty(draws, dtype=float)
        sharpe_edges = np.empty(draws, dtype=float)
        blocks_needed = int(math.ceil(n_rows / block_days))
        offsets = np.arange(block_days, dtype=int)
        for draw in range(draws):
            starts = rng.integers(0, n_rows, size=blocks_needed)
            indices = ((starts[:, None] + offsets[None, :]) % n_rows).reshape(-1)[:n_rows]
            strategy_stats = perf_stats(pd.Series(strategy[indices]))
            threshold_stats = perf_stats(pd.Series(threshold[indices]))
            annual_edges[draw] = (
                strategy_stats["annual_return"] - threshold_stats["annual_return"]
            )
            sharpe_edges[draw] = strategy_stats["sharpe"] - threshold_stats["sharpe"]
        observed_strategy = perf_stats(pd.Series(strategy))
        observed_threshold = perf_stats(pd.Series(threshold))
        rows.append(
            {
                "profile": profile,
                "split": split,
                "days": n_rows,
                "draws": draws,
                "block_days": block_days,
                "observed_annual_return_edge": (
                    observed_strategy["annual_return"] - observed_threshold["annual_return"]
                ),
                "annual_return_edge_p025": float(np.quantile(annual_edges, 0.025)),
                "annual_return_edge_median": float(np.median(annual_edges)),
                "annual_return_edge_p975": float(np.quantile(annual_edges, 0.975)),
                "probability_annual_return_edge_positive": float(np.mean(annual_edges > 0.0)),
                "observed_sharpe_edge": (
                    observed_strategy["sharpe"] - observed_threshold["sharpe"]
                ),
                "sharpe_edge_p025": float(np.quantile(sharpe_edges, 0.025)),
                "sharpe_edge_median": float(np.median(sharpe_edges)),
                "sharpe_edge_p975": float(np.quantile(sharpe_edges, 0.975)),
                "probability_sharpe_edge_positive": float(np.mean(sharpe_edges > 0.0)),
            }
        )
    return pd.DataFrame(rows)


def run_profile(
    profile: str,
    state: pd.DataFrame,
    cfg: dict[str, Any],
    base_cfg: dict[str, Any],
    train_models: bool,
) -> dict[str, Any]:
    env = RobustExecutionEnvironment(profile, state, cfg, base_cfg)
    model_dir = output_path(cfg, "model_dir") / profile
    model_dir.mkdir(parents=True, exist_ok=True)
    years = sorted(int(value) for value in pd.to_datetime(env.state["date"]).dt.year.unique())
    deployment_year = pd.Timestamp(cfg["data"]["deployment_start"]).year
    training_logs: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    if train_models:
        for year in years:
            if year < deployment_year:
                continue
            cutoff = pd.Timestamp(year, 1, 1)
            eligible = env.eligible_rows(cutoff)
            n_validation_rows = max(
                90,
                int(
                    math.ceil(
                        len(eligible)
                        * float(cfg["ppo"]["validation_fraction"])
                    )
                ),
            )
            purged_training_rows = (
                len(eligible)
                - n_validation_rows
                - int(cfg["ppo"]["purge_days"])
            )
            if purged_training_rows < int(cfg["ppo"]["min_train_rows"]):
                print(
                    f"[{profile}][{year}] robust_training_skipped "
                    f"eligible_rows={len(eligible)} "
                    f"purged_training_rows={purged_training_rows}",
                    flush=True,
                )
                continue
            annual_models: list[dict[str, Any]] = []
            for seed in cfg["ppo"]["seeds"]:
                started = time.perf_counter()
                print(
                    f"[{profile}][{year}][seed={seed}] robust_training_start eligible_rows={len(eligible)}",
                    flush=True,
                )
                checkpoint, logs, validation = train_seed(env, eligible, year, int(seed))
                annual_models.append(checkpoint)
                training_logs.extend(logs)
                validation_rows.extend(validation)
                print(
                    f"[{profile}][{year}][seed={seed}] robust_training_done "
                    f"best_epoch={checkpoint['best_epoch']} median_edge={checkpoint['validation_median_edge']:.6f} "
                    f"elapsed_seconds={time.perf_counter()-started:.1f}",
                    flush=True,
                )
            torch.save(annual_models, model_dir / f"policy_{year}.pt")
        pd.DataFrame(training_logs).to_csv(
            output_path(cfg, "training_log_pattern", profile), index=False, encoding="utf-8-sig"
        )
        pd.DataFrame(validation_rows).to_csv(
            output_path(cfg, "validation_pattern", profile), index=False, encoding="utf-8-sig"
        )
    checkpoints = _load_checkpoints(model_dir, years)
    strategy, decisions = _run_strategy_path(
        env, checkpoints, pd.Timestamp(cfg["data"]["deployment_start"])
    )
    strategy = strategy.rename(
        columns={
            "strategy_return": "strategy_return",
            "strategy_turnover": "strategy_turnover",
            "strategy_cost": "strategy_cost",
        }
    )
    daily = strategy
    for comparator in ["hold", "full", "threshold"]:
        daily = daily.merge(_run_comparator(env, comparator), on="date", how="inner")
    daily["date"] = pd.to_datetime(daily["date"])
    for name in ["formal", "hold", "full", "threshold", "strategy"]:
        daily[f"{name}_nav"] = (1.0 + daily[f"{name}_return"]).cumprod()
    metrics = _metrics(profile, daily, cfg)
    daily.to_csv(output_path(cfg, "daily_pattern", profile), index=False, encoding="utf-8-sig")
    decisions.to_parquet(output_path(cfg, "decisions_pattern", profile), index=False)
    metrics.to_csv(output_path(cfg, "metrics_pattern", profile), index=False, encoding="utf-8-sig")
    return {
        "profile": profile,
        "daily_rows": len(daily),
        "decision_rows": len(decisions),
        "ppo_days": int(decisions["policy_source"].str.contains("ppo").sum()),
        "fallback_days": int((~decisions["policy_source"].str.contains("ppo")).sum()),
    }


def audit_and_report(
    cfg: dict[str, Any],
    base_cfg: dict[str, Any],
    profiles: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    bootstrap_rows: list[pd.DataFrame] = []
    policy_source_rows: list[dict[str, Any]] = []
    for profile in profiles:
        daily = pd.read_csv(
            output_path(cfg, "daily_pattern", profile), parse_dates=["date", "signal_date"]
        )
        decisions = pd.read_parquet(output_path(cfg, "decisions_pattern", profile))
        for column in [
            "date",
            "execution_start",
            "policy_training_cutoff",
            "training_max_execution_date",
            "itransformer_information_cutoff",
        ]:
            decisions[column] = pd.to_datetime(decisions[column], errors="coerce")
        weights = decisions[[f"weight_{sleeve}" for sleeve in MODULES]]
        itransformer_available = pd.to_numeric(
            decisions["itransformer_available"], errors="coerce"
        ).fillna(0.0).gt(0.5)
        allowed_fractions = np.asarray(cfg["target"]["action_fractions"], dtype=float)
        fractions_are_discrete = np.isclose(
            decisions["action_fraction"].to_numpy(dtype=float)[:, None],
            allowed_fractions[None, :],
            atol=1e-10,
        ).any(axis=1)
        maturity_checks: list[bool] = []
        for horizon in [int(value) for value in base_cfg["actions"]["horizons_days"]]:
            qualified_column = f"it_h{horizon}_qualified"
            maturity_column = f"it_h{horizon}_matured_observations"
            if qualified_column not in decisions or maturity_column not in decisions:
                maturity_checks.append(False)
                continue
            qualified = pd.to_numeric(
                decisions[qualified_column], errors="coerce"
            ).fillna(0.0).gt(0.5)
            matured = pd.to_numeric(
                decisions[maturity_column], errors="coerce"
            ).fillna(0.0)
            maturity_checks.append(
                bool(
                    (
                        ~qualified
                        | matured.ge(
                            int(base_cfg["data"].get("reliability_min_observations", 20))
                        )
                    ).all()
                )
            )
        checks = {
            "daily_dates_unique": bool(daily["date"].is_unique),
            "signal_strictly_before_execution": bool((daily["signal_date"] < daily["date"]).all()),
            "weights_sum_to_one": bool(np.allclose(weights.sum(axis=1), 1.0, atol=1e-8)),
            "weights_finite": bool(np.isfinite(weights.to_numpy(dtype=float)).all()),
            "action_fraction_is_configured_discrete_level": bool(
                fractions_are_discrete.all()
            ),
            "itransformer_training_cutoff_not_future": bool(
                (
                    ~itransformer_available
                    | (
                        decisions["itransformer_information_cutoff"].notna()
                        & (
                            decisions["itransformer_information_cutoff"]
                            <= decisions["date"]
                        )
                    )
                ).all()
            ),
            "itransformer_qualified_horizons_are_mature": bool(
                maturity_checks and all(maturity_checks)
            ),
            "training_before_decision": bool(
                (
                    decisions["policy_training_cutoff"].isna()
                    | (decisions["policy_training_cutoff"] <= decisions["date"])
                ).all()
            ),
            "training_rewards_mature_before_cutoff": bool(
                (
                    decisions["training_max_execution_date"].isna()
                    | decisions["policy_training_cutoff"].isna()
                    | (
                        decisions["training_max_execution_date"]
                        < decisions["policy_training_cutoff"]
                    )
                ).all()
            ),
            "state_has_no_future_columns": not any(
                any(token in column.lower() for token in FORBIDDEN_STATE_TOKENS)
                for column in RobustExecutionEnvironment(
                    profile,
                    pd.read_parquet(output_path(cfg, "state")),
                    cfg,
                    base_cfg,
                ).static_cols
            ),
        }
        audits.extend(
            {
                "profile": profile,
                "check": name,
                "passed": passed,
            }
            for name, passed in checks.items()
        )
        bootstrap_rows.append(_paired_block_bootstrap(profile, daily, cfg))
        decisions["year"] = decisions["date"].dt.year
        for (year, source), count in decisions.groupby(["year", "policy_source"]).size().items():
            policy_source_rows.append(
                {
                    "profile": profile,
                    "year": int(year),
                    "policy_source": source,
                    "days": int(count),
                }
            )
        summaries.append(pd.read_csv(output_path(cfg, "metrics_pattern", profile)))
    audit = pd.DataFrame(audits)
    summary = pd.concat(summaries, ignore_index=True)
    bootstrap = pd.concat(bootstrap_rows, ignore_index=True)
    policy_sources = pd.DataFrame(policy_source_rows)
    audit.to_csv(output_path(cfg, "audit_csv"), index=False, encoding="utf-8-sig")
    bootstrap.to_csv(output_path(cfg, "bootstrap_csv"), index=False, encoding="utf-8-sig")
    policy_sources.to_csv(
        output_path(cfg, "policy_source_csv"), index=False, encoding="utf-8-sig"
    )
    summary.to_csv(output_path(cfg, "summary"), index=False, encoding="utf-8-sig")

    oos = summary.loc[summary["split"].eq("oos_2024_onward")].copy()
    lines = [
        "# Robust Execution PPO v1",
        "",
        "## 定位",
        "",
        "MarketSkill、股债九宫格、风险预算和 EP/skfolio 生成唯一方向目标；PPO 只在 0/25/50/100% 四档中决定当日向目标推进多少。",
        "iTransformer v14 只作为 point-in-time 状态特征，不直接改写预期收益或仓位。",
        "",
        "## 防过拟合约束",
        "",
        "- 2018 年起扩展窗口，逐年 walk-forward。",
        "- 训练与验证之间 purge 20 个交易日。",
        "- 验证集拆为三段，同时比较 hold、full、threshold 三个确定性基线。",
        "- 三随机种子中至少两个通过门控并同意，PPO 才能执行；其余日期自动使用 threshold。",
        "- max_epochs=1000 只是上限，同时启用 KL 限制和多段验证早停。",
        "- 信号日严格早于执行日，交易成本进入每日收益。",
        "",
        "## 2024 至今结果",
        "",
        oos[[
            "profile",
            "strategy_annual_return",
            "strategy_sharpe",
            "strategy_max_drawdown",
            "delta_return_vs_threshold",
            "delta_sharpe_vs_threshold",
            "mean_strategy_turnover",
            "strategy_sharpe_cost_10bp",
            "strategy_sharpe_cost_20bp",
        ]].to_markdown(index=False),
        "",
        "## 审计",
        "",
        f"通过 {int(audit['passed'].sum())}/{len(audit)} 项。",
    ]
    output_path(cfg, "report_md").write_text("\n".join(lines), encoding="utf-8")
    oos_bootstrap = bootstrap.loc[bootstrap["split"].eq("oos_2024_onward")].copy()
    deployment_sources = policy_sources.loc[policy_sources["year"].ge(2022)].copy()
    report_lines = [
        "# Robust Execution PPO v1",
        "",
        "## 定位",
        "",
        "MarketSkill、股债九宫格、风险预算和 EP/skfolio 生成唯一方向目标；PPO 只在 0/25/50/100% 四档中决定当日向目标推进多少。",
        "iTransformer v14 只作为 point-in-time 状态特征，不直接改写预期收益或仓位。PPO 是执行与调仓层，不是第二个选资产模型。",
        "",
        "## 防过拟合约束",
        "",
        "- 2018 年起扩展窗口，逐年 walk-forward；训练与验证之间 purge 20 个交易日。",
        "- 验证集拆为三段，同时比较 hold、full、threshold 三个确定性基线；基线在整个验证集上统一选定，不使用逐段事后最优。",
        "- 三随机种子至少两颗通过门控且决策一致时可直接执行 PPO。只有一颗通过时，PPO 相对 threshold 的影响被压到 15%；多模型不一致时最多偏离 25%；无人通过则完全 fallback。",
        "- max_epochs=1000 只是上限，同时启用 KL 限制、多段验证早停、梯度裁剪和权重衰减。",
        "- 信号日严格早于执行日，交易成本进入每日收益；iTransformer 不通过预测门控时仅作为弱状态信息。",
        "- 最终用 20 日成对区块 Bootstrap 检验 PPO 相对 threshold 的增量，不把几个基点的偶然差异包装成有效提升。",
        "",
        "## 2024 至今结果",
        "",
        oos[[
            "profile",
            "strategy_annual_return",
            "strategy_sharpe",
            "strategy_max_drawdown",
            "delta_return_vs_threshold",
            "delta_sharpe_vs_threshold",
            "mean_strategy_turnover",
            "strategy_sharpe_cost_10bp",
            "strategy_sharpe_cost_20bp",
        ]].to_markdown(index=False),
        "",
        "## 成对区块 Bootstrap",
        "",
        oos_bootstrap[[
            "profile",
            "observed_annual_return_edge",
            "annual_return_edge_p025",
            "annual_return_edge_p975",
            "probability_annual_return_edge_positive",
            "observed_sharpe_edge",
            "sharpe_edge_p025",
            "sharpe_edge_p975",
            "probability_sharpe_edge_positive",
        ]].to_markdown(index=False),
        "",
        "## PPO 实际参与情况",
        "",
        deployment_sources.to_markdown(index=False),
        "",
        "## 审计",
        "",
        f"通过 {int(audit['passed'].sum())}/{len(audit)} 项。",
        "",
        "注意：任何有限历史样本都无法保证未来绝不过拟合。这个版本能保证的是严格的因果边界、独立年度部署、失败自动回退，以及对不稳定 PPO 的仓位影响上限。",
    ]
    output_path(cfg, "report_md").write_text("\n".join(report_lines), encoding="utf-8")

    gate_payload: dict[str, Any] | None = None
    if cfg.get("formal_gate") and cfg.get("outputs", {}).get("formal_gate"):
        gate_cfg = cfg["formal_gate"]
        gate_profiles: list[dict[str, Any]] = []
        evaluation_start = pd.Timestamp(cfg["data"]["evaluation_start"])
        for profile in profiles:
            metric_row = oos.loc[oos["profile"].eq(profile)].iloc[0]
            bootstrap_row = oos_bootstrap.loc[
                oos_bootstrap["profile"].eq(profile)
            ].iloc[0]
            daily = pd.read_csv(
                output_path(cfg, "daily_pattern", profile), parse_dates=["date"]
            )
            annual = daily.loc[daily["date"].ge(evaluation_start)].copy()
            annual["year"] = annual["date"].dt.year
            yearly_edges = annual.groupby("year").apply(
                lambda frame: float(
                    (1.0 + frame["strategy_return"]).prod()
                    - (1.0 + frame["threshold_return"]).prod()
                ),
                include_groups=False,
            )
            positive_year_fraction = (
                float(yearly_edges.gt(0.0).mean()) if len(yearly_edges) else 0.0
            )
            drawdown_worsening = max(
                0.0,
                abs(float(metric_row["strategy_max_drawdown"]))
                - abs(float(metric_row["threshold_max_drawdown"])),
            )
            values = {
                "annual_return_edge": float(metric_row["delta_return_vs_threshold"]),
                "sharpe_edge": float(metric_row["delta_sharpe_vs_threshold"]),
                "drawdown_worsening": float(drawdown_worsening),
                "bootstrap_return_probability": float(
                    bootstrap_row["probability_annual_return_edge_positive"]
                ),
                "bootstrap_sharpe_probability": float(
                    bootstrap_row["probability_sharpe_edge_positive"]
                ),
                "positive_year_fraction": positive_year_fraction,
            }
            checks = {
                "annual_return_edge": values["annual_return_edge"]
                >= float(gate_cfg["minimum_oos_annual_return_edge"]),
                "sharpe_edge": values["sharpe_edge"]
                >= float(gate_cfg["minimum_oos_sharpe_edge"]),
                "drawdown_worsening": values["drawdown_worsening"]
                <= float(gate_cfg["maximum_drawdown_worsening"]),
                "bootstrap_return_probability": values[
                    "bootstrap_return_probability"
                ]
                >= float(gate_cfg["minimum_bootstrap_return_probability"]),
                "bootstrap_sharpe_probability": values[
                    "bootstrap_sharpe_probability"
                ]
                >= float(gate_cfg["minimum_bootstrap_sharpe_probability"]),
                "positive_year_fraction": values["positive_year_fraction"]
                >= float(gate_cfg["minimum_positive_year_fraction"]),
            }
            gate_profiles.append(
                {
                    "profile": profile,
                    "passed": bool(all(checks.values())),
                    "values": values,
                    "checks": checks,
                    "yearly_return_edges": {
                        str(int(year)): float(value)
                        for year, value in yearly_edges.items()
                    },
                }
            )
        itransformer_gate_passed = True
        if bool(gate_cfg.get("require_itransformer_research_gate", False)):
            itransformer_gate_path = base_cfg["data"].get(
                "itransformer_gate_report"
            )
            if itransformer_gate_path and (ROOT / itransformer_gate_path).exists():
                itransformer_gate_passed = bool(
                    json.loads(
                        (ROOT / itransformer_gate_path).read_text(encoding="utf-8")
                    ).get("enabled_for_formal_strategy", False)
                )
            else:
                itransformer_gate_passed = False
        gate_payload = {
            "version": cfg["version"],
            "evaluation_start": str(evaluation_start.date()),
            "causal_audit_passed": bool(audit["passed"].all()),
            "itransformer_global_gate_passed": itransformer_gate_passed,
            "all_profiles_passed": bool(
                gate_profiles and all(row["passed"] for row in gate_profiles)
            ),
            "production_promotion_allowed": bool(
                audit["passed"].all()
                and itransformer_gate_passed
                and gate_profiles
                and all(row["passed"] for row in gate_profiles)
            ),
            "profiles": gate_profiles,
        }
        output_path(cfg, "formal_gate").write_text(
            json.dumps(gate_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    gate_table = pd.DataFrame(
        [
            {
                "profile": row["profile"],
                "passed": row["passed"],
                **row["values"],
            }
            for row in (gate_payload or {}).get("profiles", [])
        ]
    )
    clean_lines = [
        f"# {cfg['version']} 方法与结果",
        "",
        "## 架构定位",
        "",
        f"- 目标组合：`{cfg['target']['family']}`。MarketSkill、股债九宫格、资产状态卡、风险预算与 iTransformer 相对观点先生成唯一的 EP/skfolio 目标。",
        f"- PPO 动作仅为目标推进比例：`{', '.join(f'{float(value):g}' for value in cfg['target']['action_fractions'])}`，不选择资产方向，也不选择预测期限。",
        "- 所有信号在 T 日形成，最早于下一交易日执行；线性成本与二次冲击成本均进入逐日收益。",
        "- iTransformer 只输入经历史 residual alpha 校准后的独有残差方向，并受成熟样本、绝对 RankIC、非劣容忍和九宫格约束。",
        "- PPO 按年度扩展窗口训练，训练与验证之间执行 60 日 purge，并由训练期行为预训练、多段验证、随机种子集成和确定性 threshold fallback 约束。",
        "",
        "## 2024 年以来",
        "",
        oos[
            [
                "profile",
                "strategy_annual_return",
                "strategy_sharpe",
                "strategy_max_drawdown",
                "delta_return_vs_threshold",
                "delta_sharpe_vs_threshold",
                "mean_strategy_turnover",
            ]
        ].to_markdown(index=False),
        "",
        "## 成对区块 Bootstrap",
        "",
        oos_bootstrap[
            [
                "profile",
                "observed_annual_return_edge",
                "probability_annual_return_edge_positive",
                "observed_sharpe_edge",
                "probability_sharpe_edge_positive",
            ]
        ].to_markdown(index=False),
        "",
        "## PPO 参与情况",
        "",
        deployment_sources.to_markdown(index=False),
        "",
        "## 正式门控",
        "",
        (
            gate_table.to_markdown(index=False)
            if not gate_table.empty
            else "本配置未启用正式门控。"
        ),
        "",
        f"因果与结构审计通过：{int(audit['passed'].sum())}/{len(audit)}。",
        "",
        "正式门控失败不等于代码失败，而是说明当前样本不足以证明 PPO 相对确定性执行规则具有稳定增量；此时必须保留 threshold fallback，不能把研究结果晋升为正式策略。",
    ]
    output_path(cfg, "report_md").write_text(
        "\n".join(clean_lines), encoding="utf-8"
    )
    return summary, audit


def run_pipeline(
    config_path: Path | None = None,
    profiles: list[str] | None = None,
    train_models: bool = True,
    max_epochs: int | None = None,
    seeds: list[int] | None = None,
) -> dict[str, Any]:
    cfg, base_cfg = load_config(config_path)
    torch.set_num_threads(max(1, int(cfg["ppo"].get("torch_threads", 1))))
    if max_epochs is not None:
        cfg["ppo"]["max_epochs"] = int(max_epochs)
    if seeds is not None:
        cfg["ppo"]["seeds"] = [int(value) for value in seeds]
    selected_profiles = profiles or list(cfg["profiles"])
    state = build_state_frame(cfg, base_cfg)
    results = [
        run_profile(profile, state, cfg, base_cfg, train_models=train_models)
        for profile in selected_profiles
    ]
    summary, audit = audit_and_report(cfg, base_cfg, selected_profiles)
    if not bool(audit["passed"].all()):
        failed = audit.loc[~audit["passed"], ["profile", "check"]].to_dict("records")
        raise RuntimeError(f"robust PPO audit failed: {failed}")
    gate_result: dict[str, Any] | None = None
    if cfg.get("outputs", {}).get("formal_gate"):
        gate_path = output_path(cfg, "formal_gate")
        if gate_path.exists():
            gate_result = json.loads(gate_path.read_text(encoding="utf-8"))
    return {
        "version": cfg["version"],
        "profiles": results,
        "state_rows": len(state),
        "audit_passed": int(audit["passed"].sum()),
        "audit_total": len(audit),
        "summary": str(output_path(cfg, "summary").relative_to(ROOT)),
        "report": str(output_path(cfg, "report_md").relative_to(ROOT)),
        "formal_gate": (
            str(output_path(cfg, "formal_gate").relative_to(ROOT))
            if cfg.get("outputs", {}).get("formal_gate")
            else None
        ),
        "production_promotion_allowed": (
            bool(gate_result["production_promotion_allowed"])
            if gate_result is not None
            else None
        ),
    }

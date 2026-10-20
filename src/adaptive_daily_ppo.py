from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

from src.itransformer_signal import RISK_SLEEVES
from src.ppo_overlay_env import project_simplex_bounds
from src.v1_strategy import load_config as load_v1_config


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "adaptive_daily_ppo_config.yaml"
MODULES = list(RISK_SLEEVES)


def load_config(path: Path | None = None) -> dict[str, Any]:
    with (path or DEFAULT_CONFIG).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def out_path(cfg: dict[str, Any], key: str, profile: str | None = None) -> Path:
    value = str(cfg["outputs"][key])
    if profile is not None:
        value = value.format(profile=profile)
    return ROOT / value


def _json(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value:
        return {}
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return {}


def _rank_ic(scores: np.ndarray, realized: np.ndarray) -> float:
    if not np.isfinite(scores).all() or not np.isfinite(realized).all():
        return np.nan
    return float(pd.Series(scores).corr(pd.Series(realized), method="spearman"))


def _information_cutoff(date: pd.Timestamp) -> pd.Timestamp:
    if date.year <= 2024:
        return pd.Timestamp(date.year - 1, 12, 31)
    return date.to_period("Q").start_time


def load_prediction_states(cfg: dict[str, Any]) -> pd.DataFrame:
    source = pd.read_parquet(ROOT / cfg["data"]["predictions"])
    required = {
        "as_of_date",
        "horizon_days",
        "sleeve",
        "historical_oos_relative_score",
        "walkforward_relative_score",
        "baseline_relative_score",
    }
    missing = required - set(source.columns)
    if missing:
        raise ValueError(f"Daily challenger predictions missing columns: {sorted(missing)}")
    source["as_of_date"] = pd.to_datetime(source["as_of_date"], errors="coerce")
    source["horizon_label_end_date"] = pd.to_datetime(source["horizon_label_end_date"], errors="coerce")
    historical = pd.to_numeric(source["historical_oos_relative_score"], errors="coerce")
    walkforward = pd.to_numeric(source["walkforward_relative_score"], errors="coerce")
    source["model_score"] = walkforward.where(walkforward.notna(), historical)
    source["baseline_score"] = pd.to_numeric(source["baseline_relative_score"], errors="coerce")
    source["realized"] = pd.to_numeric(source.get("realized_relative_score"), errors="coerce")
    horizons = [int(value) for value in cfg["actions"]["horizons_days"]]
    source = source[
        source["horizon_days"].isin(horizons)
        & source["sleeve"].isin(MODULES)
        & source["model_score"].notna()
        & source["baseline_score"].notna()
    ].copy()

    rows: list[dict[str, Any]] = []
    ic_rows: list[dict[str, Any]] = []
    for (date, horizon), group in source.groupby(["as_of_date", "horizon_days"], sort=True):
        group = group.set_index("sleeve").reindex(MODULES)
        raw_model = group["model_score"].to_numpy(dtype=float)
        raw_baseline = group["baseline_score"].to_numpy(dtype=float)
        if not np.isfinite(raw_model).all() or not np.isfinite(raw_baseline).all():
            continue
        model_scaled = (raw_model - raw_model.mean()) / max(float(np.std(raw_model)), 1e-8)
        baseline_scaled = (raw_baseline - raw_baseline.mean()) / max(float(np.std(raw_baseline)), 1e-8)
        row = {"date": pd.Timestamp(date), "horizon_days": int(horizon)}
        for index, sleeve in enumerate(MODULES):
            row[f"raw_model_h{int(horizon)}_{sleeve}"] = float(np.clip(model_scaled[index], -3.0, 3.0))
            row[f"raw_baseline_h{int(horizon)}_{sleeve}"] = float(np.clip(baseline_scaled[index], -3.0, 3.0))
        row["horizon_label_end_date"] = group["horizon_label_end_date"].max()
        rows.append(row)
        realized = group["realized"].to_numpy(dtype=float)
        ic_rows.append(
            {
                "date": pd.Timestamp(date),
                "horizon_days": int(horizon),
                "available_date": group["horizon_label_end_date"].max(),
                "model_rank_ic": _rank_ic(model_scaled, realized),
                "baseline_rank_ic": _rank_ic(baseline_scaled, realized),
            }
        )

    long = pd.DataFrame(rows)
    if long.empty:
        raise ValueError("No strict OOS daily iTransformer predictions are available")
    value_cols = [column for column in long.columns if column.startswith("raw_")]
    state = long.groupby("date", as_index=False)[value_cols].first()
    label_end = long.groupby("date", as_index=False)["horizon_label_end_date"].max()
    state = state.merge(label_end, on="date", how="left")
    state = state.rename(columns={"horizon_label_end_date": "max_horizon_end_date"})

    ic = pd.DataFrame(ic_rows).dropna(subset=["available_date", "model_rank_ic", "baseline_rank_ic"])
    horizon_weights: list[str] = []
    for horizon in horizons:
        history = ic[ic["horizon_days"].eq(horizon)].sort_values("available_date").copy()
        window = int(cfg["data"].get("reliability_window", 60))
        history[f"state_it_h{horizon}_model_reliability"] = history["model_rank_ic"].rolling(window, min_periods=5).mean()
        history[f"state_it_h{horizon}_baseline_reliability"] = history["baseline_rank_ic"].rolling(window, min_periods=5).mean()
        maturity_col = f"state_it_h{horizon}_matured_observations"
        history[maturity_col] = history["model_rank_ic"].rolling(window, min_periods=1).count()
        merged = pd.merge_asof(
            state[["date"]].sort_values("date"),
            history[
                [
                    "available_date",
                    f"state_it_h{horizon}_model_reliability",
                    f"state_it_h{horizon}_baseline_reliability",
                    maturity_col,
                ]
            ].sort_values("available_date"),
            left_on="date",
            right_on="available_date",
            direction="backward",
            allow_exact_matches=False,
        )
        model_rel = merged[f"state_it_h{horizon}_model_reliability"].fillna(0.0).clip(-1.0, 1.0)
        baseline_rel = merged[f"state_it_h{horizon}_baseline_reliability"].fillna(0.0).clip(-1.0, 1.0)
        edge = model_rel - baseline_rel
        maturity = merged[maturity_col].fillna(0.0)
        min_observations = int(cfg["data"].get("reliability_min_observations", 5))
        min_incremental = float(cfg["data"].get("minimum_incremental_reliability", 0.0))
        qualified = (
            model_rel.gt(0.0)
            & edge.ge(min_incremental)
            & maturity.ge(min_observations)
        )
        model_weight = (
            float(cfg["data"].get("minimum_model_weight", 0.0))
            + float(cfg["data"].get("model_reliability_scale", 0.90)) * model_rel.clip(lower=0.0)
            + float(cfg["data"].get("model_edge_scale", 0.90)) * edge.clip(lower=0.0)
        ).clip(
            float(cfg["data"].get("minimum_model_weight", 0.0)),
            float(cfg["data"].get("maximum_model_weight", 0.65)),
        ).where(qualified, 0.0)
        weight_col = f"state_it_h{horizon}_model_weight"
        state[weight_col] = model_weight
        horizon_weights.append(weight_col)
        state[f"state_it_h{horizon}_model_reliability"] = model_rel
        state[f"state_it_h{horizon}_baseline_reliability"] = baseline_rel
        state[maturity_col] = maturity.to_numpy(dtype=float)
        state[f"state_it_h{horizon}_model_qualified"] = qualified.astype(float).to_numpy()
        state[f"state_it_h{horizon}_reliability"] = pd.concat([model_rel, baseline_rel], axis=1).max(axis=1)
        blended_columns: list[str] = []
        for sleeve in MODULES:
            model_col = f"raw_model_h{horizon}_{sleeve}"
            baseline_col = f"raw_baseline_h{horizon}_{sleeve}"
            output_col = f"state_it_h{horizon}_{sleeve}"
            state[output_col] = model_weight * state[model_col] + (1.0 - model_weight) * state[baseline_col]
            blended_columns.append(output_col)
        values = state[blended_columns].to_numpy(dtype=float)
        values = values - values.mean(axis=1, keepdims=True)
        values = values / np.maximum(values.std(axis=1, keepdims=True), 1e-8)
        state.loc[:, blended_columns] = np.clip(values, -3.0, 3.0)
        state[f"state_it_h{horizon}_spread"] = np.ptp(values, axis=1)
    state["state_it_model_weight"] = state[horizon_weights].mean(axis=1)
    state = state.drop(columns=[column for column in state.columns if column.startswith("raw_")])
    state["itransformer_information_cutoff"] = state["date"].map(_information_cutoff)
    return state.sort_values("date").reset_index(drop=True)


def load_card_states(cfg: dict[str, Any]) -> pd.DataFrame:
    cards = pd.read_parquet(ROOT / cfg["data"]["state_cards"])
    cards["date"] = pd.to_datetime(cards["date"], errors="coerce")
    rows: list[dict[str, Any]] = []
    state_maps = {
        "cr_state": {"CR1": 1.0, "CR2": 0.5, "CR3": -0.5, "CR4": -1.0, "CR5": -1.0},
        "r_state": {"R1": 1.0, "R2": 0.0, "R3": -0.3, "R4": -1.0, "R5": -1.0},
        "cb_state": {"CB1": 1.0, "CB2": 0.0, "CB3": -0.3, "CB4": -1.0, "CB5": -1.0},
        "eq_state": {"EQ1": 1.0, "EQ2": 0.0, "EQ3": -0.3, "EQ4": -1.0, "EQ5": -1.0},
        "alt_state": {"ALT1": 1.0, "ALT2": 0.0, "ALT3": -0.5, "ALT4": -1.0},
        "rb_state": {"RB1": 1.0, "RB2": 0.3, "RB3": -0.5, "RB4": -1.0},
    }
    for _, card in cards.iterrows():
        detail = _json(card.get("asset_detail_states"))
        assets = detail.get("assets", {})
        row: dict[str, Any] = {
            "date": card["date"],
            "state_risk_budget_score": float(card.get("risk_budget_score", 0.5)),
            "state_liquidity_shock": float(bool(card.get("liquidity_shock", False))),
            "state_bond_score": float(card.get("bond_score", 0.5)),
            "state_equity_score": float(card.get("equity_score", 0.5)),
        }
        for field, mapping in state_maps.items():
            row[f"state_card_{field}"] = float(mapping.get(str(card.get(field)), 0.0))
        for module in MODULES:
            selected = [value for value in assets.values() if value.get("module") == module]
            scores = np.array([value.get("score", 0.5) for value in selected], dtype=float)
            finite_scores = scores[np.isfinite(scores)]
            row[f"state_card_{module}_mean"] = float(finite_scores.mean()) if len(finite_scores) else 0.5
            row[f"state_card_{module}_dispersion"] = float(finite_scores.std()) if len(finite_scores) else 0.0
            row[f"state_card_{module}_positive_ratio"] = float(
                np.mean([value.get("state") == "positive" for value in selected]) if selected else 0.0
            )
            row[f"state_card_{module}_negative_ratio"] = float(
                np.mean([value.get("state") == "negative" for value in selected]) if selected else 0.0
            )
            row[f"state_card_{module}_eligible_ratio"] = float(
                np.mean([value.get("domain_context", {}).get("overweight_eligible", False) for value in selected])
                if selected else 0.0
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("date").drop_duplicates("date", keep="last")


def build_profile_market(profile: str, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    returns = pd.read_parquet(ROOT / cfg["data"]["asset_returns"])
    returns["date"] = pd.to_datetime(returns["date"], errors="coerce")
    returns["return_1d"] = pd.to_numeric(returns["return_1d"], errors="coerce")
    calendar = pd.DatetimeIndex(
        sorted(returns.loc[returns["asset_code"].eq("equity_broad"), "date"].dropna().unique())
    )
    start = pd.Timestamp(cfg["data"]["start_date"])
    calendar = calendar[calendar >= start]
    ret_wide = returns.pivot_table(index="date", columns="asset_code", values="return_1d", aggfunc="last").reindex(calendar)
    module_map = returns.dropna(subset=["asset_code", "module"]).drop_duplicates("asset_code", keep="last").set_index("asset_code")["module"]

    asset_weights = pd.read_parquet(ROOT / cfg["data"]["asset_weights_pattern"].format(profile=profile))
    asset_weights["date"] = pd.to_datetime(asset_weights["date"], errors="coerce")
    weight_wide = asset_weights.pivot_table(index="date", columns="asset_code", values="weight", aggfunc="last").fillna(0.0).sort_index()
    all_assets = sorted(set(ret_wide.columns).union(weight_wide.columns) - {"cash"})
    ret_wide = ret_wide.reindex(columns=all_assets)
    weight_wide = weight_wide.reindex(columns=all_assets, fill_value=0.0)
    decision_dates = pd.DatetimeIndex(weight_wide.index)
    positions = decision_dates.searchsorted(calendar, side="left") - 1
    valid = positions >= 0
    daily_weights = np.zeros((len(calendar), len(all_assets)), dtype=float)
    daily_weights[valid] = weight_wide.to_numpy(dtype=float)[positions[valid]]
    ret_values = ret_wide.to_numpy(dtype=float)
    sleeve_returns = pd.DataFrame(index=calendar, columns=MODULES, dtype=float)
    for module in MODULES:
        columns = [index for index, asset in enumerate(all_assets) if module_map.get(asset) == module]
        w = daily_weights[:, columns]
        r = ret_values[:, columns]
        available = np.isfinite(r)
        numerator = np.sum(w * np.where(available, r, 0.0), axis=1)
        denominator = np.sum(w * available, axis=1)
        sleeve_returns[module] = np.divide(numerator, denominator, out=np.zeros(len(calendar)), where=denominator > 1e-12)

    sleeves = pd.read_parquet(ROOT / cfg["data"]["sleeve_weights_pattern"].format(profile=profile))
    sleeves["date"] = pd.to_datetime(sleeves["date"], errors="coerce")
    daily_base = pd.merge_asof(
        pd.DataFrame({"date": calendar}),
        sleeves[["date"] + MODULES].sort_values("date"),
        on="date",
        direction="backward",
        allow_exact_matches=False,
    ).set_index("date")
    daily_base = daily_base.ffill().dropna().reindex(calendar).ffill()
    daily_base[MODULES] = daily_base[MODULES].div(daily_base[MODULES].sum(axis=1), axis=0)
    v1 = load_v1_config()
    bounds = v1["product"]["profiles"][profile]["sleeve_bounds"]
    lo = np.array([float(bounds[sleeve][0]) for sleeve in MODULES])
    hi = np.array([float(bounds[sleeve][1]) for sleeve in MODULES])
    projected = np.vstack([project_simplex_bounds(row, lo, hi) for row in daily_base[MODULES].to_numpy(dtype=float)])
    daily_base.loc[:, MODULES] = projected
    return sleeve_returns, daily_base, lo, hi


def build_state_frame(cfg: dict[str, Any]) -> pd.DataFrame:
    predictions = load_prediction_states(cfg)
    cards = load_card_states(cfg)
    state = predictions.merge(cards, on="date", how="left")
    state_cols = [column for column in state.columns if column.startswith("state_")]
    state[state_cols] = state[state_cols].replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
    return state.sort_values("date").reset_index(drop=True)


def action_specs(cfg: dict[str, Any]) -> list[tuple[int, str, float]]:
    specs: list[tuple[int, str, float]] = []
    for horizon in cfg["actions"]["horizons_days"]:
        for name, level in zip(cfg["actions"]["names"], cfg["actions"]["levels"]):
            specs.append((int(horizon), str(name), float(level)))
    return specs


@dataclass
class AdaptiveEnvironment:
    state: pd.DataFrame
    sleeve_returns: pd.DataFrame
    base_weights: pd.DataFrame
    lo: np.ndarray
    hi: np.ndarray
    profile: str
    cfg: dict[str, Any]

    def __post_init__(self) -> None:
        self.state = self.state.copy().sort_values("date").reset_index(drop=True)
        self.calendar = pd.DatetimeIndex(self.sleeve_returns.index)
        self.calendar_positions = self.calendar.get_indexer(pd.DatetimeIndex(self.state["date"]))
        valid = self.calendar_positions >= 0
        self.state = self.state.loc[valid].reset_index(drop=True)
        self.calendar_positions = self.calendar_positions[valid]
        self.state_cols = [column for column in self.state.columns if column.startswith("state_")]
        self.specs = action_specs(self.cfg)
        self.max_transfer = float(self.cfg["actions"]["max_pair_transfer"][self.profile])
        self.min_spread = float(self.cfg["actions"]["min_score_spread"])
        base_values = self.base_weights.reindex(self.calendar)[MODULES].to_numpy(dtype=float)
        self.base_values = np.vstack([project_simplex_bounds(row, self.lo, self.hi) for row in base_values])
        self.return_values = self.sleeve_returns.reindex(self.calendar)[MODULES].fillna(0.0).to_numpy(dtype=float)
        base_turnover = np.vstack([np.zeros(len(MODULES)), np.abs(np.diff(self.base_values, axis=0))]).sum(axis=1)
        self.base_cost = base_turnover * float(self.cfg["reward"]["transaction_cost_bps"]) / 10000.0

    def eligible_rows(self, cutoff: pd.Timestamp) -> np.ndarray:
        max_end = pd.to_datetime(self.state["max_horizon_end_date"], errors="coerce")
        return np.flatnonzero(max_end.lt(cutoff).to_numpy())

    def observation(self, index: int, active: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        static = self.state.loc[index, self.state_cols].to_numpy(dtype=float)
        scaled = (static - mean) / std
        active_scaled = active / max(self.max_transfer, 1e-8)
        return np.concatenate([scaled, active_scaled]).astype(np.float32)

    def active_from_action(self, index: int, action: int) -> np.ndarray:
        horizon, _, level = self.specs[action]
        scores = self.state.loc[index, [f"state_it_h{horizon}_{s}" for s in MODULES]].to_numpy(dtype=float)
        reliability = float(self.state.loc[index, f"state_it_h{horizon}_reliability"])
        if (
            abs(level) < 1e-12
            or float(np.ptp(scores)) < self.min_spread
            or reliability < float(self.cfg["actions"].get("minimum_reliability_for_active", 0.02))
        ):
            return np.zeros(len(MODULES), dtype=float)
        top, bottom = int(np.argmax(scores)), int(np.argmin(scores))
        if level < 0:
            top, bottom = bottom, top
        base = self.base_values[self.calendar_positions[index]]
        transfer = min(abs(level) * self.max_transfer, self.hi[top] - base[top], base[bottom] - self.lo[bottom])
        active = np.zeros(len(MODULES), dtype=float)
        if transfer > 0:
            active[top] = transfer
            active[bottom] = -transfer
        return active

    def step(self, index: int, action: int, previous_active: np.ndarray) -> tuple[float, int, np.ndarray, dict[str, Any]]:
        horizon, name, level = self.specs[action]
        active = self.active_from_action(index, action)
        signal_position = int(self.calendar_positions[index])
        start = signal_position + 1
        stop = min(signal_position + horizon + 1, len(self.calendar))
        if start >= stop:
            return 0.0, len(self.state), active, {"daily": []}
        block_base = self.base_values[start:stop]
        block_returns = self.return_values[start:stop]
        candidate_weights = np.vstack([project_simplex_bounds(row + active, self.lo, self.hi) for row in block_base])
        base_daily = np.sum(block_base * block_returns, axis=1) - self.base_cost[start:stop]
        candidate_daily = np.sum(candidate_weights * block_returns, axis=1) - self.base_cost[start:stop]
        overlay_turnover = float(np.abs(active - previous_active).sum())
        overlay_cost = overlay_turnover * float(self.cfg["reward"]["transaction_cost_bps"]) / 10000.0
        candidate_daily[0] -= overlay_cost
        base_total = float(np.prod(1.0 + base_daily) - 1.0)
        candidate_total = float(np.prod(1.0 + candidate_daily) - 1.0)
        downside_worsening = max(0.0, -candidate_total) - max(0.0, -base_total)
        reward = (
            candidate_total
            - base_total
            - float(self.cfg["reward"]["downside_penalty"]) * max(0.0, downside_worsening)
            - float(self.cfg["reward"]["active_deviation_penalty"]) * float(np.abs(active).sum())
        ) * float(self.cfg["reward"]["reward_scale"])
        next_calendar_position = stop - 1
        next_index = int(np.searchsorted(self.calendar_positions, next_calendar_position, side="left"))
        daily = []
        for offset, calendar_index in enumerate(range(start, stop)):
            daily.append(
                {
                    "date": self.calendar[calendar_index],
                    "signal_date": self.state.loc[index, "date"],
                    "base_return": float(base_daily[offset]),
                    "strategy_return": float(candidate_daily[offset]),
                    "overlay_turnover": overlay_turnover if offset == 0 else 0.0,
                    "overlay_cost": overlay_cost if offset == 0 else 0.0,
                    **{f"weight_{s}": float(candidate_weights[offset, j]) for j, s in enumerate(MODULES)},
                }
            )
        return float(reward), next_index, active, {
            "daily": daily,
            "horizon_days": horizon,
            "action_name": name,
            "action_level": level,
            "overlay_turnover": overlay_turnover,
            "overlay_cost": overlay_cost,
            "candidate_total": candidate_total,
            "base_total": base_total,
            "execution_start": self.calendar[start],
            "execution_end": self.calendar[stop - 1],
            "active": active,
        }


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(obs)
        return self.actor(hidden), self.critic(hidden).squeeze(-1)


def _simulate_policy(
    env: AdaptiveEnvironment,
    start: int,
    stop: int,
    model: ActorCritic,
    mean: np.ndarray,
    std: np.ndarray,
) -> float:
    index = start
    active = np.zeros(len(MODULES))
    total = 0.0
    with torch.no_grad():
        while index < stop:
            obs = torch.tensor(env.observation(index, active, mean, std))[None, :]
            logits, _ = model(obs)
            action = int(logits.argmax(dim=1).item())
            reward, next_index, active, _ = env.step(index, action, active)
            total += reward
            index = max(index + 1, next_index)
    return float(total)


def _validation_score(
    env: AdaptiveEnvironment,
    validation_blocks: list[np.ndarray],
    model: ActorCritic,
    mean: np.ndarray,
    std: np.ndarray,
) -> tuple[float, list[float]]:
    """Use several chronological blocks so a lucky final slice cannot select a policy."""
    rewards: list[float] = []
    for block in validation_blocks:
        if len(block) == 0:
            continue
        total = _simulate_policy(env, int(block[0]), int(block[-1]) + 1, model, mean, std)
        rewards.append(float(total / len(block)))
    if not rewards:
        return -np.inf, []
    # Robust selection: reward must be broadly useful, not merely strong in one block.
    return float(np.median(rewards) + 0.25 * np.min(rewards)), rewards


def train_seed(
    env: AdaptiveEnvironment,
    eligible: np.ndarray,
    year: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    cfg = env.cfg["ppo"]
    n_val = max(60, int(math.ceil(len(eligible) * float(cfg["validation_fraction"]))))
    purge_days = int(cfg.get("purge_days", 0))
    split = len(eligible) - n_val - purge_days
    if split < 20:
        raise ValueError("Not enough pre-purge rows for PPO training")
    train_rows = eligible[:split]
    val_rows = eligible[split + purge_days :]
    validation_blocks = [block for block in np.array_split(val_rows, int(cfg.get("validation_blocks", 1))) if len(block)]
    static = env.state.loc[train_rows, env.state_cols].to_numpy(dtype=float)
    mean = np.nanmean(static, axis=0)
    std = np.nanstd(static, axis=0)
    std = np.where(std > 1e-8, std, 1.0)
    model = ActorCritic(len(env.state_cols) + len(MODULES), len(env.specs), int(cfg["hidden_dim"]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=1e-4)

    bc_obs = torch.tensor(
        np.vstack([env.observation(int(index), np.zeros(len(MODULES)), mean, std) for index in train_rows]),
        dtype=torch.float32,
    )
    teacher_actions: list[int] = []
    noop_action = env.specs.index((20, "noop", 0.0))
    for index in train_rows:
        rewards = [env.step(int(index), action, np.zeros(len(MODULES)))[0] for action in range(len(env.specs))]
        best_action = int(np.argmax(rewards))
        if rewards[best_action] < float(cfg.get("minimum_teacher_edge", 0.02)):
            best_action = noop_action
        teacher_actions.append(best_action)
    teacher = torch.tensor(teacher_actions, dtype=torch.long)
    for _ in range(int(cfg["behavior_clone_epochs"])):
        logits, values = model(bc_obs)
        loss = nn.functional.cross_entropy(logits, teacher) + 0.02 * torch.mean(values**2)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    train_start, train_stop = int(train_rows[0]), int(train_rows[-1]) + 1
    best_state = copy.deepcopy(model.state_dict())
    best_val, best_block_rewards = _validation_score(env, validation_blocks, model, mean, std)
    wait = 0
    logs: list[dict[str, Any]] = []
    for epoch in range(1, int(cfg["max_epochs"]) + 1):
        index = train_start
        active = np.zeros(len(MODULES))
        observations: list[np.ndarray] = []
        actions: list[int] = []
        rewards: list[float] = []
        discounts: list[float] = []
        old_logp: list[float] = []
        values: list[float] = []
        while index < train_stop:
            obs_array = env.observation(index, active, mean, std)
            obs = torch.tensor(obs_array)[None, :]
            with torch.no_grad():
                logits, value = model(obs)
                distribution = torch.distributions.Categorical(logits=logits)
                action = int(distribution.sample().item())
                logp = float(distribution.log_prob(torch.tensor([action])).item())
            reward, next_index, active, info = env.step(index, action, active)
            observations.append(obs_array)
            actions.append(action)
            rewards.append(reward)
            discounts.append(float(cfg["gamma"]) ** (float(info.get("horizon_days", 20)) / 20.0))
            old_logp.append(logp)
            values.append(float(value.item()))
            index = max(index + 1, next_index)

        obs_tensor = torch.tensor(np.vstack(observations), dtype=torch.float32)
        action_tensor = torch.tensor(actions, dtype=torch.long)
        reward_array = np.asarray(rewards, dtype=float)
        value_array = np.asarray(values, dtype=float)
        advantage = np.zeros(len(rewards), dtype=float)
        last = 0.0
        next_value = 0.0
        for position in reversed(range(len(rewards))):
            delta = reward_array[position] + discounts[position] * next_value - value_array[position]
            last = delta + discounts[position] * float(cfg["gae_lambda"]) * last
            advantage[position] = last
            next_value = value_array[position]
        target = advantage + value_array
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        old_logp_tensor = torch.tensor(old_logp, dtype=torch.float32)
        advantage_tensor = torch.tensor(advantage, dtype=torch.float32)
        target_tensor = torch.tensor(target, dtype=torch.float32)
        for _ in range(int(cfg["updates_per_epoch"])):
            logits, value = model(obs_tensor)
            distribution = torch.distributions.Categorical(logits=logits)
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
            val_reward, validation_block_rewards = _validation_score(env, validation_blocks, model, mean, std)
            logs.append(
                {
                    "profile": env.profile,
                    "evaluation_year": year,
                    "seed": seed,
                    "epoch": epoch,
                    "rollout_steps": len(rewards),
                    "rollout_reward": float(reward_array.sum()),
                    "validation_reward": val_reward,
                    "validation_block_rewards": json.dumps(validation_block_rewards),
                    "entropy": float(entropy.detach()),
                }
            )
            if val_reward > best_val + float(cfg["early_stopping_min_delta"]):
                best_val = val_reward
                best_state = copy.deepcopy(model.state_dict())
                best_block_rewards = validation_block_rewards
                wait = 0
            else:
                wait += 1
                if wait >= int(cfg["early_stop_patience"]):
                    break
    return {
        "state_dict": best_state,
        "mean": mean,
        "std": std,
        "state_cols": env.state_cols,
        "obs_dim": len(env.state_cols) + len(MODULES),
        "action_dim": len(env.specs),
        "hidden_dim": int(cfg["hidden_dim"]),
        "training_cutoff": pd.Timestamp(year, 1, 1),
        "seed": seed,
        "purge_days": purge_days,
        "training_rows": len(train_rows),
        "validation_rows": len(val_rows),
        "validation_block_rewards": best_block_rewards,
        "validation_reward": best_val,
    }, logs


def choose_action(models: list[dict[str, Any]], env: AdaptiveEnvironment, index: int, active: np.ndarray) -> int:
    noop_20 = env.specs.index((20, "noop", 0.0))
    qualified = [
        checkpoint
        for checkpoint in models
        if float(checkpoint.get("validation_reward", -np.inf))
        >= float(env.cfg["ppo"].get("minimum_policy_validation_reward", 0.005))
    ]
    if len(qualified) < int(env.cfg["ppo"].get("minimum_qualified_models", 2)):
        return noop_20
    seed_actions: list[int] = []
    probability_rows: list[np.ndarray] = []
    for checkpoint in qualified:
        model = ActorCritic(checkpoint["obs_dim"], checkpoint["action_dim"], checkpoint["hidden_dim"])
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        obs = env.observation(index, active, checkpoint["mean"], checkpoint["std"])
        with torch.no_grad():
            logits, _ = model(torch.tensor(obs)[None, :])
            probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        seed_actions.append(int(probabilities.argmax()))
        probability_rows.append(probabilities)
    counts = pd.Series(seed_actions).value_counts()
    if int(counts.iloc[0]) < int(env.cfg["ppo"].get("ensemble_min_agreement", 2)):
        return noop_20
    action = int(counts.index[0])
    mean_probabilities = np.mean(np.vstack(probability_rows), axis=0)
    best_probability = float(mean_probabilities[action])
    runner_up = float(np.partition(mean_probabilities, -2)[-2])
    if best_probability < float(env.cfg["ppo"].get("minimum_action_probability", 0.0)):
        return noop_20
    if best_probability - runner_up < float(env.cfg["ppo"].get("minimum_action_margin", 0.0)):
        return noop_20
    horizon, _, level = env.specs[action]
    if abs(level) > 1e-12:
        reliability = float(env.state.loc[index, f"state_it_h{horizon}_reliability"])
        if reliability < float(env.cfg["actions"].get("minimum_reliability_for_active", 0.02)):
            return env.specs.index((horizon, "noop", 0.0))
    return action


def perf_stats(returns: pd.Series) -> dict[str, float]:
    values = pd.to_numeric(returns, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    nav = (1.0 + values).cumprod()
    if nav.empty:
        return {}
    annual_return = float(nav.iloc[-1] ** (252.0 / len(nav)) - 1.0)
    annual_vol = float(values.std(ddof=1) * np.sqrt(252.0))
    drawdown = nav / nav.cummax().clip(lower=1.0) - 1.0
    return {
        "days": len(values),
        "total_return": float(nav.iloc[-1] - 1.0),
        "annual_return": annual_return,
        "annual_vol": annual_vol,
        "sharpe": float(values.mean() * 252.0 / annual_vol) if annual_vol > 1e-12 else np.nan,
        "cagr_over_vol": annual_return / annual_vol if annual_vol > 1e-12 else np.nan,
        "max_drawdown": float(drawdown.min()),
        "calmar": annual_return / abs(float(drawdown.min())) if float(drawdown.min()) < 0 else np.nan,
    }


def run_profile(profile: str, state: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    sleeve_returns, base_weights, lo, hi = build_profile_market(profile, cfg)
    env = AdaptiveEnvironment(state, sleeve_returns, base_weights, lo, hi, profile, cfg)
    years = sorted(pd.to_datetime(env.state["date"]).dt.year.unique())
    checkpoints: dict[int, list[dict[str, Any]]] = {}
    training_logs: list[dict[str, Any]] = []
    model_dir = out_path(cfg, "model_dir") / profile
    model_dir.mkdir(parents=True, exist_ok=True)
    for year in years:
        cutoff = pd.Timestamp(int(year), 1, 1)
        eligible = env.eligible_rows(cutoff)
        if len(eligible) < int(cfg["ppo"]["min_train_rows"]):
            checkpoints[int(year)] = []
            continue
        annual_models: list[dict[str, Any]] = []
        for seed in cfg["ppo"]["seeds"]:
            checkpoint, logs = train_seed(env, eligible, int(year), int(seed))
            annual_models.append(checkpoint)
            training_logs.extend(logs)
        checkpoints[int(year)] = annual_models
        torch.save(annual_models, model_dir / f"policy_{int(year)}.pt")

    index = 0
    active = np.zeros(len(MODULES))
    daily_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    common_backtest_end = pd.Timestamp(env.state["date"].max())
    while index < len(env.state):
        date = pd.Timestamp(env.state.loc[index, "date"])
        models = checkpoints.get(date.year, [])
        qualified_model_count = sum(
            float(checkpoint.get("validation_reward", -np.inf))
            >= float(cfg["ppo"].get("minimum_policy_validation_reward", 0.005))
            for checkpoint in models
        )
        action = choose_action(models, env, index, active)
        reward, next_index, active, info = env.step(index, action, active)
        if not info.get("daily"):
            break
        visible_daily = [row for row in info["daily"] if pd.Timestamp(row["date"]) <= common_backtest_end]
        if not visible_daily:
            break
        if not models:
            source = "warmup_mechanical_noop_h20"
        elif qualified_model_count < int(cfg["ppo"].get("minimum_qualified_models", 2)):
            source = "validation_gate_noop_h20"
        else:
            source = "annual_walkforward_ppo"
        decision = {
            "date": date,
            "profile": profile,
            "policy_source": source,
            "policy_training_cutoff": pd.Timestamp(date.year, 1, 1) if models else pd.NaT,
            "qualified_model_count": qualified_model_count,
            "itransformer_information_cutoff": env.state.loc[index, "itransformer_information_cutoff"],
            "action_index": action,
            "action_name": info["action_name"],
            "action_level": info["action_level"],
            "horizon_days": info["horizon_days"],
            "execution_start": info["execution_start"],
            "execution_end": info["execution_end"],
            "reward": reward,
            "overlay_turnover": info["overlay_turnover"],
            "overlay_cost": info["overlay_cost"],
        }
        for sleeve_index, sleeve in enumerate(MODULES):
            decision[f"active_{sleeve}"] = float(active[sleeve_index])
        decision_rows.append(decision)
        daily_rows.extend(visible_daily)
        index = max(index + 1, next_index)

    daily = pd.DataFrame(daily_rows).drop_duplicates("date", keep="last").sort_values("date")
    decisions = pd.DataFrame(decision_rows).sort_values("date")
    daily["base_nav"] = (1.0 + daily["base_return"]).cumprod()
    daily["strategy_nav"] = (1.0 + daily["strategy_return"]).cumprod()
    metrics_rows: list[dict[str, Any]] = []
    for split, mask in {
        "full_oos": pd.Series(True, index=daily.index),
        "oos_2024_onward": pd.to_datetime(daily["date"]).ge(pd.Timestamp(cfg["gate"]["evaluation_start"])),
    }.items():
        base_stats = perf_stats(daily.loc[mask, "base_return"])
        strategy_stats = perf_stats(daily.loc[mask, "strategy_return"])
        metrics_rows.append(
            {
                "profile": profile,
                "split": split,
                **{f"base_{key}": value for key, value in base_stats.items()},
                **{f"strategy_{key}": value for key, value in strategy_stats.items()},
                "delta_annual_return": strategy_stats.get("annual_return", np.nan) - base_stats.get("annual_return", np.nan),
                "delta_sharpe": strategy_stats.get("sharpe", np.nan) - base_stats.get("sharpe", np.nan),
                "max_drawdown_worsening": abs(min(strategy_stats.get("max_drawdown", 0.0), 0.0)) - abs(min(base_stats.get("max_drawdown", 0.0), 0.0)),
                "mean_overlay_turnover": float(daily.loc[mask, "overlay_turnover"].mean()),
            }
        )
    metrics = pd.DataFrame(metrics_rows)
    for key, frame in [("daily_pattern", daily), ("decisions_pattern", decisions), ("metrics_pattern", metrics)]:
        path = out_path(cfg, key, profile)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".parquet":
            frame.to_parquet(path, index=False)
        else:
            frame.to_csv(path, index=False, encoding="utf-8-sig")
    pd.DataFrame(training_logs).to_csv(out_path(cfg, "training_log_pattern", profile), index=False, encoding="utf-8-sig")
    return {"profile": profile, "decisions": len(decisions), "days": len(daily), **metrics.iloc[-1].to_dict()}


def audit_and_report(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    profile_gates: list[dict[str, Any]] = []
    for profile in cfg["profiles"]:
        metrics = pd.read_csv(out_path(cfg, "metrics_pattern", profile))
        summaries.append(metrics)
        daily = pd.read_csv(out_path(cfg, "daily_pattern", profile), parse_dates=["date", "signal_date"])
        decisions = pd.read_parquet(out_path(cfg, "decisions_pattern", profile))
        weights = daily[[f"weight_{s}" for s in MODULES]].to_numpy(dtype=float)
        v1 = load_v1_config()
        bounds = v1["product"]["profiles"][profile]["sleeve_bounds"]
        lo = np.array([float(bounds[s][0]) for s in MODULES])
        hi = np.array([float(bounds[s][1]) for s in MODULES])
        checks = {
            "weight_sum_one": bool(np.allclose(weights.sum(axis=1), 1.0, atol=1e-6)),
            "sleeve_bounds": bool(np.all(weights >= lo - 1e-8) and np.all(weights <= hi + 1e-8)),
            "t_plus_one": bool((daily["signal_date"] < daily["date"]).all()),
            "daily_unique": bool(not daily["date"].duplicated().any()),
            "no_nan_return": bool(daily[["base_return", "strategy_return"]].notna().all().all()),
            "policy_cutoff_before_execution": bool(
                decisions.loc[decisions["policy_source"].eq("annual_walkforward_ppo"), "policy_training_cutoff"].lt(
                    decisions.loc[decisions["policy_source"].eq("annual_walkforward_ppo"), "execution_start"]
                ).all()
            ),
            "itransformer_cutoff_not_future": bool(decisions["itransformer_information_cutoff"].le(decisions["date"]).all()),
        }
        for check, passed in checks.items():
            audits.append({"profile": profile, "check": check, "passed": passed})
        evaluation = metrics.loc[metrics["split"].eq("oos_2024_onward")].iloc[0]
        evaluation_start = pd.Timestamp(cfg["gate"]["evaluation_start"])
        active_decisions = int(
            (
                pd.to_datetime(decisions["date"]).ge(evaluation_start)
                & pd.to_numeric(decisions["action_level"], errors="coerce").abs().gt(1e-12)
            ).sum()
        )
        checks_gate = {
            "annual_return": bool(evaluation["delta_annual_return"] >= float(cfg["gate"]["min_annual_return_delta"])),
            "sharpe": bool(evaluation["delta_sharpe"] >= float(cfg["gate"]["min_sharpe_delta"])),
            "drawdown": bool(evaluation["max_drawdown_worsening"] <= float(cfg["gate"]["max_drawdown_worsening"])),
            "active_decisions": bool(active_decisions >= int(cfg["gate"].get("min_active_decisions", 3))),
        }
        profile_gates.append(
            {
                "profile": profile,
                "enabled": bool(all(checks_gate.values())),
                "checks": checks_gate,
                "active_decisions": active_decisions,
                "delta_annual_return": float(evaluation["delta_annual_return"]),
                "delta_sharpe": float(evaluation["delta_sharpe"]),
                "max_drawdown_worsening": float(evaluation["max_drawdown_worsening"]),
            }
        )
    summary = pd.concat(summaries, ignore_index=True)
    audit = pd.DataFrame(audits)
    summary.to_csv(out_path(cfg, "summary"), index=False, encoding="utf-8-sig")
    audit.to_csv(out_path(cfg, "audit_csv"), index=False, encoding="utf-8-sig")
    gate_report = {
        "enabled_for_formal_strategy": bool(profile_gates) and all(item["enabled"] for item in profile_gates),
        "evaluation_start": cfg["gate"]["evaluation_start"],
        "profiles": profile_gates,
        "caveat": "A passing gate is necessary but not sufficient; research exposure to 2025+ means this is not a pristine untouched test.",
    }
    out_path(cfg, "gate_report").write_text(json.dumps(gate_report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Daily iTransformer + Adaptive PPO",
        "",
        "- iTransformer input: 120 trading-day daily sequence; output: five sleeve relative scores for 5/20/60 trading-day horizons.",
        "- PPO action: choose one of 18 auditable horizon/tilt actions. PPO never outputs unconstrained portfolio weights.",
        "- State: strict OOS iTransformer scores, causally matured model reliability, broad market states, and aggregated 38-asset mini-card diagnostics.",
        "- Execution: after-close decision, T+1 holding, no cash, official profile sleeve bounds, 5bp turnover cost.",
        "- Training: annual expanding walk-forward; only prior decisions whose 60-day labels matured before the annual cutoff may train PPO.",
        "",
        "## Metrics",
        "",
        summary.to_markdown(index=False),
        "",
        "## Audit",
        "",
        audit.to_markdown(index=False),
        "",
        "## Formal activation gate",
        "",
        f"- Enabled: `{gate_report['enabled_for_formal_strategy']}`",
        f"- Details: `{cfg['outputs']['gate_report']}`",
    ]
    out_path(cfg, "report_md").write_text("\n".join(lines), encoding="utf-8")
    return summary, audit

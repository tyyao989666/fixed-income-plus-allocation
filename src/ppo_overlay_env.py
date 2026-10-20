from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.itransformer_signal import RISK_SLEEVES


def project_simplex_bounds(values: np.ndarray, lo: np.ndarray, hi: np.ndarray, max_iter: int = 100) -> np.ndarray:
    x = np.clip(np.asarray(values, dtype=float), lo, hi)
    for _ in range(max_iter):
        diff = 1.0 - float(x.sum())
        if abs(diff) < 1e-12:
            break
        if diff > 0:
            room = hi - x
            active = room > 1e-12
            if not active.any():
                break
            x[active] += diff * room[active] / room[active].sum()
        else:
            room = x - lo
            active = room > 1e-12
            if not active.any():
                break
            x[active] += diff * room[active] / room[active].sum()
        x = np.clip(x, lo, hi)
    return x / x.sum()


@dataclass
class OverlayStep:
    date: pd.Timestamp
    reward: float
    gross_return: float
    turnover: float
    volatility_penalty: float
    drawdown_penalty: float
    active_deviation_penalty: float
    weights: dict[str, float]


class DailyPPOOverlayEnv:
    """Small dependency-free environment for training a sleeve-tilt PPO controller.

    Action is a five-dimensional vector in [-1, 1]. It is converted into bounded
    sleeve tilts around the current v1 sleeve weights, then projected back to the
    configured no-cash simplex. Rewards use next-horizon labels from the dataset.
    """

    def __init__(self, dataset: pd.DataFrame, cfg: dict[str, Any]):
        self.df = dataset.sort_values("date").reset_index(drop=True).copy()
        self.cfg = cfg
        self.profile = str(cfg["data"].get("profile", "balanced"))
        bounds = cfg["action_space"]["sleeve_bounds"][self.profile]
        self.lo = np.array([bounds[s][0] for s in RISK_SLEEVES], dtype=float)
        self.hi = np.array([bounds[s][1] for s in RISK_SLEEVES], dtype=float)
        self.max_tilt = float(cfg["action_space"].get("max_tilt_per_step", 0.03))
        self.action_deadband = float(cfg["action_space"].get("action_deadband", 0.0))
        self.horizon = int(cfg["data"]["horizon_days"])
        self.state_cols = [c for c in self.df.columns if c.startswith("state_")]
        self.weight_cols = [f"current_weight_{s}" for s in RISK_SLEEVES]
        self.label_return_cols = [f"label_next_{self.horizon}d_return_{s}" for s in RISK_SLEEVES]
        self.label_vol_cols = [f"label_next_{self.horizon}d_vol_{s}" for s in RISK_SLEEVES]
        self.label_dd_cols = [f"label_next_{self.horizon}d_maxdd_{s}" for s in RISK_SLEEVES]
        self.idx = 0

    def reset(self) -> np.ndarray:
        self.idx = 0
        return self._observation()

    def _observation(self) -> np.ndarray:
        row = self.df.iloc[self.idx]
        return row[self.state_cols + self.weight_cols].to_numpy(dtype=np.float32)

    def _weights_from_action(self, action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        row = self.df.iloc[self.idx]
        base = row[self.weight_cols].to_numpy(dtype=float)
        base = project_simplex_bounds(base, self.lo, self.hi)
        raw_action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        if self.action_deadband > 0.0:
            raw_action = np.where(np.abs(raw_action) < self.action_deadband, 0.0, raw_action)
        tilted = base + raw_action * self.max_tilt
        weights = project_simplex_bounds(tilted, self.lo, self.hi)
        return base, weights

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        row = self.df.iloc[self.idx]
        base, weights = self._weights_from_action(action)
        future_returns = row[self.label_return_cols].to_numpy(dtype=float)
        future_vols = row[self.label_vol_cols].to_numpy(dtype=float)
        future_dds = row[self.label_dd_cols].to_numpy(dtype=float)

        gross_return = float(np.dot(weights, future_returns))
        turnover = float(np.abs(weights - base).sum())
        reward_cfg = self.cfg.get("reward", {})
        turnover_cost = turnover * float(reward_cfg.get("turnover_cost_bps", 5.0)) / 10000.0
        vol_penalty = float(np.dot(weights, future_vols)) * float(reward_cfg.get("volatility_penalty", 0.15))
        dd_penalty = abs(float(np.dot(weights, future_dds))) * float(reward_cfg.get("drawdown_penalty", 0.35))
        active_penalty = turnover * float(reward_cfg.get("active_deviation_penalty", 0.20))
        reward = gross_return - turnover_cost - vol_penalty - dd_penalty - active_penalty

        step_info = OverlayStep(
            date=pd.Timestamp(row["date"]),
            reward=float(reward),
            gross_return=gross_return,
            turnover=turnover,
            volatility_penalty=vol_penalty,
            drawdown_penalty=dd_penalty,
            active_deviation_penalty=active_penalty,
            weights={s: float(w) for s, w in zip(RISK_SLEEVES, weights)},
        )
        self.idx += 1
        done = self.idx >= len(self.df)
        obs = np.zeros(len(self.state_cols) + len(self.weight_cols), dtype=np.float32) if done else self._observation()
        return obs, float(reward), done, step_info.__dict__

    @property
    def observation_dim(self) -> int:
        return len(self.state_cols) + len(self.weight_cols)

    @property
    def action_dim(self) -> int:
        return len(RISK_SLEEVES)

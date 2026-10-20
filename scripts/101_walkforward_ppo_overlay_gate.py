from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.daily_rl_dataset import build_daily_rl_overlay_dataset, load_overlay_config  # noqa: E402
from src.itransformer_signal import RISK_SLEEVES  # noqa: E402
from src.ppo_overlay_env import DailyPPOOverlayEnv  # noqa: E402


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int, log_std_init: float = -2.0):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)
        self.log_std = nn.Parameter(torch.full((action_dim,), float(log_std_init)))

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.backbone(obs)
        mean = torch.tanh(self.actor(h))
        value = self.critic(h).squeeze(-1)
        std = torch.exp(self.log_std).expand_as(mean)
        return mean, std, value


def load_model(profile: str, cfg: dict) -> ActorCritic:
    path = ROOT / "data_processed" / f"daily_ppo_overlay_controller_{profile}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing trained model for {profile}: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model_cfg = checkpoint.get("config", cfg)
    model = ActorCritic(
        int(checkpoint["obs_dim"]),
        int(checkpoint["action_dim"]),
        int(model_cfg["ppo"]["hidden_dim"]),
        float(model_cfg["ppo"].get("log_std_init", -2.0)),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def eval_rows(df: pd.DataFrame, cfg: dict, model: ActorCritic) -> pd.DataFrame:
    env_model = DailyPPOOverlayEnv(df, cfg)
    env_noop = DailyPPOOverlayEnv(df, cfg)
    obs_model = env_model.reset()
    obs_noop = env_noop.reset()
    rows = []
    done = False
    while not done:
        with torch.no_grad():
            mean, _, _ = model(torch.tensor(obs_model, dtype=torch.float32).unsqueeze(0))
        model_action = mean.squeeze(0).numpy()
        obs_model, model_reward, done_model, model_info = env_model.step(model_action)
        obs_noop, noop_reward, done_noop, noop_info = env_noop.step(np.zeros(env_noop.action_dim))
        done = done_model or done_noop
        rows.append(
            {
                "date": model_info["date"],
                "noop_reward": float(noop_reward),
                "model_reward": float(model_reward),
                "noop_gross_return": float(noop_info["gross_return"]),
                "model_gross_return": float(model_info["gross_return"]),
                "model_turnover": float(model_info["turnover"]),
            }
        )
    return pd.DataFrame(rows)


def current_state_snapshot(row: pd.Series) -> dict[str, object]:
    snapshot: dict[str, object] = {}
    for sleeve in RISK_SLEEVES:
        snapshot[f"current_weight_{sleeve}"] = float(row.get(f"current_weight_{sleeve}", np.nan))
    state_flags = [c for c in row.index if c.startswith("state_") and pd.notna(row[c]) and abs(float(row[c])) > 0.5]
    keep = [
        c
        for c in state_flags
        if any(token in c for token in ["cr_state_", "r_state_", "cb_state_", "eq_state_", "alt_state_", "rb_state_"])
    ]
    snapshot["active_state_flags"] = ";".join(sorted(keep)[:12])
    return snapshot


def walkforward_gate(
    df: pd.DataFrame,
    perf: pd.DataFrame,
    cfg: dict,
    profile: str,
    lookback: int,
    min_obs: int,
    reward_buffer: float,
) -> tuple[pd.DataFrame, dict]:
    data = df.merge(perf, on="date", how="left").sort_values("date").reset_index(drop=True)
    rows = []
    for idx, row in data.iterrows():
        hist = data.iloc[max(0, idx - lookback) : idx].copy()
        if len(hist) >= min_obs:
            model_edge = float((hist["model_reward"] - hist["noop_reward"]).mean())
            enable_model = model_edge > reward_buffer
            reason = "rolling_model_edge_positive" if enable_model else "rolling_model_not_better_than_noop"
        else:
            model_edge = np.nan
            enable_model = False
            reason = "insufficient_realized_overlay_history"
        chosen = "model" if enable_model else "noop"
        chosen_reward = float(row["model_reward"] if enable_model else row["noop_reward"])
        chosen_gross = float(row["model_gross_return"] if enable_model else row["noop_gross_return"])
        chosen_turnover = float(row["model_turnover"] if enable_model else 0.0)
        card = {
            "profile": profile,
            "date": row["date"],
            "rolling_lookback": lookback,
            "rolling_obs": len(hist),
            "rolling_model_edge": model_edge,
            "enable_model": bool(enable_model),
            "chosen_policy": chosen,
            "decision_reason": reason,
            "chosen_reward": chosen_reward,
            "chosen_gross_return": chosen_gross,
            "chosen_turnover": chosen_turnover,
            "noop_reward": float(row["noop_reward"]),
            "model_reward": float(row["model_reward"]),
            "noop_gross_return": float(row["noop_gross_return"]),
            "model_gross_return": float(row["model_gross_return"]),
            "model_turnover": float(row["model_turnover"]),
        }
        card.update(current_state_snapshot(row))
        rows.append(card)
    cards = pd.DataFrame(rows)
    test_start = pd.Timestamp(cfg["data"]["val_end"]) + pd.offsets.Day(1)
    test = cards[pd.to_datetime(cards["date"]).ge(test_start)].copy()
    summary = {
        "profile": profile,
        "rows": int(len(cards)),
        "test_rows": int(len(test)),
        "test_start": str(test["date"].min().date()) if not test.empty else "",
        "test_end": str(test["date"].max().date()) if not test.empty else "",
        "test_enable_ratio": float(test["enable_model"].mean()) if not test.empty else float("nan"),
        "test_chosen_mean_reward": float(test["chosen_reward"].mean()) if not test.empty else float("nan"),
        "test_noop_mean_reward": float(test["noop_reward"].mean()) if not test.empty else float("nan"),
        "test_model_mean_reward": float(test["model_reward"].mean()) if not test.empty else float("nan"),
        "test_chosen_mean_gross_return": float(test["chosen_gross_return"].mean()) if not test.empty else float("nan"),
        "test_noop_mean_gross_return": float(test["noop_gross_return"].mean()) if not test.empty else float("nan"),
        "test_model_mean_gross_return": float(test["model_gross_return"].mean()) if not test.empty else float("nan"),
        "test_chosen_mean_turnover": float(test["chosen_turnover"].mean()) if not test.empty else float("nan"),
    }
    summary["test_chosen_minus_noop_reward"] = summary["test_chosen_mean_reward"] - summary["test_noop_mean_reward"]
    summary["test_chosen_minus_noop_gross_return"] = summary["test_chosen_mean_gross_return"] - summary["test_noop_mean_gross_return"]
    return cards, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="State-card walk-forward gate for the PPO overlay.")
    parser.add_argument("--lookback", type=int, default=12)
    parser.add_argument("--min-obs", type=int, default=6)
    parser.add_argument("--reward-buffer", type=float, default=0.0)
    args = parser.parse_args()

    all_summaries = []
    for profile in ["low_vol", "balanced", "high_vol"]:
        cfg = load_overlay_config()
        cfg["data"]["profile"] = profile
        cfg["data"]["strategy_version"] = "no_agent"
        df = build_daily_rl_overlay_dataset(cfg)
        model = load_model(profile, cfg)
        perf = eval_rows(df, cfg, model)
        cards, summary = walkforward_gate(df, perf, cfg, profile, args.lookback, args.min_obs, args.reward_buffer)
        card_path = ROOT / "output" / "reports" / f"daily_ppo_overlay_walkforward_cards_{profile}.csv"
        cards.to_csv(card_path, index=False, encoding="utf-8-sig")
        all_summaries.append(summary)
    summary_df = pd.DataFrame(all_summaries)
    summary_path = ROOT / "output" / "reports" / "daily_ppo_overlay_walkforward_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(json.dumps({"summary": str(summary_path.relative_to(ROOT)), "profiles": len(summary_df)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

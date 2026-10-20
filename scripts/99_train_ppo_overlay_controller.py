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

from src.daily_rl_dataset import load_overlay_config, overlay_out_path  # noqa: E402
from src.itransformer_signal import RISK_SLEEVES  # noqa: E402
from src.ppo_overlay_env import DailyPPOOverlayEnv  # noqa: E402


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int, log_std_init: float = -0.7):
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

    def dist_value(self, obs: torch.Tensor) -> tuple[torch.distributions.Normal, torch.Tensor]:
        mean, std, value = self(obs)
        return torch.distributions.Normal(mean, std), value


def collect_rollout(env: DailyPPOOverlayEnv, model: ActorCritic) -> dict[str, torch.Tensor]:
    obs_list, action_list, logp_list, reward_list, value_list = [], [], [], [], []
    obs = env.reset()
    done = False
    while not done:
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            dist, value = model.dist_value(obs_t)
            action = dist.sample()
            action_clipped = torch.clamp(action, -1.0, 1.0)
            logp = dist.log_prob(action).sum(axis=-1)
        next_obs, reward, done, _ = env.step(action_clipped.squeeze(0).numpy())
        obs_list.append(obs_t.squeeze(0))
        action_list.append(action_clipped.squeeze(0))
        logp_list.append(logp.squeeze(0))
        reward_list.append(torch.tensor(float(reward), dtype=torch.float32))
        value_list.append(value.squeeze(0))
        obs = next_obs
    return {
        "obs": torch.stack(obs_list),
        "action": torch.stack(action_list),
        "logp": torch.stack(logp_list),
        "reward": torch.stack(reward_list),
        "value": torch.stack(value_list),
    }


def advantages(reward: torch.Tensor, value: torch.Tensor, gamma: float, lam: float) -> tuple[torch.Tensor, torch.Tensor]:
    adv = torch.zeros_like(reward)
    last_gae = torch.tensor(0.0)
    next_value = torch.tensor(0.0)
    for t in reversed(range(len(reward))):
        delta = reward[t] + gamma * next_value - value[t]
        last_gae = delta + gamma * lam * last_gae
        adv[t] = last_gae
        next_value = value[t]
    ret = adv + value
    adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
    return adv.detach(), ret.detach()


def deterministic_eval(df: pd.DataFrame, cfg: dict, model: ActorCritic | None, policy: str) -> dict:
    env = DailyPPOOverlayEnv(df, cfg)
    obs = env.reset()
    rewards, gross, turnover = [], [], []
    done = False
    while not done:
        if policy == "model":
            with torch.no_grad():
                mean, _, _ = model(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))  # type: ignore[arg-type]
            action = mean.squeeze(0).numpy()
        elif policy == "noop":
            action = np.zeros(env.action_dim)
        else:
            raise ValueError(policy)
        obs, reward, done, info = env.step(action)
        rewards.append(reward)
        gross.append(float(info["gross_return"]))
        turnover.append(float(info["turnover"]))
    return {
        "policy": policy,
        "steps": len(rewards),
        "mean_reward": float(np.mean(rewards)) if rewards else float("nan"),
        "sum_reward": float(np.sum(rewards)) if rewards else float("nan"),
        "mean_gross_return": float(np.mean(gross)) if gross else float("nan"),
        "mean_turnover": float(np.mean(turnover)) if turnover else float("nan"),
    }


def oracle_action_targets(df: pd.DataFrame, cfg: dict) -> tuple[torch.Tensor, torch.Tensor]:
    horizon = int(cfg["data"]["horizon_days"])
    reward_cfg = cfg.get("reward", {})
    edge_threshold = float(cfg["ppo"].get("behavior_clone_edge_threshold", 0.002))
    action_scale = float(cfg["ppo"].get("behavior_clone_action_scale", 0.55))
    returns = df[[f"label_next_{horizon}d_return_{s}" for s in RISK_SLEEVES]].to_numpy(dtype=float)
    vols = df[[f"label_next_{horizon}d_vol_{s}" for s in RISK_SLEEVES]].to_numpy(dtype=float)
    dds = df[[f"label_next_{horizon}d_maxdd_{s}" for s in RISK_SLEEVES]].to_numpy(dtype=float)
    scores = (
        returns
        - float(reward_cfg.get("volatility_penalty", 0.15)) * vols
        - float(reward_cfg.get("drawdown_penalty", 0.35)) * np.abs(dds)
    )
    centered = scores - scores.mean(axis=1, keepdims=True)
    spread = scores.max(axis=1) - scores.min(axis=1)
    denom = np.maximum(np.std(centered, axis=1, keepdims=True), 1e-6)
    action = np.tanh(centered / denom) * action_scale
    action[spread < edge_threshold] = 0.0
    obs_cols = [c for c in df.columns if c.startswith("state_")] + [f"current_weight_{s}" for s in RISK_SLEEVES]
    return (
        torch.tensor(df[obs_cols].to_numpy(dtype=float), dtype=torch.float32),
        torch.tensor(action, dtype=torch.float32),
    )


def pretrain_actor_critic(model: ActorCritic, train_df: pd.DataFrame, cfg: dict) -> list[dict]:
    epochs = int(cfg["ppo"].get("behavior_clone_epochs", 0))
    if epochs <= 0:
        return []
    obs, target_action = oracle_action_targets(train_df, cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["ppo"].get("behavior_clone_learning_rate", 0.0005)))
    value_weight = float(cfg["ppo"].get("behavior_clone_value_weight", 0.20))
    env = DailyPPOOverlayEnv(train_df, cfg)
    noop_reward = deterministic_eval(train_df, cfg, None, "noop")["sum_reward"]
    rows = []
    for epoch in range(1, epochs + 1):
        mean, _, value = model(obs)
        bc_loss = torch.mean((mean - target_action) ** 2)
        value_target = torch.full_like(value, float(noop_reward) / max(len(train_df), 1))
        value_loss = torch.mean((value - value_target) ** 2)
        loss = bc_loss + value_weight * value_loss
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
            eval_row = deterministic_eval(train_df, cfg, model, "model")
            rows.append(
                {
                    "stage": "behavior_clone",
                    "epoch": epoch,
                    "bc_loss": float(bc_loss.detach()),
                    "value_loss": float(value_loss.detach()),
                    "train_model_sum_reward": eval_row["sum_reward"],
                    "train_model_mean_turnover": eval_row["mean_turnover"],
                    "target_nonzero_ratio": float((target_action.abs().sum(axis=1) > 0).float().mean()),
                    "observation_dim": env.observation_dim,
                }
            )
    return rows


def date_split(df: pd.DataFrame, cfg: dict) -> dict[str, pd.DataFrame]:
    dates = pd.to_datetime(df["date"])
    train_end = pd.Timestamp(cfg["data"]["train_end"])
    val_end = pd.Timestamp(cfg["data"]["val_end"])
    return {
        "train": df[dates <= train_end].copy(),
        "val": df[(dates > train_end) & (dates <= val_end)].copy(),
        "test": df[dates > val_end].copy(),
        "all": df.copy(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a lightweight PPO sleeve-tilt overlay controller.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--profile", default=None, choices=["low_vol", "balanced", "high_vol"])
    parser.add_argument("--strategy-version", default=None, choices=["no_agent", "agentic"])
    parser.add_argument("--output-suffix", default="")
    args = parser.parse_args()

    cfg = load_overlay_config(Path(args.config) if args.config else None)
    if args.profile:
        cfg["data"]["profile"] = args.profile
    if args.strategy_version:
        cfg["data"]["strategy_version"] = args.strategy_version
    if args.output_suffix:
        for key in ["ppo_model", "ppo_training_log", "ppo_evaluation"]:
            path = Path(str(cfg["outputs"][key]))
            cfg["outputs"][key] = str(path.with_name(f"{path.stem}{args.output_suffix}{path.suffix}")).replace("\\", "/")
    torch.manual_seed(int(cfg["ppo"]["seed"]))
    np.random.seed(int(cfg["ppo"]["seed"]))
    df = pd.read_parquet(overlay_out_path(cfg, "dataset"))
    splits = date_split(df, cfg)
    train_df = splits["train"]
    val_df = splits["val"]
    if train_df.empty:
        raise RuntimeError("No train rows. Build dataset first and check train_end.")
    env = DailyPPOOverlayEnv(train_df, cfg)
    model = ActorCritic(
        env.observation_dim,
        env.action_dim,
        int(cfg["ppo"]["hidden_dim"]),
        float(cfg["ppo"].get("log_std_init", -0.7)),
    )
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["ppo"]["learning_rate"]))
    epochs = int(args.epochs or cfg["ppo"]["epochs"])
    log_rows = pretrain_actor_critic(model, train_df, cfg)
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_stage = "behavior_clone"
    best_epoch = 0
    best_val_reward = deterministic_eval(val_df, cfg, model, "model")["sum_reward"] if not val_df.empty else float("-inf")
    if log_rows:
        pd.DataFrame(log_rows).to_csv(overlay_out_path(cfg, "ppo_training_log"), index=False, encoding="utf-8-sig")
        print(json.dumps(log_rows[-1], ensure_ascii=False), flush=True)

    for epoch in range(1, epochs + 1):
        batch = collect_rollout(env, model)
        adv, ret = advantages(batch["reward"], batch["value"], float(cfg["ppo"]["gamma"]), float(cfg["ppo"]["gae_lambda"]))
        old_logp = batch["logp"].detach()
        for _ in range(int(cfg["ppo"]["updates_per_epoch"])):
            dist, value = model.dist_value(batch["obs"])
            logp = dist.log_prob(batch["action"]).sum(axis=-1)
            ratio = torch.exp(logp - old_logp)
            clipped = torch.clamp(ratio, 1.0 - float(cfg["ppo"]["clip_ratio"]), 1.0 + float(cfg["ppo"]["clip_ratio"])) * adv
            policy_loss = -torch.min(ratio * adv, clipped).mean()
            value_loss = torch.mean((value - ret) ** 2)
            entropy = dist.entropy().sum(axis=-1).mean()
            loss = policy_loss + float(cfg["ppo"]["value_loss_weight"]) * value_loss - float(cfg["ppo"]["entropy_weight"]) * entropy
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        if epoch == 1 or epoch % max(1, int(args.progress_every)) == 0 or epoch == epochs:
            train_eval = deterministic_eval(train_df, cfg, model, "model")
            val_eval = deterministic_eval(val_df, cfg, model, "model") if not val_df.empty else {"sum_reward": float("nan")}
            val_reward = float(val_eval["sum_reward"])
            if not np.isnan(val_reward) and val_reward > best_val_reward:
                best_val_reward = val_reward
                best_stage = "ppo"
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            row = {
                "stage": "ppo",
                "epoch": epoch,
                "rollout_reward": float(batch["reward"].sum()),
                "policy_loss": float(policy_loss.detach()),
                "value_loss": float(value_loss.detach()),
                "entropy": float(entropy.detach()),
                "train_model_sum_reward": train_eval["sum_reward"],
                "train_model_mean_turnover": train_eval["mean_turnover"],
                "val_model_sum_reward": val_eval["sum_reward"],
                "best_val_sum_reward": best_val_reward,
            }
            log_rows.append(row)
            pd.DataFrame(log_rows).to_csv(overlay_out_path(cfg, "ppo_training_log"), index=False, encoding="utf-8-sig")
            print(json.dumps(row, ensure_ascii=False), flush=True)

    model_path = overlay_out_path(cfg, "ppo_model")
    model.load_state_dict(best_state)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": cfg,
            "obs_dim": env.observation_dim,
            "action_dim": env.action_dim,
            "best_stage": best_stage,
            "best_epoch": best_epoch,
            "best_val_reward": best_val_reward,
        },
        model_path,
    )
    pd.DataFrame(log_rows).to_csv(overlay_out_path(cfg, "ppo_training_log"), index=False, encoding="utf-8-sig")

    evaluation = {}
    for split, split_df in splits.items():
        if split_df.empty:
            continue
        evaluation[split] = {
            "noop": deterministic_eval(split_df, cfg, None, "noop"),
            "model": deterministic_eval(split_df, cfg, model, "model"),
        }
    eval_path = overlay_out_path(cfg, "ppo_evaluation")
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    eval_path.write_text(json.dumps(evaluation, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "model": str(model_path.relative_to(ROOT)),
                "evaluation": str(eval_path.relative_to(ROOT)),
                "epochs": epochs,
                "best_stage": best_stage,
                "best_epoch": best_epoch,
                "best_val_reward": best_val_reward,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

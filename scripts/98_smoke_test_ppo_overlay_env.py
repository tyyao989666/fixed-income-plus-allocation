from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.daily_rl_dataset import load_overlay_config, overlay_out_path  # noqa: E402
from src.ppo_overlay_env import DailyPPOOverlayEnv  # noqa: E402


def run_policy(env: DailyPPOOverlayEnv, policy: str) -> dict:
    rng = np.random.default_rng(42)
    obs = env.reset()
    del obs
    rewards = []
    gross = []
    turnover = []
    while True:
        if policy == "noop":
            action = np.zeros(env.action_dim)
        elif policy == "random_small":
            action = rng.normal(0.0, 0.25, size=env.action_dim)
        else:
            raise ValueError(policy)
        _, reward, done, info = env.step(action)
        rewards.append(reward)
        gross.append(float(info["gross_return"]))
        turnover.append(float(info["turnover"]))
        if done:
            break
    return {
        "policy": policy,
        "steps": len(rewards),
        "mean_reward": float(np.mean(rewards)),
        "sum_reward": float(np.sum(rewards)),
        "mean_gross_return": float(np.mean(gross)),
        "mean_turnover": float(np.mean(turnover)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test the daily PPO overlay environment.")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = load_overlay_config(Path(args.config) if args.config else None)
    dataset_path = overlay_out_path(cfg, "dataset")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Build dataset first: {dataset_path}")
    df = pd.read_parquet(dataset_path)
    env = DailyPPOOverlayEnv(df, cfg)
    report = {
        "dataset": str(dataset_path.relative_to(ROOT)),
        "rows": len(df),
        "observation_dim": env.observation_dim,
        "action_dim": env.action_dim,
        "profile": cfg["data"]["profile"],
        "strategy_version": cfg["data"]["strategy_version"],
        "policies": [run_policy(env, "noop"), run_policy(env, "random_small")],
    }
    out = overlay_out_path(cfg, "env_smoke_test")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()

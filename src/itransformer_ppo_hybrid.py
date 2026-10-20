from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

from src.itransformer_signal import RISK_SLEEVES
from src.ppo_overlay_env import project_simplex_bounds
from src.v1_strategy import V1Paths, calc_metrics, load_config, load_monthly_returns


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "itransformer_ppo_config.yaml"


def load_hybrid_config(path: Path | None = None) -> dict[str, Any]:
    with (path or DEFAULT_CONFIG).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def cfg_path(cfg: dict[str, Any], key: str, profile: str | None = None) -> Path:
    value = str(cfg["outputs"][key])
    if profile is not None:
        value = value.format(profile=profile)
    return ROOT / value


def data_path(cfg: dict[str, Any], key: str, profile: str | None = None) -> Path:
    value = str(cfg["data"][key])
    if profile is not None:
        value = value.format(profile=profile)
    return ROOT / value


def _profile_config(profile: str) -> dict[str, Any]:
    cfg = load_config()
    return copy.deepcopy(cfg["product"]["profiles"][profile])


def _monthly_returns(profile: str) -> pd.DataFrame:
    strategy_cfg = copy.deepcopy(load_config())
    strategy_cfg["strategy_version"] = "no_agent"
    strategy_cfg["product"]["selected_profile"] = profile
    return load_monthly_returns(V1Paths(), strategy_cfg)


def _profile_bounds(profile: str) -> tuple[np.ndarray, np.ndarray]:
    bounds = _profile_config(profile)["sleeve_bounds"]
    lo = np.array([float(bounds[s][0]) for s in RISK_SLEEVES], dtype=float)
    hi = np.array([float(bounds[s][1]) for s in RISK_SLEEVES], dtype=float)
    return lo, hi


def _preferred_asset_weights(profile: str, sleeve: str) -> dict[str, float]:
    raw = (
        _profile_config(profile)
        .get("asset_overrides", {})
        .get(sleeve, {})
        .get("preferred_asset_weights", {})
    )
    values = {str(k): max(float(v), 0.0) for k, v in raw.items()}
    total = sum(values.values())
    return {k: v / total for k, v in values.items()} if total > 0 else {}


def _rank_ic(scores: np.ndarray, realized: np.ndarray) -> float:
    if not np.isfinite(scores).all() or not np.isfinite(realized).all():
        return float("nan")
    return float(pd.Series(scores).corr(pd.Series(realized), method="spearman"))


def _candidate_weights(
    base: np.ndarray,
    scores: np.ndarray,
    level: float,
    max_transfer: float,
    min_spread: float,
    lo: np.ndarray,
    hi: np.ndarray,
) -> np.ndarray:
    base = project_simplex_bounds(base, lo, hi)
    if float(np.nanmax(scores) - np.nanmin(scores)) < min_spread:
        return base.copy()
    top = int(np.nanargmax(scores))
    bottom = int(np.nanargmin(scores))
    if level < 0:
        top, bottom = bottom, top
    transfer = min(abs(level) * max_transfer, hi[top] - base[top], base[bottom] - lo[bottom])
    out = base.copy()
    if transfer > 0:
        out[top] += transfer
        out[bottom] -= transfer
    return project_simplex_bounds(out, lo, hi)


def _next_month_map(dates: pd.Series) -> dict[pd.Timestamp, pd.Timestamp]:
    unique = sorted(pd.to_datetime(dates, errors="coerce").dropna().unique())
    return {pd.Timestamp(unique[i]): pd.Timestamp(unique[i + 1]) for i in range(len(unique) - 1)}


def _sleeve_next_returns(
    profile: str,
    decision_date: pd.Timestamp,
    next_date: pd.Timestamp,
    asset_weights: pd.DataFrame,
    returns: pd.DataFrame,
) -> np.ndarray:
    current = asset_weights[asset_weights["date"].eq(decision_date)].copy()
    next_returns = returns[returns["date"].eq(next_date)].set_index("asset_code")["return_1m"]
    values: list[float] = []
    for sleeve in RISK_SLEEVES:
        sub = current[current["module"].eq(sleeve)].copy()
        sub["return_1m"] = sub["asset_code"].map(next_returns)
        usable = sub[sub["return_1m"].notna()].copy()
        if not usable.empty and float(usable["weight"].sum()) > 1e-12:
            shares = usable["weight"] / usable["weight"].sum()
            values.append(float(np.dot(shares, usable["return_1m"])))
            continue
        preferred = _preferred_asset_weights(profile, sleeve)
        pairs = [(code, weight, next_returns.get(code, np.nan)) for code, weight in preferred.items()]
        pairs = [(code, weight, ret) for code, weight, ret in pairs if pd.notna(ret)]
        if not pairs:
            raise ValueError(f"No next-month return for {profile}/{sleeve} at {decision_date:%Y-%m-%d}")
        total = sum(weight for _, weight, _ in pairs)
        values.append(float(sum(weight * float(ret) for _, weight, ret in pairs) / total))
    return np.asarray(values, dtype=float)


def build_profile_dataset(profile: str, cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    cfg = cfg or load_hybrid_config()
    predictions = pd.read_parquet(data_path(cfg, "predictions"))
    predictions["as_of_date"] = pd.to_datetime(predictions["as_of_date"], errors="coerce")
    predictions["training_cutoff"] = pd.to_datetime(predictions["training_cutoff"], errors="coerce")
    predictions = predictions[predictions["sleeve"].isin(RISK_SLEEVES)].copy()
    score_wide = predictions.pivot(index="as_of_date", columns="sleeve", values="pred_return")
    score_wide = score_wide.reindex(columns=RISK_SLEEVES).dropna()
    cutoff = predictions.groupby("as_of_date")["training_cutoff"].max()
    model_weight = predictions.groupby("as_of_date")["itransformer_model_weight"].mean()

    sleeve_weights = pd.read_parquet(data_path(cfg, "baseline_sleeve_weights_pattern", profile))
    sleeve_weights["date"] = pd.to_datetime(sleeve_weights["date"], errors="coerce")
    asset_weights = pd.read_parquet(data_path(cfg, "baseline_asset_weights_pattern", profile))
    asset_weights["date"] = pd.to_datetime(asset_weights["date"], errors="coerce")
    returns = _monthly_returns(profile)
    returns["date"] = pd.to_datetime(returns["date"], errors="coerce")
    returns["return_1m"] = pd.to_numeric(returns["return_1m"], errors="coerce")
    next_dates = _next_month_map(returns["date"])
    sleeve_index = sleeve_weights.set_index("date")
    lo, hi = _profile_bounds(profile)
    action_names = list(cfg["actions"]["names"])
    action_levels = [float(v) for v in cfg["actions"]["levels"]]
    max_transfer = float(cfg["actions"]["max_pair_transfer"][profile])
    min_spread = float(cfg["actions"]["min_score_spread"])
    reward_cfg = cfg["reward"]
    start = pd.Timestamp(cfg["data"]["start_date"])
    end = pd.Timestamp(cfg["data"]["end_date"])
    rows: list[dict[str, Any]] = []

    for date, score_row in score_wide.sort_index().iterrows():
        date = pd.Timestamp(date)
        if date < start or date > end or date not in sleeve_index.index or date not in next_dates:
            continue
        next_date = next_dates[date]
        base = sleeve_index.loc[date, RISK_SLEEVES]
        if isinstance(base, pd.DataFrame):
            base = base.iloc[-1]
        base_values = project_simplex_bounds(base.to_numpy(dtype=float), lo, hi)
        raw_scores = score_row.to_numpy(dtype=float)
        centered = raw_scores - float(np.mean(raw_scores))
        scale = max(float(np.std(centered)), 1e-8)
        scores = np.clip(centered / scale, -3.0, 3.0)
        next_sleeve_returns = _sleeve_next_returns(profile, date, next_date, asset_weights, returns)
        baseline_gross = float(np.dot(base_values, next_sleeve_returns))
        row: dict[str, Any] = {
            "date": date,
            "next_date": next_date,
            "profile": profile,
            "itransformer_training_cutoff": cutoff.loc[date],
            "state_itransformer_model_weight": float(model_weight.loc[date]),
            "state_itransformer_spread": float(np.max(scores) - np.min(scores)),
            "baseline_gross_return": baseline_gross,
            "realized_rank_ic": _rank_ic(scores, next_sleeve_returns),
        }
        for idx, sleeve in enumerate(RISK_SLEEVES):
            row[f"state_itransformer_score_{sleeve}"] = float(scores[idx])
            row[f"state_base_weight_{sleeve}"] = float(base_values[idx])
            row[f"next_return_{sleeve}"] = float(next_sleeve_returns[idx])
        for name, level in zip(action_names, action_levels):
            candidate = _candidate_weights(base_values, scores, level, max_transfer, min_spread, lo, hi)
            gross = float(np.dot(candidate, next_sleeve_returns))
            active_l1 = float(np.abs(candidate - base_values).sum())
            cost = active_l1 * float(reward_cfg["transaction_cost_bps"]) / 10000.0
            downside_worsening = max(0.0, -gross) - max(0.0, -baseline_gross)
            reward = (
                gross
                - baseline_gross
                - cost
                - float(reward_cfg["downside_penalty"]) * max(0.0, downside_worsening)
                - float(reward_cfg["active_deviation_penalty"]) * active_l1
            ) * float(reward_cfg["reward_scale"])
            row[f"candidate_gross_{name}"] = gross
            row[f"candidate_reward_{name}"] = reward
            row[f"candidate_active_l1_{name}"] = active_l1
            for idx, sleeve in enumerate(RISK_SLEEVES):
                row[f"candidate_weight_{name}_{sleeve}"] = float(candidate[idx])
        rows.append(row)

    out = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    realized_ic = pd.to_numeric(out["realized_rank_ic"], errors="coerce")
    out["state_itransformer_reliability_6"] = realized_ic.shift(1).rolling(6, min_periods=2).mean().fillna(0.0)
    out["state_itransformer_reliability_12"] = realized_ic.shift(1).rolling(12, min_periods=3).mean().fillna(0.0)
    out["state_reliability_history_confidence"] = np.minimum(np.arange(len(out), dtype=float) / 12.0, 1.0)
    for sleeve in RISK_SLEEVES:
        realized = pd.to_numeric(out[f"next_return_{sleeve}"], errors="coerce").shift(1)
        out[f"state_realized_momentum_3m_{sleeve}"] = realized.rolling(3, min_periods=1).sum().fillna(0.0)
    path = cfg_path(cfg, "dataset_pattern", profile)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    return out


def state_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("state_")]


class CategoricalActorCritic(nn.Module):
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


def _advantages(reward: torch.Tensor, value: torch.Tensor, gamma: float, lam: float) -> tuple[torch.Tensor, torch.Tensor]:
    adv = torch.zeros_like(reward)
    next_value = torch.tensor(0.0)
    last = torch.tensor(0.0)
    for idx in reversed(range(len(reward))):
        delta = reward[idx] + gamma * next_value - value[idx]
        last = delta + gamma * lam * last
        adv[idx] = last
        next_value = value[idx]
    target = adv + value
    adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
    return adv.detach(), target.detach()


def _deterministic_reward(model: CategoricalActorCritic, obs: torch.Tensor, rewards: torch.Tensor) -> float:
    with torch.no_grad():
        logits, _ = model(obs)
        actions = logits.argmax(dim=1)
    return float(rewards.gather(1, actions[:, None]).sum())


def _train_seed(
    history: pd.DataFrame,
    cfg: dict[str, Any],
    seed: int,
    profile: str,
    evaluation_year: int,
) -> tuple[dict[str, torch.Tensor], dict[str, Any], list[dict[str, Any]]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    ppo_cfg = cfg["ppo"]
    actions = list(cfg["actions"]["names"])
    states = state_columns(history)
    n_val = max(2, int(math.ceil(len(history) * float(ppo_cfg["validation_fraction"]))))
    n_val = min(n_val, max(2, len(history) - 4))
    train = history.iloc[:-n_val].copy()
    val = history.iloc[-n_val:].copy()
    mean = train[states].mean().to_numpy(dtype=float)
    std = train[states].std(ddof=0).replace(0.0, 1.0).fillna(1.0).to_numpy(dtype=float)

    def tensors(frame: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
        obs = (frame[states].to_numpy(dtype=float) - mean) / std
        rewards = frame[[f"candidate_reward_{name}" for name in actions]].to_numpy(dtype=float)
        return torch.tensor(obs, dtype=torch.float32), torch.tensor(rewards, dtype=torch.float32)

    train_obs, train_rewards = tensors(train)
    val_obs, val_rewards = tensors(val)
    model = CategoricalActorCritic(len(states), len(actions), int(ppo_cfg["hidden_dim"]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(ppo_cfg["learning_rate"]), weight_decay=1e-4)
    teacher = train_rewards.argmax(dim=1)
    for _ in range(int(ppo_cfg["behavior_clone_epochs"])):
        logits, value = model(train_obs)
        loss = nn.functional.cross_entropy(logits, teacher) + 0.05 * torch.mean(value**2)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_val = _deterministic_reward(model, val_obs, val_rewards)
    best_epoch = 0
    patience = 0
    logs: list[dict[str, Any]] = []
    for epoch in range(1, int(ppo_cfg["max_epochs"]) + 1):
        with torch.no_grad():
            old_logits, old_values = model(train_obs)
            old_dist = torch.distributions.Categorical(logits=old_logits)
            sampled_actions = old_dist.sample()
            old_logp = old_dist.log_prob(sampled_actions)
            sampled_rewards = train_rewards.gather(1, sampled_actions[:, None]).squeeze(1)
            adv, targets = _advantages(
                sampled_rewards,
                old_values,
                float(ppo_cfg["gamma"]),
                float(ppo_cfg["gae_lambda"]),
            )
        for _ in range(int(ppo_cfg["updates_per_epoch"])):
            logits, values = model(train_obs)
            dist = torch.distributions.Categorical(logits=logits)
            logp = dist.log_prob(sampled_actions)
            ratio = torch.exp(logp - old_logp)
            clipped = torch.clamp(
                ratio,
                1.0 - float(ppo_cfg["clip_ratio"]),
                1.0 + float(ppo_cfg["clip_ratio"]),
            )
            policy_loss = -torch.minimum(ratio * adv, clipped * adv).mean()
            value_loss = torch.mean((values - targets) ** 2)
            entropy = dist.entropy().mean()
            loss = (
                policy_loss
                + float(ppo_cfg["value_loss_weight"]) * value_loss
                - float(ppo_cfg["entropy_weight"]) * entropy
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        every = int(ppo_cfg["evaluation_every"])
        if epoch == 1 or epoch % every == 0 or epoch == int(ppo_cfg["max_epochs"]):
            train_reward = _deterministic_reward(model, train_obs, train_rewards)
            val_reward = _deterministic_reward(model, val_obs, val_rewards)
            logs.append(
                {
                    "profile": profile,
                    "evaluation_year": evaluation_year,
                    "seed": seed,
                    "epoch": epoch,
                    "train_rows": len(train),
                    "validation_rows": len(val),
                    "train_reward": train_reward,
                    "validation_reward": val_reward,
                    "best_validation_reward": max(best_val, val_reward),
                    "policy_loss": float(policy_loss.detach()),
                    "value_loss": float(value_loss.detach()),
                    "entropy": float(entropy.detach()),
                }
            )
            if val_reward > best_val + 1e-10:
                best_val = val_reward
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
            if patience >= int(ppo_cfg["early_stop_patience"]):
                break
    metadata = {
        "state_columns": states,
        "state_mean": mean,
        "state_std": std,
        "best_epoch": best_epoch,
        "best_validation_reward": best_val,
        "train_rows": len(train),
        "validation_rows": len(val),
    }
    return best_state, metadata, logs


def _predict_probabilities(
    frame: pd.DataFrame,
    state: dict[str, torch.Tensor],
    metadata: dict[str, Any],
    cfg: dict[str, Any],
) -> np.ndarray:
    model = CategoricalActorCritic(
        len(metadata["state_columns"]),
        len(cfg["actions"]["names"]),
        int(cfg["ppo"]["hidden_dim"]),
    )
    model.load_state_dict(state)
    model.eval()
    values = frame[metadata["state_columns"]].to_numpy(dtype=float)
    values = (values - metadata["state_mean"]) / metadata["state_std"]
    with torch.no_grad():
        logits, _ = model(torch.tensor(values, dtype=torch.float32))
    return torch.softmax(logits, dim=1).numpy()


def train_walkforward_policy(profile: str, dataset: pd.DataFrame, cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    cfg = cfg or load_hybrid_config()
    action_names = list(cfg["actions"]["names"])
    warmup_action = str(cfg["actions"]["warmup_action"])
    model_dir = cfg_path(cfg, "model_dir")
    model_dir.mkdir(parents=True, exist_ok=True)
    all_logs: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    min_rows = int(cfg["ppo"]["min_train_rows"])

    for year in sorted(pd.to_datetime(dataset["date"]).dt.year.unique()):
        cutoff = pd.Timestamp(year=int(year) - 1, month=12, day=31)
        history = dataset[pd.to_datetime(dataset["next_date"]).le(cutoff)].copy()
        current = dataset[pd.to_datetime(dataset["date"]).dt.year.eq(int(year))].copy()
        if current.empty:
            continue
        if len(history) < min_rows:
            probabilities = np.zeros((len(current), len(action_names)), dtype=float)
            probabilities[:, action_names.index(warmup_action)] = 1.0
            source = "itransformer_warmup"
            max_label_date = pd.NaT
        else:
            ensemble: list[np.ndarray] = []
            for seed in cfg["ppo"]["seeds"]:
                state, metadata, logs = _train_seed(history, cfg, int(seed), profile, int(year))
                all_logs.extend(logs)
                ensemble.append(_predict_probabilities(current, state, metadata, cfg))
                torch.save(
                    {
                        "state_dict": state,
                        "metadata": metadata,
                        "config": cfg,
                        "profile": profile,
                        "evaluation_year": int(year),
                        "policy_training_cutoff": cutoff,
                        "policy_max_label_date": pd.to_datetime(history["next_date"]).max(),
                    },
                    model_dir / f"{profile}_{int(year)}_seed{int(seed)}.pt",
                )
            probabilities = np.mean(np.stack(ensemble), axis=0)
            source = "ppo_ensemble"
            max_label_date = pd.to_datetime(history["next_date"]).max()

        selected = probabilities.argmax(axis=1)
        for row_idx, (_, row) in enumerate(current.reset_index(drop=True).iterrows()):
            action = action_names[int(selected[row_idx])]
            decision: dict[str, Any] = {
                "date": row["date"],
                "profile": profile,
                "selected_action": action,
                "selected_action_probability": float(probabilities[row_idx, selected[row_idx]]),
                "policy_source": source,
                "policy_training_cutoff": cutoff if source == "ppo_ensemble" else pd.NaT,
                "policy_max_label_date": max_label_date,
                "itransformer_training_cutoff": row["itransformer_training_cutoff"],
                "itransformer_model_weight": row["state_itransformer_model_weight"],
                "itransformer_spread": row["state_itransformer_spread"],
                "reliability_6": row["state_itransformer_reliability_6"],
                "reliability_12": row["state_itransformer_reliability_12"],
            }
            for action_idx, name in enumerate(action_names):
                decision[f"prob_{name}"] = float(probabilities[row_idx, action_idx])
            for sleeve in RISK_SLEEVES:
                decision[f"base_weight_{sleeve}"] = float(row[f"state_base_weight_{sleeve}"])
                decision[f"target_weight_{sleeve}"] = float(row[f"candidate_weight_{action}_{sleeve}"])
                decision[f"itransformer_score_{sleeve}"] = float(row[f"state_itransformer_score_{sleeve}"])
            decisions.append(decision)

    log_path = cfg_path(cfg, "training_log_pattern", profile)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_logs).to_csv(log_path, index=False, encoding="utf-8-sig")
    out = pd.DataFrame(decisions).sort_values("date").reset_index(drop=True)
    decision_path = cfg_path(cfg, "decisions_pattern", profile)
    decision_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(decision_path, index=False)
    return out


def build_hybrid_weights(profile: str, decisions: pd.DataFrame, cfg: dict[str, Any] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = cfg or load_hybrid_config()
    base_sleeves = pd.read_parquet(data_path(cfg, "baseline_sleeve_weights_pattern", profile))
    base_sleeves["date"] = pd.to_datetime(base_sleeves["date"], errors="coerce")
    decision_map = decisions.set_index(pd.to_datetime(decisions["date"]))
    hybrid_sleeves = base_sleeves.copy()
    for idx, row in hybrid_sleeves.iterrows():
        date = pd.Timestamp(row["date"])
        if date in decision_map.index:
            chosen = decision_map.loc[date]
            if isinstance(chosen, pd.DataFrame):
                chosen = chosen.iloc[-1]
            for sleeve in RISK_SLEEVES:
                hybrid_sleeves.at[idx, sleeve] = float(chosen[f"target_weight_{sleeve}"])
        hybrid_sleeves.at[idx, "cash"] = 0.0
    sleeve_path = cfg_path(cfg, "sleeve_weights_pattern", profile)
    sleeve_path.parent.mkdir(parents=True, exist_ok=True)
    hybrid_sleeves.to_parquet(sleeve_path, index=False)

    base_assets = pd.read_parquet(data_path(cfg, "baseline_asset_weights_pattern", profile))
    base_assets["date"] = pd.to_datetime(base_assets["date"], errors="coerce")
    metadata = (
        _monthly_returns(profile)
        .sort_values("date")
        .drop_duplicates("asset_code", keep="last")
        .set_index("asset_code")
    )
    sleeve_target = hybrid_sleeves.set_index("date")
    rows: list[dict[str, Any]] = []
    for date, group in base_assets.groupby("date", sort=True):
        target = sleeve_target.loc[pd.Timestamp(date)]
        for sleeve in RISK_SLEEVES:
            sleeve_weight = float(target[sleeve])
            sub = group[group["module"].eq(sleeve)].copy()
            if not sub.empty and float(sub["weight"].sum()) > 1e-12:
                shares = dict(zip(sub["asset_code"], sub["weight"] / sub["weight"].sum()))
                names = dict(zip(sub["asset_code"], sub["asset_name"]))
            else:
                shares = _preferred_asset_weights(profile, sleeve)
                names = {
                    code: str(metadata.loc[code, "asset_name"]) if code in metadata.index else code
                    for code in shares
                }
            for code, share in shares.items():
                rows.append(
                    {
                        "date": pd.Timestamp(date),
                        "asset_code": str(code),
                        "asset_name": names.get(code, str(code)),
                        "module": sleeve,
                        "weight": sleeve_weight * float(share),
                    }
                )
        rows.append(
            {
                "date": pd.Timestamp(date),
                "asset_code": "cash",
                "asset_name": "现金",
                "module": "cash",
                "weight": 0.0,
            }
        )
    hybrid_assets = pd.DataFrame(rows).groupby(
        ["date", "asset_code", "asset_name", "module"], as_index=False
    )["weight"].sum()
    asset_path = cfg_path(cfg, "asset_weights_pattern", profile)
    asset_path.parent.mkdir(parents=True, exist_ok=True)
    hybrid_assets.to_parquet(asset_path, index=False)
    return hybrid_sleeves, hybrid_assets


def backtest_hybrid(profile: str, weights: pd.DataFrame, cfg: dict[str, Any] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = cfg or load_hybrid_config()
    returns = _monthly_returns(profile)
    returns["date"] = pd.to_datetime(returns["date"], errors="coerce")
    ret = returns.pivot_table(index="date", columns="asset_code", values="return_1m", aggfunc="last").sort_index()
    ret["cash"] = 0.0
    target = weights.pivot_table(index="date", columns="asset_code", values="weight", aggfunc="sum").sort_index().fillna(0.0)
    columns = sorted(set(ret.columns).union(target.columns))
    ret = ret.reindex(columns=columns)
    target = target.reindex(columns=columns).fillna(0.0)
    applied = target.shift(1).dropna(how="all")
    dates = ret.index.intersection(applied.index)
    applied = applied.loc[dates]
    raw = ret.loc[dates]
    missing_weight = applied.where(raw.isna(), 0.0).abs().sum(axis=1)
    gross = (applied * raw.fillna(0.0)).sum(axis=1)
    turnover = (applied - applied.shift(1).fillna(0.0)).abs().sum(axis=1)
    cost = turnover * float(cfg["reward"]["transaction_cost_bps"]) / 10000.0
    net = gross - cost
    nav = pd.DataFrame(
        {
            "date": dates,
            "gross_return": gross.to_numpy(),
            "turnover": turnover.to_numpy(),
            "transaction_cost": cost.to_numpy(),
            "missing_return_weight": missing_weight.to_numpy(),
            "net_return": net.to_numpy(),
            "nav": (1.0 + net).cumprod().to_numpy(),
        }
    )
    metrics = calc_metrics(nav)
    metrics = pd.concat(
        [
            metrics,
            pd.DataFrame(
                [
                    {"metric": "max_missing_return_weight", "value": float(missing_weight.max())},
                    {"metric": "months_with_missing_return_exposure", "value": float((missing_weight > 1e-12).sum())},
                ]
            ),
        ],
        ignore_index=True,
    )
    nav_path = cfg_path(cfg, "nav_pattern", profile)
    metrics_path = cfg_path(cfg, "metrics_pattern", profile)
    nav_path.parent.mkdir(parents=True, exist_ok=True)
    nav.to_csv(nav_path, index=False, encoding="utf-8-sig")
    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    return nav, metrics


def period_metrics(nav: pd.DataFrame, start: str | None = None, end: str | None = None) -> dict[str, float]:
    frame = nav.copy()
    dates = pd.to_datetime(frame["date"], errors="coerce")
    if start:
        frame = frame[dates.ge(pd.Timestamp(start))].copy()
    if end:
        frame = frame[pd.to_datetime(frame["date"]).le(pd.Timestamp(end))].copy()
    if frame.empty:
        return {}
    frame["nav"] = (1.0 + frame["net_return"].astype(float)).cumprod()
    return calc_metrics(frame).set_index("metric")["value"].to_dict()


def run_profile(profile: str, cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = cfg or load_hybrid_config()
    dataset = build_profile_dataset(profile, cfg)
    decisions = train_walkforward_policy(profile, dataset, cfg)
    _, asset_weights = build_hybrid_weights(profile, decisions, cfg)
    nav, _ = backtest_hybrid(profile, asset_weights, cfg)
    return {
        "profile": profile,
        "dataset_rows": len(dataset),
        "decision_rows": len(decisions),
        "nav_rows": len(nav),
        "first_decision": str(pd.to_datetime(decisions["date"]).min().date()),
        "last_decision": str(pd.to_datetime(decisions["date"]).max().date()),
    }


def write_comparison(cfg: dict[str, Any] | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    cfg = cfg or load_hybrid_config()
    summary_rows: list[dict[str, Any]] = []
    yearly_rows: list[dict[str, Any]] = []
    scopes = {
        "full": (None, None),
        "overlay_2022plus": ("2022-02-01", None),
        "ppo_2023plus": ("2023-02-01", None),
        "gate_2024plus": (str(cfg["gate"]["evaluation_start"]), None),
    }
    for profile in ["low_vol", "balanced", "high_vol"]:
        hybrid = pd.read_csv(cfg_path(cfg, "nav_pattern", profile), parse_dates=["date"])
        baseline = pd.read_csv(data_path(cfg, "baseline_nav_pattern", profile), parse_dates=["date"])
        for scope, (start, end) in scopes.items():
            h = period_metrics(hybrid, start, end)
            b = period_metrics(baseline, start, end)
            summary_rows.append(
                {
                    "profile": profile,
                    "scope": scope,
                    **{f"baseline_{k}": v for k, v in b.items()},
                    **{f"hybrid_{k}": v for k, v in h.items()},
                    "annual_return_delta": h.get("annual_return", np.nan) - b.get("annual_return", np.nan),
                    "sharpe_delta": h.get("sharpe", np.nan) - b.get("sharpe", np.nan),
                    "max_drawdown_delta": h.get("max_drawdown", np.nan) - b.get("max_drawdown", np.nan),
                    "avg_turnover_delta": h.get("avg_turnover", np.nan) - b.get("avg_turnover", np.nan),
                }
            )
        for year in sorted(pd.to_datetime(hybrid["date"]).dt.year.unique()):
            start = f"{int(year)}-01-01"
            end = f"{int(year)}-12-31"
            h = period_metrics(hybrid, start, end)
            b = period_metrics(baseline, start, end)
            yearly_rows.append(
                {
                    "profile": profile,
                    "year": int(year),
                    "baseline_return": b.get("annual_return", np.nan),
                    "hybrid_return": h.get("annual_return", np.nan),
                    "return_delta": h.get("annual_return", np.nan) - b.get("annual_return", np.nan),
                    "baseline_sharpe": b.get("sharpe", np.nan),
                    "hybrid_sharpe": h.get("sharpe", np.nan),
                    "sharpe_delta": h.get("sharpe", np.nan) - b.get("sharpe", np.nan),
                }
            )
    summary = pd.DataFrame(summary_rows)
    yearly = pd.DataFrame(yearly_rows)
    cfg_path(cfg, "summary").parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(cfg_path(cfg, "summary"), index=False, encoding="utf-8-sig")
    yearly.to_csv(cfg_path(cfg, "yearly"), index=False, encoding="utf-8-sig")

    gate_rows = summary[summary["scope"].eq("gate_2024plus")].copy()
    gates: dict[str, Any] = {"version": cfg["version"], "profiles": {}}
    for row in gate_rows.to_dict(orient="records"):
        checks = {
            "sharpe_delta": float(row["sharpe_delta"]) >= float(cfg["gate"]["min_sharpe_delta"]),
            "annual_return_delta": float(row["annual_return_delta"]) >= float(cfg["gate"]["min_annual_return_delta"]),
            "drawdown_worsening": float(row["max_drawdown_delta"]) >= -float(cfg["gate"]["max_drawdown_worsening"]),
        }
        gates["profiles"][str(row["profile"])] = {
            "passed": bool(all(checks.values())),
            "checks": checks,
            "annual_return_delta": float(row["annual_return_delta"]),
            "sharpe_delta": float(row["sharpe_delta"]),
            "max_drawdown_delta": float(row["max_drawdown_delta"]),
        }
    gates["all_profiles_passed"] = bool(all(v["passed"] for v in gates["profiles"].values()))
    cfg_path(cfg, "gate_report").write_text(json.dumps(gates, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary, yearly, gates


def audit_hybrid(cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    cfg = cfg or load_hybrid_config()
    checks: list[dict[str, Any]] = []

    def add(profile: str, name: str, passed: bool, evidence: Any) -> None:
        checks.append({"profile": profile, "check": name, "passed": bool(passed), "evidence": str(evidence)})

    for profile in ["low_vol", "balanced", "high_vol"]:
        decisions = pd.read_parquet(cfg_path(cfg, "decisions_pattern", profile))
        decisions["date"] = pd.to_datetime(decisions["date"], errors="coerce")
        decisions["itransformer_training_cutoff"] = pd.to_datetime(decisions["itransformer_training_cutoff"], errors="coerce")
        decisions["policy_training_cutoff"] = pd.to_datetime(decisions["policy_training_cutoff"], errors="coerce")
        decisions["policy_max_label_date"] = pd.to_datetime(decisions["policy_max_label_date"], errors="coerce")
        add(profile, "itransformer_cutoff_before_decision", bool((decisions["itransformer_training_cutoff"] < decisions["date"]).all()), len(decisions))
        ppo = decisions[decisions["policy_source"].eq("ppo_ensemble")]
        add(profile, "ppo_cutoff_before_decision", bool((ppo["policy_training_cutoff"] < ppo["date"]).all()), len(ppo))
        add(profile, "ppo_labels_matured_before_cutoff", bool((ppo["policy_max_label_date"] <= ppo["policy_training_cutoff"]).all()), len(ppo))
        add(profile, "actions_are_itransformer_conditioned", bool(decisions["selected_action"].isin(cfg["actions"]["names"]).all()), decisions["selected_action"].value_counts().to_dict())
        sleeves = pd.read_parquet(cfg_path(cfg, "sleeve_weights_pattern", profile))
        total = sleeves[RISK_SLEEVES + ["cash"]].sum(axis=1)
        add(profile, "sleeve_weights_sum_to_one", float((total - 1.0).abs().max()) <= 1e-10, float((total - 1.0).abs().max()))
        add(profile, "cash_is_zero", float(sleeves["cash"].abs().max()) <= 1e-12, float(sleeves["cash"].abs().max()))
        bounds = _profile_config(profile)["sleeve_bounds"]
        violations = 0
        for sleeve in RISK_SLEEVES:
            lo, hi = map(float, bounds[sleeve])
            violations += int(((sleeves[sleeve] < lo - 1e-10) | (sleeves[sleeve] > hi + 1e-10)).sum())
        add(profile, "sleeve_bounds_respected", violations == 0, violations)
        assets = pd.read_parquet(cfg_path(cfg, "asset_weights_pattern", profile))
        asset_total = assets.groupby("date")["weight"].sum()
        add(profile, "asset_weights_sum_to_one", float((asset_total - 1.0).abs().max()) <= 1e-10, float((asset_total - 1.0).abs().max()))
        nav = pd.read_csv(cfg_path(cfg, "nav_pattern", profile))
        add(profile, "nav_finite", bool(np.isfinite(nav[["net_return", "nav"]].to_numpy(dtype=float)).all()), len(nav))
        add(profile, "missing_return_exposure_zero", float(nav["missing_return_weight"].max()) <= 1e-12, float(nav["missing_return_weight"].max()))

    report = pd.DataFrame(checks)
    cfg_path(cfg, "audit_csv").parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(cfg_path(cfg, "audit_csv"), index=False, encoding="utf-8-sig")
    lines = [
        "# iTransformer + PPO Point-in-Time 审计",
        "",
        f"- 检查数：{len(report)}",
        f"- 失败数：{int((~report['passed']).sum())}",
        "",
        report.to_markdown(index=False),
    ]
    cfg_path(cfg, "audit_md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def write_report(summary: pd.DataFrame, yearly: pd.DataFrame, gate: dict[str, Any], cfg: dict[str, Any] | None = None) -> None:
    cfg = cfg or load_hybrid_config()
    display = summary[summary["scope"].isin(["full", "gate_2024plus"])].copy()
    lines = [
        "# iTransformer V2 + PPO 联合择时结果",
        "",
        "## 方法",
        "",
        "- iTransformer V2 使用日频 120 日路径和 5/20/60 日风险调整相对收益标签，输出五袖套 OOS 分数。",
        "- PPO 只在六种受约束动作中选择：不动，顺预测 25%/50%/100%，或反向 25%/50%；不直接生成任意仓位。",
        "- 低/中/高波最大单次袖套对转移为 0.5%/1.0%/2.0%，现金始终为 0。",
        "- PPO 每年只使用上一年末前已经实现的 iTransformer OOS 卡片和收益训练，年度内模型冻结。",
        "- 2022 年样本不足时使用顺预测 25% 的预注册 warm-up；2023 年起使用三随机种子 PPO 集成。",
        "- PPO 最多 300 epoch，按历史尾部验证集 early stopping；奖励是相对风险平价基线的增量净收益，并扣除成本、下行恶化和主动偏离。",
        "",
        "## 结果",
        "",
        display.to_markdown(index=False),
        "",
        "## 正式门控",
        "",
        "```json",
        json.dumps(gate, ensure_ascii=False, indent=2),
        "```",
        "",
        "## 重要限制",
        "",
        "- iTransformer 自身严格开发期 RankIC 仍未单独通过门控；PPO 的作用是学习何时减弱或反转预测，不改变这一事实。",
        "- 2025+ 已被早期研究观察，不是完全未触碰测试集；结果应视为历史研究证据，不能视为未来收益承诺。",
        "- 联合版本与无模型基线同时保留，只有 2024+ 组合门控通过的产品才适合进入小规模仿真或实盘观察。",
    ]
    cfg_path(cfg, "report_md").write_text("\n".join(lines) + "\n", encoding="utf-8")

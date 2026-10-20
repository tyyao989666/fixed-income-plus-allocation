"""Isolated execution-learning ablations; no change to targets or accounting."""
from __future__ import annotations

import copy
import hashlib
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from src import v45_portfolio as engine
from src.v46_shadow_research import RelativeExecutionEnvironment


def standardizer(values):
    values = np.asarray(values, float)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Normalizer requires finite T-only observations")
    mean, std = values.mean(0), values.std(0)
    return mean, np.where(std > 1e-8, std, 1.)


def soft_targets(utilities, temperature):
    if temperature <= 0 or not np.isfinite(temperature):
        raise ValueError("Invalid teacher temperature")
    x = np.asarray(utilities, float)
    if not np.isfinite(x).all() or x.shape[-1] != 4:
        raise ValueError("Invalid action values")
    x = (x-x.max(axis=-1, keepdims=True))/temperature
    ex = np.exp(x)
    return ex/ex.sum(axis=-1, keepdims=True)


class ScaledExecutionEnvironment(RelativeExecutionEnvironment):
    def raw_observation(self, index, current, context):
        n = self._state.shape[1]
        raw = super().observation(index, current, context, np.zeros(n), np.ones(n))
        raw[:n] = self._state[index]
        return raw

    def observation(self, index, current, context, mean, std):
        raw = self.raw_observation(index, current, context)
        if raw.shape != np.shape(mean) or raw.shape != np.shape(std):
            raise ValueError("Full-observation normalizer/checkpoint mismatch")
        return np.clip((raw-mean)/std, -6, 6).astype(np.float32)

    def step(self, index, action, current, context):
        raw_action = int(action)
        allowed = self.allowed_action(index, action, context)
        result = super().step(index, action, current, context)
        info = result[-1]
        info.update(raw_policy_action=raw_action, allowed_policy_action=allowed,
                    minimum_hold_suppressed=bool(raw_action != allowed))
        return result

    def action_values(self, index, current, context, horizon=5):
        """Counterfactual first action, then the same causal mechanical policy."""
        if index+horizon >= len(self.dates):
            raise IndexError("Unmatured teacher label")
        values = []
        for action in range(4):
            held, branch = current.copy(), copy.deepcopy(context)
            utility = 0.
            for offset in range(horizon):
                choice = action if offset == 0 else self.mechanical_action(index+offset, held, branch)
                reward, _, held, branch, _ = self.step(index+offset, choice, held, branch)
                utility += reward
            values.append(utility)
        # Subtracting a common baseline changes neither probabilities nor ranks.
        baseline = values[self.mechanical_action(index, current, context)]
        return np.asarray(values)-baseline


def training_normalizer(env, train):
    key = tuple(map(int, train))
    cache = getattr(env, "_v48_norm_cache", {})
    if key not in cache:
        observed = []
        # Cover drifted holdings as well as the mechanical path, using T only.
        for policy in ["mechanical", "hold", "full"]:
            current, context = env.initial_weights(int(train[0])), env.initial_context()
            for i in train:
                observed.append(env.raw_observation(i, current, context))
                action = env.mechanical_action(i, current, context) if policy == "mechanical" else (0 if policy == "hold" else 3)
                _, _, current, context, _ = env.step(i, action, current, context)
        mean, std = standardizer(observed)
        # Keep the established static-state scaler; change dynamic scaling only.
        n = env._state.shape[1]
        mean[:n], std[:n] = standardizer(env._state[train])
        cache[key] = (mean, std)
        env._v48_norm_cache = cache
    return cache[key]


def teacher_data(env, train, mean, std, variant):
    key = (tuple(map(int, train)), variant)
    cache = getattr(env, "_v48_teacher_cache", {})
    if key in cache:
        return cache[key]
    horizon = int(env.cfg["controller"]["teacher_horizon_days"])
    current, context = env.initial_weights(int(train[0])), env.initial_context()
    train_set, obs, actions, utilities, audit = set(train), [], [], [], []
    for i in train:
        if all(i+j in train_set for j in range(horizon)) and i+horizon <= train[-1]+1:
            obs.append(env.observation(i, current, context, mean, std))
            if variant == "scaled_value":
                utility = env.action_values(i, current, context, horizon)
                utilities.append(utility)
                actions.append(int(utility.argmax()))
            else:
                actions.append(env.teacher_action(i, current, horizon, context))
            audit.append({"signal_date": env.dates[i], "label_end_date": env.dates[i+horizon],
                          "train_last_execution": env.dates[train[-1]+1]})
        action = env.mechanical_action(i, current, context)
        _, _, current, context, _ = env.step(i, action, current, context)
    if not obs:
        raise ValueError("No fully matured teacher observations")
    actions = np.asarray(actions)
    temperature = None
    if variant == "scaled_value":
        utilities = np.asarray(utilities)
        temperature = max(float(np.median(np.ptp(utilities, axis=1))), .001)
        probabilities = soft_targets(utilities, temperature)
    else:
        probabilities = np.eye(4)[actions]
    result = (torch.tensor(np.vstack(obs)), torch.tensor(probabilities, dtype=torch.float32),
              actions, audit, temperature)
    cache[key] = result
    env._v48_teacher_cache = cache
    return result


def fit_seed(env, year, seed, lr, entropy_weight, split, log_callback=None):
    """Categorical clipped PPO/GAE, with a T-only cached teacher ablation."""
    ctl = env.cfg["controller"]
    train, selection, _ = split
    torch.manual_seed(seed)
    mean, std = training_normalizer(env, train)
    x_teacher, p_teacher, hard_actions, audit, temperature = teacher_data(
        env, train, mean, std, env.cfg["v48"]["variant"])
    model = engine.ActorCritic(len(mean), 4, ctl["hidden_dim"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    def teacher_loss():
        logits, _ = model(x_teacher)
        return -(p_teacher*F.log_softmax(logits, dim=-1)).sum(-1).mean()

    for _ in range(ctl.get("teacher_pretrain_epochs", 30)):
        engine.finite_update(teacher_loss(), model, optimizer)
    baseline = float(np.mean([r["reward"] for r in engine.simulate(env, selection)]))
    best, best_state, patience = -np.inf, copy.deepcopy(model.state_dict()), 0
    logs, started = [], time.monotonic()
    for epoch in range(1, min(120, ctl["max_epochs"])+1):
        current, context = env.initial_weights(int(train[0])), env.initial_context()
        observations, actions, rewards, values, old_logp = [], [], [], [], []
        for i in train:
            obs = env.observation(i, current, context, mean, std)
            with torch.no_grad():
                logits, value = model(torch.tensor(obs)[None])
                distribution = torch.distributions.Categorical(logits=logits)
                action = distribution.sample()
                logp = distribution.log_prob(action)
            reward, _, current, context, _ = env.step(i, int(action.item()), current, context)
            observations.append(obs)
            actions.append(int(action.item()))
            rewards.append(reward)
            values.append(float(value.item()))
            old_logp.append(float(logp.item()))
        advantage = np.zeros(len(train))
        last, next_value = 0., 0.
        for j in reversed(range(len(train))):
            delta = rewards[j]+ctl["gamma"]*next_value-values[j]
            last = delta+ctl["gamma"]*ctl["gae_lambda"]*last
            advantage[j], next_value = last, values[j]
        target = torch.tensor(advantage+values, dtype=torch.float32)
        adv = torch.tensor((advantage-advantage.mean())/(advantage.std()+1e-8), dtype=torch.float32)
        x, a, old = torch.tensor(np.vstack(observations)), torch.tensor(actions), torch.tensor(old_logp)
        for _ in range(ctl["updates_per_epoch"]):
            logits, value = model(x)
            distribution = torch.distributions.Categorical(logits=logits)
            ratio = torch.exp(distribution.log_prob(a)-old)
            clipped = ratio.clamp(1-ctl["clip_ratio"], 1+ctl["clip_ratio"])
            policy_loss = -torch.minimum(ratio*adv, clipped*adv).mean()
            value_loss = (value-target).square().mean()
            auxiliary = teacher_loss()
            entropy = distribution.entropy().mean()
            loss = policy_loss+ctl["value_loss_weight"]*value_loss-entropy_weight*entropy+ctl.get("teacher_policy_loss_weight", .15)*auxiliary
            engine.finite_update(loss, model, optimizer)
        record = dict(profile=env.profile, year=year, seed=seed, epoch=epoch,
                      loss=float(loss.detach()), policy_loss=float(policy_loss.detach()),
                      value_loss=float(value_loss.detach()), teacher_loss=float(auxiliary.detach()),
                      entropy=float(entropy.detach()), rollout_reward=float(sum(rewards)),
                      elapsed_seconds=time.monotonic()-started)
        if epoch == 1 or epoch % 5 == 0:
            val = float(np.mean([r["reward"] for r in engine.simulate(env, selection, model, mean, std)]))
            record.update(selection_reward=val, mechanical_selection_reward=baseline)
            if val > best+.0001:
                best, best_state, patience = val, copy.deepcopy(model.state_dict()), 0
            else:
                patience += 1
            print(f"[v48/{env.cfg['v48']['variant']}/{env.profile}/{year}/{seed}] epoch={epoch} V={val:.6g}", flush=True)
        logs.append(record)
        if log_callback:
            log_callback(record)
        if patience >= ctl.get("patience", 8):
            break
    return dict(state_dict=best_state, mean=mean, std=std, obs_dim=len(mean), seed=seed,
                selection_reward=best, mechanical_selection_reward=baseline, lr=lr,
                entropy_weight=entropy_weight, training_cutoff=str(engine.pd.Timestamp(year, 1, 1)),
                teacher_count=len(audit), teacher_class_counts=np.bincount(hard_actions, minlength=4).tolist(),
                teacher_audit=audit, teacher_temperature=temperature,
                normalization_train_last_execution=str(env.dates[train[-1]+1]),
                signature=engine.model_signature(env, year, lr, entropy_weight)), logs


def source_hash():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

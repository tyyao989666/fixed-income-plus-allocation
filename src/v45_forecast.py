"""Frozen-data v45 F0--F3 research forecasts, in raw horizon return units.

No portfolio decisions or external data access live here. G is used only after
V selection. Forecast quality is not a formal product promotion decision.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import multiprocessing
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import functional as F

from src.hierarchical_asset_itransformer import HierarchicalAssetITransformer, RISK_SLEEVES


DEFAULTS = dict(sequence_length=120, d_model=40, n_heads=4, num_layers=2,
    dim_feedforward=112, dropout=.25, seeds=[17, 29, 47], max_epochs=1000,
    pretrain_minimum_epochs=100, supervised_minimum_epochs=50, patience=40,
    evaluate_every=5, relative_min_delta=.003, min_delta=.0001,
    pretrain_batch=64, supervised_batch=96, pretrain_lr=2e-4,
    head_lr=2e-4, encoder_lr=1e-5, pretrain_weight_decay=.01,
    supervised_weight_decay=.04, gradient_clip=1., head_only_epochs=20,
    warmup_epochs=8, minimum_lr_ratio=.05, ridge_alpha_per_training_cell=.30,
    horizon_weights=[.15, .50, .35], horizons_days=[5, 20, 60],
    validation_days=126, gate_days=126, isolation_days=60,
    minimum_train_days=300, minimum_oof_days=126,
    bootstrap_repetitions=3000, bootstrap_block_days=60, threads=2,
    pretrain_max_epochs=1000, pretrain_evaluate_every=5, pretrain_patience=40,
    causal_suffix_days=[1, 5, 20], full_path_mask_fraction=.25,
    gate_asset_rank=.10, gate_sleeve_rank=.12, gate_delta=.01,
    gate_positive_seeds=2, prior_years=2, prior_asset_rank=.04,
    prior_sleeve_rank=.05, bootstrap_confidence=.90, fdr=.10,
    feature_clip_mad_multiplier=20., feature_clip_scale_floor=1., seed_workers=1,
    years=[2022, 2023, 2024, 2025, 2026])

PREDICTION_COLUMNS = ["as_of_date", "task", "entity", "module", "horizon_days",
    "model", "seed", "prediction", "baseline_prediction", "active_increment",
    "raw_increment", "gated_prediction", "active_component", "gate_reason",
    "model_signature", "selected_candidate", "source", "train_end_date",
    "gate_as_of_date", "is_oof", "component_gate_passed", "entity_training_dates"]


def configuration(path=None):
    cfg = copy.deepcopy(DEFAULTS)
    if path and Path(path).exists():
        import yaml
        root = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        mapping = {
            "model": dict(sequence_length_days="sequence_length", minimum_epochs="supervised_minimum_epochs",
                validation_every_epochs="evaluate_every", patience_epochs="patience", batch_size="supervised_batch",
                head_learning_rate="head_lr", encoder_learning_rate="encoder_lr", weight_decay="supervised_weight_decay",
                torch_threads="threads"),
            "pretraining": dict(max_epochs="pretrain_max_epochs", minimum_epochs="pretrain_minimum_epochs",
                batch_size="pretrain_batch", learning_rate="pretrain_lr", weight_decay="pretrain_weight_decay",
                validation_every_epochs="pretrain_evaluate_every", patience_epochs="pretrain_patience"),
            "walk_forward": dict(purge_days="isolation_days"),
            "forecast_gate": dict(minimum_asset_rank_ic="gate_asset_rank", minimum_sleeve_rank_ic="gate_sleeve_rank",
                minimum_delta_rank_ic_vs_F0="gate_delta", minimum_positive_seeds="gate_positive_seeds",
                prior_oos_minimum_years="prior_years", prior_oos_minimum_asset_rank_ic="prior_asset_rank",
                prior_oos_minimum_sleeve_rank_ic="prior_sleeve_rank", bootstrap_draws="bootstrap_repetitions",
                one_sided_confidence="bootstrap_confidence"), "forecast": {}}
        fixed = {("model", "candidates"): ["F0", "F1", "F2", "F3"],
            ("model", "huber_beta"): 1., ("model", "pairwise_loss_weight"): .20,
            ("walk_forward", "final_refit_after_gate"): False,
            ("pretraining", "mask_mode"): "mixed_causal_suffix"}
        for section, aliases in mapping.items():
            for key, value in root.get(section, {}).items():
                if (section, key) in fixed:
                    if value != fixed[section, key]:
                        raise ValueError(f"Unsupported {section}.{key}: {value!r}")
                    continue
                target = aliases.get(key, key)
                if target not in DEFAULTS:
                    raise ValueError(f"Unknown forecast setting {section}.{key}")
                cfg[target] = value
        data = root.get("data", {})
        if data.get("sequence_length_days", cfg["sequence_length"]) != cfg["sequence_length"]:
            raise ValueError("Conflicting data/model sequence length")
        if "horizons_days" in data:
            cfg["horizons_days"] = data["horizons_days"]
        if data.get("use_agentic", False) or data.get("fetch_external_data", False):
            raise ValueError("This trainer requires frozen, no-agent data")
    for key in ("max_epochs", "pretrain_max_epochs"):
        if not 1 <= int(cfg[key]) <= 1000:
            raise ValueError(f"{key} must be in [1, 1000]")
    if cfg["horizons_days"] != [5, 20, 60] or cfg["sequence_length"] != 120:
        raise ValueError("v45 requires 120-day windows and 5/20/60-day horizons")
    if cfg["seeds"] != [17, 29, 47]:
        raise ValueError("v45 requires seeds 17/29/47")
    if int(cfg["seed_workers"]) not in (1, 2, 3):
        raise ValueError("seed_workers must be 1, 2, or 3")
    return cfg


def dates_ns(values):
    a = np.asarray(values)
    return a.astype("datetime64[ns]").astype(np.int64) if a.dtype.kind == "M" else a.astype(np.int64)


def validate_arrays(a, check_target_values=True):
    required = ("dates", "local_values", "local_masks", "macro_values", "returns",
        "return_masks", "asset_target", "asset_target_mask", "sleeve_target",
        "sleeve_target_mask", "label_end", "module_index", "sample_indices")
    missing = set(required) - set(a)
    if missing:
        raise ValueError(f"Missing dataset keys: {sorted(missing)}")
    t, n, _ = a["local_values"].shape
    if len(a["dates"]) != t or np.any(np.diff(dates_ns(a["dates"])) <= 0):
        raise ValueError("dates must be unique and increasing")
    for key, shape in (("local_masks", a["local_values"].shape),
                       ("asset_target", (t, 3, n)), ("asset_target_mask", (t, 3, n)),
                       ("sleeve_target", (t, 3, 5)), ("sleeve_target_mask", (t, 3, 5)),
                       ("label_end", (t, 3))):
        if a[key].shape != shape:
            raise ValueError(f"{key}: expected {shape}, got {a[key].shape}")
    if set(a["module_index"].tolist()) != set(range(5)):
        raise ValueError("All five modules must be represented")
    indices = a["sample_indices"]
    if np.any(np.diff(indices) <= 0) or np.any(indices < 119) or np.any(indices >= t):
        raise ValueError("sample_indices must be unique sorted 120-day endpoints")
    for task in ("asset", "sleeve"):
        mask = a[f"{task}_target_mask"].astype(bool)
        if check_target_values and not np.isfinite(a[f"{task}_target"][mask]).all():
            raise ValueError(f"Nonfinite observed {task} target")
    if not np.isfinite(a["macro_values"]).all():
        raise ValueError("Strict macro/state inputs must be finite (with explicit masks if missing)")


def annual_splits(a, year, cfg=None):
    """Use observation positions for isolation, and actual label ends for maturity.

    A full-length split with insufficient T remains useful for shadow training.
    Incomplete V/G is never shortened to manufacture a passing fold.
    """
    cfg = cfg or DEFAULTS
    dates = dates_ns(a["dates"])
    cutoff = np.searchsorted(dates, pd.Timestamp(f"{year}-01-01").value) - 1
    ends = dates_ns(a["label_end"])
    # Some builders store observation indices rather than nanosecond timestamps.
    if ends.size and ends.max() < len(dates) and ends.max() >= 0:
        valid = (ends >= 0) & (ends < len(dates))
        ends = np.where(valid, dates[np.clip(ends, 0, len(dates)-1)], np.iinfo(np.int64).max)
    sample = np.asarray(a["sample_indices"], dtype=np.int64)
    matured = (ends > 0).all(1) & (ends <= dates[max(cutoff, 0)]).all(1)
    eligible = sample[(sample <= cutoff) & matured[sample]] if cutoff >= 0 else sample[:0]
    result = {k: np.array([], dtype=np.int64) for k in ("train", "validation", "gate", "test")}
    result["test"] = sample[(dates[sample] >= pd.Timestamp(f"{year}-01-01").value) &
        (dates[sample] < pd.Timestamp(f"{year+1}-01-01").value)]
    result.update(cutoff_index=int(cutoff), year=int(year), valid=False, reason="insufficient_split_history")
    v, g, gap = (int(cfg[k]) for k in ("validation_days", "gate_days", "isolation_days"))
    if len(eligible) < g:
        return result
    result["gate"] = eligible[-g:]
    vpool = eligible[eligible < result["gate"][0] - gap]
    if len(vpool) < v:
        return result
    result["validation"] = vpool[-v:]
    train = eligible[eligible < result["validation"][0] - gap]
    # Require every T label to mature strictly before V, and every V label before G.
    result["train"] = train[(ends[train] < dates[result["validation"][0]]).all(1)]
    boundaries = (ends[result["validation"]] < dates[result["gate"][0]]).all()
    result["valid"] = bool(len(result["train"]) >= cfg["minimum_train_days"] and boundaries)
    result["reason"] = "ok" if result["valid"] else "insufficient_training_dates_or_label_isolation"
    return result


def historical_signature(a, split, cfg, metadata):
    """Exclude future features/labels and filesystem mtimes from resume identity."""
    h = hashlib.sha256()
    h.update(json.dumps(cfg, sort_keys=True).encode())
    schema = {k: metadata.get(k) for k in ("assets", "asset_names", "asset_modules",
        "local_features", "macro_features", "horizons_days", "feature_manifest")}
    h.update(json.dumps(schema, sort_keys=True, default=str).encode())
    for path in (Path(__file__), Path(__file__).parents[1] / "scripts/train_v45_forecasts.py",
                 Path(__file__).with_name("hierarchical_asset_itransformer.py"),
                 Path(__file__).with_name("v45_data.py")):
        if path.exists():
            h.update(path.read_bytes())
    cutoff = split["cutoff_index"]
    for key in sorted(a):
        if key in ("sample_indices", "module_index"):
            value = a[key] if key == "module_index" else a[key][a[key] <= cutoff]
        elif key in ("asset_target", "asset_target_mask", "sleeve_target", "sleeve_target_mask", "label_end"):
            ix = np.concatenate([split[k] for k in ("train", "validation", "gate")])
            value = a[key][ix]
        elif len(a[key].shape) and len(a[key]) == len(a["dates"]):
            value = a[key][:cutoff+1]
        else:
            continue
        h.update(key.encode())
        h.update(str((value.shape, value.dtype)).encode())
        h.update(np.ascontiguousarray(value).tobytes())
    return h.hexdigest()


def end_times(a):
    end = dates_ns(a["label_end"])
    dates = dates_ns(a["dates"])
    if end.max() < len(dates) and end.max() >= 0:
        return np.where(end >= 0, dates[np.clip(end, 0, len(dates)-1)], np.iinfo(np.int64).max)
    return np.where(end > 0, end, np.iinfo(np.int64).max)


def raw_features(a, indices):
    """Recent causal values, availability, and verified state including nine-grid."""
    x = a["local_values"]
    m = a["local_masks"].astype(bool) & np.isfinite(x)
    x = np.where(m, x, 0.)
    parts = [x[indices], m[indices].astype(float)]
    for lag in (5, 20):
        parts.append(np.stack([x[i-lag+1:i+1].sum(0) /
            np.maximum(m[i-lag+1:i+1].sum(0), 1) for i in indices]))
    macro = a["macro_values"]
    if "macro_masks" in a:
        macro = np.concatenate([macro, a["macro_masks"].astype(np.float32)], -1)
    state = np.concatenate([macro[indices], np.stack([macro[i-19:i+1].mean(0) for i in indices])], -1)
    parts.append(np.broadcast_to(state[:, None], (len(indices), x.shape[1], state.shape[-1])))
    asset = np.concatenate(parts, -1).astype(np.float32)
    sleeve = np.stack([asset[:, a["module_index"] == j].mean(1) for j in range(5)], 1)
    return {"asset": asset, "sleeve": sleeve}


class GroupRidge:
    """Equal-date and equal-module fitting; masks never become zero labels."""
    def __init__(self, module_index, alpha=.30, minimum_dates=1,
                 clip_multiplier=20., clip_scale_floor=1.):
        self.groups = {"asset": np.asarray(module_index), "sleeve": np.zeros(5, dtype=int)}
        self.alpha = alpha
        self.minimum_dates = minimum_dates
        self.clip_multiplier = clip_multiplier
        self.clip_scale_floor = clip_scale_floor
        self.models = {}
        self.audit = []

    def fit(self, features, targets, masks):
        for task, group in self.groups.items():
            for h in range(3):
                for g in np.unique(group):
                    entities = np.flatnonzero(group == g)
                    values = features[task][:, entities]
                    identity = np.broadcast_to(np.eye(len(group), dtype=np.float32)[entities],
                        (len(values), len(entities), len(group)))
                    values = np.concatenate([values, identity], -1)
                    good = masks[task][:, h, entities].astype(bool) & np.isfinite(targets[task][:, h, entities])
                    if good.any(1).sum() < self.minimum_dates:
                        continue
                    weights = np.broadcast_to(1 / np.maximum(good.sum(1, keepdims=True), 1), good.shape)[good]
                    weights = weights * (len(weights) / weights.sum())
                    observed_x = values[good]
                    median = np.median(observed_x, axis=0)
                    mad = 1.4826*np.median(np.abs(observed_x-median), axis=0)
                    width = self.clip_multiplier*np.maximum(mad, self.clip_scale_floor)
                    lower, upper = median-width, median+width
                    fitted_x = np.clip(observed_x, lower, upper)
                    scaler = StandardScaler().fit(fitted_x, sample_weight=weights)
                    alpha = self.alpha * weights.sum()
                    model = Ridge(alpha=alpha).fit(scaler.transform(fitted_x),
                        targets[task][:, h, entities][good], sample_weight=weights)
                    self.models[task, h, int(g)] = (scaler, model, entities, lower, upper)
                    self.audit.append(dict(task=task, horizon=h, module=int(g), cells=int(good.sum()),
                        dates=int(good.any(1).sum()), weight_sum=float(weights.sum()), alpha=float(alpha),
                        feature_clip=dict(method="fit_only_unweighted_median_MAD_before_weighted_StandardScaler",
                            multiplier=self.clip_multiplier, scale_floor=self.clip_scale_floor,
                            lower=lower, upper=upper, changed_cells=int((observed_x != fitted_x).sum()))))
        return self

    def predict(self, features):
        result = {task: np.full((len(x), 3, x.shape[1]), np.nan, dtype=np.float32) for task, x in features.items()}
        for (task, h, _), (scaler, model, entities, lower, upper) in self.models.items():
            x = features[task][:, entities]
            identity = np.broadcast_to(np.eye(len(self.groups[task]), dtype=np.float32)[entities],
                (len(x), len(entities), len(self.groups[task])))
            x = np.concatenate([x, identity], -1)
            x = np.clip(x.reshape(-1, x.shape[-1]), lower, upper)
            result[task][:, h, entities] = model.predict(scaler.transform(x)).reshape(len(features[task]), len(entities))
        return result


def targets_at(a, indices):
    return ({task: a[f"{task}_target"][indices] for task in ("asset", "sleeve")},
            {task: a[f"{task}_target_mask"][indices].astype(bool) for task in ("asset", "sleeve")})


def quarterly_oof(a, train, cfg):
    """Each quarter is predicted using only T labels matured before its start."""
    result = {task: np.full_like(a[f"{task}_target"][train], np.nan) for task in ("asset", "sleeve")}
    audit = []
    quarters = pd.DatetimeIndex(dates_ns(a["dates"])[train].astype("datetime64[ns]")).to_period("Q")
    ends = end_times(a)
    for quarter in quarters.unique():
        positions = np.flatnonzero(quarters == quarter)
        block_start = pd.Timestamp(quarter.start_time).value
        past = train[(dates_ns(a["dates"])[train] < block_start) & (ends[train] < block_start).all(1)]
        if len(past) < cfg["minimum_oof_days"]:
            audit.append(dict(quarter=str(quarter), train_dates=len(past), status="unavailable"))
            continue
        y, mask = targets_at(a, past)
        effective = np.logical_or.reduce([mask[t].any((1, 2)) for t in mask])
        if effective.sum() < cfg["minimum_oof_days"]:
            continue
        model = GroupRidge(a["module_index"], cfg["ridge_alpha_per_training_cell"],
            cfg["minimum_oof_days"], cfg["feature_clip_mad_multiplier"],
            cfg["feature_clip_scale_floor"]).fit(raw_features(a, past), y, mask)
        pred = model.predict(raw_features(a, train[positions]))
        for task in result:
            result[task][positions] = pred[task]
        audit.append(dict(quarter=str(quarter), train_dates=len(past), status="past_matured_only",
            latest_label_end=int(ends[past].max()), block_start=int(block_start)))
    return result, audit


def residual_scaling(y, base, masks, modules):
    """Pooled within-task/group T median/MAD; never entity-specific rank scales."""
    residual, valid, center, scale = {}, {}, {}, {}
    for task in y:
        valid[task] = masks[task] & np.isfinite(base[task])
        r = y[task] - base[task]
        groups = modules if task == "asset" else np.zeros(5, dtype=int)
        center[task] = np.zeros((3, len(groups)), dtype=np.float32)
        scale[task] = np.ones((3, len(groups)), dtype=np.float32)
        for h in range(3):
            for group in np.unique(groups):
                ix = np.flatnonzero(groups == group)
                cells = r[:, h, ix][valid[task][:, h, ix]]
                if len(cells):
                    med = np.median(cells)
                    mad = max(float(1.4826 * np.median(np.abs(cells-med))), 1e-4)
                    center[task][h, ix], scale[task][h, ix] = med, mad
        residual[task] = np.where(valid[task], (r-center[task])/scale[task], 0).astype(np.float32)
    return residual, valid, center, scale


class ForecastNetwork(nn.Module):
    def __init__(self, a, cfg):
        super().__init__()
        width = int(cfg["d_model"])
        macro = a["macro_values"].shape[1] * (2 if "macro_masks" in a else 1)
        self.core = HierarchicalAssetITransformer(cfg["sequence_length"], a["local_values"].shape[2],
            max(macro, 1), a["module_index"], 3, cfg)
        self.asset_output = nn.Linear(width, 3)
        self.sleeve_output = nn.Linear(width, 3)
        self.suffix = int(cfg["sequence_length"])
        self.causal_suffix_days = list(cfg["causal_suffix_days"])
        self.full_path_mask_fraction = float(cfg["full_path_mask_fraction"])
        self.local_reconstruct = nn.Linear(width, self.suffix * a["local_values"].shape[2])
        self.macro_reconstruct = nn.Linear(width, self.suffix * max(macro, 1))

    def encode(self, batch):
        x, m, macro = batch
        if macro.shape[-1] == 0:
            macro = torch.zeros((*macro.shape[:-1], 1), device=x.device)
        return self.core.encode_context(x, m, macro)

    def forward(self, batch):
        asset, _, _, sleeve = self.encode(batch)
        return {"asset": self.asset_output(asset).permute(0, 2, 1),
                "sleeve": self.sleeve_output(sleeve).permute(0, 2, 1)}

    def reconstruction_loss(self, batch):
        x, mask, macro = (v.clone() for v in batch)
        target, target_macro = x.clone(), macro.clone()
        generator = None if self.training else torch.Generator(device=x.device).manual_seed(17029)
        choice = torch.randint(len(self.causal_suffix_days), (1,), device=x.device, generator=generator).item()
        suffix = self.causal_suffix_days[choice]
        hidden = torch.rand((x.shape[0], 1, x.shape[2], x.shape[3]), device=x.device,
            generator=generator) < self.full_path_mask_fraction
        hidden = hidden.expand_as(x).clone()
        hidden[:, -suffix:] = True
        valid = mask.bool() & hidden
        # A global hidden suffix prevents any token from exposing reconstruction
        # targets to another asset/feature token through cross-asset attention.
        x[hidden], mask[hidden], macro[:, -suffix:] = 0., 0., 0.
        asset, _, pool, sleeves = self.encode((x, mask, macro))
        reconstruction = self.local_reconstruct(asset).reshape(x.shape[0], x.shape[2], self.suffix, -1).permute(0, 2, 1, 3)
        error = F.smooth_l1_loss(reconstruction, target, reduction="none")
        date = (error * valid).sum((1, 2, 3)) / valid.sum((1, 2, 3)).clamp_min(1)
        local_loss = date[valid.any((1, 2, 3))].mean() if valid.any() else error.sum()*0
        # Train the existing sleeve pooling projection by the same history-only
        # objective, rather than leaving F1's frozen sleeve representation random.
        sleeve_reconstruction = self.local_reconstruct(sleeves).reshape(
            x.shape[0], 5, self.suffix, -1).permute(0, 2, 1, 3)
        sleeve_terms = []
        for group in range(5):
            ix = self.core.module_index == group
            m = valid[:, :, ix]
            count = m.sum(2)
            y = (target[:, :, ix]*m).sum(2)/count.clamp_min(1)
            e = F.smooth_l1_loss(sleeve_reconstruction[:, :, group], y, reduction="none")
            good = count > 0
            if good.any():
                sleeve_terms.append((e*good).sum((1, 2))/good.sum((1, 2)).clamp_min(1))
        if sleeve_terms:
            local_loss = local_loss + .1*torch.stack(sleeve_terms).mean()
        if target_macro.shape[-1]:
            macro_pred = self.macro_reconstruct(pool).reshape_as(target_macro)
            return local_loss + F.smooth_l1_loss(macro_pred[:, -suffix:], target_macro[:, -suffix:])
        return local_loss


def batch_at(a, indices, cfg, device):
    offsets = indices[:, None] + np.arange(1-cfg["sequence_length"], 1)
    mask = a["local_masks"][offsets].astype(bool) & np.isfinite(a["local_values"][offsets])
    values = np.where(mask, a["local_values"][offsets], 0.)
    macro = a["macro_values"][offsets]
    if "macro_masks" in a:
        macro = np.concatenate([macro, a["macro_masks"][offsets].astype(np.float32)], -1)
    return tuple(torch.as_tensor(x, dtype=torch.float32, device=device) for x in
                 (values, mask, macro))


def group_loss(pred, target, masks, modules, weights, pairwise=False):
    total = pred["asset"].sum()*0
    for task in ("asset", "sleeve"):
        groups = modules if task == "asset" else torch.zeros(5, dtype=torch.long, device=modules.device)
        for h, weight in enumerate(weights):
            group_terms = []
            for group in groups.unique():
                ix = groups == group
                p, y, m = pred[task][:, h, ix], target[task][:, h, ix], masks[task][:, h, ix]
                if pairwise:
                    difference = y[:, :, None]-y[:, None, :]
                    good = m[:, :, None] & m[:, None, :] & (difference != 0)
                    good &= torch.triu(torch.ones_like(good), diagonal=1).bool()
                    error = F.softplus(-difference.sign() * (p[:, :, None]-p[:, None, :]))
                    denominator = good.sum((1, 2))
                    date_loss = (error*good).sum((1, 2))/denominator.clamp_min(1)
                else:
                    error = F.smooth_l1_loss(p, y, beta=1., reduction="none")
                    denominator = m.sum(1)
                    date_loss = (error*m).sum(1)/denominator.clamp_min(1)
                if (denominator > 0).any():
                    group_terms.append(date_loss[denominator > 0].mean())
            if group_terms:
                total = total + weight * torch.stack(group_terms).mean()
    return total


def daily_ic(pred, target, mask):
    good = np.asarray(mask, bool) & np.isfinite(pred) & np.isfinite(target)
    p = rankdata(np.where(good, pred, np.nan), axis=1, nan_policy="omit")
    y = rankdata(np.where(good, target, np.nan), axis=1, nan_policy="omit")
    count = good.sum(1)
    p = np.where(good, p - np.nansum(p, 1, keepdims=True)/np.maximum(count[:, None], 1), 0.)
    y = np.where(good, y - np.nansum(y, 1, keepdims=True)/np.maximum(count[:, None], 1), 0.)
    denom = np.sqrt((p*p).sum(1)*(y*y).sum(1))
    return np.divide((p*y).sum(1), denom, out=np.full(len(p), np.nan), where=(denom > 0) & (count >= 2))


def finite_mean(x):
    x = np.asarray(x)
    return float(x[np.isfinite(x)].mean()) if np.isfinite(x).any() else float("nan")


def component_series(pred, base, y, masks, modules):
    result = {}
    for task in y:
        groups = modules if task == "asset" else np.zeros(5, dtype=int)
        for group in np.unique(groups):
            ix = groups == group
            for h in range(3):
                mask = masks[task][:, h, ix] & np.isfinite(base[task][:, h, ix])
                ic = daily_ic(pred[task][:, h, ix], y[task][:, h, ix], mask)
                bic = daily_ic(base[task][:, h, ix], y[task][:, h, ix], mask)
                result[task, int(group), h] = (ic, ic-bic)
    return result


def selection_score(pred, base, y, masks, modules, cfg):
    series = component_series(pred, base, y, masks, modules)
    task_series = []
    for task in ("asset", "sleeve"):
        horizons = []
        for h in range(3):
            values = [v[1] for k, v in series.items() if k[0] == task and k[2] == h]
            matrix = np.array(values)
            count = np.isfinite(matrix).sum(0)
            horizons.append(np.divide(np.nansum(matrix, 0), count, out=np.full(matrix.shape[1], np.nan), where=count > 0))
        task_series.append(np.sum(np.array(horizons) * np.array(cfg["horizon_weights"])[:, None], axis=0))
    if any(not np.isfinite(x).any() for x in task_series):
        return float("nan")
    joint = .5*task_series[0] + .5*task_series[1]
    blocks = [finite_mean(joint[i:i+20]) for i in range(0, len(joint)-19, 20)]
    if len(blocks) < 2 or not np.isfinite(blocks).all():
        return float("nan")
    return .5*finite_mean(task_series[0]) + .5*finite_mean(task_series[1]) - .1*float(np.std(blocks))


@torch.no_grad()
def infer(network, a, indices, cfg, device, representations=False):
    network.eval()
    collected = {"asset": [], "sleeve": []}
    for start in range(0, len(indices), cfg["pretrain_batch"]):
        batch = batch_at(a, indices[start:start+cfg["pretrain_batch"]], cfg, device)
        if representations:
            asset, _, _, sleeve = network.encode(batch)
            pred = {"asset": asset, "sleeve": sleeve}
        else:
            pred = network(batch)
        for task in pred:
            collected[task].append(pred[task].cpu().numpy())
    return {k: np.concatenate(v) for k, v in collected.items()}


def unscale(pred, center, scale, base):
    return {t: base[t] + pred[t]*scale[t] + center[t] for t in pred}


def _atomic_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp)
    temp.replace(path)


def train_network(network, a, split, cfg, device, checkpoint, signature, stage,
                  seed, train_target=None, train_mask=None, center=None, scale=None,
                  validation_base=None, resume=False):
    """Checkpoint current AND selected states, optimizer/schedule and RNG each epoch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    network.to(device)
    supervised = stage in ("F2", "F3")
    if supervised:
        for p in network.parameters():
            p.requires_grad_(False)
        last = list(network.core.encoder.layers[-1].parameters())
        heads = list(network.asset_output.parameters()) + list(network.sleeve_output.parameters())
        for p in heads:
            p.requires_grad_(True)
        optimizer = torch.optim.AdamW([dict(params=heads, lr=cfg["head_lr"]),
            dict(params=last, lr=cfg["encoder_lr"])], weight_decay=cfg["supervised_weight_decay"])
    else:
        optimizer = torch.optim.AdamW(network.parameters(), lr=cfg["pretrain_lr"], weight_decay=cfg["pretrain_weight_decay"])
    base_lrs = [g["lr"] for g in optimizer.param_groups]
    maximum = int(cfg["max_epochs"] if supervised else cfg["pretrain_max_epochs"])
    minimum = int(cfg["supervised_minimum_epochs"] if supervised else cfg["pretrain_minimum_epochs"])
    batch_size = cfg["supervised_batch"] if supervised else cfg["pretrain_batch"]
    best_value, best_epoch, stale, start_epoch = -float("inf"), 0, 0, 1
    best_state = copy.deepcopy(network.state_dict())
    last_validation_loss, last_score = float("nan"), float("nan")
    if resume and Path(checkpoint).exists():
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        if saved["signature"] != signature:
            raise ValueError(f"Refusing incompatible historical/config/source checkpoint: {checkpoint}")
        if saved.get("complete"):
            network.load_state_dict(saved["best_state"])
            return network, saved["summary"]
        network.load_state_dict(saved["current_state"])
        optimizer.load_state_dict(saved["optimizer"])
        best_state, best_value, best_epoch, stale = (saved[k] for k in ("best_state", "best_value", "best_epoch", "stale"))
        start_epoch = saved["epoch"]+1
        torch.set_rng_state(saved["torch_rng"].cpu())
        np.random.set_state(saved["numpy_rng"])
        random.setstate(saved["python_rng"])
        if device.type == "cuda" and saved.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
    indices = split["train"]
    valid_y, valid_masks = targets_at(a, split["validation"])
    modules = torch.as_tensor(a["module_index"], device=device)
    log_path = Path(checkpoint).with_suffix(".jsonl")
    csv_path = Path(checkpoint).with_suffix(".csv")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    epoch_times = []
    for epoch in range(start_epoch, maximum+1):
        epoch_start = time.perf_counter()
        if supervised:
            for p in last:
                p.requires_grad_(epoch > cfg["head_only_epochs"])
        # Fixed total-budget schedule makes resume independent of time elapsed.
        factor = min(epoch/cfg["warmup_epochs"], 1.) if epoch <= cfg["warmup_epochs"] else (
            cfg["minimum_lr_ratio"] + (1-cfg["minimum_lr_ratio"])*.5*
            (1+math.cos(math.pi*(epoch-cfg["warmup_epochs"])/max(maximum-cfg["warmup_epochs"], 1))))
        for group, lr in zip(optimizer.param_groups, base_lrs):
            group["lr"] = lr*factor
        network.train()
        losses, huber_losses, rank_losses = [], [], []
        order = np.random.permutation(len(indices))
        for start in range(0, len(order), batch_size):
            pos = order[start:start+batch_size]
            batch = batch_at(a, indices[pos], cfg, device)
            optimizer.zero_grad(set_to_none=True)
            if supervised:
                pred = network(batch)
                y = {t: torch.as_tensor(train_target[t][pos], device=device) for t in train_target}
                masks = {t: torch.as_tensor(train_mask[t][pos], device=device) for t in train_mask}
                huber = group_loss(pred, y, masks, modules, cfg["horizon_weights"])
                ranking = group_loss(pred, y, masks, modules, cfg["horizon_weights"], True) if stage == "F3" else huber*0
                loss = huber + .2*ranking
                huber_losses.append(float(huber.detach()))
                rank_losses.append(float(ranking.detach()))
            else:
                loss = network.reconstruction_loss(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite {stage} loss")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(network.parameters(), cfg["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
            if not all(torch.isfinite(p).all() for p in network.parameters()):
                raise FloatingPointError(f"Nonfinite {stage} parameter after optimizer step")
            losses.append(float(loss.detach()))
        evaluate_every = cfg["evaluate_every"] if supervised else cfg["pretrain_evaluate_every"]
        evaluated = epoch == 1 or epoch % evaluate_every == 0 or epoch == maximum
        improved = False
        if evaluated:
            network.eval()
            with torch.no_grad():
                if supervised:
                    pred = infer(network, a, split["validation"], cfg, device)
                    raw = unscale(pred, center, scale, validation_base)
                    last_score = selection_score(raw, validation_base, valid_y, valid_masks, a["module_index"], cfg)
                    vm = {t: valid_masks[t] & np.isfinite(validation_base[t]) for t in valid_masks}
                    vy = {t: np.where(vm[t], (valid_y[t]-validation_base[t]-center[t])/scale[t], 0).astype(np.float32) for t in vm}
                    last_validation_loss = float(group_loss(
                        {t: torch.tensor(pred[t], device=device) for t in pred},
                        {t: torch.tensor(vy[t], device=device) for t in vy},
                        {t: torch.tensor(vm[t], device=device) for t in vm}, modules, cfg["horizon_weights"]))
                    value = last_score
                    improved = np.isfinite(value) and (not np.isfinite(best_value) or value > best_value+cfg["min_delta"])
                else:
                    vals = []
                    for start in range(0, len(split["validation"]), batch_size):
                        ix = split["validation"][start:start+batch_size]
                        vals.append((float(network.reconstruction_loss(batch_at(a, ix, cfg, device))), len(ix)))
                    last_validation_loss = sum(v*n for v, n in vals)/sum(n for _, n in vals)
                    value = -last_validation_loss
                    improved = np.isfinite(value) and (not np.isfinite(best_value) or value > best_value+abs(best_value)*cfg["relative_min_delta"])
            if improved:
                best_state, best_value, best_epoch, stale = copy.deepcopy(network.state_dict()), value, epoch, 0
            else:
                stale = epoch-best_epoch
        # If a score is unavailable, retain a trained shadow checkpoint, but never
        # turn its missing validation evidence into a selection/gate pass.
        if best_epoch == 0:
            best_state = copy.deepcopy(network.state_dict())
        elapsed = time.perf_counter()-epoch_start
        epoch_times.append(elapsed)
        row = dict(stage=stage, year=split["year"], seed=seed, epoch=epoch,
            train_loss=finite_mean(losses), train_huber=finite_mean(huber_losses),
            train_pairwise=finite_mean(rank_losses), validation_loss=last_validation_loss,
            validation_score=last_score, evaluated=evaluated, selected_epoch=best_epoch,
            best_value=best_value, current_value=value if evaluated else None,
            seconds=elapsed, gradient_norm=float(gradient_norm), lr=optimizer.param_groups[0]["lr"], signature=signature)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, allow_nan=True)+"\n")
        exists = csv_path.exists()
        with csv_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if not exists:
                writer.writeheader()
            writer.writerow(row)
        patience = cfg["patience"] if supervised else cfg["pretrain_patience"]
        complete = epoch == maximum or (evaluated and epoch >= minimum and stale >= patience)
        summary = dict(trained_epochs=epoch, selected_epoch=best_epoch,
            best_value=best_value, selection_available=bool(best_epoch),
            wall_seconds=time.perf_counter()-started, mean_epoch_seconds=finite_mean(epoch_times),
            status="trained" if best_epoch else "shadow_validation_score_unavailable")
        _atomic_save(dict(signature=signature, epoch=epoch, complete=complete,
            best_state=best_state, current_state=network.state_dict(), optimizer=optimizer.state_dict(),
            best_value=best_value, best_epoch=best_epoch, stale=stale, summary=summary,
            torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(),
            python_rng=random.getstate(), cuda_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else None), checkpoint)
        if evaluated:
            print(f"{split['year']} seed={seed} {stage} epoch={epoch} train={row['train_loss']:.5g} "
                  f"val={last_validation_loss:.5g} score={last_score:.5g} best={best_epoch} s/epoch={elapsed:.3f}", flush=True)
        if complete:
            break
    network.load_state_dict(best_state)
    return network, summary


def bootstrap_lower(series, cfg, seed=17):
    """Nonoverlapping 60-observation blocks preserve missing dates, not compress them."""
    blocks = [finite_mean(series[i:i+cfg["bootstrap_block_days"]])
        for i in range(0, len(series)-cfg["bootstrap_block_days"]+1, cfg["bootstrap_block_days"])]
    blocks = np.array(blocks)
    if len(blocks) < 2 or not np.isfinite(blocks).all():
        return float("nan"), float("nan"), len(blocks)
    rng = np.random.default_rng(seed)
    means = blocks[rng.integers(0, len(blocks), (cfg["bootstrap_repetitions"], len(blocks)))].mean(1)
    return float(np.quantile(means, 1-cfg["bootstrap_confidence"])), float((1+(means <= 0).sum())/(len(means)+1)), len(blocks)


def bh_adjust(pvalues):
    p = np.asarray(pvalues, float)
    q = np.ones(len(p))
    order = np.flatnonzero(np.isfinite(p))
    order = order[np.argsort(p[order])]
    if len(order):
        # Count ALL registered comparisons, including unavailable ones.
        q[order] = np.minimum(1, np.minimum.accumulate((p[order]*len(p)/np.arange(1, len(order)+1))[::-1])[::-1])
    return q


def component_gate(key, seed_predictions, base, y, masks, modules, history, cfg,
                   contract, enough_residual_dates):
    task, group, h = key
    ix = modules == group if task == "asset" else np.ones(5, dtype=bool)
    seed_series = [component_series(p, base, y, masks, modules)[key] for p in seed_predictions]
    ensemble = {t: np.mean([p[t] for p in seed_predictions], axis=0) for t in base}
    ic, delta = component_series(ensemble, base, y, masks, modules)[key]
    seed_ic = [finite_mean(s[0]) for s in seed_series]
    seed_delta = [finite_mean(s[1]) for s in seed_series]
    agreement = []
    for i in range(len(seed_predictions)):
        for j in range(i):
            agreement.append(finite_mean(daily_ic(seed_predictions[i][task][:, h, ix],
                seed_predictions[j][task][:, h, ix], masks[task][:, h, ix])))
    blocks = [finite_mean(ic[i:i+20]) for i in range(0, len(ic)-19, 20)]
    block_fraction = float(np.mean(np.asarray(blocks) > 0)) if blocks and np.isfinite(blocks).all() else float("nan")
    lower, p, block_count = bootstrap_lower(delta, cfg)
    issued_years = sorted(set(row["year"] for row in history if np.isfinite(row["ic"])))
    history_counts = {year: sum(row["year"] == year and np.isfinite(row["ic"]) and
        np.isfinite(row["delta"]) for row in history) for year in issued_years}
    years = [year for year in issued_years if history_counts[year] >= 126]
    annual = [finite_mean([row["ic"] for row in history if row["year"] == year]) for year in years]
    historical_delta = finite_mean([row["delta"] for row in history])
    tests = dict(representation_contract=contract, residual_train_dates=enough_residual_dates,
        gate_coverage=bool(np.isfinite(ic).sum() >= 100 and np.isfinite(delta).sum() >= 100),
        complete_seeds=len(seed_predictions) == 3,
        preyear_rank_ic=finite_mean(ic) >= (cfg["gate_asset_rank"] if task == "asset" else cfg["gate_sleeve_rank"]),
        all_seed_rank_ic_positive=bool(len(seed_ic) == 3 and np.isfinite(seed_ic).all() and min(seed_ic) > 0),
        seed_prediction_agreement=finite_mean(agreement) >= (.50 if task == "asset" else .65),
        positive_validation_blocks=block_fraction >= (.60 if task == "asset" else .67),
        prior_matured_oos_two_years=len(years) >= cfg["prior_years"],
        prior_rank_ic=finite_mean(annual) >= (cfg["prior_asset_rank"] if task == "asset" else cfg["prior_sleeve_rank"]),
        prior_positive_year_fraction=bool(len(annual) and np.mean(np.array(annual) > 0) >= .50),
        paired_delta_rank_ic=finite_mean(delta) >= cfg["gate_delta"],
        prior_paired_delta_rank_ic=historical_delta >= cfg["gate_delta"],
        two_of_three_seed_delta=bool(sum(np.asarray(seed_delta) > 0) >= cfg["gate_positive_seeds"]),
        bootstrap_lower_positive=bool(np.isfinite(lower) and lower > 0))
    return dict(task=task, module=int(group), horizon_days=cfg["horizons_days"][h],
        active=bool(all(tests.values())), checks=tests,
        reason="passed" if all(tests.values()) else ";".join(k for k, value in tests.items() if not value),
        rank_ic=finite_mean(ic), delta_rank_ic=finite_mean(delta), seed_rank_ic=seed_ic,
        seed_delta_rank_ic=seed_delta, seed_agreement=finite_mean(agreement),
        positive_block_fraction=block_fraction, bootstrap_lower_90=lower,
        bootstrap_p=p, bootstrap_blocks=block_count, prior_years=years, prior_year_effective_dates=history_counts,
        prior_rank_ic=annual, prior_delta_rank_ic=historical_delta)


def ensemble(predictions):
    return {t: np.mean([p[t] for p in predictions], axis=0) for t in predictions[0]}


def entity_names(metadata, count):
    assets = metadata.get("assets", metadata.get("asset_names", []))
    if len(assets) == count:
        return [str(v.get("asset_code", v.get("code", v.get("name", i)))) if isinstance(v, dict) else str(v)
                for i, v in enumerate(assets)]
    return [f"asset_{i}" for i in range(count)]


def prediction_frame(a, indices, prediction, base, model, seed, metadata, cfg,
                     signature, selected, gates):
    frames = []
    dates = pd.to_datetime(dates_ns(a["dates"])[indices])
    timing = metadata.get("forecast_timing", {})
    for task in ("asset", "sleeve"):
        names = entity_names(metadata, len(a["module_index"])) if task == "asset" else list(RISK_SLEEVES)
        for h, horizon in enumerate(cfg["horizons_days"]):
            for entity, name in enumerate(names):
                module = int(a["module_index"][entity]) if task == "asset" else entity
                key = (task, module if task == "asset" else 0, h)
                gate = gates.get(key, dict(active=False, reason="baseline_control" if model == "F0" else "shadow_unavailable"))
                eligible = bool(gate["active"] and model == selected and seed == -1 and model != "F0")
                p, b = prediction[task][:, h, entity], base[task][:, h, entity]
                members = np.array([entity]) if task == "asset" else np.flatnonzero(a["module_index"] == module)
                current_available = (a["local_masks"][indices][:, members].any(2) &
                    a["return_masks"][indices][:, members].astype(bool)).any(1)
                finite_available = np.isfinite(p) & np.isfinite(b) & current_available
                task_counts = timing.get("entity_residual_training_dates", {}).get(task)
                entity_count = int(np.asarray(task_counts)[h, entity]) if task_counts is not None else 0
                enough_entity_history = entity_count >= cfg["minimum_train_days"]
                active = eligible & finite_available & enough_entity_history
                raw = p-b
                increment = np.where(active, raw, 0.)
                reason = gate["reason"] if model == selected else "not_selected_or_baseline_control"
                if model == selected and not enough_entity_history:
                    reason = "insufficient_entity_training_history"
                reasons = np.where((model == selected) & ~finite_available, "missing_entity_forecast", reason)
                frames.append(pd.DataFrame(dict(as_of_date=dates, task=task, entity=name,
                    module=RISK_SLEEVES[module], horizon_days=horizon, model=model, seed=seed,
                    prediction=p, baseline_prediction=b, active_increment=increment,
                    raw_increment=raw, gated_prediction=b+increment, active_component=active,
                    gate_reason=reasons,
                    model_signature=signature, selected_candidate=selected,
                    source=np.where(active, "gated_candidate", "F0_raw_feature_ridge" if model == "F0" else "shadow_candidate"),
                    train_end_date=pd.Timestamp(timing["train_end_date"]) if timing.get("train_end_date") else pd.NaT,
                    gate_as_of_date=pd.Timestamp(timing["gate_as_of_date"]) if timing.get("gate_as_of_date") else pd.NaT,
                    is_oof=True, component_gate_passed=bool(gate["active"] and model == selected),
                    entity_training_dates=entity_count)))
    return pd.concat(frames, ignore_index=True)[PREDICTION_COLUMNS]


def shared_split(a, year, cfg):
    """Normalize the data agent API, independently checking label chronology."""
    try:
        from src.v45_data import annual_splits as data_splits
    except ImportError:
        return annual_splits(a, year, cfg)
    fixed = (cfg["validation_days"], cfg["gate_days"], cfg["isolation_days"], cfg["minimum_train_days"])
    if fixed != (126, 126, 60, 300):
        return annual_splits(a, year, cfg)
    result = dict(data_splits(a, year))
    dates = dates_ns(a["dates"])
    result["cutoff_index"] = int(np.searchsorted(dates, pd.Timestamp(f"{year}-01-01").value)-1)
    result["year"] = year
    result["valid"] = not bool(result.get("reason"))
    result["reason"] = result.get("reason") or "ok"
    ends = end_times(a)
    for name in ("train", "validation", "gate"):
        ix = result[name]
        if len(ix) and not (ends[ix] <= dates[result["cutoff_index"]]).all():
            raise ValueError(f"{year} {name} contains immature labels")
    for left, right in (("train", "validation"), ("validation", "gate")):
        x, y = result[left], result[right]
        if len(x) and len(y) and (y[0]-x[-1] <= 60 or not (ends[x] < dates[y[0]]).all()):
            raise ValueError(f"{year} {left}/{right} violates isolation")
    return result


def write_json(path, value):
    """Reports use JSON null for unavailable evidence, never nonstandard NaN."""
    def clean(v):
        if isinstance(v, dict):
            return {str(k): clean(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [clean(x) for x in v]
        if isinstance(v, np.ndarray):
            return clean(v.tolist())
        if isinstance(v, (float, np.floating)):
            return float(v) if np.isfinite(v) else None
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, np.bool_):
            return bool(v)
        return v
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(clean(value), indent=2), encoding="utf-8")
    temp.replace(path)


def residual_date_counts(mask, modules):
    counts = {}
    for task in mask:
        group = modules if task == "asset" else np.zeros(5, dtype=int)
        for g in np.unique(group):
            for h in range(3):
                counts[task, int(g), h] = int((mask[task][:, h, group == g].sum(1) >= 2).sum())
    return counts


def clip_fold_features(a, train, cfg):
    """T-only neural-input bounds; F0/OOF keep their original raw features."""
    values = a["local_values"]
    valid = a["local_masks"].astype(bool) & np.isfinite(values)
    reference = np.where(valid[train], values[train], np.nan)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        median = np.nanmedian(reference, axis=0)
        mad = 1.4826*np.nanmedian(np.abs(reference-median), axis=0)
    available = np.isfinite(median)
    median = np.nan_to_num(median)
    width = cfg["feature_clip_mad_multiplier"]*np.maximum(np.nan_to_num(mad), cfg["feature_clip_scale_floor"])
    lower, upper = median-width, median+width
    clipped = np.clip(values, lower, upper)
    clipped_valid = valid & available[None]
    result = dict(a)
    result["local_values"] = np.where(clipped_valid, clipped, 0.).astype(np.float32)
    result["local_masks"] = clipped_valid
    audit = dict(method="T_only_asset_feature_median_plus_minus_20_MAD_floor_1_in_trailing_IQR_units",
        fitted_dates=train, lower=lower, upper=upper, available_in_T=available,
        changed_observed_cells=int((valid & (values != clipped)).sum()),
        changed_T_cells=int((valid[train] & (values[train] != clipped[train])).sum()),
        original_max_abs=float(np.max(np.abs(values[valid]))) if valid.any() else None,
        transformed_max_abs=float(np.max(np.abs(result["local_values"]))))
    return result, audit


def candidate_selection(validation_predictions, validation_base, y, mask, modules, cfg):
    scores = {name: selection_score(ensemble(pred), validation_base, y, mask, modules, cfg)
              for name, pred in validation_predictions.items()}
    finite = [name for name in ("F1", "F2", "F3") if np.isfinite(scores[name])]
    if not finite:
        return "F1", scores, False
    best = max(scores[name] for name in finite)
    selected = next(name for name in finite if scores[name] >= best-.005)
    return selected, scores, True


def forecast_preflight(a, cfg):
    """F0/OOF feasibility using T and V only; never opens G performance.

    Comparing B with itself tests score definedness, not predictive quality.
    No diagnostic label is ever used as a candidate forecast or fitted target.
    """
    validate_arrays(a, check_target_values=False)
    rows, folds = [], []
    for year in cfg["years"]:
        split = shared_split(a, year, cfg)
        train, validation = split["train"], split["validation"]
        if not len(train) or not len(validation):
            folds.append(dict(year=year, structural_selection_available=False,
                reason=split["reason"], gate_performance_opened=False))
            continue
        ty, tm = targets_at(a, train)
        for task in ty:
            if not np.isfinite(ty[task][tm[task]]).all():
                raise ValueError(f"Nonfinite T target in {year}/{task}")
        baseline = GroupRidge(a["module_index"], cfg["ridge_alpha_per_training_cell"],
            cfg["minimum_oof_days"], cfg["feature_clip_mad_multiplier"],
            cfg["feature_clip_scale_floor"]).fit(raw_features(a, train), ty, tm)
        base = baseline.predict(raw_features(a, validation))
        vy, vm = targets_at(a, validation)
        for task in vy:
            if not np.isfinite(vy[task][vm[task]]).all():
                raise ValueError(f"Nonfinite V target in {year}/{task}")
        oof, _ = quarterly_oof(a, train, cfg)
        residual_mask = {t: tm[t] & np.isfinite(oof[t]) for t in tm}
        counts = residual_date_counts(residual_mask, a["module_index"])
        self_series = component_series(base, base, vy, vm, a["module_index"])
        component_rows = []
        for key, (ic, _) in self_series.items():
            task, group, horizon = key
            entities = a["module_index"] == group if task == "asset" else np.ones(5, bool)
            train_dates = int(tm[task][:, horizon, entities].any(1).sum())
            paired = vm[task][:, horizon, entities] & np.isfinite(base[task][:, horizon, entities])
            row = dict(year=year, task=task,
                module=RISK_SLEEVES[group] if task == "asset" else "all_five_sleeves",
                horizon_days=cfg["horizons_days"][horizon], train_label_dates=train_dates,
                f0_fit_available=(task, horizon, group) in baseline.models,
                minimum_f0_fit_dates=cfg["minimum_oof_days"],
                oof_residual_dates=counts.get(key, 0),
                formal_residual_300_met=counts.get(key, 0) >= cfg["minimum_train_days"],
                validation_target_dates=int((vm[task][:, horizon, entities].sum(1) >= 2).sum()),
                validation_f0_paired_dates=int((paired.sum(1) >= 2).sum()),
                validation_f0_defined_rank_ic_dates=int(np.isfinite(ic).sum()),
                validation_f0_rank_ic=finite_mean(ic))
            component_rows.append(row)
            rows.append(row)
            print(f"[preflight/{year}] {task}/{row['module']}/h{row['horizon_days']} "
                f"T={train_dates} F0={row['f0_fit_available']} "
                f"OOF_residual={row['oof_residual_dates']} "
                f"V_pair={row['validation_f0_paired_dates']} V_IC={row['validation_f0_defined_rank_ic_dates']}", flush=True)
        definedness = selection_score(base, base, vy, vm, a["module_index"], cfg)
        h = hashlib.sha256()
        for key in ("dates", "asset_target", "asset_target_mask", "sleeve_target", "sleeve_target_mask"):
            h.update(np.ascontiguousarray(a[key][np.concatenate([train, validation])]).tobytes())
        folds.append(dict(year=year, structural_selection_available=bool(np.isfinite(definedness)),
            T_V_label_signature=h.hexdigest(),
            baseline_vs_itself_definedness_score=definedness,
            reason="defined_not_a_predictive_pass" if np.isfinite(definedness) else "fixed_weight_selection_unavailable_on_V",
            components=component_rows, gate_performance_opened=False))
    return dict(status="preflight_only_not_training", folds=folds, components=rows,
        gate_performance_opened=False, any_structurally_selectable_fold=any(
            f["structural_selection_available"] for f in folds),
        all_folds_structurally_selectable=bool(folds and all(f["structural_selection_available"] for f in folds)),
        thresholds=dict(oof_baseline_minimum_dates=cfg["minimum_oof_days"],
            formal_residual_minimum_dates=cfg["minimum_train_days"]),
        resolved_config=cfg,
        unchanged="targets, horizon weights, deployment thresholds, and G performance")


def write_preflight(reports_path, result):
    reports_path = Path(reports_path)
    reports_path.mkdir(parents=True, exist_ok=True)
    write_json(reports_path / "v45_forecast_preflight.json", result)
    pd.DataFrame(result["components"]).to_csv(reports_path / "v45_forecast_preflight.csv", index=False)


def require_selectable_preflight(result):
    if not result["any_structurally_selectable_fold"]:
        raise ValueError("All forecast folds have structurally unavailable fixed-weight V selection. "
            "No neural training or G performance evaluation started; inspect v45_forecast_preflight.json. "
            "Fix/register reference data, not horizon weights or quality gates.")


@contextmanager
def _seed_thread_environment(threads):
    keys = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
    previous = {key: os.environ.get(key) for key in keys}
    try:
        for key in keys:
            os.environ[key] = str(threads)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _seed_worker_initialize(threads):
    torch.set_num_threads(int(threads))
    torch.set_num_interop_threads(1)


def train_seed_worker(seed, payload):
    """One isolated seed; no candidate selection, G labels, or OOS history."""
    from threadpoolctl import threadpool_info, threadpool_limits
    with threadpool_limits(limits=int(payload["cfg"]["threads"])):
        torch.set_num_threads(int(payload["cfg"]["threads"]))
        a, cfg, split = payload["arrays"], payload["cfg"], payload["split"]
        device = torch.device(payload["device"])
        fold, signature = Path(payload["fold"]), payload["signature"]
        indices, bases = payload["indices"], payload["bases"]
        center, scale = payload["center"], payload["scale"]
        residual, residual_mask = payload["residual"], payload["residual_mask"]
        predictions = {name: {} for name in ("validation", "gate", "test")}
        stages = []
        torch.manual_seed(seed)
        pretrained = ForecastNetwork(a, cfg)
        initial = copy.deepcopy(pretrained.state_dict())
        pretrained, summary = train_network(pretrained, a, split, cfg, device,
            fold / f"seed_{seed}_pretraining.pt", signature, "pretraining", seed, resume=payload["resume"])
        changed = any(not torch.equal(initial[k].cpu(), pretrained.state_dict()[k].cpu())
            for k in initial if k.startswith("core.encoder."))
        summary.update(seed=seed, stage="pretraining", encoder_parameters_changed=changed)
        stages.append(summary)
        representations = {name: infer(pretrained, a, ix, cfg, device, True) for name, ix in indices.items()}
        probe = GroupRidge(a["module_index"], cfg["ridge_alpha_per_training_cell"], 1,
            cfg["feature_clip_mad_multiplier"], cfg["feature_clip_scale_floor"]).fit(
            representations["train"], residual, residual_mask)
        for name in predictions:
            predictions[name]["F1"] = unscale(probe.predict(representations[name]), center, scale, bases[name])
        _atomic_save(dict(signature=signature, baseline=payload["baseline"], probe=probe,
            center=center, scale=scale, oof_audit=payload["oof_audit"], cfg=cfg,
            feature_clipping=payload["feature_audit"]), fold / f"seed_{seed}_F1.pt")
        # Both supervised variants start from this seed's same SSL checkpoint.
        for candidate in ("F2", "F3"):
            network, summary = train_network(copy.deepcopy(pretrained), a, split, cfg, device,
                fold / f"seed_{seed}_{candidate}.pt", signature, candidate, seed,
                residual, residual_mask, center, scale, bases["validation"], payload["resume"])
            summary.update(seed=seed, stage=candidate)
            stages.append(summary)
            for name in predictions:
                predictions[name][candidate] = unscale(infer(network, a, indices[name], cfg, device), center, scale, bases[name])
        return dict(seed=seed, predictions=predictions, stages=stages, worker=dict(
            pid=os.getpid(), torch_threads=torch.get_num_threads(),
            blas_threads=[dict(api=v["internal_api"], threads=v["num_threads"]) for v in threadpool_info()]))


def seed_worker_results(payload, seeds, workers):
    """Yield independent results as completed; the caller restores seed order."""
    if workers == 1:
        for seed in seeds:
            yield train_seed_worker(seed, payload)
        return
    if torch.device(payload["device"]).type != "cpu":
        raise ValueError("Parallel seed workers require CPU; configure seed_workers=1 for CUDA")
    with _seed_thread_environment(payload["cfg"]["threads"]):
        pool = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
            initializer=_seed_worker_initialize, initargs=(payload["cfg"]["threads"],))
        try:
            futures = {pool.submit(train_seed_worker, seed, payload): seed for seed in seeds}
            for future in as_completed(futures):
                result = future.result()
                if result["seed"] != futures[future]:
                    raise RuntimeError("Seed worker identity mismatch")
                yield result
        except BaseException:
            # Python 3.12 has no public terminate_workers. Stop only this pool's
            # owned children; their last atomically saved checkpoints survive.
            for process in list(pool._processes.values()):
                if process.is_alive():
                    process.terminate()
            pool.shutdown(wait=True, cancel_futures=True)
            raise
        else:
            pool.shutdown(wait=True)


def train_fold(a, metadata, year, cfg, output_dir, history, device, resume=False,
               debug=False):
    output_dir = Path(output_dir)
    split = shared_split(a, year, cfg)
    fold = output_dir / str(year)
    fold.mkdir(parents=True, exist_ok=True)
    signature = historical_signature(a, split, cfg, metadata)
    report = dict(year=year, signature=signature, split={k: split[k] for k in
        ("train", "validation", "gate", "test", "valid", "reason", "cutoff_index")}, stages=[], debug=debug)
    write_json(fold / "split.json", report)
    train, val, gate, test = (split[k] for k in ("train", "validation", "gate", "test"))
    metadata = dict(metadata)
    metadata["forecast_timing"] = dict(
        train_end_date=pd.Timestamp(dates_ns(a["dates"])[train[-1]]).isoformat() if len(train) else None,
        gate_as_of_date=pd.Timestamp(dates_ns(a["dates"])[split["cutoff_index"]]).isoformat() if split["cutoff_index"] >= 0 else None)
    if not len(test):
        report["status"] = "no_test_dates"
        return report, []
    if not len(train) or not len(val) or not len(gate):
        # With no honest T/V/G there is no fitted F0 to masquerade as a baseline.
        empty = {t: np.full_like(a[f"{t}_target"][test], np.nan) for t in ("asset", "sleeve")}
        frames = [prediction_frame(a, test, empty, empty, name, seed, metadata, cfg,
            signature, "none", {}) for name in ("F0", "F1", "F2", "F3")
            for seed in ([-1] if name == "F0" else cfg["seeds"]+[-1])]
        pd.concat(frames, ignore_index=True).to_parquet(fold / "predictions.parquet", index=False)
        report.update(status="shadow_no_honest_split", representation_contract_pass=False,
            predictive_evidence_pass=False, portfolio_stage_allowed=False, gates=[])
        write_json(fold / "report.json", report)
        return report, []
    ty, tm = targets_at(a, train)
    baseline = GroupRidge(a["module_index"], cfg["ridge_alpha_per_training_cell"],
        cfg["minimum_oof_days"], cfg["feature_clip_mad_multiplier"],
        cfg["feature_clip_scale_floor"]).fit(raw_features(a, train), ty, tm)
    indices = dict(train=train, validation=val, gate=gate, test=test)
    bases = {name: baseline.predict(raw_features(a, ix)) for name, ix in indices.items() if name != "train"}
    oof, oof_audit = quarterly_oof(a, train, cfg)
    a, feature_audit = clip_fold_features(a, train, cfg)
    report["feature_clipping"] = feature_audit
    residual, residual_mask, center, scale = residual_scaling(ty, oof, tm, a["module_index"])
    entity_counts = {task: mask.sum(axis=0).astype(int) for task, mask in residual_mask.items()}
    metadata["forecast_timing"]["entity_residual_training_dates"] = entity_counts
    report["entity_residual_training_dates"] = entity_counts
    counts = residual_date_counts(residual_mask, a["module_index"])
    report.update(oof_audit=oof_audit, baseline_fit=baseline.audit,
        residual_training_dates={str(k): v for k, v in counts.items()},
        label_scaling=dict(center=center, scale=scale, units="unannualized_horizon_decimal_return"))
    predictions = {name: {candidate: [] for candidate in ("F1", "F2", "F3")}
                   for name in ("validation", "gate", "test")}
    payload = dict(arrays=a, cfg=cfg, split=split, device=str(device), fold=str(fold),
        signature=signature, indices=indices, bases=bases, center=center, scale=scale,
        residual=residual, residual_mask=residual_mask, baseline=baseline,
        oof_audit=oof_audit, feature_audit=feature_audit, resume=resume)
    completed = {}
    report["seed_workers"] = int(cfg["seed_workers"])
    for result in seed_worker_results(payload, cfg["seeds"], int(cfg["seed_workers"])):
        completed[result["seed"]] = result
        ordered = [completed[seed] for seed in cfg["seeds"] if seed in completed]
        report["stages"] = [stage for bundle in ordered for stage in bundle["stages"]]
        report["completed_seeds"] = [bundle["seed"] for bundle in ordered]
        report["worker_runtime"] = {str(bundle["seed"]): bundle["worker"] for bundle in ordered}
        write_json(fold / "progress.json", report)
    for seed in cfg["seeds"]:
        for name in predictions:
            for candidate in ("F1", "F2", "F3"):
                predictions[name][candidate].append(completed[seed]["predictions"][name][candidate])
    vy, vm = targets_at(a, val)
    selected, scores, selection_available = candidate_selection(predictions["validation"], bases["validation"], vy, vm, a["module_index"], cfg)
    report.update(selected_candidate=selected, validation_selection_scores=scores,
        selection_available=selection_available)
    contract = bool(split["valid"] and not debug and selection_available and
        cfg["seeds"] == [17, 29, 47] and all(s.get("encoder_parameters_changed", True) for s in report["stages"]))
    # An unavailable fixed-weight V score cannot authorize opening G.
    # Keep raw shadow forecasts, but do not inspect that fold's gate labels.
    gy, gm = targets_at(a, gate) if selection_available else (None, None)
    gates, gate_rows = {}, []
    selected_gate = predictions["gate"][selected]
    components = component_series(ensemble(selected_gate), bases["gate"], gy, gm,
        a["module_index"]) if selection_available else counts
    cutoff_ns = dates_ns(a["dates"])[split["cutoff_index"]]
    for key in components:
        past = [row for row in history if row["candidate"] == selected and tuple(row["key"]) == key
                and row["year"] < year and 0 < row["label_end"] <= cutoff_ns]
        if selection_available:
            value = component_gate(key, selected_gate, bases["gate"], gy, gm, a["module_index"], past,
                cfg, contract, counts.get(key, 0) >= cfg["minimum_train_days"])
        else:
            value = dict(task=key[0], module=key[1], horizon_days=cfg["horizons_days"][key[2]],
                active=False, checks=dict(selection_available=False, representation_contract=False),
                reason="fixed_weight_validation_selection_unavailable_G_unopened",
                bootstrap_p=float("nan"), gate_performance_opened=False)
        value.update(year=year, candidate=selected, residual_train_dates=counts.get(key, 0))
        gates[key] = value
        gate_rows.append(value)
    adjusted = bh_adjust([row["bootstrap_p"] for row in gate_rows])
    for row, q in zip(gate_rows, adjusted):
        row.update(bh_q=float(q), bh_fdr_auxiliary_pass=bool(q <= cfg["fdr"]),
            registered_comparisons=len(gate_rows), all_candidates_explored=3*len(gate_rows))
    report.update(gates=gate_rows, gate_performance_opened=bool(selection_available), representation_contract_pass=contract,
        predictive_evidence_pass=any(row["active"] for row in gate_rows),
        portfolio_stage_allowed=bool(contract and any(row["active"] for row in gate_rows)),
        formal_status="not_assessed_requires_separate_three_product_promotion",
        status="shadow_or_component_gated_research")
    # Stream candidate frames to avoid a multi-million-row object DataFrame.
    import pyarrow as pa
    import pyarrow.parquet as pq
    writer = None
    try:
        for candidate in ("F0", "F1", "F2", "F3"):
            bundles = [(-1, bases["test"])] if candidate == "F0" else list(zip(cfg["seeds"], predictions["test"][candidate])) + [(-1, ensemble(predictions["test"][candidate]))]
            for seed, pred in bundles:
                frame = prediction_frame(a, test, pred, bases["test"], candidate, seed, metadata, cfg,
                    signature, selected, gates)
                table = pa.Table.from_pandas(frame, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(fold / "predictions.parquet", table.schema)
                writer.write_table(table)
    finally:
        if writer:
            writer.close()
    # Historical records stay raw. Only matured records can enter a later gate.
    new_history = []
    label_ends = end_times(a)
    test_y, test_mask = targets_at(a, test)
    for candidate in ("F1", "F2", "F3"):
        series = component_series(ensemble(predictions["test"][candidate]), bases["test"], test_y, test_mask, a["module_index"])
        for key, (ic, delta) in series.items():
            for pos, i in enumerate(test):
                new_history.append(dict(candidate=candidate, key=list(key), year=year,
                    observation=int(i), label_end=int(label_ends[i, key[2]]),
                    ic=float(ic[pos]), delta=float(delta[pos])))
    _atomic_save(dict(signature=signature, history=new_history), fold / "history.pt")
    write_json(fold / "report.json", report)
    return report, new_history


def run_training(a, metadata, cfg, output_dir, predictions_path, reports_path,
                 resume=False, debug=False, device=None):
    validate_arrays(a)
    torch.set_num_threads(int(cfg["threads"]))
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir, reports_path, predictions_path = map(Path, (output_dir, reports_path, predictions_path))
    reports_path.mkdir(parents=True, exist_ok=True)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(reports_path / "v45_forecast_resolved_config.json", dict(config=cfg, debug=debug,
        device=str(device), torch_version=torch.__version__, prediction_columns=PREDICTION_COLUMNS))
    preflight = forecast_preflight(a, cfg)
    write_preflight(reports_path, preflight)
    require_selectable_preflight(preflight)
    history, reports = [], []
    for year in sorted(cfg["years"]):
        print(f"Starting v45 forecast fold {year}, {device}, threads={cfg['threads']}", flush=True)
        report, fresh = train_fold(a, metadata, year, cfg, output_dir, history, device, resume, debug)
        reports.append(report)
        history.extend(fresh)
        write_json(reports_path / "v45_forecast_progress.json", reports)
    import pyarrow.parquet as pq
    writer = None
    try:
        for report in reports:
            if report.get("status") == "no_test_dates":
                continue
            path = output_dir / str(report["year"]) / "predictions.parquet"
            if not path.exists():
                continue
            for batch in pq.ParquetFile(path).iter_batches():
                import pyarrow as pa
                table = pa.Table.from_batches([batch])
                if writer is None:
                    writer = pq.ParquetWriter(predictions_path, table.schema)
                writer.write_table(table)
    finally:
        if writer:
            writer.close()
    gates = [row for report in reports for row in report.get("gates", [])]
    trained_reports = [r for r in reports if r.get("status") != "no_test_dates"]
    summary = dict(research_only=True, debug=debug, unit="unannualized_horizon_decimal_return",
        predictions=str(predictions_path), folds=reports,
        representation_contract_pass=bool(trained_reports and all(r.get("representation_contract_pass", False) for r in trained_reports)),
        predictive_evidence_pass=any(row["active"] for row in gates),
        portfolio_stage_allowed=any(r.get("portfolio_stage_allowed", False) for r in reports),
        formal_status="not_assessed_requires_separate_three_product_promotion",
        zero_increment_trace=[dict(year=r["year"], task=r["task"], module=r["module"],
            horizon_days=r["horizon_days"], reason=r["reason"]) for r in gates if not r["active"]])
    write_json(reports_path / "v45_forecast_report.json", summary)
    write_json(reports_path / "v45_forecast_gate.json", summary)
    if gates:
        pd.json_normalize(gates).to_csv(reports_path / "v45_forecast_gates.csv", index=False)
    return summary


def synthetic_arrays(rows=1000, assets_per_module=2, local_features=3, seed=5):
    """Small auditable fixture, never a substitute for real-market training."""
    rng = np.random.default_rng(seed)
    n = 5*assets_per_module
    dates = pd.bdate_range("2018-01-02", periods=rows).asi8
    x = rng.normal(0, .5, (rows, n, local_features)).astype(np.float32)
    mask = rng.uniform(size=x.shape) > .05
    modules = np.repeat(np.arange(5), assets_per_module)
    y = np.stack([.002*x[:, :, 0]+rng.normal(0, .003, (rows, n)) for _ in range(3)], 1).astype(np.float32)
    sleeve = np.stack([y[:, :, modules == g].mean(2) for g in range(5)], -1)
    ends = np.full((rows, 3), np.iinfo(np.int64).min, dtype=np.int64)
    label_mask = np.ones_like(y, bool)
    sleeve_mask = np.ones_like(sleeve, bool)
    for h, horizon in enumerate((5, 20, 60)):
        ends[:-horizon, h] = dates[horizon:]
        label_mask[-horizon:, h], sleeve_mask[-horizon:, h] = False, False
    return dict(dates=dates, local_values=np.where(mask, x, 0), local_masks=mask,
        macro_values=np.eye(9, dtype=np.float32)[np.arange(rows) % 9],
        macro_masks=np.ones((rows, 9), bool), returns=rng.normal(0, .005, (rows, n)).astype(np.float32),
        return_masks=np.ones((rows, n), bool), asset_target=y, asset_target_mask=label_mask,
        sleeve_target=sleeve, sleeve_target_mask=sleeve_mask, label_end=ends,
        module_index=modules, sample_indices=np.arange(119, rows))


def throughput_probe(a, cfg, device=None, steps=3):
    """Real architecture forward/backward benchmark, no model artifact reuse."""
    torch.set_num_threads(cfg["threads"])
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    split = shared_split(a, cfg["years"][0], cfg)
    a, clipping = clip_fold_features(a, split["train"], cfg) if len(split["train"]) else (a, {})
    network = ForecastNetwork(a, cfg).to(device)
    optimizer = torch.optim.AdamW(network.parameters(), lr=cfg["pretrain_lr"])
    count = min(len(a["sample_indices"]), cfg["pretrain_batch"])
    batch = batch_at(a, a["sample_indices"][-count:], cfg, device)
    seconds = []
    for step in range(steps+1):
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss = network.reconstruction_loss(batch)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite throughput-probe loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), cfg["gradient_clip"], error_if_nonfinite=True)
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        if step:
            seconds.append(time.perf_counter()-started)
    return dict(device=str(device), threads=cfg["threads"], batch_size=count,
        asset_count=len(a["module_index"]), feature_count=a["local_values"].shape[-1],
        macro_input_count=a["macro_values"].shape[-1],
        numerical_policy="T_only_neural_input_clipping_and_nonfinite_gradient_rejection",
        seconds_per_pretrain_batch=float(np.median(seconds)),
        examples_per_second=float(count/np.median(seconds)),
        excludes="validation, checkpoint serialization, ridge OOF and supervised-stage costs")

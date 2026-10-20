from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
torch.set_num_threads(8)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

from src.hierarchical_asset_itransformer import (  # noqa: E402
    HierarchicalAssetITransformer,
    WindowDataset,
    load_arrays,
    load_config,
    multitask_loss,
    project_path,
    rank_ic,
)
from src.itransformer_signal import RISK_SLEEVES  # noqa: E402


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def training_signature(cfg: dict[str, Any]) -> str:
    """Bind resumable checkpoints to the exact data, logic, and train setup."""
    model_cfg = copy.deepcopy(cfg["model"])
    for operational_key in [
        "seeds",
        "progress_every_epochs",
    ]:
        model_cfg.pop(operational_key, None)
    payload = {
        "data": cfg["data"],
        "preprocessing": cfg["preprocessing"],
        "model": model_cfg,
        "walk_forward": cfg["walk_forward"],
        "dataset_sha256": _sha256_file(ROOT / cfg["outputs"]["dataset_npz"]),
        "model_source_sha256": _sha256_file(
            ROOT / "src" / "hierarchical_asset_itransformer.py"
        ),
        "trainer_source_sha256": _sha256_file(Path(__file__).resolve()),
    }
    canonical = json.dumps(
        payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def make_model(arrays, metadata: dict[str, Any], cfg: dict[str, Any]) -> HierarchicalAssetITransformer:
    return HierarchicalAssetITransformer(
        sequence_length=int(metadata["sequence_length_days"]),
        local_features=len(metadata["local_features"]),
        macro_features=len(metadata["macro_features"]),
        module_index=arrays.module_index,
        horizons=len(metadata["horizons_days"]),
        cfg=cfg["model"],
        asset_experts=len(metadata.get("asset_signal_experts", ["fixed_causal_baseline"])),
        sleeve_experts=len(metadata.get("sleeve_signal_experts", ["fixed_causal_baseline"])),
    )


def average_state_dicts(states: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not states:
        raise ValueError("At least one state_dict is required for checkpoint averaging")
    averaged: dict[str, torch.Tensor] = {}
    for key, reference in states[0].items():
        if torch.is_floating_point(reference):
            averaged[key] = torch.stack([state[key] for state in states], dim=0).mean(dim=0)
        else:
            averaged[key] = reference.clone()
    return averaged


def _robust_training_scale(
    values: np.ndarray,
    minimum: float,
    maximum: float,
) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return float(minimum)
    median = float(np.median(finite))
    mad_sigma = 1.4826 * float(np.median(np.abs(finite - median)))
    if not np.isfinite(mad_sigma) or mad_sigma <= 1e-12:
        mad_sigma = float(np.std(finite))
    if not np.isfinite(mad_sigma) or mad_sigma <= 1e-12:
        mad_sigma = float(minimum)
    return float(np.clip(mad_sigma, minimum, maximum))


def with_fold_training_loss_scales(
    cfg: dict[str, Any],
    arrays,
    train_indices: np.ndarray,
) -> dict[str, Any]:
    """Fit loss normalization on the purged training slice of one fold only."""
    mode = str(cfg["model"].get("regression_loss_scale_mode", "none"))
    if mode == "none":
        return cfg
    if mode != "fold_train_robust":
        raise ValueError(f"Unknown regression_loss_scale_mode={mode}")

    fold_cfg = copy.deepcopy(cfg)
    model_cfg = fold_cfg["model"]
    asset_floor = float(model_cfg.get("asset_regression_loss_scale_floor", 0.02))
    sleeve_floor = float(model_cfg.get("sleeve_regression_loss_scale_floor", 0.03))
    scale_ceiling = float(model_cfg.get("regression_loss_scale_ceiling", 1.50))
    horizons = arrays.asset_target.shape[1]
    modules = len(RISK_SLEEVES)
    prediction_target_mode = str(
        model_cfg.get("prediction_target_mode", "residual_rank")
    )
    asset_scale_target = arrays.asset_target[train_indices].copy()
    sleeve_scale_target = arrays.sleeve_target[train_indices].copy()
    if prediction_target_mode == "residual_rank":
        if bool(model_cfg.get("orthogonal_residual", False)):
            asset_scale_target = _orthogonalize_residual_numpy(
                asset_scale_target,
                arrays.asset_baseline[train_indices],
                arrays.asset_target_mask[train_indices],
                arrays.module_index,
            )
            sleeve_scale_target = _orthogonalize_residual_numpy(
                sleeve_scale_target,
                arrays.sleeve_baseline[train_indices],
                arrays.sleeve_target_mask[train_indices],
                None,
            )
        else:
            asset_scale_target -= arrays.asset_baseline[train_indices]
            sleeve_scale_target -= arrays.sleeve_baseline[train_indices]
    elif prediction_target_mode not in {
        "direct_rank",
        "direct_mean",
        "anchored_rank_boost",
    }:
        raise ValueError(
            f"Unknown prediction_target_mode={prediction_target_mode}"
        )

    asset_scales = np.ones((horizons, modules), dtype=np.float64)
    for horizon in range(horizons):
        for module in range(modules):
            columns = np.flatnonzero(arrays.module_index == module)
            values = asset_scale_target[:, horizon, columns]
            valid = arrays.asset_target_mask[train_indices, horizon][:, columns]
            asset_scales[horizon, module] = _robust_training_scale(
                values[valid],
                asset_floor,
                scale_ceiling,
            )

    sleeve_scales = np.ones((horizons, modules), dtype=np.float64)
    for horizon in range(horizons):
        for sleeve in range(modules):
            values = sleeve_scale_target[:, horizon, sleeve]
            valid = arrays.sleeve_target_mask[train_indices, horizon, sleeve]
            sleeve_scales[horizon, sleeve] = _robust_training_scale(
                values[valid],
                sleeve_floor,
                scale_ceiling,
            )

    model_cfg["asset_regression_loss_scale_by_module"] = asset_scales.tolist()
    model_cfg["sleeve_regression_loss_scale_by_sleeve"] = sleeve_scales.tolist()
    return fold_cfg


def append_progress(cfg: dict[str, Any], row: dict[str, Any]) -> None:
    if "training_progress" not in cfg.get("outputs", {}):
        return
    path = project_path(cfg, "training_progress")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, allow_nan=True) + "\n")


def to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def augment_training_batch(
    batch: dict[str, torch.Tensor], cfg: dict[str, Any]
) -> dict[str, torch.Tensor]:
    """Apply causal input corruption to training windows only."""
    aug = cfg["model"].get("training_augmentation", {})
    if not bool(aug.get("enabled", False)):
        return batch
    output = dict(batch)
    local_values = batch["local_values"].clone()
    local_masks = batch["local_masks"].clone()
    macro_values = batch["macro_values"].clone()

    feature_dropout = float(aug.get("local_feature_dropout", 0.0))
    if feature_dropout > 0.0:
        drop = torch.rand(
            local_values.shape[0], 1, local_values.shape[2], local_values.shape[3],
            device=local_values.device,
        ).lt(feature_dropout)
        local_values = local_values.masked_fill(drop, 0.0)
        local_masks = local_masks.masked_fill(drop, 0.0)

    macro_dropout = float(aug.get("macro_feature_dropout", 0.0))
    if macro_dropout > 0.0:
        drop = torch.rand(
            macro_values.shape[0], 1, macro_values.shape[2], device=macro_values.device
        ).lt(macro_dropout)
        macro_values = macro_values.masked_fill(drop, 0.0)

    block_probability = float(aug.get("time_block_probability", 0.0))
    block_days = min(int(aug.get("time_block_days", 0)), local_values.shape[1] - 1)
    if block_probability > 0.0 and block_days > 0:
        selected = torch.rand(local_values.shape[0], device=local_values.device).lt(block_probability)
        for row in torch.nonzero(selected, as_tuple=False).flatten().tolist():
            start = int(torch.randint(0, local_values.shape[1] - block_days, (1,)).item())
            stop = start + block_days
            local_values[row, start:stop] = 0.0
            local_masks[row, start:stop] = 0.0
            macro_values[row, start:stop] = 0.0

    noise_std = float(aug.get("noise_std", 0.0))
    if noise_std > 0.0:
        local_values = local_values + torch.randn_like(local_values) * noise_std * local_masks
        macro_values = macro_values + torch.randn_like(macro_values) * noise_std

    output["local_values"] = local_values
    output["local_masks"] = local_masks
    output["macro_values"] = macro_values
    return output


def model_forward(model, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
    return model(
        batch["local_values"],
        batch["local_masks"],
        batch["macro_values"],
        batch.get("regime_index"),
        batch.get("regime_prior_count"),
        batch.get("asset_experts"),
        batch.get("asset_expert_masks"),
        batch.get("sleeve_experts"),
        batch.get("sleeve_expert_masks"),
    )


class MaskedHistoryPretrainer(nn.Module):
    """Pretrain the exact forecasting encoder on point-in-time reconstruction."""

    def __init__(self, model: HierarchicalAssetITransformer) -> None:
        super().__init__()
        self.model = model
        d_model = int(model.cfg["d_model"])
        self.local_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, model.local_features),
        )
        self.macro_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, max(d_model // 2, 8)),
            nn.GELU(),
            nn.Linear(max(d_model // 2, 8), 1),
        )

    def forward(
        self,
        local_values: torch.Tensor,
        local_masks: torch.Tensor,
        macro_values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded_assets, encoded_macro, _, _ = self.model.encode_context(
            local_values, local_masks, macro_values
        )
        return self.local_head(encoded_assets), self.macro_head(encoded_macro).squeeze(-1)


def _masked_pretraining_inputs(
    batch: dict[str, torch.Tensor],
    cfg: dict[str, Any],
    generator: torch.Generator | None = None,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pre_cfg = cfg["model"].get("self_supervised_pretraining", {})
    local_probability = float(pre_cfg.get("local_mask_probability", 0.15))
    macro_probability = float(pre_cfg.get("macro_mask_probability", 0.15))
    mask_mode = str(pre_cfg.get("mask_mode", "full_path")).strip().lower()
    full_path_fraction = float(pre_cfg.get("full_path_mask_fraction", 1.0))
    suffix_days = [
        max(int(value), 1)
        for value in pre_cfg.get("causal_suffix_days", [1, 5, 20])
    ]
    if mask_mode not in {"full_path", "mixed_causal_suffix"}:
        raise ValueError(
            "self_supervised_pretraining.mask_mode must be full_path or "
            "mixed_causal_suffix"
        )
    if not 0.0 <= full_path_fraction <= 1.0:
        raise ValueError(
            "self_supervised_pretraining.full_path_mask_fraction must be in [0, 1]"
        )
    local_values = batch["local_values"].clone()
    local_masks = batch["local_masks"].clone()
    macro_values = batch["macro_values"].clone()
    local_target = local_values[:, -1].clone()
    macro_target = macro_values[:, -1].clone()
    local_available = local_masks[:, -1].gt(0.0)
    local_mask = (
        torch.rand(
            local_available.shape,
            device=local_values.device,
            generator=generator,
        ).lt(local_probability)
        & local_available
    )
    macro_mask = torch.rand(
        macro_target.shape,
        device=macro_values.device,
        generator=generator,
    ).lt(macro_probability)

    # A suffix mask forces the encoder to infer the recent state transition from
    # the earlier, already-observable history.  It is still strictly causal: the
    # reconstruction target is the last timestamp inside the input window, never
    # a forward return or a value after the as-of date.  A small full-path share
    # retains cross-variable imputation pressure and prevents a trivial last-step
    # copying solution.
    if mask_mode == "full_path":
        local_corruption_mask = local_mask[:, None, :, :]
        macro_corruption_mask = macro_mask[:, None, :]
    else:
        sequence_length = int(local_values.shape[1])
        local_choices = torch.randint(
            low=0,
            high=len(suffix_days),
            size=local_mask.shape,
            device=local_values.device,
            generator=generator,
        )
        macro_choices = torch.randint(
            low=0,
            high=len(suffix_days),
            size=macro_mask.shape,
            device=macro_values.device,
            generator=generator,
        )
        local_suffix = torch.as_tensor(
            suffix_days, device=local_values.device, dtype=torch.long
        )[local_choices].clamp(max=sequence_length)
        macro_suffix = torch.as_tensor(
            suffix_days, device=macro_values.device, dtype=torch.long
        )[macro_choices].clamp(max=sequence_length)
        local_full = torch.rand(
            local_mask.shape,
            device=local_values.device,
            generator=generator,
        ).lt(full_path_fraction)
        macro_full = torch.rand(
            macro_mask.shape,
            device=macro_values.device,
            generator=generator,
        ).lt(full_path_fraction)
        local_suffix = torch.where(
            local_full,
            torch.full_like(local_suffix, sequence_length),
            local_suffix,
        )
        macro_suffix = torch.where(
            macro_full,
            torch.full_like(macro_suffix, sequence_length),
            macro_suffix,
        )
        time = torch.arange(sequence_length, device=local_values.device)
        local_tail = time[None, :, None, None].ge(
            sequence_length - local_suffix[:, None, :, :]
        )
        macro_tail = time[None, :, None].ge(
            sequence_length - macro_suffix[:, None, :]
        )
        local_corruption_mask = local_tail & local_mask[:, None, :, :]
        macro_corruption_mask = macro_tail & macro_mask[:, None, :]

    # The observed/missing channel is preserved, so zero is an explicit masked
    # value rather than fake missing data.
    local_values = local_values.masked_fill(local_corruption_mask, 0.0)
    macro_values = macro_values.masked_fill(macro_corruption_mask, 0.0)
    corrupted = dict(batch)
    corrupted["local_values"] = local_values
    corrupted["local_masks"] = local_masks
    corrupted["macro_values"] = macro_values
    return corrupted, local_target, macro_target, local_mask, macro_mask


def _masked_pretraining_loss(
    pretrainer: MaskedHistoryPretrainer,
    batch: dict[str, torch.Tensor],
    cfg: dict[str, Any],
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    corrupted, local_target, macro_target, local_mask, macro_mask = (
        _masked_pretraining_inputs(batch, cfg, generator)
    )
    local_prediction, macro_prediction = pretrainer(
        corrupted["local_values"],
        corrupted["local_masks"],
        corrupted["macro_values"],
    )
    zero = local_prediction.sum() * 0.0
    local_loss = (
        F.smooth_l1_loss(
            local_prediction[local_mask],
            local_target[local_mask],
            beta=0.50,
        )
        if local_mask.any()
        else zero
    )
    macro_loss = (
        F.smooth_l1_loss(
            macro_prediction[macro_mask],
            macro_target[macro_mask],
            beta=0.50,
        )
        if macro_mask.any()
        else zero
    )
    macro_weight = float(
        cfg["model"].get("self_supervised_pretraining", {}).get(
            "macro_loss_weight", 0.50
        )
    )
    loss = local_loss + macro_weight * macro_loss
    return loss, {
        "pretrain_local_loss": float(local_loss.detach()),
        "pretrain_macro_loss": float(macro_loss.detach()),
    }


def pretrain_encoder(
    model: HierarchicalAssetITransformer,
    arrays,
    metadata: dict[str, Any],
    cfg: dict[str, Any],
    train_indices: np.ndarray,
    device: torch.device,
    seed: int,
    year: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Masked-history pretraining using only windows before the fold cutoff."""
    pre_cfg = cfg["model"].get("self_supervised_pretraining", {})
    if not bool(pre_cfg.get("enabled", False)):
        return [], {"enabled": False}
    indices = np.asarray(sorted(set(int(value) for value in train_indices)), dtype=np.int64)
    minimum_samples = int(pre_cfg.get("minimum_samples", 240))
    if len(indices) < minimum_samples:
        return [], {
            "enabled": False,
            "reason": "insufficient_samples",
            "available_samples": int(len(indices)),
        }
    validation_fraction = float(pre_cfg.get("validation_fraction", 0.15))
    validation_count = max(
        int(pre_cfg.get("minimum_validation_samples", 60)),
        int(math.ceil(len(indices) * validation_fraction)),
    )
    validation_count = min(validation_count, max(len(indices) // 3, 1))
    purge = int(pre_cfg.get("validation_purge_days", 20))
    val_indices = indices[-validation_count:]
    train_stop = max(len(indices) - validation_count - purge, 1)
    pretrain_indices = indices[:train_stop]
    if len(pretrain_indices) < minimum_samples // 2:
        pretrain_indices = indices[: len(indices) - validation_count]

    sequence_length = int(metadata["sequence_length_days"])
    train_set = WindowDataset(arrays, pretrain_indices, sequence_length, None)
    val_set = WindowDataset(arrays, val_indices, sequence_length, None)
    batch_size = int(pre_cfg.get("batch_size", cfg["model"]["batch_size"]))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    pretrainer = MaskedHistoryPretrainer(model).to(device)
    optimizer = torch.optim.AdamW(
        pretrainer.parameters(),
        lr=float(pre_cfg.get("learning_rate", 2e-4)),
        weight_decay=float(pre_cfg.get("weight_decay", 0.01)),
    )
    maximum_epochs = int(pre_cfg.get("max_epochs", 200))
    minimum_epochs = int(pre_cfg.get("minimum_epochs", 20))
    patience = int(pre_cfg.get("patience", 20))
    minimum_delta = float(pre_cfg.get("min_delta", 1e-4))
    best_loss = np.inf
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    stale = 0
    logs: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(1, maximum_epochs + 1):
        pretrainer.train()
        train_total = 0.0
        train_count = 0
        train_components: dict[str, float] = {}
        for raw in train_loader:
            batch = to_device(raw, device)
            loss, components = _masked_pretraining_loss(pretrainer, batch, cfg)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                pretrainer.parameters(),
                float(pre_cfg.get("gradient_clip", 1.0)),
            )
            optimizer.step()
            batch_rows = len(batch["index"])
            train_total += float(loss.detach()) * batch_rows
            train_count += batch_rows
            for key, value in components.items():
                train_components[key] = (
                    train_components.get(key, 0.0) + value * batch_rows
                )

        pretrainer.eval()
        val_total = 0.0
        val_count = 0
        validation_generator = torch.Generator(device=device.type)
        validation_generator.manual_seed(seed * 100_003 + year * 101 + 7)
        with torch.no_grad():
            for raw in val_loader:
                batch = to_device(raw, device)
                loss, _ = _masked_pretraining_loss(
                    pretrainer, batch, cfg, validation_generator
                )
                batch_rows = len(batch["index"])
                val_total += float(loss) * batch_rows
                val_count += batch_rows
        validation_loss = val_total / max(val_count, 1)
        improved = bool(
            epoch >= minimum_epochs
            and validation_loss < best_loss - minimum_delta
        )
        if improved:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        elif epoch >= minimum_epochs:
            stale += 1
        row = {
            "phase": "masked_history_pretraining",
            "year": year,
            "seed": seed,
            "epoch": epoch,
            "train_loss": train_total / max(train_count, 1),
            "validation_loss": validation_loss,
            "best_validation_loss": best_loss,
            "best_epoch": best_epoch,
            "elapsed_seconds": time.perf_counter() - started,
            **{
                key: value / max(train_count, 1)
                for key, value in train_components.items()
            },
        }
        logs.append(row)
        if epoch == 1 or epoch % int(pre_cfg.get("progress_every_epochs", 10)) == 0:
            print(json.dumps(row, ensure_ascii=False), flush=True)
            append_progress(cfg, row)
        if epoch >= minimum_epochs and stale >= patience:
            break
    model.load_state_dict(best_state)
    return logs, {
        "enabled": True,
        "best_epoch": int(best_epoch),
        "best_validation_loss": float(best_loss),
        "epochs_run": int(logs[-1]["epoch"]) if logs else 0,
        "train_samples": int(len(pretrain_indices)),
        "validation_samples": int(len(val_indices)),
    }


def _set_supervised_encoder_frozen(
    model: HierarchicalAssetITransformer, frozen: bool
) -> None:
    prediction_prefixes = (
        "asset_mean",
        "asset_logvar",
        "asset_mean_by_module",
        "asset_logvar_by_module",
        "macro_pool_score",
        "sleeve_head",
        "sleeve_mean",
        "sleeve_logvar",
        "asset_expert_gate",
        "sleeve_expert_gate",
        "sleeve_gate",
        "sleeve_rank_",
        "transition_head",
        "bond_transition_head",
        "equity_transition_head",
        "grid_interaction_head",
        "state_payoff_",
    )
    for name, parameter in model.named_parameters():
        if name.startswith(prediction_prefixes):
            parameter.requires_grad = True
        else:
            parameter.requires_grad = not frozen


def prediction_heads(
    output: tuple[torch.Tensor, ...],
    cfg: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if (
        len(output) == 6
        and str((cfg or {}).get("prediction_head_mode", "unconstrained"))
        == "adaptive_expert_mixture"
    ):
        return output[0], output[1], output[2], output[3]
    if len(output) == 4:
        return output
    if len(output) == 6:
        asset_mean, asset_logvar, _, sleeve_logvar, sleeve_rank_strength, _ = output
        return asset_mean, asset_logvar, sleeve_rank_strength, sleeve_logvar
    if len(output) == 7:
        asset_mean, asset_logvar, _, sleeve_logvar, sleeve_rank_strength, _, _ = output
        return asset_mean, asset_logvar, sleeve_rank_strength, sleeve_logvar
    if len(output) == 8:
        asset_mean, asset_logvar, _, sleeve_logvar, sleeve_rank_strength, _, _, _ = output
        return asset_mean, asset_logvar, sleeve_rank_strength, sleeve_logvar
    if len(output) == 10:
        asset_mean, asset_logvar, _, sleeve_logvar, sleeve_rank_strength, _, _, _, _, _ = output
        return asset_mean, asset_logvar, sleeve_rank_strength, sleeve_logvar
    raise ValueError(f"Unexpected model output count: {len(output)}")


def evaluate_loss(model, loader, cfg, device) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for raw in loader:
            batch = to_device(raw, device)
            output = model_forward(model, batch)
            loss, _ = multitask_loss(output, batch, model.module_index, cfg["model"])
            total += float(loss) * len(batch["index"])
            count += len(batch["index"])
    return total / max(count, 1)


def _orthogonalize_residual_numpy(
    values: np.ndarray,
    baseline: np.ndarray,
    mask: np.ndarray,
    groups: np.ndarray | None,
) -> np.ndarray:
    output = np.zeros_like(values, dtype=np.float32)
    column_groups = (
        [np.arange(values.shape[2], dtype=np.int64)]
        if groups is None
        else [np.flatnonzero(groups == group) for group in np.unique(groups)]
    )
    for columns in column_groups:
        if not len(columns):
            continue
        valid = mask[:, :, columns].astype(bool)
        weights = valid.astype(np.float64)
        counts = np.maximum(weights.sum(axis=2, keepdims=True), 1.0)
        selected_values = values[:, :, columns].astype(np.float64)
        selected_baseline = baseline[:, :, columns].astype(np.float64)
        value_centered = selected_values - (
            selected_values * weights
        ).sum(axis=2, keepdims=True) / counts
        baseline_centered = selected_baseline - (
            selected_baseline * weights
        ).sum(axis=2, keepdims=True) / counts
        covariance = (
            value_centered * baseline_centered * weights
        ).sum(axis=2, keepdims=True)
        baseline_variance = (
            baseline_centered**2 * weights
        ).sum(axis=2, keepdims=True)
        projection = covariance / np.maximum(baseline_variance, 1e-6)
        residual = value_centered - projection * baseline_centered
        residual -= (residual * weights).sum(axis=2, keepdims=True) / counts
        output[:, :, columns] = np.where(valid, residual, 0.0).astype(np.float32)
    return output


def _postprocess_prediction_bundle(
    bundle: dict[str, np.ndarray],
    module_index: np.ndarray,
    cfg: dict[str, Any] | None,
) -> dict[str, np.ndarray]:
    model_cfg = (cfg or {}).get("model", cfg or {})
    if not (
        str(model_cfg.get("prediction_target_mode", "residual_rank"))
        in {"residual_rank", "anchored_rank_boost"}
        and bool(model_cfg.get("orthogonal_residual", False))
    ):
        return bundle
    output = dict(bundle)
    output["asset_residual"] = _orthogonalize_residual_numpy(
        bundle["asset_residual"],
        bundle["asset_baseline"],
        bundle["asset_mask"],
        module_index,
    )
    output["sleeve_residual"] = _orthogonalize_residual_numpy(
        bundle["sleeve_residual"],
        bundle["sleeve_baseline"],
        bundle["sleeve_mask"],
        None,
    )
    return output


def evaluate_with_predictions(model, loader, cfg, device) -> tuple[float, dict[str, np.ndarray]]:
    model.eval()
    total = 0.0
    count = 0
    collected: dict[str, list[np.ndarray]] = {
        "index": [], "asset_residual": [], "asset_logvar": [],
        "sleeve_residual": [], "sleeve_logvar": [],
        "asset_target": [], "asset_baseline": [], "asset_mask": [],
        "sleeve_target": [], "sleeve_baseline": [], "sleeve_mask": [],
    }
    with torch.no_grad():
        for raw in loader:
            batch = to_device(raw, device)
            output = model_forward(model, batch)
            loss, _ = multitask_loss(output, batch, model.module_index, cfg["model"])
            batch_size = len(batch["index"])
            total += float(loss) * batch_size
            count += batch_size
            asset_mean, asset_logvar, sleeve_mean, sleeve_logvar = prediction_heads(
                output, cfg["model"]
            )
            values = {
                "index": batch["index"], "asset_residual": asset_mean,
                "asset_logvar": asset_logvar, "sleeve_residual": sleeve_mean,
                "sleeve_logvar": sleeve_logvar, "asset_target": batch["asset_target"],
                "asset_baseline": batch["asset_baseline"], "asset_mask": batch["asset_mask"],
                "sleeve_target": batch["sleeve_target"], "sleeve_baseline": batch["sleeve_baseline"],
                "sleeve_mask": batch["sleeve_mask"],
            }
            if (
                len(output) == 6
                and str(cfg["model"].get("prediction_head_mode", "unconstrained"))
                == "adaptive_expert_mixture"
            ):
                values["asset_expert_gate"] = output[4]
                values["sleeve_expert_gate"] = output[5]
                collected.setdefault("asset_expert_gate", [])
                collected.setdefault("sleeve_expert_gate", [])
            for key, value in values.items():
                collected[key].append(value.detach().cpu().numpy())
    bundle = {key: np.concatenate(values, axis=0) for key, values in collected.items()}
    bundle = _postprocess_prediction_bundle(
        bundle, model.module_index.detach().cpu().numpy(), cfg
    )
    return total / max(count, 1), bundle


def _prediction_correction(
    baseline: np.ndarray,
    model_output: np.ndarray,
    cfg: dict[str, Any] | None,
) -> np.ndarray:
    mode = str((cfg or {}).get("model", {}).get("prediction_target_mode", "residual_rank"))
    if mode in {"direct_rank", "direct_mean"}:
        return model_output - baseline
    if mode == "residual_rank":
        return model_output
    if mode == "anchored_rank_boost":
        return model_output
    raise ValueError(f"Unknown prediction_target_mode={mode}")


def _blend_prediction(
    baseline: np.ndarray,
    model_output: np.ndarray,
    baseline_weight: float,
    model_alpha: float,
    cfg: dict[str, Any] | None,
) -> np.ndarray:
    correction = _prediction_correction(baseline, model_output, cfg)
    return float(baseline_weight) * baseline + float(model_alpha) * correction


def _validation_horizon_value(
    raw: Any,
    horizon: int,
    count: int,
    default: float,
) -> float:
    if raw is None:
        return float(default)
    if isinstance(raw, (list, tuple, np.ndarray)):
        if len(raw) != count:
            raise ValueError(
                f"validation horizon parameter must contain {count} values"
            )
        return float(raw[horizon])
    return float(raw)


def _masked_softmax_numpy(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    selected = np.where(valid, values, -np.inf)
    maximum = np.max(selected)
    if not np.isfinite(maximum):
        return np.zeros_like(values, dtype=float)
    exponent = np.where(valid, np.exp(selected - maximum), 0.0)
    return exponent / max(float(exponent.sum()), 1e-12)


def _decision_net_utility_by_horizon(
    prediction: np.ndarray,
    baseline: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    groups: np.ndarray | None,
    horizon_days: list[int],
    temperature: Any,
    transaction_cost_bps: Any,
) -> tuple[np.ndarray, np.ndarray]:
    if len(horizon_days) != prediction.shape[1]:
        raise ValueError("decision_horizons_days must match prediction horizons")
    group_values = [None] if groups is None else np.unique(groups).tolist()
    means = np.full(prediction.shape[1], np.nan, dtype=float)
    positive = np.full(prediction.shape[1], np.nan, dtype=float)
    for horizon in range(prediction.shape[1]):
        tau = max(
            _validation_horizon_value(
                temperature, horizon, prediction.shape[1], 0.10
            ),
            1e-4,
        )
        cost_rate = (
            _validation_horizon_value(
                transaction_cost_bps,
                horizon,
                prediction.shape[1],
                5.0,
            )
            * 1e-4
            * 252.0
            / max(float(horizon_days[horizon]), 1.0)
        )
        values: list[float] = []
        for row in range(prediction.shape[0]):
            for group in group_values:
                valid = mask[row, horizon].astype(bool).copy()
                if group is not None:
                    valid &= groups == group
                if valid.sum() < 2:
                    continue
                pred_weight = _masked_softmax_numpy(
                    prediction[row, horizon] / tau, valid
                )
                baseline_weight = _masked_softmax_numpy(
                    baseline[row, horizon] / tau, valid
                )
                active = pred_weight - baseline_weight
                gross = float(np.sum(active[valid] * target[row, horizon, valid]))
                turnover = 0.5 * float(np.abs(active[valid]).sum())
                values.append(gross - cost_rate * turnover)
        if values:
            array = np.asarray(values, dtype=float)
            means[horizon] = float(np.mean(array))
            positive[horizon] = float(np.mean(array > 0.0))
    return means, positive


def _decision_net_utility_series(
    prediction: np.ndarray,
    baseline: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    groups: np.ndarray | None,
    temperature: float,
    transaction_cost_bps: float,
    horizon_days: int,
) -> np.ndarray:
    """Return one cost-adjusted active-utility observation per as-of date."""
    group_values = [None] if groups is None else np.unique(groups).tolist()
    tau = max(float(temperature), 1e-4)
    cost_rate = (
        float(transaction_cost_bps)
        * 1e-4
        * 252.0
        / max(float(horizon_days), 1.0)
    )
    output = np.full(prediction.shape[0], np.nan, dtype=float)
    for row in range(prediction.shape[0]):
        row_values: list[float] = []
        for group in group_values:
            valid = mask[row].astype(bool).copy()
            if group is not None:
                valid &= groups == group
            if valid.sum() < 2:
                continue
            predicted_weight = _masked_softmax_numpy(
                prediction[row] / tau, valid
            )
            baseline_weight = _masked_softmax_numpy(
                baseline[row] / tau, valid
            )
            active = predicted_weight - baseline_weight
            gross = float(np.sum(active[valid] * target[row, valid]))
            turnover = 0.5 * float(np.abs(active[valid]).sum())
            row_values.append(gross - cost_rate * turnover)
        if row_values:
            output[row] = float(np.mean(row_values))
    return output


def _normalized_task_horizon_weights(
    model_cfg: dict[str, Any],
    task: str,
    horizon_count: int,
) -> np.ndarray:
    raw = model_cfg.get(
        f"validation_{task}_horizon_weights",
        model_cfg.get(
            f"{task}_horizon_loss_weights",
            model_cfg.get(
                "validation_horizon_weights",
                model_cfg.get("horizon_loss_weights", np.ones(horizon_count)),
            ),
        ),
    )
    weights = np.asarray(raw, dtype=float)
    if len(weights) != horizon_count:
        raise ValueError(
            f"validation_{task}_horizon_weights must match forecast horizons"
        )
    weights = np.where(
        np.isfinite(weights) & (weights > 0.0),
        weights,
        0.0,
    )
    if float(weights.sum()) <= 0.0:
        raise ValueError(
            f"validation_{task}_horizon_weights must contain a positive value"
        )
    return weights / float(weights.sum())


def _normalized_asset_module_horizon_weights(
    model_cfg: dict[str, Any],
    horizon_count: int,
) -> dict[str, np.ndarray] | None:
    """Mirror module-specific training weights in validation and checkpointing."""
    raw = model_cfg.get(
        "validation_asset_module_horizon_weights",
        model_cfg.get("asset_module_horizon_loss_weights"),
    )
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(
            "validation_asset_module_horizon_weights must map sleeve names "
            "to horizon weights"
        )
    unknown = sorted(set(raw) - set(RISK_SLEEVES))
    if unknown:
        raise ValueError(
            "Unknown validation_asset_module_horizon_weights sleeves: "
            + ", ".join(unknown)
        )
    fallback = _normalized_task_horizon_weights(
        model_cfg, "asset", horizon_count
    )
    output: dict[str, np.ndarray] = {}
    for module in RISK_SLEEVES:
        values = raw.get(module)
        if values is None:
            output[module] = fallback.copy()
            continue
        weights = np.asarray(values, dtype=float)
        if len(weights) != horizon_count:
            raise ValueError(
                "validation_asset_module_horizon_weights."
                f"{module} must match forecast horizons"
            )
        weights = np.where(
            np.isfinite(weights) & (weights > 0.0),
            weights,
            0.0,
        )
        if float(weights.sum()) <= 0.0:
            raise ValueError(
                "validation_asset_module_horizon_weights."
                f"{module} must contain a positive value"
            )
        output[module] = weights / float(weights.sum())
    return output


def _weighted_finite_mean(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    finite = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not finite.any():
        return float("nan")
    return float(
        np.sum(values[finite] * weights[finite])
        / max(float(weights[finite].sum()), 1e-12)
    )


def _rank_components(
    bundle: dict[str, np.ndarray],
    module_index: np.ndarray,
    cfg: dict[str, Any] | None = None,
) -> tuple[float, dict[str, float]]:
    asset_model_by_horizon = horizon_rank_ics(
        bundle["asset_residual"], bundle["asset_target"], bundle["asset_mask"], module_index
    )
    asset_model = float(np.nanmean(asset_model_by_horizon))
    asset_correction = _prediction_correction(
        bundle["asset_baseline"], bundle["asset_residual"], cfg
    )
    validation_alpha = float(
        (cfg or {}).get("model", {}).get("validation_correction_alpha", 1.0)
    )
    asset_combined_by_horizon = horizon_rank_ics(
        bundle["asset_baseline"] + validation_alpha * asset_correction,
        bundle["asset_target"],
        bundle["asset_mask"],
        module_index,
    )
    asset_combined = float(np.nanmean(asset_combined_by_horizon))
    sleeve_model_by_horizon = horizon_rank_ics(
        bundle["sleeve_residual"], bundle["sleeve_target"], bundle["sleeve_mask"], None
    )
    sleeve_model = float(np.nanmean(sleeve_model_by_horizon))
    sleeve_correction = _prediction_correction(
        bundle["sleeve_baseline"], bundle["sleeve_residual"], cfg
    )
    sleeve_combined_by_horizon = horizon_rank_ics(
        bundle["sleeve_baseline"] + validation_alpha * sleeve_correction,
        bundle["sleeve_target"],
        bundle["sleeve_mask"],
        None,
    )
    sleeve_combined = float(np.nanmean(sleeve_combined_by_horizon))
    asset_baseline_by_horizon = horizon_rank_ics(
        bundle["asset_baseline"],
        bundle["asset_target"],
        bundle["asset_mask"],
        module_index,
    )
    if (
        str((cfg or {}).get("data", {}).get("forecast_baseline_mode", ""))
        == "zero_active_view"
    ):
        asset_baseline_by_horizon = np.where(
            np.isfinite(asset_baseline_by_horizon),
            asset_baseline_by_horizon,
            0.0,
        )
    asset_baseline = float(np.nanmean(asset_baseline_by_horizon))
    sleeve_baseline_by_horizon = horizon_rank_ics(
        bundle["sleeve_baseline"],
        bundle["sleeve_target"],
        bundle["sleeve_mask"],
        None,
    )
    if (
        str((cfg or {}).get("data", {}).get("forecast_baseline_mode", ""))
        == "zero_active_view"
    ):
        sleeve_baseline_by_horizon = np.where(
            np.isfinite(sleeve_baseline_by_horizon),
            sleeve_baseline_by_horizon,
            0.0,
        )
    sleeve_baseline = float(np.nanmean(sleeve_baseline_by_horizon))
    asset_incremental_by_horizon = (
        asset_combined_by_horizon - asset_baseline_by_horizon
    )
    sleeve_incremental_by_horizon = (
        sleeve_combined_by_horizon - sleeve_baseline_by_horizon
    )
    asset_incremental = asset_combined - asset_baseline
    sleeve_incremental = sleeve_combined - sleeve_baseline
    model_cfg = (cfg or {}).get("model", {})
    asset_horizon_weights = _normalized_task_horizon_weights(
        model_cfg,
        "asset",
        len(asset_incremental_by_horizon),
    )
    asset_module_horizon_weights = (
        _normalized_asset_module_horizon_weights(
            model_cfg,
            len(asset_incremental_by_horizon),
        )
    )
    sleeve_horizon_weights = _normalized_task_horizon_weights(
        model_cfg,
        "sleeve",
        len(sleeve_incremental_by_horizon),
    )
    asset_weighted_rank = _weighted_finite_mean(
        asset_incremental_by_horizon,
        asset_horizon_weights,
    )
    sleeve_weighted_rank = _weighted_finite_mean(
        sleeve_incremental_by_horizon,
        sleeve_horizon_weights,
    )
    asset_module_rank_components: dict[
        str, tuple[np.ndarray, np.ndarray, np.ndarray]
    ] = {}
    for module_position, module_name in enumerate(RISK_SLEEVES):
        columns = np.flatnonzero(module_index == module_position)
        if len(columns) < 2:
            continue
        module_combined = horizon_rank_ics(
            (
                bundle["asset_baseline"]
                + validation_alpha * asset_correction
            )[..., columns],
            bundle["asset_target"][..., columns],
            bundle["asset_mask"][..., columns],
            None,
        )
        module_baseline = horizon_rank_ics(
            bundle["asset_baseline"][..., columns],
            bundle["asset_target"][..., columns],
            bundle["asset_mask"][..., columns],
            None,
        )
        if (
            str((cfg or {}).get("data", {}).get("forecast_baseline_mode", ""))
            == "zero_active_view"
        ):
            module_baseline = np.where(
                np.isfinite(module_baseline),
                module_baseline,
                0.0,
            )
        asset_module_rank_components[module_name] = (
            module_combined,
            module_baseline,
            module_combined - module_baseline,
        )
    if asset_module_horizon_weights is not None:
        module_weighted_rank = [
            _weighted_finite_mean(
                components[2],
                asset_module_horizon_weights[module_name],
            )
            for module_name, components in asset_module_rank_components.items()
        ]
        finite_module_rank = np.asarray(module_weighted_rank, dtype=float)
        finite_module_rank = finite_module_rank[np.isfinite(finite_module_rank)]
        asset_weighted_rank = (
            float(np.mean(finite_module_rank))
            if len(finite_module_rank)
            else float("nan")
        )
        asset_active_rank_components = np.concatenate(
            [
                components[2][
                    (asset_module_horizon_weights[module_name] > 0.0)
                    & np.isfinite(components[2])
                ]
                for module_name, components in asset_module_rank_components.items()
            ]
        )
    else:
        asset_active_rank_components = asset_incremental_by_horizon[
            (asset_horizon_weights > 0.0)
            & np.isfinite(asset_incremental_by_horizon)
        ]
    decision_horizons = [
        int(value)
        for value in model_cfg.get(
            "decision_horizons_days",
            list(range(1, len(asset_incremental_by_horizon) + 1)),
        )
    ]
    asset_decision_utility, asset_decision_positive = (
        _decision_net_utility_by_horizon(
            bundle["asset_baseline"] + validation_alpha * asset_correction,
            bundle["asset_baseline"],
            bundle["asset_target"],
            bundle["asset_mask"],
            module_index,
            decision_horizons,
            model_cfg.get("asset_decision_temperature", 0.10),
            model_cfg.get("decision_transaction_cost_bps", 5.0),
        )
    )
    asset_module_decision_utility: dict[str, np.ndarray] = {}
    asset_module_decision_positive: dict[str, np.ndarray] = {}
    for module_position, module_name in enumerate(RISK_SLEEVES):
        columns = np.flatnonzero(module_index == module_position)
        if len(columns) < 2:
            continue
        module_utility, module_positive = _decision_net_utility_by_horizon(
            (
                bundle["asset_baseline"]
                + validation_alpha * asset_correction
            )[..., columns],
            bundle["asset_baseline"][..., columns],
            bundle["asset_target"][..., columns],
            bundle["asset_mask"][..., columns],
            None,
            decision_horizons,
            model_cfg.get("asset_decision_temperature", 0.10),
            model_cfg.get("decision_transaction_cost_bps", 5.0),
        )
        asset_module_decision_utility[module_name] = module_utility
        asset_module_decision_positive[module_name] = module_positive
    sleeve_decision_utility, sleeve_decision_positive = (
        _decision_net_utility_by_horizon(
            bundle["sleeve_baseline"] + validation_alpha * sleeve_correction,
            bundle["sleeve_baseline"],
            bundle["sleeve_target"],
            bundle["sleeve_mask"],
            None,
            decision_horizons,
            model_cfg.get("sleeve_decision_temperature", 0.10),
            model_cfg.get("decision_transaction_cost_bps", 5.0),
        )
    )
    if asset_module_horizon_weights is not None:
        weighted_module_utility = [
            _weighted_finite_mean(
                values,
                asset_module_horizon_weights[module_name],
            )
            for module_name, values in asset_module_decision_utility.items()
        ]
        finite_module_utility = np.asarray(weighted_module_utility, dtype=float)
        finite_module_utility = finite_module_utility[
            np.isfinite(finite_module_utility)
        ]
        asset_weighted_decision_utility = (
            float(np.mean(finite_module_utility))
            if len(finite_module_utility)
            else float("nan")
        )
        asset_active_decision_components = np.concatenate(
            [
                values[
                    (asset_module_horizon_weights[module_name] > 0.0)
                    & np.isfinite(values)
                ]
                for module_name, values in asset_module_decision_utility.items()
            ]
        )
    else:
        asset_weighted_decision_utility = _weighted_finite_mean(
            asset_decision_utility, asset_horizon_weights
        )
        asset_active_decision_components = asset_decision_utility[
            (asset_horizon_weights > 0.0)
            & np.isfinite(asset_decision_utility)
        ]
    selection_mode = str(
        model_cfg.get(
            "validation_selection_metric", "best_absolute_rank"
        )
    )
    if selection_mode == "best_absolute_rank":
        asset_score = np.nanmax([asset_model, asset_combined])
        sleeve_score = np.nanmax([sleeve_model, sleeve_combined])
        score = float(np.nanmean([asset_score, sleeve_score]))
    elif selection_mode == "combined_incremental":
        model_cfg = (cfg or {}).get("model", {})
        asset_weight = float(model_cfg.get("validation_asset_increment_weight", 0.55))
        sleeve_weight = float(model_cfg.get("validation_sleeve_increment_weight", 0.45))
        total_weight = max(asset_weight + sleeve_weight, 1e-12)
        asset_weight /= total_weight
        sleeve_weight /= total_weight
        weighted_increment = (
            asset_weight * asset_incremental
            + sleeve_weight * sleeve_incremental
        )
        minimum_component = np.nanmin([asset_incremental, sleeve_incremental])
        minimum_weight = float(
            model_cfg.get("validation_minimum_component_weight", 0.35)
        )
        score = float(
            (1.0 - minimum_weight) * weighted_increment
            + minimum_weight * minimum_component
        )
    elif selection_mode == "horizon_balanced_incremental":
        asset_weight = float(
            model_cfg.get("validation_asset_increment_weight", 0.55)
        )
        sleeve_weight = float(
            model_cfg.get("validation_sleeve_increment_weight", 0.45)
        )
        task_weight_total = max(asset_weight + sleeve_weight, 1e-12)
        asset_weight /= task_weight_total
        sleeve_weight /= task_weight_total
        horizon_weights = np.asarray(
            model_cfg.get(
                "validation_horizon_weights",
                model_cfg.get(
                    "horizon_loss_weights",
                    np.ones(len(asset_incremental_by_horizon)),
                ),
            ),
            dtype=float,
        )
        if len(horizon_weights) != len(asset_incremental_by_horizon):
            raise ValueError(
                "validation_horizon_weights must match the forecast horizon count"
            )
        horizon_weights = np.where(
            np.isfinite(horizon_weights) & (horizon_weights > 0.0),
            horizon_weights,
            0.0,
        )
        horizon_weights /= max(float(horizon_weights.sum()), 1e-12)
        combined_by_horizon = (
            asset_weight * asset_incremental_by_horizon
            + sleeve_weight * sleeve_incremental_by_horizon
        )
        finite_horizons = np.isfinite(combined_by_horizon)
        weighted_increment = float(
            np.sum(
                horizon_weights[finite_horizons]
                * combined_by_horizon[finite_horizons]
            )
            / max(float(horizon_weights[finite_horizons].sum()), 1e-12)
        )
        all_components = np.concatenate(
            [asset_incremental_by_horizon, sleeve_incremental_by_horizon]
        )
        all_components = all_components[np.isfinite(all_components)]
        minimum_component = (
            float(np.min(all_components)) if len(all_components) else -np.inf
        )
        downside_component = min(minimum_component, 0.0)
        minimum_weight = float(
            model_cfg.get("validation_horizon_minimum_component_weight", 0.25)
        )
        positive_fraction = (
            float(np.mean(all_components > 0.0))
            if len(all_components)
            else 0.0
        )
        positive_fraction_reward = float(
            model_cfg.get("validation_positive_horizon_reward", 0.0)
        )
        score = float(
            (1.0 - minimum_weight) * weighted_increment
            + minimum_weight * downside_component
            + positive_fraction_reward * (positive_fraction - 0.5)
        )
    elif selection_mode == "mean_view_rank":
        asset_weight = float(
            model_cfg.get("validation_asset_increment_weight", 0.50)
        )
        sleeve_weight = float(
            model_cfg.get("validation_sleeve_increment_weight", 0.50)
        )
        task_weight_total = max(asset_weight + sleeve_weight, 1e-12)
        asset_weight /= task_weight_total
        sleeve_weight /= task_weight_total
        weighted_rank = float(
            asset_weight * asset_weighted_rank
            + sleeve_weight * sleeve_weighted_rank
        )
        active_components = np.concatenate(
            [
                asset_active_rank_components,
                sleeve_incremental_by_horizon[
                    (sleeve_horizon_weights > 0.0)
                    & np.isfinite(sleeve_incremental_by_horizon)
                ],
            ]
        )
        worst_component = (
            float(np.min(active_components))
            if len(active_components)
            else -np.inf
        )
        downside_weight = float(
            model_cfg.get("validation_mean_rank_downside_weight", 0.10)
        )
        positive_fraction = (
            float(np.mean(active_components > 0.0))
            if len(active_components)
            else 0.0
        )
        positive_reward = float(
            model_cfg.get("validation_mean_rank_positive_reward", 0.0)
        )
        score = float(
            (1.0 - downside_weight) * weighted_rank
            + downside_weight * min(worst_component, 0.0)
            + positive_reward * (positive_fraction - 0.5)
        )
    elif selection_mode == "decision_aligned_utility":
        asset_weight = float(
            model_cfg.get("validation_asset_increment_weight", 0.45)
        )
        sleeve_weight = float(
            model_cfg.get("validation_sleeve_increment_weight", 0.55)
        )
        task_weight_total = max(asset_weight + sleeve_weight, 1e-12)
        asset_weight /= task_weight_total
        sleeve_weight /= task_weight_total

        asset_weighted_utility = asset_weighted_decision_utility
        sleeve_weighted_utility = _weighted_finite_mean(
            sleeve_decision_utility, sleeve_horizon_weights
        )
        weighted_utility = float(
            asset_weight * asset_weighted_utility
            + sleeve_weight * sleeve_weighted_utility
        )
        components = np.concatenate(
            [
                asset_active_decision_components,
                sleeve_decision_utility[
                    (sleeve_horizon_weights > 0.0)
                    & np.isfinite(sleeve_decision_utility)
                ],
            ]
        )
        worst_component = (
            float(np.min(components)) if len(components) else -np.inf
        )
        minimum_weight = float(
            model_cfg.get(
                "validation_decision_minimum_component_weight", 0.25
            )
        )
        positive_components = (
            float(np.mean(components > 0.0)) if len(components) else 0.0
        )
        positive_reward = float(
            model_cfg.get("validation_decision_positive_reward", 0.0)
        )
        score = float(
            (1.0 - minimum_weight) * weighted_utility
            + minimum_weight * min(worst_component, 0.0)
            + positive_reward * (positive_components - 0.5)
        )
    else:
        raise ValueError(
            f"Unknown model.validation_selection_metric={selection_mode}"
        )
    metrics = {
        "validation_asset_residual_rank_ic": asset_model,
        "validation_asset_combined_rank_ic": asset_combined,
        "validation_asset_baseline_rank_ic": asset_baseline,
        "validation_asset_incremental_rank_ic": asset_incremental,
        "validation_sleeve_residual_rank_ic": sleeve_model,
        "validation_sleeve_combined_rank_ic": sleeve_combined,
        "validation_sleeve_baseline_rank_ic": sleeve_baseline,
        "validation_sleeve_incremental_rank_ic": sleeve_incremental,
        "validation_asset_weighted_incremental_rank_ic": asset_weighted_rank,
        "validation_sleeve_weighted_incremental_rank_ic": sleeve_weighted_rank,
        "validation_correction_alpha": validation_alpha,
        "validation_asset_decision_utility": float(
            asset_weighted_utility
            if selection_mode == "decision_aligned_utility"
            else np.nanmean(asset_decision_utility)
        ),
        "validation_sleeve_decision_utility": float(
            sleeve_weighted_utility
            if selection_mode == "decision_aligned_utility"
            else np.nanmean(sleeve_decision_utility)
        ),
        "validation_asset_decision_positive_fraction": float(
            np.nanmean(asset_decision_positive)
        ),
        "validation_sleeve_decision_positive_fraction": float(
            np.nanmean(sleeve_decision_positive)
        ),
    }
    configured_horizons = list(
        (cfg or {}).get("data", {}).get(
            "horizons_days", range(len(asset_incremental_by_horizon))
        )
    )
    for horizon_position, horizon_label in enumerate(configured_horizons):
        suffix = f"h{int(horizon_label)}"
        metrics[f"validation_asset_incremental_rank_ic_{suffix}"] = float(
            asset_incremental_by_horizon[horizon_position]
        )
        metrics[f"validation_sleeve_incremental_rank_ic_{suffix}"] = float(
            sleeve_incremental_by_horizon[horizon_position]
        )
        metrics[f"validation_asset_combined_rank_ic_{suffix}"] = float(
            asset_combined_by_horizon[horizon_position]
        )
        metrics[f"validation_sleeve_combined_rank_ic_{suffix}"] = float(
            sleeve_combined_by_horizon[horizon_position]
        )
        metrics[f"validation_asset_decision_utility_{suffix}"] = float(
            asset_decision_utility[horizon_position]
        )
        metrics[f"validation_sleeve_decision_utility_{suffix}"] = float(
            sleeve_decision_utility[horizon_position]
        )
        metrics[
            f"validation_asset_decision_positive_fraction_{suffix}"
        ] = float(asset_decision_positive[horizon_position])
        metrics[
            f"validation_sleeve_decision_positive_fraction_{suffix}"
        ] = float(sleeve_decision_positive[horizon_position])
        for module_name, values in asset_module_decision_utility.items():
            metrics[
                f"validation_asset_decision_utility_{module_name}_{suffix}"
            ] = float(values[horizon_position])
            metrics[
                "validation_asset_decision_positive_fraction_"
                f"{module_name}_{suffix}"
            ] = float(
                asset_module_decision_positive[module_name][horizon_position]
            )
        for module_name, (
            module_combined,
            module_baseline,
            module_incremental,
        ) in asset_module_rank_components.items():
            metrics[
                f"validation_asset_combined_rank_ic_{module_name}_{suffix}"
            ] = float(module_combined[horizon_position])
            metrics[
                f"validation_asset_baseline_rank_ic_{module_name}_{suffix}"
            ] = float(module_baseline[horizon_position])
            metrics[
                f"validation_asset_incremental_rank_ic_{module_name}_{suffix}"
            ] = float(module_incremental[horizon_position])

    for module_name, values in asset_module_decision_utility.items():
        module_weights = (
            asset_module_horizon_weights[module_name]
            if asset_module_horizon_weights is not None
            else asset_horizon_weights
        )
        metrics[f"validation_asset_decision_utility_{module_name}"] = float(
            _weighted_finite_mean(values, module_weights)
        )
    for module_name, (
        _,
        _,
        module_incremental,
    ) in asset_module_rank_components.items():
        module_weights = (
            asset_module_horizon_weights[module_name]
            if asset_module_horizon_weights is not None
            else asset_horizon_weights
        )
        metrics[
            f"validation_asset_weighted_incremental_rank_ic_{module_name}"
        ] = float(
            _weighted_finite_mean(
                module_incremental,
                module_weights,
            )
        )
    return score, metrics


def validation_rank_score(
    bundle: dict[str, np.ndarray],
    module_index: np.ndarray,
    arrays=None,
    cfg: dict[str, Any] | None = None,
) -> tuple[float, dict[str, float]]:
    overall_score, metrics = _rank_components(bundle, module_index, cfg)
    if arrays is None or cfg is None:
        return overall_score, metrics
    indices = bundle["index"].astype(np.int64)
    minimum = int(cfg["model"].get("validation_subgroup_min_samples", 40))

    def balanced_score(groups: np.ndarray) -> float:
        scores: list[float] = []
        for value in np.unique(groups):
            selected = np.flatnonzero(groups == value)
            if len(selected) < minimum:
                continue
            subset = {key: item[selected] for key, item in bundle.items()}
            score, _ = _rank_components(subset, module_index, cfg)
            if np.isfinite(score):
                scores.append(float(score))
        return float(np.mean(scores)) if scores else np.nan

    regime_score = balanced_score(arrays.regime_index[indices])
    years = pd.to_datetime(arrays.dates[indices]).year.to_numpy(dtype=np.int64)
    year_score = balanced_score(years)
    regime_weight = float(cfg["model"].get("validation_regime_balance_weight", 0.0))
    year_weight = float(cfg["model"].get("validation_year_balance_weight", 0.0))
    if regime_weight < 0.0 or year_weight < 0.0 or regime_weight + year_weight > 1.0:
        raise ValueError("validation regime/year balance weights must be non-negative and sum to <= 1")
    weighted_scores = [(1.0 - regime_weight - year_weight, overall_score)]
    weighted_scores.append((regime_weight, regime_score if np.isfinite(regime_score) else overall_score))
    weighted_scores.append((year_weight, year_score if np.isfinite(year_score) else overall_score))
    score = float(sum(weight * value for weight, value in weighted_scores))
    metrics.update(
        {
            "validation_overall_rank_score": overall_score,
            "validation_regime_balanced_rank_score": regime_score,
            "validation_year_balanced_rank_score": year_score,
        }
    )
    return score, metrics


def build_train_sampler(arrays, train_indices: np.ndarray, cfg: dict[str, Any], seed: int):
    if not bool(cfg["model"].get("regime_balanced_sampling", False)):
        return None
    regimes = arrays.regime_index[train_indices].astype(np.int64)
    unique, counts = np.unique(regimes, return_counts=True)
    frequencies = dict(zip(unique.tolist(), counts.tolist(), strict=True))
    power = float(cfg["model"].get("regime_sampling_power", 0.5))
    cap = float(cfg["model"].get("regime_sampling_max_weight_ratio", 4.0))
    weights = np.array([frequencies[int(value)] ** (-power) for value in regimes], dtype=np.float64)
    positive = weights[weights > 0.0]
    if len(positive):
        weights = np.minimum(weights, float(np.median(positive)) * cap)
    weights = weights / max(float(weights.mean()), 1e-12)
    generator = torch.Generator().manual_seed(int(seed))
    return WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(train_indices),
        replacement=True,
        generator=generator,
    )


def predict(
    model,
    loader,
    device,
    cfg: dict[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    model.eval()
    output: dict[str, list[np.ndarray]] = {
        "index": [],
        "asset_residual": [],
        "asset_logvar": [],
        "sleeve_residual": [],
        "sleeve_logvar": [],
        "asset_target": [],
        "asset_baseline": [],
        "asset_mask": [],
        "sleeve_target": [],
        "sleeve_baseline": [],
        "sleeve_mask": [],
    }
    with torch.no_grad():
        for raw in loader:
            batch = to_device(raw, device)
            model_output = model_forward(model, batch)
            asset_mean, asset_logvar, sleeve_mean, sleeve_logvar = prediction_heads(
                model_output, model.cfg
            )
            values = {
                "index": batch["index"],
                "asset_residual": asset_mean,
                "asset_logvar": asset_logvar,
                "sleeve_residual": sleeve_mean,
                "sleeve_logvar": sleeve_logvar,
                "asset_target": batch["asset_target"],
                "asset_baseline": batch["asset_baseline"],
                "asset_mask": batch["asset_mask"],
                "sleeve_target": batch["sleeve_target"],
                "sleeve_baseline": batch["sleeve_baseline"],
                "sleeve_mask": batch["sleeve_mask"],
            }
            if (
                len(model_output) == 6
                and str(model.cfg.get("prediction_head_mode", "unconstrained"))
                == "adaptive_expert_mixture"
            ):
                values["asset_expert_gate"] = model_output[4]
                values["sleeve_expert_gate"] = model_output[5]
                output.setdefault("asset_expert_gate", [])
                output.setdefault("sleeve_expert_gate", [])
            for key, value in values.items():
                output[key].append(value.detach().cpu().numpy())
    bundle = {key: np.concatenate(values, axis=0) for key, values in output.items()}
    return _postprocess_prediction_bundle(
        bundle, model.module_index.detach().cpu().numpy(), cfg or model.cfg
    )


def mean_rank_ic(pred: np.ndarray, target: np.ndarray, mask: np.ndarray, groups: np.ndarray | None) -> float:
    scores = horizon_rank_ics(pred, target, mask, groups)
    finite_scores = np.asarray(scores, dtype=float)
    finite_scores = finite_scores[np.isfinite(finite_scores)]
    return float(finite_scores.mean()) if finite_scores.size else float("nan")


def horizon_rank_ics(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    groups: np.ndarray | None,
) -> np.ndarray:
    return np.asarray(
        [
            rank_ic(
                pred[:, horizon],
                target[:, horizon],
                mask[:, horizon],
                groups,
            )
            for horizon in range(pred.shape[1])
        ],
        dtype=float,
    )


def daily_rank_ic_series(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    groups: np.ndarray | None,
) -> np.ndarray:
    """Return one equal-subgroup cross-sectional RankIC observation per date."""
    values = np.full(pred.shape[0], np.nan, dtype=float)
    unique_groups = [None] if groups is None else np.unique(groups)
    for row in range(pred.shape[0]):
        row_values: list[float] = []
        for group in unique_groups:
            valid = mask[row].astype(bool)
            if group is not None:
                valid &= groups == group
            if valid.sum() < 2:
                continue
            prediction_rank = (
                pd.Series(pred[row, valid])
                .rank(method="average")
                .to_numpy(dtype=float)
            )
            target_rank = (
                pd.Series(target[row, valid])
                .rank(method="average")
                .to_numpy(dtype=float)
            )
            target_has_cross_section = np.std(target_rank) > 1e-12
            if (
                np.std(prediction_rank) > 1e-12
                and target_has_cross_section
            ):
                row_values.append(
                    float(np.corrcoef(prediction_rank, target_rank)[0, 1])
                )
            elif target_has_cross_section:
                # A constant prediction is the explicit no-view benchmark.
                # Treat it as zero RankIC so paired bootstrap comparisons keep
                # the same dates instead of silently dropping the benchmark.
                row_values.append(0.0)
        if row_values:
            values[row] = float(np.mean(row_values))
    return values


def paired_block_bootstrap_mean(
    delta: np.ndarray,
    block_days: int,
    draws: int,
    confidence: float,
    seed: int,
) -> dict[str, float]:
    values = np.asarray(delta, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    block_days = max(int(block_days), 2)
    draws = max(int(draws), 0)
    if n < max(20, 2 * block_days) or draws <= 0:
        return {
            "bootstrap_edge_mean": float(np.mean(values)) if n else np.nan,
            "bootstrap_edge_ci_low": -np.inf,
            "bootstrap_edge_ci_high": np.inf,
            "bootstrap_positive_probability": 0.5,
            "bootstrap_samples": float(n),
            "bootstrap_block_days": float(block_days),
        }
    confidence = float(np.clip(confidence, 0.50, 0.999))
    tail = 0.5 * (1.0 - confidence)
    blocks_needed = int(np.ceil(n / block_days))
    offsets = np.arange(block_days, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    estimates = np.empty(draws, dtype=float)
    chunk_size = 256
    for start in range(0, draws, chunk_size):
        stop = min(start + chunk_size, draws)
        starts = rng.integers(
            0,
            n,
            size=(stop - start, blocks_needed),
        )
        indices = (
            starts[:, :, None] + offsets[None, None, :]
        ) % n
        sampled = values[indices.reshape(stop - start, -1)[:, :n]]
        estimates[start:stop] = sampled.mean(axis=1)
    return {
        "bootstrap_edge_mean": float(values.mean()),
        "bootstrap_edge_ci_low": float(np.quantile(estimates, tail)),
        "bootstrap_edge_ci_high": float(np.quantile(estimates, 1.0 - tail)),
        "bootstrap_positive_probability": float(np.mean(estimates > 0.0)),
        "bootstrap_samples": float(n),
        "bootstrap_block_days": float(block_days),
    }


def choose_blend_policies(
    bundle: dict[str, np.ndarray],
    baseline_grid: list[float],
    residual_grid: list[float],
    module_index: np.ndarray,
    arrays=None,
    cfg: dict[str, Any] | None = None,
    use_task_specific_grids: bool = True,
    require_incremental_confidence: bool = False,
    tasks: tuple[str, ...] = ("asset", "sleeve"),
    allow_scoped_asset: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Select task/horizon blends on the pre-cutoff validation window only."""
    policies: dict[str, Any] = {"asset": [], "sleeve": []}
    rows: list[dict[str, Any]] = []
    definitions = {
        "asset": ("asset_baseline", "asset_residual", "asset_target", "asset_mask", module_index),
        "sleeve": ("sleeve_baseline", "sleeve_residual", "sleeve_target", "sleeve_mask", None),
    }
    model_cfg = (cfg or {}).get("model", {})
    for task in tasks:
        if task not in definitions:
            raise ValueError(f"Unknown blend task={task}")
        baseline_key, residual_key, target_key, mask_key, groups = definitions[task]
        selection_metric = str(
            model_cfg.get("blend_selection_metric", "rank_ic")
        )
        if selection_metric not in {"rank_ic", "decision_aligned_utility"}:
            raise ValueError(
                f"Unknown model.blend_selection_metric={selection_metric}"
            )
        task_residual_grid = residual_grid
        if cfg is not None and use_task_specific_grids:
            task_residual_grid = [
                float(value)
                for value in cfg["model"].get(f"{task}_residual_alpha_grid", residual_grid)
            ]
        for horizon in range(bundle[baseline_key].shape[1]):
            configured_horizons = list(
                (cfg or {}).get("data", {}).get(
                    "horizons_days",
                    range(bundle[baseline_key].shape[1]),
                )
            )
            horizon_days = int(configured_horizons[horizon])
            enabled_horizons = model_cfg.get(
                f"{task}_enabled_horizons_days"
            )
            horizon_residual_grid = task_residual_grid
            if (
                enabled_horizons is not None
                and horizon_days
                not in {int(value) for value in enabled_horizons}
            ):
                horizon_residual_grid = [0.0]
            decision_temperature = _validation_horizon_value(
                model_cfg.get(f"{task}_decision_temperature", 0.10),
                horizon,
                bundle[baseline_key].shape[1],
                0.10,
            )
            decision_transaction_cost = _validation_horizon_value(
                model_cfg.get("decision_transaction_cost_bps", 5.0),
                horizon,
                bundle[baseline_key].shape[1],
                5.0,
            )
            candidates: list[dict[str, Any]] = []
            for baseline_position, baseline_weight in enumerate(baseline_grid):
                for residual_position, residual_alpha in enumerate(
                    horizon_residual_grid
                ):
                    score = _blend_prediction(
                        bundle[baseline_key][:, horizon : horizon + 1],
                        bundle[residual_key][:, horizon : horizon + 1],
                        float(baseline_weight),
                        float(residual_alpha),
                        cfg,
                    )
                    subgroup_scores: dict[str, dict[str, float]] = {
                        "regime": {},
                        "year": {},
                    }
                    if baseline_weight == 0.0 and residual_alpha == 0.0:
                        full_ic = 0.0
                        robust_ic = 0.0
                        subgroup_std = 0.0
                        positive_subgroup_fraction = 0.0
                    else:
                        full_ic = mean_rank_ic(
                            score,
                            bundle[target_key][:, horizon : horizon + 1],
                            bundle[mask_key][:, horizon : horizon + 1],
                            groups,
                        )
                        robust_ic = full_ic
                        subgroup_std = 0.0
                        positive_subgroup_fraction = float(full_ic > 0.0)
                        if arrays is not None and cfg is not None:
                            indices = bundle["index"].astype(np.int64)
                            minimum = int(cfg["model"].get("validation_subgroup_min_samples", 40))
                            validation_groups = {
                                "regime": arrays.regime_index[indices],
                                "year": pd.to_datetime(arrays.dates[indices]).year.to_numpy(dtype=np.int64),
                            }
                            for group_name, values in validation_groups.items():
                                for value in np.unique(values):
                                    selected = np.flatnonzero(values == value)
                                    if len(selected) < minimum:
                                        continue
                                    subgroup_ic = mean_rank_ic(
                                        score[selected],
                                        bundle[target_key][selected, horizon : horizon + 1],
                                        bundle[mask_key][selected, horizon : horizon + 1],
                                        groups,
                                    )
                                    if np.isfinite(subgroup_ic):
                                        subgroup_scores[group_name][str(value)] = float(
                                            subgroup_ic
                                        )
                            regime_ic = (
                                float(np.mean(list(subgroup_scores["regime"].values())))
                                if subgroup_scores["regime"]
                                else full_ic
                            )
                            year_ic = (
                                float(np.mean(list(subgroup_scores["year"].values())))
                                if subgroup_scores["year"]
                                else full_ic
                            )
                            regime_weight = float(cfg["model"].get("blend_validation_regime_weight", 0.0))
                            year_weight = float(cfg["model"].get("blend_validation_year_weight", 0.0))
                            overall_weight = 1.0 - regime_weight - year_weight
                            if overall_weight < 0.0:
                                raise ValueError("blend validation group weights must sum to <= 1")
                            all_subgroups = [
                                *subgroup_scores["regime"].values(),
                                *subgroup_scores["year"].values(),
                            ]
                            subgroup_std = float(np.std(all_subgroups)) if all_subgroups else 0.0
                            robust_ic = (
                                overall_weight * full_ic
                                + regime_weight * regime_ic
                                + year_weight * year_ic
                                - float(cfg["model"].get("blend_stability_penalty", 0.0)) * subgroup_std
                            )
                    decision_utility_series = _decision_net_utility_series(
                        score[:, 0],
                        bundle[baseline_key][:, horizon],
                        bundle[target_key][:, horizon],
                        bundle[mask_key][:, horizon],
                        groups,
                        decision_temperature,
                        decision_transaction_cost,
                        horizon_days,
                    )
                    finite_utility = np.isfinite(decision_utility_series)
                    full_decision_utility = (
                        float(np.mean(decision_utility_series[finite_utility]))
                        if finite_utility.any()
                        else np.nan
                    )
                    utility_subgroup_scores: dict[str, dict[str, float]] = {
                        "regime": {},
                        "year": {},
                    }
                    robust_decision_utility = full_decision_utility
                    utility_subgroup_std = 0.0
                    if arrays is not None and cfg is not None:
                        indices = bundle["index"].astype(np.int64)
                        minimum = int(
                            model_cfg.get(
                                "validation_subgroup_min_samples", 40
                            )
                        )
                        validation_groups = {
                            "regime": arrays.regime_index[indices],
                            "year": pd.to_datetime(
                                arrays.dates[indices]
                            ).year.to_numpy(dtype=np.int64),
                        }
                        for group_name, values in validation_groups.items():
                            for value in np.unique(values):
                                selected = np.flatnonzero(values == value)
                                selected = selected[
                                    np.isfinite(
                                        decision_utility_series[selected]
                                    )
                                ]
                                if len(selected) < minimum:
                                    continue
                                utility_subgroup_scores[group_name][
                                    str(value)
                                ] = float(
                                    np.mean(
                                        decision_utility_series[selected]
                                    )
                                )
                        regime_utility = (
                            float(
                                np.mean(
                                    list(
                                        utility_subgroup_scores[
                                            "regime"
                                        ].values()
                                    )
                                )
                            )
                            if utility_subgroup_scores["regime"]
                            else full_decision_utility
                        )
                        year_utility = (
                            float(
                                np.mean(
                                    list(
                                        utility_subgroup_scores[
                                            "year"
                                        ].values()
                                    )
                                )
                            )
                            if utility_subgroup_scores["year"]
                            else full_decision_utility
                        )
                        regime_weight = float(
                            model_cfg.get(
                                "blend_validation_regime_weight", 0.0
                            )
                        )
                        year_weight = float(
                            model_cfg.get(
                                "blend_validation_year_weight", 0.0
                            )
                        )
                        overall_weight = 1.0 - regime_weight - year_weight
                        if overall_weight < 0.0:
                            raise ValueError(
                                "blend validation group weights must sum to <= 1"
                            )
                        utility_subgroups = [
                            *utility_subgroup_scores["regime"].values(),
                            *utility_subgroup_scores["year"].values(),
                        ]
                        utility_subgroup_std = (
                            float(np.std(utility_subgroups))
                            if utility_subgroups
                            else 0.0
                        )
                        robust_decision_utility = (
                            overall_weight * full_decision_utility
                            + regime_weight * regime_utility
                            + year_weight * year_utility
                            - float(
                                model_cfg.get(
                                    "blend_decision_stability_penalty",
                                    model_cfg.get(
                                        "blend_stability_penalty", 0.0
                                    ),
                                )
                            )
                            * utility_subgroup_std
                        )
                    if selection_metric == "decision_aligned_utility":
                        selection_score = robust_decision_utility
                        full_selection_score = full_decision_utility
                        selection_subgroup_std = utility_subgroup_std
                        selection_subgroup_scores = utility_subgroup_scores
                        daily_selection_score = decision_utility_series
                    else:
                        selection_score = robust_ic
                        full_selection_score = full_ic
                        selection_subgroup_std = subgroup_std
                        selection_subgroup_scores = subgroup_scores
                        daily_selection_score = (
                            daily_rank_ic_series(
                                score[:, 0],
                                bundle[target_key][:, horizon],
                                bundle[mask_key][:, horizon],
                                groups,
                            )
                            if require_incremental_confidence
                            else np.empty(0, dtype=float)
                        )
                    # Prefer the simpler/shrunk candidate when validation IC ties.
                    complexity = abs(float(baseline_weight)) + abs(float(residual_alpha))
                    candidates.append(
                        {
                            "task": task,
                            "horizon_index": float(horizon),
                            "baseline_weight": float(baseline_weight),
                            "residual_alpha": float(residual_alpha),
                            "rank_ic": robust_ic,
                            "full_rank_ic": full_ic,
                            "decision_utility": robust_decision_utility,
                            "full_decision_utility": full_decision_utility,
                            "selection_metric": selection_metric,
                            "selection_score": selection_score,
                            "full_selection_score": full_selection_score,
                            "subgroup_rank_ic_std": subgroup_std,
                            "subgroup_decision_utility_std": utility_subgroup_std,
                            "positive_subgroup_fraction": positive_subgroup_fraction,
                            "subgroup_edge_std": 0.0,
                            "worst_year_edge": 0.0,
                            "worst_regime_edge": 0.0,
                            "incremental_robust_rank_ic": 0.0,
                            "incremental_robust_score": 0.0,
                            "bootstrap_edge_mean": 0.0,
                            "bootstrap_edge_ci_low": -np.inf,
                            "bootstrap_edge_ci_high": np.inf,
                            "bootstrap_positive_probability": 0.5,
                            "bootstrap_samples": 0.0,
                            "bootstrap_block_days": 0.0,
                            "complexity": complexity,
                            "_subgroup_scores": subgroup_scores,
                            "_selection_subgroup_scores": selection_subgroup_scores,
                            "_daily_selection_score": daily_selection_score,
                            "_candidate_seed_offset": (
                                baseline_position * 10_000 + residual_position
                            ),
                        }
                    )
            eligible_candidates = candidates
            use_incremental_selection = False
            if cfg is not None:
                fallback_baseline_weight = float(
                    model_cfg.get(
                        "deployment_fallback_baseline_weight",
                        (
                            0.0
                            if str(
                                cfg.get("data", {}).get(
                                    "forecast_baseline_mode",
                                    "causal_anchor",
                                )
                            )
                            == "zero_active_view"
                            else 1.0
                        ),
                    )
                )
                zero_residual = [
                    row
                    for row in candidates
                    if abs(float(row["residual_alpha"])) <= 1e-12
                    and abs(
                        float(row["baseline_weight"])
                        - fallback_baseline_weight
                    )
                    <= 1e-12
                ]
                if not zero_residual:
                    raise ValueError(
                        "Blend grid is missing the configured deployment "
                        "fallback candidate: residual_alpha=0 and "
                        f"baseline_weight={fallback_baseline_weight:g}"
                    )
                if zero_residual:
                    use_incremental_selection = True
                    anchor = zero_residual[0]
                    anchor_groups = anchor.get(
                        "_selection_subgroup_scores", {}
                    )
                    edge_penalty = float(
                        cfg["model"].get("blend_edge_stability_penalty", 0.0)
                    )
                    for row in candidates:
                        subgroup_edges: list[float] = []
                        edges_by_group: dict[str, list[float]] = {
                            "regime": [],
                            "year": [],
                        }
                        candidate_groups = row.get(
                            "_selection_subgroup_scores", {}
                        )
                        for group_name in ["regime", "year"]:
                            anchor_values = anchor_groups.get(group_name, {})
                            candidate_values = candidate_groups.get(group_name, {})
                            for key in sorted(
                                set(anchor_values).intersection(candidate_values)
                            ):
                                edge = float(
                                    candidate_values[key] - anchor_values[key]
                                )
                                subgroup_edges.append(edge)
                                edges_by_group[group_name].append(edge)
                        full_edge = float(
                            row["full_selection_score"]
                            - anchor["full_selection_score"]
                        )
                        robust_edge = float(
                            row["selection_score"]
                            - anchor["selection_score"]
                        )
                        edge_std = (
                            float(np.std(subgroup_edges))
                            if subgroup_edges
                            else 0.0
                        )
                        row["positive_subgroup_fraction"] = (
                            float(np.mean(np.asarray(subgroup_edges) > 0.0))
                            if subgroup_edges
                            else float(full_edge > 0.0)
                        )
                        row["subgroup_edge_std"] = edge_std
                        row["worst_year_edge"] = (
                            float(np.min(edges_by_group["year"]))
                            if edges_by_group["year"]
                            else full_edge
                        )
                        row["worst_regime_edge"] = (
                            float(np.min(edges_by_group["regime"]))
                            if edges_by_group["regime"]
                            else full_edge
                        )
                        row["incremental_robust_rank_ic"] = (
                            float(row["rank_ic"] - anchor["rank_ic"])
                        )
                        row["incremental_robust_score"] = (
                            robust_edge - edge_penalty * edge_std
                        )
                        if require_incremental_confidence:
                            candidate_daily = np.asarray(
                                row.get("_daily_selection_score", []),
                                dtype=float,
                            )
                            anchor_daily = np.asarray(
                                anchor.get("_daily_selection_score", []),
                                dtype=float,
                            )
                            valid_daily = np.isfinite(candidate_daily) & np.isfinite(
                                anchor_daily
                            )
                            bootstrap = paired_block_bootstrap_mean(
                                candidate_daily[valid_daily]
                                - anchor_daily[valid_daily],
                                block_days=max(
                                    int(
                                        round(
                                            horizon_days
                                            * float(
                                                cfg["model"].get(
                                                    "blend_bootstrap_block_multiplier",
                                                    1.0,
                                                )
                                            )
                                        )
                                    ),
                                    5,
                                ),
                                draws=int(
                                    cfg["model"].get(
                                        "blend_bootstrap_draws", 1000
                                    )
                                ),
                                confidence=float(
                                    cfg["model"].get(
                                        "blend_bootstrap_confidence", 0.90
                                    )
                                ),
                                seed=(
                                    int(
                                        cfg["model"].get(
                                            "blend_bootstrap_seed", 20260725
                                        )
                                    )
                                    + (0 if task == "asset" else 1_000_000)
                                    + horizon * 10_000
                                    + int(row["_candidate_seed_offset"])
                                ),
                            )
                            row.update(bootstrap)
                if selection_metric == "decision_aligned_utility":
                    minimum_robust_edge = float(
                        model_cfg.get(
                            "blend_min_decision_utility_improvement", 0.0
                        )
                    )
                    minimum_full_edge = float(
                        model_cfg.get(
                            "blend_min_full_decision_utility_improvement",
                            -np.inf,
                        )
                    )
                    minimum_absolute_score = float(
                        model_cfg.get(
                            "blend_min_absolute_decision_utility", -np.inf
                        )
                    )
                else:
                    minimum_robust_edge = float(
                        model_cfg.get(
                            "blend_min_rank_ic_improvement", 0.0
                        )
                    )
                    minimum_full_edge = float(
                        model_cfg.get(
                            "blend_min_full_rank_ic_improvement", -np.inf
                        )
                    )
                    minimum_absolute_score = float(
                        model_cfg.get(
                            "blend_min_absolute_rank_ic", -np.inf
                        )
                    )
                minimum_positive_fraction = float(
                    cfg["model"].get(
                        "blend_min_positive_subgroup_fraction", 0.0
                    )
                )
                minimum_worst_year_edge = float(
                    cfg["model"].get("blend_min_worst_year_edge", -np.inf)
                )
                minimum_worst_regime_edge = float(
                    cfg["model"].get("blend_min_worst_regime_edge", -np.inf)
                )
                minimum_bootstrap_probability = float(
                    cfg["model"].get(
                        "blend_min_bootstrap_positive_probability", 0.0
                    )
                )
                minimum_bootstrap_ci_low = float(
                    cfg["model"].get("blend_min_bootstrap_edge_ci_low", -np.inf)
                )
                if zero_residual and (
                    minimum_robust_edge > 0.0 or np.isfinite(minimum_full_edge)
                ):
                    anchor = zero_residual[0]
                    promoted = [
                        row
                        for row in candidates
                        if float(row["residual_alpha"]) > 0.0
                        and float(row["incremental_robust_score"])
                        >= minimum_robust_edge
                        and float(row["full_selection_score"])
                        >= float(anchor["full_selection_score"])
                        + minimum_full_edge
                        and float(row["selection_score"])
                        >= minimum_absolute_score
                        and float(row["positive_subgroup_fraction"])
                        >= minimum_positive_fraction
                        and float(row["worst_year_edge"])
                        >= minimum_worst_year_edge
                        and float(row["worst_regime_edge"])
                        >= minimum_worst_regime_edge
                        and float(row["bootstrap_positive_probability"])
                        >= minimum_bootstrap_probability
                        and float(row["bootstrap_edge_ci_low"])
                        >= minimum_bootstrap_ci_low
                    ]
                    eligible_candidates = promoted if promoted else [anchor]
            best = max(
                eligible_candidates,
                key=lambda row: (
                    (
                        float(row["incremental_robust_score"])
                        if use_incremental_selection
                        and np.isfinite(row["incremental_robust_score"])
                        else (
                            float(row["selection_score"])
                            if np.isfinite(row["selection_score"])
                            else -np.inf
                        )
                    ),
                    (
                        0.0
                        if use_incremental_selection
                        else float(row["selection_score"])
                    ),
                    (
                        -float(row["complexity"])
                        if np.isfinite(row["complexity"])
                        else -np.inf
                    ),
                ),
            )
            public_best = {
                key: value for key, value in best.items() if not key.startswith("_")
            }
            policies[task].append(public_best)
            rows.extend(
                {
                    key: value
                    for key, value in candidate.items()
                    if not key.startswith("_")
                }
                for candidate in candidates
            )
    for row in rows:
        row.setdefault("scope", "global")
        row.setdefault("module", "")

    if (
        allow_scoped_asset
        and "asset" in tasks
        and str(model_cfg.get("asset_blend_scope", "global")) == "module"
    ):
        policies["asset_by_module"] = {}
        for module_value in sorted(np.unique(module_index).tolist()):
            columns = np.flatnonzero(module_index == module_value)
            if len(columns) < 2:
                continue
            module_bundle = dict(bundle)
            for key in (
                "asset_baseline",
                "asset_residual",
                "asset_target",
                "asset_mask",
            ):
                module_bundle[key] = bundle[key][..., columns]
            module_name = RISK_SLEEVES[int(module_value)]
            module_policies, module_rows = choose_blend_policies(
                module_bundle,
                baseline_grid,
                residual_grid,
                np.zeros(len(columns), dtype=np.int64),
                arrays=arrays,
                cfg=cfg,
                use_task_specific_grids=use_task_specific_grids,
                require_incremental_confidence=require_incremental_confidence,
                tasks=("asset",),
                allow_scoped_asset=False,
            )
            policies["asset_by_module"][module_name] = module_policies["asset"]
            for row in module_rows:
                row["scope"] = "module"
                row["module"] = module_name
            rows.extend(module_rows)
    return policies, rows


def train_seed(
    seed: int,
    year: int,
    arrays,
    metadata: dict[str, Any],
    cfg: dict[str, Any],
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    train_label_cutoff_ns: int,
    cutoff_ns: int,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, np.ndarray]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    cfg = with_fold_training_loss_scales(cfg, arrays, train_indices)
    model = make_model(arrays, metadata, cfg).to(device)
    pretraining_logs, pretraining_summary = pretrain_encoder(
        model,
        arrays,
        metadata,
        cfg,
        train_indices,
        device,
        seed,
        year,
    )
    train_set = WindowDataset(
        arrays, train_indices, int(metadata["sequence_length_days"]), train_label_cutoff_ns
    )
    val_set = WindowDataset(arrays, val_indices, int(metadata["sequence_length_days"]), cutoff_ns)
    train_sampler = build_train_sampler(arrays, train_indices, cfg, seed)
    train_loader = DataLoader(
        train_set,
        batch_size=int(cfg["model"]["batch_size"]),
        shuffle=train_sampler is None,
        sampler=train_sampler,
    )
    val_loader = DataLoader(val_set, batch_size=int(cfg["model"]["batch_size"]), shuffle=False)
    freeze_encoder_epochs = int(
        cfg["model"].get("self_supervised_pretraining", {}).get(
            "freeze_encoder_epochs", 0
        )
    )
    _set_supervised_encoder_frozen(model, freeze_encoder_epochs > 0)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["model"]["learning_rate"]),
        weight_decay=float(cfg["model"]["weight_decay"]),
    )
    maximum_epochs = int(cfg["model"]["max_epochs"])
    warmup_epochs = int(cfg["model"].get("warmup_epochs", 0))
    minimum_lr_ratio = float(cfg["model"].get("minimum_learning_rate_ratio", 0.10))

    decay_epochs = int(cfg["model"].get("scheduler_decay_epochs", maximum_epochs))

    def lr_multiplier(step: int) -> float:
        epoch = step + 1
        if warmup_epochs > 0 and epoch <= warmup_epochs:
            return max(epoch / warmup_epochs, minimum_lr_ratio)
        progress = (epoch - warmup_epochs) / max(decay_epochs - warmup_epochs, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
        return minimum_lr_ratio + (1.0 - minimum_lr_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_multiplier)
    best_loss = np.inf
    best_objective = np.inf
    best_score = -np.inf
    best_state_score = np.nan
    best_state = None
    best_epoch = 0
    stale = 0
    selection_mode = str(cfg["model"].get("checkpoint_selection_mode", "loss_rank_composite"))
    if selection_mode not in {"loss_rank_composite", "rank_constrained"}:
        raise ValueError(f"Unknown checkpoint_selection_mode={selection_mode}")
    top_k = max(int(cfg["model"].get("checkpoint_average_top_k", 1)), 1)
    candidate_pool_size = max(
        int(cfg["model"].get("checkpoint_candidate_pool_size", top_k)), top_k
    )
    checkpoint_candidates: list[tuple[float, float, int, dict[str, torch.Tensor], float]] = []
    minimum_candidate_loss_seen = np.inf
    logs: list[dict[str, Any]] = list(pretraining_logs)
    start = time.perf_counter()
    minimum_epochs = int(cfg["model"].get("minimum_epochs", 1))
    validation_every = max(int(cfg["model"].get("validation_every_epochs", 1)), 1)
    component_checkpoint_merge = bool(
        cfg["model"].get("component_checkpoint_merge", False)
    )
    if component_checkpoint_merge:
        if str(cfg["model"].get("asset_head_scope", "shared")) != "module":
            raise ValueError(
                "component_checkpoint_merge requires model.asset_head_scope=module"
            )
        if freeze_encoder_epochs < maximum_epochs:
            raise ValueError(
                "component_checkpoint_merge requires the representation encoder "
                "to remain frozen for the full supervised run"
            )
    component_best_scores: dict[str, float] = {
        **{f"asset:{module}": -np.inf for module in RISK_SLEEVES},
        "sleeve": -np.inf,
    }
    component_best_epochs: dict[str, int] = {}
    component_candidates: dict[
        str, list[tuple[float, int, dict[str, torch.Tensor]]]
    ] = {component: [] for component in component_best_scores}
    component_min_delta = float(
        cfg["model"].get("component_checkpoint_min_delta", 0.0)
    )
    component_minimum_epochs = int(
        cfg["model"].get(
            "component_checkpoint_minimum_epochs",
            minimum_epochs,
        )
    )
    component_average_top_k = max(
        int(cfg["model"].get("component_checkpoint_average_top_k", 1)),
        1,
    )
    component_candidate_pool_size = max(
        int(
            cfg["model"].get(
                "component_checkpoint_candidate_pool_size",
                component_average_top_k * 4,
            )
        ),
        component_average_top_k,
    )
    component_minimum_spacing_epochs = max(
        int(
            cfg["model"].get(
                "component_checkpoint_minimum_spacing_epochs",
                validation_every,
            )
        ),
        0,
    )
    module_lookup = {
        module: position for position, module in enumerate(RISK_SLEEVES)
    }

    def component_prefixes(component: str) -> tuple[str, ...]:
        if component == "sleeve":
            return (
                "macro_pool_score.",
                "sleeve_head.",
                "sleeve_mean.",
                "sleeve_logvar.",
            )
        module = component.split(":", 1)[1]
        module_position = module_lookup[module]
        return (
            f"asset_mean_by_module.{module_position}.",
            f"asset_logvar_by_module.{module_position}.",
        )

    def extract_component_state(
        component: str,
    ) -> dict[str, torch.Tensor]:
        prefixes = component_prefixes(component)
        return {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
            if key.startswith(prefixes)
        }

    def component_scores(metrics: dict[str, float]) -> dict[str, float]:
        component_selection_metric = str(
            cfg["model"].get(
                "component_checkpoint_selection_metric",
                "decision_aligned_utility",
            )
        )
        if component_selection_metric == "mean_view_rank":
            values = {
                f"asset:{module}": float(
                    metrics.get(
                        "validation_asset_weighted_incremental_rank_ic_"
                        f"{module}",
                        np.nan,
                    )
                )
                for module in RISK_SLEEVES
            }
            values["sleeve"] = float(
                metrics.get(
                    "validation_sleeve_weighted_incremental_rank_ic",
                    np.nan,
                )
            )
            return values
        if component_selection_metric != "decision_aligned_utility":
            raise ValueError(
                "Unknown model.component_checkpoint_selection_metric="
                f"{component_selection_metric}"
            )
        values = {
            f"asset:{module}": float(
                metrics.get(
                    f"validation_asset_decision_utility_{module}",
                    np.nan,
                )
            )
            for module in RISK_SLEEVES
        }
        values["sleeve"] = float(
            metrics.get("validation_sleeve_decision_utility", np.nan)
        )
        return values
    rank_metric_names = [
        "validation_asset_residual_rank_ic",
        "validation_asset_combined_rank_ic",
        "validation_asset_baseline_rank_ic",
        "validation_asset_incremental_rank_ic",
        "validation_sleeve_residual_rank_ic",
        "validation_sleeve_combined_rank_ic",
        "validation_sleeve_baseline_rank_ic",
        "validation_sleeve_incremental_rank_ic",
        "validation_asset_weighted_incremental_rank_ic",
        "validation_sleeve_weighted_incremental_rank_ic",
        "validation_overall_rank_score",
        "validation_regime_balanced_rank_score",
        "validation_year_balanced_rank_score",
    ]
    for horizon in metadata["horizons_days"]:
        suffix = f"h{int(horizon)}"
        rank_metric_names.extend(
            [
                f"validation_asset_incremental_rank_ic_{suffix}",
                f"validation_sleeve_incremental_rank_ic_{suffix}",
                f"validation_asset_combined_rank_ic_{suffix}",
                f"validation_sleeve_combined_rank_ic_{suffix}",
            ]
        )
        for module in RISK_SLEEVES:
            rank_metric_names.extend(
                [
                    "validation_asset_combined_rank_ic_"
                    f"{module}_{suffix}",
                    "validation_asset_baseline_rank_ic_"
                    f"{module}_{suffix}",
                    "validation_asset_incremental_rank_ic_"
                    f"{module}_{suffix}",
                ]
            )
    rank_metric_names.extend(
        [
            "validation_asset_weighted_incremental_rank_ic_"
            f"{module}"
            for module in RISK_SLEEVES
        ]
    )
    for epoch in range(1, maximum_epochs + 1):
        if freeze_encoder_epochs > 0 and epoch == freeze_encoder_epochs + 1:
            _set_supervised_encoder_frozen(model, False)
        model.train()
        train_total = 0.0
        train_count = 0
        component_sums: dict[str, float] = {}
        for raw in train_loader:
            batch = to_device(raw, device)
            batch = augment_training_batch(batch, cfg)
            output = model_forward(model, batch)
            loss, components = multitask_loss(output, batch, model.module_index, cfg["model"])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["model"]["gradient_clip"]))
            optimizer.step()
            batch_size = len(batch["index"])
            train_total += float(loss.detach()) * batch_size
            train_count += batch_size
            for key, value in components.items():
                component_sums[key] = component_sums.get(key, 0.0) + value * batch_size
        should_validate = epoch == 1 or epoch % validation_every == 0 or epoch == maximum_epochs
        if should_validate:
            val_loss, validation_bundle = evaluate_with_predictions(model, val_loader, cfg, device)
            selection_score, rank_metrics = validation_rank_score(
                validation_bundle, model.module_index.cpu().numpy(), arrays, cfg
            )
            if selection_mode == "rank_constrained":
                checkpoint_objective = -selection_score
                improved = bool(
                    epoch >= minimum_epochs
                    and np.isfinite(selection_score)
                    and selection_score > best_score + float(cfg["model"]["min_delta"])
                )
            else:
                checkpoint_objective = val_loss - float(
                    cfg["model"].get("checkpoint_rank_reward", 0.0)
                ) * selection_score
                improved = bool(
                    epoch >= minimum_epochs
                    and checkpoint_objective < best_objective - float(cfg["model"]["min_delta"])
                )
        else:
            val_loss = np.nan
            selection_score = np.nan
            checkpoint_objective = np.nan
            rank_metrics = {name: np.nan for name in rank_metric_names}
            improved = False
        component_improved = False
        if (
            component_checkpoint_merge
            and should_validate
            and epoch >= component_minimum_epochs
        ):
            current_component_scores = component_scores(rank_metrics)
            for component, score in current_component_scores.items():
                if not np.isfinite(score):
                    continue
                component_candidates[component].append(
                    (
                        float(score),
                        int(epoch),
                        extract_component_state(component),
                    )
                )
                component_candidates[component].sort(
                    key=lambda value: (-value[0], value[1])
                )
                component_candidates[component] = component_candidates[
                    component
                ][:component_candidate_pool_size]
                if (
                    score
                    > component_best_scores[component]
                    + component_min_delta
                ):
                    component_best_scores[component] = float(score)
                    component_best_epochs[component] = epoch
                    component_improved = True
        if improved:
            best_score = selection_score
            best_loss = val_loss
            best_objective = checkpoint_objective
            best_state_score = selection_score
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
        if improved or component_improved:
            stale = 0
        elif epoch >= minimum_epochs:
            stale += 1
        if should_validate and epoch >= minimum_epochs and np.isfinite(val_loss):
            minimum_candidate_loss_seen = min(minimum_candidate_loss_seen, val_loss)
            checkpoint_candidates.append(
                (
                    checkpoint_objective,
                    val_loss,
                    epoch,
                    copy.deepcopy(model.state_dict()),
                    selection_score,
                )
            )
            checkpoint_candidates.sort(key=lambda value: value[0])
            checkpoint_candidates = checkpoint_candidates[:candidate_pool_size]
        row = {
            "phase": "supervised_finetuning",
            "year": year,
            "seed": seed,
            "epoch": epoch,
            "train_loss": train_total / max(train_count, 1),
            "validation_loss": val_loss,
            "validation_selection_score": selection_score,
            "validation_checkpoint_objective": checkpoint_objective,
            "best_validation_checkpoint_objective": best_objective,
            "best_validation_selection_score": best_score,
            "selected_checkpoint_rank_score": best_state_score,
            "best_validation_loss": best_loss,
            "best_epoch": best_epoch,
            "elapsed_seconds": time.perf_counter() - start,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            **rank_metrics,
            **{key: value / max(train_count, 1) for key, value in component_sums.items()},
        }
        logs.append(row)
        if epoch == 1 or epoch % int(cfg["model"]["progress_every_epochs"]) == 0:
            print(json.dumps(row, ensure_ascii=False), flush=True)
            append_progress(cfg, row)
        scheduler.step()
        if (
            bool(cfg["model"].get("early_stopping_enabled", True))
            and should_validate
            and epoch >= minimum_epochs
            and stale >= int(cfg["model"]["patience"])
        ):
            break
    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())
        best_loss = float(val_loss)
        best_objective = float(checkpoint_objective)
        best_state_score = float(selection_score)
        best_epoch = int(logs[-1]["epoch"]) if logs else maximum_epochs
    selected_state = best_state
    checkpoint_selection = (
        "single_best_rank_constrained"
        if selection_mode == "rank_constrained"
        else "single_best_loss_rank_composite"
    )
    selected_validation_loss = best_loss
    selected_rank_score = best_state_score
    averaged_epochs: list[int] = []
    if selection_mode == "rank_constrained" and checkpoint_candidates:
        minimum_candidate_loss = minimum_candidate_loss_seen
        relative_tolerance = max(
            float(cfg["model"].get("checkpoint_rank_loss_relative_tolerance", 0.15)), 0.0
        )
        rank_floor = float(cfg["model"].get("checkpoint_rank_min_score", -np.inf))
        eligible_candidates = [
            value
            for value in checkpoint_candidates
            if value[1] <= minimum_candidate_loss * (1.0 + relative_tolerance)
            and value[4] >= rank_floor
        ]
        if not eligible_candidates:
            eligible_candidates = [min(checkpoint_candidates, key=lambda value: value[1])]
        eligible_candidates.sort(key=lambda value: (-value[4], value[1], value[2]))
        checkpoint_candidates = eligible_candidates[:top_k]
        _, best_loss, best_epoch, best_state, best_state_score = checkpoint_candidates[0]
        best_objective = -best_state_score
        selected_state = best_state
        selected_validation_loss = best_loss
        selected_rank_score = best_state_score
    checkpoint_spacing = max(int(cfg["model"].get("checkpoint_min_epoch_spacing", 0)), 0)
    if checkpoint_candidates:
        ordered_candidates = sorted(checkpoint_candidates, key=lambda value: value[0])
        spaced_candidates: list[tuple[float, float, int, dict[str, torch.Tensor], float]] = []
        for candidate in ordered_candidates:
            if checkpoint_spacing and any(
                abs(candidate[2] - selected[2]) < checkpoint_spacing
                for selected in spaced_candidates
            ):
                continue
            spaced_candidates.append(candidate)
            if len(spaced_candidates) >= top_k:
                break
        checkpoint_candidates = spaced_candidates or ordered_candidates[:top_k]
    if top_k > 1 and len(checkpoint_candidates) > 1:
        averaged_epochs = [epoch for _, _, epoch, _, _ in checkpoint_candidates]
        averaged_state = average_state_dicts([state for _, _, _, state, _ in checkpoint_candidates])
        model.load_state_dict(averaged_state)
        averaged_loss, averaged_bundle = evaluate_with_predictions(model, val_loader, cfg, device)
        averaged_rank, _ = validation_rank_score(
            averaged_bundle, model.module_index.cpu().numpy(), arrays, cfg
        )
        averaged_objective = (
            -averaged_rank
            if selection_mode == "rank_constrained"
            else averaged_loss
            - float(cfg["model"].get("checkpoint_rank_reward", 0.0)) * averaged_rank
        )
        loss_tolerance = float(cfg["model"].get("checkpoint_average_max_loss_degradation", 0.0))
        rank_tolerance = float(cfg["model"].get("checkpoint_average_max_rank_degradation", 0.0))
        if (
            averaged_loss <= best_loss + loss_tolerance
            and averaged_rank >= best_state_score - rank_tolerance
            and averaged_objective
            <= best_objective + float(cfg["model"].get("checkpoint_average_max_objective_degradation", 0.0))
        ):
            selected_state = averaged_state
            checkpoint_selection = (
                f"top_{len(checkpoint_candidates)}_rank_constrained_checkpoint_average"
                if selection_mode == "rank_constrained"
                else f"top_{len(checkpoint_candidates)}_validation_checkpoint_average"
            )
            selected_validation_loss = averaged_loss
            selected_rank_score = averaged_rank
    component_merge_accepted = False
    component_averaged_epochs: dict[str, list[int]] = {}
    nonempty_component_candidates = {
        component: values
        for component, values in component_candidates.items()
        if values
    }
    if component_checkpoint_merge and nonempty_component_candidates:
        merged_state = copy.deepcopy(selected_state)
        for component, candidates in nonempty_component_candidates.items():
            selected_candidates: list[
                tuple[float, int, dict[str, torch.Tensor]]
            ] = []
            for candidate in candidates:
                candidate_epoch = candidate[1]
                if (
                    component_minimum_spacing_epochs <= 0
                    or all(
                        abs(candidate_epoch - selected[1])
                        >= component_minimum_spacing_epochs
                        for selected in selected_candidates
                    )
                ):
                    selected_candidates.append(candidate)
                if len(selected_candidates) >= component_average_top_k:
                    break
            if not selected_candidates:
                continue
            component_averaged_epochs[component] = [
                value[1] for value in selected_candidates
            ]
            component_states = [value[2] for value in selected_candidates]
            for key in component_states[0]:
                merged_state[key] = torch.stack(
                    [state[key] for state in component_states],
                    dim=0,
                ).mean(dim=0)
        model.load_state_dict(merged_state)
        merged_loss, merged_bundle = evaluate_with_predictions(
            model, val_loader, cfg, device
        )
        merged_score, _ = validation_rank_score(
            merged_bundle,
            model.module_index.cpu().numpy(),
            arrays,
            cfg,
        )
        maximum_score_degradation = float(
            cfg["model"].get(
                "component_checkpoint_max_score_degradation", 0.0
            )
        )
        if (
            np.isfinite(merged_loss)
            and np.isfinite(merged_score)
            and merged_score
            >= selected_rank_score - maximum_score_degradation
        ):
            selected_state = merged_state
            selected_validation_loss = float(merged_loss)
            selected_rank_score = float(merged_score)
            checkpoint_selection += "_component_merged"
            component_merge_accepted = True
    model.load_state_dict(selected_state)
    validation = predict(model, val_loader, device, cfg)
    logs.append(
        {
            "phase": "checkpoint_selection",
            "year": year,
            "seed": seed,
            "epoch": "checkpoint_selection",
            "best_epoch": best_epoch,
            "best_validation_loss": best_loss,
            "best_validation_checkpoint_objective": best_objective,
            "selected_validation_loss": selected_validation_loss,
            "selected_checkpoint_rank_score": selected_rank_score,
            "checkpoint_selection": checkpoint_selection,
            "checkpoint_selection_mode": selection_mode,
            "averaged_epochs": ",".join(str(value) for value in averaged_epochs),
            "component_checkpoint_merge_accepted": component_merge_accepted,
            "component_best_epochs": json.dumps(
                component_best_epochs, sort_keys=True
            ),
            "component_averaged_epochs": json.dumps(
                component_averaged_epochs, sort_keys=True
            ),
            "epochs_run": int(logs[-1]["epoch"]) if logs else 0,
        }
    )
    checkpoint = {
        "year": year,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "best_validation_checkpoint_objective": best_objective,
        "best_validation_selection_score": best_score,
        "selected_validation_loss": selected_validation_loss,
        "selected_checkpoint_rank_score": selected_rank_score,
        "checkpoint_selection": checkpoint_selection,
        "checkpoint_selection_mode": selection_mode,
        "averaged_epochs": averaged_epochs,
        "component_checkpoint_merge_accepted": component_merge_accepted,
        "asset_regression_loss_scale_by_module": cfg["model"].get(
            "asset_regression_loss_scale_by_module"
        ),
        "sleeve_regression_loss_scale_by_sleeve": cfg["model"].get(
            "sleeve_regression_loss_scale_by_sleeve"
        ),
        "component_best_epochs": component_best_epochs,
        "component_best_scores": component_best_scores,
        "component_averaged_epochs": component_averaged_epochs,
        "epochs_run": int(logs[-2]["epoch"]) if len(logs) >= 2 else 0,
        "pretraining": pretraining_summary,
        "state_dict": {key: value.cpu() for key, value in selected_state.items()},
    }
    return checkpoint, logs, validation


def merge_seed_predictions(values: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    reference = values[0]
    average_keys = {
        "asset_residual",
        "asset_logvar",
        "sleeve_residual",
        "sleeve_logvar",
        "asset_expert_gate",
        "sleeve_expert_gate",
    }.intersection(reference)
    output = {
        key: value for key, value in reference.items() if key not in average_keys
    }
    for key in sorted(average_keys):
        output[key] = np.mean([value[key] for value in values], axis=0)
    return output


def mature_historical_predictions(
    history: list[dict[str, np.ndarray]],
    arrays,
    cutoff_ns: int,
) -> dict[str, np.ndarray] | None:
    """Combine only previously issued forecasts whose labels have matured."""
    if not history:
        return None
    combined = {
        key: np.concatenate([bundle[key] for bundle in history], axis=0)
        for key in history[0]
    }
    indices = combined["index"].astype(np.int64)
    matured = (
        (arrays.label_end[indices] > np.iinfo(np.int64).min)
        & (arrays.label_end[indices] < int(cutoff_ns))
    )
    combined["asset_mask"] = combined["asset_mask"].astype(bool) & matured[:, :, None]
    combined["sleeve_mask"] = combined["sleeve_mask"].astype(bool) & matured[:, :, None]
    usable = combined["asset_mask"].any(axis=(1, 2)) | combined["sleeve_mask"].any(
        axis=(1, 2)
    )
    if not usable.any():
        return None
    return {key: value[usable] for key, value in combined.items()}


def asset_policy_for(
    policies: dict[str, Any],
    horizon_position: int,
    module: str,
) -> dict[str, float]:
    scoped = policies.get("asset_by_module", {})
    if module in scoped and horizon_position < len(scoped[module]):
        return scoped[module][horizon_position]
    return policies["asset"][horizon_position]


def compact_blend_policy(
    policies: dict[str, Any],
    horizons: list[int],
) -> dict[str, Any]:
    """Keep progress logs readable without discarding the selected decision."""

    def compact_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for position, row in enumerate(rows):
            horizon_position = int(row.get("horizon_index", position))
            output.append(
                {
                    "horizon_days": int(horizons[horizon_position]),
                    "baseline_weight": float(row["baseline_weight"]),
                    "residual_alpha": float(row["residual_alpha"]),
                    "selection_score": float(
                        row.get("selection_score", np.nan)
                    ),
                    "bootstrap_positive_probability": float(
                        row.get("bootstrap_positive_probability", np.nan)
                    ),
                }
            )
        return output

    return {
        "asset": compact_rows(policies["asset"]),
        "sleeve": compact_rows(policies["sleeve"]),
        "asset_by_module": {
            module: compact_rows(rows)
            for module, rows in policies.get(
                "asset_by_module", {}
            ).items()
        },
    }


def blended_asset_scores(
    bundle: dict[str, np.ndarray],
    policies: dict[str, Any],
    modules: list[str],
    horizon_position: int,
    cfg: dict[str, Any],
) -> np.ndarray:
    score = np.zeros_like(bundle["asset_baseline"][:, horizon_position])
    module_array = np.asarray(modules, dtype=object)
    for module in sorted(set(modules)):
        columns = np.flatnonzero(module_array == module)
        policy = asset_policy_for(policies, horizon_position, module)
        score[:, columns] = _blend_prediction(
            bundle["asset_baseline"][:, horizon_position, columns],
            bundle["asset_residual"][:, horizon_position, columns],
            policy["baseline_weight"],
            policy["residual_alpha"],
            cfg,
        )
    return score


def append_long_predictions(
    rows_asset: list[dict[str, Any]],
    rows_sleeve: list[dict[str, Any]],
    bundle: dict[str, np.ndarray],
    arrays,
    metadata: dict[str, Any],
    policies: dict[str, Any],
    year: int,
    cfg: dict[str, Any],
) -> None:
    dates = pd.to_datetime(arrays.dates[bundle["index"]])
    horizons = metadata["horizons_days"]
    assets = metadata["assets"]
    modules = metadata["asset_modules"]
    asset_expert_names = metadata.get(
        "asset_signal_experts", ["fixed_causal_baseline"]
    )
    sleeve_expert_names = metadata.get(
        "sleeve_signal_experts", ["fixed_causal_baseline"]
    )
    asset_score = np.zeros_like(bundle["asset_baseline"])
    sleeve_score = np.zeros_like(bundle["sleeve_baseline"])
    for horizon_position in range(len(horizons)):
        sleeve_policy = policies["sleeve"][horizon_position]
        asset_score[:, horizon_position] = blended_asset_scores(
            bundle,
            policies,
            modules,
            horizon_position,
            cfg,
        )
        sleeve_score[:, horizon_position] = _blend_prediction(
            bundle["sleeve_baseline"][:, horizon_position],
            bundle["sleeve_residual"][:, horizon_position],
            sleeve_policy["baseline_weight"],
            sleeve_policy["residual_alpha"],
            cfg,
        )
    for row_position, index in enumerate(bundle["index"]):
        for horizon_position, horizon in enumerate(horizons):
            label_ns = int(arrays.label_end[index, horizon_position])
            label_end = pd.NaT if label_ns == np.iinfo(np.int64).min else pd.Timestamp(label_ns)
            sleeve_policy = policies["sleeve"][horizon_position]
            for asset_position, asset in enumerate(assets):
                asset_policy = asset_policy_for(
                    policies,
                    horizon_position,
                    modules[asset_position],
                )
                valid = bool(bundle["asset_mask"][row_position, horizon_position, asset_position])
                raw_model_score = float(
                    bundle["asset_residual"][
                        row_position, horizon_position, asset_position
                    ]
                )
                baseline_score = float(
                    bundle["asset_baseline"][
                        row_position, horizon_position, asset_position
                    ]
                )
                correction_score = float(
                    _prediction_correction(
                        np.asarray(baseline_score),
                        np.asarray(raw_model_score),
                        cfg,
                    )
                )
                raw_combined_score = baseline_score + correction_score
                row = {
                        "as_of_date": dates[row_position],
                        "horizon_label_end_date": label_end,
                        "horizon_days": int(horizon),
                        "asset_code": asset,
                        "module": modules[asset_position],
                        "baseline_relative_score": baseline_score,
                        "model_raw_score": raw_combined_score,
                        "model_residual_score": correction_score,
                        "walkforward_relative_score": float(asset_score[row_position, horizon_position, asset_position]),
                        "historical_oos_relative_score": float(asset_score[row_position, horizon_position, asset_position]),
                        "realized_relative_score": float(bundle["asset_target"][row_position, horizon_position, asset_position]) if valid else np.nan,
                        "prediction_uncertainty": float(np.exp(0.5 * bundle["asset_logvar"][row_position, horizon_position, asset_position])),
                        "baseline_weight": asset_policy["baseline_weight"],
                        "residual_alpha": asset_policy["residual_alpha"],
                        "training_cutoff_year": year,
                        "model_training_cutoff": pd.Timestamp(
                            year - 1, 12, 31
                        ),
                        "has_label": valid,
                        "score_semantics": (
                            "annualized_relative_mean_view"
                            if str(cfg["data"].get("target_mode", ""))
                            == "relative_mean_return"
                            else "relative_forecast_score"
                        ),
                        "forecast_baseline_mode": str(
                            cfg["data"].get(
                                "forecast_baseline_mode", "causal_anchor"
                            )
                        ),
                        "score_normalization_scale": float(
                            cfg["model"].get(
                                "deployment_score_normalization_scale", 0.10
                            )
                        ),
                    }
                if "asset_expert_gate" in bundle:
                    row.update(
                        {
                            f"expert_weight__{name}": float(
                                bundle["asset_expert_gate"][
                                    row_position,
                                    horizon_position,
                                    asset_position,
                                    expert_position,
                                ]
                            )
                            for expert_position, name in enumerate(
                                asset_expert_names
                            )
                        }
                    )
                rows_asset.append(row)
            for sleeve_position, sleeve in enumerate(RISK_SLEEVES):
                valid = bool(bundle["sleeve_mask"][row_position, horizon_position, sleeve_position])
                value = float(sleeve_score[row_position, horizon_position, sleeve_position])
                raw_model_score = float(
                    bundle["sleeve_residual"][
                        row_position, horizon_position, sleeve_position
                    ]
                )
                baseline_score = float(
                    bundle["sleeve_baseline"][
                        row_position, horizon_position, sleeve_position
                    ]
                )
                correction_score = float(
                    _prediction_correction(
                        np.asarray(baseline_score),
                        np.asarray(raw_model_score),
                        cfg,
                    )
                )
                raw_combined_score = baseline_score + correction_score
                row = {
                        "as_of_date": dates[row_position],
                        "label_end_date": label_end,
                        "horizon_label_end_date": label_end,
                        "has_label": valid,
                        "sleeve": sleeve,
                        "pred_relative_score": value,
                        "baseline_relative_score": baseline_score,
                        "model_raw_score": raw_combined_score,
                        "model_residual_score": correction_score,
                        "walkforward_relative_score": value,
                        "adaptive_relative_score": value,
                        "adaptive_model_weight": sleeve_policy["residual_alpha"],
                        "historical_oos_relative_score": value,
                        "historical_oos_adaptive_score": value,
                        "combined_oos_adaptive_score": value,
                        "realized_relative_score": float(bundle["sleeve_target"][row_position, horizon_position, sleeve_position]) if valid else np.nan,
                        "realized_raw_return": np.nan,
                        "prediction_uncertainty": float(np.exp(0.5 * bundle["sleeve_logvar"][row_position, horizon_position, sleeve_position])),
                        "baseline_weight": sleeve_policy["baseline_weight"],
                        "residual_alpha": sleeve_policy["residual_alpha"],
                        "training_cutoff_year": year,
                        "model_training_cutoff": pd.Timestamp(
                            year - 1, 12, 31
                        ),
                        "horizon_days": int(horizon),
                        "score_semantics": (
                            "annualized_relative_mean_view"
                            if str(cfg["data"].get("target_mode", ""))
                            == "relative_mean_return"
                            else "relative_forecast_score"
                        ),
                        "forecast_baseline_mode": str(
                            cfg["data"].get(
                                "forecast_baseline_mode", "causal_anchor"
                            )
                        ),
                        "score_normalization_scale": float(
                            cfg["model"].get(
                                "deployment_score_normalization_scale", 0.10
                            )
                        ),
                    }
                if "sleeve_expert_gate" in bundle:
                    row.update(
                        {
                            f"expert_weight__{name}": float(
                                bundle["sleeve_expert_gate"][
                                    row_position,
                                    horizon_position,
                                    sleeve_position,
                                    expert_position,
                                ]
                            )
                            for expert_position, name in enumerate(
                                sleeve_expert_names
                            )
                        }
                    )
                rows_sleeve.append(row)


def expert_usage_rows(
    frame: pd.DataFrame,
    task: str,
    expert_names: list[str],
) -> list[dict[str, Any]]:
    weight_columns = [f"expert_weight__{name}" for name in expert_names]
    if not set(weight_columns).issubset(frame.columns):
        return []
    output: list[dict[str, Any]] = []
    for (year, horizon), group in frame.groupby(
        ["training_cutoff_year", "horizon_days"], sort=True
    ):
        weights = group[weight_columns].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(dtype=float)
        valid = np.isfinite(weights).all(axis=1) & (weights.sum(axis=1) > 1e-10)
        weights = weights[valid]
        if not len(weights):
            continue
        dominant = np.argmax(weights, axis=1)
        for expert_position, expert_name in enumerate(expert_names):
            expert_weight = weights[:, expert_position]
            output.append(
                {
                    "task": task,
                    "year": int(year),
                    "horizon_days": int(horizon),
                    "expert": expert_name,
                    "mean_weight": float(np.mean(expert_weight)),
                    "std_weight": float(np.std(expert_weight)),
                    "p05_weight": float(np.quantile(expert_weight, 0.05)),
                    "p95_weight": float(np.quantile(expert_weight, 0.95)),
                    "dominant_fraction": float(
                        np.mean(dominant == expert_position)
                    ),
                    "observations": int(len(weights)),
                }
            )
    return output


def metric_rows(
    bundle: dict[str, np.ndarray],
    metadata: dict[str, Any],
    module_index: np.ndarray,
    policies: dict[str, Any],
    year: int,
    cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for horizon_position, horizon in enumerate(metadata["horizons_days"]):
        sleeve_policy = policies["sleeve"][horizon_position]
        asset_score = blended_asset_scores(
            bundle,
            policies,
            metadata["asset_modules"],
            horizon_position,
            cfg,
        )
        asset_policies = {
            module: asset_policy_for(policies, horizon_position, module)
            for module in sorted(set(metadata["asset_modules"]))
        }
        asset_baseline_weight = float(
            np.mean(
                [
                    float(policy["baseline_weight"])
                    for policy in asset_policies.values()
                ]
            )
        )
        asset_residual_alpha = float(
            np.mean(
                [
                    float(policy["residual_alpha"])
                    for policy in asset_policies.values()
                ]
            )
        )
        sleeve_score = _blend_prediction(
            bundle["sleeve_baseline"][:, horizon_position],
            bundle["sleeve_residual"][:, horizon_position],
            sleeve_policy["baseline_weight"],
            sleeve_policy["residual_alpha"],
            cfg,
        )
        asset_model = rank_ic(asset_score, bundle["asset_target"][:, horizon_position], bundle["asset_mask"][:, horizon_position], module_index)
        asset_base = rank_ic(bundle["asset_baseline"][:, horizon_position], bundle["asset_target"][:, horizon_position], bundle["asset_mask"][:, horizon_position], module_index)
        sleeve_model = rank_ic(sleeve_score, bundle["sleeve_target"][:, horizon_position], bundle["sleeve_mask"][:, horizon_position], None)
        sleeve_base = rank_ic(bundle["sleeve_baseline"][:, horizon_position], bundle["sleeve_target"][:, horizon_position], bundle["sleeve_mask"][:, horizon_position], None)
        if (
            str(cfg["data"].get("forecast_baseline_mode", ""))
            == "zero_active_view"
        ):
            asset_base = 0.0 if not np.isfinite(asset_base) else asset_base
            sleeve_base = 0.0 if not np.isfinite(sleeve_base) else sleeve_base
        output.append(
            {
                "year": year,
                "horizon_days": int(horizon),
                "asset_baseline_weight": asset_baseline_weight,
                "asset_residual_alpha": asset_residual_alpha,
                "asset_residual_alpha_by_module": json.dumps(
                    {
                        module: float(policy["residual_alpha"])
                        for module, policy in asset_policies.items()
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "sleeve_baseline_weight": sleeve_policy["baseline_weight"],
                "sleeve_residual_alpha": sleeve_policy["residual_alpha"],
                "asset_model_rank_ic": asset_model,
                "asset_baseline_rank_ic": asset_base,
                "asset_incremental_rank_ic": asset_model - asset_base,
                "sleeve_model_rank_ic": sleeve_model,
                "sleeve_baseline_rank_ic": sleeve_base,
                "sleeve_incremental_rank_ic": sleeve_model - sleeve_base,
                "samples": len(bundle["index"]),
            }
        )
    return output


def _long_component_utility(
    frame: pd.DataFrame,
    cfg: dict[str, Any],
    task: str,
) -> list[dict[str, Any]]:
    """Evaluate historically selected views on already-realized labels."""
    if task not in {"asset", "sleeve"}:
        raise ValueError(f"Unknown component task={task}")
    values = frame.copy()
    values["as_of_date"] = pd.to_datetime(
        values["as_of_date"], errors="coerce"
    )
    values["horizon_days"] = pd.to_numeric(
        values["horizon_days"], errors="coerce"
    )
    values["residual_alpha"] = pd.to_numeric(
        values["residual_alpha"], errors="coerce"
    )
    values["walkforward_relative_score"] = pd.to_numeric(
        values["walkforward_relative_score"], errors="coerce"
    )
    values["baseline_relative_score"] = pd.to_numeric(
        values["baseline_relative_score"], errors="coerce"
    )
    values["baseline_weight"] = pd.to_numeric(
        values["baseline_weight"], errors="coerce"
    )
    values["realized_relative_score"] = pd.to_numeric(
        values["realized_relative_score"], errors="coerce"
    )
    start = pd.Timestamp(cfg["gate"]["evaluation_start"])
    values = values.loc[
        values["as_of_date"].ge(start)
        & values["residual_alpha"].gt(0.0)
        & values["walkforward_relative_score"].notna()
        & values["realized_relative_score"].notna()
    ].copy()
    if values.empty:
        return []
    evidence_metric = str(
        cfg["gate"].get(
            "component_evidence_metric",
            "decision_aligned_utility",
        )
    )
    if evidence_metric not in {
        "decision_aligned_utility",
        "rank_ic",
    }:
        raise ValueError(
            "Unknown gate.component_evidence_metric="
            f"{evidence_metric}"
        )

    configured_horizons = [
        int(value) for value in cfg["data"]["horizons_days"]
    ]
    group_columns = (
        ["module", "horizon_days"]
        if task == "asset"
        else ["horizon_days"]
    )
    results: list[dict[str, Any]] = []
    for raw_key, component in values.groupby(group_columns, sort=True):
        key = raw_key if isinstance(raw_key, tuple) else (raw_key,)
        if task == "asset":
            module = str(key[0])
            horizon_days = int(key[1])
            component_name = f"asset:{module}:h{horizon_days}"
        else:
            module = ""
            horizon_days = int(key[0])
            component_name = f"sleeve:h{horizon_days}"
        if horizon_days not in configured_horizons:
            continue
        horizon_position = configured_horizons.index(horizon_days)
        temperature = _validation_horizon_value(
            cfg["model"].get(
                f"{task}_decision_temperature", 0.10
            ),
            horizon_position,
            len(configured_horizons),
            0.10,
        )
        transaction_cost_bps = _validation_horizon_value(
            cfg["model"].get("decision_transaction_cost_bps", 5.0),
            horizon_position,
            len(configured_horizons),
            5.0,
        )
        cost_rate = (
            float(transaction_cost_bps)
            * 1e-4
            * 252.0
            / max(float(horizon_days), 1.0)
        )
        observations: list[dict[str, Any]] = []
        for date, daily in component.groupby("as_of_date", sort=True):
            score = daily["walkforward_relative_score"].to_numpy(
                dtype=float
            )
            baseline = (
                daily["baseline_weight"].to_numpy(dtype=float)
                * daily["baseline_relative_score"].to_numpy(dtype=float)
            )
            target = daily["realized_relative_score"].to_numpy(dtype=float)
            valid = (
                np.isfinite(score)
                & np.isfinite(baseline)
                & np.isfinite(target)
            )
            if valid.sum() < 2:
                continue
            if evidence_metric == "rank_ic":
                prediction_ic = daily_rank_ic_series(
                    score[None, :],
                    target[None, :],
                    valid[None, :],
                    None,
                )[0]
                baseline_ic = daily_rank_ic_series(
                    baseline[None, :],
                    target[None, :],
                    valid[None, :],
                    None,
                )[0]
                edge = float(prediction_ic - baseline_ic)
            else:
                predicted_weight = _masked_softmax_numpy(
                    score / max(float(temperature), 1e-4), valid
                )
                fallback_weight = _masked_softmax_numpy(
                    baseline / max(float(temperature), 1e-4), valid
                )
                active = predicted_weight - fallback_weight
                gross = float(np.sum(active[valid] * target[valid]))
                turnover = 0.5 * float(np.abs(active[valid]).sum())
                edge = gross - cost_rate * turnover
            observations.append(
                {
                    "date": pd.Timestamp(date),
                    "edge": edge,
                }
            )
        observation_frame = pd.DataFrame(observations)
        if observation_frame.empty:
            continue
        yearly = observation_frame.groupby(
            observation_frame["date"].dt.year
        )["edge"].mean()
        quarterly = observation_frame.groupby(
            observation_frame["date"].dt.to_period("Q")
        )["edge"].mean()
        bootstrap = paired_block_bootstrap_mean(
            observation_frame["edge"].to_numpy(dtype=float),
            block_days=max(
                int(
                    round(
                        horizon_days
                        * float(
                            cfg["gate"].get(
                                "component_bootstrap_block_multiplier",
                                1.0,
                            )
                        )
                    )
                ),
                5,
            ),
            draws=int(
                cfg["gate"].get("component_bootstrap_draws", 2000)
            ),
            confidence=float(
                cfg["gate"].get("component_bootstrap_confidence", 0.90)
            ),
            seed=(
                int(cfg["gate"].get("component_bootstrap_seed", 20260727))
                + horizon_days * 10_000
                + (
                    0
                    if task == "sleeve"
                    else RISK_SLEEVES.index(module) + 1
                )
            ),
        )
        results.append(
            {
                "component": component_name,
                "task": task,
                "module": module,
                "horizon_days": horizon_days,
                "observations": int(len(observation_frame)),
                "effective_nonoverlap_observations": int(
                    math.ceil(
                        len(observation_frame)
                        / max(float(horizon_days), 1.0)
                    )
                ),
                "years": int(len(yearly)),
                "quarters": int(len(quarterly)),
                "evidence_metric": evidence_metric,
                "mean_evidence_edge": float(
                    observation_frame["edge"].mean()
                ),
                "mean_decision_utility": (
                    float(observation_frame["edge"].mean())
                    if evidence_metric == "decision_aligned_utility"
                    else np.nan
                ),
                "mean_rank_ic_edge": (
                    float(observation_frame["edge"].mean())
                    if evidence_metric == "rank_ic"
                    else np.nan
                ),
                "positive_observation_fraction": float(
                    observation_frame["edge"].gt(0.0).mean()
                ),
                "positive_year_fraction": float(yearly.gt(0.0).mean()),
                "positive_quarter_fraction": float(
                    quarterly.gt(0.0).mean()
                ),
                "worst_year_edge": float(yearly.min()),
                "worst_quarter_edge": float(quarterly.min()),
                **bootstrap,
            }
        )
    return results


def build_component_policy_gate(
    validation: pd.DataFrame,
    asset_predictions: pd.DataFrame,
    sleeve_predictions: pd.DataFrame,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    evidence = [
        *_long_component_utility(asset_predictions, cfg, "asset"),
        *_long_component_utility(sleeve_predictions, cfg, "sleeve"),
    ]
    minimum_observations = int(
        cfg["gate"].get("min_component_observations", 40)
    )
    evidence_metric = str(
        cfg["gate"].get(
            "component_evidence_metric",
            "decision_aligned_utility",
        )
    )
    if evidence_metric == "rank_ic":
        minimum_edge = float(
            cfg["gate"].get("min_component_rank_ic_edge", 0.0)
        )
    elif evidence_metric == "decision_aligned_utility":
        minimum_edge = float(
            cfg["gate"].get("min_component_decision_utility", 0.0)
        )
    else:
        raise ValueError(
            "Unknown gate.component_evidence_metric="
            f"{evidence_metric}"
        )
    minimum_positive_year_fraction = float(
        cfg["gate"].get("min_component_positive_year_fraction", 0.50)
    )
    minimum_effective_observations = int(
        cfg["gate"].get("min_component_effective_observations", 8)
    )
    minimum_years = int(cfg["gate"].get("min_component_years", 2))
    minimum_positive_quarter_fraction = float(
        cfg["gate"].get("min_component_positive_quarter_fraction", 0.50)
    )
    minimum_bootstrap_probability = float(
        cfg["gate"].get(
            "min_component_bootstrap_positive_probability",
            0.80,
        )
    )
    minimum_bootstrap_ci_low = float(
        cfg["gate"].get("min_component_bootstrap_ci_low", 0.0)
    )
    for row in evidence:
        row["passed"] = bool(
            row["observations"] >= minimum_observations
            and row["effective_nonoverlap_observations"]
            >= minimum_effective_observations
            and row["years"] >= minimum_years
            and row["evidence_metric"] == evidence_metric
            and row["mean_evidence_edge"] >= minimum_edge
            and row["positive_year_fraction"]
            >= minimum_positive_year_fraction
            and row["positive_quarter_fraction"]
            >= minimum_positive_quarter_fraction
            and row["bootstrap_positive_probability"]
            >= minimum_bootstrap_probability
            and row["bootstrap_edge_ci_low"]
            >= minimum_bootstrap_ci_low
        )

    prediction_frames = [asset_predictions, sleeve_predictions]
    finite_outputs = all(
        np.isfinite(
            pd.to_numeric(
                frame["walkforward_relative_score"], errors="coerce"
            )
        ).all()
        for frame in prediction_frames
    )
    forecast_baseline_mode = str(
        cfg["data"].get("forecast_baseline_mode", "causal_anchor")
    )
    finite_baselines = all(
        np.isfinite(
            pd.to_numeric(
                frame["baseline_relative_score"], errors="coerce"
            )
        ).all()
        for frame in prediction_frames
    )
    if forecast_baseline_mode == "zero_active_view":
        deployment_baseline_contract = finite_baselines and all(
            float(
                pd.to_numeric(
                    frame["baseline_relative_score"], errors="coerce"
                )
                .abs()
                .max()
            )
            <= 1e-12
            for frame in prediction_frames
        )
    elif forecast_baseline_mode == "causal_anchor":
        deployment_baseline_contract = finite_baselines
    else:
        deployment_baseline_contract = False

    fallback_contracts: list[bool] = []
    for frame in prediction_frames:
        residual_alpha = pd.to_numeric(
            frame["residual_alpha"], errors="coerce"
        )
        inactive = residual_alpha.fillna(0.0).le(0.0)
        observed = pd.to_numeric(
            frame.loc[inactive, "walkforward_relative_score"],
            errors="coerce",
        )
        expected = (
            pd.to_numeric(
                frame.loc[inactive, "baseline_weight"], errors="coerce"
            )
            * pd.to_numeric(
                frame.loc[inactive, "baseline_relative_score"],
                errors="coerce",
            )
        )
        fallback_contracts.append(
            bool(
                np.isfinite(observed).all()
                and np.isfinite(expected).all()
                and (observed - expected).abs().le(1e-12).all()
            )
        )
    fallback_contract = all(fallback_contracts)
    causal_label_dates = all(
        bool(
            (
                pd.to_datetime(
                    frame.loc[
                        frame["has_label"].fillna(False).astype(bool),
                        "horizon_label_end_date",
                    ],
                    errors="coerce",
                )
                > pd.to_datetime(
                    frame.loc[
                        frame["has_label"].fillna(False).astype(bool),
                        "as_of_date",
                    ],
                    errors="coerce",
                )
            ).all()
        )
        for frame in prediction_frames
    )
    active_components = [row["component"] for row in evidence]
    passed_components = [row["component"] for row in evidence if row["passed"]]
    structural_checks = {
        "finite_prediction_outputs": bool(finite_outputs),
        "known_forecast_baseline_mode": forecast_baseline_mode
        in {"zero_active_view", "causal_anchor"},
        "deployment_baseline_contract": bool(
            deployment_baseline_contract
        ),
        "fallback_contract_when_alpha_zero": bool(fallback_contract),
        "labels_strictly_after_as_of": bool(causal_label_dates),
    }
    evidence_checks = {
        "has_active_component": bool(active_components),
        "all_active_components_pass_oos_gate": bool(active_components)
        and len(passed_components) == len(active_components),
    }
    checks = {**structural_checks, **evidence_checks}
    evaluation = validation.loc[
        validation["year"].ge(
            pd.Timestamp(cfg["gate"]["evaluation_start"]).year
        )
    ]

    def finite_mean(column: str) -> float | None:
        values = pd.to_numeric(
            evaluation.get(column), errors="coerce"
        ).replace([np.inf, -np.inf], np.nan)
        return float(values.mean()) if values.notna().any() else None

    return {
        # Historical activation is already encoded row by row in residual_alpha,
        # using only forecasts whose labels had matured before that row's model
        # cutoff. Applying an ex-post full-sample evidence gate here would leak
        # later outcomes backward into earlier portfolio dates.
        "enabled_for_portfolio_stage": bool(all(structural_checks.values())),
        "enabled_for_formal_strategy": bool(all(checks.values())),
        "mode": "component_policy",
        "checks": checks,
        "structural_checks": structural_checks,
        "evidence_checks": evidence_checks,
        "historical_activation_rule": (
            "Use each row's causally calibrated residual_alpha. The formal "
            "full-sample evidence gate is reporting-only and never rewrites "
            "historical portfolio decisions."
        ),
        "active_components": active_components,
        "passed_components": passed_components,
        "component_evidence": evidence,
        "summary": {
            "asset_rank_ic": finite_mean("asset_model_rank_ic"),
            "asset_incremental_rank_ic": finite_mean(
                "asset_incremental_rank_ic"
            ),
            "sleeve_rank_ic": finite_mean("sleeve_model_rank_ic"),
            "sleeve_incremental_rank_ic": finite_mean(
                "sleeve_incremental_rank_ic"
            ),
        },
        "caveat": (
            "2024+ has already been observed during earlier strategy "
            "development and is a strict historical stress set, not a fresh "
            "untouched holdout."
        ),
    }


def write_report(
    cfg: dict[str, Any],
    metadata: dict[str, Any],
    validation: pd.DataFrame,
    gate: dict[str, Any],
    parameter_count: int,
) -> None:
    model_cfg = cfg["model"]
    early_stopping = bool(model_cfg.get("early_stopping_enabled", True))
    stopping_text = (
        f"early stopping uses patience {int(model_cfg['patience'])} and min_delta {float(model_cfg['min_delta']):g}"
        if early_stopping
        else "early stopping is disabled so every requested epoch is executed"
    )
    training_log_name = Path(cfg["outputs"]["training_log"]).name
    audit_prefix = training_log_name.removesuffix("_training_log.csv")
    sleeve_architecture = str(model_cfg.get("sleeve_architecture", "shared_regression"))
    if sleeve_architecture == "regime_moe_pairwise":
        sleeve_design = (
            "The sleeve ranking task is separated from sleeve magnitude regression. A point-in-time "
            "stock-bond regime embedding gates multiple sleeve ranking experts, and their Bradley-Terry-style "
            "strength differences are trained with a margin-filtered pairwise loss."
        )
    elif sleeve_architecture == "hierarchical_regime_pairwise":
        sleeve_design = (
            "The sleeve ranking task is separated from sleeve magnitude regression. A shared pairwise rank "
            "head learns cross-regime structure, while point-in-time bond-state, equity-state, and full 3x3-grid "
            "embeddings supply a regularized conditional residual. Its influence is shrunk by the number of "
            "same-state observations available strictly before the decision date."
        )
    elif sleeve_architecture == "regime_transition_hybrid":
        sleeve_design = (
            "The stock-bond 3x3 grid is an explicit supervised state space. The iTransformer predicts "
            "5/20/60-day transition probabilities over all nine future states. Sleeve views are the expected "
            "payoffs from a factorized global + bond-axis + equity-axis + grid-interaction state table, so the "
            "model cannot bypass the grid with an unconstrained direct sleeve rank head."
        )
    elif sleeve_architecture == "factorized_regime_transition":
        transition_target = model_cfg.get("transition_target", "terminal_state")
        target_description = (
            "future holding-period occupancy probabilities across all nine states"
            if transition_target == "path_occupancy"
            else "nine joint future-state probabilities"
        )
        sleeve_design = (
            "The stock-bond 3x3 grid remains the mandatory first allocation layer. The iTransformer predicts "
            "the three-state bond axis and three-state equity axis separately, combines them into "
            f"{target_description} with only a small interaction correction, and maps those probabilities "
            "through the fixed v4.3 grid-to-sleeve compatibility table. The neural model can forecast the grid "
            "but cannot bypass it or relearn its economic meaning."
        )
    else:
        sleeve_design = "The sleeve head jointly predicts residual magnitude and cross-sectional ordering."
    asset_count = len(metadata["assets"])
    local_feature_count = len(metadata["local_features"])
    macro_feature_count = len(metadata["macro_features"])
    sequence_length = int(metadata["sequence_length_days"])
    horizons = "/".join(str(value) for value in metadata["horizons_days"])
    baseline_description = json.dumps(
        metadata.get("sleeve_baseline_definition", metadata.get("baseline_definition", {})),
        ensure_ascii=False,
    )
    lines = [
        f"# {cfg.get('version', 'Hierarchical Asset iTransformer v7')}",
        "",
        "## Design",
        "",
        f"The model uses {asset_count} asset tokens plus {macro_feature_count} point-in-time macro/state variable tokens. For each asset, {local_feature_count} value features and {local_feature_count} availability masks are inverted into variable tokens: a shared Linear({sequence_length} -> {int(model_cfg['d_model'])}) projects each complete {sequence_length}-day path, and learned attention pooling forms one asset token. A {int(model_cfg['num_layers'])}-layer, {int(model_cfg['n_heads'])}-head cross-asset Transformer then predicts asset-level and sleeve-level {horizons}-day signals. {sleeve_design} The current model has {parameter_count:,} trainable parameters.",
        f"The sleeve causal baseline is `{baseline_description}`.",
        "",
        "Product portfolios may trade only assets already present in the formal anchor. Forecast horizons are prediction tasks, not fixed rebalancing clocks; the adaptive PPO controller separately chooses daily progress toward the target.",
        "",
        "## Training and leakage controls",
        "",
        f"- The configured run length is {int(model_cfg['max_epochs'])} epochs; {stopping_text}.",
        f"- Each annual fold uses an expanding training history and a chronological final {float(cfg['walk_forward']['validation_fraction']):.0%} validation block.",
        "- The training side is purged so every valid label end is strictly before the validation start; validation labels must mature before the prediction-year cutoff.",
        f"- Checkpoints are eligible only after {int(model_cfg.get('minimum_epochs', 1))} epochs and use the configured purged-validation loss/rank composite without any prediction-year outcome.",
        f"- Seeds {model_cfg['seeds']} are ensembled. See {audit_prefix}_convergence_summary.csv and {audit_prefix}_loss_curves.png for the complete convergence audit.",
        "",
        "## Walk-forward validation",
        "",
        validation.to_markdown(index=False),
        "",
        "## Formal gate",
        "",
        f"- Enabled: `{gate['enabled_for_portfolio_stage']}`",
        f"- Checks: `{json.dumps(gate['checks'], ensure_ascii=False)}`",
        "- 2024+ has already been observed during earlier strategy development and is not an untouched holdout.",
        "- PPO must not be credited with alpha unless this forecast gate and the later portfolio gate both pass.",
    ]
    project_path(cfg, "report_md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the hierarchical asset iTransformer v7 with annual walk-forward checkpoints.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--years", default=None, help="Comma-separated prediction years for a smoke or partial run.")
    parser.add_argument("--seeds", default=None, help="Comma-separated seed override.")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument(
        "--pretrain-max-epochs",
        type=int,
        default=None,
        help="Optional masked-history pretraining epoch override for smoke tests.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse compatible completed seed checkpoints and rebuild their validation/OOS predictions.",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    torch.set_num_threads(int(cfg["model"].get("torch_threads", torch.get_num_threads())))
    if "training_progress" in cfg.get("outputs", {}) and not args.resume:
        project_path(cfg, "training_progress").write_text("", encoding="utf-8")
    if args.seeds:
        cfg["model"]["seeds"] = [int(value) for value in args.seeds.split(",")]
    if args.max_epochs:
        cfg["model"]["max_epochs"] = int(args.max_epochs)
    if args.pretrain_max_epochs:
        pre_cfg = cfg["model"].setdefault("self_supervised_pretraining", {})
        pre_cfg["max_epochs"] = int(args.pretrain_max_epochs)
        pre_cfg["minimum_epochs"] = min(
            int(pre_cfg.get("minimum_epochs", args.pretrain_max_epochs)),
            int(args.pretrain_max_epochs),
        )
    arrays, metadata = load_arrays(cfg)
    run_signature = training_signature(cfg)
    dates = pd.to_datetime(arrays.dates)
    all_years = sorted(set(dates[dates >= pd.Timestamp(cfg["data"]["prediction_start"])].year))
    years = all_years if not args.years else [int(value) for value in args.years.split(",")]
    device = torch.device("cpu")
    model_dir = project_path(cfg, "model_dir")
    model_dir.mkdir(parents=True, exist_ok=True)
    existing_training_log = pd.DataFrame()
    existing_progress_log = pd.DataFrame()
    training_log_path = project_path(cfg, "training_log")
    if args.resume and training_log_path.exists():
        existing_training_log = pd.read_csv(training_log_path)
    if args.resume and "training_progress" in cfg.get("outputs", {}):
        progress_path = project_path(cfg, "training_progress")
        if progress_path.exists():
            progress_rows: list[dict[str, Any]] = []
            for line in progress_path.read_text(encoding="utf-8").splitlines():
                try:
                    progress_rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
            existing_progress_log = pd.DataFrame(progress_rows)
    training_logs: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    asset_rows: list[dict[str, Any]] = []
    sleeve_rows: list[dict[str, Any]] = []
    historical_oos_predictions: list[dict[str, np.ndarray]] = []
    sequence_length = int(metadata["sequence_length_days"])
    candidate_indices = arrays.sample_indices
    train_stride = int(cfg["data"]["train_stride_days"])
    configured_horizons = [
        int(value) for value in metadata["horizons_days"]
    ]
    active_horizon_days: set[int] = set()
    for task in ("asset", "sleeve"):
        enabled = cfg["model"].get(f"{task}_enabled_horizons_days")
        if enabled is not None:
            active_horizon_days.update(int(value) for value in enabled)
            continue
        weights = cfg["model"].get(
            f"{task}_horizon_loss_weights",
            cfg["model"].get("horizon_loss_weights"),
        )
        if weights is None:
            active_horizon_days.update(configured_horizons)
        else:
            active_horizon_days.update(
                horizon
                for horizon, weight in zip(configured_horizons, weights)
                if float(weight) > 0.0
            )
    active_horizon_positions = np.asarray(
        [
            position
            for position, horizon in enumerate(configured_horizons)
            if horizon in active_horizon_days
        ],
        dtype=np.int64,
    )
    if not len(active_horizon_positions):
        raise ValueError("At least one forecast horizon must be enabled")
    split_horizon_days = {
        int(value)
        for value in cfg["walk_forward"].get(
            "split_horizons_days",
            sorted(active_horizon_days),
        )
    }
    unknown_split_horizons = split_horizon_days.difference(
        configured_horizons
    )
    if unknown_split_horizons:
        raise ValueError(
            "walk_forward.split_horizons_days contains unknown horizons: "
            f"{sorted(unknown_split_horizons)}"
        )
    split_horizon_positions = np.asarray(
        [
            position
            for position, horizon in enumerate(configured_horizons)
            if horizon in split_horizon_days
        ],
        dtype=np.int64,
    )
    if not len(split_horizon_positions):
        raise ValueError(
            "walk_forward.split_horizons_days must contain a forecast horizon"
        )
    purge_horizon_positions = np.unique(
        np.concatenate(
            [active_horizon_positions, split_horizon_positions]
        )
    )
    require_all_split_horizons = bool(
        cfg["walk_forward"].get(
            "require_all_split_horizons_mature",
            False,
        )
    )
    for year in years:
        cutoff = pd.Timestamp(year, 1, 1)
        cutoff_ns = int(cutoff.value)
        eligible = candidate_indices[dates[candidate_indices] < cutoff]
        eligible = eligible[(np.arange(len(eligible)) % max(train_stride, 1)) == 0]
        maturity_reducer = (
            np.all if require_all_split_horizons else np.any
        )
        mature = np.array(
            [
                maturity_reducer(
                    (
                        arrays.label_end[index, split_horizon_positions]
                        > np.iinfo(np.int64).min
                    )
                    & (
                        arrays.label_end[index, split_horizon_positions]
                        < cutoff_ns
                    )
                )
                for index in eligible
            ]
        )
        eligible = eligible[mature]
        minimum = int(cfg["walk_forward"]["minimum_train_samples"])
        if len(eligible) < minimum:
            print(f"skip year={year}: only {len(eligible)} mature samples", flush=True)
            continue
        split = int(len(eligible) * (1.0 - float(cfg["walk_forward"]["validation_fraction"])))
        split = min(max(split, minimum), len(eligible) - int(cfg["walk_forward"]["minimum_validation_samples"]))
        preliminary_train = eligible[:split]
        val_indices = eligible[split:]
        validation_start_ns = int(dates[val_indices[0]].value)
        if bool(cfg["walk_forward"].get("purge_by_max_label_end", True)):
            train_keep = np.array(
                [
                    np.all(
                        (
                            arrays.label_end[
                                index, purge_horizon_positions
                            ]
                            <= np.iinfo(np.int64).min
                        )
                        | (
                            arrays.label_end[
                                index, purge_horizon_positions
                            ]
                            < validation_start_ns
                        )
                    )
                    for index in preliminary_train
                ]
            )
            train_indices = preliminary_train[train_keep]
        else:
            train_indices = preliminary_train
        if len(train_indices) < minimum:
            print(
                f"skip year={year}: only {len(train_indices)} samples after validation purge",
                flush=True,
            )
            continue
        checkpoints: list[dict[str, Any]] = []
        validation_predictions: list[dict[str, np.ndarray]] = []
        for seed in cfg["model"]["seeds"]:
            checkpoint_path = model_dir / f"model_{year}_seed_{seed}.pt"
            resumed = False
            if args.resume and checkpoint_path.exists():
                try:
                    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                    if checkpoint.get("training_signature") != run_signature:
                        raise ValueError(
                            "checkpoint training signature does not match current "
                            "dataset/config/source"
                        )
                    if int(checkpoint.get("year", -1)) != int(year) or int(
                        checkpoint.get("seed", -1)
                    ) != int(seed):
                        raise ValueError("checkpoint year/seed identity mismatch")
                    model = make_model(arrays, metadata, cfg).to(device)
                    model.load_state_dict(checkpoint["state_dict"])
                    validation_loader = DataLoader(
                        WindowDataset(
                            arrays,
                            val_indices,
                            sequence_length,
                            cutoff_ns,
                        ),
                        batch_size=int(cfg["model"]["batch_size"]),
                        shuffle=False,
                    )
                    validation = predict(model, validation_loader, device, cfg)
                    checkpoints.append(checkpoint)
                    validation_predictions.append(validation)
                    previous_rows = pd.DataFrame()
                    if not existing_training_log.empty:
                        previous_year = pd.to_numeric(
                            existing_training_log.get("year"), errors="coerce"
                        )
                        previous_seed = pd.to_numeric(
                            existing_training_log.get("seed"), errors="coerce"
                        )
                        previous_rows = existing_training_log.loc[
                            previous_year.eq(year) & previous_seed.eq(int(seed))
                        ]
                    if previous_rows.empty and not existing_progress_log.empty:
                        progress_year = pd.to_numeric(
                            existing_progress_log.get("year"), errors="coerce"
                        )
                        progress_seed = pd.to_numeric(
                            existing_progress_log.get("seed"), errors="coerce"
                        )
                        previous_rows = existing_progress_log.loc[
                            progress_year.eq(year) & progress_seed.eq(int(seed))
                        ].copy()
                        if not previous_rows.empty:
                            previous_rows["epoch_num"] = pd.to_numeric(
                                previous_rows["epoch"], errors="coerce"
                            )
                            previous_rows = (
                                previous_rows.sort_values("epoch_num")
                                .drop_duplicates("epoch_num", keep="last")
                                .drop(columns="epoch_num")
                            )
                    if not previous_rows.empty:
                        training_logs.extend(previous_rows.to_dict("records"))
                        if not previous_rows["epoch"].astype(str).eq("checkpoint_selection").any():
                            training_logs.append(
                                {
                                    "year": year,
                                    "seed": int(seed),
                                    "epoch": "checkpoint_selection",
                                    "best_epoch": int(checkpoint.get("best_epoch", 0)),
                                    "best_validation_loss": float(
                                        checkpoint.get("best_validation_loss", np.nan)
                                    ),
                                    "best_validation_checkpoint_objective": float(
                                        checkpoint.get(
                                            "best_validation_checkpoint_objective", np.nan
                                        )
                                    ),
                                    "selected_validation_loss": float(
                                        checkpoint.get("selected_validation_loss", np.nan)
                                    ),
                                    "selected_checkpoint_rank_score": float(
                                        checkpoint.get("selected_checkpoint_rank_score", np.nan)
                                    ),
                                    "checkpoint_selection": checkpoint.get(
                                        "checkpoint_selection", "single_best_validation_loss"
                                    ),
                                    "averaged_epochs": ",".join(
                                        str(value)
                                        for value in checkpoint.get("averaged_epochs", [])
                                    ),
                                    "epochs_run": int(
                                        checkpoint.get(
                                            "epochs_run",
                                            min(
                                                int(cfg["model"]["max_epochs"]),
                                                int(checkpoint.get("best_epoch", 0))
                                                + int(cfg["model"].get("patience", 0)),
                                            ),
                                        )
                                    ),
                                }
                            )
                    else:
                        training_logs.append(
                            {
                                "year": year,
                                "seed": int(seed),
                                "epoch": int(checkpoint.get("best_epoch", 0)),
                                "validation_loss": float(checkpoint.get("best_validation_loss", np.nan)),
                                "validation_selection_score": float(
                                    checkpoint.get("selected_checkpoint_rank_score", np.nan)
                                ),
                                "best_validation_selection_score": float(
                                    checkpoint.get("best_validation_selection_score", np.nan)
                                ),
                                "selected_checkpoint_rank_score": float(
                                    checkpoint.get("selected_checkpoint_rank_score", np.nan)
                                ),
                                "best_validation_loss": float(checkpoint.get("best_validation_loss", np.nan)),
                                "best_epoch": int(checkpoint.get("best_epoch", 0)),
                                "resumed_from_checkpoint": True,
                            }
                        )
                    print(
                        f"[year={year}][seed={seed}] resumed checkpoint "
                        f"best_epoch={checkpoint.get('best_epoch')}",
                        flush=True,
                    )
                    resumed = True
                except (KeyError, RuntimeError, ValueError) as exc:
                    print(
                        f"[year={year}][seed={seed}] incompatible checkpoint; retraining: {exc}",
                        flush=True,
                    )
            if resumed:
                continue
            print(f"[year={year}][seed={seed}] train={len(train_indices)} validation={len(val_indices)}", flush=True)
            checkpoint, logs, validation = train_seed(
                int(seed), year, arrays, metadata, cfg, train_indices, val_indices,
                validation_start_ns, cutoff_ns, device
            )
            checkpoints.append(checkpoint)
            training_logs.extend(logs)
            validation_predictions.append(validation)
            checkpoint["training_signature"] = run_signature
            torch.save(checkpoint, checkpoint_path)
        ensemble_validation = merge_seed_predictions(validation_predictions)
        validation_policies, validation_blend_rows = choose_blend_policies(
            ensemble_validation,
            [float(value) for value in cfg["model"]["baseline_weight_grid"]],
            [float(value) for value in cfg["model"]["residual_alpha_grid"]],
            arrays.module_index,
            arrays,
            cfg,
        )
        blend_source = str(
            cfg["model"].get("blend_selection_source", "current_purged_validation")
        )
        policies = validation_policies
        blend_rows = validation_blend_rows
        calibration_years: list[int] = []
        if blend_source == "prior_walkforward_oos":
            calibration = mature_historical_predictions(
                historical_oos_predictions, arrays, cutoff_ns
            )
            if calibration is not None:
                calibration_dates = pd.to_datetime(
                    arrays.dates[calibration["index"].astype(np.int64)]
                )
                calibration_years = sorted(set(calibration_dates.year.tolist()))
            minimum_calibration_years = int(
                cfg["model"].get("blend_min_calibration_years", 2)
            )
            if calibration is not None and len(calibration_years) >= minimum_calibration_years:
                policies, blend_rows = choose_blend_policies(
                    calibration,
                    [float(value) for value in cfg["model"]["baseline_weight_grid"]],
                    [float(value) for value in cfg["model"]["residual_alpha_grid"]],
                    arrays.module_index,
                    arrays,
                    cfg,
                    require_incremental_confidence=bool(
                        cfg["model"].get("blend_bootstrap_enabled", False)
                    ),
                )
            else:
                fallback_baseline_weight = float(
                    cfg["model"].get(
                        "deployment_fallback_baseline_weight",
                        (
                            0.0
                            if str(
                                cfg.get("data", {}).get(
                                    "forecast_baseline_mode",
                                    "causal_anchor",
                                )
                            )
                            == "zero_active_view"
                            else 1.0
                        ),
                    )
                )
                policies, blend_rows = choose_blend_policies(
                    ensemble_validation,
                    [fallback_baseline_weight],
                    [0.0],
                    arrays.module_index,
                    arrays,
                    cfg,
                    use_task_specific_grids=False,
                )
        elif blend_source != "current_purged_validation":
            raise ValueError(f"Unknown blend_selection_source={blend_source}")
        for row in blend_rows:
            training_logs.append(
                {
                    "year": year,
                    "seed": "ensemble",
                    "epoch": "blend",
                    "blend_selection_source": blend_source,
                    "calibration_years": ",".join(
                        str(value) for value in calibration_years
                    ),
                    **row,
                }
            )
        prediction_indices = candidate_indices[(dates[candidate_indices].year == year) & (dates[candidate_indices] >= cutoff)]
        if not len(prediction_indices):
            continue
        prediction_loader = DataLoader(
            WindowDataset(arrays, prediction_indices, sequence_length),
            batch_size=int(cfg["model"]["batch_size"]),
            shuffle=False,
        )
        seed_oos: list[dict[str, np.ndarray]] = []
        for checkpoint in checkpoints:
            model = make_model(arrays, metadata, cfg).to(device)
            model.load_state_dict(checkpoint["state_dict"])
            seed_oos.append(predict(model, prediction_loader, device, cfg))
        ensemble_oos = merge_seed_predictions(seed_oos)
        append_long_predictions(
            asset_rows,
            sleeve_rows,
            ensemble_oos,
            arrays,
            metadata,
            policies,
            year,
            cfg,
        )
        validation_rows.extend(
            metric_rows(
                ensemble_oos,
                metadata,
                arrays.module_index,
                policies,
                year,
                cfg,
            )
        )
        historical_oos_predictions.append(ensemble_oos)
        print(
            json.dumps(
                {
                    "year": year,
                    "blend_selection_source": blend_source,
                    "calibration_years": calibration_years,
                    "blend_policies": compact_blend_policy(
                        policies,
                        [int(value) for value in metadata["horizons_days"]],
                    ),
                    "prediction_samples": len(prediction_indices),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    if not asset_rows or not sleeve_rows:
        raise RuntimeError("No walk-forward predictions were produced")
    asset_predictions = pd.DataFrame(asset_rows).sort_values(["as_of_date", "horizon_days", "module", "asset_code"])
    sleeve_predictions = pd.DataFrame(sleeve_rows).sort_values(["as_of_date", "horizon_days", "sleeve"])
    validation = pd.DataFrame(validation_rows)
    pd.DataFrame(training_logs).to_csv(project_path(cfg, "training_log"), index=False, encoding="utf-8-sig")
    validation.to_csv(project_path(cfg, "validation_report"), index=False, encoding="utf-8-sig")
    asset_predictions.to_parquet(project_path(cfg, "asset_predictions"), index=False)
    sleeve_predictions.to_parquet(project_path(cfg, "sleeve_predictions"), index=False)
    if "expert_usage" in cfg.get("outputs", {}):
        usage = [
            *expert_usage_rows(
                asset_predictions,
                "asset",
                metadata.get(
                    "asset_signal_experts", ["fixed_causal_baseline"]
                ),
            ),
            *expert_usage_rows(
                sleeve_predictions,
                "sleeve",
                metadata.get(
                    "sleeve_signal_experts", ["fixed_causal_baseline"]
                ),
            ),
        ]
        pd.DataFrame(usage).to_csv(
            project_path(cfg, "expert_usage"),
            index=False,
            encoding="utf-8-sig",
        )

    if str(cfg["gate"].get("mode", "legacy_global")) == "component_policy":
        gate = build_component_policy_gate(
            validation,
            asset_predictions,
            sleeve_predictions,
            cfg,
        )
    else:
        evaluation = validation.loc[
            validation["year"].ge(
                pd.Timestamp(cfg["gate"]["evaluation_start"]).year
            )
        ]
        summary = {
            "asset_rank_ic": float(evaluation["asset_model_rank_ic"].mean()),
            "asset_incremental_rank_ic": float(
                evaluation["asset_incremental_rank_ic"].mean()
            ),
            "sleeve_rank_ic": float(
                evaluation["sleeve_model_rank_ic"].mean()
            ),
            "sleeve_incremental_rank_ic": float(
                evaluation["sleeve_incremental_rank_ic"].mean()
            ),
            "positive_year_fraction": float(
                validation.groupby("year")[
                    [
                        "asset_incremental_rank_ic",
                        "sleeve_incremental_rank_ic",
                    ]
                ]
                .mean()
                .gt(0.0)
                .all(axis=1)
                .mean()
            ),
        }
        checks = {
            "asset_rank_ic": summary["asset_rank_ic"]
            >= float(cfg["gate"]["min_asset_rank_ic"]),
            "sleeve_rank_ic": summary["sleeve_rank_ic"]
            >= float(cfg["gate"]["min_sleeve_rank_ic"]),
            "asset_incremental": summary["asset_incremental_rank_ic"]
            >= float(cfg["gate"]["min_asset_incremental_rank_ic"]),
            "sleeve_incremental": summary["sleeve_incremental_rank_ic"]
            >= float(cfg["gate"]["min_sleeve_incremental_rank_ic"]),
            "year_stability": summary["positive_year_fraction"]
            >= float(cfg["gate"]["min_positive_year_fraction"]),
        }
        gate = {
            "enabled_for_portfolio_stage": bool(all(checks.values())),
            "enabled_for_formal_strategy": bool(all(checks.values())),
            "checks": checks,
            "summary": summary,
            "caveat": (
                "2024+ has already been observed during earlier strategy "
                "development."
            ),
        }
    project_path(cfg, "gate_report").write_text(json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8")
    parameter_count = sum(
        parameter.numel()
        for parameter in make_model(arrays, metadata, cfg).parameters()
        if parameter.requires_grad
    )
    write_report(cfg, metadata, validation, gate, parameter_count)
    print(json.dumps(gate, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

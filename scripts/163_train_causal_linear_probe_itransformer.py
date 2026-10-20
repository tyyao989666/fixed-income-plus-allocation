from __future__ import annotations

"""Train strict walk-forward iTransformer representations with causal probes.

The iTransformer is trained only by masked-history reconstruction.  Frozen
states are mapped to annualized relative mean views by deterministic ridge
probes.  A purged pre-year validation block and, once available, matured prior
walk-forward forecasts decide whether each module/horizon is active.  A failed
gate emits an exact zero learned view.
"""

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.hierarchical_asset_itransformer import (  # noqa: E402
    RISK_SLEEVES,
    load_arrays,
    load_config,
    project_path,
    rank_ic,
)


def load_python_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalized_config_hash(
    cfg: dict[str, Any],
    train_indices: np.ndarray,
    metadata: dict[str, Any],
) -> str:
    dataset_path = ROOT / cfg["outputs"]["dataset_npz"]
    dataset_stat = dataset_path.stat()
    payload = {
        "version": cfg["version"],
        "model": {
            key: value
            for key, value in cfg["model"].items()
            if key != "causal_linear_probe"
        },
        "data": cfg["data"],
        "walk_forward": cfg["walk_forward"],
        "dataset": {
            "path": str(dataset_path.relative_to(ROOT)),
            "size": int(dataset_stat.st_size),
            "mtime_ns": int(dataset_stat.st_mtime_ns),
        },
        "metadata": {
            "assets": metadata["assets"],
            "asset_modules": metadata["asset_modules"],
            "horizons_days": metadata["horizons_days"],
            "sequence_length_days": metadata["sequence_length_days"],
            "local_features": metadata.get("local_features", []),
            "macro_features": metadata.get("macro_features", []),
        },
        "train_indices_sha256": hashlib.sha256(
            np.asarray(train_indices, dtype=np.int64).tobytes()
        ).hexdigest(),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def encoder_architecture_hash(
    cfg: dict[str, Any],
    metadata: dict[str, Any],
) -> str:
    """Fingerprint only the causal representation contract for warm starts."""
    payload = {
        "version": cfg["version"],
        "model": {
            key: value
            for key, value in cfg["model"].items()
            if key != "causal_linear_probe"
        },
        "data": cfg["data"],
        "metadata": {
            "assets": metadata["assets"],
            "asset_modules": metadata["asset_modules"],
            "horizons_days": metadata["horizons_days"],
            "sequence_length_days": metadata["sequence_length_days"],
            "local_features": metadata.get("local_features", []),
            "macro_features": metadata.get("macro_features", []),
        },
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def pretraining_schedule(cfg: dict[str, Any]) -> dict[str, Any]:
    pre_cfg = cfg["model"].get("self_supervised_pretraining", {})
    return {
        key: pre_cfg.get(key)
        for key in (
            "enabled",
            "mask_mode",
            "full_path_mask_fraction",
            "causal_suffix_days",
            "max_epochs",
            "minimum_epochs",
            "patience",
            "min_delta",
            "learning_rate",
            "weight_decay",
            "validation_fraction",
            "validation_purge_days",
        )
    }


def center_cross_section(values: np.ndarray) -> np.ndarray:
    centered = np.asarray(values, dtype=np.float64)
    centered = centered - centered.mean(axis=1, keepdims=True)
    return centered.astype(np.float32)


def center_cross_section_masked(
    values: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    valid = np.asarray(mask, dtype=bool)
    weights = valid.astype(np.float64)
    count = np.maximum(weights.sum(axis=1, keepdims=True), 1.0)
    mean = (values * weights).sum(axis=1, keepdims=True) / count
    return np.where(valid, values - mean, 0.0).astype(np.float32)


def entity_target_mean(
    target: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    values = np.asarray(target, dtype=np.float64)
    valid = np.asarray(mask, dtype=bool)
    count = valid.sum(axis=0)
    return np.divide(
        (values * valid).sum(axis=0),
        count,
        out=np.zeros(values.shape[1], dtype=np.float64),
        where=count > 0,
    )


def entity_state_mean(
    state: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    values = np.asarray(state, dtype=np.float64)
    valid = np.asarray(mask, dtype=bool)
    count = valid.sum(axis=0)[:, None]
    return np.divide(
        (values * valid[..., None]).sum(axis=0),
        count,
        out=np.zeros(values.shape[1:], dtype=np.float64),
        where=count > 0,
    )


def dynamic_target(
    target: np.ndarray,
    mask: np.ndarray,
    fixed_effect: np.ndarray,
) -> np.ndarray:
    residual = np.asarray(target, dtype=np.float64) - np.asarray(
        fixed_effect, dtype=np.float64
    )[None, :]
    return center_cross_section_masked(residual, mask)


def robust_residual_scale(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    floor: float = 0.02,
) -> float:
    residual = (target - prediction)[np.asarray(mask, dtype=bool)]
    residual = residual[np.isfinite(residual)]
    if not len(residual):
        return float(floor)
    median = float(np.median(residual))
    scale = 1.4826 * float(np.median(np.abs(residual - median)))
    if not np.isfinite(scale) or scale <= 1e-8:
        scale = float(np.sqrt(np.mean(np.square(residual))))
    return float(max(scale, floor))


def fit_fixed_ridge_probe(
    train_state: np.ndarray,
    train_target: np.ndarray,
    train_mask: np.ndarray,
    validation_state: np.ndarray,
    validation_target: np.ndarray,
    validation_mask: np.ndarray,
    oos_state: np.ndarray,
    probe_cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, float, dict[str, Any]]:
    valid = np.asarray(train_mask, dtype=bool).reshape(-1)
    feature_count = int(train_state.shape[-1])
    remove_fixed_effect = bool(
        probe_cfg.get("remove_entity_fixed_effect", True)
    )
    center_representation = bool(
        probe_cfg.get("center_entity_representation", True)
    )
    target_fixed_effect = (
        entity_target_mean(train_target, train_mask)
        if remove_fixed_effect
        else np.zeros(train_target.shape[1], dtype=np.float64)
    )
    state_fixed_effect = (
        entity_state_mean(train_state, train_mask)
        if center_representation
        else np.zeros(train_state.shape[1:], dtype=np.float64)
    )
    train_state_dynamic = (
        np.asarray(train_state, dtype=np.float64)
        - state_fixed_effect[None, :, :]
    )
    validation_state_dynamic = (
        np.asarray(validation_state, dtype=np.float64)
        - state_fixed_effect[None, :, :]
    )
    oos_state_dynamic = (
        np.asarray(oos_state, dtype=np.float64)
        - state_fixed_effect[None, :, :]
    )
    train_target_dynamic = dynamic_target(
        train_target,
        train_mask,
        target_fixed_effect,
    )
    validation_target_dynamic = dynamic_target(
        validation_target,
        validation_mask,
        target_fixed_effect,
    )
    minimum = max(
        int(probe_cfg.get("minimum_training_cells", 120)),
        2 * feature_count,
    )
    if int(valid.sum()) < minimum:
        validation = np.zeros(validation_target.shape, dtype=np.float32)
        oos = np.zeros(oos_state.shape[:2], dtype=np.float32)
        return validation, oos, 1.0, {
            "trained": False,
            "reason": "insufficient_training_cells",
            "training_cells": int(valid.sum()),
            "target_entity_fixed_effect": (
                target_fixed_effect.astype(float).tolist()
            ),
            "state_entity_fixed_effect": (
                state_fixed_effect.astype(float).tolist()
            ),
        }

    train_flat = train_state_dynamic.reshape(
        -1, feature_count
    )[valid]
    target_flat = train_target_dynamic.reshape(-1)[valid]
    sample_weight: np.ndarray | None = None
    if bool(probe_cfg.get("equal_date_weighting", True)):
        valid_matrix = np.asarray(train_mask, dtype=bool)
        available_per_date = np.maximum(
            valid_matrix.sum(axis=1, keepdims=True), 1
        )
        cell_weight = (
            valid_matrix.astype(np.float64) / available_per_date
        ).reshape(-1)[valid]
        sample_weight = cell_weight / max(
            float(cell_weight.mean()), 1e-12
        )
    scaler = StandardScaler().fit(
        train_flat, sample_weight=sample_weight
    )
    train_scaled = scaler.transform(train_flat)
    alpha = float(probe_cfg.get("alpha_per_training_cell", 0.30)) * len(
        train_scaled
    )
    estimator = Ridge(
        alpha=alpha,
        fit_intercept=True,
        solver=str(probe_cfg.get("solver", "lsqr")),
        tol=1e-8,
    )
    estimator.fit(
        train_scaled, target_flat, sample_weight=sample_weight
    )
    validation = estimator.predict(
        scaler.transform(
            validation_state_dynamic.reshape(-1, feature_count)
        )
    ).reshape(validation_target.shape)
    oos = estimator.predict(
        scaler.transform(
            oos_state_dynamic.reshape(-1, feature_count)
        )
    ).reshape(oos_state.shape[:2])
    clip = float(probe_cfg.get("prediction_clip", 0.60))
    validation = np.clip(
        center_cross_section(validation), -clip, clip
    )
    oos = np.clip(center_cross_section(oos), -clip, clip)
    scale = robust_residual_scale(
        validation,
        validation_target_dynamic,
        validation_mask,
    )
    artifact = {
        "trained": True,
        "training_cells": int(valid.sum()),
        "feature_count": feature_count,
        "alpha": float(alpha),
        "equal_date_weighting": bool(sample_weight is not None),
        "effective_weight_sum": float(
            sample_weight.sum()
            if sample_weight is not None
            else len(train_scaled)
        ),
        "intercept": float(estimator.intercept_),
        "coefficient": estimator.coef_.astype(float).tolist(),
        "scaler_mean": scaler.mean_.astype(float).tolist(),
        "scaler_scale": scaler.scale_.astype(float).tolist(),
        "validation_residual_scale": float(scale),
        "remove_entity_fixed_effect": remove_fixed_effect,
        "center_entity_representation": center_representation,
        "target_entity_fixed_effect": (
            target_fixed_effect.astype(float).tolist()
        ),
        "state_entity_fixed_effect": (
            state_fixed_effect.astype(float).tolist()
        ),
    }
    return (
        validation.astype(np.float32),
        oos.astype(np.float32),
        float(scale),
        artifact,
    )


def daily_rank_values(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    values = np.full(len(prediction), np.nan, dtype=float)
    for row in range(len(prediction)):
        values[row] = rank_ic(
            prediction[row : row + 1],
            target[row : row + 1],
            mask[row : row + 1],
            None,
        )
    return values


def positive_block_fraction(
    daily_values: np.ndarray,
    horizon_days: int,
    minimum_days: int,
) -> tuple[float, int]:
    finite = np.asarray(daily_values, dtype=float)
    block = max(int(horizon_days), int(minimum_days))
    blocks = [
        finite[start : start + block]
        for start in range(0, len(finite), block)
    ]
    means = np.asarray(
        [
            np.nanmean(values)
            for values in blocks
            if np.isfinite(values).any()
        ],
        dtype=float,
    )
    if not len(means):
        return 0.0, 0
    return float(np.mean(means > 0.0)), int(len(means))


def seed_prediction_agreement(
    predictions: list[np.ndarray],
    mask: np.ndarray,
) -> float:
    values: list[float] = []
    for left in range(len(predictions)):
        for right in range(left + 1, len(predictions)):
            score = rank_ic(
                predictions[left],
                predictions[right],
                mask,
                None,
            )
            if np.isfinite(score):
                values.append(float(score))
    return float(np.mean(values)) if values else 0.0


def component_history(
    history: list[dict[str, Any]],
    task: str,
    horizon_position: int,
    columns: np.ndarray | None,
    cutoff_ns: int,
) -> tuple[list[int], list[float]]:
    years: list[int] = []
    scores: list[float] = []
    for item in history:
        bundle = item["bundle"]
        indices = bundle["index"].astype(np.int64)
        label_end = item["label_end"][indices, horizon_position]
        mature = (
            (label_end > np.iinfo(np.int64).min)
            & (label_end < int(cutoff_ns))
        )
        if not mature.any():
            continue
        prediction = bundle[f"{task}_residual"][mature, horizon_position]
        target = bundle[f"{task}_target"][mature, horizon_position]
        mask = bundle[f"{task}_mask"][mature, horizon_position]
        if columns is not None:
            prediction = prediction[:, columns]
            target = target[:, columns]
            mask = mask[:, columns]
        score = rank_ic(prediction, target, mask, None)
        if np.isfinite(score):
            years.append(int(item["year"]))
            scores.append(float(score))
    return years, scores


def gate_component(
    task: str,
    horizon_days: int,
    seed_validation_scores: list[float],
    agreement: float,
    ensemble_validation_score: float,
    block_fraction: float,
    history_years: list[int],
    history_scores: list[float],
    probe_cfg: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    minimum_rank = float(
        probe_cfg["minimum_validation_rank_ic"][task]
    )
    minimum_seed_fraction = float(
        probe_cfg["minimum_seed_positive_fraction"][task]
    )
    minimum_agreement = float(
        probe_cfg["minimum_seed_prediction_agreement"][task]
    )
    minimum_block_fraction = float(
        probe_cfg["minimum_positive_block_fraction"][task]
    )
    finite_seed = np.asarray(seed_validation_scores, dtype=float)
    finite_seed = finite_seed[np.isfinite(finite_seed)]
    seed_positive_fraction = (
        float(np.mean(finite_seed > 0.0)) if len(finite_seed) else 0.0
    )
    validation_passed = bool(
        np.isfinite(ensemble_validation_score)
        and ensemble_validation_score >= minimum_rank
        and seed_positive_fraction >= minimum_seed_fraction
        and agreement >= minimum_agreement
        and block_fraction >= minimum_block_fraction
    )

    minimum_years = int(probe_cfg["prior_oos_minimum_years"])
    prior_available = len(set(history_years)) >= minimum_years
    prior_mean = (
        float(np.mean(history_scores)) if history_scores else np.nan
    )
    prior_positive_fraction = (
        float(np.mean(np.asarray(history_scores) > 0.0))
        if history_scores
        else 0.0
    )
    prior_passed = bool(
        prior_available
        and np.isfinite(prior_mean)
        and prior_mean
        >= float(probe_cfg["prior_oos_minimum_rank_ic"][task])
        and prior_positive_fraction
        >= float(
            probe_cfg["prior_oos_minimum_positive_year_fraction"]
        )
    )
    initial_allowed = bool(
        probe_cfg.get(f"{task}_initial_gate_enabled", False)
    )
    if bool(probe_cfg.get("prior_oos_gate_enabled", True)) and prior_available:
        active = validation_passed and prior_passed
        source = "purged_validation_and_prior_walkforward_oos"
    else:
        active = validation_passed and initial_allowed
        source = "purged_validation_initial_gate"
    diagnostics = {
        "task": task,
        "horizon_days": int(horizon_days),
        "active": bool(active),
        "gate_source": source,
        "ensemble_validation_rank_ic": float(
            ensemble_validation_score
        ),
        "seed_validation_rank_ic": [
            float(value) for value in seed_validation_scores
        ],
        "seed_positive_fraction": seed_positive_fraction,
        "seed_prediction_agreement": float(agreement),
        "positive_validation_block_fraction": float(block_fraction),
        "prior_oos_years": sorted(set(history_years)),
        "prior_oos_rank_ic_by_year": [
            float(value) for value in history_scores
        ],
        "prior_oos_mean_rank_ic": prior_mean,
        "prior_oos_positive_year_fraction": prior_positive_fraction,
        "validation_passed": validation_passed,
        "prior_oos_available": prior_available,
        "prior_oos_passed": prior_passed,
    }
    return active, diagnostics


def audit_fold_contract(
    arrays,
    metadata: dict[str, Any],
    cfg: dict[str, Any],
    year: int,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    oos_indices: np.ndarray,
) -> list[dict[str, Any]]:
    dates = pd.to_datetime(arrays.dates)
    cutoff = pd.Timestamp(year, 1, 1)
    validation_start = dates[validation_indices[0]]
    horizons = [int(value) for value in metadata["horizons_days"]]
    split_horizons = {
        int(value)
        for value in cfg["walk_forward"].get(
            "split_horizons_days", horizons
        )
    }
    split_positions = np.asarray(
        [
            position
            for position, horizon in enumerate(horizons)
            if horizon in split_horizons
        ],
        dtype=np.int64,
    )
    minimum_ns = np.iinfo(np.int64).min

    rows: list[dict[str, Any]] = []

    def add(check: str, passed: bool, detail: str) -> None:
        rows.append(
            {
                "year": int(year),
                "check": check,
                "passed": bool(passed),
                "detail": detail,
            }
        )

    train_set = set(np.asarray(train_indices, dtype=int).tolist())
    validation_set = set(
        np.asarray(validation_indices, dtype=int).tolist()
    )
    oos_set = set(np.asarray(oos_indices, dtype=int).tolist())
    add(
        "fold_indices_disjoint",
        not (
            train_set & validation_set
            or train_set & oos_set
            or validation_set & oos_set
        ),
        (
            f"train={len(train_set)}, validation={len(validation_set)}, "
            f"oos={len(oos_set)}"
        ),
    )
    add(
        "chronological_fold_order",
        bool(
            dates[train_indices].max() < validation_start
            and dates[validation_indices].max() < cutoff
            and np.all(dates[oos_indices].year == int(year))
        ),
        (
            f"train_end={dates[train_indices].max().date()}, "
            f"validation={validation_start.date()}.."
            f"{dates[validation_indices].max().date()}, "
            f"oos={dates[oos_indices].min().date()}.."
            f"{dates[oos_indices].max().date()}"
        ),
    )
    train_label_end = arrays.label_end[
        np.ix_(train_indices, split_positions)
    ]
    train_finite = train_label_end > minimum_ns
    add(
        "train_labels_purged_before_validation",
        bool(
            np.all(
                (~train_finite)
                | (train_label_end < int(validation_start.value))
            )
        ),
        f"validation_start={validation_start.date()}",
    )
    validation_label_end = arrays.label_end[
        np.ix_(validation_indices, split_positions)
    ]
    validation_finite = validation_label_end > minimum_ns
    add(
        "validation_labels_mature_before_oos",
        bool(
            np.all(
                (~validation_finite)
                | (validation_label_end < int(cutoff.value))
            )
        ),
        f"oos_cutoff={cutoff.date()}",
    )
    sequence_length = int(metadata["sequence_length_days"])
    all_indices = np.concatenate(
        [train_indices, validation_indices, oos_indices]
    )
    add(
        "history_windows_end_at_as_of",
        bool(
            np.all(all_indices - sequence_length + 1 >= 0)
            and np.all(np.diff(dates.values.astype(np.int64)) > 0)
        ),
        (
            f"lookback={sequence_length}, "
            f"minimum_window_start={int(all_indices.min()) - sequence_length + 1}"
        ),
    )
    expected_label_end = np.full_like(
        arrays.label_end[oos_indices], minimum_ns
    )
    for horizon_position, horizon in enumerate(horizons):
        valid = oos_indices + horizon < len(dates)
        expected_label_end[
            valid, horizon_position
        ] = dates[
            oos_indices[valid] + horizon
        ].values.astype("datetime64[ns]").astype(np.int64)
    actual_label_end = arrays.label_end[oos_indices]
    comparable = (
        (actual_label_end > minimum_ns)
        & (expected_label_end > minimum_ns)
    )
    add(
        "labels_end_at_t_plus_h",
        bool(
            comparable.any()
            and np.array_equal(
                actual_label_end[comparable],
                expected_label_end[comparable],
            )
        ),
        "targets use returns t+1 through t+h on the trading calendar",
    )
    return rows


def make_policies(
    year: int,
    validation_by_seed: list[dict[str, np.ndarray]],
    ensemble_validation: dict[str, np.ndarray],
    ensemble_oos: dict[str, np.ndarray],
    arrays,
    metadata: dict[str, Any],
    history: list[dict[str, Any]],
    probe_cfg: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    horizons = [int(value) for value in metadata["horizons_days"]]
    cutoff_ns = int(pd.Timestamp(year, 1, 1).value)
    policies: dict[str, Any] = {
        "asset": [],
        "sleeve": [],
        "asset_by_module": {module: [] for module in RISK_SLEEVES},
    }
    rows: list[dict[str, Any]] = []
    minimum_block_days = int(
        probe_cfg.get("validation_block_days_minimum", 20)
    )
    for horizon_position, horizon_days in enumerate(horizons):
        for module_position, module in enumerate(RISK_SLEEVES):
            columns = np.flatnonzero(
                arrays.module_index == module_position
            )
            seed_scores = [
                float(
                    rank_ic(
                        bundle["asset_residual"][
                            :, horizon_position, columns
                        ],
                        bundle["asset_target"][
                            :, horizon_position, columns
                        ],
                        bundle["asset_mask"][
                            :, horizon_position, columns
                        ],
                        None,
                    )
                )
                for bundle in validation_by_seed
            ]
            validation_prediction = ensemble_validation[
                "asset_residual"
            ][:, horizon_position, columns]
            validation_target = ensemble_validation["asset_target"][
                :, horizon_position, columns
            ]
            validation_mask = ensemble_validation["asset_mask"][
                :, horizon_position, columns
            ]
            validation_score = rank_ic(
                validation_prediction,
                validation_target,
                validation_mask,
                None,
            )
            static_validation_score = rank_ic(
                ensemble_validation["asset_static_prior"][
                    :, horizon_position, columns
                ],
                ensemble_validation["asset_original_target"][
                    :, horizon_position, columns
                ],
                validation_mask,
                None,
            )
            agreement = seed_prediction_agreement(
                [
                    bundle["asset_residual"][
                        :, horizon_position, columns
                    ]
                    for bundle in validation_by_seed
                ],
                validation_mask,
            )
            block_fraction, block_count = positive_block_fraction(
                daily_rank_values(
                    validation_prediction,
                    validation_target,
                    validation_mask,
                ),
                horizon_days,
                minimum_block_days,
            )
            history_years, history_scores = component_history(
                history,
                "asset",
                horizon_position,
                columns,
                cutoff_ns,
            )
            active, diagnostics = gate_component(
                "asset",
                horizon_days,
                seed_scores,
                agreement,
                validation_score,
                block_fraction,
                history_years,
                history_scores,
                probe_cfg,
            )
            policy = {
                "baseline_weight": 0.0,
                "residual_alpha": float(active),
                "selection_score": float(validation_score),
                "bootstrap_positive_probability": np.nan,
            }
            policies["asset_by_module"][module].append(policy)
            rows.append(
                {
                    "year": int(year),
                    "module": module,
                    "validation_block_count": block_count,
                    "static_fixed_effect_validation_rank_ic": float(
                        static_validation_score
                    ),
                    **diagnostics,
                }
            )

        sleeve_seed_scores = [
            float(
                rank_ic(
                    bundle["sleeve_residual"][:, horizon_position],
                    bundle["sleeve_target"][:, horizon_position],
                    bundle["sleeve_mask"][:, horizon_position],
                    None,
                )
            )
            for bundle in validation_by_seed
        ]
        sleeve_prediction = ensemble_validation["sleeve_residual"][
            :, horizon_position
        ]
        sleeve_target = ensemble_validation["sleeve_target"][
            :, horizon_position
        ]
        sleeve_mask = ensemble_validation["sleeve_mask"][
            :, horizon_position
        ]
        sleeve_score = rank_ic(
            sleeve_prediction, sleeve_target, sleeve_mask, None
        )
        sleeve_static_score = rank_ic(
            ensemble_validation["sleeve_static_prior"][
                :, horizon_position
            ],
            ensemble_validation["sleeve_original_target"][
                :, horizon_position
            ],
            sleeve_mask,
            None,
        )
        sleeve_agreement = seed_prediction_agreement(
            [
                bundle["sleeve_residual"][:, horizon_position]
                for bundle in validation_by_seed
            ],
            sleeve_mask,
        )
        sleeve_block_fraction, sleeve_block_count = (
            positive_block_fraction(
                daily_rank_values(
                    sleeve_prediction, sleeve_target, sleeve_mask
                ),
                horizon_days,
                minimum_block_days,
            )
        )
        sleeve_history_years, sleeve_history_scores = component_history(
            history,
            "sleeve",
            horizon_position,
            None,
            cutoff_ns,
        )
        sleeve_active, sleeve_diagnostics = gate_component(
            "sleeve",
            horizon_days,
            sleeve_seed_scores,
            sleeve_agreement,
            sleeve_score,
            sleeve_block_fraction,
            sleeve_history_years,
            sleeve_history_scores,
            probe_cfg,
        )
        policies["sleeve"].append(
            {
                "baseline_weight": 0.0,
                "residual_alpha": float(sleeve_active),
                "selection_score": float(sleeve_score),
                "bootstrap_positive_probability": np.nan,
            }
        )
        policies["asset"].append(
            {
                "baseline_weight": 0.0,
                "residual_alpha": float(
                    np.mean(
                        [
                            policies["asset_by_module"][module][
                                horizon_position
                            ]["residual_alpha"]
                            for module in RISK_SLEEVES
                        ]
                    )
                ),
                "selection_score": float(
                    np.mean(
                        [
                            policies["asset_by_module"][module][
                                horizon_position
                            ]["selection_score"]
                            for module in RISK_SLEEVES
                        ]
                    )
                ),
                "bootstrap_positive_probability": np.nan,
            }
        )
        rows.append(
            {
                "year": int(year),
                "module": "",
                "validation_block_count": sleeve_block_count,
                "static_fixed_effect_validation_rank_ic": float(
                    sleeve_static_score
                ),
                **sleeve_diagnostics,
            }
        )
    return policies, rows


def base_bundle(encoded: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        "index": encoded["index"],
        "asset_target": encoded["asset_target"].copy(),
        "asset_original_target": encoded["asset_target"].copy(),
        "asset_mask": encoded["asset_mask"],
        "asset_baseline": np.zeros_like(encoded["asset_target"]),
        "asset_static_prior": np.zeros_like(encoded["asset_target"]),
        "sleeve_target": encoded["sleeve_target"].copy(),
        "sleeve_original_target": encoded["sleeve_target"].copy(),
        "sleeve_mask": encoded["sleeve_mask"],
        "sleeve_baseline": np.zeros_like(encoded["sleeve_target"]),
        "sleeve_static_prior": np.zeros_like(
            encoded["sleeve_target"]
        ),
    }


def concatenate_encoded(
    *parts: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Concatenate chronological, fully mature encoded folds for final refitting."""
    if not parts:
        raise ValueError("At least one encoded fold is required")
    keys = set(parts[0])
    if any(set(part) != keys for part in parts[1:]):
        raise ValueError("Encoded folds do not share the same keys")
    combined = {
        key: np.concatenate([part[key] for part in parts], axis=0)
        for key in keys
    }
    if not np.all(np.diff(combined["index"].astype(np.int64)) > 0):
        raise ValueError("Final probe refit inputs must be strictly chronological")
    return combined


def fit_seed_probes(
    train: dict[str, np.ndarray],
    validation: dict[str, np.ndarray],
    oos: dict[str, np.ndarray],
    arrays,
    metadata: dict[str, Any],
    probe_cfg: dict[str, Any],
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    list[dict[str, Any]],
]:
    horizons = [int(value) for value in metadata["horizons_days"]]
    validation_bundle = base_bundle(validation)
    oos_bundle = base_bundle(oos)
    validation_bundle["asset_residual"] = np.zeros_like(
        validation["asset_target"]
    )
    validation_bundle["sleeve_residual"] = np.zeros_like(
        validation["sleeve_target"]
    )
    oos_bundle["asset_residual"] = np.zeros_like(oos["asset_target"])
    oos_bundle["sleeve_residual"] = np.zeros_like(oos["sleeve_target"])
    validation_scale_asset = np.ones_like(
        validation["asset_target"], dtype=np.float32
    )
    validation_scale_sleeve = np.ones_like(
        validation["sleeve_target"], dtype=np.float32
    )
    oos_scale_asset = np.ones_like(
        oos["asset_target"], dtype=np.float32
    )
    oos_scale_sleeve = np.ones_like(
        oos["sleeve_target"], dtype=np.float32
    )
    artifacts: list[dict[str, Any]] = []

    for module_position, module in enumerate(RISK_SLEEVES):
        columns = np.flatnonzero(arrays.module_index == module_position)
        for horizon_position, horizon_days in enumerate(horizons):
            val_prediction, oos_prediction, scale, artifact = (
                fit_fixed_ridge_probe(
                    train["asset_state"][:, columns],
                    train["asset_target"][
                        :, horizon_position, columns
                    ],
                    train["asset_mask"][:, horizon_position, columns],
                    validation["asset_state"][:, columns],
                    validation["asset_target"][
                        :, horizon_position, columns
                    ],
                    validation["asset_mask"][
                        :, horizon_position, columns
                    ],
                    oos["asset_state"][:, columns],
                    probe_cfg,
                )
            )
            validation_bundle["asset_residual"][
                :, horizon_position, columns
            ] = val_prediction
            oos_bundle["asset_residual"][
                :, horizon_position, columns
            ] = oos_prediction
            fixed_effect = np.asarray(
                artifact["target_entity_fixed_effect"],
                dtype=np.float64,
            )
            validation_bundle["asset_target"][
                :, horizon_position, columns
            ] = dynamic_target(
                validation["asset_target"][
                    :, horizon_position, columns
                ],
                validation["asset_mask"][
                    :, horizon_position, columns
                ],
                fixed_effect,
            )
            oos_bundle["asset_target"][
                :, horizon_position, columns
            ] = dynamic_target(
                oos["asset_target"][:, horizon_position, columns],
                oos["asset_mask"][:, horizon_position, columns],
                fixed_effect,
            )
            validation_bundle["asset_static_prior"][
                :, horizon_position, columns
            ] = center_cross_section(
                np.repeat(
                    fixed_effect[None, :],
                    len(validation["index"]),
                    axis=0,
                )
            )
            oos_bundle["asset_static_prior"][
                :, horizon_position, columns
            ] = center_cross_section(
                np.repeat(
                    fixed_effect[None, :],
                    len(oos["index"]),
                    axis=0,
                )
            )
            validation_scale_asset[
                :, horizon_position, columns
            ] = scale
            oos_scale_asset[:, horizon_position, columns] = scale
            artifacts.append(
                {
                    "task": "asset",
                    "module": module,
                    "horizon_days": horizon_days,
                    **artifact,
                }
            )

    for horizon_position, horizon_days in enumerate(horizons):
        val_prediction, oos_prediction, scale, artifact = (
            fit_fixed_ridge_probe(
                train["sleeve_state"],
                train["sleeve_target"][:, horizon_position],
                train["sleeve_mask"][:, horizon_position],
                validation["sleeve_state"],
                validation["sleeve_target"][:, horizon_position],
                validation["sleeve_mask"][:, horizon_position],
                oos["sleeve_state"],
                probe_cfg,
            )
        )
        validation_bundle["sleeve_residual"][
            :, horizon_position
        ] = val_prediction
        oos_bundle["sleeve_residual"][:, horizon_position] = (
            oos_prediction
        )
        fixed_effect = np.asarray(
            artifact["target_entity_fixed_effect"],
            dtype=np.float64,
        )
        validation_bundle["sleeve_target"][
            :, horizon_position
        ] = dynamic_target(
            validation["sleeve_target"][:, horizon_position],
            validation["sleeve_mask"][:, horizon_position],
            fixed_effect,
        )
        oos_bundle["sleeve_target"][
            :, horizon_position
        ] = dynamic_target(
            oos["sleeve_target"][:, horizon_position],
            oos["sleeve_mask"][:, horizon_position],
            fixed_effect,
        )
        validation_bundle["sleeve_static_prior"][
            :, horizon_position
        ] = center_cross_section(
            np.repeat(
                fixed_effect[None, :],
                len(validation["index"]),
                axis=0,
            )
        )
        oos_bundle["sleeve_static_prior"][
            :, horizon_position
        ] = center_cross_section(
            np.repeat(
                fixed_effect[None, :],
                len(oos["index"]),
                axis=0,
            )
        )
        validation_scale_sleeve[:, horizon_position] = scale
        oos_scale_sleeve[:, horizon_position] = scale
        artifacts.append(
            {
                "task": "sleeve",
                "module": "",
                "horizon_days": horizon_days,
                **artifact,
            }
        )
    validation_bundle["asset_scale"] = validation_scale_asset
    validation_bundle["sleeve_scale"] = validation_scale_sleeve
    oos_bundle["asset_scale"] = oos_scale_asset
    oos_bundle["sleeve_scale"] = oos_scale_sleeve
    return validation_bundle, oos_bundle, artifacts


def ensemble_bundles(
    bundles: list[dict[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    output = copy.deepcopy(bundles[0])
    for key in ("asset_residual", "sleeve_residual"):
        stacked = np.stack([bundle[key] for bundle in bundles], axis=0)
        output[key] = stacked.mean(axis=0).astype(np.float32)
        scale_key = key.replace("residual", "scale")
        scales = np.stack(
            [bundle[scale_key] for bundle in bundles], axis=0
        )
        total_scale = np.sqrt(
            np.mean(np.square(scales), axis=0)
            + np.var(stacked, axis=0)
        )
        output[key.replace("residual", "logvar")] = (
            2.0 * np.log(np.maximum(total_scale, 1e-4))
        ).astype(np.float32)
    return output


def encoder_checkpoint(
    cfg: dict[str, Any],
    year: int,
    seed: int,
) -> Path:
    path = ROOT / cfg["outputs"]["model_dir"]
    path.mkdir(parents=True, exist_ok=True)
    return path / f"encoder_{year}_seed_{seed}.pt"


def load_or_train_encoder(
    trainer,
    arrays,
    metadata: dict[str, Any],
    cfg: dict[str, Any],
    train_indices: np.ndarray,
    year: int,
    seed: int,
    device: torch.device,
    resume: bool,
    source_cfg: dict[str, Any] | None,
) -> tuple[Any, list[dict[str, Any]], dict[str, Any]]:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    model = trainer.make_model(arrays, metadata, cfg).to(device)
    if source_cfg is not None:
        path = (
            ROOT
            / source_cfg["outputs"]["model_dir"]
            / f"model_{year}_seed_{seed}.pt"
        )
        checkpoint = torch.load(
            path, map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint["state_dict"])
        return model, [], {
            "enabled": True,
            "source": str(path),
            "diagnostic_reused_supervised_checkpoint": True,
        }

    path = encoder_checkpoint(cfg, year, seed)
    signature = normalized_config_hash(cfg, train_indices, metadata)
    architecture_signature = encoder_architecture_hash(cfg, metadata)
    schedule = pretraining_schedule(cfg)
    if resume and path.exists():
        checkpoint = torch.load(
            path, map_location=device, weights_only=False
        )
        if checkpoint.get("signature") == signature:
            model.load_state_dict(checkpoint["state_dict"])
            return model, [], {
                **checkpoint.get("pretraining", {}),
                "resumed": True,
            }
    warm_start: dict[str, Any] = {"warm_started": False}
    if bool(
        cfg["model"].get("self_supervised_pretraining", {}).get(
            "warm_start_from_previous_year", False
        )
    ):
        previous_path = encoder_checkpoint(cfg, year - 1, seed)
        if previous_path.exists():
            previous = torch.load(
                previous_path, map_location=device, weights_only=False
            )
            valid_warm_start = bool(
                previous.get("architecture_signature") == architecture_signature
                and previous.get("pretraining_schedule") == schedule
                and bool(previous.get("pretraining", {}).get("enabled", False))
            )
            if valid_warm_start:
                model.load_state_dict(previous["state_dict"])
                warm_start = {
                    "warm_started": True,
                    "warm_start_source_year": int(year - 1),
                    "warm_start_source": str(previous_path),
                }
    logs, summary = trainer.pretrain_encoder(
        model,
        arrays,
        metadata,
        cfg,
        train_indices,
        device,
        int(seed),
        int(year),
    )
    torch.save(
        {
            "state_dict": model.state_dict(),
            "year": int(year),
            "seed": int(seed),
            "signature": signature,
            "architecture_signature": architecture_signature,
            "pretraining_schedule": schedule,
            "pretraining": {**summary, **warm_start},
        },
        path,
    )
    return model, logs, {**summary, **warm_start}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "config/hierarchical_asset_itransformer_v42_causal_linear_probe.yaml"
        ),
    )
    parser.add_argument("--years", default=None)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--pretrain-max-epochs", type=int, default=None)
    parser.add_argument(
        "--encoder-checkpoint-config",
        type=Path,
        default=None,
        help="Diagnostic only: reuse compatible model checkpoints.",
    )
    args = parser.parse_args()

    trainer = load_python_module(
        "linear_probe_trainer",
        ROOT / "scripts" / "135_train_hierarchical_asset_itransformer_v7.py",
    )
    helpers = load_python_module(
        "linear_probe_helpers",
        ROOT / "scripts" / "162_audit_frozen_itransformer_ridge_head.py",
    )
    cfg = load_config(args.config)
    if args.pretrain_max_epochs is not None:
        pre_cfg = cfg["model"]["self_supervised_pretraining"]
        pre_cfg["max_epochs"] = int(args.pretrain_max_epochs)
        pre_cfg["minimum_epochs"] = min(
            int(pre_cfg.get("minimum_epochs", 1)),
            int(args.pretrain_max_epochs),
        )
        pre_cfg["patience"] = min(
            int(pre_cfg.get("patience", 1)),
            int(args.pretrain_max_epochs),
        )
    source_cfg = (
        load_config(args.encoder_checkpoint_config)
        if args.encoder_checkpoint_config is not None
        else None
    )
    arrays, metadata = load_arrays(cfg)
    dates = pd.to_datetime(arrays.dates)
    available_years = sorted(
        set(
            dates[
                dates >= pd.Timestamp(cfg["data"]["prediction_start"])
            ].year
        )
    )
    years = (
        available_years
        if args.years is None
        else [int(value) for value in args.years.split(",")]
    )
    seeds = (
        [int(value) for value in cfg["model"]["seeds"]]
        if args.seeds is None
        else [int(value) for value in args.seeds.split(",")]
    )
    probe_cfg = cfg["model"]["causal_linear_probe"]
    if str(cfg["data"].get("target_mode")) != "relative_mean_return":
        raise ValueError(
            "The causal linear probe must learn relative_mean_return."
        )
    if str(
        cfg["data"].get("forecast_baseline_mode")
    ) != "zero_active_view":
        raise ValueError(
            "The learned view must use exact-zero fallback, not a hidden "
            "mechanical forecast baseline."
        )
    if str(probe_cfg.get("target")) != "relative_mean_return":
        raise ValueError(
            "Probe target and dataset target semantics do not match."
        )
    configured_seeds = [
        int(value) for value in cfg["model"]["seeds"]
    ]
    complete_seed_ensemble = set(seeds) == set(configured_seeds)
    complete_year_coverage = set(years) == set(available_years)
    device = torch.device("cpu")
    sequence_length = int(metadata["sequence_length_days"])
    batch_size = int(cfg["model"]["batch_size"])

    asset_rows: list[dict[str, Any]] = []
    sleeve_rows: list[dict[str, Any]] = []
    metric_output: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []

    for year in years:
        train_indices, validation_indices, oos_indices = (
            helpers.fold_indices(arrays, metadata, cfg, year)
        )
        current_audit = audit_fold_contract(
            arrays,
            metadata,
            cfg,
            year,
            train_indices,
            validation_indices,
            oos_indices,
        )
        audit_rows.extend(current_audit)
        failed_audit = [
            row for row in current_audit if not bool(row["passed"])
        ]
        if failed_audit:
            raise RuntimeError(
                "Point-in-time fold contract failed: "
                + json.dumps(failed_audit, ensure_ascii=False)
            )
        validation_start_ns = int(dates[validation_indices[0]].value)
        cutoff_ns = int(pd.Timestamp(year, 1, 1).value)
        train_loader = helpers.make_loader(
            arrays,
            train_indices,
            sequence_length,
            batch_size,
            validation_start_ns,
        )
        validation_loader = helpers.make_loader(
            arrays,
            validation_indices,
            sequence_length,
            batch_size,
            cutoff_ns,
        )
        oos_loader = helpers.make_loader(
            arrays,
            oos_indices,
            sequence_length,
            batch_size,
            None,
        )
        validation_by_seed: list[dict[str, np.ndarray]] = []
        oos_by_seed: list[dict[str, np.ndarray]] = []
        validation_encoded_by_seed: list[dict[str, np.ndarray]] = []
        combined_encoded_by_seed: list[dict[str, np.ndarray]] = []
        oos_encoded_by_seed: list[dict[str, np.ndarray]] = []
        for seed in seeds:
            print(
                f"[year={year}][seed={seed}] "
                f"train={len(train_indices)} validation={len(validation_indices)}",
                flush=True,
            )
            model, pretrain_logs, pretrain_summary = (
                load_or_train_encoder(
                    trainer,
                    arrays,
                    metadata,
                    cfg,
                    train_indices,
                    year,
                    seed,
                    device,
                    args.resume,
                    source_cfg,
                )
            )
            training_rows.extend(pretrain_logs)
            train_encoded = helpers.encode(
                model, train_loader, trainer, device
            )
            validation_encoded = helpers.encode(
                model, validation_loader, trainer, device
            )
            oos_encoded = helpers.encode(
                model, oos_loader, trainer, device
            )
            validation_bundle, oos_bundle, artifacts = fit_seed_probes(
                train_encoded,
                validation_encoded,
                oos_encoded,
                arrays,
                metadata,
                probe_cfg,
            )
            validation_by_seed.append(validation_bundle)
            oos_by_seed.append(oos_bundle)
            validation_encoded_by_seed.append(validation_encoded)
            combined_encoded_by_seed.append(
                concatenate_encoded(train_encoded, validation_encoded)
            )
            oos_encoded_by_seed.append(oos_encoded)
            for artifact in artifacts:
                training_rows.append(
                    {
                        "phase": "causal_linear_probe",
                        "year": int(year),
                        "seed": int(seed),
                        **pretrain_summary,
                        **artifact,
                    }
                )

        ensemble_validation = ensemble_bundles(validation_by_seed)
        ensemble_oos = ensemble_bundles(oos_by_seed)
        policies, current_gate_rows = make_policies(
            year,
            validation_by_seed,
            ensemble_validation,
            ensemble_oos,
            arrays,
            metadata,
            history,
            probe_cfg,
        )
        if bool(probe_cfg.get("final_refit_after_validation", True)):
            refitted_oos_by_seed: list[dict[str, np.ndarray]] = []
            for seed_position, seed in enumerate(seeds):
                # The gate was already fixed from the purged train-only fit and
                # validation period.  At this point every label in the combined
                # fold is known before the OOS year starts, so refitting only
                # reduces estimation error; it cannot alter the gate.
                _, refitted_oos, refit_artifacts = fit_seed_probes(
                    combined_encoded_by_seed[seed_position],
                    validation_encoded_by_seed[seed_position],
                    oos_encoded_by_seed[seed_position],
                    arrays,
                    metadata,
                    probe_cfg,
                )
                # Preserve uncertainty calibrated on the untouched validation
                # block rather than replacing it with an in-sample residual.
                refitted_oos["asset_scale"] = oos_by_seed[seed_position][
                    "asset_scale"
                ]
                refitted_oos["sleeve_scale"] = oos_by_seed[seed_position][
                    "sleeve_scale"
                ]
                refitted_oos_by_seed.append(refitted_oos)
                for artifact in refit_artifacts:
                    training_rows.append(
                        {
                            "phase": "causal_linear_probe_final_refit",
                            "year": int(year),
                            "seed": int(seed),
                            "gate_fixed_before_refit": True,
                            "refit_training_as_of": str(
                                pd.Timestamp(year, 1, 1).date()
                            ),
                            **artifact,
                        }
                    )
            ensemble_oos = ensemble_bundles(refitted_oos_by_seed)
        gate_rows.extend(current_gate_rows)
        trainer.append_long_predictions(
            asset_rows,
            sleeve_rows,
            ensemble_oos,
            arrays,
            metadata,
            policies,
            year,
            cfg,
        )
        metric_output.extend(
            trainer.metric_rows(
                ensemble_oos,
                metadata,
                arrays.module_index,
                policies,
                year,
                cfg,
            )
        )
        history.append(
            {
                "year": int(year),
                "bundle": ensemble_oos,
                "label_end": arrays.label_end,
            }
        )
        active = [
            f"{row['task']}:{row['module']}:{row['horizon_days']}"
            for row in current_gate_rows
            if bool(row["active"])
        ]
        print(
            json.dumps(
                {
                    "year": int(year),
                    "active_components": active,
                    "prediction_samples": int(len(oos_indices)),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    asset = pd.DataFrame(asset_rows).sort_values(
        ["as_of_date", "horizon_days", "module", "asset_code"]
    )
    sleeve = pd.DataFrame(sleeve_rows).sort_values(
        ["as_of_date", "horizon_days", "sleeve"]
    )
    score_semantics = "annualized_dynamic_relative_mean_innovation"
    asset["score_semantics"] = score_semantics
    sleeve["score_semantics"] = score_semantics
    asset["entity_fixed_effect_removed"] = True
    sleeve["entity_fixed_effect_removed"] = True
    validation_report = pd.DataFrame(metric_output)
    gate_frame = pd.DataFrame(gate_rows)
    training_frame = pd.DataFrame(training_rows)
    audit_frame = pd.DataFrame(audit_rows)
    asset.to_parquet(project_path(cfg, "asset_predictions"), index=False)
    sleeve.to_parquet(project_path(cfg, "sleeve_predictions"), index=False)
    validation_report.to_csv(
        project_path(cfg, "validation_report"),
        index=False,
        encoding="utf-8-sig",
    )
    training_frame.to_csv(
        project_path(cfg, "training_log"),
        index=False,
        encoding="utf-8-sig",
    )
    gate_detail_path = (
        project_path(cfg, "gate_report").with_suffix(".components.csv")
    )
    gate_frame.to_csv(
        gate_detail_path, index=False, encoding="utf-8-sig"
    )
    audit_frame.to_csv(
        project_path(cfg, "audit"), index=False, encoding="utf-8-sig"
    )

    finite = bool(
        np.isfinite(asset["model_raw_score"]).all()
        and np.isfinite(sleeve["model_raw_score"]).all()
        and np.isfinite(asset["prediction_uncertainty"]).all()
        and np.isfinite(sleeve["prediction_uncertainty"]).all()
    )
    strict_labels = bool(
        (
            asset.loc[asset["has_label"], "horizon_label_end_date"]
            > asset.loc[asset["has_label"], "as_of_date"]
        ).all()
        and (
            sleeve.loc[sleeve["has_label"], "horizon_label_end_date"]
            > sleeve.loc[sleeve["has_label"], "as_of_date"]
        ).all()
    )
    asset_zero_rows = asset["residual_alpha"].eq(0.0)
    sleeve_zero_rows = sleeve["residual_alpha"].eq(0.0)
    zero_contract = bool(
        np.allclose(
            asset.loc[
                asset_zero_rows, "walkforward_relative_score"
            ],
            0.0,
            atol=1e-12,
        )
        and np.allclose(
            sleeve.loc[
                sleeve_zero_rows, "walkforward_relative_score"
            ],
            0.0,
            atol=1e-12,
        )
    )
    training_cutoff_contract = bool(
        (
            asset["model_training_cutoff"] < asset["as_of_date"]
        ).all()
        and (
            sleeve["model_training_cutoff"] < sleeve["as_of_date"]
        ).all()
    )
    unique_output_keys = bool(
        not asset.duplicated(
            [
                "as_of_date",
                "horizon_days",
                "module",
                "asset_code",
            ]
        ).any()
        and not sleeve.duplicated(
            ["as_of_date", "horizon_days", "sleeve"]
        ).any()
    )
    fold_contract = bool(audit_frame["passed"].astype(bool).all())
    diagnostic_checkpoint_reuse = source_cfg is not None
    active_components = gate_frame.loc[
        gate_frame["active"].astype(bool)
    ]
    configured_horizon_weights = cfg["portfolio"].get("asset_score_horizon_weights")
    if configured_horizon_weights is None:
        asset_portfolio_horizons = {int(cfg["portfolio"]["asset_score_horizon_days"])}
    else:
        asset_portfolio_horizons = {
            int(horizon)
            for horizon, weight in configured_horizon_weights.items()
            if float(weight) > 0.0
        }
    portfolio_active_components = active_components.loc[
        active_components["task"].eq("sleeve")
        | (
            active_components["task"].eq("asset")
            & active_components["horizon_days"].astype(int).isin(asset_portfolio_horizons)
        )
    ]
    gate = {
        "version": cfg["version"],
        "mode": "causal_frozen_itransformer_linear_probe",
        "enabled_for_portfolio_stage": bool(
            finite
            and strict_labels
            and zero_contract
            and training_cutoff_contract
            and unique_output_keys
            and fold_contract
            and complete_seed_ensemble
            and complete_year_coverage
            and not diagnostic_checkpoint_reuse
            and not portfolio_active_components.empty
        ),
        "enabled_for_formal_strategy": False,
        "checks": {
            "finite_prediction_outputs": finite,
            "labels_strictly_after_as_of": strict_labels,
            "exact_zero_fallback": zero_contract,
            "training_cutoff_before_as_of": training_cutoff_contract,
            "unique_prediction_keys": unique_output_keys,
            "all_fold_point_in_time_contracts": fold_contract,
            "complete_configured_seed_ensemble": complete_seed_ensemble,
            "complete_available_year_coverage": complete_year_coverage,
            "relative_mean_target_with_zero_fallback": True,
            "encoder_not_reused_from_supervised_diagnostic": (
                not diagnostic_checkpoint_reuse
            ),
            "has_portfolio_consumed_active_component": bool(
                not portfolio_active_components.empty
            ),
        },
        "active_components": active_components[
            ["year", "task", "module", "horizon_days"]
        ].to_dict("records"),
        "portfolio_consumed_active_components": (
            portfolio_active_components[
                ["year", "task", "module", "horizon_days"]
            ].to_dict("records")
        ),
        "component_report": str(gate_detail_path.relative_to(ROOT)),
        "summary": {
            "asset_rank_ic": float(
                validation_report["asset_model_rank_ic"].mean()
            ),
            "asset_incremental_rank_ic": float(
                validation_report["asset_incremental_rank_ic"].mean()
            ),
            "sleeve_rank_ic": float(
                validation_report["sleeve_model_rank_ic"].mean()
            ),
            "sleeve_incremental_rank_ic": float(
                validation_report["sleeve_incremental_rank_ic"].mean()
            ),
        },
        "formal_status_reason": (
            "Diagnostic supervised-checkpoint reuse is never eligible for "
            "portfolio deployment."
            if diagnostic_checkpoint_reuse
            else (
                "Reporting-only until the configured full 2022-2026 "
                "walk-forward and portfolio bootstrap gates are complete."
            )
        ),
    }
    project_path(cfg, "gate_report").write_text(
        json.dumps(gate, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        f"# {cfg['version']}",
        "",
        "The iTransformer is retained as a masked-history representation "
        "learner. Frozen states feed deterministic ridge probes that estimate "
        "annualized relative mean views. EP/skfolio owns covariance and target "
        "weights; PPO owns execution timing only.",
        "",
        "## Component gate",
        "",
        gate_frame.to_markdown(index=False),
        "",
        "## Walk-forward metrics",
        "",
        validation_report.to_markdown(index=False),
    ]
    project_path(cfg, "report_md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "asset_predictions": str(
                    project_path(cfg, "asset_predictions")
                ),
                "sleeve_predictions": str(
                    project_path(cfg, "sleeve_predictions")
                ),
                "gate": gate,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

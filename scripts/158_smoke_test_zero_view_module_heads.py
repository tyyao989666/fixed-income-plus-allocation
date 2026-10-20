from __future__ import annotations

"""Structural tests for the zero-fallback, module-head iTransformer."""

import argparse
import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.hierarchical_asset_itransformer import (  # noqa: E402
    HierarchicalAssetITransformer,
    _masked_regression_loss,
    load_arrays,
    load_config,
)


def test_detached_uncertainty_gradient() -> dict[str, float]:
    target = torch.tensor([[[0.20, -0.10, 0.00]]], dtype=torch.float32)
    mask = torch.ones_like(target, dtype=torch.bool)
    horizon_weights = torch.ones(1)

    gradients: list[np.ndarray] = []
    logvar_gradients: list[float] = []
    for logvar_value in (-3.0, 2.0):
        prediction = torch.tensor(
            [[[0.04, -0.02, 0.01]]],
            dtype=torch.float32,
            requires_grad=True,
        )
        logvar = torch.full_like(
            prediction, logvar_value, requires_grad=True
        )
        loss = _masked_regression_loss(
            prediction,
            logvar,
            target,
            mask,
            horizon_weights,
            mode="robust_with_detached_uncertainty",
            huber_beta=0.05,
            uncertainty_nll_weight=0.10,
        )
        loss.backward()
        gradients.append(prediction.grad.detach().numpy().copy())
        logvar_gradients.append(float(logvar.grad.abs().sum()))
    np.testing.assert_allclose(
        gradients[0],
        gradients[1],
        atol=1e-7,
        rtol=1e-7,
        err_msg="variance inflation changed the mean-head gradient",
    )
    if min(logvar_gradients) <= 0.0:
        raise AssertionError("uncertainty head received no calibration gradient")
    return {
        "mean_gradient_l1": float(np.abs(gradients[0]).sum()),
        "minimum_logvar_gradient_l1": float(min(logvar_gradients)),
    }


def test_real_dataset(cfg: dict) -> dict[str, object]:
    arrays, metadata = load_arrays(cfg)
    if metadata.get("forecast_baseline_mode") != "zero_active_view":
        raise AssertionError("dataset is not in zero_active_view mode")
    if float(np.abs(arrays.asset_baseline).max()) > 0.0:
        raise AssertionError("asset fallback baseline is not exactly zero")
    if float(np.abs(arrays.sleeve_baseline).max()) > 0.0:
        raise AssertionError("sleeve fallback baseline is not exactly zero")
    if not any(
        name.startswith("causal_baseline_annualized_mean_")
        for name in metadata["local_features"]
    ):
        raise AssertionError("causal momentum anchor disappeared from inputs")
    if not any(
        name.startswith("causal_sleeve_baseline_")
        for name in metadata["macro_features"]
    ):
        raise AssertionError("causal sleeve anchor disappeared from inputs")

    model = HierarchicalAssetITransformer(
        sequence_length=int(metadata["sequence_length_days"]),
        local_features=len(metadata["local_features"]),
        macro_features=len(metadata["macro_features"]),
        module_index=arrays.module_index,
        horizons=len(metadata["horizons_days"]),
        cfg=cfg["model"],
        asset_experts=len(metadata["asset_signal_experts"]),
        sleeve_experts=len(metadata["sleeve_signal_experts"]),
    )
    if model.asset_head_scope != "module":
        raise AssertionError("asset heads were not split by sleeve")
    mean_head_parameters = [
        name
        for name, _ in model.named_parameters()
        if name.startswith("asset_mean_by_module.")
    ]
    modules_with_heads = {
        int(name.split(".")[1]) for name in mean_head_parameters
    }
    if modules_with_heads != set(range(5)):
        raise AssertionError("not every sleeve has an independent mean head")

    return {
        "rows": int(len(arrays.dates)),
        "samples": int(len(arrays.sample_indices)),
        "asset_baseline_max_abs": float(
            np.abs(arrays.asset_baseline).max()
        ),
        "sleeve_baseline_max_abs": float(
            np.abs(arrays.sleeve_baseline).max()
        ),
        "independent_asset_heads": len(modules_with_heads),
    }


def test_zero_view_blend_anchor(cfg: dict) -> dict[str, object]:
    trainer_path = (
        ROOT
        / "scripts"
        / "135_train_hierarchical_asset_itransformer_v7.py"
    )
    spec = importlib.util.spec_from_file_location(
        "hierarchical_itransformer_trainer",
        trainer_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load the iTransformer trainer")
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)

    rng = np.random.default_rng(20260727)
    observations = 180
    horizons = 3
    assets = 10
    sleeves = 5
    module_index = np.repeat(np.arange(sleeves), 2)
    dates = pd.bdate_range("2020-01-02", periods=observations)
    base_signal = np.tile(
        np.array([-0.08, 0.08], dtype=float),
        sleeves,
    )
    asset_signal = np.broadcast_to(
        base_signal,
        (observations, assets),
    ).copy()
    asset_target = asset_signal + rng.normal(
        0.0, 0.004, size=asset_signal.shape
    )
    sleeve_signal = np.broadcast_to(
        np.linspace(-0.08, 0.08, sleeves),
        (observations, sleeves),
    ).copy()
    sleeve_target = sleeve_signal + rng.normal(
        0.0, 0.004, size=sleeve_signal.shape
    )

    bundle = {
        "index": np.arange(observations, dtype=np.int64),
        "asset_baseline": np.zeros(
            (observations, horizons, assets), dtype=float
        ),
        "asset_residual": np.zeros(
            (observations, horizons, assets), dtype=float
        ),
        "asset_target": np.zeros(
            (observations, horizons, assets), dtype=float
        ),
        "asset_mask": np.ones(
            (observations, horizons, assets), dtype=bool
        ),
        "sleeve_baseline": np.zeros(
            (observations, horizons, sleeves), dtype=float
        ),
        "sleeve_residual": np.zeros(
            (observations, horizons, sleeves), dtype=float
        ),
        "sleeve_target": np.zeros(
            (observations, horizons, sleeves), dtype=float
        ),
        "sleeve_mask": np.ones(
            (observations, horizons, sleeves), dtype=bool
        ),
    }
    bundle["asset_residual"][:, 2] = asset_signal
    bundle["asset_target"][:, 2] = asset_target
    bundle["sleeve_residual"][:, 2] = sleeve_signal
    bundle["sleeve_target"][:, 2] = sleeve_target
    arrays = SimpleNamespace(
        regime_index=np.zeros(observations, dtype=np.int64),
        dates=dates.to_numpy(dtype="datetime64[ns]"),
    )
    test_cfg = copy.deepcopy(cfg)
    test_cfg["model"]["blend_bootstrap_draws"] = 250
    test_cfg["model"]["blend_min_bootstrap_edge_ci_low"] = 0.0
    test_cfg["model"]["blend_min_bootstrap_positive_probability"] = 0.80
    policies, rows = trainer.choose_blend_policies(
        bundle,
        baseline_grid=[0.0],
        residual_grid=[0.0, 1.0],
        module_index=module_index,
        arrays=arrays,
        cfg=test_cfg,
        require_incremental_confidence=True,
    )
    selected = [
        policies["asset_by_module"][module][2]["residual_alpha"]
        for module in (
            "rates",
            "credit",
            "convertible_bonds",
            "equity",
            "alternatives",
        )
    ]
    selected.append(policies["sleeve"][2]["residual_alpha"])
    if any(float(value) <= 0.0 for value in selected):
        raise AssertionError(
            "Strong synthetic forecasts were not promoted from the zero view"
        )
    promoted_rows = [
        row
        for row in rows
        if int(row["horizon_index"]) == 2
        and float(row["residual_alpha"]) > 0.0
        and (
            row.get("module") in {"rates", "credit"}
            or (
                row["task"] == "sleeve"
                and not row.get("module")
            )
        )
    ]
    if not promoted_rows:
        raise AssertionError("No promoted blend candidates were recorded")
    if any(
        float(row["bootstrap_samples"]) <= 0.0
        or not np.isfinite(float(row["bootstrap_edge_ci_low"]))
        for row in promoted_rows
    ):
        raise AssertionError(
            "Zero-view promotion bypassed the paired block bootstrap"
        )
    fallback_rows = [
        row
        for row in rows
        if int(row["horizon_index"]) == 2
        and abs(float(row["residual_alpha"])) <= 1e-12
    ]
    if not fallback_rows or any(
        abs(float(row["baseline_weight"])) > 1e-12
        for row in fallback_rows
    ):
        raise AssertionError(
            "The exact-zero deployment candidate was not the blend anchor"
        )
    return {
        "selected_positive_components": len(selected),
        "bootstrap_rows_checked": len(promoted_rows),
        "fallback_rows_checked": len(fallback_rows),
    }


def test_mean_view_rank_checkpoint_metric(cfg: dict) -> dict[str, object]:
    trainer_path = (
        ROOT
        / "scripts"
        / "135_train_hierarchical_asset_itransformer_v7.py"
    )
    spec = importlib.util.spec_from_file_location(
        "hierarchical_itransformer_rank_metric",
        trainer_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load the iTransformer trainer")
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)

    observations = 40
    horizons = 3
    sleeves = 5
    assets = sleeves * 2
    module_index = np.repeat(np.arange(sleeves), 2)
    asset_target = np.zeros((observations, horizons, assets), dtype=float)
    sleeve_target = np.zeros((observations, horizons, sleeves), dtype=float)
    for row in range(observations):
        asset_target[row, :, :] = np.tile(
            np.array([-1.0, 1.0], dtype=float),
            sleeves,
        )
        sleeve_target[row, :, :] = np.linspace(-1.0, 1.0, sleeves)

    asset_prediction = asset_target.copy()
    sleeve_prediction = sleeve_target.copy()
    # The disabled five-day target is deliberately inverted. A correct
    # task-specific horizon selector must ignore it.
    asset_prediction[:, 0] *= -1.0
    sleeve_prediction[:, 0] *= -1.0
    # Credit deliberately fails at 60 days, while the synthetic configuration
    # declares only its 20-day head active. This catches accidental reuse of
    # one global asset horizon vector during checkpoint selection.
    credit_columns = np.flatnonzero(module_index == 1)
    asset_prediction[:, 2, credit_columns] *= -1.0
    bundle = {
        "asset_baseline": np.zeros_like(asset_target),
        "asset_residual": asset_prediction,
        "asset_target": asset_target,
        "asset_mask": np.ones_like(asset_target, dtype=bool),
        "sleeve_baseline": np.zeros_like(sleeve_target),
        "sleeve_residual": sleeve_prediction,
        "sleeve_target": sleeve_target,
        "sleeve_mask": np.ones_like(sleeve_target, dtype=bool),
    }
    test_cfg = copy.deepcopy(cfg)
    test_cfg["model"]["validation_selection_metric"] = "mean_view_rank"
    test_cfg["model"]["validation_asset_horizon_weights"] = [0.0, 0.4, 0.6]
    test_cfg["model"]["validation_sleeve_horizon_weights"] = [0.0, 0.5, 0.5]
    test_cfg["model"]["validation_asset_module_horizon_weights"] = {
        "rates": [0.0, 0.0, 1.0],
        "credit": [0.0, 1.0, 0.0],
        "convertible_bonds": [0.0, 0.4, 0.6],
        "equity": [0.0, 0.4, 0.6],
        "alternatives": [0.0, 0.4, 0.6],
    }
    test_cfg["model"]["validation_mean_rank_downside_weight"] = 0.10
    score, metrics = trainer._rank_components(
        bundle,
        module_index,
        test_cfg,
    )
    if score < 0.89:
        raise AssertionError(
            "disabled horizon contaminated the mean-view checkpoint score"
        )
    required = [
        "validation_asset_weighted_incremental_rank_ic",
        "validation_sleeve_weighted_incremental_rank_ic",
        *[
            "validation_asset_weighted_incremental_rank_ic_"
            f"{module}"
            for module in (
                "rates",
                "credit",
                "convertible_bonds",
                "equity",
                "alternatives",
            )
        ],
    ]
    missing = [
        key
        for key in required
        if key not in metrics or not np.isfinite(float(metrics[key]))
    ]
    if missing:
        raise AssertionError(
            "mean-view component checkpoint metrics missing: "
            + ", ".join(missing)
        )
    if float(
        metrics["validation_asset_weighted_incremental_rank_ic_credit"]
    ) < 0.99:
        raise AssertionError(
            "credit checkpoint score used its disabled 60-day horizon"
        )
    return {
        "selection_score": float(score),
        "asset_weighted_rank_ic": float(
            metrics["validation_asset_weighted_incremental_rank_ic"]
        ),
        "sleeve_weighted_rank_ic": float(
            metrics["validation_sleeve_weighted_incremental_rank_ic"]
        ),
        "component_metrics_checked": len(required) - 2,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "config/"
            "hierarchical_asset_itransformer_v29_zero_view_module_heads.yaml"
        ),
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    report = {
        "detached_uncertainty": test_detached_uncertainty_gradient(),
        "real_dataset": test_real_dataset(cfg),
        "zero_view_blend_anchor": test_zero_view_blend_anchor(cfg),
        "mean_view_rank_checkpoint_metric": (
            test_mean_view_rank_checkpoint_metric(cfg)
        ),
        "passed": True,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

"""Structural tests for the decision-aligned relative-mean iTransformer."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.hierarchical_asset_itransformer import (  # noqa: E402
    HierarchicalAssetITransformer,
    WindowDataset,
    _decision_aligned_mean_loss,
    _relative_mean_return_labels,
    load_arrays,
    load_config,
    multitask_loss,
)


def test_label_scale_and_causality() -> dict[str, float]:
    dates = pd.bdate_range("2019-01-02", periods=420)
    columns = ["a", "b", "c"]
    rng = np.random.default_rng(20260727)
    returns = pd.DataFrame(
        rng.normal(0.0001, 0.003, size=(len(dates), len(columns))),
        index=dates,
        columns=columns,
    )
    kwargs = dict(
        horizons=[5, 20, 60],
        minimum_future_coverage=1.0,
        groups=None,
        momentum_windows=[20, 120, 252],
        momentum_weights=[0.2, 0.3, 0.5],
        clip=0.60,
    )
    target, baseline, mask, label_end, _ = _relative_mean_return_labels(
        returns, **kwargs
    )
    cutoff_position = 300
    perturbed = returns.copy()
    perturbed.iloc[cutoff_position + 1 :] += np.array([0.02, -0.01, 0.005])
    changed_target, changed_baseline, _, _, _ = _relative_mean_return_labels(
        perturbed, **kwargs
    )
    np.testing.assert_allclose(
        baseline[: cutoff_position + 1],
        changed_baseline[: cutoff_position + 1],
        atol=0.0,
        rtol=0.0,
        err_msg="future observations changed a causal baseline",
    )
    if np.allclose(
        target[cutoff_position, 1],
        changed_target[cutoff_position, 1],
    ):
        raise AssertionError("future perturbation did not change the training label")
    valid_label_end = label_end.astype("datetime64[ns]").astype(np.int64)
    date_ns = dates.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    valid = mask.any(axis=2)
    if not np.all(valid_label_end[valid] > np.broadcast_to(date_ns[:, None], valid.shape)[valid]):
        raise AssertionError("a relative-mean label does not start after as-of")
    if float(np.nanmax(np.abs(target))) > 0.600001:
        raise AssertionError("relative-mean target exceeded configured clip")
    return {
        "maximum_target": float(np.nanmax(np.abs(target))),
        "future_perturbation_delta": float(
            np.max(
                np.abs(
                    target[cutoff_position, 1]
                    - changed_target[cutoff_position, 1]
                )
            )
        ),
    }


def test_decision_surrogate() -> dict[str, float]:
    target = torch.tensor(
        [
            [
                [0.20, 0.00, -0.20],
                [0.12, 0.01, -0.13],
                [0.08, 0.00, -0.08],
            ]
        ],
        dtype=torch.float32,
    )
    baseline = torch.zeros_like(target)
    aligned = target.clone()
    inverted = -target
    mask = torch.ones_like(target, dtype=torch.bool)
    horizon_weights = torch.tensor([0.05, 0.60, 0.35])
    common = dict(
        baseline=baseline,
        target=target,
        mask=mask,
        groups=None,
        horizon_weights=horizon_weights,
        horizon_days=[5, 20, 60],
        temperature=[0.14, 0.11, 0.09],
        transaction_cost_bps=5.0,
        utility_scale=[0.04, 0.025, 0.015],
    )
    aligned_loss = _decision_aligned_mean_loss(pred=aligned, **common)
    inverted_loss = _decision_aligned_mean_loss(pred=inverted, **common)
    if not float(aligned_loss) < float(inverted_loss):
        raise AssertionError("decision surrogate does not prefer the aligned view")
    return {
        "aligned_loss": float(aligned_loss),
        "inverted_loss": float(inverted_loss),
    }


def test_real_dataset(cfg: dict) -> dict[str, object]:
    arrays, metadata = load_arrays(cfg)
    if metadata["target_mode"] != "relative_mean_return":
        raise AssertionError("unexpected target mode")
    if "causal_baseline_annualized_mean_h20" not in metadata["local_features"]:
        raise AssertionError("annualized mean baseline feature is not named correctly")
    if int((pd.to_datetime(arrays.dates).dayofweek >= 5).sum()) != 0:
        raise AssertionError("weekend dates entered the model calendar")
    sample_index = int(arrays.sample_indices[400])
    cutoff = int(arrays.dates[sample_index])
    item = WindowDataset(
        arrays,
        np.array([sample_index], dtype=np.int64),
        int(metadata["sequence_length_days"]),
        label_cutoff_ns=cutoff,
    )[0]
    label_end = arrays.label_end[sample_index]
    expected_matured = (label_end > np.iinfo(np.int64).min) & (label_end < cutoff)
    actual_matured = item["asset_mask"].numpy().any(axis=1)
    if np.any(actual_matured & ~expected_matured):
        raise AssertionError("unmatured label survived the point-in-time cutoff")

    batch_indices = arrays.sample_indices[500:504]
    loader = DataLoader(
        WindowDataset(
            arrays,
            batch_indices,
            int(metadata["sequence_length_days"]),
        ),
        batch_size=4,
        shuffle=False,
    )
    batch = next(iter(loader))
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
    output = model(
        batch["local_values"],
        batch["local_masks"],
        batch["macro_values"],
        batch["regime_index"],
        batch["regime_prior_count"],
        batch["asset_experts"],
        batch["asset_expert_masks"],
        batch["sleeve_experts"],
        batch["sleeve_expert_masks"],
    )
    asset_residual, _, sleeve_residual, _ = output
    if float(asset_residual.detach().abs().max()) > float(
        cfg["model"]["asset_prediction_output_scale"]
    ) + 1e-6:
        raise AssertionError("asset residual escaped the configured bound")
    if float(sleeve_residual.detach().abs().max()) > float(
        cfg["model"]["sleeve_prediction_output_scale"]
    ) + 1e-6:
        raise AssertionError("sleeve residual escaped the configured bound")
    loss, diagnostics = multitask_loss(
        output, batch, model.module_index, cfg["model"]
    )
    if not torch.isfinite(loss):
        raise AssertionError("real-batch loss is not finite")
    loss.backward()
    gradient = sum(
        float(parameter.grad.abs().sum())
        for parameter in model.parameters()
        if parameter.grad is not None
    )
    if not np.isfinite(gradient) or gradient <= 0.0:
        raise AssertionError("real-batch gradient is invalid")
    sparse_batch = dict(batch)
    sparse_batch["asset_mask"] = batch["asset_mask"].clone()
    sparse_batch["sleeve_mask"] = batch["sleeve_mask"].clone()
    asset_weights = cfg["model"].get(
        "asset_horizon_loss_weights",
        cfg["model"].get("horizon_loss_weights"),
    )
    sleeve_weights = cfg["model"].get(
        "sleeve_horizon_loss_weights",
        cfg["model"].get("horizon_loss_weights"),
    )
    if asset_weights is not None:
        for horizon, weight in enumerate(asset_weights):
            if float(weight) > 0.0:
                sparse_batch["asset_mask"][:, horizon] = False
    if sleeve_weights is not None:
        for horizon, weight in enumerate(sleeve_weights):
            if float(weight) > 0.0:
                sparse_batch["sleeve_mask"][:, horizon] = False
    sparse_loss, _ = multitask_loss(
        output, sparse_batch, model.module_index, cfg["model"]
    )
    if not torch.isfinite(sparse_loss):
        raise AssertionError(
            "a batch without labels for enabled horizons produced NaN"
        )
    return {
        "rows": int(len(arrays.dates)),
        "samples": int(len(arrays.sample_indices)),
        "loss": float(loss.detach()),
        "sparse_enabled_label_loss": float(sparse_loss.detach()),
        "gradient_l1": gradient,
        "asset_residual_max": float(asset_residual.detach().abs().max()),
        "sleeve_residual_max": float(sleeve_residual.detach().abs().max()),
        "asset_decision_loss": diagnostics["asset_decision_utility_loss"],
        "sleeve_decision_loss": diagnostics["sleeve_decision_utility_loss"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/hierarchical_asset_itransformer_v27_decision_mean.yaml"),
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    report = {
        "label": test_label_scale_and_causality(),
        "decision_surrogate": test_decision_surrogate(),
        "real_dataset": test_real_dataset(cfg),
        "passed": True,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

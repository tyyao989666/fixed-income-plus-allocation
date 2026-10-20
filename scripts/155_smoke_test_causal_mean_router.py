from __future__ import annotations

"""Structural and one-batch tests for the causal iTransformer mean-view router."""

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
    _causal_blended_signal_experts,
    _causal_expert_reliability_features,
    _cross_sectional_expert_skill,
    _expert_oracle_distribution,
    _forward_smoothed_expert_edge_targets,
    _masked_shrunk_expert_gate,
    _share_expert_logits,
    load_arrays,
    load_config,
    multitask_loss,
)


def assert_close(left: torch.Tensor, right: torch.Tensor, message: str) -> None:
    if not torch.allclose(left, right, atol=1e-6, rtol=1e-6):
        raise AssertionError(message)


def test_causal_experts(cfg: dict) -> dict[str, object]:
    rng = np.random.default_rng(20260725)
    dates = pd.bdate_range("2020-01-02", periods=420)
    assets = [f"a{position}" for position in range(6)]
    groups = pd.Series(
        ["rates", "rates", "rates", "equity", "equity", "equity"],
        index=assets,
    )
    returns = pd.DataFrame(
        rng.normal(0.0002, 0.01, size=(len(dates), len(assets))),
        index=dates,
        columns=assets,
    )
    returns.loc[dates[:35], "a5"] = np.nan
    definitions = list(cfg["data"]["signal_expert_blends"])
    values_before, masks_before = _causal_blended_signal_experts(
        returns,
        groups,
        definitions,
        "relative_return_utility",
        3.0,
    )
    cutoff_position = 300
    perturbed = returns.copy()
    perturbed.iloc[cutoff_position + 1 :] = rng.normal(
        0.03, 0.02, size=perturbed.iloc[cutoff_position + 1 :].shape
    )
    values_after, masks_after = _causal_blended_signal_experts(
        perturbed,
        groups,
        definitions,
        "relative_return_utility",
        3.0,
    )
    cutoff = dates[cutoff_position]
    for name in values_before:
        np.testing.assert_allclose(
            values_before[name].loc[:cutoff].to_numpy(),
            values_after[name].loc[:cutoff].to_numpy(),
            atol=0.0,
            rtol=0.0,
            err_msg=f"future perturbation changed expert {name}",
        )
        if not masks_before[name].loc[:cutoff].equals(
            masks_after[name].loc[:cutoff]
        ):
            raise AssertionError(f"future perturbation changed mask {name}")
    reliability_dates = pd.bdate_range("2021-01-04", periods=120)
    reliability_rng = np.random.default_rng(47)
    reliability_target = reliability_rng.normal(size=(120, 1, 3)).astype(
        np.float32
    )
    reliability_experts = np.stack(
        [
            reliability_target[:, 0],
            reliability_rng.normal(size=(120, 3)).astype(np.float32),
        ],
        axis=2,
    )[:, None]
    reliability_masks = np.ones_like(reliability_experts, dtype=np.float32)
    reliability_target_mask = np.ones_like(
        reliability_target, dtype=bool
    )
    reliability_label_end = np.full(
        (120, 1), np.datetime64("NaT", "ns"), dtype="datetime64[ns]"
    )
    reliability_label_end[:-5, 0] = reliability_dates.to_numpy(
        dtype="datetime64[ns]"
    )[5:]
    reliability_before, _ = _causal_expert_reliability_features(
        reliability_experts,
        reliability_masks,
        reliability_target,
        reliability_target_mask,
        reliability_label_end,
        reliability_dates,
        None,
        ["baseline", "candidate"],
        [5],
        "test",
        10,
        2,
        5,
        1,
    )
    reliability_cutoff = 80
    changed_target = reliability_target.copy()
    label_ns = reliability_label_end[:, 0].astype("datetime64[ns]").astype(
        np.int64
    )
    cutoff_ns = reliability_dates[reliability_cutoff].value
    changed_target[label_ns >= cutoff_ns] *= -7.0
    reliability_after, _ = _causal_expert_reliability_features(
        reliability_experts,
        reliability_masks,
        changed_target,
        reliability_target_mask,
        reliability_label_end,
        reliability_dates,
        None,
        ["baseline", "candidate"],
        [5],
        "test",
        10,
        2,
        5,
        1,
    )
    np.testing.assert_allclose(
        reliability_before[: reliability_cutoff + 1],
        reliability_after[: reliability_cutoff + 1],
        atol=0.0,
        rtol=0.0,
        err_msg="unmatured target changed historical reliability memory",
    )
    edge_rng = np.random.default_rng(97)
    daily_edge = edge_rng.normal(
        0.0, 0.2, size=(120, 1, 2, 3)
    ).astype(np.float32)
    daily_edge[..., 0] = 0.0
    daily_edge_mask = np.ones_like(daily_edge, dtype=bool)
    skill_before, skill_mask_before, skill_label_end = (
        _forward_smoothed_expert_edge_targets(
            daily_edge,
            daily_edge_mask,
            reliability_label_end,
            20,
            10,
            10,
            1.0,
        )
    )
    changed_edge = daily_edge.copy()
    changed_edge[reliability_cutoff:] *= -9.0
    skill_after, skill_mask_after, changed_skill_label_end = (
        _forward_smoothed_expert_edge_targets(
            changed_edge,
            daily_edge_mask,
            reliability_label_end,
            20,
            10,
            10,
            1.0,
        )
    )
    mature_before_cutoff = (
        skill_label_end.astype("datetime64[ns]").astype(np.int64)
        < cutoff_ns
    )
    np.testing.assert_allclose(
        skill_before[mature_before_cutoff],
        skill_after[mature_before_cutoff],
        atol=0.0,
        rtol=0.0,
        err_msg="future edge perturbation changed a matured router target",
    )
    if not np.array_equal(skill_mask_before, skill_mask_after):
        raise AssertionError("edge values unexpectedly changed skill availability")
    if not np.array_equal(
        skill_label_end.astype("datetime64[ns]").astype(np.int64),
        changed_skill_label_end.astype("datetime64[ns]").astype(np.int64),
    ):
        raise AssertionError("edge values unexpectedly changed skill maturity")
    return {
        "experts": list(values_before),
        "history_end_tested": str(cutoff.date()),
        "reliability_maturity_lag_tested": True,
        "forward_skill_maturity_tested": True,
    }


def test_scale_invariant_oracle() -> dict[str, float]:
    target = torch.tensor([[0.0, 1.0, 2.0, 3.0]], dtype=torch.float32)
    experts = torch.stack(
        [
            target,
            -0.001 * target,
            100.0 * target + 17.0,
        ],
        dim=2,
    )
    valid = torch.ones_like(experts, dtype=torch.bool)
    skill, available = _cross_sectional_expert_skill(
        experts,
        target,
        valid,
        "cross_sectional_correlation",
        2,
    )
    if not bool(available.all()):
        raise AssertionError("finite synthetic experts were marked unavailable")
    if not float(skill[0, 0]) > 0.999 or not float(skill[0, 2]) > 0.999:
        raise AssertionError("correctly ordered experts lost scale invariance")
    if not float(skill[0, 1]) < -0.999:
        raise AssertionError("reversed expert was not rejected")
    abstaining = _expert_oracle_distribution(
        torch.tensor([[0.20, 0.30, 0.10], [0.00, 0.50, -0.10]]),
        torch.ones((2, 3), dtype=torch.bool),
        0.10,
        "abstaining_best",
        0.20,
        0.0,
    )
    assert_close(
        abstaining[0],
        torch.tensor([1.0, 0.0, 0.0]),
        "oracle did not abstain to baseline below the minimum edge",
    )
    assert_close(
        abstaining[1],
        torch.tensor([0.0, 1.0, 0.0]),
        "oracle rejected a qualified non-baseline expert",
    )
    soft_abstaining = _expert_oracle_distribution(
        torch.tensor([[0.0, 0.01, -0.10], [0.0, 0.12, 0.06]]),
        torch.ones((2, 3), dtype=torch.bool),
        0.05,
        "abstaining_soft",
        0.03,
        0.0,
    )
    assert_close(
        soft_abstaining[0],
        torch.tensor([1.0, 0.0, 0.0]),
        "soft oracle did not abstain when every candidate edge was too small",
    )
    if not float(soft_abstaining[1, 1]) > float(
        soft_abstaining[1, 0]
    ):
        raise AssertionError("soft oracle did not reward a qualified edge")
    continuous = _expert_oracle_distribution(
        torch.tensor([[0.0, 0.01, -0.10], [0.0, 0.22, 0.05]]),
        torch.ones((2, 3), dtype=torch.bool),
        0.10,
        "continuous_abstaining",
        0.02,
        0.0,
        0.20,
        0.35,
    )
    assert_close(
        continuous.sum(dim=1),
        torch.ones(2),
        "continuous oracle weights do not sum to one",
    )
    assert_close(
        continuous[0],
        torch.tensor([1.0, 0.0, 0.0]),
        "continuous oracle did not abstain below its edge hurdle",
    )
    assert_close(
        continuous[1, 1:].sum(),
        torch.tensor(0.35),
        "continuous oracle exceeded or missed its non-baseline cap",
    )
    if not float(continuous[1, 1]) > float(continuous[1, 2]):
        raise AssertionError("continuous oracle did not favor the larger edge")
    return {
        "unit_scale_skill": float(skill[0, 0]),
        "reversed_skill": float(skill[0, 1]),
        "large_scale_skill": float(skill[0, 2]),
        "continuous_nonbaseline_cap": float(continuous[1, 1:].sum()),
    }


def test_masked_shared_gate() -> dict[str, float]:
    logits = torch.tensor(
        [
            [
                [[0.0, 1.0, -1.0], [2.0, -1.0, 0.5], [0.2, 0.4, 0.1], [0.9, -0.2, 0.3]]
            ]
        ],
        dtype=torch.float32,
    )
    groups = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    shared = _share_expert_logits(logits, "module_shared", groups)
    assert_close(shared[:, :, 0], shared[:, :, 1], "module 0 logits are not shared")
    assert_close(shared[:, :, 2], shared[:, :, 3], "module 1 logits are not shared")
    masks = torch.ones_like(shared)
    masks[:, :, 0, 0] = 0.0
    masks[:, :, 3, :] = 0.0
    prior = torch.tensor([1.0, 0.0, 0.0])
    gate = _masked_shrunk_expert_gate(shared, masks, prior, 0.65, 1.3)
    if float(gate[:, :, 0, 0].abs().max()) > 1e-12:
        raise AssertionError("masked baseline expert retained weight")
    assert_close(
        gate[:, :, :3].sum(dim=3),
        torch.ones_like(gate[:, :, :3, 0]),
        "available expert gates do not sum to one",
    )
    if float(gate[:, :, 3].abs().max()) > 1e-12:
        raise AssertionError("all-unavailable column did not produce a zero gate")
    expert_values = torch.tensor(
        [[[[1.0, -0.5, 0.25], [0.2, 0.4, -0.1], [0.5, -0.2, 0.7], [9.0, 8.0, 7.0]]]]
    )
    mixture = (gate * expert_values).sum(dim=3)
    for column in range(3):
        available_values = expert_values[0, 0, column][masks[0, 0, column].bool()]
        value = mixture[0, 0, column]
        if value < available_values.min() - 1e-6 or value > available_values.max() + 1e-6:
            raise AssertionError("router output is not a convex expert mixture")
    return {
        "max_gate_sum_error": float(
            (gate[:, :, :3].sum(dim=3) - 1.0).abs().max()
        ),
        "masked_weight_max": float(gate[:, :, 0, 0].abs().max()),
    }


def test_real_batch(cfg: dict) -> dict[str, object]:
    arrays, metadata = load_arrays(cfg)
    dates = pd.to_datetime(arrays.dates)
    valid_label_end = arrays.label_end > np.iinfo(np.int64).min
    if not bool(
        (
            ~valid_label_end
            | (arrays.label_end > dates.to_numpy(dtype="datetime64[ns]").astype(np.int64)[:, None])
        ).all()
    ):
        raise AssertionError("a forecast label does not start strictly after as-of date")
    expected_asset_experts = [
        "fixed_causal_baseline",
        *[str(item["name"]) for item in cfg["data"]["signal_expert_blends"]],
        "market_state_card_rank",
    ]
    expected_sleeve_experts = [
        "fixed_causal_baseline",
        *[str(item["name"]) for item in cfg["data"]["signal_expert_blends"]],
        "mechanical_market_view",
    ]
    if metadata["asset_signal_experts"] != expected_asset_experts:
        raise AssertionError("asset expert metadata does not match the v23 contract")
    if metadata["sleeve_signal_experts"] != expected_sleeve_experts:
        raise AssertionError("sleeve expert metadata does not match the v23 contract")
    if not np.isfinite(arrays.asset_experts).all() or not np.isfinite(
        arrays.sleeve_experts
    ).all():
        raise AssertionError("expert tensor contains NaN or inf")
    if bool(cfg["data"].get("include_forward_expert_skill_targets", False)):
        if not bool(metadata["forward_expert_skill_target"]["enabled"]):
            raise AssertionError("forward expert target metadata is disabled")
        skill_available = (
            arrays.asset_expert_skill_mask.any(axis=(2, 3))
            | arrays.sleeve_expert_skill_mask.any(axis=(2, 3))
        )
        as_of_ns = dates.to_numpy(dtype="datetime64[ns]").astype(np.int64)
        if not bool(
            (
                ~skill_available
                | (arrays.expert_skill_label_end > as_of_ns[:, None])
            ).all()
        ):
            raise AssertionError(
                "a forward expert target matures on or before its as-of date"
            )

    indices = arrays.sample_indices[
        (dates[arrays.sample_indices] >= pd.Timestamp("2021-01-01"))
        & arrays.asset_target_mask[arrays.sample_indices].any(axis=(1, 2))
        & arrays.sleeve_target_mask[arrays.sample_indices].any(axis=(1, 2))
    ][:8]
    loader = DataLoader(
        WindowDataset(
            arrays,
            indices,
            int(metadata["sequence_length_days"]),
        ),
        batch_size=len(indices),
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
    model.train()
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
    loss, diagnostics = multitask_loss(
        output, batch, model.module_index, cfg["model"]
    )
    if not bool(torch.isfinite(loss)):
        raise AssertionError("one-batch router loss is not finite")
    loss.backward()
    gate_gradient = float(
        model.asset_expert_gate.weight.grad.abs().sum()
        + model.sleeve_expert_gate.weight.grad.abs().sum()
    )
    if not np.isfinite(gate_gradient) or gate_gradient <= 0.0:
        raise AssertionError("expert router received no finite gradient")
    return {
        "dataset_dates": len(dates),
        "samples": len(arrays.sample_indices),
        "asset_experts": metadata["asset_signal_experts"],
        "sleeve_experts": metadata["sleeve_signal_experts"],
        "loss": float(loss.detach()),
        "gate_gradient_l1": gate_gradient,
        "routing_loss": diagnostics["expert_routing_loss"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "config"
        / "hierarchical_asset_itransformer_v23_causal_mean_router.yaml",
    )
    args = parser.parse_args()
    torch.set_num_threads(2)
    cfg = load_config(args.config)
    report = {
        "causal_experts": test_causal_experts(cfg),
        "scale_invariant_oracle": test_scale_invariant_oracle(),
        "masked_shared_gate": test_masked_shared_gate(),
        "real_batch": test_real_batch(cfg),
        "passed": True,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

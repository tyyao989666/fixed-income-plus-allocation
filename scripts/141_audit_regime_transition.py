from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.hierarchical_asset_itransformer import (  # noqa: E402
    HierarchicalAssetITransformer,
    WindowDataset,
    load_arrays,
    load_config,
    project_path,
)
from src.itransformer_signal import RISK_SLEEVES  # noqa: E402


def make_model(arrays, metadata: dict[str, Any], cfg: dict[str, Any]) -> HierarchicalAssetITransformer:
    return HierarchicalAssetITransformer(
        sequence_length=int(metadata["sequence_length_days"]),
        local_features=len(metadata["local_features"]),
        macro_features=len(metadata["macro_features"]),
        module_index=arrays.module_index,
        horizons=len(metadata["horizons_days"]),
        cfg=cfg["model"],
    )


def macro_f1(target: np.ndarray, prediction: np.ndarray, classes: int) -> float:
    values: list[float] = []
    for state in range(classes):
        true_positive = np.sum((target == state) & (prediction == state))
        false_positive = np.sum((target != state) & (prediction == state))
        false_negative = np.sum((target == state) & (prediction != state))
        if true_positive + false_negative == 0:
            continue
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        values.append(2.0 * precision * recall / max(precision + recall, 1e-12))
    return float(np.mean(values)) if values else np.nan


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit 3x3-grid transition probabilities and payoffs.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--year", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output-prefix")
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    architecture = cfg["model"].get("sleeve_architecture")
    if architecture not in {"regime_transition_hybrid", "factorized_regime_transition"}:
        raise ValueError("This audit requires a regime-transition sleeve architecture")
    arrays, metadata = load_arrays(cfg)
    dates = pd.to_datetime(arrays.dates)
    indices = arrays.sample_indices[dates[arrays.sample_indices].year == args.year]
    checkpoint_path = project_path(cfg, "model_dir") / f"model_{args.year}_seed_{args.seed}.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = make_model(arrays, metadata, cfg)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    loader = DataLoader(
        WindowDataset(arrays, indices, int(metadata["sequence_length_days"])),
        batch_size=int(cfg["model"]["batch_size"]),
        shuffle=False,
    )
    use_path_occupancy = cfg["model"].get("transition_target") == "path_occupancy"

    collected: dict[str, list[np.ndarray]] = {
        "index": [], "current": [], "target": [], "mask": [], "probability": []
    }
    with torch.no_grad():
        for batch in loader:
            output = model(
                batch["local_values"], batch["local_masks"], batch["macro_values"],
                batch["regime_index"], batch["regime_prior_count"]
            )
            collected["index"].append(batch["index"].numpy())
            collected["current"].append(batch["regime_index"].numpy())
            collected["target"].append(
                batch["regime_occupancy_target"].numpy()
                if use_path_occupancy
                else batch["regime_target"].numpy()
            )
            collected["mask"].append(
                batch["regime_occupancy_mask"].numpy()
                if use_path_occupancy
                else batch["regime_target_mask"].numpy()
            )
            probability_index = 8 if len(output) == 10 else 6
            collected["probability"].append(output[probability_index].numpy())
    values = {key: np.concatenate(items) for key, items in collected.items()}
    classes = int(cfg["model"]["transition_classes"])
    regime_names = metadata["regime_names"][:classes]
    summary_rows: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    for horizon_position, horizon in enumerate(metadata["horizons_days"]):
        valid = values["mask"][:, horizon_position].astype(bool)
        if use_path_occupancy:
            target_distribution = values["target"][valid, horizon_position].astype(float)
            target = target_distribution.argmax(axis=1)
        else:
            target = values["target"][valid, horizon_position].astype(int)
            target_distribution = np.eye(classes, dtype=float)[target]
        probability = values["probability"][valid, horizon_position]
        prediction = probability.argmax(axis=1)
        current = values["current"][valid].astype(int)
        if use_path_occupancy:
            target_grid = target_distribution.reshape(-1, 3, 3)
            target_bond = target_grid.sum(axis=2).argmax(axis=1)
            target_equity = target_grid.sum(axis=1).argmax(axis=1)
        else:
            target_bond = target // 3
            target_equity = target % 3
        prediction_bond = prediction // 3
        prediction_equity = prediction % 3
        current_bond = current // 3
        current_equity = current % 3
        persistence_probability = np.full_like(probability, 1.0 / classes)
        known_current = current < classes
        persistence_probability[known_current] = 0.0
        persistence_probability[
            np.flatnonzero(known_current), current[known_current]
        ] = 1.0
        entropy = -(probability * np.log(np.clip(probability, 1e-12, 1.0))).sum(axis=1)
        top2 = np.argpartition(probability, -2, axis=1)[:, -2:]
        summary_rows.append(
            {
                "horizon_days": horizon,
                "samples": len(target),
                "accuracy": float(np.mean(prediction == target)),
                "persistence_accuracy": float(np.mean(current == target)),
                "accuracy_edge": float(np.mean(prediction == target) - np.mean(current == target)),
                "top2_accuracy": float(np.mean(np.any(top2 == target[:, None], axis=1))),
                "macro_f1": macro_f1(target, prediction, classes),
                "bond_axis_accuracy": float(np.mean(prediction_bond == target_bond)),
                "bond_axis_persistence_accuracy": float(np.mean(current_bond == target_bond)),
                "bond_axis_macro_f1": macro_f1(target_bond, prediction_bond, 3),
                "equity_axis_accuracy": float(np.mean(prediction_equity == target_equity)),
                "equity_axis_persistence_accuracy": float(np.mean(current_equity == target_equity)),
                "equity_axis_macro_f1": macro_f1(target_equity, prediction_equity, 3),
                "nll": float(
                    -(target_distribution * np.log(np.clip(probability, 1e-12, 1.0)))
                    .sum(axis=1)
                    .mean()
                ),
                "persistence_nll": float(
                    -(target_distribution * np.log(np.clip(persistence_probability, 1e-12, 1.0)))
                    .sum(axis=1)
                    .mean()
                ),
                "multiclass_brier": float(
                    np.square(probability - target_distribution).sum(axis=1).mean()
                ),
                "persistence_multiclass_brier": float(
                    np.square(persistence_probability - target_distribution).sum(axis=1).mean()
                ),
                "mean_entropy": float(entropy.mean()),
            }
        )
        valid_indices = values["index"][valid].astype(int)
        for row, array_index in enumerate(valid_indices):
            label_end = int(arrays.regime_label_end[array_index, horizon_position])
            if label_end <= int(arrays.dates[array_index]):
                raise AssertionError(f"non-future transition label at index={array_index}")
            daily_rows.append(
                {
                    "as_of_date": dates[array_index],
                    "horizon_days": horizon,
                    "label_end": pd.Timestamp(label_end),
                    "current_regime": metadata["regime_names"][current[row]],
                    "actual_future_regime": regime_names[target[row]],
                    "predicted_future_regime": regime_names[prediction[row]],
                    "correct": bool(prediction[row] == target[row]),
                    "persistence_correct": bool(current[row] == target[row]),
                    "actual_regime_probability": probability[row, target[row]],
                    "maximum_probability": probability[row, prediction[row]],
                    "entropy": entropy[row],
                    **{
                        f"probability_{regime_names[state]}": probability[row, state]
                        for state in range(classes)
                    },
                }
            )
        for actual in range(classes):
            for predicted in range(classes):
                confusion_rows.append(
                    {
                        "horizon_days": horizon,
                        "actual_regime": regime_names[actual],
                        "predicted_regime": regime_names[predicted],
                        "count": int(np.sum((target == actual) & (prediction == predicted))),
                    }
                )

    if hasattr(model, "state_payoff_global"):
        state_ids = torch.arange(classes)
        state_bond = torch.div(state_ids, 3, rounding_mode="floor").clamp_max(2)
        state_equity = torch.remainder(state_ids, 3).clamp_max(2)
        interaction_scale = float(cfg["model"].get("transition_interaction_scale", 0.25))
        with torch.no_grad():
            payoff = (
                model.state_payoff_global[:, None, :]
                + model.state_payoff_bond[:, state_bond, :]
                + model.state_payoff_equity[:, state_equity, :]
                + interaction_scale * model.state_payoff_interaction
            )
            payoff = payoff - payoff.mean(dim=2, keepdim=True)
    else:
        payoff = model.grid_compatibility[None, :, :].repeat(
            len(metadata["horizons_days"]), 1, 1
        )
    payoff_rows: list[dict[str, Any]] = []
    for horizon_position, horizon in enumerate(metadata["horizons_days"]):
        for state in range(classes):
            for sleeve_position, sleeve in enumerate(RISK_SLEEVES):
                payoff_rows.append(
                    {
                        "horizon_days": horizon,
                        "regime": regime_names[state],
                        "sleeve": sleeve,
                        "relative_payoff": float(payoff[horizon_position, state, sleeve_position]),
                    }
                )

    prefix = (
        ROOT / args.output_prefix
        if args.output_prefix
        else ROOT / "output" / "reports" / f"{Path(args.config).stem}_year_{args.year}_seed_{args.seed}"
    )
    prefix.parent.mkdir(parents=True, exist_ok=True)
    outputs = {
        "summary": (pd.DataFrame(summary_rows), Path(f"{prefix}_transition_summary.csv")),
        "daily": (pd.DataFrame(daily_rows), Path(f"{prefix}_transition_daily.csv")),
        "confusion": (pd.DataFrame(confusion_rows), Path(f"{prefix}_transition_confusion.csv")),
        "payoff": (pd.DataFrame(payoff_rows), Path(f"{prefix}_state_payoff.csv")),
    }
    for _, (frame, path) in outputs.items():
        frame.to_csv(path, index=False, encoding="utf-8-sig")
    print(outputs["summary"][0].to_string(index=False))
    for name, (_, path) in outputs.items():
        print(f"{name}: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

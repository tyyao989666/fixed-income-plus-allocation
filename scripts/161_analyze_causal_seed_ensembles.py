from __future__ import annotations

"""Audit causal seed-ensemble rules on one completed walk-forward fold."""

import argparse
import copy
import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.hierarchical_asset_itransformer import (  # noqa: E402
    RISK_SLEEVES,
    WindowDataset,
    load_arrays,
    load_config,
)


def load_trainer():
    path = ROOT / "scripts" / "135_train_hierarchical_asset_itransformer_v7.py"
    spec = importlib.util.spec_from_file_location("seed_ensemble_trainer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load hierarchical iTransformer trainer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fold_indices(
    arrays,
    metadata: dict[str, Any],
    cfg: dict[str, Any],
    year: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dates = pd.to_datetime(arrays.dates)
    candidate = arrays.sample_indices
    horizons = [int(value) for value in metadata["horizons_days"]]
    split_days = {
        int(value)
        for value in cfg["walk_forward"].get("split_horizons_days", horizons)
    }
    split_positions = np.asarray(
        [position for position, horizon in enumerate(horizons) if horizon in split_days],
        dtype=np.int64,
    )
    cutoff = pd.Timestamp(year, 1, 1)
    cutoff_ns = int(cutoff.value)
    eligible = candidate[dates[candidate] < cutoff]
    stride = max(int(cfg["data"]["train_stride_days"]), 1)
    eligible = eligible[(np.arange(len(eligible)) % stride) == 0]
    reducer = (
        np.all
        if bool(
            cfg["walk_forward"].get(
                "require_all_split_horizons_mature", False
            )
        )
        else np.any
    )
    mature = np.asarray(
        [
            reducer(
                (
                    arrays.label_end[index, split_positions]
                    > np.iinfo(np.int64).min
                )
                & (arrays.label_end[index, split_positions] < cutoff_ns)
            )
            for index in eligible
        ],
        dtype=bool,
    )
    eligible = eligible[mature]
    minimum = int(cfg["walk_forward"]["minimum_train_samples"])
    split = int(
        len(eligible)
        * (1.0 - float(cfg["walk_forward"]["validation_fraction"]))
    )
    split = min(
        max(split, minimum),
        len(eligible) - int(cfg["walk_forward"]["minimum_validation_samples"]),
    )
    preliminary_train = eligible[:split]
    validation = eligible[split:]
    validation_start_ns = int(dates[validation[0]].value)
    if bool(cfg["walk_forward"].get("purge_by_max_label_end", True)):
        keep = np.asarray(
            [
                np.all(
                    (arrays.label_end[index, split_positions] <= np.iinfo(np.int64).min)
                    | (arrays.label_end[index, split_positions] < validation_start_ns)
                )
                for index in preliminary_train
            ],
            dtype=bool,
        )
        train = preliminary_train[keep]
    else:
        train = preliminary_train
    oos = candidate[dates[candidate].year == int(year)]
    return train, validation, oos


def component_rank(
    trainer,
    bundle: dict[str, np.ndarray],
    task: str,
    horizon: int,
    columns: np.ndarray | None,
) -> float:
    prediction = bundle[f"{task}_residual"][:, horizon : horizon + 1]
    target = bundle[f"{task}_target"][:, horizon : horizon + 1]
    mask = bundle[f"{task}_mask"][:, horizon : horizon + 1]
    if columns is not None:
        prediction = prediction[..., columns]
        target = target[..., columns]
        mask = mask[..., columns]
    values = trainer.horizon_rank_ics(prediction, target, mask, None)
    return float(values[0])


def seed_weights(scores: np.ndarray, method: str) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    finite = np.isfinite(scores)
    if method == "equal":
        weights = finite.astype(float)
    elif method == "top2":
        order = np.flatnonzero(finite)
        order = order[np.argsort(scores[order])[::-1]][:2]
        weights = np.zeros_like(scores)
        weights[order] = 1.0
    elif method == "positive_equal":
        weights = (finite & (scores > 0.0)).astype(float)
    elif method == "positive_score":
        weights = np.where(finite, np.maximum(scores, 0.0), 0.0)
    elif method == "best1":
        weights = np.zeros_like(scores)
        if finite.any():
            weights[int(np.nanargmax(scores))] = 1.0
    else:
        raise ValueError(f"Unknown ensemble method={method}")
    if float(weights.sum()) <= 0.0:
        return np.zeros_like(scores)
    return weights / float(weights.sum())


def causal_component_ensemble(
    trainer,
    validation: list[dict[str, np.ndarray]],
    oos: list[dict[str, np.ndarray]],
    module_index: np.ndarray,
    method: str,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    output = copy.deepcopy(oos[0])
    rows: list[dict[str, Any]] = []
    horizon_count = oos[0]["asset_residual"].shape[1]
    output["asset_residual"] = np.zeros_like(oos[0]["asset_residual"])
    output["asset_logvar"] = np.zeros_like(oos[0]["asset_logvar"])
    output["sleeve_residual"] = np.zeros_like(oos[0]["sleeve_residual"])
    output["sleeve_logvar"] = np.zeros_like(oos[0]["sleeve_logvar"])

    for module_position, module in enumerate(RISK_SLEEVES):
        columns = np.flatnonzero(module_index == module_position)
        for horizon in range(horizon_count):
            scores = np.asarray(
                [
                    component_rank(
                        trainer, bundle, "asset", horizon, columns
                    )
                    for bundle in validation
                ],
                dtype=float,
            )
            weights = seed_weights(scores, method)
            if float(weights.sum()) > 0.0:
                output["asset_residual"][:, horizon, columns] = np.sum(
                    [
                        weight * bundle["asset_residual"][:, horizon, columns]
                        for weight, bundle in zip(weights, oos, strict=True)
                    ],
                    axis=0,
                )
                output["asset_logvar"][:, horizon, columns] = np.sum(
                    [
                        weight * bundle["asset_logvar"][:, horizon, columns]
                        for weight, bundle in zip(weights, oos, strict=True)
                    ],
                    axis=0,
                )
            rows.append(
                {
                    "method": method,
                    "task": "asset",
                    "module": module,
                    "horizon_position": horizon,
                    "validation_scores": ",".join(
                        f"{value:.6f}" for value in scores
                    ),
                    "seed_weights": ",".join(
                        f"{value:.6f}" for value in weights
                    ),
                }
            )

    for horizon in range(horizon_count):
        scores = np.asarray(
            [
                component_rank(trainer, bundle, "sleeve", horizon, None)
                for bundle in validation
            ],
            dtype=float,
        )
        weights = seed_weights(scores, method)
        if float(weights.sum()) > 0.0:
            output["sleeve_residual"][:, horizon] = np.sum(
                [
                    weight * bundle["sleeve_residual"][:, horizon]
                    for weight, bundle in zip(weights, oos, strict=True)
                ],
                axis=0,
            )
            output["sleeve_logvar"][:, horizon] = np.sum(
                [
                    weight * bundle["sleeve_logvar"][:, horizon]
                    for weight, bundle in zip(weights, oos, strict=True)
                ],
                axis=0,
            )
        rows.append(
            {
                "method": method,
                "task": "sleeve",
                "module": "",
                "horizon_position": horizon,
                "validation_scores": ",".join(
                    f"{value:.6f}" for value in scores
                ),
                "seed_weights": ",".join(
                    f"{value:.6f}" for value in weights
                ),
            }
        )
    return output, rows


def median_ensemble(
    bundles: list[dict[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    output = copy.deepcopy(bundles[0])
    for key in (
        "asset_residual",
        "asset_logvar",
        "sleeve_residual",
        "sleeve_logvar",
    ):
        output[key] = np.median(
            np.stack([bundle[key] for bundle in bundles], axis=0),
            axis=0,
        )
    return output


def result_rows(
    trainer,
    bundle: dict[str, np.ndarray],
    module_index: np.ndarray,
    horizons: list[int],
    method: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for module_position, module in enumerate(RISK_SLEEVES):
        columns = np.flatnonzero(module_index == module_position)
        for horizon_position, horizon_days in enumerate(horizons):
            rows.append(
                {
                    "method": method,
                    "task": "asset",
                    "module": module,
                    "horizon_days": horizon_days,
                    "oos_rank_ic": component_rank(
                        trainer,
                        bundle,
                        "asset",
                        horizon_position,
                        columns,
                    ),
                }
            )
    for horizon_position, horizon_days in enumerate(horizons):
        rows.append(
            {
                "method": method,
                "task": "sleeve",
                "module": "",
                "horizon_days": horizon_days,
                "oos_rank_ic": component_rank(
                    trainer,
                    bundle,
                    "sleeve",
                    horizon_position,
                    None,
                ),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--seeds", default="17,29,47")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/reports/causal_seed_ensemble_audit.csv"),
    )
    parser.add_argument(
        "--weights-output",
        type=Path,
        default=Path("output/reports/causal_seed_ensemble_weights.csv"),
    )
    args = parser.parse_args()

    trainer = load_trainer()
    cfg = load_config(args.config)
    arrays, metadata = load_arrays(cfg)
    _, validation_indices, oos_indices = fold_indices(
        arrays, metadata, cfg, args.year
    )
    device = torch.device("cpu")
    validation_loader = DataLoader(
        WindowDataset(
            arrays,
            validation_indices,
            int(metadata["sequence_length_days"]),
            int(pd.Timestamp(args.year, 1, 1).value),
        ),
        batch_size=int(cfg["model"]["batch_size"]),
        shuffle=False,
    )
    oos_loader = DataLoader(
        WindowDataset(
            arrays,
            oos_indices,
            int(metadata["sequence_length_days"]),
        ),
        batch_size=int(cfg["model"]["batch_size"]),
        shuffle=False,
    )
    seeds = [int(value) for value in args.seeds.split(",")]
    validation_bundles: list[dict[str, np.ndarray]] = []
    oos_bundles: list[dict[str, np.ndarray]] = []
    model_dir = ROOT / cfg["outputs"]["model_dir"]
    for seed in seeds:
        checkpoint = torch.load(
            model_dir / f"model_{args.year}_seed_{seed}.pt",
            map_location="cpu",
            weights_only=False,
        )
        model = trainer.make_model(arrays, metadata, cfg).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        validation_bundles.append(
            trainer.predict(model, validation_loader, device, cfg)
        )
        oos_bundles.append(trainer.predict(model, oos_loader, device, cfg))

    rows: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []
    horizons = [int(value) for value in metadata["horizons_days"]]
    for seed, bundle in zip(seeds, oos_bundles, strict=True):
        rows.extend(
            result_rows(
                trainer,
                bundle,
                arrays.module_index,
                horizons,
                f"seed_{seed}",
            )
        )
    equal = trainer.merge_seed_predictions(oos_bundles)
    rows.extend(
        result_rows(
            trainer, equal, arrays.module_index, horizons, "equal_mean"
        )
    )
    rows.extend(
        result_rows(
            trainer,
            median_ensemble(oos_bundles),
            arrays.module_index,
            horizons,
            "coordinate_median",
        )
    )
    for method in ("top2", "positive_equal", "positive_score", "best1"):
        ensemble, selected = causal_component_ensemble(
            trainer,
            validation_bundles,
            oos_bundles,
            arrays.module_index,
            method,
        )
        rows.extend(
            result_rows(
                trainer,
                ensemble,
                arrays.module_index,
                horizons,
                method,
            )
        )
        for row in selected:
            row["horizon_days"] = horizons[int(row.pop("horizon_position"))]
        weight_rows.extend(selected)

    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output, index=False, encoding="utf-8-sig")
    weights_output = ROOT / args.weights_output
    weights_output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(weight_rows).to_csv(
        weights_output, index=False, encoding="utf-8-sig"
    )
    summary = (
        frame.groupby("method", as_index=False)["oos_rank_ic"]
        .agg(["mean", "median", "min"])
        .sort_values("mean", ascending=False)
    )
    print(summary.to_string())
    print(f"\nSaved: {output}")
    print(f"Saved: {weights_output}")


if __name__ == "__main__":
    main()

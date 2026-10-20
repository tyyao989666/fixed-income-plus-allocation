from __future__ import annotations

"""Test a deterministic ridge head on frozen, point-in-time iTransformer states.

This is a diagnostic, not a deployment shortcut.  It keeps the causal
iTransformer representation and replaces only the high-variance neural output
head with a convex, strongly regularized cross-sectional predictor.
"""

import argparse
import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.hierarchical_asset_itransformer import (  # noqa: E402
    RISK_SLEEVES,
    WindowDataset,
    load_arrays,
    load_config,
    rank_ic,
)


def load_trainer():
    path = ROOT / "scripts" / "135_train_hierarchical_asset_itransformer_v7.py"
    spec = importlib.util.spec_from_file_location("ridge_head_trainer", path)
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
    candidates = arrays.sample_indices
    horizons = [int(value) for value in metadata["horizons_days"]]
    split_days = {
        int(value)
        for value in cfg["walk_forward"].get(
            "split_horizons_days", horizons
        )
    }
    split_positions = np.asarray(
        [
            position
            for position, horizon in enumerate(horizons)
            if horizon in split_days
        ],
        dtype=np.int64,
    )
    cutoff_ns = int(pd.Timestamp(year, 1, 1).value)
    eligible = candidates[dates[candidates] < pd.Timestamp(year, 1, 1)]
    stride = max(int(cfg["data"].get("train_stride_days", 1)), 1)
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
                & (
                    arrays.label_end[index, split_positions]
                    < cutoff_ns
                )
            )
            for index in eligible
        ],
        dtype=bool,
    )
    eligible = eligible[mature]
    minimum = int(cfg["walk_forward"]["minimum_train_samples"])
    minimum_validation = int(
        cfg["walk_forward"]["minimum_validation_samples"]
    )
    fixed_validation_days = int(
        cfg["walk_forward"].get("validation_window_days", 0) or 0
    )
    if fixed_validation_days > 0:
        validation_count = min(
            max(fixed_validation_days, minimum_validation),
            len(eligible) - minimum,
        )
        if validation_count < minimum_validation:
            raise ValueError(
                "Insufficient mature observations for the fixed validation window"
            )
        split = len(eligible) - validation_count
    else:
        split = int(
            len(eligible)
            * (1.0 - float(cfg["walk_forward"]["validation_fraction"]))
        )
        split = min(
            max(split, minimum),
            len(eligible) - minimum_validation,
        )
    preliminary_train = eligible[:split]
    validation = eligible[split:]
    validation_start_ns = int(dates[validation[0]].value)
    if bool(cfg["walk_forward"].get("purge_by_max_label_end", True)):
        keep = np.asarray(
            [
                np.all(
                    (
                        arrays.label_end[index, split_positions]
                        <= np.iinfo(np.int64).min
                    )
                    | (
                        arrays.label_end[index, split_positions]
                        < validation_start_ns
                    )
                )
                for index in preliminary_train
            ],
            dtype=bool,
        )
        train = preliminary_train[keep]
    else:
        train = preliminary_train
    oos = candidates[dates[candidates].year == int(year)]
    return train, validation, oos


def make_loader(
    arrays,
    indices: np.ndarray,
    sequence_length: int,
    batch_size: int,
    label_cutoff_ns: int | None,
) -> DataLoader:
    return DataLoader(
        WindowDataset(
            arrays,
            indices,
            sequence_length,
            label_cutoff_ns,
        ),
        batch_size=batch_size,
        shuffle=False,
    )


def encode(model, loader: DataLoader, trainer, device: torch.device):
    collected: dict[str, list[np.ndarray]] = {
        "index": [],
        "asset_state": [],
        "sleeve_state": [],
        "asset_target": [],
        "asset_mask": [],
        "sleeve_target": [],
        "sleeve_mask": [],
    }
    model.eval()
    with torch.no_grad():
        for raw in loader:
            batch = trainer.to_device(raw, device)
            asset_state, _, _, sleeve_state = model.encode_context(
                batch["local_values"],
                batch["local_masks"],
                batch["macro_values"],
            )
            values = {
                "index": batch["index"],
                "asset_state": asset_state,
                "sleeve_state": sleeve_state,
                "asset_target": batch["asset_target"],
                "asset_mask": batch["asset_mask"],
                "sleeve_target": batch["sleeve_target"],
                "sleeve_mask": batch["sleeve_mask"],
            }
            for key, value in values.items():
                collected[key].append(value.detach().cpu().numpy())
    return {
        key: np.concatenate(values, axis=0)
        for key, values in collected.items()
    }


def centered_by_date(values: np.ndarray) -> np.ndarray:
    return values - values.mean(axis=1, keepdims=True)


def append_scalar_feature(
    state: np.ndarray,
    values: np.ndarray | None,
) -> np.ndarray:
    if values is None:
        return state
    return np.concatenate(
        [state, np.asarray(values, dtype=np.float32)[..., None]],
        axis=-1,
    )


def component_score(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> float:
    return float(rank_ic(prediction, target, mask, None))


def rank_score_target(
    values: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    output = np.zeros_like(values, dtype=np.float32)
    for row in range(len(values)):
        valid = np.asarray(mask[row], dtype=bool)
        if int(valid.sum()) < 2:
            continue
        ranks = pd.Series(values[row, valid]).rank(
            method="average", pct=True
        ).to_numpy(dtype=np.float32)
        scores = (ranks - 0.5) * 2.0
        output[row, valid] = scores - float(scores.mean())
    return output


def fit_component(
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_mask: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    validation_mask: np.ndarray,
    oos_x: np.ndarray,
    alphas: list[float],
    validation_offset: np.ndarray | None = None,
    oos_offset: np.ndarray | None = None,
) -> tuple[np.ndarray, float, float]:
    train_rows = train_mask.reshape(-1)
    if int(train_rows.sum()) < max(train_x.shape[-1] * 2, 40):
        return np.zeros(oos_y_shape(validation_y, oos_x)), np.inf, 0.0
    x_train = train_x.reshape(-1, train_x.shape[-1])[train_rows]
    y_train = train_y.reshape(-1)[train_rows]
    scaler = StandardScaler().fit(x_train)
    x_train_scaled = scaler.transform(x_train)
    validation_flat = scaler.transform(
        validation_x.reshape(-1, validation_x.shape[-1])
    )
    oos_flat = scaler.transform(oos_x.reshape(-1, oos_x.shape[-1]))
    candidates: list[tuple[float, float, np.ndarray]] = []
    for alpha in sorted({float(value) for value in alphas}, reverse=True):
        estimator = Ridge(alpha=alpha, fit_intercept=True)
        estimator.fit(x_train_scaled, y_train)
        validation_prediction = estimator.predict(validation_flat).reshape(
            validation_y.shape
        )
        if validation_offset is not None:
            validation_prediction = (
                validation_prediction + validation_offset
            )
        validation_prediction = centered_by_date(validation_prediction)
        score = component_score(
            validation_prediction,
            validation_y,
            validation_mask,
        )
        oos_prediction = estimator.predict(oos_flat).reshape(
            oos_x.shape[:2]
        )
        if oos_offset is not None:
            oos_prediction = oos_prediction + oos_offset
        candidates.append((score, alpha, centered_by_date(oos_prediction)))
    # Exact zero correction is the causal fallback for residual fitting.
    fallback_validation = (
        np.zeros_like(validation_y)
        if validation_offset is None
        else centered_by_date(validation_offset)
    )
    fallback_oos = (
        np.zeros(oos_x.shape[:2], dtype=float)
        if oos_offset is None
        else centered_by_date(oos_offset)
    )
    candidates.append(
        (
            component_score(
                fallback_validation,
                validation_y,
                validation_mask,
            ),
            np.inf,
            fallback_oos,
        )
    )
    selected = max(
        candidates,
        key=lambda value: (
            -np.inf if not np.isfinite(value[0]) else value[0],
            value[1],
        ),
    )
    return selected[2], float(selected[1]), float(selected[0])


def oos_y_shape(validation_y: np.ndarray, oos_x: np.ndarray) -> tuple[int, int]:
    del validation_y
    return int(oos_x.shape[0]), int(oos_x.shape[1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--alphas",
        default="0.01,0.1,1,10,100,1000",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/reports/frozen_itransformer_ridge_head.csv"),
    )
    parser.add_argument(
        "--include-causal-anchor",
        action="store_true",
        help="Append the same-date causal momentum/Market View anchor.",
    )
    parser.add_argument(
        "--fit-residual-to-causal-anchor",
        action="store_true",
        help="Fit a correction to the causal anchor and score anchor+correction.",
    )
    parser.add_argument(
        "--fit-rank-target",
        action="store_true",
        help="Fit future within-cross-section ranks instead of noisy magnitudes.",
    )
    args = parser.parse_args()

    trainer = load_trainer()
    cfg = load_config(args.config)
    arrays, metadata = load_arrays(cfg)
    train_indices, validation_indices, oos_indices = fold_indices(
        arrays, metadata, cfg, args.year
    )
    dates = pd.to_datetime(arrays.dates)
    sequence_length = int(metadata["sequence_length_days"])
    batch_size = int(cfg["model"]["batch_size"])
    validation_start_ns = int(dates[validation_indices[0]].value)
    cutoff_ns = int(pd.Timestamp(args.year, 1, 1).value)
    train_loader = make_loader(
        arrays,
        train_indices,
        sequence_length,
        batch_size,
        validation_start_ns,
    )
    validation_loader = make_loader(
        arrays,
        validation_indices,
        sequence_length,
        batch_size,
        cutoff_ns,
    )
    oos_loader = make_loader(
        arrays,
        oos_indices,
        sequence_length,
        batch_size,
        None,
    )

    checkpoint_path = (
        ROOT
        / cfg["outputs"]["model_dir"]
        / f"model_{args.year}_seed_{args.seed}.pt"
    )
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    device = torch.device("cpu")
    model = trainer.make_model(arrays, metadata, cfg).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    train = encode(model, train_loader, trainer, device)
    validation = encode(model, validation_loader, trainer, device)
    oos = encode(model, oos_loader, trainer, device)
    neural = trainer.predict(model, oos_loader, device, cfg)
    raw_dataset = np.load(
        ROOT / cfg["outputs"]["dataset_npz"], allow_pickle=False
    )

    alphas = [float(value) for value in str(args.alphas).split(",")]
    horizons = [int(value) for value in metadata["horizons_days"]]
    rows: list[dict[str, Any]] = []
    for module_position, module in enumerate(RISK_SLEEVES):
        columns = np.flatnonzero(arrays.module_index == module_position)
        for horizon_position, horizon_days in enumerate(horizons):
            train_anchor = (
                raw_dataset["asset_causal_anchor"][
                    train["index"], horizon_position
                ][:, columns]
                if args.include_causal_anchor
                else None
            )
            validation_anchor = (
                raw_dataset["asset_causal_anchor"][
                    validation["index"], horizon_position
                ][:, columns]
                if args.include_causal_anchor
                else None
            )
            oos_anchor = (
                raw_dataset["asset_causal_anchor"][
                    oos["index"], horizon_position
                ][:, columns]
                if args.include_causal_anchor
                else None
            )
            residual_train_anchor = (
                raw_dataset["asset_causal_anchor"][
                    train["index"], horizon_position
                ][:, columns]
                if args.fit_residual_to_causal_anchor
                else None
            )
            residual_validation_anchor = (
                raw_dataset["asset_causal_anchor"][
                    validation["index"], horizon_position
                ][:, columns]
                if args.fit_residual_to_causal_anchor
                else None
            )
            residual_oos_anchor = (
                raw_dataset["asset_causal_anchor"][
                    oos["index"], horizon_position
                ][:, columns]
                if args.fit_residual_to_causal_anchor
                else None
            )
            train_fit_target = train["asset_target"][
                :, horizon_position, columns
            ]
            if args.fit_rank_target:
                train_fit_target = rank_score_target(
                    train_fit_target,
                    train["asset_mask"][:, horizon_position, columns],
                )
            if residual_train_anchor is not None:
                train_fit_target = (
                    train_fit_target - residual_train_anchor
                )
            ridge_prediction, alpha, validation_score = fit_component(
                append_scalar_feature(
                    train["asset_state"][:, columns], train_anchor
                ),
                train_fit_target,
                train["asset_mask"][:, horizon_position, columns],
                append_scalar_feature(
                    validation["asset_state"][:, columns],
                    validation_anchor,
                ),
                validation["asset_target"][:, horizon_position, columns],
                validation["asset_mask"][:, horizon_position, columns],
                append_scalar_feature(
                oos["asset_state"][:, columns], oos_anchor
                ),
                alphas,
                validation_offset=residual_validation_anchor,
                oos_offset=residual_oos_anchor,
            )
            ridge_oos = component_score(
                ridge_prediction,
                oos["asset_target"][:, horizon_position, columns],
                oos["asset_mask"][:, horizon_position, columns],
            )
            neural_oos = component_score(
                neural["asset_residual"][:, horizon_position, columns],
                oos["asset_target"][:, horizon_position, columns],
                oos["asset_mask"][:, horizon_position, columns],
            )
            baseline_oos = (
                component_score(
                    residual_oos_anchor,
                    oos["asset_target"][:, horizon_position, columns],
                    oos["asset_mask"][:, horizon_position, columns],
                )
                if residual_oos_anchor is not None
                else 0.0
            )
            rows.append(
                {
                    "year": args.year,
                    "task": "asset",
                    "module": module,
                    "horizon_days": horizon_days,
                    "selected_alpha": alpha,
                    "validation_rank_ic": validation_score,
                    "ridge_oos_rank_ic": ridge_oos,
                    "baseline_oos_rank_ic": baseline_oos,
                    "ridge_edge_vs_baseline": ridge_oos - baseline_oos,
                    "neural_head_oos_rank_ic": neural_oos,
                    "fit_residual_to_causal_anchor": bool(
                        args.fit_residual_to_causal_anchor
                    ),
                    "fit_rank_target": bool(args.fit_rank_target),
                }
            )

    for horizon_position, horizon_days in enumerate(horizons):
        train_anchor = (
            raw_dataset["sleeve_causal_anchor"][
                train["index"], horizon_position
            ]
            if args.include_causal_anchor
            else None
        )
        validation_anchor = (
            raw_dataset["sleeve_causal_anchor"][
                validation["index"], horizon_position
            ]
            if args.include_causal_anchor
            else None
        )
        oos_anchor = (
            raw_dataset["sleeve_causal_anchor"][
                oos["index"], horizon_position
            ]
            if args.include_causal_anchor
            else None
        )
        residual_train_anchor = (
            raw_dataset["sleeve_causal_anchor"][
                train["index"], horizon_position
            ]
            if args.fit_residual_to_causal_anchor
            else None
        )
        residual_validation_anchor = (
            raw_dataset["sleeve_causal_anchor"][
                validation["index"], horizon_position
            ]
            if args.fit_residual_to_causal_anchor
            else None
        )
        residual_oos_anchor = (
            raw_dataset["sleeve_causal_anchor"][
                oos["index"], horizon_position
            ]
            if args.fit_residual_to_causal_anchor
            else None
        )
        train_fit_target = train["sleeve_target"][:, horizon_position]
        if args.fit_rank_target:
            train_fit_target = rank_score_target(
                train_fit_target,
                train["sleeve_mask"][:, horizon_position],
            )
        if residual_train_anchor is not None:
            train_fit_target = train_fit_target - residual_train_anchor
        ridge_prediction, alpha, validation_score = fit_component(
            append_scalar_feature(train["sleeve_state"], train_anchor),
            train_fit_target,
            train["sleeve_mask"][:, horizon_position],
            append_scalar_feature(
                validation["sleeve_state"], validation_anchor
            ),
            validation["sleeve_target"][:, horizon_position],
            validation["sleeve_mask"][:, horizon_position],
            append_scalar_feature(oos["sleeve_state"], oos_anchor),
            alphas,
            validation_offset=residual_validation_anchor,
            oos_offset=residual_oos_anchor,
        )
        ridge_oos = component_score(
            ridge_prediction,
            oos["sleeve_target"][:, horizon_position],
            oos["sleeve_mask"][:, horizon_position],
        )
        baseline_oos = (
            component_score(
                residual_oos_anchor,
                oos["sleeve_target"][:, horizon_position],
                oos["sleeve_mask"][:, horizon_position],
            )
            if residual_oos_anchor is not None
            else 0.0
        )
        rows.append(
            {
                    "year": args.year,
                    "task": "sleeve",
                    "module": "",
                    "horizon_days": horizon_days,
                    "include_causal_anchor": bool(
                        args.include_causal_anchor
                    ),
                "selected_alpha": alpha,
                "validation_rank_ic": validation_score,
                "ridge_oos_rank_ic": ridge_oos,
                "baseline_oos_rank_ic": baseline_oos,
                "ridge_edge_vs_baseline": ridge_oos - baseline_oos,
                "neural_head_oos_rank_ic": component_score(
                    neural["sleeve_residual"][:, horizon_position],
                    oos["sleeve_target"][:, horizon_position],
                    oos["sleeve_mask"][:, horizon_position],
                ),
                "fit_residual_to_causal_anchor": bool(
                    args.fit_residual_to_causal_anchor
                ),
                "fit_rank_target": bool(args.fit_rank_target),
            }
        )

    for row in rows:
        row.setdefault(
            "include_causal_anchor", bool(args.include_causal_anchor)
        )
        row.setdefault(
            "fit_residual_to_causal_anchor",
            bool(args.fit_residual_to_causal_anchor),
        )
        row.setdefault("fit_rank_target", bool(args.fit_rank_target))
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows)
    result.to_csv(output, index=False, encoding="utf-8-sig")
    print(result.to_string(index=False))
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()

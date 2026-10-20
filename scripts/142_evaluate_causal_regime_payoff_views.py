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
    rank_ic,
)


def make_model(arrays, metadata: dict[str, Any], cfg: dict[str, Any]) -> HierarchicalAssetITransformer:
    return HierarchicalAssetITransformer(
        sequence_length=int(metadata["sequence_length_days"]),
        local_features=len(metadata["local_features"]),
        macro_features=len(metadata["macro_features"]),
        module_index=arrays.module_index,
        horizons=len(metadata["horizons_days"]),
        cfg=cfg["model"],
    )


def prediction_bundle(model, arrays, indices, metadata, cfg, cutoff_ns: int | None = None):
    loader = DataLoader(
        WindowDataset(arrays, indices, int(metadata["sequence_length_days"]), cutoff_ns),
        batch_size=int(cfg["model"]["batch_size"]), shuffle=False
    )
    output: dict[str, list[np.ndarray]] = {
        "index": [], "current": [], "probability": [], "target": [], "mask": [],
        "sleeve_target": [], "sleeve_mask": [], "sleeve_baseline": []
    }
    model.eval()
    use_path_occupancy = cfg["model"].get("transition_target") == "path_occupancy"
    with torch.no_grad():
        for batch in loader:
            values = model(
                batch["local_values"], batch["local_masks"], batch["macro_values"],
                batch["regime_index"], batch["regime_prior_count"]
            )
            items = {
                "index": batch["index"], "current": batch["regime_index"],
                "probability": values[8] if len(values) == 10 else values[6],
                "target": (
                    batch["regime_occupancy_target"]
                    if use_path_occupancy else batch["regime_target"]
                ),
                "mask": (
                    batch["regime_occupancy_mask"]
                    if use_path_occupancy else batch["regime_target_mask"]
                ),
                "sleeve_target": batch["sleeve_target"],
                "sleeve_mask": batch["sleeve_mask"], "sleeve_baseline": batch["sleeve_baseline"],
            }
            for key, value in items.items():
                output[key].append(value.numpy())
    return {key: np.concatenate(values) for key, values in output.items()}


def persistence_probability(current: np.ndarray, classes: int) -> np.ndarray:
    probability = np.full((len(current), classes), 1.0 / classes, dtype=np.float32)
    known = current < classes
    probability[known] = 0.0
    probability[np.flatnonzero(known), current[known].astype(int)] = 1.0
    return probability


def select_transition_blend(bundle, horizons: list[int], classes: int) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    alpha_grid = [0.0, 0.25, 0.50, 0.75, 1.0]
    persistence = persistence_probability(bundle["current"], classes)
    for position, horizon in enumerate(horizons):
        valid = bundle["mask"][:, position].astype(bool)
        target_values = bundle["target"][valid, position]
        if target_values.ndim == 2:
            target_distribution = target_values.astype(float)
            target = target_distribution.argmax(axis=1)
        else:
            target = target_values.astype(int)
            target_distribution = np.eye(classes, dtype=float)[target]
        best: dict[str, float] | None = None
        for alpha in alpha_grid:
            probability = (
                alpha * bundle["probability"][valid, position]
                + (1.0 - alpha) * persistence[valid]
            )
            nll = float(
                -(target_distribution * np.log(np.clip(probability, 1e-12, 1.0)))
                .sum(axis=1)
                .mean()
            )
            accuracy = float(np.mean(probability.argmax(axis=1) == target))
            candidate = {"horizon_days": horizon, "transition_alpha": alpha, "nll": nll, "accuracy": accuracy}
            if best is None or (candidate["nll"], -candidate["accuracy"], alpha) < (
                best["nll"], -best["accuracy"], best["transition_alpha"]
            ):
                best = candidate
        rows.append(best or {})
    return rows


def robust_mean(values: np.ndarray) -> np.ndarray:
    result = np.full(values.shape[1], np.nan, dtype=float)
    for column in range(values.shape[1]):
        x = values[:, column]
        x = x[np.isfinite(x)]
        if not len(x):
            continue
        low, high = np.quantile(x, [0.05, 0.95]) if len(x) >= 20 else (x.min(), x.max())
        result[column] = np.mean(np.clip(x, low, high))
    return result


def hierarchical_payoff_table(
    arrays, indices: np.ndarray, horizon_position: int, cutoff_ns: int, classes: int,
    state_prior: float = 60.0, axis_prior: float = 120.0,
) -> tuple[np.ndarray, np.ndarray]:
    regime_valid = (
        arrays.regime_target_mask[indices, horizon_position].astype(bool)
        & (arrays.regime_label_end[indices, horizon_position] < cutoff_ns)
    )
    sleeve_valid = arrays.sleeve_target_mask[indices, horizon_position].astype(bool)
    states = arrays.regime_target[indices, horizon_position]
    target = arrays.sleeve_target[indices, horizon_position].astype(float).copy()
    target[~sleeve_valid] = np.nan
    valid_rows = regime_valid & (states >= 0) & (states < classes)
    states = states[valid_rows].astype(int)
    target = target[valid_rows]
    global_mean = robust_mean(target)
    table = np.zeros((classes, target.shape[1]), dtype=float)
    counts = np.zeros(classes, dtype=int)
    bond_means: list[np.ndarray] = []
    equity_means: list[np.ndarray] = []
    for bond in range(3):
        selected = np.isin(states, np.arange(bond * 3, bond * 3 + 3))
        mean = robust_mean(target[selected]) if selected.any() else global_mean
        weight = selected.sum() / (selected.sum() + axis_prior)
        bond_means.append(weight * mean + (1.0 - weight) * global_mean)
    for equity in range(3):
        selected = states % 3 == equity
        mean = robust_mean(target[selected]) if selected.any() else global_mean
        weight = selected.sum() / (selected.sum() + axis_prior)
        equity_means.append(weight * mean + (1.0 - weight) * global_mean)
    for state in range(classes):
        selected = states == state
        counts[state] = int(selected.sum())
        axis_mean = 0.5 * (bond_means[state // 3] + equity_means[state % 3])
        state_mean = robust_mean(target[selected]) if selected.any() else axis_mean
        weight = counts[state] / (counts[state] + state_prior)
        table[state] = weight * state_mean + (1.0 - weight) * axis_mean
    table = np.nan_to_num(table, nan=0.0)
    table -= table.mean(axis=1, keepdims=True)
    return table, counts


def select_view_blend(bundle, payoffs, transition_rows, horizons, classes):
    persistence = persistence_probability(bundle["current"], classes)
    baseline_grid = [0.0, 0.25, 0.50, 0.75, 1.0]
    view_grid = [0.10, 0.25, 0.50, 1.0]
    rows: list[dict[str, float]] = []
    for position, horizon in enumerate(horizons):
        alpha = transition_rows[position]["transition_alpha"]
        probability = alpha * bundle["probability"][:, position] + (1.0 - alpha) * persistence
        view = probability @ payoffs[position]
        best: dict[str, float] | None = None
        for baseline_weight in baseline_grid:
            for view_weight in view_grid:
                prediction = baseline_weight * bundle["sleeve_baseline"][:, position] + view_weight * view
                score = rank_ic(
                    prediction, bundle["sleeve_target"][:, position],
                    bundle["sleeve_mask"][:, position], None
                )
                candidate = {
                    "horizon_days": horizon, "baseline_weight": baseline_weight,
                    "view_weight": view_weight, "validation_rank_ic": score,
                }
                if best is None or (score, -baseline_weight - view_weight) > (
                    best["validation_rank_ic"], -best["baseline_weight"] - best["view_weight"]
                ):
                    best = candidate
        rows.append(best or {})
    return rows


def mechanical_grid_table(classes: int) -> np.ndarray:
    table = np.zeros((classes, 5), dtype=float)
    axis = np.array([1.0, 0.0, -1.0])
    for state in range(classes):
        bond = axis[state // 3]
        equity = axis[state % 3]
        defensive = float(equity < 0.0)
        table[state] = [
            bond,
            0.70 * bond,
            0.65 * equity + 0.35 * bond,
            equity,
            0.25 * bond - 0.20 * equity + 0.35 * defensive,
        ]
        table[state] -= table[state].mean()
    return table


def select_mechanical_grid_params(bundle, transition_rows, horizons, classes):
    table = mechanical_grid_table(classes)
    persistence = persistence_probability(bundle["current"], classes)
    current_view = persistence @ table
    rows: list[dict[str, float]] = []
    for position, horizon in enumerate(horizons):
        alpha = transition_rows[position]["transition_alpha"]
        probability = alpha * bundle["probability"][:, position] + (1.0 - alpha) * persistence
        transition_delta = probability @ table - current_view
        blocks = np.array_split(np.arange(len(current_view)), 4)
        candidates: list[dict[str, float]] = []
        for baseline_weight in [0.0, 0.25, 0.50, 1.0]:
            for grid_weight in [0.25, 0.50, 1.0]:
                for delta_weight in [0.0, 0.10, 0.25, 0.50]:
                    score = (
                        baseline_weight * bundle["sleeve_baseline"][:, position]
                        + grid_weight * current_view
                        + delta_weight * transition_delta
                    )
                    overall = rank_ic(
                        score, bundle["sleeve_target"][:, position],
                        bundle["sleeve_mask"][:, position], None
                    )
                    block_scores = np.array([
                        rank_ic(
                            score[block], bundle["sleeve_target"][block, position],
                            bundle["sleeve_mask"][block, position], None
                        )
                        for block in blocks
                    ], dtype=float)
                    finite = block_scores[np.isfinite(block_scores)]
                    robust = float(finite.mean() - 0.25 * finite.std()) if len(finite) else -np.inf
                    positive_fraction = float(np.mean(finite > 0.0)) if len(finite) else 0.0
                    candidates.append({
                        "horizon_days": horizon, "baseline_weight": baseline_weight,
                        "grid_weight": grid_weight, "transition_delta_weight": delta_weight,
                        "validation_rank_ic": overall, "validation_robust_rank": robust,
                        "positive_block_fraction": positive_fraction,
                    })
        no_transition = max(
            (row for row in candidates if row["transition_delta_weight"] == 0.0),
            key=lambda row: (row["validation_robust_rank"], row["validation_rank_ic"]),
        )
        with_transition = max(
            (row for row in candidates if row["transition_delta_weight"] > 0.0),
            key=lambda row: (row["validation_robust_rank"], row["validation_rank_ic"]),
        )
        if (
            with_transition["positive_block_fraction"] >= 0.75
            and with_transition["validation_robust_rank"]
            >= no_transition["validation_robust_rank"] + 0.01
        ):
            selected = with_transition
            selected["transition_gate"] = 1.0
        else:
            selected = no_transition
            selected["transition_gate"] = 0.0
        rows.append(selected)
    return rows


def select_nine_grid_hybrid_params(bundle, payoffs, transition_rows, horizons, classes):
    table = mechanical_grid_table(classes)
    persistence = persistence_probability(bundle["current"], classes)
    current_grid = persistence @ table
    blocks = np.array_split(np.arange(len(current_grid)), 4)
    rows: list[dict[str, float]] = []
    for position, horizon in enumerate(horizons):
        alpha = transition_rows[position]["transition_alpha"]
        future_probability = (
            alpha * bundle["probability"][:, position]
            + (1.0 - alpha) * persistence
        )
        empirical_path_view = future_probability @ payoffs[position]
        candidates: list[dict[str, float]] = []
        for baseline_weight in [0.0, 0.25, 0.50, 1.0]:
            for grid_weight in [0.25, 0.50, 1.0]:
                for path_weight in [0.0, 0.10, 0.25, 0.50, 1.0]:
                    score = (
                        baseline_weight * bundle["sleeve_baseline"][:, position]
                        + grid_weight * current_grid
                        + path_weight * empirical_path_view
                    )
                    overall = rank_ic(
                        score, bundle["sleeve_target"][:, position],
                        bundle["sleeve_mask"][:, position], None
                    )
                    block_scores = np.array([
                        rank_ic(
                            score[block], bundle["sleeve_target"][block, position],
                            bundle["sleeve_mask"][block, position], None
                        )
                        for block in blocks
                    ], dtype=float)
                    finite = block_scores[np.isfinite(block_scores)]
                    robust = float(finite.mean() - 0.25 * finite.std()) if len(finite) else -np.inf
                    positive_fraction = float(np.mean(finite > 0.0)) if len(finite) else 0.0
                    candidates.append({
                        "horizon_days": horizon,
                        "baseline_weight": baseline_weight,
                        "grid_weight": grid_weight,
                        "path_overlay_weight": path_weight,
                        "validation_rank_ic": overall,
                        "validation_robust_rank": robust,
                        "positive_block_fraction": positive_fraction,
                    })
        no_overlay = max(
            (row for row in candidates if row["path_overlay_weight"] == 0.0),
            key=lambda row: (row["validation_robust_rank"], row["validation_rank_ic"]),
        )
        with_overlay = max(
            (row for row in candidates if row["path_overlay_weight"] > 0.0),
            key=lambda row: (row["validation_robust_rank"], row["validation_rank_ic"]),
        )
        if (
            with_overlay["positive_block_fraction"] == 1.0
            and with_overlay["validation_robust_rank"]
            >= no_overlay["validation_robust_rank"] + 0.02
        ):
            selected = with_overlay
            selected["path_overlay_gate"] = 1.0
        else:
            selected = no_overlay
            selected["path_overlay_gate"] = 0.0
        rows.append(selected)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate causal hierarchical 3x3-grid payoff views.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--year", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output-prefix")
    args = parser.parse_args()
    cfg = load_config(ROOT / args.config)
    arrays, metadata = load_arrays(cfg)
    dates = pd.to_datetime(arrays.dates)
    cutoff = pd.Timestamp(args.year, 1, 1)
    cutoff_ns = int(cutoff.value)
    candidate = arrays.sample_indices
    eligible = candidate[dates[candidate] < cutoff]
    mature = np.array([
        np.any((arrays.label_end[index] > np.iinfo(np.int64).min) & (arrays.label_end[index] < cutoff_ns))
        for index in eligible
    ])
    eligible = eligible[mature]
    split = int(len(eligible) * (1.0 - float(cfg["walk_forward"]["validation_fraction"])))
    split = min(
        max(split, int(cfg["walk_forward"]["minimum_train_samples"])),
        len(eligible) - int(cfg["walk_forward"]["minimum_validation_samples"]),
    )
    preliminary_train, validation_indices = eligible[:split], eligible[split:]
    validation_start_ns = int(dates[validation_indices[0]].value)
    train_keep = np.array([
        np.all((arrays.label_end[index] <= np.iinfo(np.int64).min) | (arrays.label_end[index] < validation_start_ns))
        for index in preliminary_train
    ])
    train_indices = preliminary_train[train_keep]
    prediction_indices = candidate[dates[candidate].year == args.year]

    checkpoint = torch.load(
        project_path(cfg, "model_dir") / f"model_{args.year}_seed_{args.seed}.pt",
        map_location="cpu", weights_only=False
    )
    model = make_model(arrays, metadata, cfg)
    model.load_state_dict(checkpoint["state_dict"])
    validation_bundle = prediction_bundle(
        model, arrays, validation_indices, metadata, cfg, cutoff_ns
    )
    prediction = prediction_bundle(model, arrays, prediction_indices, metadata, cfg)
    horizons = [int(value) for value in metadata["horizons_days"]]
    classes = int(cfg["model"]["transition_classes"])
    transition_rows = select_transition_blend(validation_bundle, horizons, classes)

    train_payoffs, full_payoffs, count_rows = [], [], []
    for position, horizon in enumerate(horizons):
        train_table, train_counts = hierarchical_payoff_table(
            arrays, train_indices, position, validation_start_ns, classes
        )
        full_table, full_counts = hierarchical_payoff_table(
            arrays, eligible, position, cutoff_ns, classes
        )
        train_payoffs.append(train_table)
        full_payoffs.append(full_table)
        for state in range(classes):
            count_rows.append({
                "horizon_days": horizon, "regime": metadata["regime_names"][state],
                "train_count": train_counts[state], "full_pre_prediction_count": full_counts[state],
            })
    blend_rows = select_view_blend(
        validation_bundle, train_payoffs, transition_rows, horizons, classes
    )
    persistence = persistence_probability(prediction["current"], classes)
    result_rows, daily_rows = [], []
    for position, horizon in enumerate(horizons):
        alpha = transition_rows[position]["transition_alpha"]
        probability = alpha * prediction["probability"][:, position] + (1.0 - alpha) * persistence
        view = probability @ full_payoffs[position]
        params = blend_rows[position]
        final_score = (
            params["baseline_weight"] * prediction["sleeve_baseline"][:, position]
            + params["view_weight"] * view
        )
        model_ic = rank_ic(
            final_score, prediction["sleeve_target"][:, position],
            prediction["sleeve_mask"][:, position], None
        )
        baseline_ic = rank_ic(
            prediction["sleeve_baseline"][:, position], prediction["sleeve_target"][:, position],
            prediction["sleeve_mask"][:, position], None
        )
        raw_view_ic = rank_ic(
            view, prediction["sleeve_target"][:, position],
            prediction["sleeve_mask"][:, position], None
        )
        result_rows.append({
            "year": args.year, "horizon_days": horizon, **transition_rows[position], **params,
            "oos_rank_ic": model_ic, "baseline_rank_ic": baseline_ic,
            "raw_view_rank_ic": raw_view_ic,
            "incremental_rank_ic": model_ic - baseline_ic,
        })
        for row, array_index in enumerate(prediction["index"].astype(int)):
            daily_rows.append({
                "as_of_date": dates[array_index], "horizon_days": horizon,
                "current_regime": metadata["regime_names"][int(prediction["current"][row])],
                **{f"view_{metadata['target_sleeves'][sleeve]}": view[row, sleeve] for sleeve in range(view.shape[1])},
                **{f"score_{metadata['target_sleeves'][sleeve]}": final_score[row, sleeve] for sleeve in range(view.shape[1])},
            })
    mechanical_params = select_mechanical_grid_params(
        validation_bundle, transition_rows, horizons, classes
    )
    mechanical_table = mechanical_grid_table(classes)
    current_probability = persistence_probability(prediction["current"], classes)
    current_grid_view = current_probability @ mechanical_table
    mechanical_rows, mechanical_daily_rows = [], []
    for position, horizon in enumerate(horizons):
        transition_alpha = transition_rows[position]["transition_alpha"]
        future_probability = (
            transition_alpha * prediction["probability"][:, position]
            + (1.0 - transition_alpha) * current_probability
        )
        transition_delta = future_probability @ mechanical_table - current_grid_view
        params = mechanical_params[position]
        score = (
            params["baseline_weight"] * prediction["sleeve_baseline"][:, position]
            + params["grid_weight"] * current_grid_view
            + params["transition_delta_weight"] * transition_delta
        )
        score_ic = rank_ic(
            score, prediction["sleeve_target"][:, position],
            prediction["sleeve_mask"][:, position], None
        )
        current_grid_ic = rank_ic(
            current_grid_view, prediction["sleeve_target"][:, position],
            prediction["sleeve_mask"][:, position], None
        )
        delta_ic = rank_ic(
            transition_delta, prediction["sleeve_target"][:, position],
            prediction["sleeve_mask"][:, position], None
        )
        baseline_ic = rank_ic(
            prediction["sleeve_baseline"][:, position], prediction["sleeve_target"][:, position],
            prediction["sleeve_mask"][:, position], None
        )
        mechanical_rows.append({
            "year": args.year, "horizon_days": horizon,
            "transition_alpha": transition_alpha, **params,
            "oos_rank_ic": score_ic, "baseline_rank_ic": baseline_ic,
            "current_grid_rank_ic": current_grid_ic,
            "transition_delta_rank_ic": delta_ic,
            "incremental_vs_baseline": score_ic - baseline_ic,
        })
        for row, array_index in enumerate(prediction["index"].astype(int)):
            mechanical_daily_rows.append({
                "as_of_date": dates[array_index], "horizon_days": horizon,
                "current_regime": metadata["regime_names"][int(prediction["current"][row])],
                **{
                    f"current_grid_{metadata['target_sleeves'][sleeve]}": current_grid_view[row, sleeve]
                    for sleeve in range(current_grid_view.shape[1])
                },
                **{
                    f"transition_delta_{metadata['target_sleeves'][sleeve]}": transition_delta[row, sleeve]
                    for sleeve in range(transition_delta.shape[1])
                },
                **{
                    f"mechanical_score_{metadata['target_sleeves'][sleeve]}": score[row, sleeve]
                    for sleeve in range(score.shape[1])
                },
            })
    hybrid_params = select_nine_grid_hybrid_params(
        validation_bundle, train_payoffs, transition_rows, horizons, classes
    )
    hybrid_rows, hybrid_daily_rows = [], []
    for position, horizon in enumerate(horizons):
        transition_alpha = transition_rows[position]["transition_alpha"]
        future_probability = (
            transition_alpha * prediction["probability"][:, position]
            + (1.0 - transition_alpha) * current_probability
        )
        empirical_path_view = future_probability @ full_payoffs[position]
        params = hybrid_params[position]
        score = (
            params["baseline_weight"] * prediction["sleeve_baseline"][:, position]
            + params["grid_weight"] * current_grid_view
            + params["path_overlay_weight"] * empirical_path_view
        )
        score_ic = rank_ic(
            score, prediction["sleeve_target"][:, position],
            prediction["sleeve_mask"][:, position], None
        )
        hybrid_rows.append({
            "year": args.year, "horizon_days": horizon,
            "transition_alpha": transition_alpha, **params,
            "oos_rank_ic": score_ic,
            "current_grid_rank_ic": rank_ic(
                current_grid_view, prediction["sleeve_target"][:, position],
                prediction["sleeve_mask"][:, position], None
            ),
            "empirical_path_view_rank_ic": rank_ic(
                empirical_path_view, prediction["sleeve_target"][:, position],
                prediction["sleeve_mask"][:, position], None
            ),
        })
        for row, array_index in enumerate(prediction["index"].astype(int)):
            hybrid_daily_rows.append({
                "as_of_date": dates[array_index], "horizon_days": horizon,
                "current_regime": metadata["regime_names"][int(prediction["current"][row])],
                **{
                    f"path_view_{metadata['target_sleeves'][sleeve]}": empirical_path_view[row, sleeve]
                    for sleeve in range(empirical_path_view.shape[1])
                },
                **{
                    f"hybrid_score_{metadata['target_sleeves'][sleeve]}": score[row, sleeve]
                    for sleeve in range(score.shape[1])
                },
            })
    prefix = (
        ROOT / args.output_prefix if args.output_prefix
        else ROOT / "output" / "reports" / f"{Path(args.config).stem}_{args.year}_causal_payoff"
    )
    outputs = {
        "result": (pd.DataFrame(result_rows), Path(f"{prefix}_validation.csv")),
        "counts": (pd.DataFrame(count_rows), Path(f"{prefix}_state_counts.csv")),
        "daily": (pd.DataFrame(daily_rows), Path(f"{prefix}_daily.csv")),
        "mechanical": (
            pd.DataFrame(mechanical_rows), Path(f"{prefix}_mechanical_validation.csv")
        ),
        "mechanical_daily": (
            pd.DataFrame(mechanical_daily_rows), Path(f"{prefix}_mechanical_daily.csv")
        ),
        "hybrid": (
            pd.DataFrame(hybrid_rows), Path(f"{prefix}_nine_grid_hybrid_validation.csv")
        ),
        "hybrid_daily": (
            pd.DataFrame(hybrid_daily_rows), Path(f"{prefix}_nine_grid_hybrid_daily.csv")
        ),
    }
    for _, (frame, path) in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False, encoding="utf-8-sig")
    print(outputs["result"][0].to_string(index=False))
    print("\nMechanical v4.3 grid mapping")
    print(outputs["mechanical"][0].to_string(index=False))
    print("\nNine-grid core plus causal path overlay")
    print(outputs["hybrid"][0].to_string(index=False))
    for name, (_, path) in outputs.items():
        print(f"{name}: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml


# Small attention matrices are substantially slower when the Windows CPU
# backend over-subscribes threads. Set a conservative pool before importing
# project modules that may initialize PyTorch kernels.
torch.set_num_threads(8)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.itransformer_daily_signal import load_daily_config  # noqa: E402
from src.itransformer_daily_timing_v2 import (  # noqa: E402
    IdentifiedInvertedTransformer,
    nested_fold_masks,
    load_v2_arrays,
    v2_path,
)
from src.itransformer_signal import RISK_SLEEVES, prediction_metrics, set_seed  # noqa: E402


def merge_config(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merge_config(base[key], value)
        else:
            base[key] = value
    return base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the daily multi-horizon iTransformer v2 model.")
    parser.add_argument(
        "--override",
        type=Path,
        help="Optional YAML file merged over config/itransformer_daily_config.yaml.",
    )
    return parser.parse_args()


def make_model(x: np.ndarray, y: np.ndarray, cfg: dict[str, Any]) -> IdentifiedInvertedTransformer:
    return IdentifiedInvertedTransformer(
        seq_len=x.shape[1],
        n_features=x.shape[2],
        n_targets=y.shape[1] * y.shape[2],
        d_model=int(cfg["d_model"]),
        n_heads=int(cfg["n_heads"]),
        num_layers=int(cfg["num_layers"]),
        dropout=float(cfg["dropout"]),
    )


def predict(model: nn.Module, x: np.ndarray, shape: tuple[int, int], batch_size: int = 256) -> np.ndarray:
    model.eval()
    parts: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            tx = torch.tensor(x[start : start + batch_size], dtype=torch.float32)
            parts.append(model(tx).cpu().numpy())
    flat = np.vstack(parts) if parts else np.empty((0, shape[0] * shape[1]))
    return flat.reshape(len(flat), shape[0], shape[1])


def pairwise_rank_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_diff = pred.unsqueeze(-1) - pred.unsqueeze(-2)
    target_diff = target.unsqueeze(-1) - target.unsqueeze(-2)
    direction = torch.sign(target_diff)
    valid = direction.ne(0)
    if not valid.any():
        return pred.new_tensor(0.0)
    return torch.nn.functional.softplus(-direction[valid] * pred_diff[valid]).mean()


def loss_components(pred: torch.Tensor, target: torch.Tensor, cfg: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    horizons = pred.shape[1]
    configured_weights = cfg.get("horizon_loss_weights")
    if configured_weights is not None and len(configured_weights) == horizons:
        horizon_weights = pred.new_tensor([float(value) for value in configured_weights])
        horizon_weights = horizon_weights / horizon_weights.sum()
    elif horizons == 3:
        horizon_weights = pred.new_tensor([0.20, 0.60, 0.20])
    else:
        horizon_weights = pred.new_full((horizons,), 1.0 / horizons)
    huber_by_horizon = torch.stack(
        [nn.functional.smooth_l1_loss(pred[:, index], target[:, index]) for index in range(horizons)]
    )
    huber = torch.sum(huber_by_horizon * horizon_weights)
    rank_by_horizon = torch.stack(
        [pairwise_rank_loss(pred[:, index], target[:, index]) for index in range(horizons)]
    )
    rank = torch.sum(rank_by_horizon * horizon_weights)
    total = huber + float(cfg["rank_loss_weight"]) * rank
    return total, huber, rank


def primary_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    raw_future: np.ndarray,
    mask: np.ndarray,
    primary_index: int,
) -> dict[str, float]:
    selected_pred = pred[mask, primary_index]
    selected_target = target[mask, primary_index]
    metrics = prediction_metrics(selected_pred, selected_target)
    if not len(selected_pred):
        return metrics
    order = np.argsort(selected_pred, axis=1)
    top = order[:, -2:]
    bottom = order[:, :2]
    realized = raw_future[mask, primary_index]
    top_return = np.array([realized[index, top[index]].mean() for index in range(len(top))])
    bottom_return = np.array([realized[index, bottom[index]].mean() for index in range(len(bottom))])
    metrics["top2_raw_return"] = float(np.nanmean(top_return))
    metrics["bottom2_raw_return"] = float(np.nanmean(bottom_return))
    metrics["top_bottom_raw_spread"] = float(np.nanmean(top_return - bottom_return))
    return metrics


def select_residual_alpha(
    residual_prediction: np.ndarray,
    baseline: np.ndarray,
    target: np.ndarray,
    raw_future: np.ndarray,
    mask: np.ndarray,
    primary_index: int,
    grid: list[float],
) -> tuple[float, dict[str, float]]:
    candidates: list[tuple[float, float, dict[str, float]]] = []
    for value in sorted(float(alpha) for alpha in grid):
        prediction = baseline + value * residual_prediction
        metrics = primary_metrics(prediction, target, raw_future, mask, primary_index)
        score = float(metrics.get("rank_ic", np.nan))
        candidates.append((score if np.isfinite(score) else -np.inf, value, metrics))
    best = max(candidates, key=lambda item: (item[0], -item[1]))
    return best[1], best[2]


def train_with_validation(
    fold_name: str,
    seed: int,
    x: np.ndarray,
    y: np.ndarray,
    train_mask: np.ndarray,
    validation_mask: np.ndarray,
    cfg: dict[str, Any],
    primary_index: int,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]], dict[str, Any]]:
    set_seed(seed)
    model = make_model(x, y, cfg)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    train_idx = np.flatnonzero(train_mask)
    val_idx = np.flatnonzero(validation_mask)
    rng = np.random.default_rng(seed)
    best_score = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    wait = 0
    logs: list[dict[str, Any]] = []
    shape = (y.shape[1], y.shape[2])
    for epoch in range(1, int(cfg["epochs"]) + 1):
        model.train()
        order = rng.permutation(train_idx)
        batch_losses: list[float] = []
        batch_huber: list[float] = []
        batch_rank: list[float] = []
        for start in range(0, len(order), int(cfg["batch_size"])):
            idx = order[start : start + int(cfg["batch_size"])]
            tx = torch.tensor(x[idx], dtype=torch.float32)
            ty = torch.tensor(y[idx], dtype=torch.float32)
            pred = model(tx).reshape(len(idx), *shape)
            total, huber, rank = loss_components(pred, ty, cfg)
            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            batch_losses.append(float(total.detach()))
            batch_huber.append(float(huber.detach()))
            batch_rank.append(float(rank.detach()))

        val_pred = predict(model, x[val_idx], shape)
        val_target = y[val_idx]
        val_huber = float(nn.functional.smooth_l1_loss(
            torch.tensor(val_pred[:, primary_index]),
            torch.tensor(val_target[:, primary_index]),
        ))
        val_rank_ic = float(
            prediction_metrics(val_pred[:, primary_index], val_target[:, primary_index])["rank_ic"]
        )
        selection_score = val_huber - float(cfg["selection_rank_weight"]) * val_rank_ic
        logs.append(
            {
                "fold": fold_name,
                "seed": seed,
                "epoch": epoch,
                "train_loss": float(np.mean(batch_losses)),
                "train_huber": float(np.mean(batch_huber)),
                "train_rank": float(np.mean(batch_rank)),
                "val_huber": val_huber,
                "val_rank_ic": val_rank_ic,
                "selection_score": selection_score,
            }
        )
        min_delta = float(cfg.get("early_stopping_min_delta", 1e-6))
        if epoch == 1 or epoch % int(cfg.get("progress_every_epochs", 10)) == 0:
            print(
                json.dumps(
                    {
                        "stage": "validation_training",
                        "fold": fold_name,
                        "seed": seed,
                        "epoch": epoch,
                        "train_loss": logs[-1]["train_loss"],
                        "val_huber": val_huber,
                        "val_rank_ic": val_rank_ic,
                        "best_epoch_so_far": best_epoch,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        if selection_score < best_score - min_delta:
            best_score = selection_score
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            wait = 0
        else:
            wait += 1
            if wait >= int(cfg["patience"]):
                break
    if best_state is None:
        best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    diagnostics = {
        "fold": fold_name,
        "seed": seed,
        "train_rows": len(train_idx),
        "validation_rows": len(val_idx),
        "epochs_run": len(logs),
        "best_epoch": best_epoch,
        "best_selection_score": best_score,
    }
    return best_state, logs, diagnostics


def train_fixed_epochs(
    seed: int,
    epochs: int,
    x: np.ndarray,
    y: np.ndarray,
    train_mask: np.ndarray,
    cfg: dict[str, Any],
) -> dict[str, torch.Tensor]:
    set_seed(seed)
    model = make_model(x, y, cfg)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    train_idx = np.flatnonzero(train_mask)
    rng = np.random.default_rng(seed)
    shape = (y.shape[1], y.shape[2])
    for _ in range(max(1, epochs)):
        order = rng.permutation(train_idx)
        model.train()
        for start in range(0, len(order), int(cfg["batch_size"])):
            idx = order[start : start + int(cfg["batch_size"])]
            tx = torch.tensor(x[idx], dtype=torch.float32)
            ty = torch.tensor(y[idx], dtype=torch.float32)
            pred = model(tx).reshape(len(idx), *shape)
            total, _, _ = loss_components(pred, ty, cfg)
            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def main() -> None:
    args = parse_args()
    cfg = load_daily_config()
    if args.override:
        with args.override.open("r", encoding="utf-8") as handle:
            cfg = merge_config(cfg, yaml.safe_load(handle) or {})
    model_cfg = cfg["daily_timing_v2"]
    frame, x, y, baseline, raw_future, metadata = load_v2_arrays(cfg)
    residual_mode = bool(model_cfg.get("residual_target", False))
    training_target = y - baseline if residual_mode else y
    horizons = [int(value) for value in metadata["horizons_days"]]
    primary_index = horizons.index(int(model_cfg["primary_horizon_days"]))
    shape = (len(horizons), len(RISK_SLEEVES))
    all_logs: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    report_rows: list[dict[str, Any]] = []
    best_epochs: list[int] = []
    historical_oos_prediction = np.full_like(y, np.nan)
    historical_oos_adaptive = np.full_like(y, np.nan)
    oos_residual_alpha = np.full(len(frame), np.nan, dtype=float)
    model_information_cutoff = pd.Series(
        pd.NaT, index=np.arange(len(frame)), dtype="datetime64[ns]"
    )
    prediction_source = np.full(len(frame), "", dtype=object)
    fold_residual_alphas: list[float] = []
    hyperparameter_cutoffs: list[pd.Timestamp] = []
    as_of = pd.to_datetime(frame["as_of_date"], errors="coerce")
    label_end = pd.to_datetime(frame["label_end_date"], errors="coerce")
    labelled = frame["has_label"].astype(bool).to_numpy()
    train_anchor = frame["train_anchor"].astype(bool).to_numpy()
    eval_anchor = frame["evaluation_anchor"].astype(bool).to_numpy()

    for fold in model_cfg["folds"]:
        train_mask, tuning_mask, evaluation_mask = nested_fold_masks(
            frame,
            str(fold["fit_end"]),
            str(fold["tune_end"]),
            str(fold["evaluation_end"]),
        )
        seed_predictions: list[np.ndarray] = []
        for seed_value in model_cfg["seeds"]:
            seed = int(seed_value)
            state, logs, diag = train_with_validation(
                str(fold["name"]), seed, x, training_target, train_mask, tuning_mask, model_cfg, primary_index
            )
            all_logs.extend(logs)
            diagnostics.append(diag)
            best_epochs.append(int(diag["best_epoch"]))
            model = make_model(x, training_target, model_cfg)
            model.load_state_dict(state)
            seed_predictions.append(predict(model, x, shape))
        raw_ensemble = np.mean(np.stack(seed_predictions), axis=0)
        if residual_mode:
            residual_alpha, tuning_metrics = select_residual_alpha(
                raw_ensemble,
                baseline,
                y,
                raw_future,
                tuning_mask,
                primary_index,
                list(model_cfg.get("residual_alpha_grid", [0.0, 0.1, 0.25, 0.5, 1.0])),
            )
            ensemble = baseline + residual_alpha * raw_ensemble
        else:
            residual_alpha = 1.0
            tuning_metrics = primary_metrics(raw_ensemble, y, raw_future, tuning_mask, primary_index)
            ensemble = raw_ensemble
        fold_residual_alphas.append(float(residual_alpha))
        daily_oos_mask = (
            labelled
            & as_of.gt(pd.Timestamp(fold["tune_end"])).to_numpy()
            & as_of.le(pd.Timestamp(fold["evaluation_end"])).to_numpy()
        )
        historical_oos_prediction[daily_oos_mask] = ensemble[daily_oos_mask]
        historical_oos_adaptive[daily_oos_mask] = (
            ensemble[daily_oos_mask]
            if residual_mode
            else 0.65 * ensemble[daily_oos_mask] + 0.35 * baseline[daily_oos_mask]
        )
        oos_residual_alpha[daily_oos_mask] = residual_alpha
        fold_information = label_end[tuning_mask].max()
        if pd.isna(fold_information):
            fold_information = label_end[train_mask].max()
        if pd.notna(fold_information):
            fold_information = pd.Timestamp(fold_information)
            model_information_cutoff.loc[daily_oos_mask] = fold_information
            hyperparameter_cutoffs.append(fold_information)
        prediction_source[daily_oos_mask] = f"historical_oos:{fold['name']}"
        model_metrics = primary_metrics(ensemble, y, raw_future, evaluation_mask, primary_index)
        baseline_metrics = primary_metrics(baseline, y, raw_future, evaluation_mask, primary_index)
        report_rows.append({"fold": fold["name"], "source": "identified_itransformer", **model_metrics})
        report_rows.append({"fold": fold["name"], "source": "causal_momentum_baseline", **baseline_metrics})
        print(
            json.dumps(
                {
                    "fold": fold["name"],
                    "fit_rows": int(train_mask.sum()),
                    "tuning_rows": int(tuning_mask.sum()),
                    "evaluation_rows": int(evaluation_mask.sum()),
                    "model_rank_ic": model_metrics["rank_ic"],
                    "baseline_rank_ic": baseline_metrics["rank_ic"],
                    "residual_alpha": residual_alpha,
                    "tuning_rank_ic": tuning_metrics.get("rank_ic"),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    fixed_epochs = max(5, int(np.median([value for value in best_epochs if value > 0])))
    final_train_end = pd.Timestamp(model_cfg["final_train_end"])
    final_train = labelled & train_anchor & label_end.le(final_train_end).to_numpy()
    holdout_daily = labelled & as_of.ge(pd.Timestamp(model_cfg["final_holdout_start"])).to_numpy()
    holdout = holdout_daily & eval_anchor
    final_states: list[dict[str, torch.Tensor]] = []
    final_predictions: list[np.ndarray] = []
    for seed_value in model_cfg["seeds"]:
        seed = int(seed_value)
        state = train_fixed_epochs(seed, fixed_epochs, x, training_target, final_train, model_cfg)
        final_states.append(state)
        model = make_model(x, training_target, model_cfg)
        model.load_state_dict(state)
        final_predictions.append(predict(model, x, shape))
    final_raw_ensemble = np.mean(np.stack(final_predictions), axis=0)
    final_residual_alpha = float(np.median(fold_residual_alphas)) if residual_mode else 1.0
    final_ensemble = baseline + final_residual_alpha * final_raw_ensemble if residual_mode else final_raw_ensemble
    holdout_model = primary_metrics(final_ensemble, y, raw_future, holdout, primary_index)
    holdout_baseline = primary_metrics(baseline, y, raw_future, holdout, primary_index)
    report_rows.append({"fold": "holdout_2025_onward", "source": "identified_itransformer", **holdout_model})
    report_rows.append({"fold": "holdout_2025_onward", "source": "causal_momentum_baseline", **holdout_baseline})

    walkforward_prediction = np.full_like(final_ensemble, np.nan)
    adaptive_prediction = np.full_like(final_ensemble, np.nan)
    adaptive_model_weight = np.full(len(frame), np.nan, dtype=float)
    assigned = np.zeros(len(frame), dtype=bool)
    labelled_as_of = as_of[labelled]
    quarter_starts = pd.date_range(
        pd.Timestamp(model_cfg["final_holdout_start"]).to_period("Q").start_time,
        labelled_as_of.max().to_period("Q").start_time,
        freq="QS",
    )
    for quarter_start in quarter_starts:
        quarter_end = quarter_start + pd.offsets.QuarterBegin(startingMonth=quarter_start.month)
        quarter_mask = holdout_daily & as_of.ge(quarter_start).to_numpy() & as_of.lt(quarter_end).to_numpy()
        if not quarter_mask.any():
            continue
        causal_train = (
            labelled
            & train_anchor
            & label_end.lt(quarter_start).to_numpy()
        )
        quarter_seed_predictions: list[np.ndarray] = []
        for seed_value in model_cfg["seeds"]:
            seed = int(seed_value)
            state = train_fixed_epochs(seed, fixed_epochs, x, training_target, causal_train, model_cfg)
            model = make_model(x, training_target, model_cfg)
            model.load_state_dict(state)
            quarter_seed_predictions.append(predict(model, x[quarter_mask], shape))
        quarter_raw = np.mean(np.stack(quarter_seed_predictions), axis=0)
        quarter_model = (
            baseline[quarter_mask] + final_residual_alpha * quarter_raw
            if residual_mode
            else quarter_raw
        )
        walkforward_prediction[quarter_mask] = quarter_model

        matured_history = assigned & label_end.lt(quarter_start).to_numpy()
        if int(matured_history.sum()) >= 10:
            history_model_ic = float(
                primary_metrics(
                    walkforward_prediction, y, raw_future, matured_history, primary_index
                )["rank_ic"]
            )
            history_baseline_ic = float(
                primary_metrics(baseline, y, raw_future, matured_history, primary_index)["rank_ic"]
            )
            model_weight = float(np.clip(0.50 + 0.75 * (history_model_ic - history_baseline_ic), 0.20, 0.80))
        else:
            history_model_ic = float("nan")
            history_baseline_ic = float("nan")
            model_weight = 0.65
        adaptive_prediction[quarter_mask] = (
            model_weight * quarter_model + (1.0 - model_weight) * baseline[quarter_mask]
        )
        adaptive_model_weight[quarter_mask] = model_weight
        oos_residual_alpha[quarter_mask] = final_residual_alpha
        causal_information = label_end[causal_train].max()
        known_cutoffs = [
            pd.Timestamp(value)
            for value in [causal_information, *hyperparameter_cutoffs]
            if pd.notna(value)
        ]
        if known_cutoffs:
            model_information_cutoff.loc[quarter_mask] = max(known_cutoffs)
        prediction_source[quarter_mask] = f"quarterly_walkforward:{quarter_start.to_period('Q')}"
        assigned[quarter_mask] = True
        print(
            json.dumps(
                {
                    "quarter": str(quarter_start.to_period("Q")),
                    "train_rows": int(causal_train.sum()),
                    "prediction_rows": int(quarter_mask.sum()),
                    "model_weight": model_weight,
                    "matured_model_rank_ic": history_model_ic,
                    "matured_baseline_rank_ic": history_baseline_ic,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    walkforward_mask = holdout & assigned
    holdout_walkforward = primary_metrics(
        walkforward_prediction, y, raw_future, walkforward_mask, primary_index
    )
    holdout_adaptive = primary_metrics(
        adaptive_prediction, y, raw_future, walkforward_mask, primary_index
    )
    report_rows.append(
        {"fold": "holdout_2025_onward", "source": "walkforward_identified_itransformer", **holdout_walkforward}
    )
    report_rows.append(
        {"fold": "holdout_2025_onward", "source": "walkforward_adaptive_blend", **holdout_adaptive}
    )

    cv_report = pd.DataFrame(report_rows)
    cv_report.to_csv(v2_path(cfg, "timing_v2_cv_report"), index=False, encoding="utf-8-sig")
    pd.DataFrame(all_logs).to_csv(v2_path(cfg, "timing_v2_training_log"), index=False, encoding="utf-8-sig")
    checkpoint = {
        "states": final_states,
        "config": cfg,
        "metadata": metadata,
        "fixed_epochs": fixed_epochs,
        "diagnostics": diagnostics,
        "residual_target": residual_mode,
        "fold_residual_alphas": fold_residual_alphas,
        "final_residual_alpha": final_residual_alpha,
    }
    torch.save(checkpoint, v2_path(cfg, "timing_v2_model"))

    prediction_rows: list[dict[str, Any]] = []
    write_all_horizons = bool(model_cfg.get("write_all_horizons", False))
    output_horizon_indices = range(len(horizons)) if write_all_horizons else [primary_index]
    frame_dates = pd.to_datetime(frame["as_of_date"], errors="coerce").reset_index(drop=True)
    for row_index, row in frame.reset_index(drop=True).iterrows():
        for horizon_index in output_horizon_indices:
            horizon_days = horizons[horizon_index]
            horizon_end_index = row_index + horizon_days
            horizon_label_end = (
                frame_dates.iloc[horizon_end_index] if horizon_end_index < len(frame_dates) else pd.NaT
            )
            for sleeve_index, sleeve in enumerate(RISK_SLEEVES):
                output_row = {
                    "as_of_date": row["as_of_date"],
                    "label_end_date": row["label_end_date"],
                    "horizon_label_end_date": horizon_label_end,
                    "has_label": bool(row["has_label"]),
                    "sleeve": sleeve,
                    "model_training_cutoff": model_information_cutoff.iloc[row_index],
                    "prediction_source": str(prediction_source[row_index]),
                    "pred_relative_score": float(final_ensemble[row_index, horizon_index, sleeve_index]),
                    "baseline_relative_score": float(baseline[row_index, horizon_index, sleeve_index]),
                    "walkforward_relative_score": float(walkforward_prediction[row_index, horizon_index, sleeve_index]),
                    "adaptive_relative_score": float(adaptive_prediction[row_index, horizon_index, sleeve_index]),
                    "adaptive_model_weight": float(adaptive_model_weight[row_index]),
                    "residual_alpha": float(oos_residual_alpha[row_index]),
                    "historical_oos_relative_score": float(historical_oos_prediction[row_index, horizon_index, sleeve_index]),
                    "historical_oos_adaptive_score": float(historical_oos_adaptive[row_index, horizon_index, sleeve_index]),
                    "combined_oos_adaptive_score": float(
                        adaptive_prediction[row_index, horizon_index, sleeve_index]
                        if np.isfinite(adaptive_prediction[row_index, horizon_index, sleeve_index])
                        else historical_oos_adaptive[row_index, horizon_index, sleeve_index]
                    ),
                    "realized_relative_score": float(y[row_index, horizon_index, sleeve_index]),
                    "realized_raw_return": float(raw_future[row_index, horizon_index, sleeve_index]),
                }
                if write_all_horizons:
                    output_row["horizon_days"] = horizon_days
                prediction_rows.append(output_row)
    pd.DataFrame(prediction_rows).to_parquet(v2_path(cfg, "timing_v2_predictions"), index=False)

    model_cv = cv_report[
        cv_report["source"].eq("identified_itransformer")
        & cv_report["fold"].str.startswith("wf_")
    ]
    baseline_cv = cv_report[
        cv_report["source"].eq("causal_momentum_baseline")
        & cv_report["fold"].str.startswith("wf_")
    ]
    gate_cfg = model_cfg["formal_gate"]
    cv_median = float(model_cv["rank_ic"].median())
    cv_min = float(model_cv["rank_ic"].min())
    baseline_cv_median = float(baseline_cv["rank_ic"].median())
    holdout_ic = float(holdout_adaptive["rank_ic"])
    checks = {
        "cv_median_rank_ic": cv_median >= float(gate_cfg["min_cv_median_rank_ic"]),
        "cv_min_fold_rank_ic": cv_min >= float(gate_cfg["min_cv_fold_rank_ic"]),
        "holdout_rank_ic": holdout_ic >= float(gate_cfg["min_holdout_rank_ic"]),
        "incremental_cv_rank_ic": cv_median - baseline_cv_median >= float(gate_cfg["min_incremental_cv_rank_ic"]),
    }
    if residual_mode:
        checks["residual_alpha_active"] = float(np.median(fold_residual_alphas)) >= float(
            gate_cfg.get("min_selected_residual_alpha", 0.05)
        )
    gate = {
        "enabled_for_formal_strategy": bool(all(checks.values())),
        "checks": checks,
        "thresholds": gate_cfg,
        "cv_median_rank_ic": cv_median,
        "cv_min_rank_ic": cv_min,
        "baseline_cv_median_rank_ic": baseline_cv_median,
        "incremental_cv_rank_ic": cv_median - baseline_cv_median,
        "holdout_rank_ic": holdout_ic,
        "static_holdout_rank_ic": float(holdout_model["rank_ic"]),
        "walkforward_holdout_rank_ic": float(holdout_walkforward["rank_ic"]),
        "adaptive_holdout_rank_ic": float(holdout_adaptive["rank_ic"]),
        "holdout_top_bottom_raw_spread": float(holdout_adaptive["top_bottom_raw_spread"]),
        "fixed_epochs": fixed_epochs,
        "final_train_rows": int(final_train.sum()),
        "holdout_rows": int(holdout.sum()),
        "holdout_daily_prediction_rows": int((holdout_daily & assigned).sum()),
        "residual_target": residual_mode,
        "fold_residual_alphas": fold_residual_alphas,
        "final_residual_alpha": final_residual_alpha,
        "caveat": "2025+ was exposed by earlier research versions; this is a conservative historical gate, not a pristine untouched test",
    }
    v2_path(cfg, "timing_v2_gate_report").write_text(
        json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(gate, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

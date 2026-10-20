from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.itransformer_daily_signal import load_daily_config  # noqa: E402
from src.itransformer_daily_timing import (  # noqa: E402
    load_timing_arrays,
    timing_path,
    timing_split_masks,
)
from src.itransformer_signal import (  # noqa: E402
    InvertedTimeSeriesTransformer,
    RISK_SLEEVES,
    prediction_metrics,
    set_seed,
    softmax_np,
)


def pairwise_rank_loss(pred: torch.Tensor, target: torch.Tensor, temperature: float) -> torch.Tensor:
    pred_diff = pred.unsqueeze(2) - pred.unsqueeze(1)
    target_diff = target.unsqueeze(2) - target.unsqueeze(1)
    direction = torch.sign(target_diff)
    valid = direction.ne(0)
    if not valid.any():
        return pred.new_tensor(0.0)
    return torch.nn.functional.softplus(
        -direction[valid] * pred_diff[valid] / max(float(temperature), 1e-6)
    ).mean()


def make_model(x: np.ndarray, y: np.ndarray, model_cfg: dict) -> InvertedTimeSeriesTransformer:
    return InvertedTimeSeriesTransformer(
        seq_len=x.shape[1],
        n_features=x.shape[2],
        n_targets=y.shape[1],
        d_model=int(model_cfg["d_model"]),
        n_heads=int(model_cfg["n_heads"]),
        num_layers=int(model_cfg["num_layers"]),
        dropout=float(model_cfg["dropout"]),
    )


def predict(model: nn.Module, x: np.ndarray, batch_size: int = 128) -> np.ndarray:
    model.eval()
    parts: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = torch.tensor(x[start : start + batch_size], dtype=torch.float32)
            parts.append(model(batch).cpu().numpy())
    return np.vstack(parts) if parts else np.empty((0, 0))


def train_seed(
    seed: int,
    x: np.ndarray,
    y: np.ndarray,
    masks: dict[str, np.ndarray],
    model_cfg: dict,
) -> tuple[dict[str, torch.Tensor], list[dict], dict]:
    set_seed(seed)
    model = make_model(x, y, model_cfg)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(model_cfg["learning_rate"]),
        weight_decay=float(model_cfg["weight_decay"]),
    )
    train_idx = np.flatnonzero(masks["train"])
    val_idx = np.flatnonzero(masks["val"])
    batch_size = int(model_cfg["batch_size"])
    patience = int(model_cfg["patience"])
    best_score = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    wait = 0
    rows: list[dict] = []
    rng = np.random.default_rng(seed)

    for epoch in range(1, int(model_cfg["epochs"]) + 1):
        model.train()
        order = rng.permutation(train_idx)
        epoch_total: list[float] = []
        epoch_huber: list[float] = []
        epoch_rank: list[float] = []
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            tx = torch.tensor(x[idx], dtype=torch.float32)
            ty = torch.tensor(y[idx], dtype=torch.float32)
            pred = model(tx)
            huber = nn.SmoothL1Loss()(pred, ty)
            rank = pairwise_rank_loss(pred, ty, float(model_cfg["rank_temperature"]))
            loss = huber + float(model_cfg["rank_loss_weight"]) * rank
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_total.append(float(loss.detach()))
            epoch_huber.append(float(huber.detach()))
            epoch_rank.append(float(rank.detach()))

        val_pred = predict(model, x[val_idx])
        val_loss = float(np.mean((val_pred - y[val_idx]) ** 2)) if len(val_idx) else float("nan")
        val_metrics = prediction_metrics(val_pred, y[val_idx])
        val_rank_ic = float(val_metrics.get("rank_ic", np.nan))
        score = val_loss - float(model_cfg["selection_rank_weight"]) * np.nan_to_num(val_rank_ic)
        rows.append(
            {
                "seed": seed,
                "epoch": epoch,
                "train_loss": float(np.mean(epoch_total)),
                "train_huber_loss": float(np.mean(epoch_huber)),
                "train_rank_loss": float(np.mean(epoch_rank)),
                "val_mse": val_loss,
                "val_rank_ic": val_rank_ic,
                "selection_score": score,
            }
        )
        if score < best_score - 1e-9:
            best_score = score
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is None:
        best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    diagnostics = {
        "seed": seed,
        "epochs_run": len(rows),
        "best_epoch": best_epoch,
        "best_selection_score": best_score,
    }
    return best_state, rows, diagnostics


def main() -> None:
    cfg = load_daily_config()
    model_cfg = cfg["daily_timing"]
    frame, x, y, feature_cols = load_timing_arrays(cfg)
    masks = timing_split_masks(frame, cfg)
    if min(int(mask.sum()) for mask in masks.values()) == 0:
        raise RuntimeError(f"Empty timing split: { {key: int(value.sum()) for key, value in masks.items()} }")

    states: list[dict[str, torch.Tensor]] = []
    all_logs: list[dict] = []
    seed_diagnostics: list[dict] = []
    seed_predictions: list[np.ndarray] = []
    for seed in [int(value) for value in model_cfg["seeds"]]:
        state, logs, diagnostics = train_seed(seed, x, y, masks, model_cfg)
        states.append(state)
        all_logs.extend(logs)
        seed_diagnostics.append(diagnostics)
        model = make_model(x, y, model_cfg)
        model.load_state_dict(state)
        seed_predictions.append(predict(model, x))

    ensemble = np.mean(np.stack(seed_predictions, axis=0), axis=0)
    checkpoint = {
        "states": states,
        "config": cfg,
        "feature_columns": feature_cols,
        "sequence_length_days": x.shape[1],
        "feature_count": x.shape[2],
        "seed_diagnostics": seed_diagnostics,
    }
    model_path = timing_path(cfg, "timing_model")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, model_path)

    split_name = np.full(len(frame), "", dtype=object)
    for name, mask in masks.items():
        split_name[mask] = name
    probabilities = softmax_np(ensemble, temperature=float(cfg["model"].get("probability_temperature", 0.03)))
    prediction_rows: list[dict] = []
    for i, row in frame.reset_index(drop=True).iterrows():
        order = (-ensemble[i]).argsort().tolist()
        realized_order = (-y[i]).argsort().tolist()
        for j, sleeve in enumerate(RISK_SLEEVES):
            prediction_rows.append(
                {
                    "as_of_date": row["as_of_date"],
                    "label_end_date": row["label_end_date"],
                    "split": split_name[i],
                    "sleeve": sleeve,
                    "pred_return": float(ensemble[i, j]),
                    "pred_prob": float(probabilities[i, j]),
                    "pred_rank": int(order.index(j) + 1),
                    "realized_return": float(y[i, j]),
                    "realized_rank": int(realized_order.index(j) + 1),
                }
            )
    predictions = pd.DataFrame(prediction_rows)
    predictions.to_parquet(timing_path(cfg, "timing_predictions"), index=False)

    report_rows: list[dict] = []
    for split, mask in masks.items():
        metrics = prediction_metrics(ensemble[mask], y[mask])
        report_rows.append(
            {
                "model": "daily_timing_itransformer_3seed_ensemble",
                "split": split,
                **metrics,
                "sequence_length_days": x.shape[1],
                "horizon_days": int(model_cfg["horizon_days"]),
                "feature_count": x.shape[2],
                "seeds": ",".join(str(value) for value in model_cfg["seeds"]),
            }
        )
    report = pd.DataFrame(report_rows)
    report.to_csv(timing_path(cfg, "timing_validation_report"), index=False, encoding="utf-8-sig")
    pd.DataFrame(all_logs).to_csv(timing_path(cfg, "timing_training_log"), index=False, encoding="utf-8-sig")

    test = report.set_index("split").loc["test"]
    gate_cfg = model_cfg["formal_gate"]
    checks = {
        "test_rank_ic": float(test["rank_ic"]) >= float(gate_cfg["min_test_rank_ic"]),
        "test_top2_realized_return": float(test["top2_realized_return"])
        >= float(gate_cfg["min_test_top2_realized_return"]),
    }
    gate = {
        "enabled_for_formal_strategy": bool(all(checks.values())),
        "checks": checks,
        "thresholds": gate_cfg,
        "test_metrics": {key: float(test[key]) for key in ["rows", "mse", "rank_ic", "top1_hit_rate", "top2_realized_return", "long_short_spread"]},
        "split_rows": {key: int(mask.sum()) for key, mask in masks.items()},
        "selection_rule": "architecture and early stopping use train/validation only; test is used once as a deployment gate",
    }
    timing_path(cfg, "timing_gate_report").write_text(json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "model": str(model_path.relative_to(ROOT)),
                "rows": len(frame),
                "split_rows": gate["split_rows"],
                "seed_diagnostics": seed_diagnostics,
                "test_rank_ic": gate["test_metrics"]["rank_ic"],
                "formal_enabled": gate["enabled_for_formal_strategy"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

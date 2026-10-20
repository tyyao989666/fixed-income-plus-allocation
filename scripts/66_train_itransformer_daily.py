from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.itransformer_daily_signal import (  # noqa: E402
    daily_out_path,
    load_daily_config,
    load_daily_dataset_arrays,
    split_daily_masks,
)
from src.itransformer_signal import (  # noqa: E402
    InvertedTimeSeriesTransformer,
    RISK_SLEEVES,
    prediction_metrics,
    set_seed,
    softmax_np,
)


def tensor_pair(x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.tensor(x[mask], dtype=torch.float32), torch.tensor(y[mask], dtype=torch.float32)


def evaluate(model: nn.Module, x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> tuple[float, np.ndarray]:
    model.eval()
    if mask.sum() == 0:
        return float("nan"), np.empty((0, y.shape[1]))
    with torch.no_grad():
        tx, ty = tensor_pair(x, y, mask)
        pred = model(tx)
        loss = nn.SmoothL1Loss()(pred, ty).item()
    return loss, pred.detach().cpu().numpy()


def pairwise_rank_loss(pred: torch.Tensor, target: torch.Tensor, temperature: float = 0.03) -> torch.Tensor:
    pred_diff = pred.unsqueeze(2) - pred.unsqueeze(1)
    target_diff = target.unsqueeze(2) - target.unsqueeze(1)
    direction = torch.sign(target_diff)
    valid = direction.ne(0)
    if not valid.any():
        return pred.new_tensor(0.0)
    return torch.nn.functional.softplus(-direction[valid] * pred_diff[valid] / max(temperature, 1e-6)).mean()


def training_loss(pred: torch.Tensor, target: torch.Tensor, cfg: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    huber = nn.SmoothL1Loss()(pred, target)
    rank = pairwise_rank_loss(pred, target, float(cfg.get("model", {}).get("rank_temperature", 0.03)))
    weight = float(cfg.get("model", {}).get("rank_loss_weight", 0.0))
    return huber + weight * rank, huber, rank


def write_daily_prediction_frame(cfg: dict, df: pd.DataFrame, pred: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    probs = softmax_np(pred, temperature=float(cfg.get("model", {}).get("probability_temperature", 0.03)))
    rows = []
    for i, row in df.reset_index(drop=True).iterrows():
        order = (-pred[i]).argsort().tolist()
        real_order = (-y[i]).argsort().tolist()
        for j, sleeve in enumerate(RISK_SLEEVES):
            rows.append(
                {
                    "as_of_date": row["as_of_date"],
                    "label_date": row["label_date"],
                    "sleeve": sleeve,
                    "pred_return": float(pred[i, j]),
                    "pred_prob": float(probs[i, j]),
                    "realized_return": float(y[i, j]),
                    "pred_rank": int(order.index(j) + 1),
                    "realized_rank": int(real_order.index(j) + 1),
                    "model_version": cfg["version"],
                    "input_frequency": "daily_path",
                }
            )
    out = pd.DataFrame(rows)
    path = daily_out_path(cfg, "predictions")
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the daily-path iTransformer sleeve-return signal model.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--config", default=None, help="Optional daily iTransformer YAML config path.")
    parser.add_argument("--output-suffix", default="", help="Suffix added to model/report/prediction outputs.")
    args = parser.parse_args()

    cfg = load_daily_config(Path(args.config) if args.config else None)
    if args.output_suffix:
        for key in ["predictions", "validation_report", "training_log", "model"]:
            path = Path(str(cfg["outputs"][key]))
            cfg["outputs"][key] = str(path.with_name(f"{path.stem}{args.output_suffix}{path.suffix}")).replace("\\", "/")
    if args.seed is not None:
        cfg["model"]["seed"] = int(args.seed)
    set_seed(int(cfg["model"]["seed"]))
    df, x, y, feature_cols = load_daily_dataset_arrays(cfg)
    masks = split_daily_masks(df, cfg)
    if masks["train"].sum() == 0:
        raise RuntimeError("No training rows in daily-path dataset.")

    model = InvertedTimeSeriesTransformer(
        seq_len=x.shape[1],
        n_features=x.shape[2],
        n_targets=y.shape[1],
        d_model=int(cfg["model"]["d_model"]),
        n_heads=int(cfg["model"]["n_heads"]),
        num_layers=int(cfg["model"]["num_layers"]),
        dropout=float(cfg["model"]["dropout"]),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["model"]["learning_rate"]),
        weight_decay=float(cfg["model"]["weight_decay"]),
    )
    train_x, train_y = tensor_pair(x, y, masks["train"])
    epochs = int(cfg["model"]["epochs"])
    patience = int(cfg["model"]["patience"])
    best_val = float("inf")
    best_state = None
    wait = 0
    log_rows = []

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        pred = model(train_x)
        loss, huber_loss, rank_loss = training_loss(pred, train_y, cfg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss = float(loss.item())
        val_loss, _ = evaluate(model, x, y, masks["val"])
        test_loss, _ = evaluate(model, x, y, masks["test"])
        log_rows.append(
            {
                "seed": int(cfg["model"]["seed"]),
                "epoch": epoch,
                "train_loss": train_loss,
                "train_huber_loss": float(huber_loss.item()),
                "train_rank_loss": float(rank_loss.item()),
                "val_loss": val_loss,
                "test_loss": test_loss,
            }
        )

        score = val_loss if not np.isnan(val_loss) else train_loss
        if score < best_val:
            best_val = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model_path = daily_out_path(cfg, "model")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": cfg,
            "feature_columns": feature_cols,
            "seq_len": x.shape[1],
            "n_features": x.shape[2],
        },
        model_path,
    )

    model.eval()
    with torch.no_grad():
        all_pred = model(torch.tensor(x, dtype=torch.float32)).detach().cpu().numpy()
    pred_df = write_daily_prediction_frame(cfg, df, all_pred, y)

    log_path = daily_out_path(cfg, "training_log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(log_rows).to_csv(log_path, index=False, encoding="utf-8-sig")

    rows = []
    for split in ["train", "val", "test", "prediction"]:
        metrics = prediction_metrics(all_pred[masks[split]], y[masks[split]])
        metrics["split"] = split
        metrics["model"] = "itransformer_daily_path_signal"
        metrics["seed"] = int(cfg["model"]["seed"])
        metrics["seq_len_days"] = x.shape[1]
        metrics["feature_count"] = x.shape[2]
        rows.append(metrics)
    report = pd.DataFrame(rows)[
        [
            "model",
            "seed",
            "split",
            "rows",
            "mse",
            "rank_ic",
            "top1_hit_rate",
            "top2_realized_return",
            "long_short_spread",
            "seq_len_days",
            "feature_count",
        ]
    ]
    report.to_csv(daily_out_path(cfg, "validation_report"), index=False, encoding="utf-8-sig")

    print(
        json.dumps(
            {
                "model": cfg["outputs"]["model"],
                "predictions": cfg["outputs"]["predictions"],
                "validation_report": cfg["outputs"]["validation_report"],
                "training_log": cfg["outputs"]["training_log"],
                "prediction_rows": len(pred_df),
                "epochs_run": len(log_rows),
                "best_val_loss": best_val,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

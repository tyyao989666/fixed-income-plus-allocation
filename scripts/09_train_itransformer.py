from __future__ import annotations

import json
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.itransformer_signal import (  # noqa: E402
    InvertedTimeSeriesTransformer,
    load_config,
    load_dataset_arrays,
    out_path,
    append_live_prediction_frame,
    prediction_metrics,
    set_seed,
    split_masks,
    write_prediction_frame,
)


def add_suffix(path_value: str, suffix: str) -> str:
    if not suffix:
        return path_value
    path = Path(path_value)
    return str(path.with_name(f"{path.stem}{suffix}{path.suffix}")).replace("\\", "/")


def apply_output_suffix(cfg: dict, suffix: str) -> dict:
    if not suffix:
        return cfg
    for key in ["predictions", "validation_report", "training_log", "model"]:
        cfg["outputs"][key] = add_suffix(str(cfg["outputs"][key]), suffix)
    return cfg


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the iTransformer sleeve-return signal model.")
    parser.add_argument("--seed", type=int, default=None, help="Override model.seed for one run.")
    parser.add_argument("--output-suffix", default="", help="Suffix added before model/report/prediction output extensions.")
    args = parser.parse_args()

    cfg = load_config()
    if args.seed is not None:
        cfg["model"]["seed"] = int(args.seed)
    cfg = apply_output_suffix(cfg, args.output_suffix)
    set_seed(int(cfg["model"]["seed"]))
    df, x, y, feature_cols = load_dataset_arrays(cfg)
    masks = split_masks(df, cfg)

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
    criterion = nn.SmoothL1Loss()

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
        loss = criterion(pred, train_y)
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

    model_path = out_path(cfg, "model")
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
    pred_df = write_prediction_frame(cfg, df, all_pred, y, "predictions")
    pred_df = append_live_prediction_frame(cfg, pred_df, model, metadata_key="metadata", path_key="predictions")

    log_path = out_path(cfg, "training_log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(log_rows).to_csv(log_path, index=False, encoding="utf-8-sig")

    rows = []
    for split in ["train", "val", "test", "prediction"]:
        metrics = prediction_metrics(all_pred[masks[split]], y[masks[split]])
        metrics["split"] = split
        metrics["model"] = "itransformer_signal"
        metrics["seed"] = int(cfg["model"]["seed"])
        rows.append(metrics)
    report = pd.DataFrame(rows)[["model", "seed", "split", "rows", "mse", "rank_ic", "top1_hit_rate", "top2_realized_return", "long_short_spread"]]
    report.to_csv(out_path(cfg, "validation_report"), index=False, encoding="utf-8-sig")

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

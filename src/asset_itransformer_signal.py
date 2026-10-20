from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.point_in_time import causal_fill_features, normalization_metadata, transform_point_in_time_features
from src.itransformer_signal import (
    ITransformerPaths,
    InvertedTimeSeriesTransformer,
    build_feature_frame,
    load_config,
    load_monthly_returns_with_proxy,
    out_path,
    prediction_metrics,
    set_seed,
    softmax_np,
)


ROOT = Path(__file__).resolve().parents[1]

SELECTED_ASSETS = [
    "rate_30y_curve_proxy",
    "rate_7_10y",
    "credit_ind_aa",
    "credit_ind_aa_plus",
    "credit_lgfv_aa_plus",
    "cb_balanced_large_high",
    "cb_balanced_small_low",
    "cb_bond_large_low",
    "cb_bond_small_low",
    "equity_broad",
    "equity_tech_growth",
    "equity_fin_property",
    "equity_dividend_utility",
    "alt_gold",
    "alt_commodity",
    "alt_nonferrous",
    "alt_chemical",
    "alt_us_bond",
    "alt_reits",
]


def selected_asset_returns(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    returns = load_monthly_returns_with_proxy(cfg)
    returns = returns[returns["asset_code"].isin(SELECTED_ASSETS)].copy()
    returns = returns.dropna(subset=["return_1m"])
    info = returns.sort_values("date").drop_duplicates("asset_code", keep="last")[
        ["asset_code", "asset_name", "module"]
    ].copy()
    wide = (
        returns.pivot_table(index="date", columns="asset_code", values="return_1m", aggfunc="last")
        .reindex(columns=SELECTED_ASSETS)
        .sort_index()
    )
    return wide, info


def build_asset_feature_frame(cfg: dict[str, Any]) -> tuple[pd.DataFrame, list[str], pd.DataFrame, pd.DataFrame]:
    features, feature_cols = build_feature_frame(cfg)
    asset_ret, info = selected_asset_returns(cfg)
    asset_ret = asset_ret[asset_ret.index >= pd.Timestamp(cfg["data"]["start_date"])]
    asset_features = asset_ret.copy()
    for code in SELECTED_ASSETS:
        asset_features[f"{code}_ret_1m"] = asset_features[code]
        asset_features[f"{code}_mom_3m"] = asset_features[code].rolling(3).sum()
        asset_features[f"{code}_vol_6m"] = asset_features[code].rolling(6).std()
    asset_features = asset_features[[c for c in asset_features.columns if c not in SELECTED_ASSETS]].reset_index()
    out = features.merge(asset_features, on="date", how="left").sort_values("date")
    added = [c for c in out.columns if c not in {"date"} and c not in feature_cols]
    all_cols = feature_cols + added
    out, all_cols = causal_fill_features(
        out,
        all_cols,
        cfg.get("data", {}).get("preprocessing", {}),
    )
    return out[["date"] + all_cols], all_cols, asset_ret, info


def build_asset_dataset(cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    feature_frame, feature_cols, asset_ret, info = build_asset_feature_frame(cfg)
    horizon = int(cfg["data"]["horizon_months"])
    seq_len = int(cfg["data"]["sequence_length"])
    labels = asset_ret.shift(-horizon).add_prefix("target_").reset_index().rename(columns={"index": "date"})
    data = feature_frame.merge(labels, on="date", how="inner").sort_values("date")
    target_cols = [f"target_{c}" for c in SELECTED_ASSETS]
    data = data.dropna(subset=target_cols).reset_index(drop=True)
    preprocessing = cfg.get("data", {}).get("preprocessing", {})
    scaled, feature_cols = transform_point_in_time_features(data, feature_cols, preprocessing)

    rows = []
    for idx in range(seq_len - 1, len(data)):
        window = scaled.iloc[idx - seq_len + 1 : idx + 1].to_numpy(dtype=float)
        row: dict[str, Any] = {
            "as_of_date": data.loc[idx, "date"],
            "label_date": data.loc[idx, "date"] + pd.offsets.MonthEnd(horizon),
        }
        for t in range(seq_len):
            for j, col in enumerate(feature_cols):
                row[f"x_t{t:02d}_{col}"] = float(window[t, j])
        targets = data.loc[idx, target_cols].astype(float)
        for code in SELECTED_ASSETS:
            row[f"target_{code}"] = float(targets[f"target_{code}"])
        row["winner"] = SELECTED_ASSETS[int(np.argmax(targets.to_numpy(dtype=float)))]
        rows.append(row)
    out = pd.DataFrame(rows)
    out_path(cfg, "asset_dataset").parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path(cfg, "asset_dataset"), index=False)
    meta = {
        "version": "asset_itransformer_signal",
        "seq_len": seq_len,
        "horizon_months": horizon,
        "feature_count": len(feature_cols),
        "feature_columns": feature_cols,
        "target_assets": SELECTED_ASSETS,
        "asset_info": info.to_dict(orient="records"),
        "rows": len(out),
        "start": str(out["as_of_date"].min().date()) if not out.empty else "",
        "end": str(out["as_of_date"].max().date()) if not out.empty else "",
        "train_end": cfg["data"]["train_end"],
        "val_end": cfg["data"]["val_end"],
        "normalization": normalization_metadata(preprocessing),
    }
    out_path(cfg, "asset_metadata").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def load_asset_dataset_arrays(cfg: dict[str, Any] | None = None) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    cfg = cfg or load_config()
    df = pd.read_parquet(out_path(cfg, "asset_dataset"))
    meta = json.loads(out_path(cfg, "asset_metadata").read_text(encoding="utf-8"))
    feature_cols = meta["feature_columns"]
    seq_len = int(meta["seq_len"])
    x_cols = [f"x_t{t:02d}_{col}" for t in range(seq_len) for col in feature_cols]
    y_cols = [f"target_{code}" for code in SELECTED_ASSETS]
    x = df[x_cols].to_numpy(dtype=np.float32).reshape(len(df), seq_len, len(feature_cols))
    y = df[y_cols].to_numpy(dtype=np.float32)
    return df, x, y, feature_cols


def split_masks(df: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, np.ndarray]:
    dates = pd.to_datetime(df["as_of_date"])
    train_end = pd.Timestamp(cfg["data"]["train_end"])
    val_end = pd.Timestamp(cfg["data"]["val_end"])
    pred_start = pd.Timestamp(cfg["data"]["prediction_start"])
    return {
        "train": dates.le(train_end).to_numpy(),
        "val": dates.gt(train_end).to_numpy() & dates.le(val_end).to_numpy(),
        "test": dates.gt(val_end).to_numpy(),
        "prediction": dates.ge(pred_start).to_numpy(),
    }


def write_asset_prediction_frame(cfg: dict[str, Any], df: pd.DataFrame, pred: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    probs = softmax_np(pred, temperature=float(cfg.get("model", {}).get("probability_temperature", 0.03)))
    meta = json.loads(out_path(cfg, "asset_metadata").read_text(encoding="utf-8"))
    info = pd.DataFrame(meta["asset_info"]).drop_duplicates("asset_code").set_index("asset_code").to_dict(orient="index")
    rows = []
    for i, row in df.reset_index(drop=True).iterrows():
        for j, code in enumerate(SELECTED_ASSETS):
            item = info.get(code, {})
            rows.append(
                {
                    "as_of_date": row["as_of_date"],
                    "label_date": row["label_date"],
                    "asset_code": code,
                    "asset_name": item.get("asset_name", code),
                    "module": item.get("module", ""),
                    "pred_return": float(pred[i, j]),
                    "pred_prob": float(probs[i, j]),
                    "realized_return": float(y[i, j]),
                    "pred_rank": int((-pred[i]).argsort().tolist().index(j) + 1),
                    "realized_rank": int((-y[i]).argsort().tolist().index(j) + 1),
                }
            )
    out = pd.DataFrame(rows)
    out_path(cfg, "asset_predictions").parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path(cfg, "asset_predictions"), index=False)
    return out


def build_asset_live_feature_sequences(cfg: dict[str, Any]) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    meta = json.loads(out_path(cfg, "asset_metadata").read_text(encoding="utf-8"))
    feature_cols = list(meta["feature_columns"])
    seq_len = int(meta["seq_len"])
    horizon = int(meta["horizon_months"])
    feature_frame, _, _, _ = build_asset_feature_frame(cfg)
    feature_frame = feature_frame.sort_values("date").drop_duplicates("date").copy()
    for col in feature_cols:
        if col not in feature_frame.columns:
            feature_frame[col] = 0.0
    if str(meta.get("normalization", {}).get("method", "")).startswith("causal_"):
        scaled, _ = transform_point_in_time_features(
            feature_frame,
            feature_cols,
            cfg.get("data", {}).get("preprocessing", meta.get("normalization", {})),
        )
        scaled = scaled.to_numpy(dtype=float)
    else:
        values = feature_frame[feature_cols].to_numpy(dtype=float)
        med = np.asarray(meta["scaler_median"], dtype=float)
        mad = np.asarray(meta["scaler_mad"], dtype=float)
        mad = np.where(mad < 1e-8, 1.0, mad)
        scaled = np.clip((values - med) / mad, -5.0, 5.0)
    rows = []
    windows = []
    for idx in range(seq_len - 1, len(feature_frame)):
        as_of = pd.Timestamp(feature_frame.iloc[idx]["date"])
        rows.append({"as_of_date": as_of, "label_date": as_of + pd.offsets.MonthEnd(horizon)})
        windows.append(scaled[idx - seq_len + 1 : idx + 1])
    if not rows:
        return pd.DataFrame(columns=["as_of_date", "label_date"]), np.empty((0, seq_len, len(feature_cols)), dtype=np.float32), feature_cols
    return pd.DataFrame(rows), np.asarray(windows, dtype=np.float32), feature_cols


def append_asset_live_prediction_frame(cfg: dict[str, Any], labeled_predictions: pd.DataFrame, model: torch.nn.Module) -> pd.DataFrame:
    live_df, live_x, _ = build_asset_live_feature_sequences(cfg)
    if live_df.empty:
        return labeled_predictions
    max_labeled_date = pd.to_datetime(labeled_predictions["as_of_date"]).max() if not labeled_predictions.empty else pd.Timestamp.min
    live_df = live_df[pd.to_datetime(live_df["as_of_date"]).gt(max_labeled_date)].reset_index(drop=True)
    if live_df.empty:
        return labeled_predictions
    live_x = live_x[-len(live_df) :]
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(live_x, dtype=torch.float32)).detach().cpu().numpy()
    probs = softmax_np(pred, temperature=float(cfg.get("model", {}).get("probability_temperature", 0.03)))
    meta = json.loads(out_path(cfg, "asset_metadata").read_text(encoding="utf-8"))
    info = pd.DataFrame(meta["asset_info"]).drop_duplicates("asset_code").set_index("asset_code").to_dict(orient="index")
    rows = []
    for i, row in live_df.iterrows():
        order = (-pred[i]).argsort().tolist()
        for j, code in enumerate(SELECTED_ASSETS):
            item = info.get(code, {})
            rows.append(
                {
                    "as_of_date": row["as_of_date"],
                    "label_date": row["label_date"],
                    "asset_code": code,
                    "asset_name": item.get("asset_name", code),
                    "module": item.get("module", ""),
                    "pred_return": float(pred[i, j]),
                    "pred_prob": float(probs[i, j]),
                    "realized_return": np.nan,
                    "pred_rank": int(order.index(j) + 1),
                    "realized_rank": np.nan,
                }
            )
    out = pd.concat([labeled_predictions, pd.DataFrame(rows)], ignore_index=True, sort=False)
    out_path(cfg, "asset_predictions").parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path(cfg, "asset_predictions"), index=False)
    return out


def train_asset_model(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = cfg or load_config()
    set_seed(int(cfg["model"]["seed"]))
    df, x, y, feature_cols = load_asset_dataset_arrays(cfg)
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["model"]["learning_rate"]), weight_decay=float(cfg["model"]["weight_decay"]))
    criterion = torch.nn.SmoothL1Loss()
    train_x = torch.tensor(x[masks["train"]], dtype=torch.float32)
    train_y = torch.tensor(y[masks["train"]], dtype=torch.float32)
    epochs = int(cfg["model"]["epochs"])
    patience = int(cfg["model"]["patience"])
    best_val = float("inf")
    best_state = None
    wait = 0
    logs = []
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(train_x), train_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            all_pred = model(torch.tensor(x, dtype=torch.float32)).detach().cpu().numpy()
        train_loss = float(loss.item())
        val_loss = float(criterion(torch.tensor(all_pred[masks["val"]]), torch.tensor(y[masks["val"]])).item()) if masks["val"].sum() else np.nan
        test_loss = float(criterion(torch.tensor(all_pred[masks["test"]]), torch.tensor(y[masks["test"]])).item()) if masks["test"].sum() else np.nan
        logs.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "test_loss": test_loss})
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
    torch.save({"state_dict": model.state_dict(), "config": cfg, "feature_columns": feature_cols}, out_path(cfg, "asset_model"))
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(x, dtype=torch.float32)).detach().cpu().numpy()
    pred_df = write_asset_prediction_frame(cfg, df, pred, y)
    pred_df = append_asset_live_prediction_frame(cfg, pred_df, model)
    pd.DataFrame(logs).to_csv(out_path(cfg, "asset_training_log"), index=False, encoding="utf-8-sig")
    rows = []
    for split in ["train", "val", "test", "prediction"]:
        m = prediction_metrics(pred[masks[split]], y[masks[split]])
        m["split"] = split
        m["model"] = "asset_itransformer_signal"
        rows.append(m)
    report = pd.DataFrame(rows)[["model", "split", "rows", "mse", "rank_ic", "top1_hit_rate", "top2_realized_return", "long_short_spread"]]
    report.to_csv(out_path(cfg, "asset_validation_report"), index=False, encoding="utf-8-sig")
    return {"prediction_rows": len(pred_df), "epochs_run": len(logs), "best_val_loss": best_val}

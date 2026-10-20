from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

from src.point_in_time import causal_fill_features, normalization_metadata, transform_point_in_time_features


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data_processed"
REPORTS = ROOT / "output" / "reports"


RISK_SLEEVES = ["rates", "credit", "convertible_bonds", "equity", "alternatives"]


@dataclass(frozen=True)
class ITransformerPaths:
    config: Path = ROOT / "config" / "itransformer_config.yaml"
    returns: Path = PROCESSED / "asset_returns_monthly.parquet"
    features: Path = PROCESSED / "feature_panel_monthly_v43.parquet"
    rate_curves: Path = PROCESSED / "rate_curves_daily_v43.parquet"


def load_config(path: Path | None = None) -> dict[str, Any]:
    with (path or ITransformerPaths().config).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def out_path(cfg: dict[str, Any], key: str) -> Path:
    return ROOT / cfg["outputs"][key]


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def build_rate_curve_proxy_returns(cfg: dict[str, Any], paths: ITransformerPaths | None = None) -> pd.DataFrame:
    paths = paths or ITransformerPaths()
    proxy_cfg = cfg.get("synthetic_assets", {}).get("rate_30y_curve_proxy", {})
    if not proxy_cfg.get("enabled", False):
        return pd.DataFrame()
    curves = pd.read_parquet(paths.rate_curves)
    curves["date"] = pd.to_datetime(curves["date"], errors="coerce")
    y_col = str(proxy_cfg.get("yield_column", "treasury_y30"))
    daily = curves[["date", y_col]].dropna().sort_values("date").copy()
    daily[y_col] = pd.to_numeric(daily[y_col], errors="coerce")
    monthly = daily.set_index("date")[y_col].resample("ME").last().dropna().to_frame("yield_pct")
    duration = float(proxy_cfg.get("modified_duration", 18.5))
    carry = monthly["yield_pct"].shift(1) / 100.0 / 12.0
    dy = monthly["yield_pct"].diff() / 100.0
    ret = carry - duration * dy
    close = (1.0 + ret.fillna(0.0)).cumprod() * float(proxy_cfg.get("base_close", 100.0))
    out = pd.DataFrame(
        {
            "date": monthly.index,
            "asset_code": str(proxy_cfg.get("asset_code", "rate_30y_curve_proxy")),
            "asset_name": str(proxy_cfg.get("asset_name", "30Y国债曲线代理")),
            "module": "rates",
            "close": close.to_numpy(dtype=float),
            "return_1m": ret.to_numpy(dtype=float),
            "data_source": f"synthetic:{y_col}:duration_{duration:g}",
        }
    )
    start = pd.Timestamp(cfg["data"]["start_date"])
    return out[out["date"].ge(start)].reset_index(drop=True)


def load_monthly_returns_with_proxy(cfg: dict[str, Any], paths: ITransformerPaths | None = None) -> pd.DataFrame:
    paths = paths or ITransformerPaths()
    df = pd.read_parquet(paths.returns)
    proxy = build_rate_curve_proxy_returns(cfg, paths)
    if not proxy.empty:
        proxy_code = proxy["asset_code"].iloc[0]
        df = df[df["asset_code"].astype(str).ne(proxy_code)].copy()
        df = pd.concat([df, proxy], ignore_index=True, sort=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["return_1m"] = pd.to_numeric(df["return_1m"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    df.loc[df["return_1m"].abs() > 0.80, "return_1m"] = np.nan
    return df.dropna(subset=["date", "asset_code", "module"]).sort_values(["date", "asset_code"]).reset_index(drop=True)


def build_sleeve_returns(cfg: dict[str, Any], paths: ITransformerPaths | None = None) -> pd.DataFrame:
    returns = load_monthly_returns_with_proxy(cfg, paths)
    preferred_rates = {"rate_30y_curve_proxy", "rate_7_10y"}
    parts = []
    for sleeve in RISK_SLEEVES:
        sub = returns[returns["module"].eq(sleeve)].copy()
        if sleeve == "rates":
            sub = sub[sub["asset_code"].isin(preferred_rates)]
        part = sub.groupby("date", as_index=False)["return_1m"].mean()
        part["sleeve"] = sleeve
        parts.append(part)
    long = pd.concat(parts, ignore_index=True)
    wide = long.pivot(index="date", columns="sleeve", values="return_1m").reindex(columns=RISK_SLEEVES).sort_index()
    return wide


def selected_feature_columns(features: pd.DataFrame, cfg: dict[str, Any]) -> list[str]:
    prefixes = tuple(cfg["data"].get("feature_prefixes", []))
    numeric_cols = [c for c in features.columns if c not in {"date", "month"} and pd.api.types.is_numeric_dtype(features[c])]
    cols = [c for c in numeric_cols if not prefixes or c.startswith(prefixes)]
    train = features[pd.to_datetime(features["date"], errors="coerce").le(pd.Timestamp(cfg["data"]["train_end"]))]
    quality = []
    for c in cols:
        s = pd.to_numeric(train[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        miss = float(s.isna().mean())
        var = float(s.var(skipna=True)) if s.notna().sum() > 2 else 0.0
        if miss <= 0.50 and var > 1e-12:
            quality.append((miss, -var, c))
    quality = sorted(quality)
    return [c for _, _, c in quality[: int(cfg["data"].get("max_features", 64))]]


def build_feature_frame(cfg: dict[str, Any], paths: ITransformerPaths | None = None) -> tuple[pd.DataFrame, list[str]]:
    paths = paths or ITransformerPaths()
    features = pd.read_parquet(paths.features)
    features["date"] = pd.to_datetime(features["date"], errors="coerce")
    features = features.dropna(subset=["date"]).sort_values("date").drop_duplicates("date")
    features = features[features["date"].ge(pd.Timestamp(cfg["data"]["start_date"]))].copy()
    feature_cols = selected_feature_columns(features, cfg)
    frame = features[["date"] + feature_cols].copy()
    for c in feature_cols:
        frame[c] = pd.to_numeric(frame[c], errors="coerce").replace([np.inf, -np.inf], np.nan)

    sleeve_ret = build_sleeve_returns(cfg, paths)
    sleeve_ret = sleeve_ret[sleeve_ret.index >= pd.Timestamp(cfg["data"]["start_date"])]
    sleeve_features = sleeve_ret.copy()
    for sleeve in RISK_SLEEVES:
        sleeve_features[f"{sleeve}_ret_1m"] = sleeve_features[sleeve]
        sleeve_features[f"{sleeve}_mom_3m"] = sleeve_features[sleeve].rolling(3).sum()
        sleeve_features[f"{sleeve}_vol_6m"] = sleeve_features[sleeve].rolling(6).std()
    sleeve_features = sleeve_features[[c for c in sleeve_features.columns if c not in RISK_SLEEVES]].reset_index()
    all_features = frame.merge(sleeve_features, on="date", how="left").sort_values("date")
    added_cols = [c for c in all_features.columns if c not in {"date"} and c not in feature_cols]
    all_cols = feature_cols + added_cols
    all_features, all_cols = causal_fill_features(
        all_features,
        all_cols,
        cfg.get("data", {}).get("preprocessing", {}),
    )
    return all_features[["date"] + all_cols], all_cols


def robust_scale(train_values: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    med = np.nanmedian(train_values, axis=0)
    mad = np.nanmedian(np.abs(train_values - med), axis=0) * 1.4826
    mad = np.where(mad < 1e-8, 1.0, mad)
    scaled = np.clip((values - med) / mad, -5.0, 5.0)
    return scaled, med, mad


def build_dataset(cfg: dict[str, Any] | None = None, paths: ITransformerPaths | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    paths = paths or ITransformerPaths()
    feature_frame, feature_cols = build_feature_frame(cfg, paths)
    sleeve_ret = build_sleeve_returns(cfg, paths)
    horizon = int(cfg["data"]["horizon_months"])
    seq_len = int(cfg["data"]["sequence_length"])
    labels = sleeve_ret.shift(-horizon).add_prefix("target_").reset_index().rename(columns={"index": "date"})
    data = feature_frame.merge(labels, on="date", how="inner").sort_values("date")
    data = data.dropna(subset=[f"target_{s}" for s in RISK_SLEEVES]).reset_index(drop=True)

    preprocessing = cfg.get("data", {}).get("preprocessing", {})
    scaled, feature_cols = transform_point_in_time_features(data, feature_cols, preprocessing)

    rows: list[dict[str, Any]] = []
    for idx in range(seq_len - 1, len(data)):
        window = scaled.iloc[idx - seq_len + 1 : idx + 1].to_numpy(dtype=float)
        row: dict[str, Any] = {
            "as_of_date": data.loc[idx, "date"],
            "label_date": data.loc[idx, "date"] + pd.offsets.MonthEnd(horizon),
        }
        for t in range(seq_len):
            for j, col in enumerate(feature_cols):
                row[f"x_t{t:02d}_{col}"] = float(window[t, j])
        targets = data.loc[idx, [f"target_{s}" for s in RISK_SLEEVES]].astype(float)
        for sleeve in RISK_SLEEVES:
            row[f"target_{sleeve}"] = float(targets[f"target_{sleeve}"])
        row["winner"] = RISK_SLEEVES[int(np.argmax(targets.to_numpy(dtype=float)))]
        rows.append(row)
    out = pd.DataFrame(rows)
    path = out_path(cfg, "dataset")
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    meta = {
        "version": cfg["version"],
        "seq_len": seq_len,
        "horizon_months": horizon,
        "feature_count": len(feature_cols),
        "feature_columns": feature_cols,
        "target_sleeves": RISK_SLEEVES,
        "rows": len(out),
        "start": str(out["as_of_date"].min().date()) if not out.empty else "",
        "end": str(out["as_of_date"].max().date()) if not out.empty else "",
        "train_end": cfg["data"]["train_end"],
        "val_end": cfg["data"]["val_end"],
        "normalization": normalization_metadata(preprocessing),
    }
    meta_path = out_path(cfg, "metadata")
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def load_dataset_arrays(cfg: dict[str, Any] | None = None) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    cfg = cfg or load_config()
    df = pd.read_parquet(out_path(cfg, "dataset"))
    meta = json.loads(out_path(cfg, "metadata").read_text(encoding="utf-8"))
    feature_cols = meta["feature_columns"]
    seq_len = int(meta["seq_len"])
    x_cols = [f"x_t{t:02d}_{col}" for t in range(seq_len) for col in feature_cols]
    y_cols = [f"target_{s}" for s in RISK_SLEEVES]
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


def softmax_np(x: np.ndarray, temperature: float = 0.01) -> np.ndarray:
    z = x / max(temperature, 1e-8)
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)


def prediction_metrics(pred: np.ndarray, y: np.ndarray) -> dict[str, float]:
    if len(y) == 0:
        return {"rows": 0}
    pred_rank = np.argsort(np.argsort(pred, axis=1), axis=1)
    true_rank = np.argsort(np.argsort(y, axis=1), axis=1)
    rank_ic = [pd.Series(pred_rank[i]).corr(pd.Series(true_rank[i]), method="spearman") for i in range(len(y))]
    top1 = np.argmax(pred, axis=1)
    true_top1 = np.argmax(y, axis=1)
    top2_pred = np.argsort(pred, axis=1)[:, -2:]
    top2_realized = np.array([y[i, top2_pred[i]].mean() for i in range(len(y))])
    return {
        "rows": float(len(y)),
        "mse": float(np.mean((pred - y) ** 2)),
        "rank_ic": float(np.nanmean(rank_ic)),
        "top1_hit_rate": float(np.mean(top1 == true_top1)),
        "top2_realized_return": float(np.mean(top2_realized)),
        "long_short_spread": float(np.mean(y[np.arange(len(y)), top1] - y[np.arange(len(y)), np.argmin(pred, axis=1)])),
    }


class InvertedTimeSeriesTransformer(nn.Module):
    def __init__(self, seq_len: int, n_features: int, n_targets: int, d_model: int, n_heads: int, num_layers: int, dropout: float):
        super().__init__()
        self.value_embedding = nn.Linear(seq_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_targets),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = x.transpose(1, 2)
        h = self.value_embedding(tokens)
        h = self.encoder(h)
        pooled = self.norm(h).mean(dim=1)
        return self.head(pooled)


def write_prediction_frame(cfg: dict[str, Any], df: pd.DataFrame, pred: np.ndarray, y: np.ndarray, path_key: str) -> pd.DataFrame:
    temperature = float(cfg.get("model", {}).get("probability_temperature", 0.03))
    probs = softmax_np(pred, temperature=temperature)
    rows = []
    for i, row in df.reset_index(drop=True).iterrows():
        for j, sleeve in enumerate(RISK_SLEEVES):
            rows.append(
                {
                    "as_of_date": row["as_of_date"],
                    "label_date": row["label_date"],
                    "sleeve": sleeve,
                    "pred_return": float(pred[i, j]),
                    "pred_prob": float(probs[i, j]),
                    "realized_return": float(y[i, j]),
                    "pred_rank": int((-pred[i]).argsort().tolist().index(j) + 1),
                    "realized_rank": int((-y[i]).argsort().tolist().index(j) + 1),
                }
            )
    out = pd.DataFrame(rows)
    path = out_path(cfg, path_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    return out


def build_live_feature_sequences(cfg: dict[str, Any], metadata_key: str = "metadata") -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    meta = json.loads(out_path(cfg, metadata_key).read_text(encoding="utf-8"))
    feature_cols = list(meta["feature_columns"])
    seq_len = int(meta["seq_len"])
    horizon = int(meta["horizon_months"])
    features, _ = build_feature_frame(cfg)
    features = features.sort_values("date").drop_duplicates("date").copy()
    for col in feature_cols:
        if col not in features.columns:
            features[col] = 0.0
    if str(meta.get("normalization", {}).get("method", "")).startswith("causal_"):
        scaled, _ = transform_point_in_time_features(
            features,
            feature_cols,
            cfg.get("data", {}).get("preprocessing", meta.get("normalization", {})),
        )
        scaled = scaled.to_numpy(dtype=float)
    else:
        values = features[feature_cols].to_numpy(dtype=float)
        med = np.asarray(meta["scaler_median"], dtype=float)
        mad = np.asarray(meta["scaler_mad"], dtype=float)
        mad = np.where(mad < 1e-8, 1.0, mad)
        scaled = np.clip((values - med) / mad, -5.0, 5.0)
    rows: list[dict[str, Any]] = []
    windows = []
    for idx in range(seq_len - 1, len(features)):
        as_of = pd.Timestamp(features.iloc[idx]["date"])
        rows.append({"as_of_date": as_of, "label_date": as_of + pd.offsets.MonthEnd(horizon)})
        windows.append(scaled[idx - seq_len + 1 : idx + 1])
    if not rows:
        return pd.DataFrame(columns=["as_of_date", "label_date"]), np.empty((0, seq_len, len(feature_cols)), dtype=np.float32), feature_cols
    return pd.DataFrame(rows), np.asarray(windows, dtype=np.float32), feature_cols


def append_live_prediction_frame(
    cfg: dict[str, Any],
    labeled_predictions: pd.DataFrame,
    model: nn.Module,
    metadata_key: str = "metadata",
    path_key: str = "predictions",
) -> pd.DataFrame:
    live_df, live_x, _ = build_live_feature_sequences(cfg, metadata_key=metadata_key)
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
    rows = []
    for i, row in live_df.iterrows():
        order = (-pred[i]).argsort().tolist()
        for j, sleeve in enumerate(RISK_SLEEVES):
            rows.append(
                {
                    "as_of_date": row["as_of_date"],
                    "label_date": row["label_date"],
                    "sleeve": sleeve,
                    "pred_return": float(pred[i, j]),
                    "pred_prob": float(probs[i, j]),
                    "realized_return": np.nan,
                    "pred_rank": int(order.index(j) + 1),
                    "realized_rank": np.nan,
                }
            )
    out = pd.concat([labeled_predictions, pd.DataFrame(rows)], ignore_index=True, sort=False)
    path = out_path(cfg, path_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    return out

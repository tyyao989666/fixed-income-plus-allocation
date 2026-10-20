from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "output" / "figures" / "itransformer"
REPORT_DIR = ROOT / "output" / "reports"
PROCESSED = ROOT / "data_processed"

sys.path.insert(0, str(ROOT))

from src.itransformer_signal import RISK_SLEEVES, load_config, out_path  # noqa: E402


SLEEVE_LABELS = {
    "rates": "利率",
    "credit": "信用",
    "convertible_bonds": "转债",
    "equity": "权益",
    "alternatives": "另类",
}


def set_plot_style() -> None:
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 180
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]


def plot_loss_curve() -> Path:
    log = pd.read_csv(REPORT_DIR / "itransformer_training_log.csv")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(log["epoch"], log["train_loss"], label="train", linewidth=2.0)
    ax.plot(log["epoch"], log["val_loss"], label="val", linewidth=2.0)
    ax.plot(log["epoch"], log["test_loss"], label="test", linewidth=1.5, alpha=0.7)
    best_idx = int(log["val_loss"].astype(float).idxmin())
    ax.scatter(log.loc[best_idx, "epoch"], log.loc[best_idx, "val_loss"], color="#D95F02", zorder=3, label="best val")
    ax.set_title("iTransformer 训练损失")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("SmoothL1Loss")
    ax.grid(True, alpha=0.22)
    ax.legend(frameon=False)
    path = FIG_DIR / "itransformer_training_loss.png"
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def metric_frame() -> pd.DataFrame:
    base = pd.read_csv(REPORT_DIR / "baseline_prediction_report.csv")
    it = pd.read_csv(REPORT_DIR / "itransformer_validation_report.csv")
    if "seed" not in base.columns:
        base["seed"] = np.nan
    return pd.concat([base, it], ignore_index=True, sort=False)


def plot_metric_compare() -> Path:
    df = metric_frame()
    keep = df[df["split"].isin(["val", "test", "prediction"])].copy()
    metrics = ["rank_ic", "top1_hit_rate", "long_short_spread"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(14, 4))
    for ax, metric in zip(axes, metrics):
        pivot = keep.pivot(index="split", columns="model", values=metric).reindex(["val", "test", "prediction"])
        pivot.plot(kind="bar", ax=ax, width=0.72)
        ax.set_title(metric)
        ax.grid(True, axis="y", alpha=0.22)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Baseline vs iTransformer 信号指标")
    fig.tight_layout()
    path = FIG_DIR / "itransformer_metric_compare.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def top1_by_month(pred_path: Path) -> pd.DataFrame:
    pred = pd.read_parquet(pred_path)
    pred["as_of_date"] = pd.to_datetime(pred["as_of_date"])
    pred["label_date"] = pd.to_datetime(pred["label_date"])
    top_pred = pred.loc[pred.groupby("as_of_date")["pred_return"].idxmax(), ["as_of_date", "label_date", "sleeve"]].rename(columns={"sleeve": "pred_top1"})
    top_true = pred.loc[pred.groupby("as_of_date")["realized_return"].idxmax(), ["as_of_date", "sleeve"]].rename(columns={"sleeve": "realized_top1"})
    out = top_pred.merge(top_true, on="as_of_date", how="left")
    out["hit"] = out["pred_top1"].eq(out["realized_top1"])
    return out.sort_values("as_of_date")


def plot_prediction_timeline() -> Path:
    top = top1_by_month(PROCESSED / "itransformer_predictions_monthly.parquet")
    y_map = {s: i for i, s in enumerate(RISK_SLEEVES)}
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.scatter(top["label_date"], top["pred_top1"].map(y_map), marker="o", s=44, label="pred top1", alpha=0.85)
    ax.scatter(top["label_date"], top["realized_top1"].map(y_map), marker="x", s=54, label="realized top1", alpha=0.85)
    hit_rate = top["hit"].mean()
    ax.set_title(f"每月预测最强袖套 vs 实际最强袖套: hit={hit_rate:.1%}")
    ax.set_yticks(range(len(RISK_SLEEVES)))
    ax.set_yticklabels([SLEEVE_LABELS[s] for s in RISK_SLEEVES])
    ax.grid(True, alpha=0.22)
    ax.legend(frameon=False)
    path = FIG_DIR / "itransformer_top1_timeline.png"
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_multiseed_summary() -> Path | None:
    path = REPORT_DIR / "itransformer_multiseed_report.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    test = df[df["split"].eq("test")].copy()
    if test.empty:
        return None
    metrics = ["rank_ic", "top1_hit_rate", "long_short_spread"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(14, 4))
    for ax, metric in zip(axes, metrics):
        ax.bar(test["seed"].astype(str), test[metric], color="#4E79A7", alpha=0.85)
        ax.axhline(test[metric].mean(), color="#D95F02", linestyle="--", linewidth=1.5, label="mean")
        ax.set_title(f"test {metric}")
        ax.grid(True, axis="y", alpha=0.22)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("iTransformer 多 seed 稳定性")
    fig.tight_layout()
    out = FIG_DIR / "itransformer_multiseed_test.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def render_html(figures: list[Path], cfg: dict) -> Path:
    metrics = metric_frame()
    html = [
        "<html><head><meta charset='utf-8'><title>iTransformer Signal Report</title>",
        "<style>body{font-family:Arial,'Microsoft YaHei',sans-serif;margin:28px;color:#222} img{max-width:100%;border:1px solid #ddd;margin:12px 0} table{border-collapse:collapse;margin:12px 0} th,td{border:1px solid #ddd;padding:6px 8px;text-align:right} th:first-child,td:first-child{text-align:left}</style>",
        "</head><body>",
        "<h1>iTransformer Signal Report</h1>",
        f"<p>epochs={cfg['model']['epochs']}, patience={cfg['model']['patience']}, seed={cfg['model']['seed']}, loss=SmoothL1Loss.</p>",
        "<h2>Validation Metrics</h2>",
        metrics.to_html(index=False, float_format=lambda x: f"{x:.4f}"),
        "<h2>Figures</h2>",
    ]
    for fig in figures:
        rel = fig.relative_to(ROOT).as_posix()
        html.append(f"<h3>{fig.stem}</h3><img src='../../{rel}'>")
    html.append("</body></html>")
    out = out_path(cfg, "visual_report")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(html), encoding="utf-8")
    return out


def main() -> None:
    cfg = load_config()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    set_plot_style()
    figures = [plot_loss_curve(), plot_metric_compare(), plot_prediction_timeline()]
    multi = plot_multiseed_summary()
    if multi is not None:
        figures.append(multi)
    html = render_html(figures, cfg)
    manifest = {
        "figures": [str(p.relative_to(ROOT)) for p in figures],
        "html_report": str(html.relative_to(ROOT)),
    }
    manifest_path = REPORT_DIR / "itransformer_visual_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()

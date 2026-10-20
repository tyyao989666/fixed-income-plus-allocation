from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config/hierarchical_asset_itransformer_v7.yaml"


def load_training_log(log_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(log_path)
    frame["epoch_num"] = pd.to_numeric(frame["epoch"], errors="coerce")
    frame["seed_num"] = pd.to_numeric(frame["seed"], errors="coerce")
    return frame


def load_resumable_progress(cfg: dict, fallback: pd.DataFrame) -> pd.DataFrame:
    progress_value = cfg.get("outputs", {}).get("training_progress")
    if not progress_value:
        return fallback
    progress_path = ROOT / progress_value
    if not progress_path.exists():
        return fallback
    rows = []
    for line in progress_path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row.get("epoch"), int):
            rows.append(row)
    if not rows:
        return fallback
    frame = pd.DataFrame(rows)
    frame["epoch_num"] = pd.to_numeric(frame["epoch"], errors="coerce")
    frame["seed_num"] = pd.to_numeric(frame["seed"], errors="coerce")
    frame = frame.drop_duplicates(["year", "seed_num", "epoch_num"], keep="last")

    model_dir = ROOT / cfg["outputs"]["model_dir"]
    checkpoint_rows = []
    for path in sorted(model_dir.glob("model_*_seed_*.pt")):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        checkpoint_rows.append(
            {
                "year": int(checkpoint["year"]),
                "seed": int(checkpoint["seed"]),
                "seed_num": int(checkpoint["seed"]),
                "epoch": "checkpoint_selection",
                "epoch_num": float("nan"),
                "best_epoch": int(checkpoint["best_epoch"]),
                "best_validation_loss": float(checkpoint["best_validation_loss"]),
                "selected_checkpoint_rank_score": float(
                    checkpoint["selected_checkpoint_rank_score"]
                ),
                "checkpoint_selection": checkpoint["checkpoint_selection"],
                "epochs_run": int(checkpoint["epochs_run"]),
            }
        )
    if checkpoint_rows:
        frame = pd.concat([frame, pd.DataFrame(checkpoint_rows)], ignore_index=True, sort=False)
    return frame


def build_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    numeric_seed = frame.dropna(subset=["seed_num"])
    for (year, seed), all_rows in numeric_seed.groupby(["year", "seed_num"]):
        group = all_rows.dropna(subset=["epoch_num"]).sort_values("epoch_num")
        if group.empty:
            continue
        last = group.iloc[-1]
        checkpoint_rows = all_rows.loc[all_rows["epoch"].astype(str).eq("checkpoint_selection")]
        selected = checkpoint_rows.iloc[-1] if not checkpoint_rows.empty else last
        best_epoch = int(pd.to_numeric(selected["best_epoch"], errors="coerce"))
        selected_epochs_run = pd.to_numeric(selected.get("epochs_run"), errors="coerce")
        epochs_run = (
            int(selected_epochs_run)
            if pd.notna(selected_epochs_run)
            else int(last["epoch_num"])
        )
        checkpoint_method = selected.get("checkpoint_selection", "single_validation_best")
        if pd.isna(checkpoint_method):
            checkpoint_method = "single_validation_best"
        rows.append(
            {
                "year": int(year),
                "seed": int(seed),
                "epochs_run": epochs_run,
                "best_epoch": best_epoch,
                "initial_train_loss": float(group.iloc[0]["train_loss"]),
                "final_train_loss": float(last["train_loss"]),
                "initial_validation_loss": float(group.iloc[0]["validation_loss"]),
                "best_validation_loss": float(selected["best_validation_loss"]),
                "final_validation_loss": float(last["validation_loss"]),
                "selected_checkpoint_rank_ic": float(selected["selected_checkpoint_rank_score"]),
                "checkpoint_selection": str(checkpoint_method),
                "hit_max_epoch": bool(int(last["epoch_num"]) >= 300),
            }
        )
    return pd.DataFrame(rows).sort_values(["year", "seed"])


def plot_curves(frame: pd.DataFrame, summary: pd.DataFrame, figure_path: Path, title: str) -> None:
    years = sorted(frame["year"].astype(int).unique())
    fig, axes = plt.subplots(len(years), 1, figsize=(12, 3.2 * len(years)), constrained_layout=True)
    if len(years) == 1:
        axes = [axes]
    colors = {17: "#0b3c5d", 29: "#d97706", 47: "#14866d"}
    for axis, year in zip(axes, years, strict=True):
        annual = frame.loc[frame["year"].eq(year)]
        for seed, group in annual.groupby("seed_num"):
            seed = int(seed)
            group = group.sort_values("epoch_num")
            color = colors.get(seed, None)
            axis.plot(
                group["epoch_num"],
                group["train_loss"],
                color=color,
                linewidth=1.3,
                alpha=0.75,
                label=f"seed {seed} train",
            )
            axis.plot(
                group["epoch_num"],
                group["validation_loss"],
                color=color,
                linewidth=1.6,
                linestyle="--",
                label=f"seed {seed} validation",
            )
            best_epoch = int(
                summary.loc[
                    summary["year"].eq(year) & summary["seed"].eq(seed), "best_epoch"
                ].iloc[0]
            )
            axis.axvline(best_epoch, color=color, linewidth=0.8, alpha=0.35)
        axis.set_title(f"Walk-forward prediction year {year}")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Loss")
        axis.grid(axis="y", color="#d7dde5", linewidth=0.7, alpha=0.8)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(ncol=3, fontsize=8, frameon=False)
    fig.suptitle(f"{title}: train and purged-validation loss", fontsize=15)
    fig.savefig(figure_path, dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot hierarchical iTransformer training convergence.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    log_path = ROOT / cfg["outputs"]["training_log"]
    prefix = log_path.name.removesuffix("_training_log.csv")
    summary_path = log_path.with_name(f"{prefix}_convergence_summary.csv")
    figure_path = log_path.with_name(f"{prefix}_loss_curves.png")
    training_log = load_resumable_progress(cfg, load_training_log(log_path))
    summary = build_summary(training_log)
    frame = training_log.dropna(subset=["epoch_num", "seed_num"]).copy()
    maximum_epochs = int(cfg["model"]["max_epochs"])
    summary["hit_max_epoch"] = summary["epochs_run"].ge(maximum_epochs)
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    plot_curves(frame, summary, figure_path, str(cfg.get("version", "Hierarchical Asset iTransformer")))
    print(summary.to_string(index=False))
    print(f"summary: {summary_path.relative_to(ROOT)}")
    print(f"figure: {figure_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

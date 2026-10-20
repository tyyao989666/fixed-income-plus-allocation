from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
DEFAULT_CONFIG = ROOT / "config" / "itransformer_daily_enhanced_config.yaml"


def run(args: list[str]) -> None:
    print(" ".join(args), flush=True)
    subprocess.run(args, cwd=ROOT, check=True)


def suffixed(path_value: str, suffix: str) -> Path:
    path = Path(path_value)
    return ROOT / path.with_name(f"{path.stem}{suffix}{path.suffix}")


def metrics(pred: np.ndarray, realized: np.ndarray) -> dict[str, float]:
    pred_rank = np.argsort(np.argsort(pred, axis=1), axis=1)
    true_rank = np.argsort(np.argsort(realized, axis=1), axis=1)
    rank_ic = [pd.Series(pred_rank[i]).corr(pd.Series(true_rank[i]), method="spearman") for i in range(len(pred))]
    top = np.argmax(pred, axis=1)
    bottom = np.argmin(pred, axis=1)
    return {
        "rows": len(pred),
        "mse": float(np.mean((pred - realized) ** 2)),
        "rank_ic": float(np.nanmean(rank_ic)),
        "top1_hit_rate": float(np.mean(top == np.argmax(realized, axis=1))),
        "top2_realized_return": float(np.mean([realized[i, np.argsort(pred[i])[-2:]].mean() for i in range(len(pred))])),
        "long_short_spread": float(np.mean(realized[np.arange(len(pred)), top] - realized[np.arange(len(pred)), bottom])),
    }


def ensemble_report(cfg: dict, seeds: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    for seed in seeds:
        path = suffixed(cfg["outputs"]["predictions"], f"_seed{seed}")
        frame = pd.read_parquet(path)
        frame["seed"] = seed
        frames.append(frame)
    all_predictions = pd.concat(frames, ignore_index=True)
    keys = ["as_of_date", "label_date", "sleeve"]
    ensemble = all_predictions.groupby(keys, as_index=False).agg(
        pred_return=("pred_return", "mean"),
        pred_std=("pred_return", "std"),
        realized_return=("realized_return", "first"),
    )
    rows = []
    dates = ensemble[keys[:2]].drop_duplicates().sort_values("as_of_date")
    for _, date_row in dates.iterrows():
        active = ensemble[ensemble["as_of_date"].eq(date_row["as_of_date"])]
        rank = active["pred_return"].rank(ascending=False, method="first").astype(int)
        ensemble.loc[active.index, "pred_rank"] = rank
    ensemble["model_version"] = cfg["version"] + "_ensemble"
    slug = str(cfg["version"]).replace("/", "_")
    ensemble_path = ROOT / "data_processed" / f"{slug}_ensemble_predictions.parquet"
    ensemble.to_parquet(ensemble_path, index=False)

    for split, mask in {
        "train": ensemble["as_of_date"].le(pd.Timestamp(cfg["data"]["train_end"])),
        "val": ensemble["as_of_date"].gt(pd.Timestamp(cfg["data"]["train_end"])) & ensemble["as_of_date"].le(pd.Timestamp(cfg["data"]["val_end"])),
        "test": ensemble["as_of_date"].gt(pd.Timestamp(cfg["data"]["val_end"])),
        "prediction": ensemble["as_of_date"].ge(pd.Timestamp(cfg["data"]["prediction_start"])),
    }.items():
        active = ensemble[mask].pivot(index="as_of_date", columns="sleeve", values=["pred_return", "realized_return"])
        if active.empty:
            continue
        pred = active["pred_return"].to_numpy()
        realized = active["realized_return"].to_numpy()
        row = metrics(pred, realized)
        row.update({"model": cfg["version"] + "_ensemble", "split": split, "seed_count": len(seeds)})
        rows.append(row)
    report = pd.DataFrame(rows)
    report.to_csv(ROOT / "output" / "reports" / f"{slug}_ensemble_report.csv", index=False, encoding="utf-8-sig")
    return ensemble, report


def write_comparison(enhanced: pd.DataFrame, cfg: dict) -> Path:
    reports = []
    base_path = ROOT / "output" / "reports" / "itransformer_daily_validation_report.csv"
    if base_path.exists():
        base = pd.read_csv(base_path)
        base["variant"] = "原日频路径"
        reports.append(base)
    enhanced = enhanced.copy()
    enhanced["variant"] = "OHLCV+排序损失多seed集成"
    reports.append(enhanced)
    comparison = pd.concat(reports, ignore_index=True, sort=False)
    path = ROOT / "output" / "reports" / f"{cfg['version']}_comparison.csv"
    comparison.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build, train and compare the enhanced daily-path iTransformer.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG.relative_to(ROOT)))
    parser.add_argument("--seeds", default="", help="Comma-separated seeds; defaults to config experiment.seeds.")
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()
    config_path = ROOT / args.config
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()] or [int(x) for x in cfg["experiment"]["seeds"]]

    if not args.skip_build:
        run([PYTHON, "scripts/65_build_itransformer_daily_dataset.py", "--config", args.config])
    for seed in seeds:
        run([PYTHON, "scripts/66_train_itransformer_daily.py", "--config", args.config, "--seed", str(seed), "--output-suffix", f"_seed{seed}"])
    _, report = ensemble_report(cfg, seeds)
    comparison_path = write_comparison(report, cfg)
    print(json.dumps({"seeds": seeds, "version": cfg["version"], "comparison": str(comparison_path.relative_to(ROOT))}, ensure_ascii=False))


if __name__ == "__main__":
    main()

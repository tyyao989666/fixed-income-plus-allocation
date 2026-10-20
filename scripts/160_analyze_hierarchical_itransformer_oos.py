from __future__ import annotations

"""Audit every raw hierarchical iTransformer view on strict OOS rows."""

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.hierarchical_asset_itransformer import load_config  # noqa: E402


def daily_rank_ic(frame: pd.DataFrame) -> float:
    valid = frame[["model_raw_score", "realized_relative_score"]].dropna()
    if (
        len(valid) < 2
        or valid["model_raw_score"].nunique() < 2
        or valid["realized_relative_score"].nunique() < 2
    ):
        return float("nan")
    return float(
        valid["model_raw_score"]
        .rank(method="average")
        .corr(
            valid["realized_relative_score"].rank(method="average")
        )
    )


def circular_block_bootstrap(
    values: np.ndarray,
    block_days: int,
    draws: int,
    confidence: float,
    seed: int,
) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    block_days = max(int(block_days), 2)
    result = {
        "bootstrap_ci_low": float("-inf"),
        "bootstrap_ci_high": float("inf"),
        "bootstrap_positive_probability": 0.5,
    }
    if len(values) < max(20, 2 * block_days) or draws <= 0:
        return result
    rng = np.random.default_rng(int(seed))
    blocks_needed = int(math.ceil(len(values) / block_days))
    offsets = np.arange(block_days, dtype=np.int64)
    estimates = np.empty(int(draws), dtype=float)
    for start in range(0, int(draws), 256):
        stop = min(start + 256, int(draws))
        starts = rng.integers(
            0,
            len(values),
            size=(stop - start, blocks_needed),
        )
        indices = (
            starts[:, :, None] + offsets[None, None, :]
        ) % len(values)
        samples = values[
            indices.reshape(stop - start, -1)[:, : len(values)]
        ]
        estimates[start:stop] = samples.mean(axis=1)
    tail = 0.5 * (1.0 - float(confidence))
    return {
        "bootstrap_ci_low": float(np.quantile(estimates, tail)),
        "bootstrap_ci_high": float(np.quantile(estimates, 1.0 - tail)),
        "bootstrap_positive_probability": float(
            np.mean(estimates > 0.0)
        ),
    }


def component_rows(
    frame: pd.DataFrame,
    task: str,
    draws: int,
    confidence: float,
    seed: int,
) -> list[dict[str, Any]]:
    grouping = ["horizon_days"]
    if task == "asset":
        grouping.append("module")
    rows: list[dict[str, Any]] = []
    for key, component in frame.groupby(grouping, sort=True):
        if task == "asset":
            horizon, module = key
            date_grouping = ["as_of_date", "module"]
        else:
            horizon = key[0] if isinstance(key, tuple) else key
            module = ""
            date_grouping = ["as_of_date"]
        daily = (
            component.groupby(date_grouping, sort=True)
            .apply(daily_rank_ic, include_groups=False)
            .rename("rank_ic")
            .reset_index()
        )
        daily["as_of_date"] = pd.to_datetime(daily["as_of_date"])
        daily = daily.loc[daily["rank_ic"].notna()].copy()
        if daily.empty:
            continue
        yearly = daily.groupby(daily["as_of_date"].dt.year)[
            "rank_ic"
        ].mean()
        active_years = sorted(
            component.loc[
                pd.to_numeric(
                    component["residual_alpha"], errors="coerce"
                ).fillna(0.0)
                > 0.0,
                "as_of_date",
            ]
            .dt.year.unique()
            .tolist()
        )
        bootstrap = circular_block_bootstrap(
            daily["rank_ic"].to_numpy(dtype=float),
            int(horizon),
            int(draws),
            float(confidence),
            int(seed)
            + int(horizon) * 10_000
            + (0 if task == "asset" else 1_000_000)
            + sum(ord(char) for char in str(module)),
        )
        row: dict[str, Any] = {
            "task": task,
            "module": str(module),
            "horizon_days": int(horizon),
            "observations": int(len(daily)),
            "effective_nonoverlap_observations": int(
                math.ceil(len(daily) / max(int(horizon), 1))
            ),
            "years": int(len(yearly)),
            "active_years": ",".join(str(value) for value in active_years),
            "mean_rank_ic": float(daily["rank_ic"].mean()),
            "positive_year_fraction": float((yearly > 0.0).mean()),
            "worst_year_rank_ic": float(yearly.min()),
            **bootstrap,
        }
        for year, value in yearly.items():
            row[f"rank_ic_{int(year)}"] = float(value)
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
    )
    parser.add_argument("--draws", type=int, default=4000)
    parser.add_argument("--confidence", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    cfg = load_config(args.config)
    asset_path = ROOT / cfg["outputs"]["asset_predictions"]
    sleeve_path = ROOT / cfg["outputs"]["sleeve_predictions"]
    asset = pd.read_parquet(asset_path)
    sleeve = pd.read_parquet(sleeve_path)
    for frame in (asset, sleeve):
        frame["as_of_date"] = pd.to_datetime(frame["as_of_date"])
        frame["has_label"] = frame["has_label"].astype(bool)
    asset = asset.loc[asset["has_label"]].copy()
    sleeve = sleeve.loc[sleeve["has_label"]].copy()

    rows = [
        *component_rows(
            asset,
            "asset",
            args.draws,
            args.confidence,
            args.seed,
        ),
        *component_rows(
            sleeve,
            "sleeve",
            args.draws,
            args.confidence,
            args.seed,
        ),
    ]
    report = pd.DataFrame(rows).sort_values(
        ["task", "horizon_days", "module"]
    )
    output = args.output
    if output is None:
        output = (
            ROOT
            / "output"
            / "reports"
            / f"{cfg['version']}_raw_oos_evidence.csv"
        )
    elif not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output, index=False, encoding="utf-8-sig")
    print(report.to_string(index=False))
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()

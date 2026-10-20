from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.hierarchical_asset_itransformer import (  # noqa: E402
    HierarchicalAssetITransformer,
    WindowDataset,
    load_arrays,
    load_config,
    project_path,
)


def make_model(arrays, metadata: dict[str, Any], cfg: dict[str, Any]) -> HierarchicalAssetITransformer:
    return HierarchicalAssetITransformer(
        sequence_length=int(metadata["sequence_length_days"]),
        local_features=len(metadata["local_features"]),
        macro_features=len(metadata["macro_features"]),
        module_index=arrays.module_index,
        horizons=len(metadata["horizons_days"]),
        cfg=cfg["model"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit point-in-time 3x3 regime shrinkage effects.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--year", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output-prefix")
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    if cfg["model"].get("sleeve_architecture") != "hierarchical_regime_pairwise":
        raise ValueError("This audit requires hierarchical_regime_pairwise")
    arrays, metadata = load_arrays(cfg)
    dates = pd.to_datetime(arrays.dates)
    indices = arrays.sample_indices[dates[arrays.sample_indices].year == args.year]
    if not len(indices):
        raise ValueError(f"No prediction samples for year={args.year}")

    checkpoint_path = project_path(cfg, "model_dir") / f"model_{args.year}_seed_{args.seed}.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = make_model(arrays, metadata, cfg)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    loader = DataLoader(
        WindowDataset(arrays, indices, int(metadata["sequence_length_days"])),
        batch_size=int(cfg["model"]["batch_size"]),
        shuffle=False,
    )

    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            actual = model(
                batch["local_values"],
                batch["local_masks"],
                batch["macro_values"],
                batch["regime_index"],
                batch["regime_prior_count"],
            )
            shared_only = model(
                batch["local_values"],
                batch["local_masks"],
                batch["macro_values"],
                batch["regime_index"],
                torch.zeros_like(batch["regime_prior_count"]),
            )
            effect = (actual[4] - shared_only[4]).abs()
            for position, array_index in enumerate(batch["index"].numpy()):
                regime_index = int(batch["regime_index"][position])
                prior_count = int(batch["regime_prior_count"][position])
                expected_count = int(np.sum(arrays.regime_index[:array_index] == regime_index))
                if prior_count != expected_count:
                    raise AssertionError(
                        f"non-causal regime count at index={array_index}: {prior_count} != {expected_count}"
                    )
                rows.append(
                    {
                        "as_of_date": dates[array_index],
                        "regime_index": regime_index,
                        "regime": metadata["regime_names"][regime_index],
                        "regime_prior_count": prior_count,
                        "adjustment_weight": float(actual[5][position, 0]),
                        "rank_effect_mean_abs": float(effect[position].mean()),
                        "rank_effect_max_abs": float(effect[position].max()),
                        "regime_residual_rms": float(actual[6][position].square().mean().sqrt()),
                    }
                )

    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby("regime", sort=False)
        .agg(
            samples=("as_of_date", "size"),
            first_prior_count=("regime_prior_count", "first"),
            last_prior_count=("regime_prior_count", "last"),
            mean_adjustment_weight=("adjustment_weight", "mean"),
            mean_rank_effect_abs=("rank_effect_mean_abs", "mean"),
            max_rank_effect_abs=("rank_effect_max_abs", "max"),
            mean_regime_residual_rms=("regime_residual_rms", "mean"),
        )
        .reset_index()
    )
    overall = pd.DataFrame(
        [
            {
                "regime": "all",
                "samples": len(detail),
                "first_prior_count": np.nan,
                "last_prior_count": np.nan,
                "mean_adjustment_weight": detail["adjustment_weight"].mean(),
                "mean_rank_effect_abs": detail["rank_effect_mean_abs"].mean(),
                "max_rank_effect_abs": detail["rank_effect_max_abs"].max(),
                "mean_regime_residual_rms": detail["regime_residual_rms"].mean(),
            }
        ]
    )
    summary = pd.concat([overall, summary], ignore_index=True)

    if args.output_prefix:
        prefix = ROOT / args.output_prefix
    else:
        prefix = ROOT / "output" / "reports" / Path(args.config).stem
        prefix = Path(f"{prefix}_year_{args.year}_seed_{args.seed}_regime")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    detail_path = Path(f"{prefix}_daily_effect.csv")
    summary_path = Path(f"{prefix}_summary.csv")
    detail.to_csv(detail_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))
    print(f"detail: {detail_path.relative_to(ROOT)}")
    print(f"summary: {summary_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

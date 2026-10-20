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


def summarize_gate(frame: pd.DataFrame, expert_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped = [("all", frame)] + list(frame.groupby("regime", sort=False))
    for regime, group in grouped:
        gate = group[expert_columns].to_numpy(dtype=float)
        entropy = -(gate * np.log(np.clip(gate, 1e-12, 1.0))).sum(axis=1)
        dominant = gate.argmax(axis=1)
        row: dict[str, Any] = {
            "regime": regime,
            "samples": len(group),
            "mean_entropy": float(entropy.mean()),
            "effective_experts": float(np.exp(entropy.mean())),
            "mean_dominant_weight": float(gate.max(axis=1).mean()),
            "dominant_expert": int(np.bincount(dominant, minlength=len(expert_columns)).argmax()),
        }
        for expert, column in enumerate(expert_columns):
            row[f"expert_{expert}_mean_weight"] = float(gate[:, expert].mean())
            row[f"expert_{expert}_dominant_fraction"] = float(np.mean(dominant == expert))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit regime-MoE gate usage without using future labels.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--year", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output-prefix")
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    if cfg["model"].get("sleeve_architecture") != "regime_moe_pairwise":
        raise ValueError("This audit requires model.sleeve_architecture=regime_moe_pairwise")
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
    collected_indices: list[np.ndarray] = []
    collected_regimes: list[np.ndarray] = []
    collected_gates: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            output = model(
                batch["local_values"],
                batch["local_masks"],
                batch["macro_values"],
                batch["regime_index"],
            )
            collected_indices.append(batch["index"].numpy())
            collected_regimes.append(batch["regime_index"].numpy())
            collected_gates.append(output[5].numpy())

    sample_indices = np.concatenate(collected_indices)
    regime_indices = np.concatenate(collected_regimes)
    gates = np.concatenate(collected_gates)
    regime_names = metadata["regime_names"]
    expert_columns = [f"expert_{expert}_weight" for expert in range(gates.shape[1])]
    detail = pd.DataFrame(
        {
            "as_of_date": dates[sample_indices],
            "regime_index": regime_indices,
            "regime": [regime_names[int(value)] for value in regime_indices],
            **{column: gates[:, expert] for expert, column in enumerate(expert_columns)},
        }
    )
    detail["dominant_expert"] = gates.argmax(axis=1)
    detail["dominant_weight"] = gates.max(axis=1)
    detail["gate_entropy"] = -(gates * np.log(np.clip(gates, 1e-12, 1.0))).sum(axis=1)
    detail["effective_experts"] = np.exp(detail["gate_entropy"])
    summary = summarize_gate(detail, expert_columns)

    if args.output_prefix:
        prefix = ROOT / args.output_prefix
    else:
        prefix = ROOT / "output" / "reports" / Path(args.config).stem
        prefix = Path(f"{prefix}_year_{args.year}_seed_{args.seed}_moe")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    detail_path = Path(f"{prefix}_daily_gate.csv")
    summary_path = Path(f"{prefix}_expert_usage.csv")
    detail.to_csv(detail_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))
    print(f"detail: {detail_path.relative_to(ROOT)}")
    print(f"summary: {summary_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

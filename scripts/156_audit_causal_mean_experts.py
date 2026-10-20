from __future__ import annotations

"""Audit whether the causal expert library contains routable OOS opportunity."""

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.hierarchical_asset_itransformer import (  # noqa: E402
    load_arrays,
    load_config,
    rank_ic,
)


def oracle_mixture(
    experts: np.ndarray,
    expert_masks: np.ndarray,
    target: np.ndarray,
    target_mask: np.ndarray,
    groups: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    output = np.zeros_like(target, dtype=np.float32)
    output_mask = np.zeros_like(target_mask, dtype=bool)
    unique_groups = [None] if groups is None else np.unique(groups).tolist()
    for row in range(len(target)):
        for group in unique_groups:
            columns = (
                np.arange(target.shape[1])
                if group is None
                else np.flatnonzero(groups == group)
            )
            best_expert = None
            best_ic = -np.inf
            best_valid = None
            for expert in range(experts.shape[2]):
                valid = (
                    target_mask[row, columns].astype(bool)
                    & expert_masks[row, columns, expert].astype(bool)
                )
                if valid.sum() < 2:
                    continue
                prediction_rank = pd.Series(
                    experts[row, columns, expert][valid]
                ).rank(method="average")
                target_rank = pd.Series(
                    target[row, columns][valid]
                ).rank(method="average")
                if prediction_rank.std() <= 1e-12 or target_rank.std() <= 1e-12:
                    continue
                ic = float(np.corrcoef(prediction_rank, target_rank)[0, 1])
                if ic > best_ic:
                    best_ic = ic
                    best_expert = expert
                    best_valid = valid
            if best_expert is None or best_valid is None:
                continue
            output[row, columns] = experts[row, columns, best_expert]
            output_mask[row, columns] = best_valid
    return output, output_mask


def equal_available_mixture(
    experts: np.ndarray, expert_masks: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    masks = expert_masks.astype(float)
    counts = masks.sum(axis=2)
    values = (experts * masks).sum(axis=2) / np.maximum(counts, 1.0)
    return values.astype(np.float32), counts > 0.0


def audit_task(
    task: str,
    experts: np.ndarray,
    expert_masks: np.ndarray,
    target: np.ndarray,
    target_mask: np.ndarray,
    names: list[str],
    groups: np.ndarray | None,
    dates: pd.DatetimeIndex,
    horizons: list[int],
    prediction_start: pd.Timestamp,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    scopes: list[tuple[str, np.ndarray]] = [
        (str(year), (dates.year == year) & (dates >= prediction_start))
        for year in sorted(set(dates[dates >= prediction_start].year))
    ]
    scopes.append(("pooled", dates >= prediction_start))
    for scope, date_mask in scopes:
        selected = np.flatnonzero(date_mask)
        if not len(selected):
            continue
        for horizon_position, horizon in enumerate(horizons):
            selected_target = target[selected, horizon_position]
            selected_target_mask = target_mask[selected, horizon_position]
            selected_experts = experts[selected, horizon_position]
            selected_expert_masks = expert_masks[selected, horizon_position]
            baseline_ic = rank_ic(
                selected_experts[:, :, 0],
                selected_target,
                selected_target_mask & selected_expert_masks[:, :, 0].astype(bool),
                groups,
            )
            for expert_position, expert_name in enumerate(names):
                valid = (
                    selected_target_mask
                    & selected_expert_masks[:, :, expert_position].astype(bool)
                )
                score = rank_ic(
                    selected_experts[:, :, expert_position],
                    selected_target,
                    valid,
                    groups,
                )
                rows.append(
                    {
                        "task": task,
                        "scope": scope,
                        "horizon_days": int(horizon),
                        "candidate": expert_name,
                        "rank_ic": score,
                        "baseline_rank_ic": baseline_ic,
                        "incremental_rank_ic": score - baseline_ic,
                        "dates": int(len(selected)),
                        "diagnostic_only": False,
                    }
                )
            equal_value, equal_mask = equal_available_mixture(
                selected_experts, selected_expert_masks
            )
            equal_ic = rank_ic(
                equal_value,
                selected_target,
                selected_target_mask & equal_mask,
                groups,
            )
            rows.append(
                {
                    "task": task,
                    "scope": scope,
                    "horizon_days": int(horizon),
                    "candidate": "equal_available_expert_blend",
                    "rank_ic": equal_ic,
                    "baseline_rank_ic": baseline_ic,
                    "incremental_rank_ic": equal_ic - baseline_ic,
                    "dates": int(len(selected)),
                    "diagnostic_only": False,
                }
            )
            oracle_value, oracle_mask = oracle_mixture(
                selected_experts,
                selected_expert_masks,
                selected_target,
                selected_target_mask,
                groups,
            )
            oracle_ic = rank_ic(
                oracle_value,
                selected_target,
                oracle_mask,
                groups,
            )
            rows.append(
                {
                    "task": task,
                    "scope": scope,
                    "horizon_days": int(horizon),
                    "candidate": "same_date_hindsight_oracle_upper_bound",
                    "rank_ic": oracle_ic,
                    "baseline_rank_ic": baseline_ic,
                    "incremental_rank_ic": oracle_ic - baseline_ic,
                    "dates": int(len(selected)),
                    "diagnostic_only": True,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "config"
        / "hierarchical_asset_itransformer_v23_causal_mean_router.yaml",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    arrays, metadata = load_arrays(cfg)
    dates = pd.to_datetime(arrays.dates)
    horizons = [int(value) for value in metadata["horizons_days"]]
    rows = [
        *audit_task(
            "asset",
            arrays.asset_experts,
            arrays.asset_expert_masks,
            arrays.asset_target,
            arrays.asset_target_mask,
            list(metadata["asset_signal_experts"]),
            arrays.module_index,
            dates,
            horizons,
            pd.Timestamp(cfg["data"]["prediction_start"]),
        ),
        *audit_task(
            "sleeve",
            arrays.sleeve_experts,
            arrays.sleeve_expert_masks,
            arrays.sleeve_target,
            arrays.sleeve_target_mask,
            list(metadata["sleeve_signal_experts"]),
            None,
            dates,
            horizons,
            pd.Timestamp(cfg["data"]["prediction_start"]),
        ),
    ]
    report = pd.DataFrame(rows)
    output_path = ROOT / cfg["outputs"].get(
        "expert_library_audit",
        f"output/reports/{cfg['version']}_expert_library_audit.csv",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_path, index=False, encoding="utf-8-sig")
    pooled = report.loc[report["scope"].eq("pooled")].copy()
    print(pooled.to_string(index=False), flush=True)
    print(
        json.dumps(
            {
                "rows": len(report),
                "output": str(output_path.relative_to(ROOT)),
                "baseline_beaten_by_at_least_one_causal_expert": bool(
                    pooled.loc[
                        ~pooled["diagnostic_only"]
                        & ~pooled["candidate"].eq("fixed_causal_baseline"),
                        "incremental_rank_ic",
                    ].gt(0.0).any()
                ),
                "hindsight_oracle_is_diagnostic_only": True,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

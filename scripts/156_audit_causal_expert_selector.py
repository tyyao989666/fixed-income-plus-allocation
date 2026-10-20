from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.hierarchical_asset_itransformer import (  # noqa: E402
    _daily_expert_edges_by_group,
    load_arrays,
    load_config,
)


def _causal_selector_path(
    daily_edges: np.ndarray,
    daily_masks: np.ndarray,
    label_end: np.ndarray,
    dates: np.ndarray,
    halflife: float,
    minimum_observations: int,
    activation_threshold: float,
    availability_lag_days: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Select an expert using only edges whose return labels have matured."""
    length, horizon_count, group_count, expert_count = daily_edges.shape
    selected = np.zeros((length, horizon_count, group_count), dtype=np.int64)
    selected_edge = np.zeros((length, horizon_count, group_count), dtype=np.float32)
    date_ns = dates.astype(np.int64)
    missing_time = np.iinfo(np.int64).min
    alpha = 1.0 - np.exp(-np.log(2.0) / max(float(halflife), 1e-6))
    lag = max(int(availability_lag_days), 1)

    for horizon in range(horizon_count):
        events: list[list[int]] = [[] for _ in range(length)]
        for issue_position in range(length):
            maturity = int(label_end[issue_position, horizon])
            if maturity == missing_time:
                continue
            available_position = int(np.searchsorted(date_ns, maturity, side="left")) + lag
            if available_position < length:
                events[available_position].append(issue_position)

        reliability = np.zeros((group_count, expert_count), dtype=float)
        counts = np.zeros((group_count, expert_count), dtype=np.int64)
        for date_position in range(length):
            for issue_position in events[date_position]:
                valid = daily_masks[issue_position, horizon]
                values = daily_edges[issue_position, horizon]
                first = counts == 0
                reliability = np.where(
                    valid & first,
                    values,
                    np.where(valid, (1.0 - alpha) * reliability + alpha * values, reliability),
                )
                counts += valid.astype(np.int64)

            for group in range(group_count):
                candidate_scores = reliability[group, 1:].copy()
                candidate_counts = counts[group, 1:]
                candidate_scores[candidate_counts < minimum_observations] = -np.inf
                if not np.isfinite(candidate_scores).any():
                    continue
                candidate = int(np.argmax(candidate_scores)) + 1
                if reliability[group, candidate] <= activation_threshold:
                    continue
                if not daily_masks[date_position, horizon, group, candidate]:
                    continue
                selected[date_position, horizon, group] = candidate
                selected_edge[date_position, horizon, group] = daily_edges[
                    date_position, horizon, group, candidate
                ]
    return selected, selected_edge


def _summarize(
    task: str,
    dates: pd.DatetimeIndex,
    horizons: list[int],
    selected: np.ndarray,
    selected_edge: np.ndarray,
    expert_names: list[str],
    halflife: int,
    minimum_observations: int,
    threshold: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    scopes: list[tuple[str, np.ndarray]] = [
        (str(year), dates.year == year) for year in sorted(set(dates.year))
    ]
    scopes.append(("pooled", np.ones(len(dates), dtype=bool)))
    for scope, date_mask in scopes:
        for horizon_position, horizon in enumerate(horizons):
            choices = selected[date_mask, horizon_position].reshape(-1)
            edges = selected_edge[date_mask, horizon_position].reshape(-1)
            active = choices > 0
            row: dict[str, object] = {
                "task": task,
                "scope": scope,
                "horizon_days": int(horizon),
                "halflife_days": int(halflife),
                "minimum_observations": int(minimum_observations),
                "activation_threshold": float(threshold),
                "observations": int(len(edges)),
                "activation_fraction": float(active.mean()) if len(active) else np.nan,
                "mean_rank_ic_edge": float(edges.mean()) if len(edges) else np.nan,
                "positive_edge_fraction_when_active": (
                    float((edges[active] > 0.0).mean()) if active.any() else np.nan
                ),
            }
            for expert_position, name in enumerate(expert_names):
                row[f"selection_fraction__{name}"] = (
                    float((choices == expert_position).mean()) if len(choices) else np.nan
                )
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit whether matured historical expert skill can causally beat the fixed anchor."
    )
    parser.add_argument(
        "--config",
        default="config/hierarchical_asset_itransformer_v25_forward_skill_router.yaml",
    )
    parser.add_argument("--halflives", default="20,42,84")
    parser.add_argument("--thresholds", default="0.00,0.02,0.05,0.10,0.15")
    parser.add_argument("--minimum-observations", type=int, default=20)
    parser.add_argument("--availability-lag-days", type=int, default=1)
    parser.add_argument(
        "--output",
        default="output/reports/causal_expert_selector_diagnostic.csv",
    )
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    arrays, metadata = load_arrays(cfg)
    dates = pd.to_datetime(arrays.dates)
    horizons = [int(value) for value in metadata["horizons_days"]]
    asset_edges, asset_masks, _ = _daily_expert_edges_by_group(
        arrays.asset_experts,
        arrays.asset_expert_masks,
        arrays.asset_target,
        arrays.asset_target_mask,
        arrays.module_index,
    )
    sleeve_edges, sleeve_masks, _ = _daily_expert_edges_by_group(
        arrays.sleeve_experts,
        arrays.sleeve_expert_masks,
        arrays.sleeve_target,
        arrays.sleeve_target_mask,
        None,
    )

    halflives = [int(value) for value in args.halflives.split(",")]
    thresholds = [float(value) for value in args.thresholds.split(",")]
    rows: list[dict[str, object]] = []
    for halflife in halflives:
        for threshold in thresholds:
            asset_selected, asset_selected_edge = _causal_selector_path(
                asset_edges,
                asset_masks,
                arrays.label_end,
                arrays.dates,
                halflife,
                args.minimum_observations,
                threshold,
                args.availability_lag_days,
            )
            sleeve_selected, sleeve_selected_edge = _causal_selector_path(
                sleeve_edges,
                sleeve_masks,
                arrays.label_end,
                arrays.dates,
                halflife,
                args.minimum_observations,
                threshold,
                args.availability_lag_days,
            )
            rows.extend(
                _summarize(
                    "asset",
                    dates,
                    horizons,
                    asset_selected,
                    asset_selected_edge,
                    list(metadata["asset_signal_experts"]),
                    halflife,
                    args.minimum_observations,
                    threshold,
                )
            )
            rows.extend(
                _summarize(
                    "sleeve",
                    dates,
                    horizons,
                    sleeve_selected,
                    sleeve_selected_edge,
                    list(metadata["sleeve_signal_experts"]),
                    halflife,
                    args.minimum_observations,
                    threshold,
                )
            )

    output = pd.DataFrame(rows)
    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False, encoding="utf-8-sig")
    focus = output.loc[
        output["scope"].isin(["2022", "2023", "2024", "2025", "2026", "pooled"]),
        [
            "task",
            "scope",
            "horizon_days",
            "halflife_days",
            "activation_threshold",
            "activation_fraction",
            "mean_rank_ic_edge",
            "positive_edge_fraction_when_active",
        ],
    ]
    print(focus.to_string(index=False))
    print(
        json.dumps(
            {
                "output": str(output_path.relative_to(ROOT)),
                "rows": int(len(output)),
                "point_in_time_rule": (
                    "An expert edge becomes selectable only after its complete future-return "
                    "label matures plus the configured trading-day lag."
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

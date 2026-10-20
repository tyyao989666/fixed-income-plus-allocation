from __future__ import annotations

"""Recalibrate saved strict-OOS forecasts without retraining the network."""

import argparse
import importlib.util
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
    RISK_SLEEVES,
    load_arrays,
    load_config,
    project_path,
)


def load_trainer():
    path = ROOT / "scripts" / "135_train_hierarchical_asset_itransformer_v7.py"
    spec = importlib.util.spec_from_file_location(
        "hierarchical_itransformer_trainer",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import trainer from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare_long_frame(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["as_of_date"] = pd.to_datetime(
        output["as_of_date"], errors="coerce"
    )
    output["horizon_label_end_date"] = pd.to_datetime(
        output["horizon_label_end_date"], errors="coerce"
    )
    output["horizon_days"] = pd.to_numeric(
        output["horizon_days"], errors="coerce"
    ).astype("Int64")
    output["has_label"] = output["has_label"].fillna(False).astype(bool)
    return output


def long_frames_to_bundle(
    asset_frame: pd.DataFrame,
    sleeve_frame: pd.DataFrame,
    arrays,
    metadata: dict[str, Any],
) -> dict[str, np.ndarray]:
    dates = sorted(
        set(asset_frame["as_of_date"].dropna()).union(
            sleeve_frame["as_of_date"].dropna()
        )
    )
    if not dates:
        raise ValueError("No prediction dates are available for recalibration")

    source_date_to_index = {
        pd.Timestamp(date): position
        for position, date in enumerate(pd.to_datetime(arrays.dates))
    }
    missing_dates = [
        pd.Timestamp(date)
        for date in dates
        if pd.Timestamp(date) not in source_date_to_index
    ]
    if missing_dates:
        raise ValueError(
            "Prediction dates are absent from the source dataset: "
            f"{missing_dates[:5]}"
        )

    assets = list(metadata["assets"])
    horizons = [int(value) for value in metadata["horizons_days"]]
    date_position = {
        pd.Timestamp(date): position for position, date in enumerate(dates)
    }
    asset_position = {asset: position for position, asset in enumerate(assets)}
    sleeve_position = {
        sleeve: position for position, sleeve in enumerate(RISK_SLEEVES)
    }
    horizon_position = {
        horizon: position for position, horizon in enumerate(horizons)
    }
    rows = len(dates)
    horizon_count = len(horizons)
    asset_count = len(assets)
    sleeve_count = len(RISK_SLEEVES)
    bundle: dict[str, np.ndarray] = {
        "index": np.asarray(
            [source_date_to_index[pd.Timestamp(date)] for date in dates],
            dtype=np.int64,
        ),
        "asset_baseline": np.zeros(
            (rows, horizon_count, asset_count), dtype=np.float32
        ),
        "asset_residual": np.zeros(
            (rows, horizon_count, asset_count), dtype=np.float32
        ),
        "asset_target": np.zeros(
            (rows, horizon_count, asset_count), dtype=np.float32
        ),
        "asset_mask": np.zeros(
            (rows, horizon_count, asset_count), dtype=bool
        ),
        "sleeve_baseline": np.zeros(
            (rows, horizon_count, sleeve_count), dtype=np.float32
        ),
        "sleeve_residual": np.zeros(
            (rows, horizon_count, sleeve_count), dtype=np.float32
        ),
        "sleeve_target": np.zeros(
            (rows, horizon_count, sleeve_count), dtype=np.float32
        ),
        "sleeve_mask": np.zeros(
            (rows, horizon_count, sleeve_count), dtype=bool
        ),
    }

    for row in asset_frame.itertuples(index=False):
        date = pd.Timestamp(row.as_of_date)
        horizon = int(row.horizon_days)
        if (
            date not in date_position
            or horizon not in horizon_position
            or row.asset_code not in asset_position
        ):
            continue
        i = date_position[date]
        j = horizon_position[horizon]
        k = asset_position[row.asset_code]
        bundle["asset_baseline"][i, j, k] = float(
            row.baseline_relative_score
        )
        bundle["asset_residual"][i, j, k] = float(
            row.model_residual_score
        )
        valid = bool(row.has_label) and pd.notna(
            row.realized_relative_score
        )
        if valid:
            bundle["asset_target"][i, j, k] = float(
                row.realized_relative_score
            )
            bundle["asset_mask"][i, j, k] = True

    for row in sleeve_frame.itertuples(index=False):
        date = pd.Timestamp(row.as_of_date)
        horizon = int(row.horizon_days)
        if (
            date not in date_position
            or horizon not in horizon_position
            or row.sleeve not in sleeve_position
        ):
            continue
        i = date_position[date]
        j = horizon_position[horizon]
        k = sleeve_position[row.sleeve]
        bundle["sleeve_baseline"][i, j, k] = float(
            row.baseline_relative_score
        )
        bundle["sleeve_residual"][i, j, k] = float(
            row.model_residual_score
        )
        valid = bool(row.has_label) and pd.notna(
            row.realized_relative_score
        )
        if valid:
            bundle["sleeve_target"][i, j, k] = float(
                row.realized_relative_score
            )
            bundle["sleeve_mask"][i, j, k] = True
    return bundle


def fallback_policies(
    cfg: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    baseline_weight = float(
        cfg["model"].get(
            "deployment_fallback_baseline_weight",
            (
                0.0
                if str(
                    cfg["data"].get(
                        "forecast_baseline_mode",
                        "causal_anchor",
                    )
                )
                == "zero_active_view"
                else 1.0
            ),
        )
    )

    def rows(task: str) -> list[dict[str, Any]]:
        return [
            {
                "task": task,
                "horizon_index": float(position),
                "baseline_weight": baseline_weight,
                "residual_alpha": 0.0,
                "selection_metric": str(
                    cfg["model"].get(
                        "blend_selection_metric",
                        "rank_ic",
                    )
                ),
                "selection_score": 0.0,
                "bootstrap_positive_probability": 0.5,
            }
            for position, _ in enumerate(metadata["horizons_days"])
        ]

    return {
        "asset": rows("asset"),
        "sleeve": rows("sleeve"),
        "asset_by_module": {
            module: rows("asset") for module in RISK_SLEEVES
        },
    }


def apply_policies(
    frame: pd.DataFrame,
    policies: dict[str, Any],
    metadata: dict[str, Any],
    task: str,
    trainer,
) -> pd.DataFrame:
    output = frame.copy()
    horizon_position = {
        int(horizon): position
        for position, horizon in enumerate(metadata["horizons_days"])
    }
    baseline_weights: list[float] = []
    residual_alphas: list[float] = []
    for row in output.itertuples(index=False):
        position = horizon_position[int(row.horizon_days)]
        if task == "asset":
            policy = trainer.asset_policy_for(
                policies,
                position,
                str(row.module),
            )
        else:
            policy = policies["sleeve"][position]
        baseline_weights.append(float(policy["baseline_weight"]))
        residual_alphas.append(float(policy["residual_alpha"]))
    output["baseline_weight"] = baseline_weights
    output["residual_alpha"] = residual_alphas
    score = (
        output["baseline_weight"]
        * pd.to_numeric(
            output["baseline_relative_score"], errors="coerce"
        )
        + output["residual_alpha"]
        * pd.to_numeric(
            output["model_residual_score"], errors="coerce"
        )
    )
    output["walkforward_relative_score"] = score
    output["historical_oos_relative_score"] = score
    if task == "sleeve":
        for column in (
            "pred_relative_score",
            "adaptive_relative_score",
            "historical_oos_adaptive_score",
            "combined_oos_adaptive_score",
        ):
            if column in output:
                output[column] = score
        if "adaptive_model_weight" in output:
            output["adaptive_model_weight"] = output["residual_alpha"]
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-config",
        type=Path,
        default=Path(
            "config/hierarchical_asset_itransformer_v31_fold_scaled_mean.yaml"
        ),
    )
    parser.add_argument(
        "--target-config",
        type=Path,
        default=Path(
            "config/hierarchical_asset_itransformer_v32_rank_gated_mean.yaml"
        ),
    )
    args = parser.parse_args()

    source_cfg = load_config(args.source_config)
    target_cfg = load_config(args.target_config)
    arrays, metadata = load_arrays(target_cfg)
    trainer = load_trainer()

    asset_source = prepare_long_frame(
        pd.read_parquet(project_path(source_cfg, "asset_predictions"))
    )
    sleeve_source = prepare_long_frame(
        pd.read_parquet(project_path(source_cfg, "sleeve_predictions"))
    )
    years = sorted(
        set(asset_source["as_of_date"].dt.year.dropna().astype(int))
        & set(sleeve_source["as_of_date"].dt.year.dropna().astype(int))
    )
    if not years:
        raise ValueError("Source predictions contain no common years")

    asset_output: list[pd.DataFrame] = []
    sleeve_output: list[pd.DataFrame] = []
    validation_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    minimum_years = int(
        target_cfg["model"].get("blend_min_calibration_years", 2)
    )
    for year in years:
        cutoff = pd.Timestamp(year, 1, 1)
        asset_current = asset_source.loc[
            asset_source["as_of_date"].dt.year.eq(year)
        ].copy()
        sleeve_current = sleeve_source.loc[
            sleeve_source["as_of_date"].dt.year.eq(year)
        ].copy()
        current_bundle = long_frames_to_bundle(
            asset_current,
            sleeve_current,
            arrays,
            metadata,
        )
        asset_calibration = asset_source.loc[
            asset_source["as_of_date"].lt(cutoff)
            & asset_source["horizon_label_end_date"].lt(cutoff)
            & asset_source["has_label"]
        ].copy()
        sleeve_calibration = sleeve_source.loc[
            sleeve_source["as_of_date"].lt(cutoff)
            & sleeve_source["horizon_label_end_date"].lt(cutoff)
            & sleeve_source["has_label"]
        ].copy()
        calibration_years = sorted(
            set(asset_calibration["as_of_date"].dt.year.astype(int))
            & set(sleeve_calibration["as_of_date"].dt.year.astype(int))
        )
        if len(calibration_years) >= minimum_years:
            calibration_bundle = long_frames_to_bundle(
                asset_calibration,
                sleeve_calibration,
                arrays,
                metadata,
            )
            policies, rows = trainer.choose_blend_policies(
                calibration_bundle,
                [
                    float(value)
                    for value in target_cfg["model"][
                        "baseline_weight_grid"
                    ]
                ],
                [
                    float(value)
                    for value in target_cfg["model"][
                        "residual_alpha_grid"
                    ]
                ],
                arrays.module_index,
                arrays,
                target_cfg,
                require_incremental_confidence=bool(
                    target_cfg["model"].get(
                        "blend_bootstrap_enabled",
                        False,
                    )
                ),
            )
        else:
            policies = fallback_policies(target_cfg, metadata)
            rows = []
        for row in rows:
            calibration_rows.append(
                {
                    "prediction_year": year,
                    "calibration_years": ",".join(
                        str(value) for value in calibration_years
                    ),
                    **row,
                }
            )

        asset_year = apply_policies(
            asset_current,
            policies,
            metadata,
            "asset",
            trainer,
        )
        sleeve_year = apply_policies(
            sleeve_current,
            policies,
            metadata,
            "sleeve",
            trainer,
        )
        asset_output.append(asset_year)
        sleeve_output.append(sleeve_year)
        validation_rows.extend(
            trainer.metric_rows(
                current_bundle,
                metadata,
                arrays.module_index,
                policies,
                year,
                target_cfg,
            )
        )
        print(
            json.dumps(
                {
                    "year": year,
                    "calibration_years": calibration_years,
                    "policies": trainer.compact_blend_policy(
                        policies,
                        [
                            int(value)
                            for value in metadata["horizons_days"]
                        ],
                    ),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    asset_predictions = pd.concat(
        asset_output, ignore_index=True
    ).sort_values(
        ["as_of_date", "horizon_days", "module", "asset_code"]
    )
    sleeve_predictions = pd.concat(
        sleeve_output, ignore_index=True
    ).sort_values(["as_of_date", "horizon_days", "sleeve"])
    validation = pd.DataFrame(validation_rows)

    asset_predictions.to_parquet(
        project_path(target_cfg, "asset_predictions"), index=False
    )
    sleeve_predictions.to_parquet(
        project_path(target_cfg, "sleeve_predictions"), index=False
    )
    validation.to_csv(
        project_path(target_cfg, "validation_report"),
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(calibration_rows).to_csv(
        project_path(target_cfg, "training_log"),
        index=False,
        encoding="utf-8-sig",
    )
    gate = trainer.build_component_policy_gate(
        validation,
        asset_predictions,
        sleeve_predictions,
        target_cfg,
    )
    gate["recalibration_source"] = str(args.source_config)
    gate["network_retrained"] = False
    project_path(target_cfg, "gate_report").write_text(
        json.dumps(gate, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    parameter_count = sum(
        parameter.numel()
        for parameter in trainer.make_model(
            arrays, metadata, target_cfg
        ).parameters()
    )
    trainer.write_report(
        target_cfg,
        metadata,
        validation,
        gate,
        parameter_count,
    )
    print(
        json.dumps(
            {
                "asset_predictions": str(
                    project_path(target_cfg, "asset_predictions")
                ),
                "sleeve_predictions": str(
                    project_path(target_cfg, "sleeve_predictions")
                ),
                "gate": gate,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

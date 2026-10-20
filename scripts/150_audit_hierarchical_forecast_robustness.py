from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.hierarchical_asset_itransformer import load_config, project_path  # noqa: E402


def _row_spearman(score: pd.DataFrame, target: pd.DataFrame) -> pd.Series:
    valid = score.notna() & target.notna()
    score_rank = score.where(valid).rank(axis=1, method="average")
    target_rank = target.where(valid).rank(axis=1, method="average")
    score_centered = score_rank.sub(score_rank.mean(axis=1), axis=0)
    target_centered = target_rank.sub(target_rank.mean(axis=1), axis=0)
    denominator = np.sqrt(
        score_centered.pow(2).sum(axis=1)
        * target_centered.pow(2).sum(axis=1)
    )
    values = (
        (score_centered * target_centered).sum(axis=1)
        / denominator.replace(0.0, np.nan)
    )
    values.loc[valid.sum(axis=1).lt(2)] = np.nan
    return values


def _daily_rank_ic(
    frame: pd.DataFrame,
    item_column: str,
    score_column: str,
    subgroup_column: str | None,
) -> pd.Series:
    groups = (
        [(None, frame)]
        if subgroup_column is None
        else list(frame.groupby(subgroup_column, sort=False))
    )
    pieces: list[pd.DataFrame] = []
    for subgroup, group in groups:
        score = group.pivot(
            index="as_of_date", columns=item_column, values=score_column
        )
        target = (
            group.pivot(
                index="as_of_date",
                columns=item_column,
                values="realized_relative_score",
            )
            .reindex_like(score)
        )
        values = _row_spearman(score, target)
        pieces.append(
            pd.DataFrame(
                {
                    "as_of_date": values.index,
                    "subgroup": "all" if subgroup is None else str(subgroup),
                    "rank_ic": values.to_numpy(dtype=float),
                }
            )
        )
    detail = pd.concat(pieces, ignore_index=True).dropna(subset=["rank_ic"])
    # Preserve the trainer's equal module/date weighting for the point estimate,
    # but aggregate modules on the same date before time-series resampling.
    return detail.groupby("as_of_date", sort=True)["rank_ic"].mean()


def _masked_softmax(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    selected = np.where(valid, values, -np.inf)
    maximum = float(np.max(selected))
    if not np.isfinite(maximum):
        return np.zeros_like(values, dtype=float)
    exponent = np.where(valid, np.exp(selected - maximum), 0.0)
    return exponent / max(float(exponent.sum()), 1e-12)


def _daily_decision_utility(
    frame: pd.DataFrame,
    score_column: str,
    subgroup_column: str | None,
    temperature: float,
    transaction_cost_bps: float,
    horizon_days: int,
) -> pd.Series:
    """Cost-adjusted active mean utility relative to the causal baseline."""
    tau = max(float(temperature), 1e-4)
    cost_rate = (
        float(transaction_cost_bps)
        * 1e-4
        * 252.0
        / max(float(horizon_days), 1.0)
    )
    output: dict[pd.Timestamp, float] = {}
    for as_of_date, day in frame.groupby("as_of_date", sort=True):
        subgroups = (
            [(None, day)]
            if subgroup_column is None
            else list(day.groupby(subgroup_column, sort=False))
        )
        values: list[float] = []
        for _, group in subgroups:
            prediction = pd.to_numeric(
                group[score_column], errors="coerce"
            ).to_numpy(dtype=float)
            baseline = pd.to_numeric(
                group["baseline_relative_score"], errors="coerce"
            ).to_numpy(dtype=float)
            target = pd.to_numeric(
                group["realized_relative_score"], errors="coerce"
            ).to_numpy(dtype=float)
            valid = (
                np.isfinite(prediction)
                & np.isfinite(baseline)
                & np.isfinite(target)
            )
            if valid.sum() < 2:
                continue
            predicted_weight = _masked_softmax(prediction / tau, valid)
            baseline_weight = _masked_softmax(baseline / tau, valid)
            active = predicted_weight - baseline_weight
            gross = float(np.sum(active[valid] * target[valid]))
            turnover = 0.5 * float(np.abs(active[valid]).sum())
            values.append(gross - cost_rate * turnover)
        if values:
            output[pd.Timestamp(as_of_date)] = float(np.mean(values))
    return pd.Series(output, dtype=float).sort_index()


def _paired_circular_block_bootstrap(
    delta: pd.Series,
    block_days: int,
    draws: int,
    seed: int,
) -> dict[str, float | int]:
    values = pd.to_numeric(delta, errors="coerce").dropna().to_numpy(dtype=float)
    n = len(values)
    if n < max(20, block_days * 2):
        return {
            "bootstrap_samples": n,
            "block_days": block_days,
            "bootstrap_draws": 0,
            "incremental_ci_low": np.nan,
            "incremental_ci_high": np.nan,
            "probability_incremental_positive": np.nan,
        }
    rng = np.random.default_rng(seed)
    blocks_needed = int(np.ceil(n / block_days))
    offsets = np.arange(block_days, dtype=np.int64)
    estimates = np.empty(draws, dtype=float)
    for draw in range(draws):
        starts = rng.integers(0, n, size=blocks_needed)
        indices = (starts[:, None] + offsets[None, :]) % n
        estimates[draw] = float(values[indices.reshape(-1)[:n]].mean())
    return {
        "bootstrap_samples": n,
        "block_days": block_days,
        "bootstrap_draws": draws,
        "incremental_ci_low": float(np.quantile(estimates, 0.025)),
        "incremental_ci_high": float(np.quantile(estimates, 0.975)),
        "probability_incremental_positive": float(np.mean(estimates > 0.0)),
    }


def _load_predictions(cfg: dict[str, Any], output_key: str) -> pd.DataFrame:
    path = project_path(cfg, output_key)
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path)
    frame["as_of_date"] = pd.to_datetime(frame["as_of_date"], errors="coerce")
    frame["horizon_label_end_date"] = pd.to_datetime(
        frame["horizon_label_end_date"], errors="coerce"
    )
    frame["has_label"] = frame["has_label"].fillna(False).astype(bool)
    frame["year"] = frame["as_of_date"].dt.year.astype("Int64")
    return frame


def _structural_checks(
    asset: pd.DataFrame,
    sleeve: pd.DataFrame,
    minimum_calibration_years: int,
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    definitions = {
        "asset": (
            asset,
            ["as_of_date", "horizon_days", "module", "asset_code"],
        ),
        "sleeve": (
            sleeve,
            ["as_of_date", "horizon_days", "sleeve"],
        ),
    }
    for task, (frame, keys) in definitions.items():
        labelled = frame.loc[frame["has_label"]].copy()
        finite_columns = [
            "baseline_relative_score",
            "model_raw_score",
            "model_residual_score",
            "walkforward_relative_score",
            "realized_relative_score",
            "prediction_uncertainty",
        ]
        finite = np.isfinite(labelled[finite_columns].to_numpy(dtype=float)).all()
        raw_identity_error = (
            labelled["model_raw_score"]
            - labelled["baseline_relative_score"]
            - labelled["model_residual_score"]
        ).abs()
        checks.extend(
            [
                {
                    "check": f"{task}_unique_prediction_keys",
                    "passed": not bool(frame.duplicated(keys).any()),
                    "detail": f"rows={len(frame)}",
                },
                {
                    "check": f"{task}_label_starts_after_as_of",
                    "passed": bool(
                        (
                            labelled["horizon_label_end_date"]
                            > labelled["as_of_date"]
                        ).all()
                    ),
                    "detail": f"labelled_rows={len(labelled)}",
                },
                {
                    "check": f"{task}_finite_labelled_values",
                    "passed": bool(finite),
                    "detail": ", ".join(finite_columns),
                },
                {
                    "check": f"{task}_raw_score_identity",
                    "passed": bool(raw_identity_error.le(1e-6).all()),
                    "detail": (
                        "model_raw_score = baseline_relative_score + "
                        "model_residual_score; "
                        f"max_error={float(raw_identity_error.max()):.3e}"
                    ),
                },
                {
                    "check": f"{task}_domestic_weekday_calendar",
                    "passed": not bool(frame["as_of_date"].dt.dayofweek.ge(5).any()),
                    "detail": (
                        f"{frame['as_of_date'].min().date()} to "
                        f"{frame['as_of_date'].max().date()}"
                    ),
                },
                {
                    "check": f"{task}_prediction_year_identity",
                    "passed": bool(
                        frame["training_cutoff_year"]
                        .astype("Int64")
                        .eq(frame["year"])
                        .all()
                    ),
                    "detail": "training_cutoff_year must equal issued forecast year",
                },
            ]
        )
        years = sorted(frame["year"].dropna().astype(int).unique().tolist())
        fallback_years = years[:minimum_calibration_years]
        fallback = frame.loc[frame["year"].isin(fallback_years)]
        checks.append(
            {
                "check": f"{task}_strict_initial_oos_fallback",
                "passed": bool(fallback["residual_alpha"].abs().le(1e-12).all()),
                "detail": (
                    f"minimum_calibration_years={minimum_calibration_years}; "
                    f"fallback_years={fallback_years}"
                ),
            }
        )
    return checks


def _metric_rows(
    task: str,
    frame: pd.DataFrame,
    item_column: str,
    subgroup_column: str | None,
    alpha_grid: list[float],
    draws: int,
    seed: int,
    cfg: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metric_rows: list[dict[str, Any]] = []
    alpha_rows: list[dict[str, Any]] = []
    scopes = [
        (str(year), frame.loc[frame["year"].eq(year)].copy())
        for year in sorted(frame["year"].dropna().astype(int).unique())
    ]
    scopes.append(("pooled", frame.copy()))
    for scope, scoped in scopes:
        for horizon in sorted(scoped["horizon_days"].unique()):
            selected = scoped.loc[
                scoped["horizon_days"].eq(horizon) & scoped["has_label"]
            ].copy()
            baseline = _daily_rank_ic(
                selected,
                item_column,
                "baseline_relative_score",
                subgroup_column,
            )
            deployed = _daily_rank_ic(
                selected,
                item_column,
                "walkforward_relative_score",
                subgroup_column,
            )
            raw_model = _daily_rank_ic(
                selected,
                item_column,
                "model_raw_score",
                subgroup_column,
            )
            model_cfg = cfg["model"]
            configured_horizons = [
                int(value) for value in cfg["data"]["horizons_days"]
            ]
            horizon_position = configured_horizons.index(int(horizon))

            def horizon_value(raw: Any, default: float) -> float:
                if isinstance(raw, (list, tuple)):
                    return float(raw[horizon_position])
                return float(default if raw is None else raw)

            temperature = horizon_value(
                model_cfg.get(f"{task}_decision_temperature"), 0.10
            )
            transaction_cost_bps = horizon_value(
                model_cfg.get("decision_transaction_cost_bps"), 5.0
            )
            raw_utility = _daily_decision_utility(
                selected,
                "model_raw_score",
                subgroup_column,
                temperature,
                transaction_cost_bps,
                int(horizon),
            )
            deployed_utility = _daily_decision_utility(
                selected,
                "walkforward_relative_score",
                subgroup_column,
                temperature,
                transaction_cost_bps,
                int(horizon),
            )
            common = baseline.index.intersection(deployed.index)
            delta = deployed.reindex(common) - baseline.reindex(common)
            bootstrap = _paired_circular_block_bootstrap(
                delta,
                block_days=max(int(horizon), 5),
                draws=draws,
                seed=seed + int(horizon) + (0 if scope == "pooled" else int(scope)),
            )
            utility_bootstrap = _paired_circular_block_bootstrap(
                deployed_utility,
                block_days=max(int(horizon), 5),
                draws=draws,
                seed=(
                    seed
                    + 1_000_000
                    + int(horizon)
                    + (0 if scope == "pooled" else int(scope))
                ),
            )
            metric_rows.append(
                {
                    "task": task,
                    "scope": scope,
                    "horizon_days": int(horizon),
                    "baseline_rank_ic": float(baseline.mean()),
                    "raw_model_rank_ic": float(raw_model.mean()),
                    "deployed_rank_ic": float(deployed.mean()),
                    "deployed_incremental_rank_ic": float(delta.mean()),
                    "positive_date_fraction": float((delta > 0.0).mean()),
                    "raw_model_decision_utility": float(raw_utility.mean()),
                    "deployed_decision_utility": float(
                        deployed_utility.mean()
                    ),
                    "deployed_utility_positive_date_fraction": float(
                        (deployed_utility > 0.0).mean()
                    ),
                    "deployed_utility_ci_low": utility_bootstrap[
                        "incremental_ci_low"
                    ],
                    "deployed_utility_ci_high": utility_bootstrap[
                        "incremental_ci_high"
                    ],
                    "probability_deployed_utility_positive": utility_bootstrap[
                        "probability_incremental_positive"
                    ],
                    "mean_residual_alpha": float(
                        selected["residual_alpha"].mean()
                    ),
                    **bootstrap,
                }
            )
            for alpha in alpha_grid:
                score_column = "_diagnostic_score"
                selected[score_column] = (
                    selected["baseline_relative_score"]
                    + float(alpha) * selected["model_residual_score"]
                )
                candidate = _daily_rank_ic(
                    selected,
                    item_column,
                    score_column,
                    subgroup_column,
                )
                common = baseline.index.intersection(candidate.index)
                candidate_delta = candidate.reindex(common) - baseline.reindex(common)
                candidate_bootstrap = _paired_circular_block_bootstrap(
                    candidate_delta,
                    block_days=max(int(horizon), 5),
                    draws=draws,
                    seed=(
                        seed
                        + 10_000
                        + int(horizon)
                        + int(round(float(alpha) * 10_000))
                        + (0 if scope == "pooled" else int(scope))
                    ),
                )
                candidate_utility = _daily_decision_utility(
                    selected,
                    score_column,
                    subgroup_column,
                    temperature,
                    transaction_cost_bps,
                    int(horizon),
                )
                candidate_utility_bootstrap = (
                    _paired_circular_block_bootstrap(
                        candidate_utility,
                        block_days=max(int(horizon), 5),
                        draws=draws,
                        seed=(
                            seed
                            + 2_000_000
                            + int(horizon)
                            + int(round(float(alpha) * 10_000))
                            + (
                                0
                                if scope == "pooled"
                                else int(scope)
                            )
                        ),
                    )
                )
                alpha_rows.append(
                    {
                        "task": task,
                        "scope": scope,
                        "horizon_days": int(horizon),
                        "diagnostic_alpha": float(alpha),
                        "rank_ic": float(candidate.mean()),
                        "incremental_rank_ic": float(candidate_delta.mean()),
                        "positive_date_fraction": float(
                            (candidate_delta > 0.0).mean()
                        ),
                        "decision_utility": float(
                            candidate_utility.mean()
                        ),
                        "utility_positive_date_fraction": float(
                            (candidate_utility > 0.0).mean()
                        ),
                        "utility_ci_low": candidate_utility_bootstrap[
                            "incremental_ci_low"
                        ],
                        "utility_ci_high": candidate_utility_bootstrap[
                            "incremental_ci_high"
                        ],
                        "probability_utility_positive": (
                            candidate_utility_bootstrap[
                                "probability_incremental_positive"
                            ]
                        ),
                        **candidate_bootstrap,
                    }
                )
    return metric_rows, alpha_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit annual walk-forward iTransformer forecasts with paired "
            "moving-block bootstrap confidence intervals."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "config"
        / "hierarchical_asset_itransformer_v21_continuous_utility.yaml",
    )
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument(
        "--alpha-grid",
        default="0,0.01,0.025,0.05,0.075,0.10,0.15,0.25,0.50,0.75,1.0",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    alpha_grid = [float(value) for value in args.alpha_grid.split(",")]
    asset = _load_predictions(cfg, "asset_predictions")
    sleeve = _load_predictions(cfg, "sleeve_predictions")
    checks = _structural_checks(
        asset,
        sleeve,
        int(cfg["model"].get("blend_min_calibration_years", 2)),
    )
    metric_rows: list[dict[str, Any]] = []
    alpha_rows: list[dict[str, Any]] = []
    for definition in [
        ("asset", asset, "asset_code", "module"),
        ("sleeve", sleeve, "sleeve", None),
    ]:
        metrics, curves = _metric_rows(
            *definition,
            alpha_grid=alpha_grid,
            draws=max(int(args.bootstrap_draws), 0),
            seed=int(args.seed),
            cfg=cfg,
        )
        metric_rows.extend(metrics)
        alpha_rows.extend(curves)

    stem = str(cfg["version"])
    report_dir = ROOT / "output" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    checks_path = report_dir / f"{stem}_forecast_causality_checks.csv"
    metrics_path = report_dir / f"{stem}_forecast_robustness.csv"
    curves_path = report_dir / f"{stem}_diagnostic_alpha_curves.csv"
    gate_path = report_dir / f"{stem}_forecast_robustness_gate.json"
    checks_frame = pd.DataFrame(checks)
    metrics_frame = pd.DataFrame(metric_rows)
    curves_frame = pd.DataFrame(alpha_rows)
    checks_frame.to_csv(checks_path, index=False, encoding="utf-8-sig")
    metrics_frame.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    curves_frame.to_csv(curves_path, index=False, encoding="utf-8-sig")

    pooled = metrics_frame.loc[metrics_frame["scope"].eq("pooled")]
    enabled_rows = []
    for task in ["asset", "sleeve"]:
        enabled = cfg["model"].get(f"{task}_enabled_horizons_days")
        task_rows = pooled.loc[pooled["task"].eq(task)]
        if enabled is not None:
            task_rows = task_rows.loc[
                task_rows["horizon_days"].isin(
                    [int(value) for value in enabled]
                )
            ]
        enabled_rows.append(task_rows)
    pooled_enabled = pd.concat(enabled_rows, ignore_index=True)
    active_pooled = pooled_enabled.loc[
        pooled_enabled["mean_residual_alpha"].abs().gt(1e-12)
    ]
    decision_gate = str(
        cfg["model"].get("blend_selection_metric", "rank_ic")
    ) == "decision_aligned_utility"
    gate = {
        "structural_checks_passed": bool(checks_frame["passed"].all()),
        "structural_checks": {
            row["check"]: bool(row["passed"]) for row in checks
        },
        "forecast_is_active": bool(
            active_pooled.shape[0] > 0
        ),
        "all_active_pooled_edges_nonnegative": bool(
            (
                active_pooled[
                    "deployed_decision_utility"
                    if decision_gate
                    else "deployed_incremental_rank_ic"
                ]
                >= 0.0
            ).all()
        ),
        "promotion_metric": (
            "cost_adjusted_decision_utility"
            if decision_gate
            else "incremental_rank_ic"
        ),
        "caveat": (
            "Diagnostic alpha curves are post-hoc research diagnostics and "
            "must never be used to tune the same forecast year."
        ),
        "outputs": {
            "checks": str(checks_path.relative_to(ROOT)),
            "metrics": str(metrics_path.relative_to(ROOT)),
            "diagnostic_alpha_curves": str(curves_path.relative_to(ROOT)),
        },
    }
    gate_path.write_text(
        json.dumps(gate, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(checks_frame.to_string(index=False))
    print()
    print(metrics_frame.to_string(index=False))
    print()
    print(json.dumps(gate, ensure_ascii=False, indent=2))
    if not gate["structural_checks_passed"]:
        raise SystemExit("Forecast causality audit failed")


if __name__ == "__main__":
    main()

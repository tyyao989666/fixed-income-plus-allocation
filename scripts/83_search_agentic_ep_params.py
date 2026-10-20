from __future__ import annotations

import copy
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_v1_full_pipeline import profile_cfg  # noqa: E402
from src.v1_strategy import (  # noqa: E402
    V1Paths,
    build_asset_weights,
    build_risk_budget_base,
    build_sleeve_weights,
    run_backtest,
)


REPORTS = ROOT / "output" / "reports"
PROCESSED = ROOT / "data_processed"


def candidate_grid() -> list[dict]:
    return [
        {
            "name": "agentic_base",
            "ep": {},
            "view_scale": {},
        },
        {
            "name": "view_scale_006_dev025",
            "ep": {"view_mu_scale": 0.006, "max_active_deviation": 0.025, "max_abs_mu_tilt": 0.012},
            "view_scale": {"low_vol": 0.60, "balanced": 0.60, "high_vol": 0.40},
        },
        {
            "name": "defensive_downweight_006",
            "ep": {
                "view_mu_scale": 0.006,
                "max_active_deviation": 0.025,
                "max_abs_mu_tilt": 0.012,
                "view_pair_scales": {
                    "alternatives>equity": 0.25,
                    "rates>convertible_bonds": 0.35,
                    "rates>equity": 0.75,
                },
            },
            "view_scale": {"low_vol": 0.50, "balanced": 0.55, "high_vol": 0.35},
        },
        {
            "name": "defensive_downweight_008",
            "ep": {
                "view_mu_scale": 0.008,
                "max_active_deviation": 0.030,
                "max_abs_mu_tilt": 0.014,
                "view_pair_scales": {
                    "alternatives>equity": 0.15,
                    "rates>convertible_bonds": 0.25,
                    "rates>equity": 0.65,
                    "convertible_bonds>credit": 1.20,
                },
            },
            "view_scale": {"low_vol": 0.50, "balanced": 0.60, "high_vol": 0.40},
        },
        {
            "name": "view_scale_008_dev030",
            "ep": {"view_mu_scale": 0.008, "max_active_deviation": 0.030, "max_abs_mu_tilt": 0.014},
            "view_scale": {"low_vol": 0.65, "balanced": 0.65, "high_vol": 0.45},
        },
        {
            "name": "view_scale_010_dev035",
            "ep": {"view_mu_scale": 0.010, "max_active_deviation": 0.035, "max_abs_mu_tilt": 0.016},
            "view_scale": {"low_vol": 0.70, "balanced": 0.70, "high_vol": 0.50},
        },
        {
            "name": "view_scale_012_dev040_looser",
            "ep": {
                "view_mu_scale": 0.012,
                "max_active_deviation": 0.040,
                "max_abs_mu_tilt": 0.018,
                "base_deviation_penalty": 7.0,
                "turnover_penalty": 0.35,
            },
            "view_scale": {"low_vol": 0.75, "balanced": 0.75, "high_vol": 0.55},
        },
    ]


def run_one(profile: str, candidate: dict) -> dict:
    cfg = copy.deepcopy(profile_cfg(profile, "agentic"))
    suffix = f"search_agentic_{candidate['name']}_{profile}"
    cfg["outputs"] = {
        "risk_budget_base": f"data_processed/{suffix}_risk_budget_base_monthly.parquet",
        "sleeve_weights": f"data_processed/{suffix}_sleeve_weights_monthly.parquet",
        "weights": f"data_processed/{suffix}_target_weights_monthly.parquet",
        "nav": f"output/reports/{suffix}_backtest_nav.csv",
        "metrics": f"output/reports/{suffix}_backtest_metrics.csv",
        "attribution": f"output/reports/{suffix}_attribution_by_sleeve.csv",
        "robustness": f"output/reports/{suffix}_robustness_summary.csv",
        "data_quality": f"output/reports/{suffix}_data_quality.csv",
        "ep_posterior": f"data_processed/{suffix}_ep_posterior_monthly.parquet",
        "ep_signals": f"output/reports/{suffix}_ep_signals.csv",
        "optimizer_diagnostics": f"output/reports/{suffix}_optimizer_diagnostics.csv",
    }
    cfg.setdefault("strategy_version_overrides", {})["agentic"] = {
        "ep_skfolio": candidate["ep"],
        "view_tilt_profile_scale": candidate["view_scale"],
    }
    paths = V1Paths(views=PROCESSED / "ep_views_agentic_daily_v1.parquet")
    build_risk_budget_base(cfg, paths)
    build_sleeve_weights(cfg, paths)
    build_asset_weights(cfg, paths)
    _, metrics = run_backtest(cfg, paths)
    values = dict(zip(metrics["metric"], metrics["value"]))
    return {
        "candidate": candidate["name"],
        "profile": profile,
        "annual_return": float(values["annual_return"]),
        "annual_vol": float(values["annual_vol"]),
        "sharpe": float(values["sharpe"]),
        "max_drawdown": float(values["max_drawdown"]),
        "avg_turnover": float(values["avg_turnover"]),
    }


def main() -> None:
    rows = []
    for candidate in candidate_grid():
        for profile in ["low_vol", "balanced", "high_vol"]:
            row = run_one(profile, candidate)
            print(row, flush=True)
            rows.append(row)
    out = pd.DataFrame(rows)
    out["score"] = out["sharpe"] + 0.20 * out["annual_return"] - 0.50 * out["max_drawdown"].abs()
    summary = (
        out.groupby("candidate", as_index=False)
        .agg(
            mean_sharpe=("sharpe", "mean"),
            min_sharpe=("sharpe", "min"),
            mean_return=("annual_return", "mean"),
            max_drawdown_worst=("max_drawdown", "min"),
            mean_turnover=("avg_turnover", "mean"),
            mean_score=("score", "mean"),
        )
        .sort_values(["mean_score", "mean_sharpe"], ascending=False)
    )
    REPORTS.mkdir(parents=True, exist_ok=True)
    out.to_csv(REPORTS / "agentic_ep_param_search_detail.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(REPORTS / "agentic_ep_param_search_summary.csv", index=False, encoding="utf-8-sig")
    print("\nSUMMARY")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

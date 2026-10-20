from __future__ import annotations

import argparse
import copy
import itertools
import json
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.v1_strategy import calc_metrics, load_config, run_backtest  # noqa: E402


PROFILES = ["low_vol", "balanced", "high_vol"]
DEV_FOLDS = {
    "dev_2018_2020": ("2018-01-01", "2020-12-31"),
    "dev_2021_2022": ("2021-01-01", "2022-12-31"),
    "validation_2023_2024": ("2023-01-01", "2024-12-31"),
}
HOLDOUT = ("2025-01-01", "2026-12-31")
TEMP_DATA = ROOT / "data_processed" / "profile_differentiation_search"
TEMP_REPORTS = ROOT / "output" / "reports" / "profile_differentiation_search"


def candidate(
    candidate_id: str,
    profile: str,
    budget: list[float],
    bounds: list[list[float]],
    ep: list[float],
    trend: list[float],
    rate_30y_weight: float,
    quality_assets: bool = True,
    alt_mix: str = "default",
    equity_mix: str = "default",
    covariance_method: str = "sample_shrink",
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "profile": profile,
        "budget": dict(zip(["rates", "credit", "convertible_bonds", "equity", "alternatives"], budget)),
        "bounds": dict(zip(["rates", "credit", "convertible_bonds", "equity", "alternatives"], bounds)),
        "ep": {
            "risk_aversion": ep[0],
            "base_deviation_penalty": ep[1],
            "turnover_penalty": ep[2],
            "max_active_deviation": ep[3],
            "max_sleeve_turnover": ep[4],
        },
        "trend": {"tilt_per_score": trend[0], "max_sleeve_tilt": trend[1]},
        "rate_30y_weight": rate_30y_weight,
        "quality_assets": quality_assets,
        "alt_mix": alt_mix,
        "equity_mix": equity_mix,
        "covariance_method": covariance_method,
    }


def build_candidates() -> list[dict[str, Any]]:
    low_bounds = [[0.38, 0.68], [0.25, 0.55], [0.00, 0.04], [0.00, 0.03], [0.05, 0.18]]
    bal_bounds = [[0.25, 0.58], [0.20, 0.48], [0.04, 0.10], [0.04, 0.12], [0.08, 0.22]]
    high_bounds = [[0.18, 0.42], [0.06, 0.25], [0.10, 0.24], [0.12, 0.30], [0.18, 0.35]]
    rows = [
        candidate("low_raw_current", "low_vol", [.40, .40, .04, .02, .14], low_bounds, [1.15, 12, .45, .020, .08], [.034, .09], .62, False),
        candidate("low_quality_current", "low_vol", [.40, .40, .04, .02, .14], low_bounds, [1.15, 12, .45, .020, .08], [.034, .09], .62),
        candidate("low_rates_anchor", "low_vol", [.46, .35, .01, .01, .17], [[.42, .70], [.23, .50], [0, .03], [0, .02], [.08, .20]], [1.25, 14, .55, .015, .06], [.025, .06], .62),
        candidate("low_credit_carry", "low_vol", [.35, .48, .01, .01, .15], [[.34, .62], [.32, .58], [0, .03], [0, .02], [.08, .18]], [1.30, 15, .55, .015, .06], [.022, .055], .58),
        candidate("low_rates_alt", "low_vol", [.45, .32, .01, .01, .21], [[.40, .68], [.22, .48], [0, .03], [0, .02], [.12, .22]], [1.20, 14, .50, .018, .07], [.026, .065], .62),
        candidate("low_smooth_core", "low_vol", [.40, .40, .04, .02, .14], low_bounds, [1.35, 18, .65, .012, .05], [.020, .05], .58),
        candidate("low_moderate_alt", "low_vol", [.42, .38, .02, .01, .17], [[.40, .68], [.25, .52], [0, .03], [0, .02], [.09, .20]], [1.25, 15, .55, .015, .06], [.024, .06], .58),
        candidate("low_rates_max", "low_vol", [.50, .32, .01, .00, .17], [[.45, .72], [.22, .46], [0, .02], [0, .01], [.10, .20]], [1.30, 16, .60, .012, .05], [.020, .05], .66),
        candidate("low_defensive_alt", "low_vol", [.38, .36, .01, .00, .25], [[.36, .64], [.24, .50], [0, .02], [0, .01], [.14, .26]], [1.25, 15, .55, .015, .06], [.022, .055], .58, True, "hedge"),

        candidate("balanced_raw_current", "balanced", [.30, .30, .12, .12, .16], bal_bounds, [1.15, 12, .45, .020, .08], [.034, .09], .62, False),
        candidate("balanced_quality_current", "balanced", [.30, .30, .12, .12, .16], bal_bounds, [1.15, 12, .45, .020, .08], [.034, .09], .62),
        candidate("balanced_all_weather", "balanced", [.28, .27, .07, .18, .20], [[.23, .54], [.17, .43], [.03, .09], [.07, .18], [.10, .25]], [1.05, 10, .40, .025, .09], [.034, .085], .60),
        candidate("balanced_equity_plus", "balanced", [.27, .25, .06, .23, .19], [[.22, .52], [.16, .40], [.03, .08], [.09, .23], [.10, .24]], [.95, 9, .38, .030, .10], [.038, .095], .58),
        candidate("balanced_alt_plus", "balanced", [.28, .25, .06, .16, .25], [[.23, .54], [.17, .41], [.03, .08], [.07, .18], [.13, .28]], [1.00, 10, .40, .026, .09], [.034, .085], .58, True, "diversified"),
        candidate("balanced_risk_plus", "balanced", [.25, .23, .08, .24, .20], [[.20, .50], [.15, .38], [.04, .10], [.10, .25], [.11, .25]], [.90, 8, .35, .032, .11], [.040, .10], .58),
        candidate("balanced_core_tilt", "balanced", [.32, .25, .06, .16, .21], [[.27, .57], [.17, .42], [.03, .08], [.07, .18], [.11, .25]], [1.05, 11, .42, .024, .08], [.032, .08], .62),
        candidate("balanced_carry", "balanced", [.28, .32, .06, .14, .20], [[.23, .53], [.23, .47], [.03, .08], [.06, .16], [.10, .24]], [1.10, 12, .45, .022, .08], [.030, .075], .58),
        candidate("balanced_dynamic", "balanced", [.27, .26, .07, .20, .20], [[.22, .53], [.17, .42], [.03, .09], [.08, .21], [.10, .25]], [.95, 9, .38, .030, .10], [.042, .105], .60),

        candidate("high_raw_current", "high_vol", [.16, .12, .18, .24, .30], high_bounds, [1.05, 8, .35, .030, .10], [.040, .09], .62, False),
        candidate("high_quality_current", "high_vol", [.16, .12, .18, .24, .30], high_bounds, [1.05, 8, .35, .030, .10], [.040, .09], .62, True, "diversified"),
        candidate("high_equity_alt", "high_vol", [.12, .07, .08, .40, .33], [[.12, .34], [.04, .15], [.05, .15], [.20, .42], [.22, .40]], [.80, 6, .30, .040, .13], [.045, .11], .55, True, "diversified"),
        candidate("high_equity_focus", "high_vol", [.14, .08, .07, .43, .28], [[.13, .35], [.04, .16], [.05, .14], [.22, .44], [.19, .36]], [.78, 6, .28, .042, .14], [.048, .12], .55, True, "diversified"),
        candidate("high_alt_focus", "high_vol", [.12, .06, .07, .32, .43], [[.12, .34], [.03, .14], [.05, .14], [.18, .36], [.26, .44]], [.82, 6, .30, .040, .13], [.045, .11], .55, True, "inflation"),
        candidate("high_balanced_risk", "high_vol", [.16, .07, .08, .37, .32], [[.14, .36], [.04, .15], [.05, .15], [.20, .40], [.21, .39]], [.85, 7, .32, .038, .12], [.044, .11], .58, True, "diversified"),
        candidate("high_low_cb", "high_vol", [.16, .08, .05, .39, .32], [[.14, .36], [.04, .16], [.03, .11], [.21, .42], [.21, .39]], [.82, 6, .30, .040, .13], [.046, .115], .58, True, "diversified"),
        candidate("high_equity_core", "high_vol", [.20, .06, .06, .38, .30], [[.16, .38], [.03, .14], [.04, .12], [.20, .41], [.20, .37]], [.85, 7, .32, .038, .12], [.044, .11], .60, True, "diversified"),
        candidate("high_robust_risk", "high_vol", [.18, .08, .08, .35, .31], [[.16, .38], [.04, .16], [.05, .14], [.19, .38], [.20, .38]], [.90, 8, .35, .035, .11], [.042, .10], .58, True, "hedge"),
    ]
    rows.extend(
        [
            candidate("low_credit_carry_r52", "low_vol", [.35, .48, .01, .01, .15], [[.34, .62], [.32, .58], [0, .03], [0, .02], [.08, .18]], [1.30, 15, .55, .015, .06], [.022, .055], .52),
            candidate("low_credit_carry_r62", "low_vol", [.35, .48, .01, .01, .15], [[.34, .62], [.32, .58], [0, .03], [0, .02], [.08, .18]], [1.30, 15, .55, .015, .06], [.022, .055], .62),
            candidate("low_credit_carry_smooth", "low_vol", [.35, .48, .01, .01, .15], [[.34, .62], [.32, .58], [0, .03], [0, .02], [.08, .18]], [1.40, 18, .70, .010, .045], [.018, .045], .58),
            candidate("low_credit_alt_neighbor", "low_vol", [.36, .45, .01, .01, .17], [[.34, .62], [.30, .56], [0, .03], [0, .02], [.10, .20]], [1.30, 16, .60, .013, .05], [.020, .05], .56),

            candidate("balanced_core_active", "balanced", [.30, .25, .05, .20, .20], [[.24, .54], [.17, .42], [.03, .08], [.09, .22], [.11, .25]], [1.00, 10, .40, .026, .09], [.036, .09], .58, True, "default", "active"),
            candidate("balanced_active_plus", "balanced", [.28, .25, .05, .23, .19], [[.22, .52], [.16, .40], [.03, .08], [.10, .24], [.10, .24]], [.95, 9, .38, .030, .10], [.039, .095], .58, True, "default", "active"),
            candidate("balanced_active_alt", "balanced", [.28, .24, .05, .21, .22], [[.22, .52], [.16, .39], [.03, .08], [.10, .23], [.12, .27]], [.95, 9, .38, .030, .10], [.039, .095], .56, True, "diversified", "active"),
            candidate("balanced_core_ewma", "balanced", [.32, .25, .06, .16, .21], [[.27, .57], [.17, .42], [.03, .08], [.07, .18], [.11, .25]], [1.05, 11, .42, .024, .08], [.032, .08], .62, True, "default", "default", "ewma_shrink"),
            candidate("balanced_core_rienet", "balanced", [.32, .25, .06, .16, .21], [[.27, .57], [.17, .42], [.03, .08], [.07, .18], [.11, .25]], [1.05, 11, .42, .024, .08], [.032, .08], .62, True, "default", "default", "rienet_inspired"),
            candidate("balanced_core_active_trend", "balanced", [.30, .25, .05, .20, .20], [[.24, .54], [.17, .42], [.03, .08], [.09, .22], [.11, .25]], [.95, 9, .36, .034, .11], [.050, .12], .58, True, "default", "active"),
            candidate("balanced_barbell", "balanced", [.33, .22, .04, .20, .21], [[.27, .56], [.14, .38], [.02, .07], [.09, .22], [.12, .26]], [1.00, 10, .38, .030, .10], [.042, .105], .60, True, "hedge", "active"),
            candidate("balanced_dynamic_active", "balanced", [.29, .24, .05, .21, .21], [[.23, .53], [.16, .40], [.03, .08], [.09, .23], [.11, .26]], [.92, 8, .34, .038, .12], [.055, .13], .58, True, "hedge", "active"),

            candidate("high_moderate_hedge", "high_vol", [.17, .09, .08, .33, .33], [[.15, .38], [.05, .18], [.05, .15], [.17, .36], [.21, .38]], [.92, 8, .35, .035, .11], [.042, .10], .58, True, "hedge"),
            candidate("high_moderate_active", "high_vol", [.18, .08, .06, .35, .33], [[.16, .39], [.04, .16], [.04, .12], [.18, .38], [.21, .38]], [.88, 7, .33, .037, .12], [.044, .105], .58, True, "hedge", "active"),
            candidate("high_current_wider", "high_vol", [.16, .12, .14, .28, .30], [[.16, .40], [.05, .20], [.07, .18], [.16, .34], [.19, .37]], [.95, 8, .35, .034, .11], [.042, .10], .60, True, "hedge"),
            candidate("high_equity_hedge", "high_vol", [.18, .07, .05, .37, .33], [[.16, .38], [.04, .15], [.03, .11], [.19, .40], [.21, .39]], [.85, 7, .32, .038, .12], [.045, .11], .58, True, "hedge", "active"),
            candidate("high_robust_default_alt", "high_vol", [.18, .08, .08, .35, .31], [[.16, .38], [.04, .16], [.05, .14], [.19, .38], [.20, .38]], [.90, 8, .35, .035, .11], [.042, .10], .58, True, "default"),
            candidate("high_robust_active", "high_vol", [.18, .08, .08, .35, .31], [[.16, .38], [.04, .16], [.05, .14], [.19, .38], [.20, .38]], [.90, 8, .35, .035, .11], [.042, .10], .58, True, "hedge", "active"),
            candidate("high_moderate_ewma", "high_vol", [.17, .09, .08, .33, .33], [[.15, .38], [.05, .18], [.05, .15], [.17, .36], [.21, .38]], [.92, 8, .35, .035, .11], [.042, .10], .58, True, "hedge", "default", "ewma_shrink"),
            candidate("high_moderate_rienet", "high_vol", [.17, .09, .08, .33, .33], [[.15, .38], [.05, .18], [.05, .15], [.17, .36], [.21, .38]], [.92, 8, .35, .035, .11], [.042, .10], .58, True, "hedge", "default", "rienet_inspired"),
            candidate("high_dynamic_hedge", "high_vol", [.17, .09, .07, .33, .34], [[.15, .38], [.04, .17], [.04, .13], [.17, .37], [.22, .39]], [.85, 7, .30, .050, .15], [.060, .15], .58, True, "hedge", "active"),
            candidate("high_quality_mid", "high_vol", [.18, .10, .10, .28, .34], [[.16, .40], [.05, .19], [.06, .16], [.15, .33], [.22, .39]], [.95, 9, .38, .032, .10], [.040, .10], .60, True, "hedge"),
            candidate("high_barbell", "high_vol", [.23, .05, .05, .30, .37], [[.19, .43], [.03, .12], [.03, .10], [.17, .35], [.23, .41]], [.90, 8, .34, .036, .11], [.045, .11], .60, True, "hedge", "active"),
            candidate("high_quality_active", "high_vol", [.16, .12, .12, .28, .32], [[.15, .39], [.06, .21], [.07, .18], [.16, .34], [.20, .38]], [.92, 8, .34, .036, .11], [.045, .11], .60, True, "hedge", "active"),
        ]
    )
    return rows


def quality_asset_overrides(profile: str, alt_mix: str, equity_mix: str = "default") -> dict[str, Any]:
    credit = {
        "credit_ind_aa_plus": .35,
        "credit_ind_aa": .30,
        "credit_lgfv_aa_plus": .25,
        "credit_ind_aaa": .10,
    }
    cb = {
        "cb_balanced_large_high": .50,
        "cb_balanced_small_low": .30,
        "cb_bond_large_low": .10,
        "cb_bond_small_low": .10,
    }
    if profile == "low_vol":
        equity = {"equity_broad": .40, "equity_dividend_utility": .35, "equity_fin_property": .15, "equity_manufacturing": .10}
    elif profile == "balanced":
        equity = {
            "equity_broad": .30, "equity_csi1000": .20, "equity_chinext": .12,
            "equity_dividend_utility": .15, "equity_manufacturing": .08,
            "equity_healthcare": .05, "equity_consumption": .05,
            "equity_fin_property": .03, "equity_cyclical": .02,
        }
    else:
        equity = {
            "equity_chinext": .30, "equity_csi1000": .26, "equity_broad": .20,
            "equity_dividend_utility": .10, "equity_manufacturing": .06,
            "equity_fin_property": .03, "equity_healthcare": .03, "equity_consumption": .02,
        }
    if equity_mix == "active" and profile == "balanced":
        equity = {
            "equity_csi1000": .25, "equity_chinext": .18, "equity_broad": .25,
            "equity_dividend_utility": .12, "equity_manufacturing": .08,
            "equity_fin_property": .04, "equity_healthcare": .03,
            "equity_consumption": .03, "equity_cyclical": .02,
        }
    elif equity_mix == "active" and profile == "high_vol":
        equity = {
            "equity_chinext": .34, "equity_csi1000": .30, "equity_broad": .17,
            "equity_dividend_utility": .07, "equity_manufacturing": .05,
            "equity_fin_property": .02, "equity_healthcare": .03, "equity_consumption": .02,
        }
    alternatives = {
        "default": {"alt_gold": .55, "alt_commodity": .15, "alt_nonferrous": .10, "alt_chemical": .08, "alt_us_bond": .07, "alt_reits": .05},
        "hedge": {"alt_gold": .50, "alt_commodity": .14, "alt_nonferrous": .10, "alt_chemical": .06, "alt_us_bond": .12, "alt_reits": .08},
        "diversified": {"alt_gold": .38, "alt_commodity": .21, "alt_nonferrous": .16, "alt_chemical": .10, "alt_us_bond": .08, "alt_reits": .07},
        "inflation": {"alt_gold": .30, "alt_commodity": .26, "alt_nonferrous": .20, "alt_chemical": .14, "alt_us_bond": .05, "alt_reits": .05},
    }[alt_mix]
    return {
        "credit": {"preferred_asset_weights": credit},
        "convertible_bonds": {"preferred_asset_weights": cb},
        "equity": {"preferred_asset_weights": equity},
        "alternatives": {"preferred_asset_weights": alternatives},
    }


def configure_case(spec: dict[str, Any]) -> dict[str, Any]:
    cfg = load_config()
    profile = spec["profile"]
    cfg["strategy_version"] = "no_agent"
    cfg["product"]["selected_profile"] = profile
    prof = cfg["product"]["profiles"][profile]
    prof["target_risk_contribution"] = copy.deepcopy(spec["budget"])
    prof["sleeve_bounds"] = {**copy.deepcopy(spec["bounds"]), "cash": [0.0, 0.0]}
    prof["ep_skfolio_overrides"] = copy.deepcopy(spec["ep"])
    prof.setdefault("signal_overlay_overrides", {})["trend_following"] = copy.deepcopy(spec["trend"])
    if spec["quality_assets"]:
        prof["asset_overrides"] = quality_asset_overrides(profile, spec["alt_mix"], spec.get("equity_mix", "default"))
    cfg["asset_mapping"]["rates_long_duration"]["preferred_asset_weights"] = {
        "rate_30y_curve_proxy": float(spec["rate_30y_weight"]),
        "rate_7_10y": 1.0 - float(spec["rate_30y_weight"]),
    }
    cfg["signal_overlays"]["itransformer_rank"]["enabled"] = False
    cfg["signal_overlays"]["asset_itransformer"]["enabled"] = False
    cfg["ep_skfolio"]["covariance_estimator"]["method"] = str(spec.get("covariance_method", "sample_shrink"))
    prefix = spec["candidate_id"]
    cfg["outputs"] = {
        "weights": f"data_processed/profile_differentiation_search/{prefix}_weights.parquet",
        "sleeve_weights": f"data_processed/profile_differentiation_search/{prefix}_sleeves.parquet",
        "risk_budget_base": f"data_processed/profile_differentiation_search/{prefix}_risk.parquet",
        "nav": f"output/reports/profile_differentiation_search/{prefix}_nav.csv",
        "metrics": f"output/reports/profile_differentiation_search/{prefix}_metrics.csv",
        "data_quality": f"output/reports/profile_differentiation_search/{prefix}_data_quality.csv",
    }
    return cfg


def period_metrics(nav: pd.DataFrame, start: str, end: str) -> dict[str, float]:
    sub = nav[pd.to_datetime(nav["date"]).between(pd.Timestamp(start), pd.Timestamp(end))].copy()
    if sub.empty:
        return {"annual_return": np.nan, "annual_vol": np.nan, "sharpe": np.nan, "max_drawdown": np.nan, "avg_turnover": np.nan}
    sub["nav"] = (1.0 + sub["net_return"]).cumprod()
    return calc_metrics(sub).set_index("metric")["value"].to_dict()


def score_profile(profile: str, fold_metrics: list[dict[str, float]], dev: dict[str, float]) -> float:
    sharpes = np.asarray([m["sharpe"] for m in fold_metrics], dtype=float)
    returns = np.asarray([m["annual_return"] for m in fold_metrics], dtype=float)
    drawdowns = np.asarray([m["max_drawdown"] for m in fold_metrics], dtype=float)
    target_ranges = {"low_vol": (.028, .038), "balanced": (.038, .052), "high_vol": (.060, .085)}
    lo, hi = target_ranges[profile]
    vol = float(dev["annual_vol"])
    vol_distance = max(lo - vol, 0.0) + max(vol - hi, 0.0)
    return float(
        np.nanmean(sharpes)
        + .45 * np.nanmin(sharpes)
        + 1.5 * np.nanmean(returns)
        + .50 * np.nanmin(drawdowns)
        - .12 * float(dev["avg_turnover"])
        - 20.0 * vol_distance
    )


def run_candidate(spec: dict[str, Any]) -> dict[str, Any]:
    row = {k: v for k, v in spec.items() if k not in {"budget", "bounds", "ep", "trend"}}
    row["spec_json"] = json.dumps(spec, ensure_ascii=False, sort_keys=True)
    try:
        cfg = configure_case(spec)
        nav, full_metrics = run_backtest(cfg)
        full = full_metrics.set_index("metric")["value"].to_dict()
        folds = [period_metrics(nav, start, end) for start, end in DEV_FOLDS.values()]
        dev = period_metrics(nav, "2018-01-01", "2024-12-31")
        holdout = period_metrics(nav, *HOLDOUT)
        sleeves = pd.read_parquet(ROOT / cfg["outputs"]["sleeve_weights"])
        means = sleeves[["rates", "credit", "convertible_bonds", "equity", "alternatives"]].mean()
        row.update({f"full_{k}": v for k, v in full.items()})
        row.update({f"dev_{k}": v for k, v in dev.items()})
        row.update({f"holdout_{k}": v for k, v in holdout.items()})
        for fold_name, values in zip(DEV_FOLDS, folds):
            row.update({f"{fold_name}_{k}": v for k, v in values.items()})
        row.update({f"mean_weight_{k}": float(v) for k, v in means.items()})
        row["mean_weight_risk_assets"] = float(means["convertible_bonds"] + means["equity"])
        row["mean_weight_bonds"] = float(means["rates"] + means["credit"])
        row["dev_score"] = score_profile(spec["profile"], folds, dev)
        row["error"] = ""
    except Exception as exc:
        row["error"] = repr(exc)
        row["dev_score"] = -np.inf
    return row


def combine_profiles(detail: pd.DataFrame) -> pd.DataFrame:
    usable = detail[detail["error"].fillna("").eq("")].copy()
    pools = [usable[usable["profile"].eq(profile)].sort_values("dev_score", ascending=False).head(20) for profile in PROFILES]
    rows = []
    for low, balanced, high in itertools.product(*(pool.to_dict(orient="records") for pool in pools)):
        vol_order = low["dev_annual_vol"] + .004 <= balanced["dev_annual_vol"] and balanced["dev_annual_vol"] + .010 <= high["dev_annual_vol"]
        risk_order = (
            low["mean_weight_risk_assets"] <= .05
            and balanced["mean_weight_risk_assets"] >= low["mean_weight_risk_assets"] + .05
            and high["mean_weight_risk_assets"] >= balanced["mean_weight_risk_assets"] + .10
        )
        bond_order = low["mean_weight_bonds"] >= .78 and balanced["mean_weight_bonds"] <= .78 and high["mean_weight_bonds"] <= .52
        return_order = balanced["dev_annual_return"] >= low["dev_annual_return"] - .005 and high["dev_annual_return"] >= balanced["dev_annual_return"] - .005
        constraints_ok = bool(vol_order and risk_order and bond_order and return_order)
        score = low["dev_score"] + balanced["dev_score"] + high["dev_score"]
        score += 4.0 * (balanced["dev_annual_vol"] - low["dev_annual_vol"])
        score += 3.0 * (high["dev_annual_vol"] - balanced["dev_annual_vol"])
        if not constraints_ok:
            score -= 2.0
        rows.append(
            {
                "low_candidate": low["candidate_id"],
                "balanced_candidate": balanced["candidate_id"],
                "high_candidate": high["candidate_id"],
                "constraints_ok": constraints_ok,
                "selection_score": score,
                "low_dev_return": low["dev_annual_return"],
                "balanced_dev_return": balanced["dev_annual_return"],
                "high_dev_return": high["dev_annual_return"],
                "low_dev_vol": low["dev_annual_vol"],
                "balanced_dev_vol": balanced["dev_annual_vol"],
                "high_dev_vol": high["dev_annual_vol"],
                "low_dev_sharpe": low["dev_sharpe"],
                "balanced_dev_sharpe": balanced["dev_sharpe"],
                "high_dev_sharpe": high["dev_sharpe"],
                "low_holdout_sharpe": low["holdout_sharpe"],
                "balanced_holdout_sharpe": balanced["holdout_sharpe"],
                "high_holdout_sharpe": high["holdout_sharpe"],
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(["constraints_ok", "selection_score"], ascending=[False, False]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Search robust, differentiated V1 product profiles without using the 2025+ holdout for selection.")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--keep-temp", action="store_true")
    args = parser.parse_args()

    TEMP_DATA.mkdir(parents=True, exist_ok=True)
    TEMP_REPORTS.mkdir(parents=True, exist_ok=True)
    specs = build_candidates()
    rows = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(run_candidate, spec): spec["candidate_id"] for spec in specs}
        for index, future in enumerate(as_completed(futures), 1):
            result = future.result()
            rows.append(result)
            print(f"done {index}/{len(specs)} {result['candidate_id']} score={result.get('dev_score')}", flush=True)

    detail = pd.DataFrame(rows).sort_values(["profile", "dev_score"], ascending=[True, False])
    combinations = combine_profiles(detail)
    detail_path = ROOT / "output" / "reports" / "v1_profile_differentiation_search_detail.csv"
    combo_path = ROOT / "output" / "reports" / "v1_profile_differentiation_search_combinations.csv"
    detail.to_csv(detail_path, index=False, encoding="utf-8-sig")
    combinations.to_csv(combo_path, index=False, encoding="utf-8-sig")
    print(combinations.head(10).to_string(index=False))
    print(json.dumps({"detail": str(detail_path.relative_to(ROOT)), "combinations": str(combo_path.relative_to(ROOT)), "cases": len(specs)}, ensure_ascii=False))

    if not args.keep_temp:
        shutil.rmtree(TEMP_DATA, ignore_errors=True)
        shutil.rmtree(TEMP_REPORTS, ignore_errors=True)


if __name__ == "__main__":
    main()

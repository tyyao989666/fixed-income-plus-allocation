from __future__ import annotations

import argparse
import itertools
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.v1_strategy import RISK_SLEEVES, build_asset_weights, load_config, run_backtest, sleeve_bounds  # noqa: E402


PROFILES = ["low_vol", "balanced", "high_vol"]
TEMP_DATA = ROOT / "data_processed" / "grid_search_v1"
TEMP_REPORTS = ROOT / "output" / "reports" / "grid_search_v1"


def parse_list(value: str, cast=float) -> list[Any]:
    return [cast(x.strip()) for x in value.split(",") if x.strip()]


def apply_budget_scales(cfg: dict[str, Any], profile: str, cb_scale: float, equity_scale: float, credit_share: float) -> None:
    target = cfg["product"]["profiles"][profile]["target_risk_contribution"]
    old = {k: float(v) for k, v in target.items()}
    new = dict(old)
    new["convertible_bonds"] = old["convertible_bonds"] * cb_scale
    new["equity"] = old["equity"] * equity_scale
    diff = sum(old.values()) - sum(new.values())
    alt_share = 1.0 - credit_share
    if diff >= 0:
        new["credit"] += diff * credit_share
        new["alternatives"] += diff * alt_share
    else:
        need = -diff
        donors = {"credit": credit_share, "alternatives": alt_share}
        for sleeve, share in donors.items():
            new[sleeve] = max(0.01, new[sleeve] - need * share)
    total = sum(max(v, 0.0) for v in new.values())
    cfg["product"]["profiles"][profile]["target_risk_contribution"] = {k: max(v, 0.0) / total for k, v in new.items()}

    bounds = cfg["product"]["profiles"][profile]["sleeve_bounds"]
    if cb_scale < 1.0:
        lo, hi = map(float, bounds["convertible_bonds"])
        bounds["convertible_bonds"] = [lo, max(lo + 0.01, hi * max(cb_scale, 0.65))]
    elif cb_scale > 1.0:
        lo, hi = map(float, bounds["convertible_bonds"])
        bounds["convertible_bonds"] = [lo, min(0.25, hi * cb_scale)]
    if equity_scale < 1.0:
        lo, hi = map(float, bounds["equity"])
        bounds["equity"] = [lo, max(lo + 0.01, hi * max(equity_scale, 0.70))]
    elif equity_scale > 1.0:
        lo, hi = map(float, bounds["equity"])
        bounds["equity"] = [lo, min(0.25, hi * equity_scale)]


def apply_params(cfg: dict[str, Any], profile: str, params: dict[str, Any]) -> None:
    cfg["product"]["selected_profile"] = profile
    cfg["asset_mapping"]["rates_long_duration"]["preferred_assets"] = ["rate_30y_curve_proxy", "rate_7_10y"]
    cfg["asset_mapping"]["rates_long_duration"]["preferred_asset_weights"] = {
        "rate_30y_curve_proxy": float(params["rate_30y_weight"]),
        "rate_7_10y": 1.0 - float(params["rate_30y_weight"]),
    }
    cfg["signal_overlays"]["itransformer_rank"]["tilt"] = float(params["itr_tilt"])
    cfg["signal_overlays"]["itransformer_rank"]["min_top_prob"] = float(params["min_top_prob"])
    cfg["signal_overlays"]["trend_following"]["tilt_per_score"] = float(params["trend_tilt"])
    cfg["signal_overlays"]["trend_following"]["max_sleeve_tilt"] = float(params["trend_max"])
    cfg["signal_overlays"]["trend_following"]["enabled_profiles"] = (
        ["low_vol", "balanced", "high_vol"] if params["trend_high_vol"] else ["low_vol", "balanced"]
    )
    cfg["signal_overlays"]["asset_itransformer"]["blend"] = float(params["asset_blend"])
    cfg["signal_overlays"]["asset_itransformer"]["temperature"] = float(params["asset_temp"])
    apply_budget_scales(cfg, profile, float(params["cb_scale"]), float(params["equity_scale"]), float(params["credit_share"]))


def set_temp_outputs(cfg: dict[str, Any], case_id: str, profile: str) -> None:
    prefix = f"{case_id}_{profile}"
    cfg["outputs"] = {
        "weights": f"data_processed/grid_search_v1/{prefix}_weights.parquet",
        "sleeve_weights": f"data_processed/grid_search_v1/{prefix}_sleeves.parquet",
        "risk_budget_base": f"data_processed/grid_search_v1/{prefix}_risk_base.parquet",
        "nav": f"output/reports/grid_search_v1/{prefix}_nav.csv",
        "metrics": f"output/reports/grid_search_v1/{prefix}_metrics.csv",
        "attribution": f"output/reports/grid_search_v1/{prefix}_attribution.csv",
        "robustness": f"output/reports/grid_search_v1/{prefix}_robustness.csv",
        "data_quality": f"output/reports/grid_search_v1/{prefix}_data_quality.csv",
    }


def run_case(case_id: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for profile in PROFILES:
        cfg = load_config()
        apply_params(cfg, profile, params)
        set_temp_outputs(cfg, case_id, profile)
        try:
            nav, metrics = run_backtest(cfg)
            weights = build_asset_weights(cfg)
            sleeve = pd.read_parquet(ROOT / cfg["outputs"]["sleeve_weights"])
            bounds = sleeve_bounds(cfg)
            max_bound_violation = 0.0
            for sleeve_name, (lo, hi) in bounds.items():
                if sleeve_name in sleeve.columns:
                    max_bound_violation = max(
                        max_bound_violation,
                        float((lo - sleeve[sleeve_name]).clip(lower=0).max()),
                        float((sleeve[sleeve_name] - hi).clip(lower=0).max()),
                    )
            weight_sum_error = float((weights.groupby("date")["weight"].sum() - 1.0).abs().max())
            cash_weight = float(weights.loc[weights["asset_code"].eq("cash"), "weight"].max()) if weights["asset_code"].eq("cash").any() else 0.0
            m = metrics.set_index("metric")["value"].to_dict()
            row = {
                "case_id": case_id,
                "profile": profile,
                **params,
                **m,
                "weight_sum_error": weight_sum_error,
                "max_cash_weight": cash_weight,
                "max_bound_violation": max_bound_violation,
                "max_abs_monthly_return": float(nav["net_return"].abs().max()) if not nav.empty else float("nan"),
                "error": "",
            }
        except Exception as exc:
            row = {"case_id": case_id, "profile": profile, **params, "error": repr(exc)}
        rows.append(row)
    return rows


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    ok = detail[detail["error"].fillna("").eq("")].copy()
    rows = []
    for case_id, group in ok.groupby("case_id"):
        if len(group) != len(PROFILES):
            continue
        vols = group.set_index("profile").reindex(PROFILES)["annual_vol"].to_numpy()
        bound_ok = (
            group["weight_sum_error"].max() <= 1e-6
            and group["max_cash_weight"].max() <= 1e-12
            and group["max_bound_violation"].max() <= 1e-8
        )
        row = {
            "case_id": case_id,
            "cases": len(group),
            "avg_annual_return": group["annual_return"].mean(),
            "avg_annual_vol": group["annual_vol"].mean(),
            "avg_sharpe": group["sharpe"].mean(),
            "min_sharpe": group["sharpe"].min(),
            "worst_drawdown": group["max_drawdown"].min(),
            "avg_turnover": group["avg_turnover"].mean(),
            "vol_order_ok": bool(vols[0] <= vols[1] <= vols[2]),
            "constraint_ok": bool(bound_ok),
        }
        first = group.iloc[0]
        for col in [
            "rate_30y_weight",
            "itr_tilt",
            "min_top_prob",
            "trend_tilt",
            "trend_max",
            "trend_high_vol",
            "asset_blend",
            "asset_temp",
            "cb_scale",
            "equity_scale",
            "credit_share",
        ]:
            row[col] = first[col]
        row["score"] = (
            row["avg_sharpe"]
            + 2.0 * row["avg_annual_return"]
            + 0.45 * row["min_sharpe"]
            + 0.4 * row["worst_drawdown"]
            - 0.20 * row["avg_turnover"]
            + (0.03 if row["vol_order_ok"] else 0.0)
            + (0.05 if row["constraint_ok"] else -1.0)
        )
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["score", "avg_sharpe", "avg_annual_return"], ascending=False)
    return out


def build_param_grid(args: argparse.Namespace) -> list[dict[str, Any]]:
    trend_pairs = []
    for item in args.trend_pairs.split(","):
        left, right = item.split(":")
        trend_pairs.append((float(left), float(right)))
    keys = [
        "rate_30y_weight",
        "itr_tilt",
        "min_top_prob",
        "trend_pair",
        "trend_high_vol",
        "asset_blend",
        "asset_temp",
        "cb_scale",
        "equity_scale",
        "credit_share",
    ]
    values = [
        parse_list(args.rate_30y_weights),
        parse_list(args.itr_tilts),
        parse_list(args.min_top_probs),
        trend_pairs,
        [x.lower() == "true" for x in args.trend_high_vol.split(",")],
        parse_list(args.asset_blends),
        parse_list(args.asset_temps),
        parse_list(args.cb_scales),
        parse_list(args.equity_scales),
        parse_list(args.credit_shares),
    ]
    params = []
    for combo in itertools.product(*values):
        item = dict(zip(keys, combo))
        item["trend_tilt"], item["trend_max"] = item.pop("trend_pair")
        params.append(item)
    return params


def main() -> None:
    parser = argparse.ArgumentParser(description="Grid-search V1 strategy parameters across all product profiles.")
    parser.add_argument("--rate-30y-weights", default="0.65,0.70,0.75,0.80")
    parser.add_argument("--itr-tilts", default="0.05,0.06,0.07")
    parser.add_argument("--min-top-probs", default="0.30,0.35,0.40")
    parser.add_argument("--trend-pairs", default="0.020:0.060,0.024:0.070,0.030:0.080")
    parser.add_argument("--trend-high-vol", default="true")
    parser.add_argument("--asset-blends", default="0.70,0.80,0.90")
    parser.add_argument("--asset-temps", default="0.035")
    parser.add_argument("--cb-scales", default="0.80,1.00,1.20")
    parser.add_argument("--equity-scales", default="0.80,1.00,1.20")
    parser.add_argument("--credit-shares", default="0.45,0.55,0.65")
    parser.add_argument("--limit", type=int, default=0, help="Run only the first N combinations for smoke tests.")
    parser.add_argument("--keep-temp", action="store_true", help="Keep per-case temporary backtest files.")
    parser.add_argument("--checkpoint-every", type=int, default=10, help="Write partial detail/summary every N cases.")
    args = parser.parse_args()

    TEMP_DATA.mkdir(parents=True, exist_ok=True)
    TEMP_REPORTS.mkdir(parents=True, exist_ok=True)
    grid = build_param_grid(args)
    if args.limit:
        grid = grid[: args.limit]
    detail_path = ROOT / "output" / "reports" / "v1_grid_search_detail.csv"
    summary_path = ROOT / "output" / "reports" / "v1_grid_search_summary.csv"
    detail_rows: list[dict[str, Any]] = []
    for idx, params in enumerate(grid, 1):
        case_id = f"case_{idx:04d}"
        detail_rows.extend(run_case(case_id, params))
        if args.checkpoint_every and (idx % args.checkpoint_every == 0):
            partial = pd.DataFrame(detail_rows)
            partial.to_csv(detail_path, index=False, encoding="utf-8-sig")
            summarize(partial).to_csv(summary_path, index=False, encoding="utf-8-sig")
        if idx % 25 == 0 or idx == len(grid):
            print(f"done {idx}/{len(grid)}", flush=True)

    detail = pd.DataFrame(detail_rows)
    detail.to_csv(detail_path, index=False, encoding="utf-8-sig")
    summary = summarize(detail)
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    if not summary.empty:
        print(summary.head(20).to_string(index=False))
    print({"detail": str(detail_path.relative_to(ROOT)), "summary": str(summary_path.relative_to(ROOT)), "cases": len(grid)})

    if not args.keep_temp:
        shutil.rmtree(TEMP_DATA, ignore_errors=True)
        shutil.rmtree(TEMP_REPORTS, ignore_errors=True)


if __name__ == "__main__":
    main()

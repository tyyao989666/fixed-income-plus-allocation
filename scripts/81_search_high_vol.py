from __future__ import annotations

import argparse
import itertools
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.v1_strategy import build_asset_weights, load_config, run_backtest, sleeve_bounds  # noqa: E402


TEMP_DATA = ROOT / "data_processed" / "high_vol_search"
TEMP_REPORTS = ROOT / "output" / "reports" / "high_vol_search"
REPORT_DIR = ROOT / "output" / "reports"


BUDGET_PRESETS: dict[str, dict[str, float]] = {
    "current": {
        "rates": 0.16,
        "credit": 0.12,
        "convertible_bonds": 0.18,
        "equity": 0.24,
        "alternatives": 0.30,
    },
    "equity_cb": {
        "rates": 0.13,
        "credit": 0.08,
        "convertible_bonds": 0.22,
        "equity": 0.30,
        "alternatives": 0.27,
    },
    "cb_barbell": {
        "rates": 0.14,
        "credit": 0.07,
        "convertible_bonds": 0.27,
        "equity": 0.27,
        "alternatives": 0.25,
    },
    "alt_equity": {
        "rates": 0.12,
        "credit": 0.07,
        "convertible_bonds": 0.18,
        "equity": 0.30,
        "alternatives": 0.33,
    },
}


BOUNDS_PRESETS: dict[str, dict[str, list[float]]] = {
    "current": {
        "rates": [0.18, 0.42],
        "credit": [0.06, 0.25],
        "convertible_bonds": [0.10, 0.24],
        "equity": [0.12, 0.30],
        "alternatives": [0.18, 0.35],
        "cash": [0.00, 0.00],
    },
    "risk_wide": {
        "rates": [0.16, 0.40],
        "credit": [0.04, 0.20],
        "convertible_bonds": [0.10, 0.28],
        "equity": [0.14, 0.36],
        "alternatives": [0.18, 0.38],
        "cash": [0.00, 0.00],
    },
    "equity_cb_wide": {
        "rates": [0.14, 0.38],
        "credit": [0.04, 0.18],
        "convertible_bonds": [0.12, 0.32],
        "equity": [0.16, 0.40],
        "alternatives": [0.16, 0.36],
        "cash": [0.00, 0.00],
    },
}


EQUITY_PRESETS: dict[str, dict[str, float]] = {
    "current": {
        "equity_chinext": 0.22,
        "equity_csi1000": 0.20,
        "equity_tech_growth": 0.18,
        "equity_broad": 0.14,
        "equity_dividend_utility": 0.08,
        "equity_fin_property": 0.06,
        "equity_manufacturing": 0.06,
        "equity_healthcare": 0.03,
        "equity_consumption": 0.02,
        "equity_cyclical": 0.01,
    },
    "growth": {
        "equity_chinext": 0.26,
        "equity_csi1000": 0.23,
        "equity_tech_growth": 0.23,
        "equity_broad": 0.10,
        "equity_manufacturing": 0.06,
        "equity_healthcare": 0.04,
        "equity_dividend_utility": 0.03,
        "equity_consumption": 0.02,
        "equity_cyclical": 0.02,
        "equity_fin_property": 0.01,
    },
    "balanced_growth": {
        "equity_chinext": 0.23,
        "equity_csi1000": 0.22,
        "equity_tech_growth": 0.21,
        "equity_broad": 0.13,
        "equity_dividend_utility": 0.06,
        "equity_manufacturing": 0.06,
        "equity_healthcare": 0.03,
        "equity_fin_property": 0.03,
        "equity_consumption": 0.02,
        "equity_cyclical": 0.01,
    },
}


ALT_PRESETS: dict[str, dict[str, float]] = {
    "current": {
        "alt_gold": 0.42,
        "alt_commodity": 0.20,
        "alt_nonferrous": 0.15,
        "alt_chemical": 0.13,
        "alt_us_bond": 0.05,
        "alt_reits": 0.05,
    },
    "commodity": {
        "alt_gold": 0.32,
        "alt_commodity": 0.24,
        "alt_nonferrous": 0.20,
        "alt_chemical": 0.16,
        "alt_us_bond": 0.04,
        "alt_reits": 0.04,
    },
}


def parse_list(value: str, cast=float) -> list[Any]:
    return [cast(x.strip()) for x in value.split(",") if x.strip()]


def set_outputs(cfg: dict[str, Any], case_id: str) -> None:
    prefix = f"{case_id}_high_vol"
    cfg["outputs"] = {
        "weights": f"data_processed/high_vol_search/{prefix}_weights.parquet",
        "sleeve_weights": f"data_processed/high_vol_search/{prefix}_sleeves.parquet",
        "risk_budget_base": f"data_processed/high_vol_search/{prefix}_risk_base.parquet",
        "nav": f"output/reports/high_vol_search/{prefix}_nav.csv",
        "metrics": f"output/reports/high_vol_search/{prefix}_metrics.csv",
        "attribution": f"output/reports/high_vol_search/{prefix}_attribution.csv",
        "robustness": f"output/reports/high_vol_search/{prefix}_robustness.csv",
        "data_quality": f"output/reports/high_vol_search/{prefix}_data_quality.csv",
        "ep_posterior": f"data_processed/high_vol_search/{prefix}_ep_posterior.parquet",
        "ep_signals": f"output/reports/high_vol_search/{prefix}_ep_signals.csv",
        "optimizer_diagnostics": f"output/reports/high_vol_search/{prefix}_optimizer_diagnostics.csv",
    }


def apply_params(cfg: dict[str, Any], params: dict[str, Any]) -> None:
    profile = cfg["product"]["profiles"]["high_vol"]
    cfg["product"]["selected_profile"] = "high_vol"
    profile["target_risk_contribution"] = dict(BUDGET_PRESETS[str(params["budget_preset"])])
    profile["sleeve_bounds"] = {k: list(v) for k, v in BOUNDS_PRESETS[str(params["bounds_preset"])].items()}
    profile.setdefault("asset_overrides", {}).setdefault("equity", {})["preferred_asset_weights"] = dict(EQUITY_PRESETS[str(params["equity_preset"])])
    profile.setdefault("asset_overrides", {}).setdefault("alternatives", {})["preferred_asset_weights"] = dict(ALT_PRESETS[str(params["alt_preset"])])
    profile["ep_skfolio_overrides"] = {
        "risk_aversion": float(params["risk_aversion"]),
        "base_deviation_penalty": float(params["deviation_penalty"]),
        "turnover_penalty": float(params["turnover_penalty"]),
        "max_active_deviation": float(params["max_active_deviation"]),
        "max_sleeve_turnover": float(params["max_sleeve_turnover"]),
    }
    profile["signal_overlay_overrides"] = {
        "itransformer_rank": {
            "tilt": float(params["itr_tilt"]),
            "min_top_prob": float(params["min_top_prob"]),
        },
        "trend_following": {
            "tilt_per_score": float(params["trend_tilt"]),
            "max_sleeve_tilt": float(params["trend_max"]),
        },
    }
    cfg.setdefault("view_tilt_profile_scale", {})["high_vol"] = float(params["view_scale"])


def run_case(case_id: str, params: dict[str, Any]) -> dict[str, Any]:
    cfg = load_config()
    apply_params(cfg, params)
    set_outputs(cfg, case_id)
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
        m = metrics.set_index("metric")["value"].to_dict()
        row = {
            "case_id": case_id,
            **params,
            **m,
            "weight_sum_error": float((weights.groupby("date")["weight"].sum() - 1.0).abs().max()),
            "max_cash_weight": float(weights.loc[weights["asset_code"].eq("cash"), "weight"].max()) if weights["asset_code"].eq("cash").any() else 0.0,
            "max_bound_violation": max_bound_violation,
            "max_abs_monthly_return": float(nav["net_return"].abs().max()) if not nav.empty else float("nan"),
            "error": "",
        }
    except Exception as exc:
        row = {"case_id": case_id, **params, "error": repr(exc)}
    return row


def score_results(detail: pd.DataFrame, max_drawdown_floor: float) -> pd.DataFrame:
    ok = detail[detail["error"].fillna("").eq("")].copy()
    if ok.empty:
        return ok
    ok["constraint_ok"] = (
        (ok["weight_sum_error"] <= 1e-6)
        & (ok["max_cash_weight"] <= 1e-12)
        & (ok["max_bound_violation"] <= 1e-8)
    )
    dd_penalty = (max_drawdown_floor - ok["max_drawdown"]).clip(lower=0.0)
    ok["objective_score"] = (
        10.0 * ok["annual_return"]
        + 0.90 * ok["sharpe"]
        + 0.15 * ok["calmar"]
        + 5.0 * ok["max_drawdown"]
        - 0.30 * ok["annual_vol"]
        - 0.12 * ok["avg_turnover"]
        - 30.0 * dd_penalty
        + ok["constraint_ok"].astype(float) * 0.20
    )
    return ok.sort_values(["objective_score", "annual_return", "sharpe"], ascending=False)


def build_grid(args: argparse.Namespace) -> list[dict[str, Any]]:
    keys = [
        "budget_preset",
        "bounds_preset",
        "equity_preset",
        "alt_preset",
        "view_scale",
        "itr_tilt",
        "min_top_prob",
        "trend_tilt",
        "trend_max",
        "risk_aversion",
        "deviation_penalty",
        "turnover_penalty",
        "max_active_deviation",
        "max_sleeve_turnover",
    ]
    values = [
        parse_list(args.budget_presets, str),
        parse_list(args.bounds_presets, str),
        parse_list(args.equity_presets, str),
        parse_list(args.alt_presets, str),
        parse_list(args.view_scales),
        parse_list(args.itr_tilts),
        parse_list(args.min_top_probs),
        parse_list(args.trend_tilts),
        parse_list(args.trend_maxes),
        parse_list(args.risk_aversions),
        parse_list(args.deviation_penalties),
        parse_list(args.turnover_penalties),
        parse_list(args.max_active_deviations),
        parse_list(args.max_sleeve_turnovers),
    ]
    out = []
    for combo in itertools.product(*values):
        item = dict(zip(keys, combo))
        for field, presets in [("budget_preset", BUDGET_PRESETS), ("bounds_preset", BOUNDS_PRESETS), ("equity_preset", EQUITY_PRESETS), ("alt_preset", ALT_PRESETS)]:
            if str(item[field]) not in presets:
                raise ValueError(f"Unknown {field}: {item[field]}")
        out.append(item)
    return out


def write_notes(summary: pd.DataFrame, detail_path: Path, summary_path: Path, max_drawdown_floor: float) -> Path:
    path = REPORT_DIR / "high_vol_search_notes.md"
    if summary.empty:
        path.write_text("# High Vol Search Notes\n\nNo valid cases.\n", encoding="utf-8")
        return path
    best = summary.iloc[0].to_dict()
    top = summary.head(12).copy()
    cols = [
        "case_id",
        "objective_score",
        "annual_return",
        "annual_vol",
        "sharpe",
        "max_drawdown",
        "avg_turnover",
        "budget_preset",
        "bounds_preset",
        "equity_preset",
        "alt_preset",
        "view_scale",
        "itr_tilt",
        "trend_tilt",
        "trend_max",
        "risk_aversion",
        "deviation_penalty",
        "max_active_deviation",
    ]
    text = "\n".join(
        [
            "# High Vol Search Notes",
            "",
            "目标：只优化 `high_vol`，不新增产品线，不改变 low_vol/balanced。",
            "",
            f"- detail: `{detail_path.relative_to(ROOT)}`",
            f"- summary: `{summary_path.relative_to(ROOT)}`",
            f"- drawdown floor used in score: `{max_drawdown_floor:.2%}`",
            "",
            "## Best Candidate",
            "",
            "```json",
            json.dumps(best, ensure_ascii=False, indent=2),
            "```",
            "",
            "## Top Candidates",
            "",
            top[cols].to_markdown(index=False),
            "",
            "## Reading",
            "",
            "- 分数偏向年化收益和 Sharpe，同时惩罚回撤、波动、换手和约束违规。",
            "- 搜索只作为 high_vol 参数选择依据；正式落地后仍需重跑三产品并做约束检查。",
        ]
    )
    path.write_text(text, encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Search high_vol parameters without changing low_vol/balanced.")
    parser.add_argument("--budget-presets", default="current,equity_cb,cb_barbell,alt_equity")
    parser.add_argument("--bounds-presets", default="current,risk_wide,equity_cb_wide")
    parser.add_argument("--equity-presets", default="current,growth,balanced_growth")
    parser.add_argument("--alt-presets", default="current,commodity")
    parser.add_argument("--view-scales", default="0.25,0.35,0.45")
    parser.add_argument("--itr-tilts", default="0.07,0.09")
    parser.add_argument("--min-top-probs", default="0.30,0.35")
    parser.add_argument("--trend-tilts", default="0.034,0.040")
    parser.add_argument("--trend-maxes", default="0.090,0.110")
    parser.add_argument("--risk-aversions", default="0.90,1.05")
    parser.add_argument("--deviation-penalties", default="8.0,10.0")
    parser.add_argument("--turnover-penalties", default="0.35")
    parser.add_argument("--max-active-deviations", default="0.030,0.040")
    parser.add_argument("--max-sleeve-turnovers", default="0.10")
    parser.add_argument("--max-drawdown-floor", type=float, default=-0.065)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--checkpoint-every", type=int, default=20)
    parser.add_argument("--keep-temp", action="store_true")
    args = parser.parse_args()

    TEMP_DATA.mkdir(parents=True, exist_ok=True)
    TEMP_REPORTS.mkdir(parents=True, exist_ok=True)
    grid = build_grid(args)
    if args.limit:
        grid = grid[: args.limit]

    detail_path = REPORT_DIR / "high_vol_search_detail.csv"
    summary_path = REPORT_DIR / "high_vol_search_summary.csv"
    rows: list[dict[str, Any]] = []
    for idx, params in enumerate(grid, 1):
        rows.append(run_case(f"high_case_{idx:04d}", params))
        if args.checkpoint_every and idx % args.checkpoint_every == 0:
            detail = pd.DataFrame(rows)
            detail.to_csv(detail_path, index=False, encoding="utf-8-sig")
            score_results(detail, args.max_drawdown_floor).to_csv(summary_path, index=False, encoding="utf-8-sig")
        if idx % 20 == 0 or idx == len(grid):
            print(f"done {idx}/{len(grid)}", flush=True)

    detail = pd.DataFrame(rows)
    detail.to_csv(detail_path, index=False, encoding="utf-8-sig")
    summary = score_results(detail, args.max_drawdown_floor)
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    notes = write_notes(summary, detail_path, summary_path, args.max_drawdown_floor)
    if not summary.empty:
        print(summary.head(15).to_string(index=False), flush=True)
    print(json.dumps({"detail": str(detail_path.relative_to(ROOT)), "summary": str(summary_path.relative_to(ROOT)), "notes": str(notes.relative_to(ROOT)), "cases": len(grid)}, ensure_ascii=False), flush=True)

    if not args.keep_temp:
        shutil.rmtree(TEMP_DATA, ignore_errors=True)
        shutil.rmtree(TEMP_REPORTS, ignore_errors=True)


if __name__ == "__main__":
    main()

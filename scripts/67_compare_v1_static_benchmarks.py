from __future__ import annotations

import json
import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.v1_strategy import load_config, load_monthly_returns  # noqa: E402


REPORTS = ROOT / "output" / "reports"
FIGURES = ROOT / "output" / "figures" / "v1"
PROFILES = ["low_vol", "balanced", "high_vol"]
PROFILE_NAMES = {
    "low_vol": "V1 low_vol",
    "balanced": "V1 balanced",
    "high_vol": "V1 high_vol",
    "high_vol_aggressive": "V1 high_vol_aggressive",
}
BENCHMARKS: dict[str, dict[str, Any]] = {
    "pure_bond_core": {
        "name": "纯债核心",
        "weights": {
            "rate_7_10y": 0.40,
            "credit_lgfv_aaa": 0.20,
            "credit_ind_aaa": 0.20,
            "credit_lgfv_aa_plus": 0.10,
            "credit_ind_aa_plus": 0.10,
        },
    },
    "pure_bond_30y_plus": {
        "name": "纯债+30Y增强",
        "weights": {
            "rate_30y_curve_proxy": 0.35,
            "rate_7_10y": 0.35,
            "credit_lgfv_aa_plus": 0.15,
            "credit_ind_aa_plus": 0.15,
        },
    },
    "stock_bond_20_80": {
        "name": "股债28",
        "weights": {
            "rate_7_10y": 0.35,
            "credit_lgfv_aa_plus": 0.25,
            "credit_ind_aa_plus": 0.20,
            "equity_broad": 0.20,
        },
    },
    "stock_bond_30_70": {
        "name": "股债37",
        "weights": {
            "rate_7_10y": 0.30,
            "credit_lgfv_aa_plus": 0.20,
            "credit_ind_aa_plus": 0.20,
            "equity_broad": 0.30,
        },
    },
    "static_fip_balanced": {
        "name": "静态固收+",
        "weights": {
            "rate_7_10y": 0.30,
            "credit_lgfv_aa_plus": 0.25,
            "credit_ind_aa_plus": 0.15,
            "cb_balanced_large_low": 0.10,
            "equity_broad": 0.10,
            "alt_gold": 0.10,
        },
    },
}


def calc_metrics(nav: pd.DataFrame) -> dict[str, float]:
    r = pd.to_numeric(nav["net_return"], errors="coerce").fillna(0.0)
    n = len(r)
    value = pd.to_numeric(nav["nav"], errors="coerce")
    total_return = float(value.iloc[-1] - 1.0)
    annual_return = float(value.iloc[-1] ** (12.0 / n) - 1.0) if n else np.nan
    annual_vol = float(r.std(ddof=0) * np.sqrt(12.0)) if n else np.nan
    sharpe = annual_return / annual_vol if annual_vol and annual_vol > 0 else np.nan
    drawdown = value / value.cummax() - 1.0
    max_drawdown = float(drawdown.min())
    calmar = annual_return / abs(max_drawdown) if max_drawdown < 0 else np.nan
    return {
        "months": float(n),
        "total_return": total_return,
        "annual_return": annual_return,
        "annual_vol": annual_vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
    }


def load_return_matrix() -> pd.DataFrame:
    cfg = load_config()
    returns = load_monthly_returns(cfg=cfg)
    pivot = (
        returns.pivot_table(index="date", columns="asset_code", values="return_1m", aggfunc="last")
        .sort_index()
        .replace([np.inf, -np.inf], np.nan)
    )
    return pivot


def normalize_weights(weights: dict[str, float], columns: pd.Index) -> pd.Series:
    missing = [asset for asset in weights if asset not in columns]
    if missing:
        raise KeyError(f"Benchmark assets missing from return matrix: {missing}")
    w = pd.Series(weights, dtype=float)
    w = w / w.sum()
    return w.reindex(columns).fillna(0.0)


def benchmark_nav(name: str, weights: dict[str, float], returns: pd.DataFrame, dates: pd.Series, cost_bps: float = 5.0) -> pd.DataFrame:
    w = normalize_weights(weights, returns.columns)
    sub = returns.reindex(pd.to_datetime(dates)).fillna(0.0)
    gross = sub @ w
    turnover = pd.Series(0.0, index=sub.index)
    if len(turnover):
        turnover.iloc[0] = 1.0
    cost = turnover * cost_bps / 10000.0
    net = gross - cost
    nav = (1.0 + net).cumprod()
    return pd.DataFrame(
        {
            "date": sub.index,
            "portfolio": name,
            "gross_return": gross.to_numpy(dtype=float),
            "turnover": turnover.to_numpy(dtype=float),
            "transaction_cost": cost.to_numpy(dtype=float),
            "net_return": net.to_numpy(dtype=float),
            "nav": nav.to_numpy(dtype=float),
        }
    )


def load_v1_navs(profiles: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    nav_rows = []
    metric_rows = []
    for profile in profiles:
        nav = pd.read_csv(REPORTS / f"v1_{profile}_backtest_nav.csv")
        nav["date"] = pd.to_datetime(nav["date"], errors="coerce")
        nav["portfolio"] = f"v1_{profile}"
        nav_rows.append(nav[["date", "portfolio", "gross_return", "turnover", "transaction_cost", "net_return", "nav"]])
        metrics = calc_metrics(nav)
        metrics.update({"portfolio": f"v1_{profile}", "type": "strategy", "中文名称": PROFILE_NAMES.get(profile, f"V1 {profile}")})
        metric_rows.append(metrics)
    return pd.concat(nav_rows, ignore_index=True), pd.DataFrame(metric_rows)


def add_excess_columns(metrics: pd.DataFrame) -> pd.DataFrame:
    out = metrics.copy()
    bench = out[out["portfolio"].eq("stock_bond_20_80")]
    if bench.empty:
        return out
    b = bench.iloc[0]
    out["excess_ann_return_vs_股债28"] = out["annual_return"] - float(b["annual_return"])
    out["sharpe_diff_vs_股债28"] = out["sharpe"] - float(b["sharpe"])
    out["drawdown_diff_vs_股债28"] = out["max_drawdown"] - float(b["max_drawdown"])
    return out


def write_plot(nav: pd.DataFrame) -> str:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return ""
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    for name, sub in nav.groupby("portfolio"):
        if name.startswith("v1_"):
            ax.plot(sub["date"], sub["nav"], linewidth=2.2, label=name)
        else:
            ax.plot(sub["date"], sub["nav"], linewidth=1.2, linestyle="--", label=name)
    ax.set_title("V1 strategies vs static benchmarks")
    ax.set_ylabel("NAV")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    path = FIGURES / "v1_static_benchmark_nav_compare.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path.relative_to(ROOT))


def write_markdown(metrics: pd.DataFrame, plot_path: str, nav: pd.DataFrame) -> None:
    display = metrics.copy()
    for col in ["total_return", "annual_return", "annual_vol", "max_drawdown", "excess_ann_return_vs_股债28", "drawdown_diff_vs_股债28"]:
        if col in display:
            display[col] = display[col].map(lambda x: f"{x:.2%}" if pd.notna(x) else "")
    for col in ["sharpe", "calmar", "sharpe_diff_vs_股债28"]:
        if col in display:
            display[col] = display[col].map(lambda x: f"{x:.3f}" if pd.notna(x) else "")
    cols = [
        "中文名称",
        "portfolio",
        "type",
        "annual_return",
        "annual_vol",
        "sharpe",
        "max_drawdown",
        "calmar",
        "excess_ann_return_vs_股债28",
        "sharpe_diff_vs_股债28",
    ]
    start = pd.to_datetime(nav["date"]).min().date()
    end = pd.to_datetime(nav["date"]).max().date()
    lines = [
        "# V1策略与静态Benchmark对比",
        "",
        f"口径：{start} 至 {end}，月频收益，benchmark 首月扣 5bp 建仓成本，之后静态月度再平衡。",
        "",
    ]
    if plot_path:
        normalized_plot_path = plot_path.replace("\\", "/")
        lines.extend([f"![NAV compare](../../{normalized_plot_path})", ""])
    lines.extend(
        [
            display[cols].to_markdown(index=False),
            "",
            "解读：",
            "",
            "- `股债28` 是 80% 债券代理 + 20% 沪深300ETF代理（510300.SH）。",
            "- `纯债+30Y增强` 使用 30Y曲线代理，因此收益和夏普会受久期代理假设影响。",
            "- V1 如果相对股债28的年化超额、夏普差值为正，同时最大回撤不明显扩大，才说明动态框架有实际增益。",
        ]
    )
    (REPORTS / "v1_static_benchmark_comparison.md").write_text("\n".join(lines), encoding="utf-8")


def parse_profiles(value: str) -> list[str]:
    if value == "all":
        return PROFILES
    return [x.strip() for x in value.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare V1 strategies with static benchmarks.")
    parser.add_argument("--profiles", default="all", help="all, or comma-separated profiles")
    args = parser.parse_args()
    profiles = parse_profiles(args.profiles)

    REPORTS.mkdir(parents=True, exist_ok=True)
    returns = load_return_matrix()
    v1_nav, v1_metrics = load_v1_navs(profiles)
    date_anchor = "v1_balanced" if "balanced" in profiles else f"v1_{profiles[0]}"
    dates = v1_nav[v1_nav["portfolio"].eq(date_anchor)]["date"]

    nav_rows = [v1_nav]
    metric_rows = [v1_metrics]
    for code, spec in BENCHMARKS.items():
        nav = benchmark_nav(code, spec["weights"], returns, dates)
        metrics = calc_metrics(nav)
        metrics.update({"portfolio": code, "type": "benchmark", "中文名称": spec["name"]})
        nav_rows.append(nav)
        metric_rows.append(pd.DataFrame([metrics]))

    nav_all = pd.concat(nav_rows, ignore_index=True)
    metrics_all = pd.concat(metric_rows, ignore_index=True)
    metrics_all = add_excess_columns(metrics_all)
    order_map = {f"v1_{p}": i for i, p in enumerate(profiles)}
    order_map.update({b: i + len(profiles) for i, b in enumerate(BENCHMARKS)})
    metrics_all = metrics_all.sort_values("portfolio", key=lambda s: s.map(order_map).fillna(999)).reset_index(drop=True)

    nav_path = REPORTS / "v1_static_benchmark_nav.csv"
    metrics_path = REPORTS / "v1_static_benchmark_comparison.csv"
    nav_all.to_csv(nav_path, index=False, encoding="utf-8-sig")
    metrics_all.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    plot_path = write_plot(nav_all)
    write_markdown(metrics_all, plot_path, nav_all)
    print(
        json.dumps(
            {
                "nav": str(nav_path.relative_to(ROOT)),
                "metrics": str(metrics_path.relative_to(ROOT)),
                "report": "output/reports/v1_static_benchmark_comparison.md",
                "plot": plot_path,
                "rows": len(metrics_all),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

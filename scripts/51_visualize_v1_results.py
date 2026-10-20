from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.v1_strategy import RISK_SLEEVES, load_config, load_monthly_returns  # noqa: E402


FIG_DIR = ROOT / "output" / "figures" / "v1"
REPORT_DIR = ROOT / "output" / "reports"
PROFILES = ["low_vol", "balanced", "high_vol"]
SLEEVES = RISK_SLEEVES + ["cash"]
SLEEVE_LABELS = {
    "rates": "利率债",
    "credit": "信用债",
    "convertible_bonds": "转债",
    "equity": "权益",
    "alternatives": "另类",
    "cash": "现金",
}
SLEEVE_COLORS = {
    "rates": "#1F4E79",
    "credit": "#4F9D69",
    "convertible_bonds": "#F0A202",
    "equity": "#C44E52",
    "alternatives": "#8064A2",
    "cash": "#9AA4AF",
}
PROFILE_LABELS = {
    "low_vol": "低波固收+",
    "balanced": "中波固收+",
    "high_vol": "高波固收+",
    "high_vol_aggressive": "进取固收+",
}
PROFILE_COLORS = {
    "low_vol": "#1F4E79",
    "balanced": "#F0A202",
    "high_vol": "#C44E52",
    "high_vol_aggressive": "#7B3294",
}
THEME = {
    "navy": "#1F3A5F",
    "blue": "#2C7FB8",
    "light_blue": "#D9EAF7",
    "grid": "#D6DEE8",
    "text": "#243447",
    "muted": "#6B7280",
    "positive": "#1F7A4D",
    "negative": "#B94A48",
    "gray": "#9AA4AF",
    "paper": "#FFFFFF",
}


def set_plot_style() -> None:
    plt.rcParams["figure.dpi"] = 170
    plt.rcParams["savefig.dpi"] = 240
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.edgecolor"] = "#AEB7C2"
    plt.rcParams["axes.labelcolor"] = THEME["text"]
    plt.rcParams["xtick.color"] = THEME["muted"]
    plt.rcParams["ytick.color"] = THEME["muted"]
    plt.rcParams["grid.color"] = THEME["grid"]
    plt.rcParams["grid.linestyle"] = "--"
    plt.rcParams["grid.linewidth"] = 0.7
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "KaiTi",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]


def profile_label(profile: str) -> str:
    return PROFILE_LABELS.get(profile, profile)


def pct_fmt(value: float, digits: int = 2) -> str:
    if pd.isna(value):
        return "-"
    return f"{value * 100:.{digits}f}%"


def num_fmt(value: float, digits: int = 2) -> str:
    if pd.isna(value):
        return "-"
    return f"{value:.{digits}f}"


def profile_paths(profile: str) -> dict[str, Path]:
    return {
        "nav": REPORT_DIR / f"v1_{profile}_backtest_nav.csv",
        "metrics": REPORT_DIR / f"v1_{profile}_backtest_metrics.csv",
        "attribution": REPORT_DIR / f"v1_{profile}_attribution_by_sleeve.csv",
        "sleeve_weights": ROOT / "data_processed" / f"v1_{profile}_sleeve_weights_monthly.parquet",
        "asset_weights": ROOT / "data_processed" / f"v1_{profile}_target_weights_monthly.parquet",
        "monthly_attribution": REPORT_DIR / f"v1_{profile}_monthly_attribution.csv",
    }


def read_nav(profile: str) -> pd.DataFrame:
    nav = pd.read_csv(profile_paths(profile)["nav"])
    nav["date"] = pd.to_datetime(nav["date"], errors="coerce")
    nav = nav.dropna(subset=["date"]).sort_values("date")
    nav["drawdown"] = nav["nav"] / nav["nav"].cummax() - 1.0
    nav["pnl"] = nav["nav"] - 1.0
    return nav


def read_sleeve_weights(profile: str) -> pd.DataFrame:
    df = pd.read_parquet(profile_paths(profile)["sleeve_weights"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.dropna(subset=["date"]).sort_values("date")


def read_asset_weights(profile: str) -> pd.DataFrame:
    df = pd.read_parquet(profile_paths(profile)["asset_weights"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.dropna(subset=["date"]).sort_values(["date", "asset_code"])


def profile_cfg(profile: str) -> dict[str, Any]:
    cfg = load_config()
    cfg["product"]["selected_profile"] = profile
    return cfg


def compute_monthly_attribution(profile: str) -> pd.DataFrame:
    cfg = profile_cfg(profile)
    lag = int(cfg["rebalance"]["signal_lag_periods"])
    returns = load_monthly_returns(cfg=cfg)
    returns = returns[["date", "asset_code", "return_1m"]].copy()
    weights = read_asset_weights(profile)

    ret = returns.pivot_table(index="date", columns="asset_code", values="return_1m", aggfunc="last").sort_index()
    ret["cash"] = 0.0
    target_w = weights.pivot_table(index="date", columns="asset_code", values="weight", aggfunc="sum").sort_index().fillna(0.0)
    module_map = weights.drop_duplicates("asset_code").set_index("asset_code")["module"].to_dict()
    module_map["cash"] = "cash"

    cols = sorted(set(ret.columns).union(target_w.columns))
    ret = ret.reindex(columns=cols).fillna(0.0)
    applied_w = target_w.reindex(columns=cols).fillna(0.0).shift(lag).dropna(how="all")
    dates = ret.index.intersection(applied_w.index)
    contrib = applied_w.loc[dates] * ret.loc[dates]

    rows: list[dict[str, Any]] = []
    for date, row in contrib.iterrows():
        by_module: dict[str, float] = {s: 0.0 for s in SLEEVES}
        for asset_code, value in row.items():
            module = module_map.get(asset_code)
            if module is None:
                if abs(float(value)) < 1e-15:
                    continue
                module = "unknown"
            by_module[module] = by_module.get(module, 0.0) + float(value)
        out_row = {"date": date, "profile": profile}
        out_row.update({s: by_module.get(s, 0.0) for s in SLEEVES})
        out_row["gross_return_recomputed"] = sum(by_module.values())
        rows.append(out_row)

    out = pd.DataFrame(rows).sort_values("date")
    out.to_csv(profile_paths(profile)["monthly_attribution"], index=False, encoding="utf-8-sig")
    return out


def metrics_frame(profiles: list[str]) -> pd.DataFrame:
    rows = []
    for profile in profiles:
        metrics = pd.read_csv(profile_paths(profile)["metrics"]).set_index("metric")["value"]
        rows.append(
            {
                "profile": profile,
                "产品": profile_label(profile),
                "累计收益": metrics.get("total_return", np.nan),
                "年化收益": metrics.get("annual_return", np.nan),
                "年化波动": metrics.get("annual_vol", np.nan),
                "Sharpe": metrics.get("sharpe", np.nan),
                "最大回撤": metrics.get("max_drawdown", np.nan),
                "Calmar": metrics.get("calmar", np.nan),
                "月均换手": metrics.get("avg_turnover", np.nan),
            }
        )
    return pd.DataFrame(rows)


def compute_asset_contribution(profile: str) -> pd.DataFrame:
    cfg = profile_cfg(profile)
    lag = int(cfg["rebalance"]["signal_lag_periods"])
    returns = load_monthly_returns(cfg=cfg)
    returns = returns[["date", "asset_code", "return_1m"]].copy()
    weights = read_asset_weights(profile)
    meta = weights.drop_duplicates("asset_code").set_index("asset_code")[["asset_name", "module"]]

    ret = returns.pivot_table(index="date", columns="asset_code", values="return_1m", aggfunc="last").sort_index()
    target_w = weights.pivot_table(index="date", columns="asset_code", values="weight", aggfunc="sum").sort_index().fillna(0.0)
    cols = sorted(set(ret.columns).union(target_w.columns))
    ret = ret.reindex(columns=cols).fillna(0.0)
    applied_w = target_w.reindex(columns=cols).fillna(0.0).shift(lag).dropna(how="all")
    dates = ret.index.intersection(applied_w.index)
    contrib = applied_w.loc[dates] * ret.loc[dates]
    long = contrib.stack().rename("contribution").reset_index()
    long.columns = ["date", "asset_code", "contribution"]
    long = long.merge(meta.reset_index(), how="left", on="asset_code")
    long["asset_name"] = long["asset_name"].fillna(long["asset_code"])
    long["module"] = long["module"].fillna("unknown")
    return long


def set_date_axis(ax: plt.Axes) -> None:
    ax.grid(True, alpha=0.35)
    ax.margins(x=0.01)
    for label in ax.get_xticklabels():
        label.set_rotation(35)
        label.set_ha("right")


def plot_nav_compare(profiles: list[str]) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 7.2), sharex=True, gridspec_kw={"height_ratios": [2.1, 1.0]})
    for profile in profiles:
        nav = read_nav(profile)
        axes[0].plot(nav["date"], nav["nav"], linewidth=2.25, label=profile_label(profile), color=PROFILE_COLORS.get(profile))
        axes[1].fill_between(nav["date"], nav["drawdown"], 0, alpha=0.14, color=PROFILE_COLORS.get(profile))
        axes[1].plot(nav["date"], nav["drawdown"], linewidth=1.6, label=profile_label(profile), color=PROFILE_COLORS.get(profile))
    axes[0].set_title("V1 产品线净值对比", fontsize=14, color=THEME["navy"], weight="bold")
    axes[0].set_ylabel("NAV")
    axes[0].legend(ncol=len(profiles), frameon=False, loc="upper left")
    axes[0].grid(True, alpha=0.35)
    axes[1].set_title("回撤")
    axes[1].set_ylabel("Drawdown")
    axes[1].yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    axes[1].grid(True, alpha=0.35)
    set_date_axis(axes[1])
    fig.tight_layout()
    path = FIG_DIR / "v1_nav_drawdown_compare.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_profile_pnl(profile: str) -> Path:
    nav = read_nav(profile)
    colors = np.where(nav["net_return"] >= 0, THEME["positive"], THEME["negative"])
    fig, axes = plt.subplots(3, 1, figsize=(12.5, 8.6), sharex=True, gridspec_kw={"height_ratios": [2.1, 1.0, 1.0]})
    axes[0].plot(nav["date"], nav["nav"], color=PROFILE_COLORS.get(profile, THEME["navy"]), linewidth=2.25)
    axes[0].set_title(f"{profile_label(profile)} 净值 / 月度PnL / 回撤", fontsize=14, color=THEME["navy"], weight="bold")
    axes[0].set_ylabel("NAV")
    axes[0].grid(True, alpha=0.35)
    axes[1].bar(nav["date"], nav["net_return"], width=22, color=colors, alpha=0.88)
    axes[1].set_ylabel("月度PnL")
    axes[1].yaxis.set_major_formatter(lambda x, _: f"{x:.1%}")
    axes[1].grid(True, axis="y", alpha=0.35)
    axes[2].fill_between(nav["date"], nav["drawdown"], 0, color=THEME["negative"], alpha=0.30)
    axes[2].set_ylabel("回撤")
    axes[2].yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    axes[2].grid(True, alpha=0.35)
    set_date_axis(axes[2])
    fig.tight_layout()
    path = FIG_DIR / f"v1_{profile}_pnl_drawdown.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_sleeve_weights(profile: str) -> Path:
    df = read_sleeve_weights(profile).set_index("date")
    fig, ax = plt.subplots(figsize=(12.8, 5.8))
    ax.stackplot(
        df.index,
        [df[s].astype(float) for s in SLEEVES],
        labels=[SLEEVE_LABELS[s] for s in SLEEVES],
        colors=[SLEEVE_COLORS[s] for s in SLEEVES],
        alpha=0.95,
        step="mid",
    )
    ax.set_title(f"{profile_label(profile)} 大类持仓变化", fontsize=14, color=THEME["navy"], weight="bold")
    ax.set_ylabel("权重")
    ax.yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.set_ylim(0, 1.0)
    ax.grid(True, axis="y", alpha=0.35)
    ax.legend(ncol=6, loc="upper left", bbox_to_anchor=(0, 1.14), frameon=True, fontsize=9)
    set_date_axis(ax)
    fig.tight_layout()
    path = FIG_DIR / f"v1_{profile}_sleeve_weights_stack.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_asset_weights(profile: str, top_n: int = 14) -> Path:
    df = read_asset_weights(profile)
    labels = df.drop_duplicates("asset_code").set_index("asset_code")["asset_name"].to_dict()
    avg = df.groupby("asset_code")["weight"].mean().sort_values(ascending=False)
    keep = list(avg.head(top_n).index)
    df["asset_plot"] = np.where(df["asset_code"].isin(keep), df["asset_code"], "other")
    pivot = df.groupby(["date", "asset_plot"], as_index=False)["weight"].sum().pivot(index="date", columns="asset_plot", values="weight").fillna(0.0)
    ordered = [x for x in keep if x in pivot.columns]
    if "other" in pivot.columns:
        ordered.append("other")
    pivot = pivot.reindex(columns=ordered)
    plot_labels = [labels.get(x, x) if x != "other" else "其他" for x in ordered]
    module_by_asset = df.drop_duplicates("asset_code").set_index("asset_code")["module"].to_dict()
    module_cmaps = {
        "rates": plt.get_cmap("Blues"),
        "credit": plt.get_cmap("Greens"),
        "convertible_bonds": plt.get_cmap("YlOrBr"),
        "equity": plt.get_cmap("Reds"),
        "alternatives": plt.get_cmap("Purples"),
    }
    counts: dict[str, int] = {}
    colors = []
    for asset in ordered:
        if asset == "other":
            colors.append(THEME["gray"])
            continue
        module = module_by_asset.get(asset, "unknown")
        counts[module] = counts.get(module, 0) + 1
        cmap = module_cmaps.get(module, plt.get_cmap("tab20"))
        colors.append(cmap(0.34 + 0.10 * (counts[module] % 5)))

    fig, ax = plt.subplots(figsize=(13.8, 6.4))
    ax.stackplot(pivot.index, [pivot[c].astype(float) for c in ordered], labels=plot_labels, colors=colors, alpha=0.95, step="mid")
    ax.set_title(f"{profile_label(profile)} 资产持仓变化 Top {top_n}", fontsize=14, color=THEME["navy"], weight="bold")
    ax.set_ylabel("权重")
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.grid(True, axis="y", alpha=0.35)
    ax.legend(ncol=5, loc="upper left", bbox_to_anchor=(0, 1.20), frameon=True, fontsize=8)
    set_date_axis(ax)
    fig.tight_layout()
    path = FIG_DIR / f"v1_{profile}_asset_weights_stack.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_turnover(profile: str) -> Path:
    nav = read_nav(profile)
    fig, ax = plt.subplots(figsize=(12.5, 4.2))
    ax.bar(nav["date"], nav["turnover"], width=22, color=THEME["gray"], alpha=0.72, label="换手")
    ax.plot(nav["date"], nav["transaction_cost"], color=THEME["negative"], linewidth=1.8, label="交易成本")
    ax.set_title(f"{profile_label(profile)} 换手与交易成本", fontsize=13, color=THEME["navy"], weight="bold")
    ax.yaxis.set_major_formatter(lambda x, _: f"{x:.1%}")
    ax.grid(True, axis="y", alpha=0.35)
    ax.legend(frameon=False)
    set_date_axis(ax)
    fig.tight_layout()
    path = FIG_DIR / f"v1_{profile}_turnover_cost.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_profile_attribution(profile: str) -> Path:
    attr = compute_monthly_attribution(profile)
    total = attr[SLEEVES].sum().sort_values(ascending=True)
    colors = [SLEEVE_COLORS.get(x, "#777777") for x in total.index]
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.3), gridspec_kw={"width_ratios": [1.0, 1.6]})
    axes[0].barh([SLEEVE_LABELS.get(x, x) for x in total.index], total.values, color=colors)
    axes[0].set_title(f"{profile_label(profile)} 总归因", color=THEME["navy"], weight="bold")
    axes[0].xaxis.set_major_formatter(lambda x, _: f"{x:.1%}")
    axes[0].grid(True, axis="x", alpha=0.35)
    xmax = max(abs(total.min()), abs(total.max()), 0.01)
    axes[0].set_xlim(-xmax * 1.25, xmax * 1.25)
    for i, value in enumerate(total.values):
        axes[0].text(
            value + (0.015 * xmax if value >= 0 else -0.015 * xmax),
            i,
            f"{value:.1%}",
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=9,
            color=THEME["text"],
        )

    cumulative = attr.set_index("date")[SLEEVES].cumsum()
    for sleeve in SLEEVES:
        axes[1].plot(cumulative.index, cumulative[sleeve], label=SLEEVE_LABELS[sleeve], color=SLEEVE_COLORS[sleeve], linewidth=1.8)
    axes[1].set_title("分大类累计贡献")
    axes[1].yaxis.set_major_formatter(lambda x, _: f"{x:.1%}")
    axes[1].grid(True, alpha=0.35)
    axes[1].legend(ncol=3, frameon=False, fontsize=9)
    set_date_axis(axes[1])
    fig.tight_layout()
    path = FIG_DIR / f"v1_{profile}_attribution.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_attribution_compare(profiles: list[str]) -> Path:
    data = []
    for profile in profiles:
        attr = compute_monthly_attribution(profile)
        row = {"profile": profile}
        row.update(attr[SLEEVES].sum().to_dict())
        data.append(row)
    df = pd.DataFrame(data).set_index("profile").reindex(columns=SLEEVES)

    fig, ax = plt.subplots(figsize=(12.5, 5.2))
    x = np.arange(len(df.index))
    bottom = np.zeros(len(df.index))
    for sleeve in SLEEVES:
        values = df[sleeve].to_numpy(dtype=float)
        ax.bar(x, values, bottom=bottom, label=SLEEVE_LABELS[sleeve], color=SLEEVE_COLORS[sleeve])
        bottom += values
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([profile_label(x) for x in df.index])
    ax.set_title("V1 产品线归因对比", fontsize=14, color=THEME["navy"], weight="bold")
    ax.yaxis.set_major_formatter(lambda y, _: f"{y:.1%}")
    ax.grid(True, axis="y", alpha=0.35)
    ax.legend(ncol=6, frameon=False)
    fig.tight_layout()
    path = FIG_DIR / "v1_attribution_compare.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_metrics_table(profiles: list[str]) -> Path:
    metrics = metrics_frame(profiles)
    display = metrics[["产品", "累计收益", "年化收益", "年化波动", "Sharpe", "最大回撤", "Calmar", "月均换手"]].copy()
    for col in ["累计收益", "年化收益", "年化波动", "最大回撤", "月均换手"]:
        display[col] = display[col].map(lambda x: pct_fmt(x, 2))
    for col in ["Sharpe", "Calmar"]:
        display[col] = display[col].map(lambda x: num_fmt(x, 2))

    fig, ax = plt.subplots(figsize=(12.5, 2.9))
    ax.axis("off")
    table = ax.table(
        cellText=display.values,
        colLabels=display.columns.tolist(),
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1.0, 1.35)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#C7D2DE")
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_facecolor(THEME["light_blue"])
            cell.set_text_props(weight="bold", color=THEME["navy"])
        elif row % 2 == 1:
            cell.set_facecolor("#F7FAFC")
        else:
            cell.set_facecolor("white")
        if col == 0 and row > 0:
            cell.set_text_props(weight="bold", color=THEME["text"])
    ax.set_title("V1 三产品绩效概览", fontsize=14, color=THEME["navy"], weight="bold", pad=12)
    fig.tight_layout()
    path = FIG_DIR / "v1_metrics_table.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_benchmark_nav_compare(profiles: list[str]) -> Path | None:
    path_csv = REPORT_DIR / "v1_static_benchmark_nav.csv"
    if not path_csv.exists():
        return None
    df = pd.read_csv(path_csv)
    if "date" not in df.columns:
        return None
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    if {"portfolio", "nav"}.issubset(df.columns):
        df = df.pivot_table(index="date", columns="portfolio", values="nav", aggfunc="last").reset_index()
    prefer = [f"v1_{p}" for p in profiles] + ["pure_bond_core", "pure_bond_30y_plus", "stock_bond_20_80", "static_fip"]
    keep = [c for c in prefer if c in df.columns]
    if not keep:
        return None
    label_map = {
        **{f"v1_{p}": profile_label(p) for p in profiles},
        "pure_bond_core": "纯债核心",
        "pure_bond_30y_plus": "纯债+30Y增强",
        "stock_bond_20_80": "股债28",
        "static_fip": "静态固收+",
    }
    colors = {
        **{f"v1_{p}": PROFILE_COLORS.get(p, "#7B3294") for p in profiles},
        "pure_bond_core": "#7A869A",
        "pure_bond_30y_plus": "#5B8C5A",
        "stock_bond_20_80": "#8E6C8A",
        "static_fip": "#A66A3F",
    }
    fig, ax = plt.subplots(figsize=(12.8, 5.4))
    for col in keep:
        lw = 2.25 if col.startswith("v1_") else 1.45
        alpha = 1.0 if col.startswith("v1_") else 0.72
        ax.plot(df["date"], df[col], label=label_map.get(col, col), color=colors.get(col), linewidth=lw, alpha=alpha)
    ax.set_title("V1 与静态基准净值对比", fontsize=14, color=THEME["navy"], weight="bold")
    ax.set_ylabel("NAV")
    ax.legend(ncol=4, frameon=False, loc="upper left")
    set_date_axis(ax)
    fig.tight_layout()
    out = FIG_DIR / "v1_benchmark_nav_compare_pro.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_asset_contribution(profile: str, top_n: int = 12) -> Path:
    contrib = compute_asset_contribution(profile)
    total = contrib.groupby(["asset_code", "asset_name", "module"], as_index=False)["contribution"].sum()
    total = total.sort_values("contribution", ascending=False)
    top_pos = total.head(top_n // 2)
    top_neg = total.tail(top_n // 2).sort_values("contribution")
    show = pd.concat([top_neg, top_pos], ignore_index=True).drop_duplicates("asset_code")
    show = show.sort_values("contribution")
    colors = [SLEEVE_COLORS.get(m, THEME["gray"]) if v >= 0 else "#B8C0CC" for m, v in zip(show["module"], show["contribution"])]

    fig, ax = plt.subplots(figsize=(12.2, 5.4))
    y = np.arange(len(show))
    ax.barh(y, show["contribution"], color=colors, height=0.58)
    ax.set_yticks(y)
    ax.set_yticklabels(show["asset_name"])
    ax.set_title(f"{profile_label(profile)} 资产累计贡献 Top {len(show)}", fontsize=14, color=THEME["navy"], weight="bold")
    ax.set_xlabel("累计贡献")
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:.1%}")
    ax.grid(True, axis="x", alpha=0.35)
    xmax = max(abs(show["contribution"].min()), abs(show["contribution"].max()), 0.01)
    ax.set_xlim(-xmax * 1.35, xmax * 1.35)
    for i, value in enumerate(show["contribution"]):
        ax.text(
            value + (0.012 * xmax if value >= 0 else -0.012 * xmax),
            i,
            f"{value:.1%}",
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=9,
            color=THEME["text"],
        )
    fig.tight_layout()
    path = FIG_DIR / f"v1_{profile}_asset_contribution_top.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_yearly_sleeve_contribution(profile: str) -> Path:
    attr = compute_monthly_attribution(profile)
    attr["year"] = pd.to_datetime(attr["date"]).dt.year
    yearly = attr.groupby("year")[SLEEVES].sum()
    fig, ax = plt.subplots(figsize=(12.5, 5.2))
    x = np.arange(len(yearly.index))
    pos_bottom = np.zeros(len(yearly.index))
    neg_bottom = np.zeros(len(yearly.index))
    for sleeve in SLEEVES:
        values = yearly[sleeve].to_numpy(dtype=float)
        bottom = np.where(values >= 0, pos_bottom, neg_bottom)
        ax.bar(x, values, bottom=bottom, label=SLEEVE_LABELS[sleeve], color=SLEEVE_COLORS[sleeve], width=0.62)
        pos_bottom += np.where(values >= 0, values, 0.0)
        neg_bottom += np.where(values < 0, values, 0.0)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(yearly.index.astype(str))
    ax.set_title(f"{profile_label(profile)} 年度大类贡献", fontsize=14, color=THEME["navy"], weight="bold")
    ax.yaxis.set_major_formatter(lambda y, _: f"{y:.1%}")
    ax.grid(True, axis="y", alpha=0.35)
    ax.legend(ncol=6, frameon=False, loc="upper left")
    fig.tight_layout()
    path = FIG_DIR / f"v1_{profile}_yearly_sleeve_contribution.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def metrics_table_html(profiles: list[str]) -> str:
    df = metrics_frame(profiles)
    cards = []
    for _, row in df.iterrows():
        cards.append(
            f"""
            <div class="metric-card">
              <div class="metric-title">{row['产品']}</div>
              <div class="metric-main">{pct_fmt(row['年化收益'])}</div>
              <div class="metric-sub">年化收益 / Sharpe {num_fmt(row['Sharpe'])}</div>
              <div class="metric-grid">
                <span>波动</span><b>{pct_fmt(row['年化波动'])}</b>
                <span>回撤</span><b>{pct_fmt(row['最大回撤'])}</b>
                <span>换手</span><b>{pct_fmt(row['月均换手'])}</b>
              </div>
            </div>
            """
        )
    table = df[["产品", "累计收益", "年化收益", "年化波动", "Sharpe", "最大回撤", "Calmar", "月均换手"]].copy()
    for col in ["累计收益", "年化收益", "年化波动", "最大回撤", "月均换手"]:
        table[col] = table[col].map(lambda x: pct_fmt(x, 2))
    for col in ["Sharpe", "Calmar"]:
        table[col] = table[col].map(lambda x: num_fmt(x, 2))
    return f"<div class='metric-cards'>{''.join(cards)}</div>{table.to_html(index=False, escape=False, classes='data-table')}"


def write_html_report(profiles: list[str], figures: dict[str, list[Path]]) -> Path:
    def img(path: Path) -> str:
        rel = path.relative_to(REPORT_DIR).as_posix() if path.is_relative_to(REPORT_DIR) else ("../" + path.relative_to(ROOT / "output").as_posix())
        return f'<figure><img src="{rel}" alt="{path.name}"></figure>'

    sections = [metrics_table_html(profiles)]
    sections.append("<h2>产品线总览</h2>")
    sections.append("<div class='figure-grid'>")
    sections.extend(img(p) for p in figures["common"])
    sections.append("</div>")
    for profile in profiles:
        sections.append(f"<h2>{profile_label(profile)}</h2>")
        sections.append("<div class='figure-grid'>")
        sections.extend(img(p) for p in figures[profile])
        sections.append("</div>")

    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>V1 固收+动态配置可视化报告</title>
  <style>
    :root {{
      --navy: #1F3A5F;
      --blue: #2C7FB8;
      --light-blue: #D9EAF7;
      --text: #243447;
      --muted: #6B7280;
      --line: #D6DEE8;
      --paper: #FFFFFF;
      --bg: #F4F7FB;
    }}
    body {{
      font-family: "Microsoft YaHei", Arial, sans-serif;
      margin: 0;
      color: var(--text);
      background: var(--bg);
    }}
    .page {{
      max-width: 1440px;
      margin: 0 auto;
      padding: 30px 34px 46px;
    }}
    .hero {{
      background: linear-gradient(135deg, #1F3A5F 0%, #2C7FB8 100%);
      color: white;
      border-radius: 6px;
      padding: 24px 28px;
      box-shadow: 0 10px 30px rgba(31, 58, 95, 0.18);
    }}
    h1 {{ margin: 0 0 8px; font-size: 28px; letter-spacing: 0; }}
    h2 {{
      margin: 30px 0 16px;
      color: var(--navy);
      font-size: 20px;
      border-bottom: 2px solid var(--line);
      padding-bottom: 9px;
    }}
    .note {{ color: rgba(255,255,255,0.82); margin: 0; }}
    .metric-cards {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 14px;
      margin: 20px 0 16px;
    }}
    .metric-card {{
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 16px 18px;
      box-shadow: 0 2px 8px rgba(31, 58, 95, 0.06);
    }}
    .metric-title {{ color: var(--muted); font-weight: 700; }}
    .metric-main {{ color: var(--navy); font-size: 30px; font-weight: 800; margin-top: 8px; }}
    .metric-sub {{ color: var(--muted); font-size: 13px; margin: 4px 0 12px; }}
    .metric-grid {{
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 6px 12px;
      font-size: 13px;
      color: var(--muted);
    }}
    .metric-grid b {{ color: var(--text); }}
    table.data-table {{
      width: 100%;
      border-collapse: collapse;
      background: white;
      margin: 14px 0 24px;
      border: 1px solid var(--line);
      box-shadow: 0 2px 8px rgba(31, 58, 95, 0.05);
    }}
    .data-table th, .data-table td {{
      border-bottom: 1px solid var(--line);
      padding: 10px 12px;
      text-align: right;
      white-space: nowrap;
    }}
    .data-table th {{
      background: var(--light-blue);
      color: var(--navy);
      font-weight: 800;
    }}
    .data-table th:first-child, .data-table td:first-child {{ text-align: left; font-weight: 700; }}
    .data-table tr:nth-child(even) td {{ background: #F8FBFE; }}
    .figure-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 18px;
      align-items: start;
    }}
    figure {{
      margin: 0;
      background: white;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px;
      box-shadow: 0 2px 8px rgba(31, 58, 95, 0.06);
    }}
    img {{ width: 100%; display: block; }}
    @media (max-width: 980px) {{
      .metric-cards, .figure-grid {{ grid-template-columns: 1fr; }}
      .page {{ padding: 18px; }}
    }}
  </style>
</head>
<body>
  <main class="page">
  <section class="hero">
    <h1>V1 固收+动态配置可视化报告</h1>
    <p class="note">包含 PnL/净值回撤、持仓变化、换手成本、资产与大类归因。数据截止以 v1 当前回测产物为准。</p>
  </section>
  {''.join(sections)}
  </main>
</body>
</html>
"""
    path = REPORT_DIR / "v1_visual_report.html"
    path.write_text(html, encoding="utf-8")
    return path


def parse_profiles(value: str) -> list[str]:
    if value == "all":
        return PROFILES
    return [x.strip() for x in value.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate V1 PnL, holdings and attribution visual reports.")
    parser.add_argument("--profiles", default="all", help="all, or comma-separated profiles: low_vol,balanced,high_vol")
    parser.add_argument("--top-assets", type=int, default=14, help="Top assets shown in asset-level holding stack.")
    args = parser.parse_args()

    profiles = parse_profiles(args.profiles)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    set_plot_style()

    common: list[Path] = [plot_metrics_table(profiles), plot_nav_compare(profiles), plot_attribution_compare(profiles)]
    benchmark_fig = plot_benchmark_nav_compare(profiles)
    if benchmark_fig is not None:
        common.insert(2, benchmark_fig)
    figures: dict[str, list[Path]] = {"common": common}
    for profile in profiles:
        figures[profile] = [
            plot_profile_pnl(profile),
            plot_sleeve_weights(profile),
            plot_asset_weights(profile, args.top_assets),
            plot_asset_contribution(profile),
            plot_yearly_sleeve_contribution(profile),
            plot_turnover(profile),
            plot_profile_attribution(profile),
        ]

    report = write_html_report(profiles, figures)
    manifest = {
        "profiles": profiles,
        "figures": {key: [str(p.relative_to(ROOT)) for p in paths] for key, paths in figures.items()},
        "monthly_attribution": [str(profile_paths(p)["monthly_attribution"].relative_to(ROOT)) for p in profiles],
        "html_report": str(report.relative_to(ROOT)),
    }
    manifest_path = REPORT_DIR / "v1_visual_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.itransformer_ppo_hybrid import cfg_path, load_hybrid_config


FIGURE_DIR = ROOT / "output" / "figures" / "v1"
PROFILES = ["low_vol", "balanced", "high_vol"]
NAMES = {"low_vol": "低波", "balanced": "中波", "high_vol": "高波"}
COLORS = {"low_vol": "#1D4ED8", "balanced": "#059669", "high_vol": "#DC2626"}


def setup_style() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["axes.edgecolor"] = "#CBD5E1"
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.color"] = "#E2E8F0"
    plt.rcParams["grid.alpha"] = 0.75


def load_nav(cfg: dict, profile: str, hybrid: bool) -> pd.DataFrame:
    if hybrid:
        path = ROOT / str(cfg["outputs"]["nav_pattern"]).format(profile=profile)
    else:
        path = ROOT / str(cfg["data"]["baseline_nav_pattern"]).format(profile=profile)
    frame = pd.read_csv(path, parse_dates=["date"])
    frame["rebased_nav"] = (1.0 + frame["net_return"].astype(float)).cumprod()
    return frame


def plot_nav(cfg: dict) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharex=True)
    for ax, profile in zip(axes, PROFILES):
        base = load_nav(cfg, profile, False)
        hybrid = load_nav(cfg, profile, True)
        ax.plot(base["date"], base["rebased_nav"], color="#64748B", linewidth=1.8, label="风险预算基线")
        ax.plot(hybrid["date"], hybrid["rebased_nav"], color=COLORS[profile], linewidth=2.2, label="iTransformer + PPO")
        ax.set_title(f"{NAMES[profile]}产品", fontsize=13, fontweight="bold")
        ax.legend(frameon=False, fontsize=9)
        ax.tick_params(axis="x", rotation=30)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("三产品净值：正式基线 vs iTransformer V2 + PPO", fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = FIGURE_DIR / f"{cfg['outputs']['figure_prefix']}_nav_compare.png"
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_incremental(cfg: dict) -> Path:
    fig, ax = plt.subplots(figsize=(13, 5.2))
    for profile in PROFILES:
        base = load_nav(cfg, profile, False).set_index("date")["rebased_nav"]
        hybrid = load_nav(cfg, profile, True).set_index("date")["rebased_nav"]
        dates = base.index.intersection(hybrid.index)
        ratio = hybrid.loc[dates] / base.loc[dates]
        start = dates[dates >= pd.Timestamp("2022-02-01")]
        ratio = ratio.loc[start] / ratio.loc[start].iloc[0]
        ax.plot(ratio.index, ratio, color=COLORS[profile], linewidth=2.0, label=NAMES[profile])
    ax.axhline(1.0, color="#475569", linewidth=1.0, linestyle="--")
    ax.set_title("iTransformer + PPO 相对基线增量净值（2022-02=1）", fontsize=15, fontweight="bold")
    ax.set_ylabel("增量净值")
    ax.legend(frameon=False, ncol=3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = FIGURE_DIR / f"{cfg['outputs']['figure_prefix']}_incremental_nav.png"
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize a configured iTransformer V2 + PPO overlay.")
    parser.add_argument("--config", default=None, help="Optional hybrid YAML config.")
    args = parser.parse_args()
    cfg = load_hybrid_config(Path(args.config) if args.config else None)
    report = cfg_path(cfg, "visual_report")
    setup_style()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    nav_path = plot_nav(cfg)
    incremental_path = plot_incremental(cfg)
    summary = pd.read_csv(cfg_path(cfg, "summary"))
    table = summary[summary["scope"].eq("full")][
        ["profile", "baseline_annual_return", "hybrid_annual_return", "baseline_sharpe", "hybrid_sharpe", "sharpe_delta"]
    ].copy()
    table["profile"] = table["profile"].map(NAMES)
    for col in ["baseline_annual_return", "hybrid_annual_return"]:
        table[col] = table[col].map(lambda value: f"{value:.2%}")
    for col in ["baseline_sharpe", "hybrid_sharpe", "sharpe_delta"]:
        table[col] = table[col].map(lambda value: f"{value:.3f}")
    table.columns = ["产品", "基线年化", "联合年化", "基线Sharpe", "联合Sharpe", "Sharpe增量"]
    rel_nav = Path(os.path.relpath(nav_path, report.parent)).as_posix()
    rel_incremental = Path(os.path.relpath(incremental_path, report.parent)).as_posix()
    html = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>iTransformer + PPO 联合择时</title>
<style>body{{font-family:'Microsoft YaHei',Arial,sans-serif;margin:0;background:#f8fafc;color:#0f172a}}main{{max-width:1500px;margin:auto;padding:32px}}h1{{font-size:28px}}p{{color:#475569}}section{{background:white;border:1px solid #e2e8f0;margin:18px 0;padding:22px;border-radius:6px}}img{{width:100%;display:block}}table{{border-collapse:collapse;width:100%}}th,td{{padding:10px 12px;border-bottom:1px solid #e2e8f0;text-align:right}}th:first-child,td:first-child{{text-align:left}}th{{background:#f1f5f9}}</style></head>
<body><main><h1>iTransformer V2 + PPO 联合择时</h1><p>联合版本与风险预算基线并行保留；所有动作均为月末决策、下月执行。</p>
<section>{table.to_html(index=False, border=0, escape=False)}</section>
<section><img src="{rel_nav}" alt="净值对比"></section>
<section><img src="{rel_incremental}" alt="增量净值"></section>
</main></body></html>"""
    report.write_text(html, encoding="utf-8")
    print(json.dumps({"report": str(report.relative_to(ROOT)), "figures": [str(nav_path.relative_to(ROOT)), str(incremental_path.relative_to(ROOT))]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.v1_strategy import (  # noqa: E402
    V1Paths,
    build_asset_weights,
    build_risk_budget_base,
    build_sleeve_weights,
    load_config,
    run_backtest,
)


PROFILES = ["low_vol", "balanced", "high_vol"]
CASES: dict[str, dict[str, Any]] = {
    "asset_cov_off": {"enabled": False},
    "asset_cov_rienet_b25": {"enabled": True, "blend": 0.25},
    "asset_cov_rienet_b35": {"enabled": True, "blend": 0.35},
    "asset_cov_rienet_b50": {"enabled": True, "blend": 0.50},
}


def outputs(case: str, profile: str) -> dict[str, str]:
    prefix = f"v1_{case}_{profile}"
    return {
        "risk_budget_base": f"data_processed/{prefix}_risk_budget_base_monthly.parquet",
        "sleeve_weights": f"data_processed/{prefix}_sleeve_weights_monthly.parquet",
        "weights": f"data_processed/{prefix}_target_weights_monthly.parquet",
        "nav": f"output/reports/{prefix}_backtest_nav.csv",
        "metrics": f"output/reports/{prefix}_backtest_metrics.csv",
        "attribution": f"output/reports/{prefix}_attribution_by_sleeve.csv",
        "robustness": f"output/reports/{prefix}_robustness_summary.csv",
        "data_quality": f"output/reports/{prefix}_data_quality.csv",
        "ep_posterior": f"data_processed/{prefix}_ep_posterior_monthly.parquet",
        "ep_signals": f"output/reports/{prefix}_ep_signals.csv",
        "optimizer_diagnostics": f"output/reports/{prefix}_optimizer_diagnostics.csv",
    }


def run_case(case: str, overlay_updates: dict[str, Any], profile: str) -> dict[str, Any]:
    cfg = copy.deepcopy(load_config())
    cfg["strategy_version"] = "no_agent"
    cfg["product"]["selected_profile"] = profile
    cfg["outputs"] = outputs(case, profile)
    overlay = cfg.setdefault("signal_overlays", {}).setdefault("asset_covariance_cleaner", {})
    overlay.update(overlay_updates)
    paths = V1Paths()
    build_risk_budget_base(cfg, paths)
    build_sleeve_weights(cfg, paths)
    build_asset_weights(cfg, paths)
    nav, metrics = run_backtest(cfg, paths)
    m = metrics.set_index("metric")["value"].to_dict()
    out = {
        "case": case,
        "profile": profile,
        "months": int(m.get("months", 0)),
        "annual_return": float(m.get("annual_return", float("nan"))),
        "annual_vol": float(m.get("annual_vol", float("nan"))),
        "sharpe": float(m.get("sharpe", float("nan"))),
        "max_drawdown": float(m.get("max_drawdown", float("nan"))),
        "calmar": float(m.get("calmar", float("nan"))),
        "avg_turnover": float(m.get("avg_turnover", float("nan"))),
        "latest_nav": float(nav["nav"].iloc[-1]) if not nav.empty else float("nan"),
    }
    print(json.dumps(out, ensure_ascii=False), flush=True)
    return out


def write_summary(rows: list[dict[str, Any]]) -> None:
    summary = pd.DataFrame(rows)
    base = summary[summary["case"].eq("asset_cov_off")].set_index("profile")
    out = []
    for _, row in summary.iterrows():
        b = base.loc[row["profile"]]
        data = row.to_dict()
        data["sharpe_delta_vs_off"] = row["sharpe"] - b["sharpe"]
        data["annual_return_delta_vs_off"] = row["annual_return"] - b["annual_return"]
        data["max_drawdown_delta_vs_off"] = row["max_drawdown"] - b["max_drawdown"]
        out.append(data)
    comp = pd.DataFrame(out)
    csv_path = ROOT / "output" / "reports" / "v1_asset_covariance_overlay_comparison.csv"
    comp.to_csv(csv_path, index=False, encoding="utf-8-sig")

    lines = [
        "# V1 资产层 RIEnet-inspired 协方差 Overlay 对比",
        "",
        "本实验不改变五大袖套总权重，不改变 EP views，只在每个袖套内部用过去可见收益估计清洗协方差，生成内部 GMV 权重并与原资产权重 blend。",
        "",
        "## 当前排序",
        "",
    ]
    for profile in PROFILES:
        sub = comp[comp["profile"].eq(profile)].sort_values("sharpe", ascending=False)
        for i, (_, row) in enumerate(sub.iterrows(), 1):
            lines.append(
                f"{i}. `{profile}` `{row['case']}`：Sharpe={row['sharpe']:.4f}，"
                f"年化收益={row['annual_return']:.4%}，最大回撤={row['max_drawdown']:.4%}，"
                f"相对关闭 Sharpe={row['sharpe_delta_vs_off']:+.4f}。"
            )
        lines.append("")
    md_path = ROOT / "output" / "reports" / "v1_asset_covariance_overlay_comparison.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(csv_path, flush=True)
    print(md_path, flush=True)


def main() -> None:
    rows = []
    for case, updates in CASES.items():
        for profile in PROFILES:
            rows.append(run_case(case, updates, profile))
    write_summary(rows)


if __name__ == "__main__":
    main()

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
METHODS: dict[str, dict[str, Any]] = {
    "sample_shrink": {
        "method": "sample_shrink",
        "shrinkage_to_diag": 0.20,
    },
    "ewma_shrink": {
        "method": "ewma_shrink",
        "shrinkage_to_diag": 0.25,
        "half_life_months": 12,
    },
    "rienet_inspired": {
        "method": "rienet_inspired",
        "rienet_cleaner": {
            "half_life_months": 18,
            "winsorize_z": 4.0,
            "eigen_shrinkage": 0.45,
            "vol_shrinkage_to_median": 0.20,
            "max_condition_number": 25.0,
            "min_eigenvalue": 0.0001,
        },
    },
    "rienet_more_stable": {
        "method": "rienet_inspired",
        "rienet_cleaner": {
            "half_life_months": 24,
            "winsorize_z": 3.5,
            "eigen_shrinkage": 0.60,
            "vol_shrinkage_to_median": 0.30,
            "max_condition_number": 18.0,
            "min_eigenvalue": 0.0001,
        },
    },
}


def versioned_outputs(method: str, profile: str) -> dict[str, str]:
    prefix = f"v1_cov_{method}_{profile}"
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


def run_one(method: str, method_cfg: dict[str, Any], profile: str) -> dict[str, Any]:
    cfg = copy.deepcopy(load_config())
    cfg["strategy_version"] = "no_agent"
    cfg["product"]["selected_profile"] = profile
    cfg["outputs"] = versioned_outputs(method, profile)
    cfg.setdefault("ep_skfolio", {})["covariance_estimator"] = method_cfg

    paths = V1Paths()
    build_risk_budget_base(cfg, paths)
    build_sleeve_weights(cfg, paths)
    build_asset_weights(cfg, paths)
    nav, metrics = run_backtest(cfg, paths)

    m = metrics.set_index("metric")["value"].to_dict()
    diag_path = ROOT / cfg["outputs"]["optimizer_diagnostics"]
    diag = pd.read_csv(diag_path) if diag_path.exists() else pd.DataFrame()
    out = {
        "method": method,
        "profile": profile,
        "months": int(m.get("months", 0)),
        "annual_return": float(m.get("annual_return", float("nan"))),
        "annual_vol": float(m.get("annual_vol", float("nan"))),
        "sharpe": float(m.get("sharpe", float("nan"))),
        "max_drawdown": float(m.get("max_drawdown", float("nan"))),
        "calmar": float(m.get("calmar", float("nan"))),
        "avg_turnover": float(m.get("avg_turnover", float("nan"))),
        "latest_nav": float(nav["nav"].iloc[-1]) if not nav.empty else float("nan"),
        "avg_active_deviation": float(pd.to_numeric(diag.get("opt_active_deviation"), errors="coerce").mean()) if not diag.empty else float("nan"),
        "avg_sleeve_turnover": float(pd.to_numeric(diag.get("opt_sleeve_turnover"), errors="coerce").mean()) if not diag.empty else float("nan"),
        "avg_portfolio_vol_monthly": float(pd.to_numeric(diag.get("opt_portfolio_vol_monthly"), errors="coerce").mean()) if not diag.empty else float("nan"),
    }
    print(json.dumps(out, ensure_ascii=False), flush=True)
    return out


def write_notes(summary: pd.DataFrame) -> None:
    base = summary[summary["method"].eq("sample_shrink")].set_index("profile")
    rows = []
    for _, row in summary.iterrows():
        b = base.loc[row["profile"]]
        rows.append(
            {
                **row.to_dict(),
                "sharpe_delta_vs_sample": row["sharpe"] - b["sharpe"],
                "max_drawdown_delta_vs_sample": row["max_drawdown"] - b["max_drawdown"],
                "annual_return_delta_vs_sample": row["annual_return"] - b["annual_return"],
            }
        )
    comp = pd.DataFrame(rows)
    comp_path = ROOT / "output" / "reports" / "v1_covariance_estimator_comparison.csv"
    comp.to_csv(comp_path, index=False, encoding="utf-8-sig")

    lines = [
        "# V1 协方差估计器对比",
        "",
        "本实验只替换 EP/skfolio 优化器使用的协方差估计器，不改变 MarketSkill/Agentic views、EP posterior、产品约束和交易成本。",
        "",
        "## 方法",
        "",
        "- `sample_shrink`：历史默认口径，滚动样本协方差向对角阵收缩。",
        "- `ewma_shrink`：指数加权协方差，再向对角阵收缩。",
        "- `rienet_inspired`：RIEnet 思路的轻量协方差清洗：lag 加权、相关矩阵特征值清洗、边际波动收缩、PSD 修复。",
        "- `rienet_more_stable`：更强特征值和波动收缩的稳健版本。",
        "",
        "## 结果文件",
        "",
        f"- `{comp_path.relative_to(ROOT)}`",
        "",
        "## 当前排序",
        "",
    ]
    ranking = comp.sort_values(["profile", "sharpe"], ascending=[True, False])
    for profile in PROFILES:
        sub = ranking[ranking["profile"].eq(profile)]
        if sub.empty:
            continue
        best = sub.iloc[0]
        lines.append(
            f"- `{profile}`：Sharpe 最优 `{best['method']}`，Sharpe={best['sharpe']:.4f}，"
            f"年化收益={best['annual_return']:.4%}，最大回撤={best['max_drawdown']:.4%}。"
        )
    (ROOT / "output" / "reports" / "v1_covariance_estimator_comparison.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    results = []
    for method, method_cfg in METHODS.items():
        for profile in PROFILES:
            results.append(run_one(method, method_cfg, profile))
    summary = pd.DataFrame(results)
    write_notes(summary)
    print("\n=== done ===", flush=True)
    print("output/reports/v1_covariance_estimator_comparison.csv", flush=True)
    print("output/reports/v1_covariance_estimator_comparison.md", flush=True)


if __name__ == "__main__":
    main()

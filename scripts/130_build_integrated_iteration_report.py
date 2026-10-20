from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "output" / "reports"

ITERATIONS = {
    "v2_baseline": "integrated_daily_view_ppo_v2_baseline_summary.csv",
    "v3_horizon_normalized": "integrated_daily_view_ppo_v3_horizon_normalized_summary.csv",
    "v4_dual_teacher": "integrated_daily_view_ppo_v4_dual_teacher_summary.csv",
    "v5_product_base": "integrated_daily_view_ppo_v5_product_base_summary.csv",
    "v6_reduced_action": "integrated_daily_view_ppo_v6_reduced_action_summary.csv",
    "v7_sleeve_reliability": "integrated_daily_view_ppo_v7_sleeve_reliability_summary.csv",
}

DECISIONS = {
    "v2_baseline": ("rejected", "Initial common action space; retained only as the starting benchmark."),
    "v3_horizon_normalized": ("rejected", "Horizon-normalized teacher weakened balanced and high-vol results."),
    "v4_dual_teacher": ("rejected", "Dual teachers added complexity without robust three-product improvement."),
    "v5_product_base": ("rejected", "Product-specific fallback helped high-vol but did not sufficiently improve low/balanced."),
    "v6_reduced_action": ("selected", "Best overall product separation and Sharpe balance; current deployed challenger."),
    "v7_sleeve_reliability": ("rejected", "Sleeve reliability added 30 states and runtime but did not improve aggregate results."),
}

METRIC_COLUMNS = [
    "strategy_annual_return",
    "strategy_annual_vol",
    "strategy_sharpe",
    "strategy_max_drawdown",
    "delta_annual_return_vs_formal",
    "delta_sharpe_vs_formal",
    "delta_annual_return_vs_rule_only",
    "delta_sharpe_vs_rule_only",
    "mean_strategy_turnover",
]


def performance_stats(returns: pd.Series) -> dict[str, float]:
    values = pd.to_numeric(returns, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    nav = (1.0 + values).cumprod()
    annual_return = float(nav.iloc[-1] ** (252.0 / len(nav)) - 1.0)
    annual_vol = float(values.std(ddof=0) * np.sqrt(252.0))
    drawdown = nav / nav.cummax() - 1.0
    return {
        "days": int(len(values)),
        "annual_return": annual_return,
        "annual_vol": annual_vol,
        "sharpe": annual_return / annual_vol if annual_vol > 1e-12 else np.nan,
        "max_drawdown": float(drawdown.min()),
    }


def build_annual_robustness() -> tuple[pd.DataFrame, pd.DataFrame]:
    annual_rows: list[dict[str, object]] = []
    action_rows: list[dict[str, object]] = []
    for profile in ["low_vol", "balanced", "high_vol"]:
        daily = pd.read_csv(REPORT_DIR / f"integrated_daily_view_ppo_{profile}_daily.csv")
        daily["date"] = pd.to_datetime(daily["date"])
        daily = daily.loc[daily["date"].ge("2024-01-01")].copy()
        for year, group in daily.groupby(daily["date"].dt.year):
            stats = {
                name: performance_stats(group[f"{name}_return"])
                for name in ["formal", "rule_only", "strategy"]
            }
            row: dict[str, object] = {"profile": profile, "year": int(year)}
            for name, values in stats.items():
                row.update({f"{name}_{key}": value for key, value in values.items()})
            row["delta_sharpe_vs_formal"] = stats["strategy"]["sharpe"] - stats["formal"]["sharpe"]
            row["delta_sharpe_vs_rule_only"] = stats["strategy"]["sharpe"] - stats["rule_only"]["sharpe"]
            annual_rows.append(row)

        decisions = pd.read_parquet(ROOT / "data_processed" / f"integrated_daily_view_ppo_{profile}_decisions.parquet")
        decisions["date"] = pd.to_datetime(decisions["date"])
        decisions = decisions.loc[decisions["date"].ge("2024-01-01")].copy()
        counts = decisions.groupby([decisions["date"].dt.year.rename("year"), "action_name"]).size()
        for (year, action), count in counts.items():
            action_rows.append({"profile": profile, "year": int(year), "action_name": action, "count": int(count)})
    return pd.DataFrame(annual_rows), pd.DataFrame(action_rows)


def main() -> None:
    rows: list[pd.DataFrame] = []
    for version, filename in ITERATIONS.items():
        frame = pd.read_csv(REPORT_DIR / filename)
        frame = frame.loc[frame["split"].eq("oos_2024_onward"), ["profile", *METRIC_COLUMNS]].copy()
        frame.insert(0, "version", version)
        frame["decision"], frame["decision_reason"] = DECISIONS[version]
        rows.append(frame)

    comparison = pd.concat(rows, ignore_index=True)
    profile_order = pd.CategoricalDtype(["low_vol", "balanced", "high_vol"], ordered=True)
    comparison["profile"] = comparison["profile"].astype(profile_order)
    comparison = comparison.sort_values(["version", "profile"]).reset_index(drop=True)

    current = pd.read_csv(REPORT_DIR / "integrated_daily_view_ppo_three_profile_summary.csv")
    current = current.loc[current["split"].eq("oos_2024_onward"), ["profile", *METRIC_COLUMNS]].copy()
    selected = comparison.loc[comparison["version"].eq("v6_reduced_action"), ["profile", *METRIC_COLUMNS]].copy()
    check = current.merge(selected, on="profile", suffixes=("_current", "_v6"))
    max_difference = max(
        float((check[f"{column}_current"] - check[f"{column}_v6"]).abs().max())
        for column in METRIC_COLUMNS
    )
    if max_difference > 1e-12:
        raise RuntimeError(f"Current results do not match selected v6; max difference={max_difference:.3e}")

    csv_path = REPORT_DIR / "integrated_daily_view_ppo_iteration_comparison.csv"
    comparison.to_csv(csv_path, index=False, encoding="utf-8-sig")
    annual, action_counts = build_annual_robustness()
    annual_path = REPORT_DIR / "integrated_daily_view_ppo_annual_robustness.csv"
    action_path = REPORT_DIR / "integrated_daily_view_ppo_annual_action_counts.csv"
    annual.to_csv(annual_path, index=False, encoding="utf-8-sig")
    action_counts.to_csv(action_path, index=False, encoding="utf-8-sig")

    selected_rows = comparison.loc[comparison["version"].eq("v6_reduced_action")].copy()
    display = selected_rows[
        [
            "profile",
            "strategy_annual_return",
            "strategy_annual_vol",
            "strategy_sharpe",
            "strategy_max_drawdown",
            "delta_sharpe_vs_formal",
            "delta_sharpe_vs_rule_only",
        ]
    ].copy()
    for column in ["strategy_annual_return", "strategy_annual_vol", "strategy_max_drawdown"]:
        display[column] = display[column].map(lambda value: f"{value:.3%}")
    for column in ["strategy_sharpe", "delta_sharpe_vs_formal", "delta_sharpe_vs_rule_only"]:
        display[column] = display[column].map(lambda value: f"{value:.4f}")

    sharpe_pivot = comparison.pivot(index="version", columns="profile", values="strategy_sharpe")
    annual_display = annual[
        ["profile", "year", "strategy_annual_return", "strategy_sharpe", "strategy_max_drawdown", "delta_sharpe_vs_formal", "delta_sharpe_vs_rule_only"]
    ].copy()
    for column in ["strategy_annual_return", "strategy_max_drawdown"]:
        annual_display[column] = annual_display[column].map(lambda value: f"{value:.3%}")
    for column in ["strategy_sharpe", "delta_sharpe_vs_formal", "delta_sharpe_vs_rule_only"]:
        annual_display[column] = annual_display[column].map(lambda value: f"{value:.4f}")
    notes = [
        "# 日频 iTransformer + PPO 迭代记录",
        "",
        "## 结论",
        "",
        "当前保留 `v6_reduced_action`。它不是单产品指标最高的参数拼接，而是在同一组事前规则下兼顾三产品风险分层、Sharpe、回撤、可解释性和运行复杂度的综合选择。",
        "",
        "低波和中波的策略 Sharpe 已同时超过正式锚点与固定规则对照；高波超过正式锚点，且回撤略优于固定规则，但 Sharpe 仍比固定规则低约 0.009。因此该链路仍是研究挑战者，不能表述为已经全面战胜控制组。",
        "",
        "## 当前 v6 结果",
        "",
        display.to_markdown(index=False),
        "",
        "## 各轮 Sharpe",
        "",
        sharpe_pivot.to_markdown(floatfmt=".4f"),
        "",
        "## 分年度稳健性",
        "",
        annual_display.to_markdown(index=False),
        "",
        "2026 仅包含截至 2026-04-16 的部分年度，年化结果只用于诊断。逐年动作计数保存在 `integrated_daily_view_ppo_annual_action_counts.csv`。",
        "",
        "## 参数试验判断",
        "",
        "- `v3`：把不同持有期教师收益年化归一后，排序噪声被放大，未带来稳健改善。",
        "- `v4`：双教师增加模型选择自由度，但样本量不足以支持额外复杂度。",
        "- `v5`：产品化默认动作方向正确，高波应以规则控制组为默认基座。",
        "- `v6`：低波/中波移除裸规则 PPO 动作，高波保留规则默认；只有验证优势、家族概率和概率差同时过门槛才切换。",
        "- `v7`：增加袖套级时序可靠度后状态维度和耗时上升，但三产品综合结果下降，已回退。",
        "- 块级置信门诊断：把年度验证段切成三个连续时间块后，高波 2025 在两个种子上仍显示正的块级下界，但随后真实 2025 仍落后规则。这说明问题包含状态迁移，不能靠事后提高验证阈值可靠修复，因此未把该门加入正式 v6。",
        "",
        "## 防止过拟合的边界",
        "",
        "这些版本都在已经被观察过的 2024+ 区间比较，属于研究诊断，不是新的独立样本外证明。后续不应继续围绕该区间网格搜索；更可信的下一步是冻结 v6，进行逐日纸面跟踪、滚动仿真或等待新增数据。",
        "逐年结果也显示 2025 并非全面占优，因此后续新增模块必须先定义参数和门槛，再在新增日期上检验，不能用 2025 的已知损失日反推规则。",
        "",
        f"当前结果与保存的 v6 最大数值差为 `{max_difference:.3e}`。",
    ]
    notes_path = REPORT_DIR / "integrated_daily_view_ppo_iteration_notes.md"
    notes_path.write_text("\n".join(notes), encoding="utf-8")
    print(
        {
            "comparison": str(csv_path),
            "annual_robustness": str(annual_path),
            "annual_actions": str(action_path),
            "notes": str(notes_path),
            "max_difference": max_difference,
        }
    )


if __name__ == "__main__":
    main()

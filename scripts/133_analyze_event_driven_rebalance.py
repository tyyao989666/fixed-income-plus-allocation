from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.adaptive_daily_ppo import perf_stats  # noqa: E402


REPORTS = ROOT / "output" / "reports"
PROFILES = ["low_vol", "balanced", "high_vol"]


def moving_block_indices(length: int, block: int, rng: np.random.Generator) -> np.ndarray:
    starts = rng.integers(0, max(length - block + 1, 1), size=int(np.ceil(length / block)))
    return np.concatenate([np.arange(start, min(start + block, length)) for start in starts])[:length]


def bootstrap_profile(profile: str, draws: int = 2000, block: int = 20) -> dict[str, float | str]:
    daily = pd.read_csv(
        REPORTS / f"event_driven_rebalance_ppo_v2_{profile}_daily.csv", parse_dates=["date"]
    )
    daily = daily.loc[daily["date"].ge("2024-01-01")].reset_index(drop=True)
    event = daily["strategy_return"].to_numpy(dtype=float)
    v6 = daily["v6_return"].to_numpy(dtype=float)
    rng = np.random.default_rng(20260716 + PROFILES.index(profile))
    return_delta: list[float] = []
    sharpe_delta: list[float] = []
    for _ in range(draws):
        indices = moving_block_indices(len(daily), block, rng)
        event_stats = perf_stats(pd.Series(event[indices]))
        v6_stats = perf_stats(pd.Series(v6[indices]))
        return_delta.append(event_stats["annual_return"] - v6_stats["annual_return"])
        sharpe_delta.append(event_stats["sharpe"] - v6_stats["sharpe"])
    return_array = np.asarray(return_delta)
    sharpe_array = np.asarray(sharpe_delta)
    return {
        "profile": profile,
        "draws": draws,
        "block_trading_days": block,
        "annual_return_delta_p2_5": float(np.quantile(return_array, 0.025)),
        "annual_return_delta_median": float(np.median(return_array)),
        "annual_return_delta_p97_5": float(np.quantile(return_array, 0.975)),
        "probability_annual_return_delta_positive": float(np.mean(return_array > 0.0)),
        "sharpe_delta_p2_5": float(np.quantile(sharpe_array, 0.025)),
        "sharpe_delta_median": float(np.median(sharpe_array)),
        "sharpe_delta_p97_5": float(np.quantile(sharpe_array, 0.975)),
        "probability_sharpe_delta_positive": float(np.mean(sharpe_array > 0.0)),
    }


def annual_rows(profile: str) -> list[dict[str, float | int | str]]:
    daily = pd.read_csv(
        REPORTS / f"event_driven_rebalance_ppo_v2_{profile}_daily.csv", parse_dates=["date"]
    )
    daily = daily.loc[daily["date"].ge("2024-01-01")].copy()
    rows: list[dict[str, float | int | str]] = []
    for year, group in daily.groupby(daily["date"].dt.year):
        event = perf_stats(group["strategy_return"])
        v6 = perf_stats(group["v6_return"])
        mechanical = perf_stats(group["mechanical_return"])
        rows.append(
            {
                "profile": profile,
                "year": int(year),
                "days": int(event["days"]),
                "event_annual_return": event["annual_return"],
                "event_sharpe": event["sharpe"],
                "event_max_drawdown": event["max_drawdown"],
                "v6_annual_return": v6["annual_return"],
                "v6_sharpe": v6["sharpe"],
                "delta_annual_return_vs_v6": event["annual_return"] - v6["annual_return"],
                "delta_sharpe_vs_v6": event["sharpe"] - v6["sharpe"],
                "delta_sharpe_vs_mechanical": event["sharpe"] - mechanical["sharpe"],
            }
        )
    return rows


def main() -> None:
    v1 = pd.read_csv(REPORTS / "event_driven_rebalance_ppo_three_profile_summary.csv")
    v2 = pd.read_csv(REPORTS / "event_driven_rebalance_ppo_v2_three_profile_summary.csv")
    comparison_rows: list[dict[str, object]] = []
    for version, frame in [("v1_family_options", v1), ("v2_unified_target", v2)]:
        selected = frame.loc[frame["split"].eq("oos_2024_onward")]
        for _, row in selected.iterrows():
            comparison_rows.append(
                {
                    "version": version,
                    "profile": row["profile"],
                    "annual_return": row["strategy_annual_return"],
                    "annual_vol": row["strategy_annual_vol"],
                    "sharpe": row["strategy_sharpe"],
                    "max_drawdown": row["strategy_max_drawdown"],
                    "v6_annual_return": row["v6_annual_return"],
                    "v6_sharpe": row["v6_sharpe"],
                    "delta_annual_return_vs_v6": row["delta_annual_return_vs_v6"],
                    "delta_sharpe_vs_v6": row["delta_sharpe_vs_v6"],
                    "delta_sharpe_vs_mechanical": row["delta_sharpe_vs_mechanical"],
                    "mean_turnover": row["mean_strategy_turnover"],
                }
            )
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(
        REPORTS / "event_driven_rebalance_validation_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    bootstrap = pd.DataFrame([bootstrap_profile(profile) for profile in PROFILES])
    bootstrap.to_csv(
        REPORTS / "event_driven_rebalance_bootstrap.csv", index=False, encoding="utf-8-sig"
    )
    annual = pd.DataFrame([row for profile in PROFILES for row in annual_rows(profile)])
    annual.to_csv(
        REPORTS / "event_driven_rebalance_annual_robustness.csv",
        index=False,
        encoding="utf-8-sig",
    )

    interval = pd.read_csv(REPORTS / "event_driven_rebalance_ppo_v2_interval_summary.csv")
    diagnostic_rows: list[dict[str, object]] = []
    policy_interval_rows: list[dict[str, object]] = []
    for profile in PROFILES:
        decisions = pd.read_parquet(
            ROOT / "data_processed" / f"event_driven_rebalance_ppo_v2_{profile}_decisions.parquet"
        )
        diagnostic_rows.append(
            {
                "profile": profile,
                "decision_days": len(decisions),
                "event_ppo_days": int(decisions["policy_source"].eq("event_ppo").sum()),
                "rebalance_days": int(decisions["rebalanced"].sum()),
                "prediction_target_days": int(decisions["horizon_weights"].ne("{}").sum()),
                "positive_itransformer_level_days": int(
                    pd.to_numeric(decisions["effective_model_level"], errors="coerce").fillna(0.0).gt(0.0).sum()
                ),
                "max_hold_age": int(decisions["hold_age_before"].max()),
            }
        )
        rebalances = decisions.loc[
            decisions["rebalanced"].astype(bool),
            ["policy_source", "hold_age_before"],
        ].copy()
        rebalances["realized_interval"] = rebalances["hold_age_before"] + 1
        for source, group in rebalances.groupby("policy_source"):
            values = group["realized_interval"].astype(float)
            policy_interval_rows.append(
                {
                    "profile": profile,
                    "policy_source": source,
                    "rebalance_count": len(values),
                    "mean_interval": values.mean(),
                    "median_interval": values.median(),
                    "p75_interval": values.quantile(0.75),
                    "p90_interval": values.quantile(0.90),
                    "max_interval": values.max(),
                    "unique_intervals": values.nunique(),
                }
            )
    diagnostics = pd.DataFrame(diagnostic_rows).merge(interval, on="profile", how="left")
    diagnostics.to_csv(
        REPORTS / "event_driven_rebalance_diagnostics.csv", index=False, encoding="utf-8-sig"
    )
    policy_intervals = pd.DataFrame(policy_interval_rows)
    policy_intervals.to_csv(
        REPORTS / "event_driven_rebalance_policy_source_intervals.csv",
        index=False,
        encoding="utf-8-sig",
    )
    learned_intervals = policy_intervals.loc[
        policy_intervals["policy_source"].eq("event_ppo")
    ].copy()

    v2_display = comparison.loc[comparison["version"].eq("v2_unified_target")].copy()
    report = [
        "# 真正自适应调仓验证结论",
        "",
        "## 结论先行",
        "",
        "事件驱动架构已经实现：系统每天决定继续持有或终止旧仓，没有 1/5/20/60 日固定持有期；5/20/60 只作为 iTransformer 多尺度预测输入。技术审计全部通过，但当前样本上没有超过 v6，因此不能替换正式挑战者。",
        "",
        "v1 将终止与四个目标家族一起学习，最终目标选择坍缩为 hold/formal/rules，预测目标没有实际落仓。v2 改为 EP/skfolio 每日生成唯一的 iTransformer 增强目标，PPO 只学习二元终止，解决了目标家族坍缩，但三产品 2024+ 的收益和 Sharpe 仍低于 v6。",
        "",
        "## v2 对比 v6",
        "",
        v2_display[
            [
                "profile",
                "annual_return",
                "annual_vol",
                "sharpe",
                "max_drawdown",
                "delta_annual_return_vs_v6",
                "delta_sharpe_vs_v6",
                "delta_sharpe_vs_mechanical",
                "mean_turnover",
            ]
        ].to_markdown(index=False, floatfmt=".6f"),
        "",
        "## 调仓确实自适应",
        "",
        diagnostics.to_markdown(index=False, floatfmt=".4f"),
        "",
        "调仓间隔最短为 1 日，最长达到 129/136/224 日，且每个产品有 19–26 种不同间隔。不存在最大持有期，长间隔来自连续 hold。",
        "",
        "### 学习策略与兜底分解",
        "",
        policy_intervals.to_markdown(index=False, floatfmt=".4f"),
        "",
        "最长持有阶段内，PPO 确实连续选择 hold，因此并非机械月历伪装出来的长间隔。但只看由 `event_ppo` 触发的再平衡，三个产品的中位间隔都是 1 日，75% 分位仅为 1/2/1 日。当前策略表现为长时间 hold 与若干连续调仓簇并存；它在定义上是真正事件驱动，在经济质量上仍有明显的动作迟滞与簇集问题。",
        "",
        "## 配对时间块 Bootstrap",
        "",
        bootstrap.to_markdown(index=False, floatfmt=".6f"),
        "",
        "Bootstrap 使用 20 个交易日连续块、2,000 次抽样，保持局部自相关。它只是对已观察 2024+ 路径的不确定性诊断，不会把该区间重新变成独立样本外。",
        "",
        "## 分年度表现",
        "",
        annual.to_markdown(index=False, floatfmt=".6f"),
        "",
        "2026 仅到 4 月中旬，年化值只用于诊断。",
        "",
        "## 为什么没有超过 v6",
        "",
        "1. 可学习的独立路径只有约 4.3 年，真正调仓事件仅 220 次左右；对 135 维状态和神经终止边界而言样本仍然太少。",
        "2. v2 的统一预测目标本身在中高波上弱于 v6，终止策略只能改善进入时点，无法把较弱目标变成更强目标。",
        "3. 低波 PPO 相对同目标机械阈值有小幅正增益，但不足以抵消目标路径与 v6 的差距；中高波终止策略反而降低机械目标 Sharpe。",
        "4. 交易成本确实形成无交易区间，但 PPO 触发后的调仓存在簇集，说明目标变化与终止概率尚未被持续时间模型充分平滑。",
        "",
        "## 决策",
        "",
        "- 正式保留 v6，不把事件驱动版本包装成提升。",
        "- v2 作为研究分支保留，因为它已完整解决固定持有期问题，并通过时点审计。",
        "- 下一次有效验证应使用新增日期纸面跟踪；在已观察的 2024+ 上继续网格搜索阈值、epoch 或奖励系数会形成明显过拟合。",
        "- 若继续研究，优先扩充更长的 Point-in-Time 日频状态历史，并把终止学习改成低维生存/危险率模型或离线 RL 置信下界，而不是继续加深网络。",
    ]
    (REPORTS / "event_driven_rebalance_final_assessment.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    print(
        {
            "comparison_rows": len(comparison),
            "bootstrap_rows": len(bootstrap),
            "annual_rows": len(annual),
            "report": str(REPORTS / "event_driven_rebalance_final_assessment.md"),
        }
    )


if __name__ == "__main__":
    main()

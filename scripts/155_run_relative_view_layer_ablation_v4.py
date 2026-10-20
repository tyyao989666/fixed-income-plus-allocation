from __future__ import annotations

import copy
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.adaptive_daily_ppo import perf_stats
from src.robust_execution_ppo import (
    ROOT,
    RobustExecutionEnvironment,
    _run_comparator,
    build_state_frame,
    load_config,
)


CONFIG = ROOT / "config" / "view_execution_ppo_v4_config.yaml"
SUMMARY_OUTPUT = (
    ROOT / "output" / "reports" / "view_execution_ppo_v4_layer_ablation.csv"
)
DAILY_OUTPUT = (
    ROOT
    / "output"
    / "reports"
    / "view_execution_ppo_v4_layer_ablation_daily.csv"
)
REPORT_OUTPUT = (
    ROOT
    / "output"
    / "reports"
    / "view_execution_ppo_v4_layer_ablation.md"
)


def _strategy_stats(
    daily: pd.DataFrame,
    column: str,
    turnover_column: str,
    mask: pd.Series,
) -> dict[str, float]:
    stats = perf_stats(daily.loc[mask, column])
    return {
        "annual_return": float(stats["annual_return"]),
        "annual_vol": float(stats["annual_vol"]),
        "sharpe": float(stats["sharpe"]),
        "max_drawdown": float(stats["max_drawdown"]),
        "calmar": float(stats["calmar"]),
        "mean_turnover": float(daily.loc[mask, turnover_column].mean()),
    }


def main() -> None:
    cfg, base_cfg = load_config(CONFIG)
    state = build_state_frame(cfg, base_cfg, write_output=False)
    all_daily: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []

    for profile in cfg["profiles"]:
        model_env = RobustExecutionEnvironment(profile, state, cfg, base_cfg)
        mechanical_base_cfg = copy.deepcopy(base_cfg)
        mechanical_base_cfg["actions"]["max_itransformer_weight"] = 0.0
        mechanical_env = RobustExecutionEnvironment(
            profile, state, cfg, mechanical_base_cfg
        )

        mechanical = _run_comparator(mechanical_env, "threshold").rename(
            columns={
                "threshold_return": "mechanical_ep_return",
                "threshold_turnover": "mechanical_ep_turnover",
                "threshold_cost": "mechanical_ep_cost",
            }
        )
        model = _run_comparator(model_env, "threshold").rename(
            columns={
                "threshold_return": "itransformer_ep_return",
                "threshold_turnover": "itransformer_ep_turnover",
                "threshold_cost": "itransformer_ep_cost",
            }
        )
        production_path = (
            ROOT
            / str(cfg["outputs"]["daily_pattern"]).format(profile=profile)
        )
        production = pd.read_csv(production_path)
        production["date"] = pd.to_datetime(production["date"], errors="coerce")
        production = production[
            [
                "date",
                "formal_return",
                "strategy_return",
                "strategy_turnover",
                "strategy_cost",
            ]
        ].rename(
            columns={
                "formal_return": "formal_anchor_return",
                "strategy_return": "ppo_execution_return",
                "strategy_turnover": "ppo_execution_turnover",
                "strategy_cost": "ppo_execution_cost",
            }
        )

        daily = (
            production.merge(mechanical, on="date", how="inner")
            .merge(model, on="date", how="inner")
            .sort_values("date")
            .reset_index(drop=True)
        )
        daily["profile"] = profile
        daily["formal_anchor_turnover"] = 0.0
        for name in [
            "formal_anchor",
            "mechanical_ep",
            "itransformer_ep",
            "ppo_execution",
        ]:
            daily[f"{name}_nav"] = (1.0 + daily[f"{name}_return"]).cumprod()
        all_daily.append(daily)

        splits = {
            "full_history": pd.Series(True, index=daily.index),
            "walk_forward": daily["date"].ge(
                pd.Timestamp(cfg["data"]["deployment_start"])
            ),
            "oos_2024_onward": daily["date"].ge(
                pd.Timestamp(cfg["data"]["evaluation_start"])
            ),
        }
        strategy_columns = {
            "formal_anchor": (
                "formal_anchor_return",
                "formal_anchor_turnover",
            ),
            "mechanical_ep": (
                "mechanical_ep_return",
                "mechanical_ep_turnover",
            ),
            "itransformer_ep": (
                "itransformer_ep_return",
                "itransformer_ep_turnover",
            ),
            "ppo_execution": (
                "ppo_execution_return",
                "ppo_execution_turnover",
            ),
        }
        for split, mask in splits.items():
            stats = {
                name: _strategy_stats(daily, return_col, turnover_col, mask)
                for name, (return_col, turnover_col) in strategy_columns.items()
            }
            row: dict[str, object] = {
                "profile": profile,
                "split": split,
                "days": int(mask.sum()),
            }
            for name, values in stats.items():
                row.update(
                    {f"{name}_{metric}": value for metric, value in values.items()}
                )
            row.update(
                {
                    "mechanical_return_edge_vs_formal": (
                        stats["mechanical_ep"]["annual_return"]
                        - stats["formal_anchor"]["annual_return"]
                    ),
                    "mechanical_sharpe_edge_vs_formal": (
                        stats["mechanical_ep"]["sharpe"]
                        - stats["formal_anchor"]["sharpe"]
                    ),
                    "itransformer_return_edge_vs_mechanical": (
                        stats["itransformer_ep"]["annual_return"]
                        - stats["mechanical_ep"]["annual_return"]
                    ),
                    "itransformer_sharpe_edge_vs_mechanical": (
                        stats["itransformer_ep"]["sharpe"]
                        - stats["mechanical_ep"]["sharpe"]
                    ),
                    "ppo_return_edge_vs_itransformer": (
                        stats["ppo_execution"]["annual_return"]
                        - stats["itransformer_ep"]["annual_return"]
                    ),
                    "ppo_sharpe_edge_vs_itransformer": (
                        stats["ppo_execution"]["sharpe"]
                        - stats["itransformer_ep"]["sharpe"]
                    ),
                }
            )
            summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    combined_daily = pd.concat(all_daily, ignore_index=True)
    SUMMARY_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_OUTPUT, index=False, encoding="utf-8-sig")
    combined_daily.to_csv(DAILY_OUTPUT, index=False, encoding="utf-8-sig")

    oos = summary.loc[summary["split"].eq("oos_2024_onward")].copy()
    report_columns = [
        "profile",
        "formal_anchor_annual_return",
        "formal_anchor_sharpe",
        "mechanical_return_edge_vs_formal",
        "mechanical_sharpe_edge_vs_formal",
        "itransformer_return_edge_vs_mechanical",
        "itransformer_sharpe_edge_vs_mechanical",
        "ppo_return_edge_vs_itransformer",
        "ppo_sharpe_edge_vs_itransformer",
    ]
    REPORT_OUTPUT.write_text(
        "\n".join(
            [
                "# V4 分层消融结果",
                "",
                "同一日期、T+1、交易成本和资产约束下，依次比较：",
                "",
                "1. 当前正式锚点；",
                "2. 机械 Market View + EP/skfolio + 阈值执行；",
                "3. 在第 2 层加入因果 iTransformer 残差观点；",
                "4. 在第 3 层加入通过年度门控的 PPO 执行器。",
                "",
                "## 2024 年以来",
                "",
                oos[report_columns].to_markdown(index=False),
                "",
                "正值代表该层相对上一层改善，负值代表该层拖累。"
            ]
        ),
        encoding="utf-8",
    )
    print(oos[report_columns].to_string(index=False))
    print(f"summary={SUMMARY_OUTPUT}")
    print(f"daily={DAILY_OUTPUT}")
    print(f"report={REPORT_OUTPUT}")


if __name__ == "__main__":
    main()

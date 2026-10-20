"""Generate the three-layer article draft and auditable data draft for v43."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/reports"
VERSION = "hierarchical_asset_itransformer_v43_clean_calendar"


def read_yaml(path: str) -> dict:
    with (ROOT / path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def percent(value: float) -> str:
    return f"{value:.2%}"


def asset_table() -> pd.DataFrame:
    universe = pd.read_parquet(ROOT / "data_processed/asset_universe.parquet")
    returns = pd.read_parquet(ROOT / "data_processed/asset_returns_daily.parquet")
    returns["date"] = pd.to_datetime(returns["date"], errors="coerce")
    returns["return_1d"] = pd.to_numeric(returns["return_1d"], errors="coerce")
    coverage = (
        returns.groupby("asset_code", as_index=False)
        .agg(
            actual_start=("date", "min"),
            actual_end=("date", "max"),
            observations=("date", "size"),
            return_missing_rate=("return_1d", lambda value: float(value.isna().mean())),
        )
    )
    out = universe.merge(coverage, on="asset_code", how="left")
    out["proxy_class"] = "direct_or_fund_price"
    out.loc[out["source_type"].astype(str).str.contains("underlying_index|wealth_index"), "proxy_class"] = "index_proxy"
    out.loc[out["source_type"].astype(str).str.contains("curve_proxy"), "proxy_class"] = "curve_duration_proxy"
    out.loc[out["source_type"].astype(str).str.contains("derived_style_bucket"), "proxy_class"] = "point_in_time_research_bucket"
    out.loc[out["asset_code"].eq("rate_30y"), "proxy_class"] = "long_duration_treasury_proxy"
    out.loc[out["asset_code"].eq("alt_reits"), "proxy_class"] = "equal_weight_reits_research_index"
    out.loc[out["asset_code"].eq("alt_us_bond"), "proxy_class"] = "us10y_curve_duration_proxy"
    return out.sort_values(["module", "asset_code"]).reset_index(drop=True)


def markdown_asset_table(frame: pd.DataFrame) -> str:
    headers = ["袖套", "代码", "名称", "来源/代码", "代理类型", "实际起止", "缺失率", "重要说明"]
    rows = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for _, row in frame.iterrows():
        source = str(row.get("data_source", ""))
        code = str(row.get("ts_code", ""))
        source_label = f"{source}<br>{code}" if code and code != "nan" else source
        start = pd.Timestamp(row["actual_start"]).date() if pd.notna(row["actual_start"]) else ""
        end = pd.Timestamp(row["actual_end"]).date() if pd.notna(row["actual_end"]) else ""
        note = str(row.get("note", "")).replace("|", "/")
        rows.append(
            "| " + " | ".join(
                [
                    str(row["module"]),
                    str(row["asset_code"]),
                    str(row["name"]),
                    source_label,
                    str(row["proxy_class"]),
                    f"{start} 至 {end}",
                    percent(float(row["return_missing_rate"])),
                    note,
                ]
            ) + " |"
        )
    return "\n".join(rows)


def build_article_draft(summary: pd.DataFrame, baselines: pd.DataFrame, gate: dict, model_gate: dict) -> str:
    profile_rows = summary.loc[summary["split"].eq("oos_2024_onward")].copy()
    result_rows = []
    for _, row in profile_rows.iterrows():
        result_rows.append(
            f"| {row['profile']} | {percent(row['formal_annual_return'])} | {row['formal_sharpe']:.2f} | "
            f"{percent(row['strategy_annual_return'])} | {row['strategy_sharpe']:.2f} | "
            f"{row['delta_sharpe_vs_formal']:+.3f} | {percent(row['strategy_max_drawdown'])} |"
        )
    baseline_rows = []
    for _, row in baselines.iterrows():
        baseline_rows.append(
            f"| {row['profile']} | {row['strategy']} | {percent(row['annual_return'])} | "
            f"{row['sharpe']:.2f} | {percent(row['max_drawdown'])} | {row['annualized_turnover'] if 'annualized_turnover' in row else row['mean_daily_turnover'] * 252:.2f} |"
        )
    active = model_gate.get("portfolio_consumed_active_components", [])
    active_text = "、".join(
        f"{item['year']} {item['module']} {item['horizon_days']}日" for item in active
    ) or "无"
    return f"""# 固收+多资产动态配置：三层框架文章底稿

## 写作定位

本文不是把深度学习、优化和强化学习并列展示，而是用控制变量说明每一层到底解决什么问题：先用风险平价建立不依赖预测的底仓，再用可解释的 Market View 与 EP/skfolio 在边界内进行主动配置，最后用 iTransformer 与 PPO 做受门控的局部相对观点和日频调仓时机。所有比较共享同一 38 资产池、同一交易日、T+1 执行、无现金约束和 5bp 换手成本。

正式数据决策区间锁定为 2018-01-02 至 2026-07-15；组合样本外评价重点为 2024-01-02 至 2026-07-15。本文不把基础价格文件较晚的观测误写为完整最新状态。

```mermaid
flowchart TD
  D["38资产日频收益与特征"] --> A0["第一层 A0: 分层风险平价"]
  D --> S["状态仪表盘 + 股债九宫格 + 38张资产卡"]
  S --> A1["第二层 A1: 机械 Market View"]
  A0 --> E["EP/skfolio: 协方差、风险预算、上下限"]
  A1 --> E
  S --> IT["第三层: iTransformer 相对观点"]
  IT --> G["年度walk-forward + 多种子因果门控"]
  G --> E
  E --> T["每日目标组合"]
  T --> P["PPO: 是否调仓及调多少"]
  P --> R["T+1回测、5bp成本、Bootstrap"]
```

## 第一章：第一层，分层风险平价底仓（A0）

### 1.1 要解决的问题

在没有任何方向预测时，固收+首先必须保证风险来源不过度集中。A0 将资产先按利率、信用、转债、权益、另类五个袖套分组，在袖套之间按照产品风险预算做风险平价，在袖套内部按逆波动分散。它不使用 Market View、iTransformer 或 PPO，因此是所有后续增量的干净对照。

### 1.2 输入、规则和输出

| 环节 | 输入 | 处理 | 输出 |
|---|---|---|---|
| 袖套风险预算 | 五袖套历史日收益、产品风险贡献目标、袖套上下限 | 252日滚动协方差；迭代风险贡献求解 | 月度袖套权重 |
| 袖套内配置 | 该袖套资产历史波动 | 逆波动归一，不预测收益 | 月度资产权重 |
| 交易回放 | 月末权重、下一交易日收益 | T+1 生效，5bp x 换手扣成本 | A0 日收益、净值、权重和风险指标 |

### 1.3 产品层风险预算

低波优先信用与利率，平衡提高权益和另类，高波明显提高权益与另类的风险贡献，同时保留利率的分散化作用。现金目标均为零；风险预算不是固定名义权重，而是在协方差与上下限下被映射成可执行权重。

### 1.4 A0 应如何写结果

不要把 A0 写成“收益预测策略”。它的成功标准是：低波回撤最小、产品风险从低到高递进、长期净值平稳、换手低且不依赖未来数据。A0 的低波 Sharpe 可能高于主动版本，这通常反映更低的风险暴露，而不是主动配置无效。

## 第二章：第二层，Market View 与 EP/skfolio 配置层（A1）

### 2.1 要解决的问题

风险平价只知道风险，不知道当前利率、信用、权益、转债和另类机会是否不同。A1 在不破坏风险预算的前提下，加入一套可解释的、机械化的 Market View。观点不是仓位指令，而是对期望收益排序的受限修正。

### 2.2 状态信息的三层组织

1. **总仪表盘**：利率、信用、转债、权益、另类、风险预算和流动性压力。
2. **股债九宫格**：债券强/中/弱与权益强/中/弱的 3x3 环境，只做跨资产背景与条件变量，不直接输出单资产仓位。
3. **38 张资产卡**：每个资产的收益、动量、波动、回撤、可得估值/利差/宏观代理与缺失掩码。

### 2.3 A1 的流程

| 环节 | 输入 | 规则 | 输出 |
|---|---|---|---|
| Market View | 当日状态卡、九宫格、资产卡 | 机械规则生成五袖套相对观点与置信度 | 利率/信用/转债/权益/另类观点 |
| EP/skfolio | A0 风险预算、历史收益、收缩协方差、观点、产品上下限 | 观点只修正均值；协方差和边界主导风险 | 目标组合 |
| 执行 | 月末决策与下一交易日收益 | T+1，5bp 成本，无现金 | A1 日收益、净值、权重 |

### 2.4 关键写作原则

EP/skfolio 不负责预测，也不让高风险资产因为“观点强”就无限加仓。它的任务是把观点转换成受约束的、可解释的目标权重。A1 是最终策略的正式锚点，也是第三层必须超越的直接对照。

## 第三章：第三层，iTransformer 与 PPO 自适应层（A2）

### 3.1 iTransformer 的职责：预测相对机会，而不是价格方向

模型读取过去 120 个交易日的局部资产特征、宏观特征、状态卡、九宫格与机械观点特征，预测 5/20/60 日的横截面相对平均收益。训练标签从决策日下一交易日开始，协方差不进入预测目标。模型输出被视为对 A1 的**增量相对观点**，不是独立的全局交易信号。

### 3.2 防泄露与防过拟合

- 按年 walk-forward，预测某年只使用该年开始前已完全成熟的标签。
- 每年三随机种子；掩码历史预训练后进行监督训练，最长 1000 epoch，但由验证早停和 checkpoint 选择决定实际使用轮数。
- 不通过因果门控的资产/期限/年份，观点严格回退为零，不允许模型凭弱预测改变仓位。
- 本次真正进入投资组合的有效组件为：{active_text}。

### 3.3 PPO 的职责：日频择时执行

PPO 不直接选择资产，不创造方向，不改变风险预算。它在每天观察当前持仓、目标组合距离、状态卡、预测置信度、风险与交易成本后，选择继续持有或向目标组合移动 25%/50%/100%。因此它回答的是“今天是否调仓、调多少”，而不是“买什么”。

### 3.4 正式门控

预测层可以进入组合研究：`enabled_for_portfolio_stage={model_gate.get('enabled_for_portfolio_stage')}`；但本轮完整正式策略门控为：`enabled_as_formal={gate.get('enabled_as_formal')}`。未正式放行的原因是低波产品的 Bootstrap Sharpe 优势概率未达到预设阈值。文章中应将该版本表述为严格评估下的挑战者，不应写成全产品正式替代。

## 第四章：控制变量结果与可写结论

### 4.1 A0 与 A1 基线

| 产品 | 版本 | 年化收益 | Sharpe | 最大回撤 | 年化换手 |
|---|---|---:|---:|---:|---:|
{chr(10).join(baseline_rows)}

### 4.2 A1 正式锚点与 A2 最终策略，2024 年以来

| 产品 | A1年化收益 | A1 Sharpe | A2年化收益 | A2 Sharpe | A2相对A1 Sharpe | A2最大回撤 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(result_rows)}

### 4.3 建议的结论表述

1. A0 提供稳定、可解释的风险预算底仓，但不承担主动获取相对机会的任务。
2. A1 的 Market View + EP/skfolio 是目前主要主动配置来源：它提高了中高波产品的绝对收益，同时付出合理的波动与回撤代价。
3. A2 的 iTransformer 只在严格因果门控允许的少数模块、年份和期限提供增量观点；PPO 只优化调仓时机。因此，A2 的收益改善是小幅、受约束且必须经 Bootstrap 复核的。
4. 中波和高波的 A2 相对 A1 有更强的正向证据；低波的改善不够稳健，正式使用仍应保留 A1 锚点。

## 附录：读者复现清单

- 使用 `config/hierarchical_asset_itransformer_v43_clean_calendar.yaml` 固定 38 资产、交易日锚定和最终决策日。
- 先运行 A0/A1 基线，再运行 iTransformer、因果探针和 PPO，不得只展示最终版本。
- 每个比较必须保持相同资产池、日期、T+1、成本、风险预算和上下限。
- 报告指标至少包含年化收益、波动、Sharpe、最大回撤、换手、成本、门控比例和 block bootstrap 区间。
"""


def build_data_draft(assets: pd.DataFrame, metadata: dict, model_gate: dict) -> str:
    modules = assets.groupby("module")["asset_code"].nunique().to_dict()
    local = metadata.get("local_features", [])
    macro = metadata.get("macro_features", [])
    source_rules = """
| 数据类别 | 主文件 | 频率 | 用途 | 点时规则 |
|---|---|---|---|---|
| 资产收益与 OHLCV | `data_processed/asset_returns_daily.parquet` | 日频/交易日 | 38 资产收益、局部特征、回测 | 仅使用当日及以前记录；源切换日收益剔除 |
| 日频特征面板 | `data_processed/feature_panel_daily_v43.parquet` | 日频/交易日 | 利率、信用、资金、转债、权益、另类状态特征 | 滚动计算只向后看 |
| 月频宏观面板 | `data_processed/macro_factors_monthly_v43.parquet` | 月频 | 宏观背景与低频状态 | 按真实发布后可得日对齐，不提前写入未发布值 |
| 市场状态卡 | `data_processed/market_state_cards_daily_v1.parquet` | 日频/交易日 | 仪表盘、九宫格、机械 View、模型上下文 | 仅用截至当日的特征生成 |
| 机械观点 | `data_processed/market_skill_views_mechanical_daily_v1.parquet` | 日频/交易日 | A1 市场观点、iTransformer 输入 | 规则在当天状态卡上生成 |
| 模型数据集 | `data_processed/hierarchical_asset_itransformer_v43_clean_calendar_dataset.npz` | 日频窗口 | 120 日序列与 5/20/60 日标签 | 标签起点严格为 t+1 |
"""
    return f"""# 固收+多资产动态配置：数据底稿（v43 Clean Calendar）

## 1. 口径总览

| 项目 | 正式口径 |
|---|---|
| 正式版本 | `{VERSION}` |
| 资产数量 | 38 |
| 袖套数量 | 5：利率、信用、转债、权益、另类 |
| 基础价格覆盖 | 2018-01-02 至 2026-07-21，个别资产按上市/可得日开始 |
| 模型决策结束日 | 2026-07-15 |
| 模型交易日 | 宽基权益锚定的真实交易日，周末数为 0 |
| iTransformer 序列长度 | 120 个交易日 |
| 预测期限 | 5、20、60 个交易日 |
| 回测执行 | 决策日收盘后产生信号，下一交易日生效 |
| 交易成本 | 5bp x 单边换手 |
| 现金 | 目标权重为 0 |

38 资产按袖套数量为：利率 {modules.get('rates', 0)}、信用 {modules.get('credit', 0)}、转债 {modules.get('convertible_bonds', 0)}、权益 {modules.get('equity', 0)}、另类 {modules.get('alternatives', 0)}。

## 2. 数据文件与责任边界

{source_rules}

## 3. 38 资产主数据字典

**代理类型说明**：`direct_or_fund_price` 是基金价格或直接价格；`index_proxy` 是指数或财富指数代理；`curve_duration_proxy` 与 `long_duration_treasury_proxy` 是利率曲线久期构造；`point_in_time_research_bucket` 是遵守可得日规则的研究组合，不等同于单一可交易产品。

{markdown_asset_table(assets)}

## 4. 特征底稿

### 4.1 每资产局部特征

局部特征来自 38 个资产的日频收益与风险轨迹，并在样本内做因果变换。数值特征同时带有缺失掩码，不把“没有历史/未上市”伪装成零收益。局部特征原始定义：

{', '.join(f'`{item}`' for item in local)}。

### 4.2 宏观、状态与规则特征

全局特征包括利率曲线、信用利差、资金、转债估值、权益估值、另类宏观代理、六类状态分数、股债九宫格和机械观点。进入模型的宏观特征如下：

{', '.join(f'`{item}`' for item in macro)}。

额外状态输入包括：`card_bond_score`、`card_equity_score`、`card_risk_budget_score`、`card_liquidity_shock`、信用/利率/转债/权益/另类/风险预算状态码，九宫格 one-hot，以及五袖套的机械 Market View。

## 5. 标签与防未来函数

1. 在日期 t 的模型输入只包含 t 及以前可见的数据。
2. 未来标签为 t+1 至 t+h 的年化对数平均收益；h 属于 5、20、60 日。
3. 标签在横截面去均值得到相对表现，模型不利用未来协方差或未来波动归一化。
4. 年度 walk-forward 训练只使用在预测年度开始前完全成熟的标签；验证集与训练集按最长标签期净化。
5. 宏观月频数据按可得日而非统计期末前置；尚未发布的数据不写入标签或特征。
6. 数据源切换日的收益剔除；周末行已经清理；数据集交易日周末数为 0。

## 6. 代理与研究限制清单

| 风险点 | 当前处理 | 文章必须披露的限制 |
|---|---|---|
| 30Y 利率债 | 使用 10年以上国债财富指数的久期代理，模型规范名为 `rate_30y_curve_proxy` | 不等同于纯30Y ETF，也不应与 `rate_30y` 视为两只资产 |
| 美债 | 使用美国10Y收益率构造 7.5 久期代理 | 非美元计价、汇率和真实QDII可交易摩擦未完全等价 |
| REITs | 首批9只公募 REITs 的等权研究指数 | 上市历史从 2021-06-22 开始，非官方可交易指数 |
| 城投AA/AA+ | iFind 指数增量数据 | 细分评级口径、源切换和可交易复制需持续审计 |
| 城投AAA | 城投债 ETF 代理 | 不能表述为严格评级拆分的城投 AAA 指数 |
| 转债风格桶 | 前一日价格与余额的点时分桶 | 是研究组合，未加入完整成分级交易冲击和停牌摩擦 |
| 部分权益行业 | 底层指数代理 | 不含 ETF 跟踪误差、申赎和实际冲击成本 |
| 另类宏观特征 | 日频代理与月频宏观结合 | 历史可得性不同，缺失由掩码与点时前向规则处理，不意味着数据完全同质 |

## 7. 当前模型观点的实际使用范围

预测门控通过了点时、标签、种子与零回退检查，允许进入组合研究；但只有以下组件被实际放行使用：

{json.dumps(model_gate.get('portfolio_consumed_active_components', []), ensure_ascii=False, indent=2)}

其余组件的学习观点严格为零。因此文章不能声称“模型每天对所有资产全面预测”；准确说法是：模型每天产生候选相对观点，但只有通过历史因果门控的模块/期限才进入优化。

## 8. 可复现性清单

1. 使用 `config/asset_universe.yaml` 和 `data_processed/asset_universe.parquet` 确定资产与来源。
2. 运行 `scripts/47_build_v43_basic_data.py` 与 `scripts/48_audit_v43_no_annual_data.py`。
3. 运行状态卡、机械观点和 `scripts/134_build_hierarchical_asset_itransformer_v7.py --config config/hierarchical_asset_itransformer_v43_clean_calendar.yaml`。
4. 先生成 A0/A1：`scripts/171_run_v43_control_variable_baselines.py`。
5. 再生成 A2：训练 iTransformer、因果线性探针、EP/skfolio 与 PPO。
6. 每次重跑都检查资产数=38、权重和=1、现金=0、交易日周末数=0、标签在决策日之后。
"""


def main() -> None:
    assets = asset_table()
    metadata = json.loads((OUT / f"{VERSION}_metadata.json").read_text(encoding="utf-8"))
    summary = pd.read_csv(OUT / f"{VERSION}_three_profile_summary.csv")
    baselines = pd.read_csv(OUT / "v43_clean_calendar_control_variable_baselines.csv")
    gate = json.loads((OUT / f"{VERSION}_portfolio_gate.json").read_text(encoding="utf-8"))
    model_gate = json.loads((OUT / f"{VERSION}_gate.json").read_text(encoding="utf-8"))
    (OUT / "v43_three_layer_article_draft.md").write_text(
        build_article_draft(summary, baselines, gate, model_gate), encoding="utf-8"
    )
    (OUT / "v43_data_draft.md").write_text(
        build_data_draft(assets, metadata, model_gate), encoding="utf-8"
    )
    print("created: output/reports/v43_three_layer_article_draft.md")
    print("created: output/reports/v43_data_draft.md")


if __name__ == "__main__":
    main()

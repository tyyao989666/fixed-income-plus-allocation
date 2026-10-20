# V1 一键运行说明

## 全流程一键跑

默认会重建 v43 数据、审计数据、生成 MarketSkill 状态和 views，然后并行跑低波/中波/高波三套产品：

```powershell
python scripts\run_v1_full_pipeline.py
```

需要先刷新 iFinD、Tushare 与本地转债库时，使用正式全量命令：

```powershell
python scripts\run_v1_full_pipeline.py --refresh-latest-data --refresh-daily-itransformer-v2 --market-view-mode mechanical --active-market-views mechanical --strategy-version no_agent --profiles low_vol,balanced,high_vol --workers 3
```

`--refresh-as-of YYYY-MM-DD` 可指定刷新截止日；`--refresh-latest-data` 不允许与 `--skip-data` 同时使用。

## 常用模式

只跑中波产品：

```powershell
python scripts\run_v1_full_pipeline.py --profiles balanced --no-parallel
```

数据和 MarketSkill 已经生成，只重跑三个产品组合/回测：

```powershell
python scripts\run_v1_full_pipeline.py --skip-data --skip-market-skill
```

显式重跑正式无 Agent 版本：

```powershell
python scripts\run_v1_full_pipeline.py --skip-data --skip-market-skill --strategy-version no_agent --profiles low_vol,balanced,high_vol --workers 3
```

使用已有 Agentic views 做双版本对比，不调用 LLM API：

```powershell
python scripts\run_v1_full_pipeline.py --skip-data --skip-market-skill --strategy-version compare --profiles low_vol,balanced,high_vol --workers 3
```

指定并行 worker 数：

```powershell
python scripts\run_v1_full_pipeline.py --workers 3
```

## 主要输出

共用信号输出：

- `data_processed/market_state_cards_daily_v1.parquet`
- `data_processed/market_skill_views_daily_v1.parquet`
- `data_processed/ep_views_daily_v1.parquet`

每个产品 profile 独立输出：

- `data_processed/v1_{profile}_risk_budget_base_monthly.parquet`
- `data_processed/v1_{profile}_sleeve_weights_monthly.parquet`
- `data_processed/v1_{profile}_target_weights_monthly.parquet`
- `output/reports/v1_{profile}_backtest_nav.csv`
- `output/reports/v1_{profile}_backtest_metrics.csv`
- `output/reports/v1_{profile}_attribution_by_sleeve.csv`
- `output/reports/v1_{profile}_robustness_summary.csv`

完成后会生成汇总清单：

- `output/reports/v1_full_pipeline_manifest.json`

可视化报告：

- `output/reports/v1_visual_report.html`
- `output/figures/v1/*.png`

数据口径工作簿：

- `output/reports/v1_data_usage_readme.xlsx`

## 运行口径

- 产品类型来自 `config/v1_strategy_config.yaml` 的 `low_vol`、`balanced`、`high_vol`。
- 每个产品先按目标风险贡献和滚动协方差生成风险预算基石。
- MarketSkill views 只做小幅偏离，不直接给最终仓位。
- 回测使用月频 T+1，换手成本按配置扣除。
- `v1_*_backtest_metrics.csv` 必须满足 `max_missing_return_weight=0`、`months_with_missing_return_exposure=0`。

## 数据刷新与收益代理

`scripts/116_refresh_latest_data.py` 负责增量刷新、原始响应落盘、旧文件备份和代理序列重建。也可单独运行：

```powershell
python scripts\116_refresh_latest_data.py --as-of 2026-07-16
```

- `alt_us_bond`：使用 iFinD 美债 10Y 收益率，按完整缺失月份逐月查询补齐，再按久期 7.5 构造曲线总收益代理。
- `equity_csi1000`：使用 Tushare `000852.SH` 指数收盘序列，避免 ETF 份额折算造成的价格伪跳点。
- `rate_30y_curve_proxy`：使用中国国债收益率曲线构造，不伪装成 30Y ETF 的历史价格。
- 转债风格指数按前一交易日价格分组、前一交易日余额加权，并限制单券权重；不使用当前存续样本回填历史。
- 不使用未来插值。刷新报告为 `output/reports/latest_data_refresh_report.md`，逐源状态为 `output/reports/latest_data_refresh_status.csv`。

## 参数搜索

三产品差异化搜索：

```powershell
python scripts\105_search_profile_differentiation.py --workers 3
```

选参只使用 `2018-2024`：`2018-2020`、`2021-2022`、`2023-2024` 三段。`2025-2026` 只做最终验收，不参与候选排序。

在正式产品参数不变的前提下，继续运行两层策略消融：

```powershell
python scripts\107_search_asset_momentum_overlay.py --workers 3
python scripts\108_search_sleeve_risk_brake.py --workers 3
python scripts\109_search_risk_on_rotation.py --workers 3
python scripts\110_search_alt_state_rotation.py --workers 3
python scripts\111_compare_sleeve_return_aggregation.py --workers 3
python scripts\112_search_market_view_preprocessing.py --workers 3
python scripts\113_search_view_reliability.py --workers 3
python scripts\114_search_view_application_channel.py --workers 3
python scripts\115_select_profile_view_channels.py
```

- `107` 的正式选择是权益/另类袖套内 6/12 月相对动量，三产品混合比例 6%/22%/38%。
- `108` 的正式选择是 3/6/12 月强风险刹车，转债/权益/另类转弱时只向利率和信用小幅转移。
- `111` 的正式选择是 `preferred_asset_weights` 袖套收益口径，风险预算和 EP 只使用各 profile 实际资产锚。
- `112` 的正式选择是 `monthly_pair_mean`：同一月、同一方向资产对的重复日频观点合并为一条，置信度取已观察日卡片的均值，避免同一状态持续多日时被机械重复放大。
- `113` 的历史命中率可信度校准未通过最终验收，正式保持关闭；它只保留为 Point-in-Time 消融模块。
- `114/115` 发现同一 view 同时直接推底仓、又进入 EP posterior mean 会重复计数。正式采用分产品单通道：低波 `EP mean only`，中波 `direct only × 0.5`，高波 `direct only × 1.0`。
- `109` 的追涨增风险和 `110` 的 ALT1 状态模板未通过，正式保持关闭。
- 所有信号和观点聚合都只读取决策月及以前数据，并由下一月权重执行；搜索输出在 `output/reports/v1_*_search_*.csv` 和对应 `.md`。

## iTransformer 与 PPO

项目同时保留三代实验，正式风险预算主流程不会被任何未过门控模型覆盖：

1. 旧月频 iTransformer：预测下一月五袖套收益，样本外失败，关闭。
2. 旧月末 V2 + PPO：使用日频路径但只在月末出分和动作，保留为历史对照；入口是 `scripts/123_run_itransformer_ppo_pipeline.py`。
3. 当前日频挑战者：逐交易日出分，按 5/20/60 日动作期限持有，并在下一交易日执行。

只使用已经训练好的日频残差预测重跑三产品 PPO：

```powershell
python scripts\128_run_daily_adaptive_pipeline.py --skip-state-cards --skip-itransformer
```

完整重建会先生成 38 张逐资产 mini-card，再训练残差 iTransformer：

```powershell
python scripts\128_run_daily_adaptive_pipeline.py
```

日频模型输入为 `120 × 64`，结构为 `d_model=64`、2 层、4 头，输出五袖套 5/20/60 日相对分数。最大 300 epoch，验证集 early stopping；并非必须跑满 300。直预测折内 RankIC 在 2022/2023/2024 为 `-0.030/+0.283/-0.105`，跨年不稳定。残差版只学习相对因果动量基线的增量，折内选择的残差权重为 `0/0.25/0`，增量 RankIC 门控失败。

日频 PPO 状态共 66 列，含 15 个 iTransformer 分数、5/20/60 日因果可靠性、市场总状态和 38 张资产卡的模块聚合诊断。动作空间为 `3 个期限 × 6 个幅度 = 18`，低/中/高波最大袖套对转移为 0.5%/1.0%/2.0%。每年只用截止日之前已经成熟的最长 60 日标签，三个种子中至少两个必须通过历史验证收益门，未通过时执行 `noop`。所有动作均 T+1、现金为 0、受袖套上下限约束并扣 5bp 成本。

截至 `2026-07-16`，共同 2024+ 区间的 Sharpe 增量为低/中/高波 `-0.034/-0.013/-0.001`，正式启用门为 `false`。运行结果见 `output/reports/adaptive_daily_ppo_method_and_results.md`，模型比较见 `output/reports/itransformer_daily_model_variant_comparison.md`。

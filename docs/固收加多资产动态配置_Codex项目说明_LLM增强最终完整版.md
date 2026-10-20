# 固收+多资产动态配置研究项目说明（LLM增强最终完整版｜Codex可直接执行版）

## 0. 项目定位与总目标

请帮我在桌面搭建一个 Python 量化研究项目，用于研究 **固收+多资产动态配置策略**。

本项目不是单纯做基金历史收益统计，也不是做 A 股个股选股。研究目标是构建一个完整的固收+多资产配置框架：

```text
资产池构建
→ 静态配置基准
→ 动态风险估计
→ iTransformer 预测资产占优概率与排序分数
→ LLM 情景矩阵解释与结构化 views 生成
→ Entropy Pooling 融合模型观点与 LLM views
→ skfolio 组合优化
→ Regime / 风险预算约束适配
→ 组合回测与多层归因
```

核心研究问题包括：

1. 固收+组合底层到底由哪些风险来源构成；
2. 不做择时，仅使用等权和风险平价，组合表现如何；
3. 如何构建前瞻性风险矩阵，为组合优化提供更稳定的风险输入；
4. 如何使用 iTransformer 预测未来一段时间哪些资产模块更可能占优；
5. 如何使用 LLM 将 iTransformer 输出、估值分位、风险状态和当前组合回撤解释为可复现的情景标签、相对收益 views 和约束调整建议；
6. 如何通过 Entropy Pooling 将 iTransformer 与 LLM 的结构化观点转成后验情景概率；
7. 如何用 skfolio 在 CVaR、换手率、仓位上下限和风险预算约束下生成最终权重；
8. 最终策略收益到底来自久期、信用、转债、权益、另类资产，还是模型择时、LLM 情景解释、EP 观点融合和组合优化本身。

请注意：

- 第一版不要做 A 股个股选股；
- 第一版不要做股票多因子横截面策略；
- 第一版不要做强化学习；
- 第一版不要做 A 股个股涨跌停 mask-first 工程；
- 第一版重点是底层资产池、动态风险矩阵、iTransformer、LLM 情景矩阵、Entropy Pooling、skfolio 和归因框架；
- LLM 第一版可以实现，但必须只做“情景解释器 + views 生成器 + 约束调节器”，不能直接输出最终组合权重；
- LLM 输出必须是 JSON，必须有 schema 校验、prompt 版本记录、输入输出落库和可关闭的 rule-based fallback；
- PPO 可以预留为后续扩展，但第一版不要实现。

---

## 1. 数据源优先级原则

### 1.1 总原则

本项目数据获取采用：

```text
Tushare 优先
Wind 兜底
本地缓存优先于重复请求
```

具体规则：

1. 如果 Tushare 能提供所需数据，优先使用 Tushare；
2. 如果 Tushare 没有该数据、字段不完整、历史长度不足、频率不满足要求，再使用 Wind；
3. 如果 Tushare 和 Wind 都能提供同类数据，优先保留 Tushare 数据，但记录 Wind 作为备选来源；
4. 所有数据表必须记录 `data_source` 字段，取值为 `tushare`、`wind`、`manual_mapping` 或 `derived`；
5. 所有原始数据必须落地保存，不允许每次运行重复拉取；
6. 每个数据接口都要有缓存机制；
7. 所有数据必须做质量检查，包括缺失率、重复值、时间范围、异常值和字段完整性；
8. 所有字段名、代码映射、数据源优先级写入 config，不要在代码中硬编码；
9. 如果某个接口或字段在当前 Tushare / Wind 版本中不存在，请在 quality report 中标记为 `missing_source`，并给出替代字段建议。

---

### 1.2 优先使用 Tushare 的数据

优先尝试从 Tushare 获取：

```text
A股指数行情
ETF / 基金日行情
股票行业指数行情
可转债个券基础信息
可转债日行情
可转债转股溢价率等基础指标
A股成交额、换手率、估值
部分宏观数据
部分利率数据
部分基金基础信息
交易日历
```

---

### 1.3 使用 Wind 兜底的数据

以下数据如果 Tushare 不完整，则使用 Wind：

```text
中债债券指数
不同期限国债指数
不同期限政金债指数
信用债指数
城投债指数
产业债指数
AAA / AA+ / AA 信用利差
城投 / 产业利差
信用债净融资
信用债到期压力
公开市场操作数据
DR007 / R007 / Shibor
国债期货连续合约
行业 ETF 或行业指数缺失部分
REITs 指数或基金数据
QDII 美债基金或海外资产数据
部分宏观披露日数据
```

---

### 1.4 手工映射数据

部分分类需要自建映射表：

```text
行业 ETF 映射表
权益行业群组映射表
信用债评级和类型映射表
转债风格分类规则
资产模块分类表
ETF 与底层资产映射表
```

这些放在：

```text
config/asset_universe.yaml
config/code_mapping.yaml
```

---

### 1.5 数据源适配器要求

请实现统一的数据源适配层：

```text
DataSourceManager
├── get_data()
├── get_from_cache()
├── fetch_from_tushare()
├── fetch_from_wind()
├── validate_data()
└── save_raw_data()
```

调用逻辑：

```text
先检查本地缓存；
如果缓存存在且未过期，直接读取缓存；
如果缓存不存在，先调用 Tushare；
如果 Tushare 失败或字段不完整，再调用 Wind；
最终返回标准化后的 DataFrame。
```

每次取数返回的 DataFrame 至少包含：

```text
date
code
字段
data_source
fetch_time
```

---

## 2. 文章与项目结构

本文最终写成五章，对应代码项目也按这五个模块组织。

```text
第一章｜静态配置基准：资产宇宙与风险平价
第二章｜动态风险估计与信号数据底座
第三章｜动态配置策略：iTransformer + LLM 情景矩阵 + Entropy Pooling + skfolio
第四章｜Regime 识别与风险预算适配
第五章｜归因验证与稳健性检验
```

---

## 3. Python 项目目录结构

请按照下面结构搭建项目：

```text
fixed_income_plus_allocation/
│
├── config/
│   ├── data_source_config.yaml
│   ├── tushare_fields.yaml
│   ├── wind_fields.yaml
│   ├── wind_codes.yaml
│   ├── asset_universe.yaml
│   ├── code_mapping.yaml
│   ├── feature_config.yaml
│   ├── model_config.yaml
│   ├── llm_config.yaml
│   ├── scenario_matrix.yaml
│   ├── llm_prompt_templates.yaml
│   ├── llm_output_schema.json
│   ├── scenario_thresholds.yaml
│   ├── market_state_card_schema.yaml
│   ├── confidence_rules.yaml
│   ├── ep_view_mapping.yaml
│   ├── constraint_priority.yaml
│   ├── transaction_cost.yaml
│   ├── backtest_timeline.yaml
│   ├── acceptance_criteria.yaml
│   ├── ablation_config.yaml
│   ├── optimization_config.yaml
│   └── backtest_config.yaml
│
├── data_raw/
│   ├── tushare/
│   ├── wind/
│   ├── manual/
│   ├── rates/
│   ├── credit/
│   ├── convertible_bonds/
│   ├── equity/
│   ├── alternatives/
│   ├── macro/
│   └── liquidity/
│
├── data_processed/
│   ├── asset_universe.parquet
│   ├── asset_returns_daily.parquet
│   ├── asset_returns_monthly.parquet
│   ├── feature_panel_daily.parquet
│   ├── feature_panel_monthly.parquet
│   ├── covariance_forecasts.parquet
│   ├── itransformer_dataset.parquet
│   ├── itransformer_predictions.parquet
│   ├── market_state_cards.parquet
│   ├── llm_scenario_outputs.parquet
│   ├── llm_relative_views.parquet
│   ├── ep_scenario_weights.parquet
│   ├── optimized_weights.parquet
│   └── backtest_results.parquet
│
├── output/
│   ├── tables/
│   ├── figures/
│   ├── reports/
│   └── models/
│
├── scripts/
│   ├── 00_test_data_connections.py
│   ├── 01_fetch_asset_data.py
│   ├── 02_build_asset_universe.py
│   ├── 03_build_returns.py
│   ├── 04_build_features.py
│   ├── 05_static_baseline.py
│   ├── 06_dynamic_risk_model.py
│   ├── 07_build_itransformer_dataset.py
│   ├── 08_train_itransformer.py
│   ├── 09_build_market_state_card.py
│   ├── 10_generate_llm_scenario_views.py
│   ├── 11_translate_views_to_ep.py
│   ├── 12_entropy_pooling.py
│   ├── 13_skfolio_optimization.py
│   ├── 14_regime_detection.py
│   ├── 15_backtest.py
│   ├── 16_attribution.py
│   ├── 17_quality_check.py
│   ├── 18_build_scenario_thresholds.py
│   ├── 19_validate_market_state_card.py
│   ├── 20_validate_llm_outputs.py
│   ├── 21_postprocess_llm_confidence.py
│   ├── 22_build_ep_soft_constraints.py
│   ├── 23_merge_base_overlay_weights.py
│   ├── 24_run_ablation_tests.py
│   └── 25_generate_acceptance_report.py
│
├── src/
│   ├── data_source_manager.py
│   ├── tushare_client.py
│   ├── wind_client.py
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── risk_model.py
│   ├── itransformer_model.py
│   ├── market_state_card.py
│   ├── llm_scenario_engine.py
│   ├── prompt_builder.py
│   ├── scenario_schema.py
│   ├── view_translator.py
│   ├── entropy_pooling.py
│   ├── optimizer_skfolio.py
│   ├── regime.py
│   ├── backtester.py
│   ├── attribution.py
│   └── utils.py
│
├── requirements.txt
└── README.md
```

代码要求：

1. 使用 Python、pandas、numpy、scipy、sklearn、torch、tushare、WindPy、skfolio、pyarrow，并预留 LLM provider 接口；
2. 优先调用 Tushare，Tushare 缺失或字段不足时调用 Wind；
3. 所有 Wind API 请求必须检查 ErrorCode；
4. Tushare 和 Wind 请求都必须分批，避免一次性请求过多代码；
5. 所有原始数据保存在 data_raw；
6. 所有清洗后数据保存在 data_processed；
7. 所有图表和结果保存在 output；
8. 所有字段名、资产代码、参数尽量写入 config，不要硬编码；
9. 每个脚本都可以独立运行；
10. 每一步都要输出数据质量检查；
11. 任何 rolling 特征、标准化、目标变量都不能使用未来数据；
12. 时间序列模型训练不能随机打乱样本；
13. 回测必须使用 walk-forward 或严格时间切分。

---

# 第一章｜静态配置基准：资产宇宙与风险平价

## 4.1 本章目标

第一章不做收益预测，不使用 iTransformer，不使用 Entropy Pooling。

本章只回答：

> 不做择时，只靠资产分散和风险平价，固收+多资产组合可以做到什么水平？

本章输出的等权组合、风险平价组合和历史收益矩阵，将作为后续 iTransformer、Entropy Pooling 和 skfolio 的基准与输入。

---

## 4.2 资产宇宙定义

资产池直接从固收+底层收益来源出发，分为五大模块：

```text
1. 利率债模块
2. 信用债模块
3. 可转债模块
4. 权益行业模块
5. 另类资产模块
```

---

## 4.3 利率债模块

用于刻画久期暴露、票息、骑乘和利率下行收益。

需要构造或获取以下资产：

```text
rate_1_3y        利率债 1-3Y
rate_3_5y        利率债 3-5Y
rate_7_10y       利率债 7-10Y
rate_30y         30Y 超长久期国债
```

优先使用：

```text
Tushare 可获取的债券 ETF / 基金行情
Wind 中债指数 / 国债 ETF / 政金债 ETF 数据
```

如果 Tushare 有对应 ETF 或基金行情，优先使用 Tushare。  
如果没有合适 ETF 或期限指数，则使用 Wind 指数作为研究层资产，后续再映射到具体可交易基金或 ETF。

---

## 4.4 信用债模块

用于刻画票息收益、信用利差压缩、信用下沉和城投/产业相对价值。

按照评级 × 类型拆分：

```text
credit_lgfv_aaa        AAA 城投债
credit_lgfv_aa_plus    AA+ 城投债
credit_lgfv_aa         AA 城投债

credit_ind_aaa         AAA 产业债
credit_ind_aa_plus     AA+ 产业债
credit_ind_aa          AA 产业债
```

数据优先级：

```text
Tushare 如果有可用信用债指数或债券行情，优先使用；
如果 Tushare 无法提供完整评级、城投/产业分类和信用利差，则使用 Wind。
```

后续可以构造：

```text
信用利差因子 = 信用债收益 - 同期限利率债收益
信用下沉因子 = AA 信用债收益 - AAA 信用债收益
城投产业因子 = 城投债收益 - 产业债收益
```

---

## 4.5 可转债模块

转债是固收+的核心增强资产。需要自建 12 个转债子指数。

三维划分：

```text
转债风格：偏债 / 平衡 / 偏股
正股市值：大盘 / 小盘
价格水平：低价 / 高价
```

最终形成 12 个子指数：

```text
cb_bond_large_low
cb_bond_large_high
cb_bond_small_low
cb_bond_small_high

cb_balanced_large_low
cb_balanced_large_high
cb_balanced_small_low
cb_balanced_small_high

cb_equity_large_low
cb_equity_large_high
cb_equity_small_low
cb_equity_small_high
```

数据优先级：

```text
Tushare 优先获取可转债个券基础信息、日行情、转股溢价率、正股信息；
如果 Tushare 字段不足，再用 Wind 补充纯债价值、YTM、余额、评级、强赎状态等字段。
```

### 转债风格初步规则

可以先用平价、转股溢价率、价格进行简化分类：

```text
偏债型：价格较低、平价较低、债性更强；
平衡型：价格和平价处于中间区间；
偏股型：平价较高、价格较高、股性更强。
```

### 大盘/小盘规则

```text
大盘：正股市值高于样本中位数；
小盘：正股市值低于样本中位数。
```

### 高价/低价规则

第一版可以使用：

```text
低价：转债价格 <= 120 元；
高价：转债价格 > 120 元。
```

后续可以改成横截面分位数。

### 转债指数加权方式

第一版使用等权。  
后续对比余额加权。

---

## 4.6 权益行业模块

权益部分不做个股，按 6-8 个行业群组使用行业 ETF 或行业指数。

建议资产：

```text
equity_tech_growth        科技成长
equity_manufacturing      高端制造
equity_consumption        消费
equity_healthcare         医药
equity_fin_property       金融地产
equity_cyclical           周期资源
equity_dividend_utility   公用事业 / 红利低波
equity_broad              宽基权益
```

数据优先级：

```text
优先用 Tushare 获取 ETF / 指数行情；
如果行业 ETF 或行业指数在 Tushare 中缺失，再用 Wind 行业指数或 ETF 数据。
```

---

## 4.7 另类资产模块

另类资产用于分散化和极端环境防守。

```text
alt_reits          REITs
alt_gold           黄金 ETF
alt_commodity      商品 ETF / 南华商品指数
alt_us_bond        QDII 美债
```

数据优先级：

```text
如果 Tushare 有对应 ETF / 基金行情，优先使用；
如果 Tushare 不覆盖 REITs 指数、QDII 美债、商品指数或海外相关变量，则用 Wind。
```

另类资产第一版可以纳入，但上限较低。  
建议后续优化中设置：

```text
另类资产总权重 <= 15%
单一另类资产 <= 5%
```

---

## 4.8 静态基准组合

### 4.8.1 等权基准

代表“不做择时、不做风险优化”的静态组合。

```text
每个资产权重 = 1 / N
```

可以额外构造模块等权：

```text
五大模块先等权；
模块内部再等权。
```

这样避免转债子指数数量过多导致组合过度偏向转债。

### 4.8.2 风险平价基准

代表“不预测收益，只使用风险信息”的静态组合。

先实现两种方法：

#### 方法一：波动率倒数加权

```text
w_i ∝ 1 / σ_i
```

其中 σ_i 为资产过去窗口内的年化波动率。

#### 方法二：等风险贡献 ERC

目标是让每个资产对组合总风险贡献接近：

```text
RC_i = w_i × (Σw)_i / portfolio_vol
```

---

## 4.9 波动率估计方法

比较以下方法：

```text
1. 收盘价收益率标准差
2. Parkinson 波动率
3. Yang-Zhang 波动率
```

说明：

- 对 ETF 和有 OHLC 数据的资产，可以比较 Parkinson 和 Yang-Zhang；
- 对债券指数、信用债指数、部分自建指数，主口径仍使用收益率标准差；
- Parkinson 和 Yang-Zhang 不强制用于所有资产。

---

## 4.10 回看窗口

比较：

```text
20日：响应快，但噪声大；
60日：稳定性和敏感度相对均衡；
120日：稳定，但反应慢。
```

第一版主参数使用 60 日，20 日和 120 日用于稳健性检验。

---

## 4.11 再平衡方式

比较两种：

```text
1. 固定月度再平衡；
2. 偏离度触发再平衡。
```

偏离度触发规则：

```text
如果实际权重与目标权重差异小于 3% 或 5%，则不交易；
只有超过阈值时才调仓。
```

---

## 4.12 第一章输出

输出文件：

```text
asset_universe.parquet
asset_returns_daily.parquet
asset_returns_monthly.parquet
static_equal_weight_backtest.parquet
static_inverse_vol_backtest.parquet
static_erc_backtest.parquet
static_baseline_metrics.xlsx
```

输出指标：

```text
年化收益
年化波动
最大回撤
Sharpe
Calmar
月度胜率
换手率
CVaR
回撤修复时间
```

核心对比表：

```text
等权组合 vs 波动率倒数组合 vs ERC 风险平价组合
```

---

# 第二章｜动态风险估计与信号数据底座

## 5.1 本章目标

第二章主要为第三章的 skfolio 组合优化做准备。

它不负责预测收益方向，而是提供前瞻性风险输入。

本章回答：

> 当前组合承担这些资产暴露时，未来可能面临多大的波动、相关性和尾部损失？

第二章输出：

```text
1. 每个资产未来波动率预测；
2. 动态相关性矩阵；
3. 前瞻性协方差矩阵；
4. 用于 skfolio 优化的风险输入；
5. 用于 Regime 识别的风险状态变量；
6. 用于 iTransformer 的特征数据底座。
```

---

## 5.2 为什么需要动态风险估计

第一章的等权和风险平价使用的是历史风险。  
但固收+组合最怕的不是普通波动，而是风险状态突然变化：

```text
权益波动突然放大；
转债和权益相关性上升；
股债相关性在危机中抬升；
信用利差突然走阔；
资金面收紧导致债券和转债同时承压。
```

因此需要从静态历史协方差升级到前瞻性风险矩阵。

---

## 5.3 三类协方差估计方法

### 方法一：Ledoit-Wolf 收缩协方差

目的：

```text
解决样本协方差在资产数量较多、小样本下不稳定的问题。
```

适用：

```text
作为基础风险矩阵；
作为静态风险平价和 skfolio 优化的稳健协方差输入。
```

---

### 方法二：EWMA 动态协方差

目的：

```text
对近期风险变化更敏感，捕捉波动率快速上升。
```

适用：

```text
风险预警；
短期仓位约束；
高波动阶段的风险控制。
```

---

### 方法三：GARCH + Ledoit-Wolf 混合方法

实用口径：

```text
单资产波动率：使用 GARCH(1,1) 或 EWMA 预测；
资产间相关性：使用 Ledoit-Wolf 收缩相关矩阵；
最终协方差矩阵：
Σ_t = D_t × Corr_t × D_t
```

其中：

```text
D_t = 预测波动率对角矩阵
Corr_t = Ledoit-Wolf 收缩相关矩阵
```

这个方法比完整 DCC-GARCH 更稳，也更适合第一版落地。

---

## 5.4 风险模型评价标准

不要只比较协方差预测误差。  
评价重点是：

> 用该风险矩阵做出来的组合，样本外表现是否更好。

评价指标：

```text
最大回撤是否降低；
CVaR 是否改善；
Calmar 是否提高；
组合权重是否更稳定；
极端权重是否减少；
高波动阶段是否降低损失。
```

---

## 5.5 iTransformer 输入特征体系

第二章同时需要构建 iTransformer 的输入变量。

输入变量分为七类：

```text
1. 资产自身特征；
2. 利率曲线特征；
3. 信用利差特征；
4. 转债估值与供需特征；
5. 权益行业与估值特征；
6. 流动性特征；
7. 宏观与海外特征。
```

---

## 5.6 资产自身特征

对每个资产生成：

```text
ret_1d
ret_5d
ret_20d
ret_60d
vol_20d
vol_60d
max_drawdown_60d
momentum_20d
momentum_60d
price_position_60d
```

---

## 5.7 利率曲线特征

```text
1Y 国债收益率
3Y 国债收益率
5Y 国债收益率
10Y 国债收益率
30Y 国债收益率
10Y-1Y 期限利差
30Y-10Y 超长端利差
2×5Y - 1Y - 10Y 蝶式因子
TS / TF / T / TL 国债期货收益
国债期货成交量
国债期货持仓量
```

---

## 5.8 信用特征

```text
AAA 信用利差
AA+ 信用利差
AA 信用利差
AA+ - AAA 等级利差
AA - AAA 等级利差
城投债利差
产业债利差
城投 - 产业利差
信用债净融资
信用债到期规模
信用债取消发行规模
```

---

## 5.9 转债特征

```text
中证转债指数收益
转债价格中位数
转债平均价格
转股溢价率中位数
纯债溢价率中位数
YTM 中位数
低价转债占比
高价转债占比
双低转债占比
转债成交额
转债市场余额
新券发行规模
强赎退出规模
```

---

## 5.10 权益特征

```text
沪深300收益
中证500收益
中证1000收益
创业板指收益
红利指数收益
万得全A收益
行业 ETF 收益
A股成交额
两融余额
PE/PB 分位数
股债性价比 ERP
权益波动率
```

---

## 5.11 流动性特征

```text
DR007
R007
R007 - DR007
Shibor 3M
1Y AAA 同业存单收益率
同业存单收益率变化
公开市场净投放
MLF利率
逆回购利率
```

---

## 5.12 宏观与海外特征

```text
PMI
社融同比
社融新增规模
M2同比
CPI同比
PPI同比
工业增加值同比
固定资产投资同比
出口同比
美元指数
美国10年国债收益率
美国实际利率
VIX
人民币汇率
黄金价格
南华商品指数
```

注意：

```text
宏观数据必须按照披露日处理，不能提前使用未来数据。
```

---

## 5.13 第二章输出文件

```text
feature_panel_daily.parquet
feature_panel_monthly.parquet
covariance_ledoit_wolf.parquet
covariance_ewma.parquet
covariance_garch_lw.parquet
risk_state_variables.parquet
risk_model_comparison.xlsx
```

---

# 第三章｜动态配置策略：iTransformer + LLM 情景矩阵 + Entropy Pooling + skfolio

## 6.1 本章目标

第三章是核心策略引擎。

整体链条：

```text
iTransformer 生成资产占优概率、预期收益和排序分数
→ Market State Card 汇总模型输出、估值、风险、流动性和当前组合状态
→ LLM 基于情景矩阵生成结构化 scenario labels、relative views 和 constraint adjustments
→ Entropy Pooling 将 iTransformer + LLM views 转成后验情景概率
→ skfolio 在风险、CVaR、换手和约束下优化组合权重
```

分工：

```text
iTransformer：预测未来谁更可能占优；
LLM：不预测原始收益，不直接给最终权重，只负责把模型输出解释成可复现的情景标签、相对收益 views 和约束调整建议；
EP：把结构化观点转成后验情景权重；
skfolio：在收益观点、风险矩阵和约束条件下计算最终权重。
```

---

## 6.2 iTransformer 模型定位

iTransformer 不直接输出组合权重。

它输出未来 20 个交易日各资产或资产模块的占优概率、预期收益或排序分数。

默认任务：

```text
预测 horizon = 20 trading days
输入 lookback_window = 120 trading days
调仓频率 = 月度或双周
```

---

## 6.3 iTransformer 输入

输入矩阵：

```text
X_t = 过去 120 个交易日 × K 个特征
```

K 包括第二章构建的全部特征：

```text
资产自身收益与波动；
利率曲线；
信用利差；
转债估值；
权益行业表现；
流动性；
宏观；
海外变量。
```

---

## 6.4 iTransformer 结构建议

第一版使用轻量结构：

```yaml
itransformer:
  lookback_window: 120
  prediction_horizon: 20

  model:
    enc_layers: 2
    n_heads: 4
    d_model: 64
    d_ff: 256
    dropout: 0.1
    activation: gelu

  input:
    frequency: daily
    standardization: rolling_zscore
    use_future_data: false

  output:
    predict_expected_return: true
    predict_outperform_probability: true
    predict_rank_score: true
```

含义：

```text
2层 Encoder；
每层 4 个 attention heads；
每个变量作为一个 token；
模型学习变量之间的跨资产关系；
输出未来20日每个资产的占优概率、预期收益和排序分数。
```

第一版用 2 层 4 头作为 baseline。  
稳健性测试再比较：

```yaml
itransformer_experiment:
  baseline:
    enc_layers: 2
    n_heads: 4
    d_model: 64
    d_ff: 256
    dropout: 0.1

  medium:
    enc_layers: 3
    n_heads: 4
    d_model: 128
    d_ff: 512
    dropout: 0.1

  large_heads:
    enc_layers: 3
    n_heads: 8
    d_model: 128
    d_ff: 512
    dropout: 0.1
```

模型选择不能只看训练 loss，要看样本外 Calmar、最大回撤、CVaR、RankIC、Top-K 命中率和换手率。

---

## 6.5 iTransformer 输出

模型输出包括：

```text
expected_return_i_next_20d
p_outperform_i_next_20d
rank_score_i_next_20d
```

其中 i 是资产池中的每个子资产。

也可以聚合成模块观点：

```text
p_long_duration_outperform
p_credit_outperform
p_lgfv_outperform
p_cb_balanced_low_price_outperform
p_equity_growth_outperform
p_gold_outperform
```

---

## 6.6 目标变量定义

### 目标一：未来收益率回归

```text
y_i,t = asset_i 未来20日收益率
```

### 目标二：是否跑赢基准

```text
target_i,t = 1 if asset_i未来20日收益 > 固收底仓组合未来20日收益 else 0
```

### 目标三：横截面排序

```text
rank_i,t = asset_i未来20日收益在资产池中的排名
```

第一版建议优先使用分类和排序，收益率回归作为辅助。

---

## 6.7 iTransformer 损失函数

iTransformer 使用多任务损失函数：

```text
Loss_iTransformer
= λ1 × Loss_return
+ λ2 × Loss_classification
+ λ3 × Loss_ranking
+ λ4 × Loss_calibration
+ λ5 × Loss_financial
```

### 1）收益率预测损失：Huber Loss

未来收益率噪声较大，直接使用 MSE 容易受到极端值影响。第一版建议使用 Huber Loss。

```text
Loss_return = Huber(y_return_pred, y_return_true)
```

### 2）跑赢概率损失：Binary Cross Entropy / Focal Loss

对于每个资产，构造是否跑赢基准的二分类标签：

```text
target_i = 1 if asset_i未来20日收益 > 固收底仓未来20日收益 else 0
```

基础损失使用 BCE：

```text
Loss_classification = BCE(p_outperform, target)
```

如果类别不平衡，可以使用 Focal Loss。第一版默认 BCE。

### 3）排序损失：ListNet / Pairwise Ranking Loss

本项目最终关心的是未来哪类资产更值得配置，所以资产排序比单点收益预测更重要。

可以构造横截面排序目标：

```text
rank_i,t = asset_i未来20日收益在所有资产中的排名
```

第一版建议实现 ListNet：

```text
Loss_ranking = CrossEntropy(softmax(rank_score_pred), softmax(future_return_true))
```

后续可以加入 Pairwise Ranking Loss。

### 4）概率校准损失：Brier Score

iTransformer 输出的概率后续会进入 Entropy Pooling，因此概率不能只追求方向正确，还要尽量校准。

```text
Loss_calibration = mean((p_outperform - target)^2)
```

### 5）金融目标损失：RankIC / Top-K 命中率惩罚

可以设计：

```text
Loss_financial = - RankIC(pred_rank_score, future_return)
```

或者：

```text
Loss_financial = - TopK_Return_Spread
```

其中：

```text
TopK_Return_Spread =
未来20日模型预测排名前K资产的平均收益
- 预测排名后K资产的平均收益
```

第一版可以先不反向传播该项，只用于模型选择。第二版再纳入训练损失。

### 6）推荐第一版损失函数

第一版建议使用：

```text
Loss_iTransformer
= 0.20 × Huber_Return_Loss
+ 0.35 × BCE_Outperform_Loss
+ 0.30 × ListNet_Ranking_Loss
+ 0.15 × Brier_Score
```

本文主口径更重视：

```text
资产占优方向
资产横截面排序
概率校准
```

收益率点预测只作为辅助任务。

---

## 6.8 训练过程中的 loss 诊断

训练阶段需要输出随 epoch 变化的 loss 曲线：

```text
train_total_loss
valid_total_loss
train_return_loss
valid_return_loss
train_classification_loss
valid_classification_loss
train_ranking_loss
valid_ranking_loss
train_brier_loss
valid_brier_loss
```

正常情况：

```text
训练集 loss 整体下降，但中间会波动；
验证集 loss 可能先下降后上升；
如果验证集 loss 上升，应使用 early stopping；
最终模型不能只按最低 train loss 选择。
```

请输出以下训练诊断图：

```text
1. train_total_loss 与 valid_total_loss 曲线；
2. return_loss / classification_loss / ranking_loss / brier_loss 分项曲线；
3. 每个 walk-forward 窗口的最佳 valid loss；
4. valid loss 与样本外 Calmar 的对比；
5. valid loss 与样本外最大回撤的对比；
6. RankIC、Top-K 命中率随 epoch 变化图；
7. early stopping 对应的最佳 epoch 标记。
```

---

## 6.9 iTransformer 输出如何转成观点

例如模型输出：

```text
p_7_10Y_rate_outperform = 0.65
p_AAplus_LGFV_outperform = 0.60
p_balanced_low_price_cb_outperform = 0.68
p_tech_growth_outperform = 0.55
p_gold_outperform = 0.58
```

可以转成如下观点：

```text
长久期 > 短久期
AA+城投 > AAA产业
平衡低价转债 > 偏股高价转债
科技成长 > 宽基权益
黄金 > 权益Beta
```

这些观点进入 Entropy Pooling。

---

---

## 6.10 LLM 情景矩阵层：定位、输入、输出与安全边界

### 6.10.1 LLM 的定位

LLM 在本项目中不是收益预测器，也不是组合优化器。

LLM 的角色是：

```text
情景解释器
+ 结构化 views 生成器
+ 约束调节器
+ 投研语言与量化信号之间的翻译层
```

LLM 不能做的事情：

```text
不能直接输出最终组合权重；
不能编造输入中不存在的新闻、政策、市场信息；
不能覆盖 iTransformer 的原始预测；
不能绕过风险预算、CVaR 和仓位约束；
不能使用未来数据；
不能在回测过程中改变 prompt 版本。
```

LLM 可以做的事情：

```text
1. 根据 iTransformer 输出判断当前更接近哪个股债九宫格情景；
2. 根据估值分位、流动性、信用利差、转债估值、权益风格等信息识别叠加标签；
3. 生成相对收益观点 relative_views；
4. 生成 Entropy Pooling 可用的 ep_views；
5. 生成 skfolio 约束调整建议；
6. 标记冲突信号并降低置信度；
7. 把模型结果转成可解释的投研语言，方便归因和报告输出。
```

---

### 6.10.2 LLM 输入：Market State Card

不要把大量原始表格直接塞给 LLM。  
请先由代码生成标准化的 `market_state_card`，再把该卡片传给 LLM。

每个调仓日生成一张卡片：

```json
{
  "date": "YYYY-MM-DD",
  "horizon": "20d",
  "core_model_outputs": {
    "p_rate_7_10y_outperform": 0.0,
    "p_rate_30y_outperform": 0.0,
    "p_credit_outperform": 0.0,
    "p_cb_outperform": 0.0,
    "p_cb_balanced_low_price_outperform": 0.0,
    "p_equity_outperform": 0.0,
    "p_equity_growth_outperform": 0.0,
    "p_gold_outperform": 0.0,
    "rank_scores": {}
  },
  "risk_state": {
    "equity_vol_percentile": 0.0,
    "cb_vol_percentile": 0.0,
    "bond_vol_percentile": 0.0,
    "cross_asset_corr_percentile": 0.0,
    "portfolio_cvar": 0.0,
    "risk_regime": ""
  },
  "valuation_state": {
    "cb_premium_percentile": 0.0,
    "cb_price_median_percentile": 0.0,
    "cb_ytm_percentile": 0.0,
    "equity_erp_percentile": 0.0,
    "equity_pe_percentile": 0.0,
    "credit_spread_percentile": 0.0,
    "long_duration_crowding_percentile": 0.0
  },
  "liquidity_state": {
    "dr007_level": 0.0,
    "dr007_change_20d": 0.0,
    "r007_dr007_spread_percentile": 0.0,
    "cd_1y_change_20d": 0.0,
    "omo_net_injection_20d": 0.0
  },
  "momentum_state": {
    "rate_7_10y_ret_20d": 0.0,
    "rate_30y_ret_20d": 0.0,
    "credit_ret_20d": 0.0,
    "cb_index_ret_20d": 0.0,
    "wind_all_a_ret_20d": 0.0,
    "growth_equity_ret_20d": 0.0,
    "dividend_equity_ret_20d": 0.0,
    "gold_ret_20d": 0.0
  },
  "current_portfolio_state": {
    "current_weights": {},
    "current_drawdown": 0.0,
    "drawdown_limit": 0.05,
    "remaining_risk_budget": 0.0,
    "turnover_last_month": 0.0
  }
}
```

Market State Card 生成要求：

```text
1. 所有字段必须来自当日或之前可得数据；
2. 宏观数据必须按披露日生效；
3. 所有分位数必须使用 rolling expanding 或历史窗口计算，不能使用未来全样本；
4. 卡片必须保存到 data_processed/market_state_cards.parquet；
5. 每次 LLM 调用必须记录对应的 card_id。
```

---

### 6.10.3 LLM 情景矩阵：主框架与叠加层

LLM 情景矩阵分成两层：

```text
第一层：股债九宫格，唯一主情景框架；
第二层：性价比与风险叠加矩阵，包括转债、信用、利率、权益、另类和风险预算。
```

#### A. 股债九宫格：唯一主框架

| 权益 × 债券 | B+ 债强 | B0 债震荡 | B- 债弱 |
|---|---|---|---|
| E+ 股强 | C1 股债双强 | C2 股强债震荡 | C3 股强债弱 |
| E0 股震荡 | C4 股震荡债强 | C5 股债双震荡 | C6 股震荡债弱 |
| E- 股弱 | C7 债强股弱 | C8 股弱债震荡 | C9 股债双弱 |

九宫格基础仓位：

| 核心情景 | 利率债 | 信用债 | 转债 | 权益 | 另类 | 逻辑 |
|---|---:|---:|---:|---:|---:|---|
| C1 股债双强 | 25% | 25% | 20% | 25% | 5% | 股债都能赚钱，组合均衡进攻 |
| C2 股强债震荡 | 20% | 25% | 20% | 30% | 5% | 权益主导，债券做底仓 |
| C3 股强债弱 | 15% | 25% | 25% | 32% | 3% | 权益强但利率上行，久期压低 |
| C4 股震荡债强 | 40% | 30% | 12% | 8% | 10% | 债券主导，权益低配 |
| C5 股债双震荡 | 30% | 35% | 15% | 12% | 8% | 票息、滚降和低估值资产为主 |
| C6 股震荡债弱 | 20% | 35% | 15% | 20% | 10% | 信用 carry 与权益结构机会更重要 |
| C7 债强股弱 | 50% | 30% | 8% | 4% | 8% | 久期资产主导，权益低配 |
| C8 股弱债震荡 | 35% | 35% | 8% | 7% | 15% | 防守、票息和黄金为主 |
| C9 股债双弱 | 35% | 35% | 3% | 2% | 25% | 股债相关性上升，黄金和短久期防守 |

#### B. 转债性价比矩阵：估值 × 权益趋势

| 转债估值 / 权益趋势 | 权益强 | 权益震荡 | 权益弱 |
|---|---|---|---|
| CB1 低估高性价比 | 转债20%-25%，平衡低价/偏股低价，转债优于直接权益 | 转债15%-20%，平衡低价/双低，赚估值修复 | 转债8%-15%，偏债低价/平衡低价 |
| CB2 合理偏便宜 | 转债15%-20%，可作为权益替代 | 转债10%-15% | 转债5%-10% |
| CB3 合理偏贵 | 转债8%-12%，权益可能优于转债 | 转债5%-10% | 转债0%-5% |
| CB4 高估低性价比 | 转债5%-8%，优先权益 | 转债0%-5%，优先信用债/利率债 | 转债0%-3%，优先纯债/黄金 |
| CB5 极端拥挤 | 转债0%-5%，即使股强也优先权益ETF | 转债0%-3% | 转债0% |

转债高估时的替代规则：

```text
权益强：权益 > 转债
权益震荡：信用债 / 3-5Y利率债 > 转债
权益弱：纯债 / 黄金 > 转债
转债极端拥挤：转债失去“固收+安全垫”意义，不适合作为增强主线
```

#### C. 信用债矩阵：利差补偿 × 尾部风险

| 信用状态 | 利差位置 | 尾部风险 | 信用债总仓位 | 结构建议 | 逻辑 |
|---|---|---|---:|---|---|
| CR1 高利差低风险 | 高 | 低 | 40%-45% | AA+城投、AAA产业，可小幅下沉 | 利差补偿充分 |
| CR2 利差压缩中段 | 中 | 低 | 35%-40% | AA+城投、AAA产业 | 还能赚压缩，但不能大幅下沉 |
| CR3 利差过低 | 低 | 低 | 25%-35% | AAA/AA+为主 | 赚票息，不主动下沉 |
| CR4 高利差高风险 | 高 | 高 | 20%-30% | AAA和高流动性AA+ | 高利差可能只是风险补偿 |
| CR5 利差低且风险升 | 低 | 高 | 15%-25% | AAA短久期 | 利差不足以覆盖风险 |
| CR6 城投优于产业 | 中高 | 可控 | 35%-45% | AA+城投为主 | 城投利差压缩更明确 |
| CR7 产业优于城投 | 中 | 可控 | 30%-40% | AAA/AA+产业 | 产业景气改善时更有弹性 |

#### D. 利率债矩阵：方向 × 期限结构 × 拥挤度

| 利率状态 | 利率债仓位 | 1-3Y | 3-5Y | 7-10Y | 30Y | 逻辑 |
|---|---:|---:|---:|---:|---:|---|
| R1 久期进攻 | 45%-60% | 5%-10% | 20%-25% | 40%-45% | 20%-30% | 利率下行初期，长久期资本利得强 |
| R2 稳健骑乘 | 25%-40% | 15%-25% | 45%-55% | 20%-30% | 0%-10% | 震荡时3-5Y兼顾票息、滚降和波动 |
| R3 短端友好 | 30%-45% | 25%-35% | 40%-50% | 15%-25% | 0%-10% | 短端下行更多，中短久期更稳 |
| R4 长端承压 | 10%-25% | 45%-55% | 35%-45% | 0%-15% | 0%-5% | 长端上行时压低久期 |
| R5 股债双杀下的利率债 | 35%-50% | 50%-60% | 30%-40% | 0%-10% | 0% | 利率债承担流动性和防守功能 |
| R6 债券拥挤 | 25%-35% | 20%-30% | 45%-55% | 15%-25% | 0%-5% | 长端涨多后赔率下降 |

#### E. 权益矩阵：估值 × 趋势 × 风格

| 权益状态 | 判断 | 权益总仓位 | 行业配置 | 逻辑 |
|---|---|---:|---|---|
| EQ1 低估修复 | ERP高、估值低分位、宽基企稳 | 20%-35% | 宽基、消费、医药、红利、部分成长 | 估值有安全边际 |
| EQ2 成长趋势 | 成长行业强，成交放大 | 25%-40% | 科技成长、高端制造、创业板、中证1000 | 成长弹性高 |
| EQ3 红利防御 | 风险偏好弱，高股息占优 | 5%-20% | 红利低波、公用事业、金融地产 | 权益低配时保留稳定收益 |
| EQ4 周期通胀 | 商品、PPI、周期资源走强 | 10%-25% | 周期资源、商品ETF、高股息周期 | 通胀交易下周期好于成长 |
| EQ5 高估拥挤 | PE/PB高分位、成交拥挤 | 0%-15% | 降高估成长，保留红利低波 | 高估值下回撤风险大 |
| EQ6 行业轮动 | 指数一般，行业强弱明显 | 10%-25% | 强行业 > 宽基，弱行业低配 | 做行业相对收益 |

#### F. 另类资产矩阵

| 状态 | 触发条件 | 另类仓位 | 具体资产 | 逻辑 |
|---|---|---:|---|---|
| ALT1 黄金避险 | 风险偏好下降，权益波动升，实际利率下行 | 5%-15% | 黄金ETF | 对冲权益和转债回撤 |
| ALT2 商品通胀 | PPI、商品价格、周期资源走强 | 5%-15% | 商品ETF、周期资源 | 对冲通胀和利率上行 |
| ALT3 美债占优 | 美国利率下行，全球避险 | 3%-10% | QDII美债 | 海外久期和避险属性 |
| ALT4 REITs修复 | 利率稳定或下行，分派率有吸引力 | 3%-8% | REITs | 类固收收益和分散化 |
| ALT5 另类中性 | 无明确信号 | 0%-5% | 黄金少量 | 分散工具 |
| ALT6 股债双杀保护 | 流动性冲击或风险事件 | 15%-25% | 黄金为主，少量商品/美债 | 股债相关性上升时缓冲组合 |

#### G. 风险预算矩阵

| 风险预算状态 | 配置约束 |
|---|---|
| RB1 预算充足 | 权益上限 +5%，转债上限 +5%，允许使用进攻型 views |
| RB2 预算中性 | 维持正常约束 |
| RB3 预算紧张 | 权益上限 -10%，转债上限 -5%至-10%，固收底仓 +10% |
| RB4 防守状态 | 权益0%-5%，转债0%-5%，信用下沉0%-5%，固收底仓75%-90% |

风险预算优先级最高：

```text
RB4 / LIQ4 出现时，直接覆盖大部分进攻型信号；
RB3 会压低转债和权益观点的置信度；
CB1、CR1 这类“看起来便宜”的信号，只有在 RB1/RB2 下才会被充分使用。
```

---

### 6.10.4 LLM 输出 JSON Schema

LLM 输出必须是 JSON。  
不允许输出 JSON 以外的自然语言。  
若输出不符合 schema，必须自动重试；重试失败则使用 rule-based fallback。

```json
{
  "date": "",
  "prompt_version": "",
  "model_version": "",
  "card_id": "",
  "scenario_summary": "",
  "core_stock_bond_scenario_weights": {
    "C1_stock_bond_both_strong": 0.0,
    "C2_stock_strong_bond_neutral": 0.0,
    "C3_stock_strong_bond_weak": 0.0,
    "C4_stock_neutral_bond_strong": 0.0,
    "C5_stock_bond_both_neutral": 0.0,
    "C6_stock_neutral_bond_weak": 0.0,
    "C7_bond_strong_stock_weak": 0.0,
    "C8_stock_weak_bond_neutral": 0.0,
    "C9_stock_bond_both_weak": 0.0
  },
  "overlay_states": {
    "rate_duration": {"state": "", "confidence": 0.0},
    "credit": {"state": "", "confidence": 0.0},
    "convertible_bond": {"state": "", "confidence": 0.0},
    "equity": {"state": "", "confidence": 0.0},
    "alternative": {"state": "", "confidence": 0.0},
    "risk_budget": {"state": "", "confidence": 0.0}
  },
  "conflict_flags": {
    "has_conflict": false,
    "cb_equity_conflict": false,
    "credit_risk_conflict": false,
    "duration_crowding_conflict": false,
    "risk_budget_signal_conflict": false
  },
  "suggested_module_weight_center": {
    "rate_bond": 0.0,
    "credit_bond": 0.0,
    "convertible_bond": 0.0,
    "equity": 0.0,
    "alternative": 0.0
  },
  "sub_asset_tilt": {
    "rate_bond": [],
    "credit_bond": [],
    "convertible_bond": [],
    "equity": [],
    "alternative": []
  },
  "relative_attractiveness": {
    "cb_vs_equity": {
      "winner": "",
      "confidence": 0.0,
      "reason": ""
    },
    "cb_vs_credit": {
      "winner": "",
      "confidence": 0.0,
      "reason": ""
    },
    "credit_vs_rate": {
      "winner": "",
      "confidence": 0.0,
      "reason": ""
    },
    "long_duration_vs_short_duration": {
      "winner": "",
      "confidence": 0.0,
      "reason": ""
    }
  },
  "relative_views": [
    {
      "asset_a": "",
      "asset_b": "",
      "relation": ">",
      "confidence": 0.0,
      "horizon": "20d",
      "reason": ""
    }
  ],
  "ep_views": [
    {
      "type": "relative_return",
      "expression": "",
      "confidence": 0.0,
      "soft_or_hard": "soft"
    }
  ],
  "skfolio_constraint_adjustment": {
    "rate_bond_cap": "neutral",
    "long_duration_cap": "neutral",
    "credit_cap": "neutral",
    "credit_downshift_cap": "neutral",
    "cb_cap": "neutral",
    "high_price_cb_cap": "neutral",
    "equity_cap": "neutral",
    "alternative_cap": "neutral",
    "cvar_constraint": "normal",
    "turnover_threshold": "normal",
    "risk_asset_budget": "neutral"
  }
}
```

字段要求：

```text
1. core_stock_bond_scenario_weights 之和必须等于 1；
2. suggested_module_weight_center 五大资产之和必须等于 1；
3. 单条 relative_view 的 confidence 默认不得超过 0.75；
4. 若 has_conflict=true，相关 confidence 不得超过 0.60；
5. 若 risk_budget 为 RB3/RB4，权益和转债相关 confidence 自动下调；
6. 若 LLM 建议与硬约束冲突，以硬约束为准；
7. 所有 reason 必须引用 market_state_card 中已有字段，不允许编造外部信息。
```

---

### 6.10.5 LLM Prompt 模板

Prompt 文件放在：

```text
config/llm_prompt_templates.yaml
```

建议模板：

```text
你是一个固收+多资产配置研究助手。

你不能直接输出最终组合权重。
你只能基于输入的 market_state_card 输出：
1. 股债九宫格场景概率；
2. 转债、信用、利率、权益、另类、风险预算叠加标签；
3. 五大资产配置中枢；
4. 细分资产倾向；
5. 相对性价比判断；
6. Entropy Pooling 可用的 relative views；
7. skfolio 可用的约束调整建议。

重要规则：
- 股债九宫格是唯一主框架；
- 转债高估值、信用利差补偿、久期拥挤、权益估值等都是叠加层；
- 如果转债高估且权益强，优先权益而不是高溢价转债；
- 如果转债高估且权益震荡，优先信用债或3-5Y利率债；
- 如果转债高估且权益弱，优先纯债或黄金；
- 如果信用利差高但尾部风险也高，不能简单加信用下沉；
- 如果长端债券拥挤，债强也不能无脑买30Y；
- 如果风险预算紧张，必须降低权益、转债、信用下沉等风险资产；
- 输出必须是合法 JSON，不要输出 JSON 以外的文字；
- 不允许编造输入中没有的信息。

输入：
{market_state_card}

输出：
严格按照 llm_output_schema.json。
```

---

### 6.10.6 LLM 到 Entropy Pooling 的映射

LLM 输出的 `relative_views` 进入 EP。

例子：

```json
{
  "asset_a": "equity_growth",
  "asset_b": "cb_high_premium",
  "relation": ">",
  "confidence": 0.68,
  "horizon": "20d",
  "reason": "convertible bond valuation is expensive while equity trend remains positive"
}
```

转成 EP 观点：

```text
E_q[equity_growth_return - cb_high_premium_return] >= 0
```

不同 confidence 对应不同观点强度：

| confidence | EP 处理 |
|---:|---|
| 0.50-0.55 | 不进入 EP，只记录 |
| 0.55-0.65 | 软约束，低权重 |
| 0.65-0.75 | 软约束，中权重 |
| >0.75 | 默认不允许，除非人工显式放开 |

`view_translator.py` 需要实现：

```text
relative_view_to_ep_constraint()
confidence_to_omega()
conflict_adjust_confidence()
validate_asset_names()
```

---

### 6.10.7 LLM 到 skfolio 的映射

LLM 不能输出最终权重，只能输出约束调整建议。

示例：

```json
"skfolio_constraint_adjustment": {
  "cb_cap": "tighten_to_0_5_percent",
  "high_price_cb_cap": "tighten_to_0_2_percent",
  "equity_cap": "keep_10_15_percent",
  "credit_cap": "raise_to_35_45_percent",
  "cvar_constraint": "normal_or_slightly_tighten"
}
```

代码需要把自然语言枚举映射成数值约束：

```yaml
constraint_mapping:
  cb_cap:
    neutral: [0.00, 0.25]
    tighten_to_0_5_percent: [0.00, 0.05]
    tighten_to_5_10_percent: [0.05, 0.10]
    raise_to_15_25_percent: [0.15, 0.25]

  high_price_cb_cap:
    neutral: [0.00, 0.10]
    tighten_to_0_2_percent: [0.00, 0.02]
    tighten_to_0_3_percent: [0.00, 0.03]

  equity_cap:
    keep_10_15_percent: [0.10, 0.15]
    keep_25_35_percent: [0.25, 0.35]
    raise_to_30_40_percent: [0.30, 0.40]
    tighten_to_0_5_percent: [0.00, 0.05]

  credit_cap:
    raise_to_35_45_percent: [0.35, 0.45]
    keep_25_35_percent: [0.25, 0.35]
    tighten_to_15_25_percent: [0.15, 0.25]
```

硬约束优先级：

```text
监管/产品硬约束
> 风险预算硬约束
> 流动性冲击约束
> LLM 约束建议
> iTransformer views
```

---

### 6.10.8 LLM 日志、审计和可复现要求

每次 LLM 调用必须落库：

```text
date
card_id
prompt_version
model_version
temperature
input_hash
output_hash
raw_prompt_path
raw_output_json
validated_output_json
schema_validation_result
fallback_used
retry_count
```

保存位置：

```text
data_processed/llm_scenario_outputs.parquet
output/reports/llm_audit_log.csv
output/reports/llm_prompt_version_report.md
```

LLM 参数建议：

```yaml
llm:
  enabled: true
  provider: configurable
  temperature: 0.0
  max_retries: 2
  require_json_schema: true
  allow_free_text: false
  fallback_to_rule_engine: true
  max_confidence: 0.75
  conflict_confidence_cap: 0.60
  risk_budget_tight_confidence_cap: 0.55
```

---

### 6.10.9 Rule-based fallback

如果 LLM 不可用或输出不合格，必须使用规则引擎兜底。

规则引擎基于固定阈值生成：

```text
core_stock_bond_scenario_weights
overlay_states
relative_views
constraint_adjustment
```

例如：

```text
如果 cb_premium_percentile > 0.80 且 equity_momentum > 0：
    cb_vs_equity = equity_better
    cb_cap = tighten_to_5_10_percent

如果 cb_premium_percentile > 0.80 且 equity_momentum <= 0：
    cb_vs_credit = credit_better
    cb_cap = tighten_to_0_5_percent

如果 credit_spread_percentile > 0.70 且 credit_tail_risk_low:
    credit_cap = raise_to_35_45_percent

如果 credit_spread_percentile > 0.70 且 credit_tail_risk_high:
    credit_downshift_cap = tighten_to_0_5_percent

如果 remaining_risk_budget < warning_threshold:
    equity_cap = tighten
    cb_cap = tighten
    cvar_constraint = tighten
```

rule-based fallback 输出也必须符合同一个 JSON schema。

---

### 6.10.10 LLM 相关输出文件

```text
market_state_cards.parquet
llm_scenario_outputs.parquet
llm_relative_views.parquet
llm_ep_views.parquet
llm_constraint_adjustments.parquet
llm_audit_log.csv
llm_prompt_version_report.md
```

---

## 6.11 Entropy Pooling

Entropy Pooling 的作用：

```text
历史收益情景 = prior
iTransformer 观点 = views
EP = 在尽量少改变历史分布的前提下，使后验分布满足观点
输出 = posterior scenario probabilities
```

EP 输入：

```text
历史资产收益矩阵 R
iTransformer 观点
观点置信度
```

EP 输出：

```text
每个历史情景的后验权重 posterior_weight_t
```

EP 优化目标：

```text
min_q  Σ q_t × log(q_t / p_t)
```

约束条件：

```text
1. Σ q_t = 1
2. q_t >= 0
3. 后验分布满足 iTransformer 观点
```

举例：

如果模型判断“转债强、长久期强、权益波动可控”，EP 应提高历史上类似月份的情景权重，降低股债双杀和转债估值压缩月份的情景权重。

输出文件：

```text
ep_views.parquet
ep_scenario_weights.parquet
posterior_return_distribution.parquet
```

---

## 6.12 skfolio 组合优化

skfolio 用于最终组合优化和回测。

输入：

```text
资产收益矩阵
EP 后验情景概率
第二章生成的协方差矩阵
仓位上下限
CVaR约束
换手约束
交易成本
```

主策略：

```text
EP + Mean-CVaR
```

对照策略：

```text
等权
波动率倒数
Risk Budgeting
HRP / HERC
Minimum Variance
Maximum Sharpe
Mean-CVaR
EP + Risk Budgeting
EP + Mean-CVaR
```

skfolio 优化目标可以写成：

```text
max_w  E_q[R_p] - λ1 × CVaR_q(R_p) - λ2 × Turnover(w) - λ3 × ConcentrationPenalty(w)
```

其中：

```text
R_p = w'R
E_q[R_p] = 在 EP 后验情景概率 q 下的组合期望收益
CVaR_q(R_p) = 在后验情景概率下的组合尾部风险
Turnover(w) = Σ |w_t - w_{t-1}|
ConcentrationPenalty(w) = Σ w_i^2
```

也可以写成最小化问题：

```text
min_w  - E_q[R_p]
+ λ1 × CVaR_q(R_p)
+ λ2 × Turnover(w)
+ λ3 × Σ w_i^2
```

---

## 6.13 组合约束

建议设置：

```text
单一资产权重 <= 30%
利率债总权重 >= 20%
固收底仓总权重 >= 30%
信用下沉资产权重 <= 20%
转债总权重 <= 25%
偏股型高价转债权重 <= 10%
权益总敞口 <= 40%
另类资产总权重 <= 15%
单一另类资产 <= 5%
月度换手率 <= 30%
CVaR 不超过阈值
```

其中：

```text
固收底仓 = 利率债 + 高等级信用债
风险资产 = 转债 + 权益 + 商品 + REITs
```

---

## 6.14 Risk Budgeting 目标函数

风险平价和风险预算使用风险贡献偏离作为目标。

组合波动率：

```text
σ_p = sqrt(w'Σw)
```

第 i 个资产的风险贡献：

```text
RC_i = w_i × (Σw)_i / σ_p
```

如果目标风险预算为 b_i，则 Risk Budgeting 优化目标为：

```text
Loss_RiskBudgeting = Σ (RC_i - b_i × σ_p)^2
```

其中：

```text
等风险贡献 ERC：b_i = 1 / N
不等风险预算：b_i 根据 Regime 或策略观点动态设定
```

---

## 6.15 剩余风险预算机制

固收+组合要引入剩余风险预算：

```text
剩余风险预算 = 回撤红线 - 当前组合回撤
```

例如：

```text
回撤红线 = 5%
当前回撤 = 2%
剩余风险预算 = 3%
```

规则：

```text
剩余风险预算充足：可以提高转债、权益、长久期权重；
剩余风险预算下降：降低风险资产；
接近回撤红线：组合回到固收底仓和低波资产。
```

这个机制体现：

```text
风控即择时。
```

也可以直接调整约束：

```text
如果剩余风险预算下降：
权益上限下降
转债上限下降
固收底仓下限上升
CVaR约束收紧
```

---

## 6.16 第三章输出文件

```text
itransformer_features.parquet
itransformer_labels.parquet
itransformer_predictions.parquet
itransformer_training_log.csv
market_state_cards.parquet
llm_scenario_outputs.parquet
llm_relative_views.parquet
llm_ep_views.parquet
llm_constraint_adjustments.parquet
model_views.parquet
ep_scenario_weights.parquet
posterior_return_distribution.parquet
optimized_weights.parquet
optimization_report.xlsx
```

---

# 第四章｜Regime 识别与风险预算适配

## 7.1 本章目标

Regime 不直接预测收益，也不直接输出权重。

它负责调节第三章策略的参数。

核心逻辑：

```text
Regime 决定约束和风险预算；
iTransformer 决定量化观点方向；
LLM 负责把量化输出解释成情景标签、relative views 和约束建议；
EP 决定后验情景概率；
skfolio 决定最终权重。
```

---

## 7.2 Regime 识别方法

第一版实现规则法：

```text
滚动波动率分位数 × 滚动相关性分位数
```

划分四种状态：

```text
低波低相关
低波高相关
高波低相关
高波高相关
```

后续增强：

```text
HMM
K-means
层次聚类
```

如果后续使用 HMM，则 HMM 的目标为最大化观测序列似然：

```text
max  log P(observed returns | hidden regimes)
```

第一版不要求实现复杂 HMM 损失。

---

## 7.3 Regime 映射

| Regime | 市场含义 | 参数调整 |
|---|---|---|
| 低波低相关 | 顺风环境，资产分散有效 | 放宽权益和转债上限，提高 iTransformer 观点权重 |
| 低波高相关 | 趋势状态，资产同向运行 | 提高趋势类信号权重，降低换手阈值 |
| 高波低相关 | 分化环境 | 回归风险平价，降低过度择时 |
| 高波高相关 | 危机环境 | 提高固收底仓，降低权益和转债，收紧 CVaR |

---

## 7.4 Regime 调节参数

Regime 调节：

```text
权益上限
转债上限
另类资产上限
CVaR约束强度
EP观点置信度
换手率阈值
风险预算比例
再平衡频率
```

例如：

```text
危机概率高时：
权益上限从 40% 降到 20%
转债上限从 25% 降到 10%
固收底仓下限从 30% 提高到 60%
CVaR 约束收紧
EP 观点置信度降低
```

---

## 7.5 第四章输出

```text
regime_probability.parquet
regime_label.parquet
regime_adjusted_constraints.parquet
regime_backtest_comparison.xlsx
```

---

# 第五章｜归因验证与稳健性检验

## 8.1 本章目标

第五章回答：

> 策略到底赚了什么钱？
> 是资产 Beta，还是模型择时？
> iTransformer、EP、skfolio 各自有没有增量？

---

## 8.2 资产模块归因

把组合收益拆成：

```text
组合收益
= 利率债贡献
+ 信用债贡献
+ 转债贡献
+ 权益贡献
+ 另类资产贡献
+ 残差
```

贡献计算：

```text
Contribution_i,t = Weight_i,t-1 × Return_i,t
```

---

## 8.3 底层因子归因

正文展示 8 类因子：

```text
1. 久期因子
2. 曲线与 Carry 因子
3. 信用因子
4. 转债因子
5. 权益 Beta 因子
6. 权益风格 / 行业因子
7. 流动性因子
8. 另类资产因子
```

研究底稿层可以构造 20 个左右因子：

```text
Short_Duration_Factor
Mid_Duration_Factor
Long_Duration_Factor
Ultra_Long_Duration_Factor
Curve_Steepening_Factor
Carry_Rolldown_Factor
Credit_Spread_Factor
Credit_Downshift_Factor
LGFV_Industrial_Factor
CB_Beta_Factor
CB_Equity_Delta_Factor
CB_Bond_Floor_Factor
CB_Valuation_Factor
CB_LowPrice_Factor
CB_Size_Factor
Equity_Beta_Factor
Growth_Value_Factor
Small_Large_Factor
Liquidity_Stress_Factor
Alternative_Hedge_Factor
```

---

## 8.4 转债因子必须单独细拆

转债因子包括：

```text
CB_Beta_Factor = 中证转债指数收益
CB_Equity_Delta_Factor = 偏股型转债收益 - 偏债型转债收益
CB_Bond_Floor_Factor = 偏债型转债收益 - 中证转债收益
CB_Valuation_Factor = - Δ转股溢价率中位数
CB_LowPrice_Factor = 低价转债收益 - 高价转债收益
CB_Size_Factor = 小盘转债收益 - 大盘转债收益
```

---

## 8.5 因子回归归因

使用 OLS 和 Ridge：

```text
R_portfolio,t =
α
+ β1 × Duration_Factor_t
+ β2 × Credit_Factor_t
+ β3 × CB_Factor_t
+ β4 × Equity_Factor_t
+ β5 × Liquidity_Factor_t
+ ...
+ ε_t
```

需要输出：

```text
因子暴露 beta
因子收益贡献
残差 alpha
解释度 R²
滚动 beta
```

同时做 24 个月滚动窗口归因。

---

## 8.6 正交化因子

由于转债和权益、信用和利率高度相关，需要构造纯因子：

```text
CB_Pure_Factor =
中证转债收益
- β1 × 万得全A收益
- β2 × 中证全债收益
```

```text
Credit_Pure_Factor =
信用债指数收益
- β × 国债指数收益
```

```text
Growth_Pure_Factor =
创业板收益
- β × 万得全A收益
```

避免同一风险来源被重复归因。

---

## 8.7 策略流程归因

按照策略链条拆：

```text
等权基准
→ 风险平价
→ iTransformer
→ iTransformer + LLM
→ iTransformer + LLM + EP
→ iTransformer + LLM + EP + skfolio
→ Regime 适配完整策略
```

计算增量：

```text
风险平价贡献 = 风险平价组合 - 等权组合
iTransformer贡献 = iTransformer择时组合 - 风险平价组合
LLM贡献 = iTransformer + LLM组合 - iTransformer组合
EP贡献 = iTransformer + LLM + EP组合 - iTransformer + LLM组合
skfolio贡献 = 完整策略 - iTransformer + LLM + EP组合
Regime贡献 = Regime适配策略 - 无Regime策略
```

---

## 8.8 样本外验证

必须使用时间切分：

```text
2018-2021：训练集
2022-2024：验证集
2025至今：测试集 / 模拟实盘
```

或者使用滚动窗口：

```text
过去3年训练
未来1个月测试
每月向前滚动
```

不能随机切分。

---

## 8.9 稳健性检验

需要测试：

```text
核心参数 ±30% 扰动
交易成本 0 / 5 / 10 / 20bp
再平衡频率：月度 / 双周 / 偏离度触发
CVaR 阈值不同设定
权益上限不同设定
转债上限不同设定
不同协方差估计方法
不同资产池口径
不同 iTransformer lookback window
不同 iTransformer 层数、head 数和 d_model
```

---

## 8.10 第五章输出

```text
asset_contribution.parquet
factor_attribution.parquet
rolling_factor_beta.parquet
strategy_contribution.parquet
robustness_report.xlsx
final_backtest_report.xlsx
```

---

# 9. 第一版执行顺序

请严格按照下面顺序开发，不要一开始直接写深度学习。

## P0：Tushare / Wind 连接和资产池数据

完成：

```text
Tushare 连接测试
WindPy 连接测试
资产代码配置
资产价格 / 指数数据抓取
数据源优先级测试
资产收益矩阵构建
数据质量检查
```

输出：

```text
asset_universe.parquet
asset_returns_daily.parquet
asset_returns_monthly.parquet
P0_quality_report.xlsx
```

---

## P1：静态基准

完成：

```text
等权组合
波动率倒数组合
ERC 风险平价组合
固定月度再平衡
偏离度触发再平衡
```

输出：

```text
static_baseline_metrics.xlsx
static_baseline_backtest.parquet
```

---

## P2：动态风险矩阵

完成：

```text
Ledoit-Wolf 协方差
EWMA 协方差
GARCH + Ledoit-Wolf 混合矩阵
风险模型对比
```

输出：

```text
covariance_forecasts.parquet
risk_model_comparison.xlsx
```

---

## P3：特征和目标变量

完成：

```text
所有市场特征
iTransformer 输入数据
未来20日收益 / 跑赢标签 / 排名标签
```

输出：

```text
feature_panel_daily.parquet
itransformer_dataset.parquet
```

---

## P4：基准预测模型

先不要直接上 iTransformer。  
先做：

```text
Logistic Regression
Random Forest
LightGBM
```

输出：

```text
baseline_predictions.parquet
baseline_model_metrics.xlsx
```

---

## P5：iTransformer

完成：

```text
多变量时间序列输入
资产占优概率预测
walk-forward 样本外预测
loss 曲线和训练诊断图
```

输出：

```text
itransformer_predictions.parquet
itransformer_metrics.xlsx
itransformer_training_log.csv
```

---

## P6：LLM 情景矩阵 + EP + skfolio

完成：

```text
Market State Card 生成
LLM 情景矩阵输出
LLM JSON schema 校验
relative_views 生成
EP views 转换
Entropy Pooling 后验情景概率
skfolio Mean-CVaR 优化
Risk Budgeting / HRP 对照
权重输出
```

输出：

```text
market_state_cards.parquet
llm_scenario_outputs.parquet
llm_relative_views.parquet
llm_ep_views.parquet
llm_constraint_adjustments.parquet
ep_scenario_weights.parquet
optimized_weights.parquet
optimization_report.xlsx
llm_audit_log.csv
```

---

## P7：Regime 和回测

完成：

```text
Regime 识别
Regime 参数调整
完整策略回测
```

输出：

```text
regime_probability.parquet
portfolio_backtest.parquet
portfolio_metrics.xlsx
```

---

## P8：归因和稳健性

完成：

```text
资产模块归因
底层因子归因
策略流程归因
稳健性检验
```

输出：

```text
attribution_report.xlsx
robustness_report.xlsx
final_report.xlsx
```

---

# 10. 代码实现注意事项

1. 优先使用 Tushare，Tushare 没有或字段不足时再使用 Wind；
2. Tushare 和 Wind 取数都必须分批；
3. Wind API 必须检查 ErrorCode；
4. 所有数据要有交易日对齐；
5. 所有数据表必须包含 `data_source` 字段；
6. 资产收益矩阵要处理缺失值；
7. 新上市资产不能提前进入样本；
8. 自建转债指数必须避免未来函数；
9. 转债分类使用当日可得数据；
10. 宏观数据必须按披露日生效；
11. rolling 特征只能使用过去数据；
12. 标准化只能使用训练集或 rolling z-score；
13. iTransformer 训练不能随机打乱时间序列；
14. 回测必须使用样本外 walk-forward；
15. 交易成本必须纳入；
16. 组合权重必须满足约束；
17. 每一步都要保存中间结果；
18. 每一步都要输出质量检查报告；
19. 模型选择不能只看训练 loss；
20. 最终以样本外 Calmar、最大回撤、CVaR、RankIC 和换手率综合判断。

---

# 11. PPO 后续扩展说明

第一版不实现 PPO。

如果后续加入 PPO，它只作为策略控制器，不直接输出资产权重。

PPO 状态包括：

```text
iTransformer 占优概率
EP 后验分布统计
skfolio 目标权重
当前组合权重
当前回撤
剩余风险预算
Regime 概率
过去20日组合收益
过去20日组合波动
过去20日换手
CVaR
```

PPO 动作包括：

```text
是否调仓
调仓强度
风险预算倍率
优化器选择
权益 / 转债上限调整
```

PPO 奖励函数可以写成：

```text
Reward
= Portfolio_Return
- λ1 × Drawdown
- λ2 × CVaR
- λ3 × Turnover
- λ4 × RiskBudgetViolation
```

PPO 损失包括：

```text
Loss_PPO
= Policy_Clipped_Loss
+ value_loss_coef × Value_Loss
- entropy_coef × Entropy
```

PPO 只作为后续增强模块，不进入第一版主策略。

---

# 12. 最终主策略定义

最终主策略为：

```text
iTransformer + LLM 情景矩阵 + Entropy Pooling + skfolio Mean-CVaR + Regime 风险预算
```

完整流程：

```text
1. 使用 Tushare / Wind API 获取资产池数据；
2. Tushare 优先，Wind 兜底；
3. 构建利率债、信用债、转债、权益和另类资产收益矩阵；
4. 用等权、波动率倒数和 ERC 建立静态基准；
5. 用 Ledoit-Wolf、EWMA、GARCH+LW 构建动态风险矩阵；
6. 用 iTransformer 预测未来20日各资产占优概率、预期收益和排序分数；
7. 生成 Market State Card，汇总模型输出、风险状态、估值分位、流动性和当前组合回撤；
8. 用 LLM 情景矩阵生成股债九宫格概率、叠加标签、relative views 和 skfolio 约束调整建议；
9. 用 Entropy Pooling 调整历史情景概率；
10. 用 skfolio 在 CVaR、换手率和仓位约束下优化组合；
11. 用 Regime 状态调整风险预算和约束强度；
12. 回测并做资产贡献、因子归因和策略流程归因。
```

---

# 13. 最终需要生成的核心结果表

```text
1. 资产池清单表
2. 数据源覆盖情况表：Tushare覆盖 / Wind补充 / 手工映射
3. 静态基准表现表
4. 风险模型对比表
5. iTransformer 训练 loss 与验证 loss 表
6. iTransformer 预测效果表
7. LLM 情景矩阵输出表
8. LLM relative views 与约束调整表
9. LLM prompt 版本与审计日志表
10. EP 情景权重变化表
11. skfolio 优化权重表
12. 完整策略回测表现表
13. 资产模块归因表
14. 因子回归归因表
15. 策略流程贡献表
16. 稳健性检验表
```

---

# 14. 最终评价指标

核心评价指标：

```text
年化收益
年化波动
最大回撤
Calmar Ratio
Sharpe Ratio
Sortino Ratio
CVaR
CDaR
月度胜率
换手率
回撤修复时间
尾部亏损
RankIC
Top-K 命中率
```

固收+策略的评价优先级：

```text
Calmar Ratio
> 最大回撤
> CVaR
> RankIC / Top-K 命中率
> 年化收益
> Sharpe
> 月度胜率
> 换手率
```

---

# 15. 最终写作口径

文章主线可以概括为：

本文首先构建由利率债、信用债、可转债、权益行业和另类资产组成的固收+多资产池，并在数据层面采用 Tushare 优先、Wind 兜底的方式构建可复用的数据底座；随后用等权和风险平价建立静态配置基准，并通过 Ledoit-Wolf、EWMA 和 GARCH+Ledoit-Wolf 构建前瞻性风险矩阵，为组合优化提供风险输入；在动态策略层面，使用 iTransformer 捕捉利率、信用、转债、权益和流动性变量之间的跨资产时序关系，生成未来20个交易日各资产模块的占优概率；随后将模型输出、估值分位、风险状态、流动性状态和当前组合回撤整合为 Market State Card，并通过 LLM 情景矩阵将其转化为股债九宫格概率、转债/信用/利率/权益/另类资产叠加标签、相对收益 views 和约束调整建议；再通过 Entropy Pooling 将结构化观点转化为后验情景概率，最后借助 skfolio 在 CVaR、换手率和仓位约束下完成组合优化。策略结果通过资产模块归因、底层因子归因和策略流程归因进行验证，判断收益究竟来自市场风险暴露，还是来自 iTransformer、LLM 情景解释、EP 和 skfolio 的策略增量。


---

# 附录 A｜LLM 相关配置文件建议

## A.1 config/llm_config.yaml

```yaml
llm:
  enabled: true
  provider: configurable
  model_name: configurable
  temperature: 0.0
  max_retries: 2
  require_json_schema: true
  allow_free_text: false
  fallback_to_rule_engine: true
  max_confidence: 0.75
  conflict_confidence_cap: 0.60
  risk_budget_tight_confidence_cap: 0.55

logging:
  save_raw_prompt: true
  save_raw_output: true
  save_validated_output: true
  save_input_hash: true
  save_output_hash: true

paths:
  market_state_cards: data_processed/market_state_cards.parquet
  llm_outputs: data_processed/llm_scenario_outputs.parquet
  llm_views: data_processed/llm_relative_views.parquet
  audit_log: output/reports/llm_audit_log.csv
```

---

## A.2 config/scenario_matrix.yaml

```yaml
core_stock_bond_scenarios:
  C1:
    name: stock_bond_both_strong
    base_weight: {rate_bond: 0.25, credit_bond: 0.25, convertible_bond: 0.20, equity: 0.25, alternative: 0.05}
  C2:
    name: stock_strong_bond_neutral
    base_weight: {rate_bond: 0.20, credit_bond: 0.25, convertible_bond: 0.20, equity: 0.30, alternative: 0.05}
  C3:
    name: stock_strong_bond_weak
    base_weight: {rate_bond: 0.15, credit_bond: 0.25, convertible_bond: 0.25, equity: 0.32, alternative: 0.03}
  C4:
    name: stock_neutral_bond_strong
    base_weight: {rate_bond: 0.40, credit_bond: 0.30, convertible_bond: 0.12, equity: 0.08, alternative: 0.10}
  C5:
    name: stock_bond_both_neutral
    base_weight: {rate_bond: 0.30, credit_bond: 0.35, convertible_bond: 0.15, equity: 0.12, alternative: 0.08}
  C6:
    name: stock_neutral_bond_weak
    base_weight: {rate_bond: 0.20, credit_bond: 0.35, convertible_bond: 0.15, equity: 0.20, alternative: 0.10}
  C7:
    name: bond_strong_stock_weak
    base_weight: {rate_bond: 0.50, credit_bond: 0.30, convertible_bond: 0.08, equity: 0.04, alternative: 0.08}
  C8:
    name: stock_weak_bond_neutral
    base_weight: {rate_bond: 0.35, credit_bond: 0.35, convertible_bond: 0.08, equity: 0.07, alternative: 0.15}
  C9:
    name: stock_bond_both_weak
    base_weight: {rate_bond: 0.35, credit_bond: 0.35, convertible_bond: 0.03, equity: 0.02, alternative: 0.25}

priority:
  - hard_product_constraints
  - risk_budget
  - liquidity_shock
  - core_stock_bond_scenario
  - relative_value_overlays
  - itransformer_views
```

---

## A.3 config/llm_output_schema.json

请按 6.10.4 中的 JSON schema 落地为文件，并在代码中使用 JSON Schema Validator 校验。  
schema 校验失败时：

```text
第一次：要求 LLM 只修复 JSON 格式；
第二次：仍失败则使用 rule-based fallback；
所有失败都写入 audit log。
```

---

# 附录 B｜LLM 模块伪代码

```python
def run_llm_scenario_pipeline(date):
    card = build_market_state_card(date)

    if not llm_config.enabled:
        output = rule_based_scenario_engine(card)
        output["fallback_used"] = True
    else:
        prompt = build_prompt(card, prompt_template_version)
        raw_output = call_llm(prompt, temperature=0.0)
        output = validate_json_schema(raw_output)

        if not output.valid:
            raw_output = retry_fix_json(raw_output)
            output = validate_json_schema(raw_output)

        if not output.valid:
            output = rule_based_scenario_engine(card)
            output["fallback_used"] = True

    output = apply_confidence_caps(output)
    output = enforce_hard_constraints(output)
    ep_views = translate_relative_views_to_ep(output["relative_views"])
    constraints = map_constraint_adjustments(output["skfolio_constraint_adjustment"])

    save_llm_output(date, card, output, ep_views, constraints)
    return ep_views, constraints
```

---

# 附录 C｜需要重点做的 LLM 消融实验

必须比较：

```text
1. 无 LLM：iTransformer → EP → skfolio
2. Rule-based LLM fallback：iTransformer → rule matrix → EP → skfolio
3. LLM only views：iTransformer → LLM relative views → EP → skfolio
4. LLM views + constraints：完整版本
```

比较指标：

```text
样本外 Calmar
最大回撤
CVaR
RankIC / Top-K 命中率
换手率
回撤修复时间
极端月份表现
```

如果 LLM 版本解释更好但回测更差，正文需要如实说明，不应强行采用。

---

# 16. 必须补充的工程化增强模块（最终版）

本节用于把前文的“研究框架”进一步补成“Codex 可执行系统”。前文已经定义了资产池、iTransformer、LLM 情景矩阵、Entropy Pooling、skfolio、Regime 和归因框架；本节进一步补充：

```text
1. 情景标签的可计算阈值；
2. Market State Card 字段字典；
3. LLM 历史回测防泄露规则；
4. LLM confidence 后处理公式；
5. LLM views 到 EP 的 soft constraint 实现；
6. 九宫格基础权重与叠加矩阵的合成算法；
7. Regime、风险预算、LLM 的优先级边界；
8. 回测时间线、交易成本、流动性摩擦；
9. LLM 独立评价指标；
10. 更完整的消融实验和验收标准。
```

请 Codex 将本节作为最终实现要求的一部分，不要只把它当作文字说明。

---

## 16.1 新增配置文件要求

请在 `config/` 下新增以下配置文件：

```text
config/
├── scenario_thresholds.yaml          # 情景标签阈值
├── market_state_card_schema.yaml     # Market State Card 字段字典
├── confidence_rules.yaml             # LLM confidence 后处理规则
├── ep_view_mapping.yaml              # LLM views 到 EP soft constraints 的映射规则
├── constraint_priority.yaml          # 约束优先级与冲突处理规则
├── transaction_cost.yaml             # 分资产交易成本与流动性参数
├── backtest_timeline.yaml            # 信号生成与成交时间线
├── acceptance_criteria.yaml          # 每阶段验收标准
└── ablation_config.yaml              # 消融实验配置
```

同时在 `scripts/` 下新增：

```text
scripts/
├── 18_build_scenario_thresholds.py
├── 19_validate_market_state_card.py
├── 20_validate_llm_outputs.py
├── 21_postprocess_llm_confidence.py
├── 22_build_ep_soft_constraints.py
├── 23_merge_base_overlay_weights.py
├── 24_run_ablation_tests.py
└── 25_generate_acceptance_report.py
```

---

# 17. 情景标签可计算阈值设计

## 17.1 总原则

前文的情景矩阵不能只停留在描述层面。所有标签必须尽量变成可计算字段。

每个标签至少包含：

```text
1. required_fields：判断所需字段；
2. hard_rules：硬阈值；
3. soft_score：综合打分；
4. fallback_rule：字段缺失时的替代规则；
5. confidence_cap：该标签对应的最高置信度；
6. data_quality_requirement：最低数据质量要求。
```

默认采用滚动分位数，而不是全样本分位数，避免未来函数。

```text
rolling_percentile_window = 756 trading days   # 约3年
min_periods = 252 trading days                 # 至少1年数据
```

所有分位数计算必须只使用 `t` 时点以前可得数据。

---

## 17.2 核心股债九宫格阈值

九宫格是唯一主情景框架，由权益方向 E 和债券方向 B 组成。

```text
E+ = 权益偏强
E0 = 权益震荡
E- = 权益偏弱

B+ = 债券偏强
B0 = 债券震荡
B- = 债券偏弱
```

### 17.2.1 权益方向分数 equity_direction_score

建议由四部分构成：

```text
equity_direction_score =
  0.35 × equity_momentum_score
+ 0.30 × itransformer_equity_score
+ 0.20 × equity_breadth_score
+ 0.15 × equity_valuation_support_score
```

其中：

```text
equity_momentum_score = sigmoid(0.5 × ret_20d_z + 0.5 × ret_60d_z)
itransformer_equity_score = mean(p_outperform for equity assets)
equity_breadth_score = percentile(上涨行业数量 / 行业总数)
equity_valuation_support_score = equity_erp_percentile
```

判断规则：

```yaml
equity_direction:
  E_plus:
    equity_direction_score: ">=0.60"
    description: "权益偏强"
  E_zero:
    equity_direction_score: "0.40-0.60"
    description: "权益震荡"
  E_minus:
    equity_direction_score: "<=0.40"
    description: "权益偏弱"
```

### 17.2.2 债券方向分数 bond_direction_score

```text
bond_direction_score =
  0.35 × rate_bond_momentum_score
+ 0.30 × itransformer_rate_score
+ 0.20 × yield_down_score
+ 0.15 × liquidity_support_score
```

其中：

```text
rate_bond_momentum_score = sigmoid(0.5 × rate_bond_ret_20d_z + 0.5 × rate_bond_ret_60d_z)
itransformer_rate_score = mean(p_outperform for rate bond assets)
yield_down_score = percentile(- change_10y_yield_20d)
liquidity_support_score = percentile(- change_dr007_20d)
```

判断规则：

```yaml
bond_direction:
  B_plus:
    bond_direction_score: ">=0.60"
    description: "债券偏强"
  B_zero:
    bond_direction_score: "0.40-0.60"
    description: "债券震荡"
  B_minus:
    bond_direction_score: "<=0.40"
    description: "债券偏弱"
```

### 17.2.3 九宫格映射

```yaml
core_stock_bond_scenario:
  C1_Eplus_Bplus:  {equity: E_plus,  bond: B_plus,  base_weight: {rate_bond: 0.25, credit_bond: 0.25, convertible_bond: 0.20, equity: 0.25, alternative: 0.05}}
  C2_Eplus_Bzero:  {equity: E_plus,  bond: B_zero,  base_weight: {rate_bond: 0.20, credit_bond: 0.25, convertible_bond: 0.20, equity: 0.30, alternative: 0.05}}
  C3_Eplus_Bminus: {equity: E_plus,  bond: B_minus, base_weight: {rate_bond: 0.15, credit_bond: 0.25, convertible_bond: 0.25, equity: 0.32, alternative: 0.03}}
  C4_Ezero_Bplus:  {equity: E_zero,  bond: B_plus,  base_weight: {rate_bond: 0.40, credit_bond: 0.30, convertible_bond: 0.12, equity: 0.08, alternative: 0.10}}
  C5_Ezero_Bzero:  {equity: E_zero,  bond: B_zero,  base_weight: {rate_bond: 0.30, credit_bond: 0.35, convertible_bond: 0.15, equity: 0.12, alternative: 0.08}}
  C6_Ezero_Bminus: {equity: E_zero,  bond: B_minus, base_weight: {rate_bond: 0.20, credit_bond: 0.35, convertible_bond: 0.15, equity: 0.20, alternative: 0.10}}
  C7_Eminus_Bplus: {equity: E_minus, bond: B_plus,  base_weight: {rate_bond: 0.50, credit_bond: 0.30, convertible_bond: 0.08, equity: 0.04, alternative: 0.08}}
  C8_Eminus_Bzero: {equity: E_minus, bond: B_zero,  base_weight: {rate_bond: 0.35, credit_bond: 0.35, convertible_bond: 0.08, equity: 0.07, alternative: 0.15}}
  C9_Eminus_Bminus:{equity: E_minus, bond: B_minus, base_weight: {rate_bond: 0.35, credit_bond: 0.35, convertible_bond: 0.03, equity: 0.02, alternative: 0.25}}
```

如果模型输出的是九宫格概率，而不是单一标签，则基础权重采用概率加权：

```text
base_weight = Σ_k scenario_probability_k × base_weight_k
```

---

## 17.3 转债状态阈值 CB1-CB5

转债性价比由四类变量决定：

```text
1. 估值：转股溢价率、纯债溢价率、YTM；
2. 价格结构：价格中位数、高价券占比、低价券占比；
3. 交易拥挤：成交额、换手率、涨跌分化；
4. 权益环境：权益趋势和波动率。
```

建议生成字段：

```text
cb_premium_percentile
cb_bond_premium_percentile
cb_price_median_percentile
cb_ytm_percentile
low_price_cb_ratio
high_price_cb_ratio
cb_turnover_percentile
cb_volume_percentile
cb_volatility_percentile
```

阈值建议：

```yaml
convertible_bond_state:
  CB1_cheap_high_value:
    conditions:
      cb_premium_percentile: "<=0.30"
      cb_price_median_percentile: "<=0.40"
      low_price_cb_ratio: ">=0.40"
    target_cb_weight: "15%-25%"
    preference: ["balanced_low_price", "bond_like_low_price", "double_low"]

  CB2_slightly_cheap:
    conditions:
      cb_premium_percentile: "0.30-0.50"
      cb_price_median_percentile: "0.30-0.60"
    target_cb_weight: "10%-20%"
    preference: ["balanced", "low_price", "liquid_large_cap"]

  CB3_slightly_expensive:
    conditions:
      cb_premium_percentile: "0.50-0.75"
      cb_price_median_percentile: "0.50-0.75"
    target_cb_weight: "5%-15%"
    preference: ["low_price_only", "balanced_low_premium"]

  CB4_expensive_low_value:
    conditions:
      cb_premium_percentile: ">=0.75"
      cb_price_median_percentile: ">=0.70"
      cb_ytm_percentile: "<=0.35"
    target_cb_weight: "0%-8%"
    preference: ["low_price_low_premium_only"]
    replacement_rule:
      equity_strong: "replace_with_equity"
      equity_range_bound: "replace_with_credit_or_3_5y_rate_bond"
      equity_weak: "replace_with_rate_bond_or_gold"

  CB5_extreme_crowding:
    conditions:
      cb_premium_percentile: ">=0.90"
      cb_price_median_percentile: ">=0.85"
      high_price_cb_ratio: ">=0.35"
      cb_turnover_percentile: ">=0.85"
    target_cb_weight: "0%-5%"
    preference: ["avoid_cb", "only_tiny_bond_like_low_price"]
```

如果 CB4 或 CB5 出现，必须触发以下约束：

```yaml
cb_overvaluation_constraints:
  cb_cap: "tighten"
  high_price_cb_cap: "0%-3%"
  high_premium_cb_cap: "0%-3%"
  cb_view_confidence_cap: 0.60
```

---

## 17.4 信用状态阈值 CR1-CR7

信用状态由两个维度决定：

```text
1. 信用利差位置：信用利差历史分位；
2. 尾部风险：取消发行、评级下调、信用债成交流动性、区域/行业压力。
```

### 17.4.1 信用尾部风险打分

```text
credit_tail_risk_score =
  0.25 × cancellation_issuance_percentile
+ 0.25 × rating_downgrade_count_percentile
+ 0.20 × credit_spread_volatility_percentile
+ 0.15 × low_rating_spread_widening_percentile
+ 0.15 × liquidity_discount_percentile
```

如果部分字段缺失，按可得字段重新归一化权重。

### 17.4.2 信用状态阈值

```yaml
credit_state:
  CR1_high_spread_low_risk:
    credit_spread_percentile: ">=0.70"
    credit_tail_risk_score: "<=0.35"
    credit_weight: "40%-45%"
    structure: {AAA_LGFV: 0.20, AAplus_LGFV: 0.35, AA_LGFV: 0.10, AAA_industrial: 0.25, AAplus_industrial: 0.10, AA_industrial: 0.00}

  CR2_mid_spread_low_risk:
    credit_spread_percentile: "0.40-0.70"
    credit_tail_risk_score: "<=0.35"
    credit_weight: "35%-40%"
    structure: {AAA_LGFV: 0.25, AAplus_LGFV: 0.35, AA_LGFV: 0.05, AAA_industrial: 0.25, AAplus_industrial: 0.10, AA_industrial: 0.00}

  CR3_low_spread_low_risk:
    credit_spread_percentile: "<=0.40"
    credit_tail_risk_score: "<=0.35"
    credit_weight: "25%-35%"
    structure: {AAA_LGFV: 0.30, AAplus_LGFV: 0.25, AA_LGFV: 0.00, AAA_industrial: 0.35, AAplus_industrial: 0.10, AA_industrial: 0.00}

  CR4_high_spread_high_risk:
    credit_spread_percentile: ">=0.70"
    credit_tail_risk_score: ">=0.65"
    credit_weight: "20%-30%"
    structure: {AAA_LGFV: 0.35, AAplus_LGFV: 0.25, AA_LGFV: 0.00, AAA_industrial: 0.35, AAplus_industrial: 0.05, AA_industrial: 0.00}

  CR5_low_spread_high_risk:
    credit_spread_percentile: "<=0.40"
    credit_tail_risk_score: ">=0.65"
    credit_weight: "15%-25%"
    structure: {AAA_LGFV: 0.45, AAplus_LGFV: 0.15, AA_LGFV: 0.00, AAA_industrial: 0.35, AAplus_industrial: 0.05, AA_industrial: 0.00}

  CR6_lgfv_better_than_industrial:
    lgfv_industrial_spread_percentile: ">=0.60"
    lgfv_tail_risk_score: "<=0.50"
    credit_weight: "35%-45%"
    structure: {AAA_LGFV: 0.25, AAplus_LGFV: 0.40, AA_LGFV: 0.05, AAA_industrial: 0.20, AAplus_industrial: 0.10, AA_industrial: 0.00}

  CR7_industrial_better_than_lgfv:
    industrial_profit_momentum_score: ">=0.60"
    industrial_tail_risk_score: "<=0.50"
    credit_weight: "30%-40%"
    structure: {AAA_LGFV: 0.20, AAplus_LGFV: 0.20, AA_LGFV: 0.00, AAA_industrial: 0.35, AAplus_industrial: 0.25, AA_industrial: 0.00}
```

---

## 17.5 利率债状态阈值 R1-R6

利率债状态必须同时考虑：

```text
1. 利率方向；
2. 曲线变化；
3. 久期交易拥挤度；
4. 股债相关性；
5. 资金面。
```

需要生成字段：

```text
change_1y_yield_20d
change_3y_yield_20d
change_5y_yield_20d
change_10y_yield_20d
change_30y_yield_20d
curve_10y_1y_change_20d
curve_30y_10y_change_20d
rate_bond_ret_20d
long_duration_ret_60d_percentile
long_duration_crowding_score
dr007_change_20d
bond_vol_percentile
```

久期拥挤度：

```text
long_duration_crowding_score =
  0.35 × long_duration_ret_60d_percentile
+ 0.25 × 30y_volume_percentile
+ 0.20 × 30y_position_percentile
+ 0.20 × inverse_30y_yield_percentile
```

阈值建议：

```yaml
rate_state:
  R1_duration_attack:
    change_10y_yield_20d: "<0"
    change_30y_yield_20d: "<0"
    long_duration_crowding_score: "<=0.70"
    rate_weight: "45%-60%"
    term_structure: {1_3y: "5%-10%", 3_5y: "20%-25%", 7_10y: "40%-45%", 30y: "20%-30%"}

  R2_carry_roll_down:
    abs_change_10y_yield_20d: "low"
    curve_roll_down_score: ">=0.50"
    rate_weight: "25%-40%"
    term_structure: {1_3y: "15%-25%", 3_5y: "45%-55%", 7_10y: "20%-30%", 30y: "0%-10%"}

  R3_short_end_friendly:
    dr007_change_20d: "<0"
    change_1y_yield_20d: "< change_10y_yield_20d"
    rate_weight: "30%-45%"
    term_structure: {1_3y: "25%-35%", 3_5y: "40%-50%", 7_10y: "15%-25%", 30y: "0%-10%"}

  R4_long_end_under_pressure:
    change_10y_yield_20d: ">0"
    change_30y_yield_20d: ">0"
    rate_weight: "10%-25%"
    term_structure: {1_3y: "45%-55%", 3_5y: "35%-45%", 7_10y: "0%-15%", 30y: "0%-5%"}

  R5_stock_bond_double_selloff_rate_defense:
    bond_ret_20d: "<0"
    equity_ret_20d: "<0"
    liquidity_stress_score: ">=0.65"
    rate_weight: "35%-50%"
    term_structure: {1_3y: "50%-60%", 3_5y: "30%-40%", 7_10y: "0%-10%", 30y: "0%"}

  R6_duration_crowded:
    long_duration_crowding_score: ">=0.75"
    rate_weight: "25%-35%"
    term_structure: {1_3y: "20%-30%", 3_5y: "45%-55%", 7_10y: "15%-25%", 30y: "0%-5%"}
```

---

## 17.6 权益状态阈值 EQ1-EQ6

权益状态由估值、趋势和风格共同决定。

```yaml
equity_state:
  EQ1_low_valuation_repair:
    equity_erp_percentile: ">=0.65"
    equity_momentum_score: ">=0.50"
    equity_weight: "20%-35%"
    preference: ["broad_index", "consumption", "healthcare", "dividend", "selective_growth"]

  EQ2_growth_trend:
    growth_momentum_percentile: ">=0.65"
    growth_vs_value_momentum: ">0"
    equity_weight: "25%-40%"
    preference: ["technology_growth", "advanced_manufacturing", "chinext", "csi1000"]

  EQ3_dividend_defensive:
    risk_appetite_score: "<=0.40"
    dividend_lowvol_momentum_percentile: ">=0.55"
    equity_weight: "5%-20%"
    preference: ["dividend_lowvol", "utilities", "financial_property"]

  EQ4_cyclical_inflation:
    commodity_momentum_percentile: ">=0.60"
    ppi_momentum_score: ">=0.55"
    equity_weight: "10%-25%"
    preference: ["cyclical_resources", "commodity_related", "high_dividend_cyclicals"]

  EQ5_expensive_crowded:
    equity_valuation_percentile: ">=0.80"
    equity_turnover_percentile: ">=0.80"
    equity_weight: "0%-15%"
    preference: ["reduce_expensive_growth", "keep_dividend_lowvol"]

  EQ6_sector_rotation:
    market_index_momentum_score: "0.40-0.60"
    sector_dispersion_percentile: ">=0.65"
    equity_weight: "10%-25%"
    preference: ["strong_sector_etf", "underweight_weak_sector", "avoid_broad_only"]
```

---

## 17.7 另类资产状态阈值 ALT1-ALT6

```yaml
alternative_state:
  ALT1_gold_hedge:
    risk_appetite_score: "<=0.40"
    gold_momentum_percentile: ">=0.55"
    real_rate_change_20d: "<=0"
    alternative_weight: "5%-15%"

  ALT2_commodity_inflation:
    commodity_momentum_percentile: ">=0.60"
    ppi_momentum_score: ">=0.55"
    alternative_weight: "5%-15%"

  ALT3_us_bond_outperform:
    us_10y_yield_change_20d: "<0"
    global_risk_aversion_score: ">=0.55"
    alternative_weight: "3%-10%"

  ALT4_reits_repair:
    reits_momentum_percentile: ">=0.55"
    domestic_rate_state: ["R1_duration_attack", "R2_carry_roll_down"]
    alternative_weight: "3%-8%"

  ALT5_neutral:
    no_strong_signal: true
    alternative_weight: "0%-5%"

  ALT6_stock_bond_double_selloff_protection:
    equity_ret_20d: "<0"
    bond_ret_20d: "<0"
    cross_asset_corr_percentile: ">=0.70"
    alternative_weight: "15%-25%"
```

---

## 17.8 风险预算状态阈值 RB1-RB4

风险预算是最终总闸门。

```text
current_drawdown = 当前净值 / 历史高点 - 1
remaining_risk_budget = drawdown_limit - abs(current_drawdown)
```

默认：

```yaml
risk_budget:
  drawdown_limit: 0.05
  warning_level: 0.03
  defensive_level: 0.045
```

阈值：

```yaml
risk_budget_state:
  RB1_sufficient:
    current_drawdown_abs: "<=0.015"
    cvar_percentile: "<=0.60"
    action: "allow_moderate_risk_up"

  RB2_neutral:
    current_drawdown_abs: "0.015-0.030"
    cvar_percentile: "<=0.70"
    action: "normal_constraints"

  RB3_tight:
    current_drawdown_abs: "0.030-0.045"
    action: "reduce_equity_cb_credit_downshift"
    caps: {equity_cap_delta: "-10%", cb_cap_delta: "-5%-10%", safe_asset_floor_delta: "+10%"}

  RB4_defensive:
    current_drawdown_abs: ">=0.045"
    action: "defensive_only"
    caps: {equity_cap: "0%-5%", cb_cap: "0%-5%", credit_downshift_cap: "0%-5%", safe_asset_floor: "75%-90%"}
```

---

# 18. Market State Card 字段字典

## 18.1 字段原则

Market State Card 是 LLM 唯一输入。LLM 不允许直接读取原始数据库、新闻、真实日期和未来收益。

每个字段必须包含：

```text
field_name
meaning
frequency
calculation
source
asof_rule
missing_value_policy
```

---

## 18.2 Market State Card 标准字段

| 字段 | 含义 | 频率 | 计算方式 | 数据源 | 防未来函数要求 |
|---|---|---|---|---|---|
| `rebalance_id` | 匿名调仓编号 | 调仓频率 | 从1递增 | derived | 历史回测时替代真实日期 |
| `horizon` | 预测期限 | 固定 | 默认20个交易日 | config | 无 |
| `asof_mode` | 模式 | 每期 | historical_backtest / live | config | 回测和实盘分开 |
| `date_visible_to_llm` | 是否向 LLM 暴露日期 | 每期 | 回测必须 false | config | 防止 LLM 日期泄露 |
| `equity_direction_score` | 权益方向分数 | 日/调仓 | 见17.2 | derived | 只用 t 日前数据 |
| `bond_direction_score` | 债券方向分数 | 日/调仓 | 见17.2 | derived | 只用 t 日前数据 |
| `cb_premium_percentile` | 转股溢价率分位 | 日 | 当前中位数在滚动窗口中的分位 | Tushare/Wind | 滚动窗口，不用全样本 |
| `cb_price_median_percentile` | 转债价格中位数分位 | 日 | 价格中位数滚动分位 | Tushare/Wind | 滚动窗口 |
| `cb_ytm_percentile` | 转债 YTM 分位 | 日 | YTM 中位数滚动分位 | Wind/Tushare | 滚动窗口 |
| `low_price_cb_ratio` | 低价转债占比 | 日 | 价格<=120的转债数量/样本数量 | Tushare/Wind | 当日可得 |
| `high_price_cb_ratio` | 高价转债占比 | 日 | 价格>=130或>=140的转债数量/样本数量 | Tushare/Wind | 当日可得 |
| `cb_turnover_percentile` | 转债换手拥挤度 | 日 | 成交额/余额分位 | Tushare/Wind | 滚动窗口 |
| `credit_spread_percentile` | 信用利差分位 | 日/周 | 当前信用利差滚动分位 | Wind | 滚动窗口 |
| `credit_tail_risk_score` | 信用尾部风险分数 | 周/月 | 取消发行、评级下调、流动性折价综合 | Wind/manual | 使用披露后数据 |
| `lgfv_industrial_spread_percentile` | 城投-产业利差分位 | 日/周 | 城投利差-产业利差 | Wind | 滚动窗口 |
| `change_10y_yield_20d` | 10Y国债收益率20日变化 | 日 | 当前-20日前 | Wind/Tushare | t日前数据 |
| `change_30y_yield_20d` | 30Y国债收益率20日变化 | 日 | 当前-20日前 | Wind | t日前数据 |
| `long_duration_crowding_score` | 长久期拥挤度 | 日/周 | 见17.5 | Wind | 滚动窗口 |
| `dr007_change_20d` | DR007 20日变化 | 日 | 当前-20日前 | Wind/Tushare | t日前数据 |
| `liquidity_stress_score` | 流动性压力分数 | 日 | R007-DR007、存单利率、资金波动综合 | Wind | t日前数据 |
| `equity_erp_percentile` | 股债性价比分位 | 日 | ERP滚动分位 | Wind/Tushare | t日前数据 |
| `equity_valuation_percentile` | 权益估值分位 | 日 | PE/PB滚动分位 | Wind/Tushare | t日前数据 |
| `growth_momentum_percentile` | 成长风格动量分位 | 日 | 成长指数收益滚动分位 | Wind/Tushare | t日前数据 |
| `sector_dispersion_percentile` | 行业分化程度 | 日 | 行业收益横截面标准差分位 | Wind/Tushare | t日前数据 |
| `gold_momentum_percentile` | 黄金动量分位 | 日 | 黄金ETF或黄金指数收益滚动分位 | Tushare/Wind | t日前数据 |
| `commodity_momentum_percentile` | 商品动量分位 | 日 | 商品指数收益滚动分位 | Wind | t日前数据 |
| `current_drawdown_abs` | 当前回撤绝对值 | 日 | abs(nav / hwm - 1) | derived | t日净值后可得 |
| `remaining_risk_budget` | 剩余风险预算 | 日 | drawdown_limit - current_drawdown_abs | derived | t日净值后可得 |
| `portfolio_cvar_percentile` | 组合 CVaR 分位 | 日/调仓 | 滚动历史 CVaR 分位 | derived | t日前收益 |
| `current_weight` | 当前组合权重 | 调仓 | 上期持仓漂移后权重 | derived | t日可得 |
| `data_quality_score` | 数据质量分数 | 每期 | 缺失率、异常值、字段覆盖 | derived | 当期数据检查 |

---

## 18.3 宏观字段的披露日处理

宏观数据必须包含：

```text
data_period：数据所属期
release_date：实际发布日期
effective_date：策略可使用日期，一般为 release_date 后第一个交易日
```

示例：

```text
PMI_2026_05 的 data_period = 2026-05
release_date = 2026-05-31
effective_date = 2026-06-01 或下一个交易日
```

策略在 `effective_date` 前不得使用该数据。

---

# 19. LLM 历史回测防泄露规则

## 19.1 核心风险

如果历史回测时把真实日期传给 LLM，LLM 可能利用训练语料中关于该时期的事后知识，造成隐性未来函数。

因此历史回测必须执行日期防泄露规则。

---

## 19.2 严格回测模式

```yaml
llm_backtest_firewall:
  historical_backtest:
    expose_calendar_date_to_llm: false
    expose_news_to_llm: false
    expose_future_returns_to_llm: false
    expose_raw_macro_release_text: false
    allow_web_search: false
    use_only_market_state_card: true
    freeze_prompt_version: true
    freeze_schema_version: true
    cache_raw_llm_output: true
```

Market State Card 中：

```json
{
  "rebalance_id": 87,
  "horizon": "20d",
  "asof_mode": "historical_backtest",
  "date_visible_to_llm": false
}
```

真实日期只保存在 audit log 中，不传给 LLM。

---

## 19.3 实盘模式

实盘或模拟实盘可以保留真实日期，但仍不能让 LLM 自己联网搜索新闻。

```yaml
live_mode:
  expose_calendar_date_to_llm: true
  allow_web_search: false
  input_source: "market_state_card_only"
```

如果未来要加入新闻摘要，必须单独做新闻数据源、时间戳和可得性校验，不允许 LLM 自己临时搜索。

---

# 20. LLM confidence 后处理规则

## 20.1 不直接相信 LLM 自己给出的 confidence

LLM 输出的 confidence 只能作为初始值，必须经过后处理。

最终置信度：

```text
final_confidence = min(
    llm_confidence,
    itransformer_confidence_cap,
    data_quality_cap,
    conflict_cap,
    risk_budget_cap,
    valuation_cap,
    max_allowed_confidence
)
```

默认：

```yaml
confidence_limits:
  max_allowed_confidence: 0.75
  conflict_cap: 0.60
  risk_budget_tight_cap: 0.55
  risk_budget_defensive_cap: 0.45
  low_data_quality_cap: 0.50
  cb_expensive_view_cap: 0.60
```

---

## 20.2 iTransformer 置信度上限

对二分类占优概率：

```text
itransformer_confidence_raw = abs(p_outperform - 0.5) × 2
```

为了避免过度压低，可以做平滑映射：

```text
itransformer_confidence_cap = 0.50 + 0.50 × itransformer_confidence_raw
```

示例：

```text
p_outperform = 0.65
raw = 0.30
cap = 0.65
```

---

## 20.3 Confidence 到 EP view 权重的映射

```yaml
confidence_to_ep:
  ignore:
    range: "<0.55"
    action: "do_not_enter_ep_only_log"
  low:
    range: "0.55-0.65"
    view_strength: 0.5
    lambda_view: 0.5
  medium:
    range: "0.65-0.75"
    view_strength: 1.0
    lambda_view: 1.0
  high:
    range: ">0.75"
    action: "not_allowed_by_default"
```

---

## 20.4 冲突置信度处理

以下情况必须降置信度：

```text
1. LLM view 与 iTransformer 方向相反；
2. 转债估值高但 LLM 建议加转债；
3. 信用尾部风险高但 LLM 建议信用下沉；
4. RB3/RB4 下 LLM 建议增加权益或转债；
5. LIQ4 下 LLM 建议增加信用下沉或小盘转债。
```

处理规则：

```text
若存在单一冲突：final_confidence <= 0.60
若存在多个冲突：final_confidence <= 0.50
若违反产品硬约束：直接丢弃该 view
```

---

# 21. LLM views 到 EP 的 soft constraint 实现

## 21.1 不建议第一版使用硬约束

LLM views 可能互相冲突，也可能和历史情景分布不一致。第一版 EP 应以 soft constraints 为主。

目标函数：

```text
min_q  KL(q || p)
      + λ_view × Σ_j confidence_j × violation_j(q)^2
```

其中：

```text
KL(q || p) = Σ_t q_t log(q_t / p_t)
```

基础约束：

```text
Σ_t q_t = 1
q_t >= 0
```

---

## 21.2 Relative return view

如果 LLM 输出：

```json
{
  "asset_a": "equity_growth",
  "asset_b": "cb_high_premium",
  "relation": ">",
  "confidence": 0.68,
  "horizon": "20d"
}
```

则 EP soft constraint 为：

```text
E_q[r_equity_growth - r_cb_high_premium] >= threshold
```

第一版 threshold 默认：

```text
threshold = 0
```

违约项：

```text
violation = max(0, threshold - E_q[r_a - r_b])
```

---

## 21.3 Directional view

如果 LLM 输出：

```json
{
  "type": "directional_return",
  "asset": "rate_7_10y",
  "view": "positive",
  "confidence": 0.66
}
```

则：

```text
E_q[r_rate_7_10y] >= 0
```

---

## 21.4 Risk view

如果 LLM 输出：

```json
{
  "type": "risk_constraint",
  "expression": "cb_tail_loss should be reduced",
  "confidence": 0.70
}
```

则不进入 EP 收益约束，而进入 skfolio 风险约束：

```text
cb_cap tighten
high_price_cb_cap tighten
CVaR penalty increase
```

---

## 21.5 EP 不可行或失败处理

如果 EP 求解失败：

```text
1. 降低所有 view confidence × 0.8；
2. 删除最低 confidence 的 view；
3. 改用 rule-based fallback views；
4. 若仍失败，则使用 prior scenario probabilities；
5. 所有失败必须写入 ep_failure_log.csv。
```

---

# 22. 九宫格基础权重 + 叠加矩阵合成算法

## 22.1 合成逻辑

LLM 输出的配置中枢不是最终权重，只是 skfolio 的 anchor 或 constraint input。

完整流程：

```text
Step 1：九宫格概率加权得到 base_weight
Step 2：叠加 CB / CR / R / EQ / ALT 修正，得到 delta_weight
Step 3：风险预算与 Regime 覆盖
Step 4：clip 到上下限
Step 5：归一化
Step 6：作为 skfolio anchor，不直接交易
```

---

## 22.2 伪代码

```python
def build_preliminary_weight(
    scenario_probs,
    scenario_base_weights,
    overlay_states,
    overlay_delta_rules,
    risk_budget_state,
    hard_bounds,
):
    # Step 1: base weight from nine-grid probabilities
    base_weight = sum(
        scenario_probs[k] * scenario_base_weights[k]
        for k in scenario_probs
    )

    # Step 2: overlay adjustments
    delta = zeros_like(base_weight)
    for dim, state in overlay_states.items():
        delta += overlay_delta_rules[dim][state]

    pre_weight = base_weight + delta

    # Step 3: risk budget override
    pre_weight = apply_risk_budget_override(pre_weight, risk_budget_state)

    # Step 4: clip by hard bounds
    pre_weight = clip(pre_weight, hard_bounds.lower, hard_bounds.upper)

    # Step 5: normalize
    pre_weight = pre_weight / pre_weight.sum()

    return pre_weight
```

---

## 22.3 示例：C5 + CB4

若当前核心情景为：

```text
C5 股债双震荡
```

基础仓位：

```text
利率债 30%
信用债 35%
转债 15%
权益 12%
另类 8%
```

若叠加：

```text
CB4 转债高估低性价比
```

修正后：

```text
利率债 32%-35%
信用债 38%-42%
转债 0%-5%
权益 10%-15%
另类 5%-8%
```

原因：

```text
股债都没有明确趋势，转债估值又贵，转债没有方向也没有赔率。
这时与其配置高溢价转债，不如配置 3-5Y 利率债和 AAA/AA+ 信用债赚票息。
```

---

# 23. Regime、风险预算、LLM 的边界和优先级

## 23.1 分工

```text
Regime：识别市场风险状态，如高波动、高相关、危机环境；
风险预算：根据当前组合回撤和 CVaR 决定能不能承担风险；
LLM：解释 Market State Card，生成情景标签、relative views 和约束建议；
skfolio：在硬约束、风险预算和 EP 后验分布下求最终权重。
```

LLM 不允许覆盖 Regime 和风险预算。

---

## 23.2 约束优先级

```yaml
constraint_priority:
  1_product_hard_constraints:
    description: "产品合同、监管和投资范围约束，最高优先级"
  2_risk_budget:
    description: "回撤红线、CVaR、剩余风险预算"
  3_liquidity_regime:
    description: "流动性冲击、高波高相关、危机 regime"
  4_llm_constraint_adjustment:
    description: "LLM 根据情景矩阵提出的软约束建议"
  5_itransformer_views:
    description: "模型预测观点"
  6_return_optimization:
    description: "收益最大化目标，最低优先级"
```

如果发生冲突：

```text
高优先级约束覆盖低优先级约束；
低优先级信号不删除，但降 confidence 并记录 conflict log。
```

---

# 24. 回测时间线和交易执行假设

## 24.1 标准时间线

```text
T日收盘后：
  1. 更新价格、收益、估值分位、风险状态；
  2. 生成 Market State Card；
  3. iTransformer 生成未来20日预测；
  4. LLM 生成情景标签、views 和约束建议；
  5. confidence 后处理；
  6. EP 生成后验情景概率；
  7. skfolio 生成目标权重。

T+1日：
  按设定成交价执行调仓。
```

第一版成交价假设：

```yaml
execution_price:
  default: "next_close"
  sensitivity_tests: ["next_open", "next_vwap", "next_close"]
```

如果使用 T 日收盘价生成信号，则最早只能在 T+1 执行，不能用 T 日收盘价成交。

---

## 24.2 调仓频率

```yaml
rebalance:
  main_frequency: "monthly"
  alternatives: ["biweekly", "threshold_triggered"]
  threshold_trigger:
    weight_deviation: "3%-5%"
    risk_budget_trigger: true
```

---

# 25. 交易成本和流动性摩擦

## 25.1 分资产交易成本

第一版默认交易成本：

```yaml
transaction_cost:
  rate_bond_etf: 0.5bp
  treasury_bond_index_proxy: 0.0bp
  credit_bond_fund: 2bp
  convertible_bond_index_proxy: 5bp
  equity_etf: 3bp
  reits: 5bp
  gold_etf: 3bp
  commodity_etf: 5bp
  qdii_us_bond: 10bp
```

稳健性检验：

```text
0bp / 5bp / 10bp / 20bp 单边成本
```

---

## 25.2 可转债自建指数的流动性过滤

自建转债子指数必须做以下处理：

```text
1. 上市未满20个交易日剔除；
2. 余额低于阈值的个券剔除或降权；
3. 日成交额低于阈值的个券剔除或降权；
4. 进入强赎期或临近退市个券单独处理；
5. 停牌日不参与调仓；
6. 异常价格、负溢价异常值需做 winsorize 或剔除；
7. 个券权重上限，例如单券不超过5%；
8. 子指数至少包含一定数量个券，不足时标记 insufficient_sample。
```

建议配置：

```yaml
convertible_bond_liquidity_filter:
  min_listing_days: 20
  min_balance: 100_000_000
  min_daily_amount: 5_000_000
  max_single_bond_weight: 0.05
  min_bonds_per_subindex: 10
  handle_forced_redemption: "exclude_after_announcement_or_downweight"
```

---

# 26. LLM 独立评价指标

除了组合表现，还要单独评价 LLM 模块。

## 26.1 工程稳定性指标

```text
schema_pass_rate：JSON schema 通过率
fallback_rate：fallback 使用率
retry_count：平均重试次数
empty_output_rate：空输出比例
parse_error_rate：解析失败比例
```

## 26.2 观点质量指标

```text
view_hit_ratio：relative views 方向命中率
high_confidence_view_hit_ratio：高置信 views 命中率
view_ic：view implied score 与未来收益的相关性
avg_views_per_period：每期 views 数量
view_turnover：views 换手率
view_conflict_rate：views 之间冲突率
```

## 26.3 风险和交易影响指标

```text
llm_turnover_increase：加入 LLM 后换手增加
llm_drawdown_contribution：加入 LLM 后最大回撤变化
llm_cvar_contribution：加入 LLM 后 CVaR 变化
constraint_violation_count：LLM 建议违反硬约束次数
llm_weight_impact：LLM 对最终权重的边际影响
```

输出文件：

```text
output/tables/llm_engineering_metrics.xlsx
output/tables/llm_view_quality_metrics.xlsx
output/tables/llm_risk_impact_metrics.xlsx
```

---

# 27. 更完整的消融实验

在原有消融实验基础上，增加以下组别：

```yaml
ablation_tests:
  baseline_equal_weight:
    description: "等权基准"

  static_risk_parity:
    description: "静态风险平价"

  itransformer_only:
    description: "只用 iTransformer views，不使用 LLM"

  rule_based_scenario_engine:
    description: "不用 LLM，只用规则情景矩阵"

  llm_views_only:
    description: "LLM 只输出 relative views，不调约束"

  llm_constraints_only:
    description: "LLM 只调约束，不进入 EP views"

  llm_labels_only:
    description: "LLM 只输出情景标签，不生成 views"

  llm_views_plus_constraints:
    description: "LLM 同时输出 views 和 constraints"

  llm_date_hidden:
    description: "回测中不向 LLM 暴露真实日期"

  llm_date_visible:
    description: "仅用于检测日期泄露风险，不作为主结果"

  full_model:
    description: "iTransformer + LLM + EP + skfolio + Regime"
```

评价维度：

```text
年化收益
最大回撤
Calmar
CVaR
换手率
LLM view hit ratio
schema pass rate
fallback rate
```

主结果必须使用 `llm_date_hidden`。

---

# 28. 各阶段验收标准 Acceptance Criteria

## 28.1 P0 数据连接与资产池

完成标准：

```text
1. Tushare 连接测试通过；
2. Wind 连接测试通过或给出明确不可用原因；
3. asset_universe.parquet 非空；
4. asset_returns_daily.parquet 非空；
5. 所有资产存在 date、code、return、data_source 字段；
6. 缺失率报告输出；
7. 至少覆盖 2018-01-01 至今，若部分资产上市较晚需记录 start_date；
8. 新上市资产不得提前进入回测。
```

输出：

```text
output/reports/P0_acceptance_report.md
```

---

## 28.2 P1 静态基准

完成标准：

```text
1. 等权组合回测完成；
2. 波动率倒数组合回测完成；
3. ERC 风险平价组合回测完成；
4. 每期权重 sum(w)=1；
5. 无权重为 NaN；
6. 输出年化收益、波动、最大回撤、Sharpe、Calmar、CVaR。
```

---

## 28.3 P2 动态风险矩阵

完成标准：

```text
1. Ledoit-Wolf 协方差可生成；
2. EWMA 协方差可生成；
3. GARCH+LW 可运行或提供 fallback；
4. 每期协方差矩阵半正定检查；
5. 协方差矩阵维度与资产池一致；
6. 风险模型对组合表现的影响表输出。
```

---

## 28.4 P3 特征和标签

完成标准：

```text
1. feature_panel_daily.parquet 非空；
2. itransformer_dataset.parquet 非空；
3. 每个特征有字段字典；
4. 宏观数据按 effective_date 生效；
5. rolling 特征无未来函数；
6. 未来20日标签正确向后移动。
```

---

## 28.5 P4 基准预测模型

完成标准：

```text
1. Logistic Regression / Random Forest / LightGBM 至少一个跑通；
2. walk-forward 预测结果非空；
3. RankIC、Top-K 命中率输出；
4. 作为 iTransformer 对照组保存。
```

---

## 28.6 P5 iTransformer

完成标准：

```text
1. itransformer_predictions.parquet 非空；
2. 每个 rebalance_date 都有预测；
3. train/valid loss 曲线输出；
4. return_loss / classification_loss / ranking_loss / brier_loss 分项输出；
5. RankIC、Top-K 命中率输出；
6. walk-forward 验证完成；
7. 不使用随机时间切分。
```

---

## 28.7 P6 LLM + EP + skfolio

完成标准：

```text
1. market_state_cards.parquet 非空；
2. Market State Card schema 校验通过率 >= 99%；
3. LLM JSON schema pass rate >= 95%；
4. fallback_rate 有记录；
5. llm_relative_views.parquet 非空；
6. final_confidence 已完成后处理；
7. ep_scenario_weights 每期 sum(q)=1；
8. optimized_weights 每期 sum(w)=1；
9. 所有权重满足硬约束；
10. EP failure log 即使为空也要生成。
```

---

## 28.8 P7 Regime 和回测

完成标准：

```text
1. regime_probability.parquet 非空；
2. regime 与风险预算约束已进入最终 constraint engine；
3. 回测净值曲线输出；
4. 成本前后结果均输出；
5. T 日信号、T+1 执行逻辑检查通过。
```

---

## 28.9 P8 归因和稳健性

完成标准：

```text
1. 资产模块归因完成；
2. 因子回归归因完成；
3. 策略流程归因完成；
4. 参数敏感性完成；
5. 交易成本敏感性完成；
6. 消融实验完成；
7. final_report.xlsx 和 final_report.md 输出。
```

---

# 29. 更新后的最终执行顺序

最终执行顺序如下：

```text
P0 数据源连接与资产池
P1 静态基准
P2 动态风险矩阵
P3 特征和标签
P4 基准预测模型
P5 iTransformer
P6 Market State Card
P7 LLM 情景矩阵与 JSON 校验
P8 LLM confidence 后处理
P9 EP soft constraints
P10 skfolio 优化
P11 Regime / 风险预算覆盖
P12 回测
P13 归因
P14 消融实验
P15 验收报告
```

对应脚本建议：

```text
00_test_data_connections.py
01_fetch_asset_data.py
02_build_asset_universe.py
03_build_returns.py
04_build_features.py
05_static_baseline.py
06_dynamic_risk_model.py
07_build_itransformer_dataset.py
08_train_itransformer.py
09_build_market_state_card.py
10_generate_llm_scenario_views.py
11_validate_llm_outputs.py
12_postprocess_llm_confidence.py
13_build_ep_soft_constraints.py
14_entropy_pooling.py
15_merge_base_overlay_weights.py
16_skfolio_optimization.py
17_regime_detection.py
18_backtest.py
19_attribution.py
20_run_ablation_tests.py
21_generate_acceptance_report.py
```

---

# 30. 最终给 Codex 的硬性要求摘要

Codex 实现时必须遵守：

```text
1. 所有情景标签必须来自 scenario_thresholds.yaml 或 LLM JSON，不能在代码里硬编码；
2. 所有 Market State Card 字段必须通过 schema 校验；
3. 历史回测模式下不向 LLM 暴露真实日期；
4. LLM 不允许联网，不允许编造新闻和政策；
5. LLM 输出必须 JSON schema 校验；
6. LLM confidence 必须后处理，不能直接使用原始 confidence；
7. EP 第一版使用 soft constraints，不使用强硬约束；
8. 九宫格基础权重和叠加矩阵只作为 skfolio anchor，不直接作为最终交易权重；
9. 风险预算和产品硬约束优先级高于 LLM；
10. 所有回测信号在 T 日收盘后生成，最早 T+1 执行；
11. 交易成本和流动性过滤必须纳入；
12. LLM 模块必须单独评价，不只看最终净值；
13. 所有阶段必须生成 acceptance report；
14. 最终主结果必须来自样本外 walk-forward，不允许全样本拟合。
```

---

# 31. 最终版主策略定义（补强后）

最终策略定义为：

```text
Tushare/Wind 数据底座
→ 多资产收益矩阵与特征矩阵
→ 静态基准与动态风险矩阵
→ iTransformer 预测未来20日资产占优概率、收益和排序分数
→ 生成 Market State Card
→ LLM 在日期隐藏和 schema 约束下输出情景标签、relative views 和约束建议
→ confidence 后处理与冲突检查
→ 九宫格基础权重 + 叠加矩阵生成 preliminary anchor
→ Entropy Pooling soft constraints 生成后验情景概率
→ skfolio 在 CVaR、换手率、硬约束、风险预算和 anchor 下优化最终权重
→ T+1 执行回测
→ 资产归因、因子归因、策略流程归因、LLM 独立评价和消融实验
```

一句话：

> 本项目不是让 LLM 直接做投资决策，而是让 LLM 在严格输入、严格 schema、严格置信度后处理和严格回测防泄露约束下，把 iTransformer 与市场状态翻译为可检验、可落库、可回测的结构化 views 和约束建议，最终仍由 EP 和 skfolio 完成概率融合与组合优化。

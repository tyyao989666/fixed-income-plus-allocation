# 固收+多资产动态配置项目：数据完整性检查指令（给 Codex）

## 0. 你的任务

请你作为代码审查与数据审计助手，检查当前项目是否已经具备运行《固收+多资产动态配置：iTransformer + LLM 情景矩阵 + Entropy Pooling + skfolio》所需的全部数据。

你不要重写策略逻辑，不要直接训练模型，不要直接跑完整回测。  
你的任务是：

```text
1. 扫描项目目录；
2. 检查原始数据是否齐全；
3. 检查清洗后数据是否齐全；
4. 检查字段是否齐全；
5. 检查数据频率、时间范围、缺失率、数据源、未来函数风险；
6. 检查能否生成 iTransformer、LLM Market State Card、EP 和 skfolio 所需输入；
7. 输出一份 data_completeness_report.md 和 data_completeness_report.xlsx。
```

最终请明确回答：

```text
当前数据是否足够进入下一步建模？
如果不够，缺什么数据？
缺失数据应该优先从 Tushare 抓，还是从 Wind 抓？
哪些字段可以先用替代变量？
哪些字段必须补齐后才能继续？
```

---

## 1. 数据源优先级

数据源优先级必须遵循：

```text
Tushare 优先
Wind 兜底
本地缓存优先于重复请求
```

检查时请确认每张表都有以下字段或元信息：

```text
date
code
data_source
fetch_time 或 update_time
```

如果某张表是派生表，需要有：

```text
derived_from
calculation_method
asof_date 或 effective_date
```

如果是宏观数据，必须检查：

```text
data_date：数据所属期
release_date：实际披露日
effective_date：策略可使用日
```

禁止使用未来数据。

---

## 2. 需要检查的核心目录

请优先检查以下目录是否存在：

```text
config/
data_raw/
data_raw/tushare/
data_raw/wind/
data_raw/manual/
data_processed/
output/
output/tables/
output/figures/
output/reports/
output/models/
scripts/
src/
```

如果目录不存在，请在报告中标记：

```text
status = missing_directory
```

---

## 3. 需要检查的配置文件

请检查 `config/` 下是否有以下配置文件：

```text
data_source_config.yaml
tushare_fields.yaml
wind_fields.yaml
wind_codes.yaml
asset_universe.yaml
code_mapping.yaml
industry_etf_mapping.yaml
credit_rating_mapping.yaml
credit_lgfv_industry_mapping.yaml
cb_style_classification_rules.yaml
cb_index_constituent_rules.yaml
scenario_thresholds.yaml
market_state_card_schema.yaml
confidence_rules.yaml
transaction_cost.yaml
constraint_priority.yaml
feature_config.yaml
model_config.yaml
optimization_config.yaml
backtest_config.yaml
```

每个配置文件需要检查：

```text
是否存在；
是否为空；
是否能被 Python 正常读取；
是否包含必要字段；
是否存在重复 code；
是否存在未映射资产；
是否存在字段名和代码硬编码在脚本中，而不是配置文件中。
```

---

## 4. 第一优先级：资产池行情数据

这是最重要的数据。没有资产收益矩阵，后面所有模型都不能运行。

请检查是否能生成：

```text
data_processed/asset_universe.parquet
data_processed/asset_returns_daily.parquet
data_processed/asset_returns_monthly.parquet
```

### 4.1 总体检查

`asset_returns_daily.parquet` 至少需要包含：

```text
date
asset_code
asset_name
asset_class
close 或 nav 或 index_level
return_1d
data_source
```

必须检查：

```text
时间范围是否覆盖 2018-01-01 至今，或项目配置中的 start_date 至 end_date；
是否所有资产按交易日对齐；
每个资产缺失率；
是否存在重复 date-code；
收益率是否存在异常极端值；
新上市资产是否只在上市后进入样本；
是否存在未来填充。
```

### 4.2 五大资产模块必须覆盖

必须至少覆盖以下五大模块：

```text
1. 利率债
2. 信用债
3. 可转债
4. 权益
5. 另类资产
```

每个资产都需要有：

```text
asset_code
asset_name
asset_class
module
sub_module
data_source
tradable_proxy
start_date
end_date
```

---

## 5. 利率债数据检查

### 5.1 必备资产

请检查资产池中是否有以下 4 个利率债资产：

```text
rate_1_3y
rate_3_5y
rate_7_10y
rate_30y
```

每个资产需要日频行情或指数点位：

```text
date
code
open
high
low
close 或 index_level 或 nav
volume 可选
amount 可选
return_1d
data_source
```

### 5.2 利率曲线特征

请检查是否具备以下利率曲线数据：

```text
1Y 国债收益率
3Y 国债收益率
5Y 国债收益率
10Y 国债收益率
30Y 国债收益率
10Y - 1Y 期限利差
30Y - 10Y 超长端利差
2 × 5Y - 1Y - 10Y 蝶式因子
TS 国债期货
TF 国债期货
T 国债期货
TL 国债期货
国债期货成交量
国债期货持仓量
```

### 5.3 利率债情景标签所需字段

为了判断 R1-R6，请检查是否能计算：

```text
rate_1y_change_20d
rate_3y_change_20d
rate_5y_change_20d
rate_10y_change_20d
rate_30y_change_20d
term_spread_10y_1y
term_spread_30y_10y
rate_vol_20d
rate_vol_60d
long_duration_return_20d
long_duration_crowding_percentile
```

如果无法计算 `long_duration_crowding_percentile`，请标记为：

```text
status = missing_optional_but_recommended
```

并建议用以下变量替代：

```text
30Y利率债过去60日收益分位
TL国债期货成交量分位
TL国债期货持仓量分位
30Y-10Y利差历史分位
```

---

## 6. 信用债数据检查

### 6.1 必备信用债资产

请检查资产池中是否有以下 6 个信用债资产：

```text
credit_lgfv_aaa
credit_lgfv_aa_plus
credit_lgfv_aa
credit_ind_aaa
credit_ind_aa_plus
credit_ind_aa
```

每个资产需要日频或周频指数点位：

```text
date
code
index_level 或 nav 或 close
return_1d
rating
sector_type: lgfv / industrial
data_source
```

### 6.2 信用特征

请检查是否具备：

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
信用债成交额
信用债收益率
信用债久期
```

### 6.3 信用状态 CR1-CR7 所需字段

必须能计算：

```text
credit_spread_percentile
credit_spread_change_20d
aa_plus_aaa_spread_percentile
aa_aaa_spread_percentile
lgfv_spread_percentile
industrial_spread_percentile
lgfv_industrial_spread_percentile
credit_tail_risk_score
credit_net_financing_20d
credit_maturity_pressure_30d
credit_cancelled_issuance_30d
```

如果 `credit_tail_risk_score` 不存在，请检查是否能用以下字段合成：

```text
信用利差快速走阔
AA-AAA等级利差快速走阔
取消发行规模上升
信用债成交额下降
信用债净融资转负
弱资质信用债收益率上行
```

### 6.4 信用债检查结论

请判断当前数据能否区分：

```text
CR1 高利差低风险
CR2 利差压缩中段
CR3 利差过低
CR4 高利差高风险
CR5 利差低且风险升
CR6 城投优于产业
CR7 产业优于城投
```

如果不能，请明确缺哪些字段。

---

## 7. 可转债数据检查

这是最重要的模块之一，请重点检查。

### 7.1 可转债个券原始数据

请检查是否存在个券级别数据，至少包括：

```text
trade_date
cb_code
cb_name
open
high
low
close
volume
amount
outstanding_balance
conversion_price
conversion_value 或 parity
conversion_premium_rate
pure_bond_value
pure_bond_premium_rate
ytm
rating
stock_code
stock_name
stock_close
stock_total_mv
stock_float_mv
listing_date
delisting_date
remaining_maturity
redeem_status
call_announcement_date
suspend_status
data_source
```

Tushare 优先字段：

```text
可转债基础信息
可转债日行情
转股溢价率
正股行情
正股市值
```

Wind 兜底字段：

```text
纯债价值
纯债溢价率
YTM
余额
评级
强赎状态
退市/摘牌状态
```

### 7.2 转债流动性过滤

请检查是否有规则或字段支持：

```text
上市未满 N 日剔除
余额过低剔除
成交额过低剔除
停牌剔除
强赎公告后特殊处理
退市/摘牌处理
高价高溢价转债单独标记
低流动性转债降权
```

必须检查：

```text
是否有 cb_liquidity_filter；
是否有 cb_universe_by_date；
是否每个交易日只使用当日可得信息；
是否强赎和退市处理没有未来函数。
```

### 7.3 自建 12 个转债子指数

请检查是否能生成以下 12 个子指数：

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

分类维度：

```text
转债风格：偏债 / 平衡 / 偏股
正股市值：大盘 / 小盘
价格水平：低价 / 高价
```

请检查分类规则是否写在：

```text
config/cb_style_classification_rules.yaml
config/cb_index_constituent_rules.yaml
```

不要把分类阈值硬编码在脚本中。

### 7.4 转债市场特征

请检查是否能计算：

```text
cb_index_return
cb_price_median
cb_price_mean
cb_conversion_premium_median
cb_pure_bond_premium_median
cb_ytm_median
low_price_cb_ratio
high_price_cb_ratio
double_low_cb_ratio
high_price_high_premium_cb_ratio
low_price_low_premium_cb_ratio
cb_turnover
cb_amount
cb_market_balance
new_cb_issuance_amount
forced_redemption_amount
cb_premium_percentile
cb_price_median_percentile
cb_ytm_percentile
cb_turnover_percentile
```

### 7.5 转债情景 CB1-CB5 检查

请判断是否能支持：

```text
CB1 低估高性价比
CB2 合理偏便宜
CB3 合理偏贵
CB4 高估低性价比
CB5 极端拥挤
```

如果缺字段，请按重要性排序：

```text
一级必需：
转股溢价率中位数
价格中位数
YTM中位数
低价转债占比
高价转债占比

二级推荐：
纯债溢价率
转债市场余额
成交额/换手率
强赎退出规模
新券发行规模
```

---

## 8. 权益数据检查

### 8.1 权益资产池

请检查是否覆盖以下权益模块：

```text
equity_tech_growth
equity_manufacturing
equity_consumption
equity_healthcare
equity_fin_property
equity_cyclical
equity_dividend_utility
equity_broad
```

可以使用行业 ETF 或行业指数代理。

### 8.2 权益市场特征

请检查是否具备：

```text
沪深300收益
中证500收益
中证1000收益
创业板指收益
红利指数收益
万得全A收益
行业ETF收益
A股成交额
A股换手率
两融余额
PE分位数
PB分位数
股息率
ERP / 股债性价比
权益波动率
行业扩散度
成长/红利相对强弱
大盘/小盘相对强弱
周期/成长相对强弱
消费医药相对强弱
```

### 8.3 权益情景 EQ1-EQ6 检查

请判断是否能支持：

```text
EQ1 低估修复
EQ2 成长趋势
EQ3 红利防御
EQ4 周期通胀
EQ5 高估拥挤
EQ6 行业轮动
```

关键字段：

```text
equity_valuation_percentile
equity_erp_percentile
equity_vol_percentile
growth_value_momentum
dividend_growth_momentum
small_large_momentum
cyclical_growth_momentum
industry_dispersion_score
industry_breadth
```

---

## 9. 另类资产数据检查

### 9.1 另类资产池

请检查是否覆盖：

```text
alt_reits
alt_gold
alt_commodity
alt_us_bond
```

对应数据：

```text
REITs指数或REITs基金行情
黄金ETF行情
商品ETF行情或南华商品指数
QDII美债基金净值或海外债券指数
```

### 9.2 海外与另类特征

请检查是否具备：

```text
gold_price
gold_return
commodity_index_return
nanhua_commodity_index
reits_return
us_bond_return
us_10y_yield
us_real_rate
vix
dxy
usd_cny
global_risk_off_score
```

### 9.3 ALT1-ALT6 检查

请判断是否能支持：

```text
ALT1 黄金避险
ALT2 商品通胀
ALT3 美债占优
ALT4 REITs修复
ALT5 另类中性
ALT6 股债双杀保护
```

如果缺少 REITs 或 QDII 美债数据，可以先标记为：

```text
optional_for_first_version
```

但黄金和商品建议必须有。

---

## 10. 流动性数据检查

请检查是否具备：

```text
DR007
R007
R007 - DR007
Shibor 3M
1Y AAA 同业存单收益率
同业存单收益率变化
公开市场净投放
MLF 利率
逆回购利率
资金利率分位数
非银压力指标
liquidity_stress_score
```

必须能计算：

```text
dr007_change_20d
r007_dr007_spread_percentile
cd_1y_change_20d
omo_net_injection_20d
liquidity_stress_score
```

这些字段用于：

```text
判断流动性冲击；
压缩权益和转债上限；
判断信用债和转债流动性风险；
生成 Market State Card。
```

---

## 11. 宏观与海外数据检查

请检查是否具备：

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
美国10年期国债收益率
美国实际利率
VIX
人民币汇率
黄金价格
南华商品指数
```

所有宏观数据必须具备：

```text
data_date
release_date
effective_date
```

请检查是否存在未来函数：

```text
如果某个宏观数据在 release_date 之前被使用，必须标记为 lookahead_bias。
```

---

## 12. iTransformer 数据检查

### 12.1 输入特征

请检查是否能生成：

```text
data_processed/feature_panel_daily.parquet
data_processed/feature_panel_monthly.parquet
data_processed/itransformer_dataset.parquet
```

输入特征至少包括：

```text
资产自身特征
利率曲线特征
信用利差特征
转债估值与供需特征
权益行业与估值特征
流动性特征
宏观与海外特征
```

资产自身特征：

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

### 12.2 目标变量

请检查是否能生成：

```text
future_return_20d
future_return_60d 可选
outperform_fixed_income_floor_20d
cross_section_rank_20d
top_k_label_20d
downside_risk_20d
```

核心标签：

```text
y_return_i_t = asset_i 未来20日收益率

target_i_t = 1 if asset_i未来20日收益 > 固收底仓未来20日收益 else 0

rank_i_t = asset_i未来20日收益在资产池中的排名
```

### 12.3 未来函数检查

必须确认：

```text
rolling 特征只使用过去数据；
目标变量只用于训练标签，不进入当期特征；
标准化使用 rolling z-score 或训练集拟合；
walk-forward 划分中没有随机打乱时间；
T 日收盘生成信号，最早 T+1 执行。
```

---

## 13. LLM Market State Card 数据检查

请检查是否能生成：

```text
data_processed/market_state_cards.parquet
```

每个 rebalance date 必须有一张 Market State Card。

### 13.1 iTransformer 输出字段

```text
p_rate_1_3y_outperform
p_rate_3_5y_outperform
p_rate_7_10y_outperform
p_rate_30y_outperform
p_credit_outperform
p_cb_balanced_low_price_outperform
p_equity_growth_outperform
p_gold_outperform
expected_return_i_next_20d
rank_score_i_next_20d
```

### 13.2 风险状态字段

```text
equity_vol_percentile
cb_vol_percentile
bond_vol_percentile
cross_asset_correlation_percentile
portfolio_cvar
market_cvar_state
regime_probability_low_vol_low_corr
regime_probability_high_vol_high_corr
```

### 13.3 估值状态字段

```text
cb_premium_percentile
cb_price_median_percentile
cb_ytm_percentile
equity_valuation_percentile
equity_erp_percentile
credit_spread_percentile
long_duration_crowding_percentile
```

### 13.4 流动性状态字段

```text
dr007_change_20d
r007_dr007_spread_percentile
cd_1y_change_20d
omo_net_injection_20d
liquidity_stress_score
```

### 13.5 组合状态字段

```text
current_portfolio_weight
current_drawdown
drawdown_limit
remaining_risk_budget
turnover_last_month
current_cvar
risk_asset_weight
```

### 13.6 LLM 防泄露检查

请检查历史回测中是否隐藏真实日期：

```text
date_visible_to_llm = false
```

严格回测模式下，Market State Card 不应把真实 calendar date 传给 LLM，只保留：

```text
rebalance_id
horizon
asof_mode
date_visible_to_llm
```

真实日期可以保留在 audit log，不给 LLM 输入。

---

## 14. Entropy Pooling 数据检查

请检查是否能生成：

```text
data_processed/llm_outputs.parquet
data_processed/ep_views.parquet
data_processed/ep_scenario_weights.parquet
data_processed/posterior_return_distribution.parquet
```

### 14.1 LLM 输出检查

每期 LLM 输出必须包含：

```text
scenario_summary
core_stock_bond_scenario_weights
overlay_states
relative_attractiveness
relative_views
ep_views
constraint_adjustment
confidence
conflict_flags
fallback_used
prompt_version
model_version
schema_pass
```

### 14.2 EP views 检查

每条 view 必须有：

```text
view_id
view_type
asset_a
asset_b
relation
confidence_raw
confidence_final
horizon
reason
source: llm / rule_based / itransformer
```

### 14.3 EP 后验概率检查

`ep_scenario_weights.parquet` 必须满足：

```text
每个 rebalance date 的 posterior_weight 总和 = 1
posterior_weight >= 0
不存在 NaN
如果 EP 不可行，必须 fallback 到 prior 或 rule-based soft views
```

---

## 15. skfolio 优化数据检查

请检查是否能生成：

```text
data_processed/optimized_weights.parquet
data_processed/portfolio_backtest.parquet
```

skfolio 所需输入：

```text
资产收益矩阵
EP 后验情景概率
动态协方差矩阵
交易成本
仓位上下限
换手率约束
CVaR 阈值
风险预算约束
资产分组约束
```

每期优化权重必须满足：

```text
sum(weights) = 1
weights >= 0
单一资产上限
模块权重上下限
权益上限
转债上限
信用下沉上限
另类资产上限
CVaR 约束
换手率约束
```

如果无法满足约束，请记录：

```text
optimization_status
failed_reason
fallback_portfolio
```

---

## 16. 交易成本和流动性摩擦检查

请检查是否有：

```text
config/transaction_cost.yaml
```

建议成本字段：

```text
rate_bond_etf: 0.5bp - 2bp
credit_bond_fund: 2bp - 5bp
convertible_bond: 5bp - 10bp
equity_etf: 3bp - 5bp
reits: 5bp - 10bp
qdii_us_bond: 10bp - 20bp
```

请检查回测中是否真的扣除了交易成本：

```text
turnover
transaction_cost
net_return_after_cost
```

---

## 17. 最终需要生成的数据文件

请检查以下文件是否存在、是否非空、字段是否齐全：

```text
data_processed/asset_universe.parquet
data_processed/asset_returns_daily.parquet
data_processed/asset_returns_monthly.parquet
data_processed/feature_panel_daily.parquet
data_processed/feature_panel_monthly.parquet
data_processed/covariance_forecasts.parquet
data_processed/itransformer_dataset.parquet
data_processed/itransformer_predictions.parquet
data_processed/market_state_cards.parquet
data_processed/llm_outputs.parquet
data_processed/ep_views.parquet
data_processed/ep_scenario_weights.parquet
data_processed/posterior_return_distribution.parquet
data_processed/optimized_weights.parquet
data_processed/portfolio_backtest.parquet
data_processed/asset_contribution.parquet
data_processed/factor_attribution.parquet
```

---

## 18. 数据完整性评分

请给每一类数据打分：

```text
0 = 完全缺失
1 = 只有部分原始数据，无法建模
2 = 能生成基础收益矩阵，但特征不足
3 = 能支持静态基准和部分动态风险模型
4 = 能支持 iTransformer + LLM + EP + skfolio 主链条
5 = 数据完整，且通过质量检查、未来函数检查和回测输入检查
```

评分维度：

```text
资产池行情
利率债数据
信用债数据
可转债数据
权益数据
另类资产数据
流动性数据
宏观海外数据
iTransformer 数据集
Market State Card
EP 数据
skfolio 优化数据
交易成本
回测结果
归因数据
```

---

## 19. 最终报告格式

请输出：

```text
output/reports/data_completeness_report.md
output/reports/data_completeness_report.xlsx
```

报告必须包括以下部分：

```text
1. 总体结论
2. 数据完整性总分
3. 哪些数据已具备
4. 哪些数据缺失
5. 哪些字段缺失但可用替代变量
6. 哪些字段缺失会阻断后续建模
7. Tushare 可补数据清单
8. Wind 必须补的数据清单
9. 未来函数风险检查结果
10. 缺失率和异常值检查结果
11. LLM Market State Card 是否可生成
12. iTransformer 数据集是否可生成
13. EP 和 skfolio 输入是否可生成
14. 下一步优先修复清单
```

---

## 20. 输出一个机器可读 JSON 总结

除了 Markdown 和 Excel 报告，请额外输出：

```text
output/reports/data_completeness_summary.json
```

JSON 格式：

```json
{
  "overall_status": "pass / partial_pass / fail",
  "overall_score": 0,
  "blocking_issues": [],
  "missing_required_data": [],
  "missing_optional_data": [],
  "tushare_to_fetch": [],
  "wind_to_fetch": [],
  "files_existing": [],
  "files_missing": [],
  "lookahead_bias_risks": [],
  "can_run_static_baseline": false,
  "can_run_dynamic_risk_model": false,
  "can_train_itransformer": false,
  "can_generate_market_state_card": false,
  "can_run_llm_views": false,
  "can_run_entropy_pooling": false,
  "can_run_skfolio_optimization": false,
  "can_run_backtest": false,
  "next_priority_actions": []
}
```

---

## 21. 最终检查结论模板

最后请用以下格式输出结论：

```text
结论：

当前数据完整性状态：pass / partial_pass / fail

可以运行：
- [ ] 静态配置基准
- [ ] 动态风险矩阵
- [ ] iTransformer
- [ ] LLM Market State Card
- [ ] Entropy Pooling
- [ ] skfolio 优化
- [ ] 完整回测
- [ ] 归因分析

最大阻断项：
1.
2.
3.

优先补充数据：
1.
2.
3.

Tushare 优先补：
1.
2.
3.

Wind 必须补：
1.
2.
3.

可用替代变量：
1.
2.
3.
```

---

## 22. 重要提醒

请严格遵守：

```text
不要用未来数据；
不要随机切分时间序列；
不要把真实历史日期暴露给 LLM 回测输入；
不要让 LLM 直接输出最终权重；
不要让 LLM 覆盖产品硬约束、风险预算和流动性冲击约束；
不要忽略转债高估值导致的低性价比；
不要把信用利差高简单等同于信用债便宜；
不要把债强简单等同于无脑买 30Y。
```

请先完成数据检查，再决定是否进入建模阶段。

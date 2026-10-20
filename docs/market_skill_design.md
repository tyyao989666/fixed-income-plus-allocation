# MarketSkill v4.3 设计说明

MarketSkill 借鉴 PokerSkill 的核心思想：先用确定性上下文引擎读取状态，再把 LLM 或机械 views 限制在规则允许的窄空间内。第一版不调用 LLM，不输出最终仓位，只生成状态卡和已校验的相对 views。

## 架构

1. Context Engine  
   从 `feature_panel_daily_v43.parquet` 计算 CR/R/CB/EQ/ALT/RB 状态、风险标签、九宫格环境、可投范围和 view filters；同时读取 `asset_returns_daily.parquet`，为正式数据中的 38 个资产生成 Point-in-Time mini-card。

2. Skill Retrieval  
   按当前状态从 `config/market_skill_config.yaml` 选择 P1-P5 技能片段：
   - P1：永久硬规则与输出约束
   - P2：风险预算基石原则
   - P3：固收+稳定原则
   - P4：当前状态对应规则
   - P5：压力情景 override

3. Bounded Views  
   机械 views 只表达可投袖套之间的相对方向，例如 `rates_long_duration > equity`。Validator 会拦截违反状态边界的 views。现金不是观点资产，也不能被任何 view 配出正权重。

4. EP-ready Views  
   `ep_views_daily_v1.parquet` 只保留 validator 通过的 views，供后续 Entropy Pooling 或优化器使用。

5. Asset mini-cards  
   每张资产卡包含 5/20/60 日收益、20 日波动、60 日回撤、历史覆盖率、陈旧天数、252 日因果分位、模块内排名、模块状态、硬过滤和 `overweight_eligible`。上市前或历史不足的资产保持 unavailable，不做未来回填。资产趋势必须与模块状态和 view filters 共同解释，不能单独越过信用、利率、转债、权益或另类的硬规则。

6. 口径一致性  
   `eligible_universe` 直接由 `view_filters` 的反向布尔值生成，避免同一状态同时出现允许和禁止超配。转债市场中位转股溢价率的原始单位为百分数，估值评分前按配置除以 100；状态卡同时保留原始百分数和标准化小数供审计。

## 输出

- `data_processed/market_state_cards_daily_v1.parquet`
- `data_processed/market_skill_views_daily_v1.parquet`
- `data_processed/ep_views_daily_v1.parquet`
- `output/reports/market_state_card_latest.md`
- `output/reports/market_skill_views_report.md`

## 边界

- 不生成最终组合权重。
- 不用未发布宏观数据。
- 不把 proxy 信号包装成真值。
- 不把 LLM 自由文本直接送入优化器。

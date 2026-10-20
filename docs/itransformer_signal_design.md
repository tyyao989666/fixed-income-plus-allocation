# 日频 iTransformer + PPO 信号设计

## 1. 定位

iTransformer 是预测专家，不是仓位生成器。风险预算决定底仓；股债九宫格和六模块状态决定边界；Market Views 给出可解释方向；EP/skfolio 负责受约束求解；PPO 只在预先计算的可行候选中路由。

最新一体化入口是 `scripts/129_run_integrated_daily_view_ppo.py`。旧的月末联合层与 `scripts/128_run_daily_adaptive_pipeline.py` 仍保留作消融对照，但不代表当前推荐架构。

## 2. 数据与标签

- 输入窗口：决策日及以前 120 个中国交易日。
- 变量：64 个 Point-in-Time 特征，覆盖收益、动量、波动、回撤、利率曲线、信用利差、资金面、转债、权益估值、黄金、商品、美债和 REITs。
- 输出：5 个袖套 × 5/20/60 日，共 15 个风险调整相对分数。
- 袖套：`rates`、`credit`、`convertible_bonds`、`equity`、`alternatives`。
- 标签：下一交易日起的未来复合收益，减去仅用过去 252 日计算的期望收益，再除以过去 60 日波动并做横截面去均值。
- 最长标签只有在 `horizon_label_end_date` 到达后，才能进入训练、验证或可靠度统计。

30Y 利率债继续使用中国国债收益率曲线构造的总收益代理，不冒充 ETF 实盘价格。缺少真实成交量的代理资产不伪造 OHLCV。

## 3. iTransformer

当前直接 challenger：

```yaml
d_model: 64
n_heads: 4
num_layers: 2
dropout: 0.25
epochs: 300
```

输入张量为 `batch × 120 × 64`。模型把变量而不是时间点作为 token，加入变量身份 embedding，再用 15 个袖套/期限 query 读取编码结果。损失由 SmoothL1 与五袖套横截面 pairwise rank loss 组成。300 是上限，每折由独立调参段 early stopping，不能把评估段用于选 epoch。

## 4. Champion-Challenger

直接 iTransformer 的 2022/2023/2024 折外 RankIC 约为 `-0.030/+0.283/-0.105`，跨年不稳定。因果动量基线的 CV 中位 RankIC 约为 `0.081`，2025+ 约为 `0.152`。

因此部署时同时保留基线与模型：

```text
incremental_reliability = matured_model_rank_ic - matured_baseline_rank_ic
iTransformer_weight = clip((incremental - 0.02) / (0.20 - 0.02), 0, 1) × 0.75
hybrid_forecast = baseline + iTransformer_weight × (model - baseline)
```

所有 RankIC 只使用标签已经成熟的历史。模型相对基线没有正增量时，iTransformer 动作被 mask。

排名优先残差实验使用更高排序损失、0.35 dropout 和更强正则，折外 RankIC 为 `0.077/0.071/0.105`；但 CV 中位数仍低于基线，折残差权重为 `0/0.50/0`，最终权重为 0，所以不替换当前预测。

## 5. 九宫格优先

每个决策日先计算股债九宫格，再处理预测：

1. 九宫格形成债券/权益方向和状态持续时间。
2. `view_filters` 对信用、转债、权益和长久期利率做硬阻断。
3. 38 张资产卡汇总为五袖套均值、离散度、正负状态比例和可投资比例。
4. 预测为正时按九宫格兼容度衰减；被硬阻断的正向观点归零。
5. 预测只进入 `posterior_mu`，协方差仍来自过去 252 日收益。

九宫格不是 PPO 的事后标签，而是动作生成前的边界层。

## 6. PPO

PPO 使用两层 64 维 categorical Actor-Critic，种子为 17/29/47。动作共 9 个：

```text
noop
formal_anchor_20d
rules_only_20d
baseline_5d / baseline_20d / baseline_60d
model_5d / model_20d / model_60d
```

模型动作的强度不是额外离散档位，而由事前可靠度连续决定。三个种子先对 `noop/formal/rules/baseline/model` 家族投票，再在获胜家族内选择平均概率最高的期限，避免稀疏类别把模型票拆散。

每个自然年年初扩展训练，只使用该年开始前已经成熟的标签，年度内冻结。行为克隆老师在训练期逐动作比较成本后奖励；类别权重使用平方根逆频率，防止少数模型动作完全消失。至少两个种子验证通过才允许 PPO 接管。

当前 v6 采用产品化部署门：低波和中波默认 `formal_anchor`，允许 `formal/baseline/model/noop`；高波默认 `rules_only`，允许全部五个家族。低波/中波的所有候选组合仍使用机械 Market Views，只是 PPO 不能直接选择裸规则路径。候选必须相对默认动作有至少 `0.002` 的验证奖励优势，家族概率至少 `0.40`，且领先第二名至少 `0.08` 才能覆盖默认动作。

## 7. 奖励与组合求解

奖励以正式策略为基准：

```text
excess compound return
- downside worsening
- profile-specific relative volatility worsening
- profile-specific relative drawdown worsening
- active deviation penalty
```

低波惩罚最严格，高波最宽松。交易使用 T+1，换手成本 5bp。正式锚点是时变路径，已验证在连续 1,040 日全锚点测试中逐日收益差异为 0。

EP/skfolio 采用 MeanRisk + CLARABEL。为避免均值观点被惩罚项完全压平，日频挑战者使用产品化偏离上限：low/balanced/high 分别为 1.2%/2.2%/3.6%。最终再应用趋势刹车、袖套上下限、九宫格硬约束和无现金约束。

## 8. 当前结果

共同区间 2024-01-01 起：

| 产品 | 年化收益 | Sharpe | 最大回撤 | 相对正式收益 | 相对正式 Sharpe |
|---|---:|---:|---:|---:|---:|
| low_vol | 6.377% | 3.095 | -1.419% | +0.196% | +0.090 |
| balanced | 8.032% | 2.647 | -2.189% | +0.105% | +0.029 |
| high_vol | 15.146% | 1.973 | -5.861% | +0.532% | +0.027 |

Point-in-Time、T+1、权重和硬约束审计 0 失败。低波/中波相对规则 Sharpe 为 `+0.080/+0.036`，高波为 `-0.009`。正式门禁仍为 `false`，另一个原因是低/中/高波的 iTransformer 增量动作只有 `0/1/1` 次。正确表述是“完整可运行的研究挑战者”，不是已证明的稳定 Alpha。2024+ 已用于多轮诊断，下一步应冻结 v6 做新增样本前瞻检验，而不是继续在同一区间网格搜索。

## 9. 运行与输出

```powershell
python scripts\129_run_integrated_daily_view_ppo.py
```

主要输出：

```text
data_processed/integrated_daily_view_ppo_{profile}_decisions.parquet
data_processed/integrated_daily_view_ppo_{profile}_asset_weights_daily.parquet
output/reports/integrated_daily_view_ppo_{profile}_daily.csv
output/reports/integrated_daily_view_ppo_three_profile_summary.csv
output/reports/integrated_daily_view_ppo_point_in_time_audit.csv
output/reports/integrated_daily_view_ppo_formal_gate.json
output/reports/integrated_daily_view_ppo_method_and_results.md
output/reports/integrated_daily_view_ppo_iteration_comparison.csv
output/reports/integrated_daily_view_ppo_iteration_notes.md
output/reports/integrated_daily_view_ppo_annual_robustness.csv
output/reports/integrated_daily_view_ppo_annual_action_counts.csv
output/reports/integrated_daily_view_ppo_visual_report.html
```

## 10. 无固定期限的事件驱动分支

### 10.1 为什么另建分支

v6 的 `5/20/60` 同时出现在预测期限和动作期限中，虽然 PPO 会在不同期限间路由，下一次允许换仓的时间仍带有预注册日历。事件驱动 v2 将两者拆开：

```text
5/20/60 日 iTransformer/因果预测
  -> 按历史已成熟可靠度、预测分散度和九宫格一致性动态融合
  -> EP/skfolio 每日生成唯一可行目标 adaptive_fused
  -> PPO 每日判断继续持有旧仓或终止并切换目标
  -> T+1 执行、5bp 成本、产品硬约束
```

因此 `5/20/60` 只是预期收益的观察尺度，不再决定仓位必须持有多少天。系统没有最大持有期，也没有月末强制调仓。

### 10.2 终止状态与训练

终止控制器读取当日 Point-in-Time 状态、九宫格、预测可靠度，以及持仓路径变量：持有年龄、建仓以来收益、相对 v6 收益、持仓内回撤、EWMA 波动、上次动作、当前仓位到新目标的距离。交易成本直接进入逐日奖励，使 `hold` 成为有经济含义的无交易区间。

每个自然年年初扩展训练，训练数据只允许使用该年开始前已经成熟的标签；年度内 checkpoint 冻结。信号日为 `t`，执行与收益从 `t+1` 开始。PPO 不能改变风险预算、九宫格硬过滤、袖套上下限、无现金约束或 EP/skfolio 可行域。

### 10.3 两次结构实验

- v1 family-options：联合学习终止和 formal/rules/baseline/model 目标家族。结果坍缩到 formal/rules，预测目标落仓次数为零，因此拒绝。
- v2 unified-target：iTransformer 增强的 EP/skfolio 目标先统一生成，PPO 只做二元 `hold/adaptive_fused`。它解决了家族坍缩，预测目标在低/中/高波分别出现 209/216/223 日。

### 10.4 验证结果

2024 年起交易成本后：

| 产品 | 事件驱动年化 | 事件驱动 Sharpe | 最大回撤 | 相对 v6 Sharpe |
|---|---:|---:|---:|---:|
| low_vol | 6.366% | 3.004 | -1.562% | -0.091 |
| balanced | 7.842% | 2.549 | -2.256% | -0.098 |
| high_vol | 14.819% | 1.929 | -6.031% | -0.045 |

实际全样本调仓间隔为 1–129、1–136、1–224 个交易日，每个产品有 19–26 种间隔，说明固定期限已被真正移除。27 项 Point-in-Time、标签成熟、T+1、同样本和权重审计全部通过。

但进一步按策略来源拆分后，PPO 触发再平衡的中位间隔仍为 1 日，呈现“长时间 hold + 连续调仓簇”。20 日配对区块 bootstrap 中，相对 v6 的 Sharpe 增量为正概率仅为 19.3%/9.6%/3.5%。因此事件驱动 v2 是有效的架构验证，不是业绩升级。

### 10.5 运行与结论

```powershell
python scripts\132_run_event_driven_rebalance_ppo.py --config config\event_driven_rebalance_ppo_v2_config.yaml
python scripts\133_analyze_event_driven_rebalance.py
```

完整结论在 `output/reports/event_driven_rebalance_final_assessment.md`。正式保留 v6；下一步若继续研究，应增加更长 Point-in-Time 历史，并优先比较低维生存/危险率终止模型和带置信下界的离线 RL，而不是继续在已观察的 2024+ 区间搜索 epoch、阈值或奖励权重。

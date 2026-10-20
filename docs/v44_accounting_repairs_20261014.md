# 固收+ v44 修复说明

本版本修正审阅发现的组合会计和求解器问题。历史 v43 的CSV、模型和报告保留，修正版另用v44前缀。预测仍来自已保存的因果 iTransformer 编码器与 Ridge 相对收益头；执行目标和PPO奖励已经改变，旧PPO不能直接复用。

## 2026-10-14后续状态

v44已在2026-10-14完成三产品、三年度、三种子的27组PPO训练，流水线约52分钟；预测模型未在该轮重训。2024年以来低/中/高波年化收益约6.30%/7.82%/13.59%，标准Sharpe约2.969/2.477/1.718。正式组合门仍为False。

后续边界审查发现：skfolio的max_turnover是逐袖套约束，而备用分支使用总L1；自动提高换手上限、投影原上一期权重、失败回退和不可行盒约束也需整改。资产份额截断后归一化不能严格保证75%上限，但该项未在保存的v44目标中触发。mean_rank归一化还会消掉整体置信度。

因此，本文件的通过项只对应当时已覆盖的会计与正常输入测试，不能解释为所有求解器边界通过；下表的后端一致性也不覆盖绑定总L1换手约束。原v44报告和模型保留不改写。新的修复、对照与训练计划见 [v45完整方案](v45_solver_prediction_execution_plan_20261014.md)，目前仅是待执行设计。

## 已修改的行为

| 问题 | 修正版行为 | 验证 |
|---|---|---|
| skfolio没有接收外部协方差 | EmpiricalPrior接入SuppliedCovariance，实际风险目标、日志和SLSQP使用同一矩阵 | 固定行情，仅改变风险矩阵就改变最优权重；两后端含惩罚目标下权重接近 |
| 两后端一个优化相对风险、一个优化绝对风险 | 移除MeanRisk的target_weights，底仓偏离由显式惩罚和约束管理 | 低方差资产增加权重，而不被锁在底仓 |
| A1额外晚执行一日 | 同一交易日收盘形成目标，下一交易日开始贡献收益 | 三日例子验证收益与成本同步 |
| 月内隐含恢复权重却不计成本 | 持仓按真实收益漂移；没有执行则换手与成本为0 | A0/A1/PPO正式锚点均验证自然漂移 |
| A0/A1财富指数、A2曲线代理 | 全部从同一冻结NPZ的38资产矩阵开始；30Y统一为模型原有固定久期曲线代理 | 原始掩码保留，生成统一收益哈希 |
| 风险估计还读取旧原始收益 | 日频目标生成器改读v44冻结长表，含正确的30Y资产名和module | 与执行器同源；不在原始财富指数和曲线代理之间混切 |
| 报告与bootstrap指标不同 | Sharpe统一为日均收益/样本标准差×sqrt(252)，rf=0；CAGR/vol单列；bootstrap年化收益差也改为复合年化差 | 重算及公式测试；初期亏损相对初始NAV=1计算回撤 |
| 缺失日悄悄当0收益 | 已有估值且连续缺口不超过5个本地交易观察日，明确延用上一估值；同时禁止该资产执行；初始无估值、超龄持仓缺口停止 | 缺口CSV台账、未上市前不回填、不能在无报价日买入测试 |
| 月末标签可能对应周末 | 月度输入卡和观点先限制在冻结交易日历，名义月末再映射到当月最后交易日收盘 | 周末月末测试；普通非月末的发布日不向前挪 |
| 教师标签与真实执行会计不同 | 教师动作也用共享会计、成本、基准漂移和报价限制 | 5日监督窗口仍限制在训练阶段；在线不读教师结果 |
| 旧PPO错误复用 | 检查点签名包含会计源代码、控制器参数及当年前历史收益/报价/目标/状态 | 改未来数据不影响签名，改历史数据影响签名；旧无签名模型被拒绝 |
| 组合门可能绕过预测模型正式门 | 正式组合还要求预测正式门和修复后PPO检查点成立 | 无新PPO的机械预览不能晋升正式 |

缺口政策是一项估值约定，不是补齐真实成交数据。休市和数据损坏原因尚未逐一核验，台账明确写为未验证。不能把延用值称为真实观测，也不能据此宣称所有供应商数据严格Point-in-Time。最长5日不是为了选择收益，超限会阻断持仓回测。

全部策略默认从已投资组合开始，不再单独计初次建仓成本；后续真实买卖都按L1换手×5bp计成本。费用被视为从组合价值中按比例扣除，权重漂移用资产收益更新。观察期统一从2022-01-05（第一模型信号2022-01-04的下一交易日）至2026-07-15。

这里的T+1是日级收益记账约定：下一交易观察日的close-to-close收益作用于执行权重。尚未拆分“旧持仓隔夜收益”和“下一日开盘成交后的新持仓盘中收益”，也未模拟成交价/滑点。因曲线和指数代理缺少真实开盘成交数据，这不能称为逐笔或开盘价实盘回测。

日频优化的观点强度与风险矩阵按配置的21日风险缩放进入同一目标周期；字段daily_mu_scale是历史名称，不能解释为已校准的实际日均收益。月度锚点使用月频均值与风险。修复显式传入风险矩阵，不代表观点均值已经获得统计校准。

## 实际验证和运行边界

行为测试覆盖：T+1时点、月内漂移、买卖成本、初始回撤、标准Sharpe、报价缺失、周末月末、外部协方差敏感性、求解后端一致性、未来数据不进入检查点签名、旧检查点拒绝等。

另有合成180日样本上的教师预训练/PPO一轮梯度更新测试，检验损失与参数有限、验证和放行接口贯通；它只验证训练代码，不作为实际市场训练或策略绩效。

修复流程会重建三产品的无Agent月度底仓与MeanRisk目标，重算A0/A1，并以机械执行方式验证模型观点到日目标到成本/净值的全样本路径。机械预览的ppo_days为0；不能把它标成重训后的PPO业绩。预测模型正式门仍由原报告决定，不会因为修复代码而改成True。

## 运行

在项目根目录执行修复后的数据/基线/机械执行验证：

```powershell
& 'C:\Users\lenovo\AppData\Local\Programs\Python\Python312\python.exe' -X utf8 scripts\run_v44_repaired_pipeline.py
```

加入新的PPO正式训练（继承3种子，每年最多120epoch，选模型与放行各10%、20日隔离）：

```powershell
& 'C:\Users\lenovo\AppData\Local\Programs\Python\Python312\python.exe' -X utf8 scripts\run_v44_repaired_pipeline.py --train-ppo
```

编码器的1000epoch上限与PPO的120epoch上限是不同训练任务。本次会计修复不通过增加epoch掩盖评价偏差。

仅测试：`scripts/176_test_accounting_repairs.py`。仅重建基线可在一键入口加`--baselines-only`。新组合配置为`config/hierarchical_asset_itransformer_v44_accounting_fixed.yaml`，继承旧预测配置，组合输出和控制器模型目录独立。

## 产物

- `output/reports/v44_accounting_fixed_input_manifest.json`：冻结收益口径、哈希、月频截止、缺口政策。
- `output/reports/v44_accounting_fixed_pipeline_manifest.json`：本轮各阶段命令和退出码。
- `output/reports/v44_accounting_fixed_validation.md`及`.csv`：同日期、T+1、成本、权重、缺口和Sharpe的输出核验。
- `output/reports/v44_accounting_fixed_environment.json`：本机本轮验证的库版本。
- `output/reports/v44_accounting_fixed_valuation_gaps.csv`：缺口年龄、是否延用、禁止交易标记。
- `output/reports/v44_accounting_fixed_control_variable_baselines.csv`：同口径A0/A1业绩。
- `output/reports/hierarchical_asset_itransformer_v44_accounting_fixed_three_profile_summary.csv`：日频执行预览或新训练结果，需结合本轮manifest辨别。
- `data_processed/v44_frozen_returns_daily.parquet`、`v44_frozen_returns_long.parquet`：统一日频矩阵/长表。
- `data_processed/v44_frozen_returns_monthly.parquet`：同源完整月收益。
- `data_processed/v44_state_cards_daily.parquet`、`v44_mechanical_views_daily.parquet`：冻结交易日历内的规则输入。
- `data_processed/v44_{profile}_target_weights_monthly.parquet`：修正版月度目标。

## 尚不能宣称已解决的事项

宏观真实历史发布日和修订版本、供应商历史复权/退市信息、信用/REITs代理真实性、每个缺口原因均需进一步证据核验。2024+已经参与开发，无法靠重跑恢复为未触碰测试集。完整KL熵池化没有在此次修复中实现，当前应称“均值观点融合+skfolio”。旧外发ZIP保留为历史底稿，它不包含完整独立可运行依赖；本轮没有把它升级成完整工程包。

会计修复提高了部分控制变量比较的可信度，PPO已按v44会计重训；但新发现的求解器契约问题、预测独立增量和未触碰的前瞻验证仍未解决。v45须先修求解器并重算共同基准，不能把修复影响误算为模型提升。

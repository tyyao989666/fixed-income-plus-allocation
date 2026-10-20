# V56 单次后续实验预注册

登记于2026-10-16北京时间夜间审查，训练启动前冻结。尚未启动。

## 原因与证据范围

代码审查发现一个明确的目标不一致，而非依据2024–2026组合收益挑参数：

- V55的F3 Huber和pairwise均作用于标准化残差。
- 部署与验证比较的是F0基准加上还原后的残差，即总观点。
- 残差的排序不一定等于总观点排序。示例：F0为[4,0]，真实总目标为[3,1]，真实残差为[-1,1]。两种排序方向完全相反。
- 仅强迫残差排序可能在模型已给出正确总排序时仍施加反向梯度。新单元测试直接验证这一情形，不使用测试期收益。

2022–2023历史验证阶段也没有显示F3必然优于F2：六个年度×种子中一次相等、一次变差、四次提高。不能宣称已证实排名损失是唯一根因。本实验只验证代码层面已发现的目标一致性问题。

## 唯一学习变更

保留Huber残差拟合，把F3的pairwise项作用到下式：

```text
总观点 = residual_prediction * residual_scale + residual_center + TRAIN_OOF_F0
训练目标 = 原V55机会分数目标
总损失 = 原Huber损失 + 0.2 * 总观点pairwise损失
```

TRAIN_OOF_F0来自同一历史季度OOF基准。计算等价式为：

```text
总观点 = (residual_prediction - standardized_residual_label) * residual_scale
         + original_objective_label
```

该等价式仅在训练损失内使用训练行标签，不作为网络输入、推理特征或未来交易观点。只允许捕获的batch属于本年度train，不能用于validation/gate/test行。masked cell继续屏蔽，非法尺度或非训练批次立即报错。

## 保持不变

同一V55冻结数据及38资产、九宫格、120日窗口、5/20/60目标、两层四头、三种子17/29/47、2022–2026年度划分、purge、早停、1000epoch上限、F0/F1/F2、原候选选择与全部预测门槛。没有新增超参数搜索。GPU单进程、seed_workers=1。

F3此处排序单位仍是V55机会分数，以便与当前验证目标一致；不能声称已经优化真实组合收益或成本后夏普。导出EP观点前仍需反变换，正式PPO仍受执行账本/现金流认证约束。

## 运行与停止条件

原V53和V55均完成且原训练进程退出、V55解码审计通过后才能启动。独立脚本 `scripts/run_v56_total_view_ranking.py`，独立目录 `total_view_rank_v56`，不得覆盖原模型或共享源。全部从头训练，签名包含目标修正、运行脚本、预注册文档和数据哈希，不挪用不兼容检查点。

本自动任务最多跑这一项后续实验，不再新增V57或循环改参。失败先查根因，最多一次同实验恢复。测试期诊断只做描述，不用于改权重或门槛；2024–2026不宣称全新样本外。

完成后核查是否改善历史验证总观点排序、F3相对F2/F0增量、种子稳定性和原门控，再在共同有效样本上报告解码收益误差/IC。无论通过与否均记录，不只展示最好年份。若没有稳定增量，保留“不通过”结论并暂停夜间自动任务，不能强行发布模型。

## 入口

```powershell
& '.venv-gpu-v55/Scripts/python.exe' -X utf8 scripts/test_v56_total_view_ranking.py
& '.venv-gpu-v55/Scripts/python.exe' -X utf8 scripts/run_v56_total_view_ranking.py
& '.venv-gpu-v55/Scripts/python.exe' -u -X utf8 scripts/run_v56_total_view_ranking.py --train --resume
```

最后一条需由隐藏后台进程启动并重定向日志；前两条不是正式训练。GPU不可用或原控制未完成时直接停止，不静默降为CPU。

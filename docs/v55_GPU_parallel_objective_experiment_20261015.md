# V55 学习目标改造与 GPU 并行实验

## 当前实际运行方式

两条独立进程并行，不更改正在训练的对照输入和检查点：

| 分支 | 目标 | 计算设备 | 产物 |
| --- | --- | --- | --- |
| V53 对照 | 未来 5/20/60 日原始相对收益 | 原 CPU 环境 | unified_v53 |
| V55 挑战 | 事前风险调整后、相对自身已成熟历史常态的机会偏离 | RTX 5070 Ti Laptop GPU | temporal_opportunity_v55 |

两条保留 38 资产、120 日序列、九宫格、2 层/4 头/d_model40 的 iTransformer、三种子17/29/47、2022–2026 年度划分以及原验证/门控。保留 1,000 epoch 上限和早停，不改成 5 epoch 冒烟，也不为提高放行比例降低门槛。

只有一张 GPU，默认不让多个训练进程争同一张卡；并行是“CPU 对照＋GPU 新目标”。GPU 分支内部顺序训练各年度和种子。mask-safe 编码器要求 seed_workers=1，不能借 Windows spawn 并行静默恢复旧编码器。

## 新目标的准确含义

原始相对收益记为 x(t,h)。先用资产在信号日前的日收益计算风险尺度：

```text
sigma(t) = std(r[t-60:t])，至少40个有效观察
sigma(t) 至少为 1%/sqrt(252)
scale(t,h) = sigma(t) * sqrt(h)
u(t,h) = x(t,h) / scale(t,h)
location(t,h) = 过去252个信号位置内已成熟 u 的均值 * n/(n+252)
z(t,h) = u(t,h) - location(t,h)
```

location 至少需要126个有效历史标签，且只允许使用标签起点 s≤t-h-1，也就是终点 s+h≤t-1 的标签。不能直接把过去行号中的“未来收益标签”当成当时已经知道。缺失、历史不足继续屏蔽，没有前向或后向填充。

这不是“再套一层残差学习”。F0/F1/F2/F3 原有残差链仍在，但它们共同学习的新目标先消除了部分风险尺度及资产历史固定强弱。假设是模型更适合辨认当期状态偏离，而非主要重现久期或长期收益差异；只是可检验假设，不保证胜过原目标。

风险尺度目前采用自身过去日收益波动，不是未来波动，也不是已经实现的未来 Sharpe。历史常态估计来自成熟标签且做收缩，不把高度重叠的126个标签当成126个独立市场周期。60/40、252/126、收缩252和1%底线均为本轮预设，不根据2024–2026收益反复挑选。

## 与 EP/skfolio、PPO 的单位接口

机会分数 z 没有收益率单位，不能直接当作 EP 的 mean view，不能直接加到组合收益或 PPO 奖励。

预测完成后按信号日可见参数逆变换：

```text
raw_relative_return_view = (predicted_z + location(t,h)) * scale(t,h)
raw_active_increment = score_active_increment * scale(t,h)
```

分数存于 `score_predictions.parquet`，逆变换后的相对收益观点另存于 `decoded_relative_return_views.parquet`。尺度/历史常态在最新没有未来标签的日期也可计算，但历史不足时逆变换结果为缺失，不伪造观点。

`run_v55_gpu_objective.py` 修正了复用训练引擎的报告单位声明；文件名中遗留 v45 仅表示复用引擎命名，不表示新分数是原始收益率。原门控的通过也只是新目标上的预测证据，不能替代同口径原始收益、成本与产品回测。

PPO 仍是后续调仓执行层，不由本轮分数直接替代。此前尚未认证的票息、退出现金流和实盘成本问题继续保留，本轮没有自动启动正式 PPO 或产品回测。

## GPU 环境与速度实测

显卡为 NVIDIA GeForce RTX 5070 Ti Laptop GPU，约12GB显存，驱动592.01。原 Python 的 torch=2.11.0+cpu，之前确实没有使用 CUDA。

新环境 `.venv-gpu-v55` 使用 system-site-packages 复用已有研究依赖，仅在环境内安装官方 torch==2.11.0+cu130；未卸载/替换系统 CPU PyTorch。CUDA 前向及反向均已执行成功，GPU capability为12.0，wheel包含sm_120。支持依据见 [PyTorch Blackwell 发布说明](https://pytorch.org/blog/pytorch-2-7/)，实际安装包由官方 cu130 索引提供。

同一 mask-safe 模型、batch64、3次预热后测5批，计入窗口提取、CPU→GPU传输、前向、反向、优化器：

- CPU 中位每批0.3441秒。
- CUDA 中位每批0.06375秒，约5.40倍批处理加速。
- 测量峰值 GPU allocated 约383.8MB，不代表完整训练所有阶段总显存。

基准是在 CPU 对照同时运行的情况下测得，不是无竞争的纯硬件比较；不含 OOF 岭回归、验证及检查点序列化，不能宣称整轮必然快5.4倍。完整训练会输出真实每轮秒数。

## 验证、控制变量与结果解释

6项目标单元测试、163项实际数据检查通过，覆盖未来标签不可用于当前常态、当前收益不进入过去风险尺度、缺失保留、九宫格/输入不变、成熟终点及逆变换。11项 mask-safe 预测接口测试也在 GPU 环境中通过。

新标签有效样本因因果标准化预热减少，年度 T 日期数为315/557/799/1041/1284，V/G均126。原始标签保留在新数据包中，未删除日期或其他资产。

两条实验共有输入、种子集合、年度隔离和门槛，但训练标签、有效样本以及 CPU/CUDA 随机数实现不同；不是严格只有一个变量变化。若挑战分支有改善，还需补同硬件、共同可用样本的目标消融，不能直接归因于“换目标必然有效”。比较应在相同成熟日期上，以解码后的原始相对收益与成本后决策效果进行，而不是比较两种单位的损失大小。

正式晋升仍需独立门控、三产品风险分层、原始收益单位、执行账本和成本测试。2024年以来已多次研究，不再宣称完全未触碰样本外。

## 查看进度和运行入口

```powershell
python scripts/show_v55_parallel_progress.py
.venv-gpu-v55\Scripts\python.exe scripts/run_v55_gpu_objective.py --benchmark
.venv-gpu-v55\Scripts\python.exe scripts/run_v55_gpu_objective.py --train --resume
```

当前长任务由 `scripts/start_v55_gpu_challenger.ps1` 启动，初始PID85484，启动登记为 `output/reports/temporal_opportunity_v55/GPU_training_launch.json`。实际进程和逐轮进展需要实时读取，不只看静态 training 字段。原 CPU 对照仍为独立任务。运行中不要重建这两套数据或修改受哈希约束的模型源代码。

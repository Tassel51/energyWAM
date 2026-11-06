# GPU 2、3 显存容量调优结果（2026-11-01）

> 最新正式执行方案已改为以吞吐为目标的双副本并行，见 [GPU23_SPEED_TUNING_20261101_CN.md](GPU23_SPEED_TUNING_20261101_CN.md)。本文件保留此前配置与过程记录。

已选择接近显存上限、通过真实4B完整步压力测试的配置，并恢复正式训练。
所有测试均保持K=2完整二阶展开、全量可训练语言主干、eager attention、bf16主干和fp32候选。
GPU 2/3分别映射进程内cuda:0/1；两卡仍是一个按语言层划分的模型。

## 最终配置

| 参数 | Stage I 双向 | Stage II 联合 |
|---|---|---|
| batch | 2 | 2 |
| GPU激活预算（GPU 2/3） | 16 / 19 GiB | 20 / 22.5 GiB |
| 非重入activation checkpointing | 开启 | 开启 |
| 预算之外的activation | 卸载CPU | 卸载CPU |
| AdamW8bit moments | forward前预分配零状态 | forward前预分配，切阶段reset后重新预分配 |
| 首轮目标步数 | 200 | 100 |

预分配仅创建本来就需要的optimizer状态，不提前更新参数；activation驻留保存真实反向传播所需张量。
没有分配无关tensor来占用显存。batch4的8/8GiB驻留预算试验耗时超过6分钟后，
仍未完成首个action方向backward，按时间上限停止；未将其宣称为完整步吞吐或稳定性结果。

## 完整步压力测试

从22,725条训练轨迹中用Qwen tokenizer查找最长指令：episode20263，46个文本tokens，
任务SeparateFreezerRack。使用实际RGB、动作、状态、未来DINO目标和固定150k统计；
提前分配optimizer状态，让首步测试包含后续训练所需的常驻moments。

| 指标 | Stage I（16/19） | Stage II（20/22.5） |
|---|---|---|
| 单步训练计算时间 | 154.93秒 | 70.27秒 |
| GPU 2/3已分配峰值 | 43.95 / 43.57 GiB | 43.75 / 43.69 GiB |
| GPU 2/3预留峰值 | 44.92 / 44.70 GiB | 44.71 / 44.64 GiB |
| nvidia-smi高占用观测 | 46,488 / 46,260 MiB | 46,272 / 46,202 MiB |
| 参数梯度非有限数 | 0 | 0 |

每卡显示容量49,140MiB，观测占用约94–95%，保留约2.6–2.9GiB显示容量余量。
Torch统计与nvidia-smi不同：后者包含CUDA运行时等开销。不同训练阶段的占用仍会波动。
单步计时不含数据加载；正式日志的samples_per_second也是训练计算吞吐，不能替代端到端吞吐。
本轮不是收敛或闭环成功率评估，也未证明所有可能实现的理论最大batch。

## 边界与对照

- batch2 + 重计算 + 完全关闭CPU卸载：真实4B发生OOM，报错时一卡已分配46.44GiB。
- batch2 + 重计算 + 12/12GiB预算：连续2步通过；第二步138.38秒，分配峰值39.29/35.92GiB。
- batch4 + 重计算 + 8/8GiB预算：观测45,074/42,870MiB，未完成首步，达到测试时间上限后停止。
- 最终高占用档使用最长训练指令；输入长度、预分配和冷/热状态不同，不能直接据此声称严格加速比例。

小随机Qwen的数值对照验证完整K=2梯度：单卡/双卡、完整求和/逐方向反向、CPU卸载、
非重入重计算、不对称GPU预算；同时验证AdamW8bit预分配、resume，以及joint切阶段reset后的零moments与下一步训练。
最终相关回归20项通过，另有34项较早相关CPU回归通过（两组有重复，不能相加）。
真实4B容量测试分别完成上述表中的完整步并通过有限梯度检查。

## 运行和记录

正式任务：`supervisorctl status energywam-gpu23`。
调优进程已停止并关闭autostart，防止将来与正式训练同时抢占GPU。

```bash
cat logs/gpu23_pipeline_status.json
tail -f logs/train_gpu23_bi_20261101.log
tail -f logs/runs/energywam_gpu23_bi_20261101/train_log.jsonl
```

- 配置：`configs/gpu23_bi_20261101.yaml`、`configs/gpu23_joint_20261101.yaml`。
- 机器可读汇总：`logs/gpu23_capacity_report_20261101.json`。
- 各档完整日志与指标：`logs/tuning_gpu23/`、`logs/tuning_gpu23_*.log`。
- 高频显存观测：`logs/gpu23_capacity_samples_20261101.csv`（覆盖最终joint测试及恢复启动期间）。
- 数值/optimizer回归：`logs/gpu23_capacity_final_tests_20261101.log`。

正式训练首步完成即保存滚动断点，其后每10步保存，减少后续调整时丢失已完成更新。
容量测试的临时权重未作为正式训练初始化；正式任务仍从原始预训练模型和固定训练统计开始。

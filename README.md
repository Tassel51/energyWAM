# LTE-WAM / EnergyWAM

基于Qwen3-VL-4B语言主干的机器人标量能量模型研究代码。当前使用文本、冻结DINOv3当前图像特征和状态作为条件，Stage1同时学习Action与Future两个条件预测方向。

**接手与复现请先阅读：[完整交接文档](项目交接文档.md)。** 文档包含真实架构、依赖、外部模型/数据、路径准备、训练/验证、恢复、资源监测、历史实验和已知限制。

## 当前可运行入口

- `scripts/prepare_remote_run.py`：为另一台机器生成配置和受监测队列；先核验数据/模型，不自动开训。
- `scripts/train_shared_sharded.py`：一份模型拆两卡；有效batch16、同时计算4样本、K2完整展开。
- `scripts/run_iteration_search.py`：空闲GPU/空间检查、资源监测、有限数值保护及断点继续。
- 冻结清单：`manifests/experiments/`；TRAIN标准化：`stats/`。
- 当前实验：三任务完整轨迹池的一轮训练，以及16固定样本的Future损失权重1/9短对照。研究结果仍在验证，不是最终benchmark成功率。

## 环境

先准备Python3.12与适配机器的PyTorch/torchvision CUDA构建，再安装`requirements-dev.txt`。实际源环境在`docs/reproducibility/source_environment_20261106.json`。`requirements.txt`不包含平台相关PyTorch wheel，旧环境参考保留为`requirements-legacy-20261031.txt`；可选W&B依赖单列。

```bash
python scripts/prepare_remote_run.py --help
```

运行前必须准备外部Qwen/DINO完整权重和相同RoboCasa365 LeRobot视图；Git不含这些资源、训练checkpoint或密钥。

## 阅读顺序

1. [完整交接与命令](项目交接文档.md)
2. [模型及实验历史](docs/LTE_WAM_MODEL_AND_EXPERIMENT_SUMMARY_CN.md)
3. [数据任务构成](docs/ROBOCASA365_DATASET_BREAKDOWN_CN.md)
4. [早期README（历史记录，含已过时配置）](README_LEGACY_CN.md)

旧论文草稿、ARCHITECTURE和HPC文档记录历史阶段；与当前交接冲突时，核对实际代码、冻结配置和日志。不要直接重启历史supervisor或Stage2流程。

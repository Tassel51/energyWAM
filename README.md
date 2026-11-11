# LTE-WAM / EnergyWAM

基于Qwen3-VL-4B语言主干的机器人标量能量模型研究代码。当前使用文本、冻结DINOv3当前图像特征和状态作为条件，Stage1同时学习Action与Future两个条件预测方向。

**接手与复现请先阅读：[完整交接文档](PROJECT_HANDOFF.md)。** 文档包含真实架构、依赖、外部模型/数据、路径准备、训练/验证、恢复、资源监测、历史实验和已知限制。

## 当前可运行入口

- `scripts/prepare_remote_run.py`：为另一台机器生成配置和受监测队列；先核验数据/模型，不自动开训。
- `scripts/train_shared_sharded.py`：一份模型拆两卡；有效batch16、同时计算4样本、K2完整展开。
- `scripts/run_iteration_search.py`：空闲GPU/空间检查、资源监测、有限数值保护及断点继续。
- 冻结清单：`manifests/experiments/`；TRAIN标准化：`stats/`。
- 当前研究：OpenDrawer 单 episode、67个训练窗口的小样本拟合，比较外层学习率、能量头学习率、随机种子和可学习内层步长。seed45已完成210步；学校双步长减半队列于11月11日部署，实时状态应查询队列。研究结果仍在验证，不是最终benchmark成功率。
- 学校服务器运行与查询：[远程运行说明](docs/LAB_REMOTE_RUN.md)；完整历史：[实验记录](HISTORICAL_EXPERIMENTS.md)。

## 2026-11-11 服务器迁移备份

本仓库保存当前工作区源码、配置、测试、实验文档，以及此前被忽略的历史冻结源码。历史版本位于 `docs/reproducibility/source_archives/`，`manifest.json` 记录每份原路径、备份路径、大小和SHA256；内容相同的源码ZIP只保留一份。

原始实验文本记录另存于 `docs/reproducibility/experiment_records/` 的分包归档，包含云端实际配置/清单/逐步日志/评价与实验室部署receipt。自定义50任务视图的四份meta原文件保存在 `docs/reproducibility/trainmix_meta_backup/`；它们不包含视频/Parquet payload。恢复方式和排除清单见各目录manifest与README。

外部权重/数据另见[迁移清单](docs/reproducibility/server_retirement_artifacts_20261111.md)。Git仓库不包含训练checkpoint、原始数据集、预训练模型、个人认证文件或虚拟环境。归档包含已筛选的原始文本记录；其余二进制实验产物仍需独立保存，文档和W&B不能替代它们。

本次近期改动对应CPU测试为88项通过。完整suite尝试190通过、10失败、1跳过，达到失败上限后停止；旧测试硬编码DINO路径、缺外部starVLA仓库及旧cfg=None fixture的具体限制见[验证记录](docs/reproducibility/github_backup_validation_20261111.json)。未将它报告为全suite通过，未启动真实GPU训练。

`prepare_remote_run.py`目前只覆盖旧三任务和tiny-weight1/9入口。恢复当前67窗口实验时，应使用对应实验保存的配置、窗口清单和统计文件，并调整数据/输出路径及GPU分配；不要直接执行历史清理或部署脚本。`configs/remote/`中的路径和GPU UUID为历史机器配置。

## 环境

先准备Python3.12与适配机器的PyTorch/torchvision CUDA构建，再安装`requirements-dev.txt`。实际源环境在`docs/reproducibility/source_environment_20261106.json`。`requirements.txt`不包含平台相关PyTorch wheel，旧环境参考保留为`requirements-legacy-20261031.txt`；可选W&B依赖单列。

```bash
python scripts/prepare_remote_run.py --help
```

运行前必须准备外部Qwen/DINO完整权重和相同RoboCasa365 LeRobot视图；Git不含这些资源、训练checkpoint或密钥。

## 阅读顺序

1. [完整交接与命令](PROJECT_HANDOFF.md)
2. [模型及实验历史](docs/LTE_WAM_MODEL_AND_EXPERIMENT_SUMMARY_CN.md)
3. [数据任务构成](docs/ROBOCASA365_DATASET_BREAKDOWN_CN.md)
4. [早期README（历史记录，含已过时配置）](README_LEGACY_CN.md)

旧论文草稿、ARCHITECTURE和HPC文档记录历史阶段；与当前交接冲突时，核对实际代码、冻结配置和日志。不要直接重启历史supervisor或Stage2流程。

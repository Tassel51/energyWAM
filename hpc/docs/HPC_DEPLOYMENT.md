# EnergyWAM 超算短训练

2026-10-29 用户已授权本项目所需代码、数据和模型部署及短训练。范围仅 EnergyWAM；不恢复 LDA 上传服务，不移除全局 `UPLOADS_PAUSED`，不影响其他实验。

## 已准备

远端根目录：`/home/bingxing2/home/scx8178/energywam_storage`。

- `datasets/robocasa365/v1.0/target/atomic/OpenDrawer/20250816/lerobot`：官方 target/human，LeRobot v2.1、PandaOmron；514 episodes、137431 frames、1542 videos、514 trajectory parquet。官方 Box 包 SHA256：`f89d586e828126a568486e5c3a30c2fe9c91e43d8b18c760969bf850e9e6f005`。
- `models/Qwen3-VL-4B-Instruct`：两个完整权重分片及原始 tokenizer/processor/config。
- `models/dinov3-vits16-pretrain-lvd1689m`：原始冻结 DINOv3-ViT-S 权重和配置。
- 全部模型文件通过本地原件的逐文件 SHA256；清单在 `logs/model_manifest.json`。权重由超算直接访问 ModelScope 下载，小型配置使用本地原件，避免镜像配置变化。
- `env/bin/python`：ARM64 Python3.11；本项目独立 venv，transformers4.57.0、tokenizers0.22.1、huggingface-hub0.36.0、peft0.17.1、omegaconf2.3.0 安装于本项目环境。只读借用 `~/repa_arm64_py311/lib/python3.11/site-packages` 的 torch2.10.0+cu126、视频、parquet 等底层依赖；没有修改该环境。HF Qwen3VL/processor/DINO 导入已验证。
- `code/energyWAM`：独立源码快照，不覆盖其他项目。

模型总占用约8.4GiB，数据约464MiB，新环境约167MiB；临时数据包仍保留以便核验。共享盘约650TiB，首次查询剩余约33TiB。用户/组 quota 查询均超时，**账号存储限额未知**，共享剩余空间不等于个人额度。全量4B AdamW checkpoint 会额外占用数十GB，短训练需预留足够空间；当前没有训练 checkpoint。

## 实验条件与边界

`hpc/configs/hpc_smoke_gpu.yaml` 为工程验证：真实预训练4B，Qwen视觉与DINO冻结，语言主干参与训练，LoRA关闭，eager attention、bf16主干、float32候选，H16/action12/state32、左相机、future offsets[5,11,16]，K2 full unroll、同点联合更新，训练split标准化和N(0,I)初始化。

首轮按8 train +2 val episodes进行明确的smoke子集测试；统计只拟合train中的32个窗口，必须标记coverage=subset。它不替代全训练集统计，不提供正式模型效果结论。

执行顺序：

1. `logs/preflight.slurm`：单GPU、32CPU、15分钟硬上限；先拟合统计，真实模型跑1个完整训练步，记录梯度、峰值显存和同步墙钟时间，不保存大型checkpoint。
2. 只有首轮成功，才提交 `logs/short_training.slurm`：单GPU、32CPU、30分钟硬上限；bi10步 → joint10步 → 同阶段恢复10步 → 4个留出样本的离线动作误差。
3. `hpc/scripts/auto_train_when_available.py` 最多提交以上2个作业，互斥锁防重复；不取消已有作业，不修改Slurm限制。账号不允许提交时每60秒检查，最长等待6小时，过期退出并记录原因。作业失败即停止，不静默启用LoRA、冻结语言主干、缩减真实几何或截断展开。

`hpc_frozen_qwen_diag.yaml` 仅为明确标注的辅助配置，自动流程不使用它。`gpu_preflight.py --offload-saved-tensors` 是未在真实GPU验证的可选精确activation offload诊断开关，自动流程不使用它。

## 当前阻塞与监控

首次sbatch返回 `AssocMaxSubmitJobLimit`；`sacctmgr` 明确显示本账号 `MaxSubmitJobs=0`。这属于平台账号策略；换分区不能绕过。当前尚无EnergyWAM训练作业，真实4B反向、显存、吞吐、checkpoint及闭环成功率未验证。

查看账号限制：

```bash
papp-cloud-proxy ssh scx8178@n32ha14p 'sacctmgr -nP show assoc where user=scx8178 format=Cluster,Account,User,MaxSubmitJobs'
```

限额恢复后自动等待进程负责提交。监控主状态：`logs/training_status.json`；详细等待日志：`logs/training_supervisor.log`；Slurm日志：`logs/preflight-JOBID.log` 和 `logs/short-training-JOBID.log`。Ctrl+C只退出监控，不停止等待进程或后台作业。

完整4B二阶图可能超出单张A100 40GB显存；当前不能承诺可容纳或预计20k步训练时间。只在实测成功后按稳定单步时间估算。RoboCasa仿真资产和闭环评估环境未部署，本轮为训练和离线评估，不自动下载无关大资产。

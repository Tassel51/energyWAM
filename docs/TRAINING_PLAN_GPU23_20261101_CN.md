# 本机 GPU 2、3 训练方案（2026-11-01）

> 最新正式执行方案已改为以吞吐为目标的双副本并行，见 [GPU23_SPEED_TUNING_20261101_CN.md](GPU23_SPEED_TUNING_20261101_CN.md)。本文件保留此前配置与过程记录。

用户已确认“第 3、4 张卡”指物理编号 2、3。进程统一使用
`CUDA_VISIBLE_DEVICES=2,3`，进程内 `cuda:0/1` 对应物理 GPU 2/3。
GPU 0、1 不参与。Python 环境：`/usr/local/miniconda3/envs/py312/bin/python`。

## 数据和权重

- 数据：`/datasets/robocasa365/v1.0/target`，50 个任务，PandaOmron。
- 合并软链视图：`/datasets/robocasa365_trainmix/robocasa365_target_50tasks`。
- 核查：25,307 episodes / 14,957,899 frames；25,307 parquet、75,921 mp4
  链接目标均存在。这是文件/元数据检查，不是逐帧解码或逐文件哈希检查。
- 固定划分：train 22,725 / val 1,285 / test 1,297 episodes；互不重叠，均覆盖50任务。
  清单位于 `manifests/`，检查记录 `logs/dataset_audit_20261101.json`。
- Qwen：`/datasets/models/Qwen3-VL-4B-Instruct`（链接到本机 `/model` 权重）。
- DINO：`/datasets/models/dinov3-vits16-pretrain-lvd1689m`，冻结。
- 归一化：严格训练集拟合的150,000窗口**子集**统计，并非全部训练集统计。
  `stats/trainmix_150k_paper_standardized_v1.json`；沿用打乱的完整训练清单。
  训练数据加载范围仍是完整22,725训练轨迹，不受150k统计窗口数限制。

## 方法和执行顺序

1. 真实预训练4B模型完成2步双卡工程验证，使用旧256窗口子集统计，仅用于验证运行。
   该checkpoint不用于正式训练初始化。
2. 重新拟合150k窗口统计：动作/未来DINO逐维z-score，冻结后全流程复用。
   DINO按32张批量提取；float64 Chan矩统计向量化，保留有效mask、逐通道计数。
3. Stage I：`configs/gpu23_bi_20261101.yaml`，首轮预算200步。
   Future→Action 与 Action→Future 同一batch都计算，共享同一Energy；
   学习率1e-5，warmup20，batch2，梯度裁剪1。
4. Stage I 后先在val清单上做32样本joint离线动作误差检查。
5. Stage II：`configs/gpu23_joint_20261101.yaml`，从Stage I权重初始化，
   optimizer/scheduler显式reset；联合同时更新A/Z，学习率5e-6，首轮预算100步（warmup10）。
6. Stage II 后同样做val离线检查。闭环成功率另需仿真评测；离线MSE不能替代成功率。

首轮200+100步用于真实数据稳定性与学习信号检查，不声称收敛。历史单卡短训约5分钟/步，直接20k+5k可能需要数月，因此先跑有界首轮。后续20k/5k规模应根据本次吞吐、loss和验证结果调整。运行速度和显存以真实日志为准。
32样本离线检查是小子集，不代表整个验证集或全部任务的均衡评测。
正式论文实验后续应按任务均衡扩大验证，报告随机初始化动作误差、相同seed对照、
条件/联合动作误差、未来latent误差、耦合诊断、运行成本和闭环成功率。

## 双卡与内存

本实现未有DDP训练入口；单卡历史试验发生OOM。新增可选语言层分配：
36层按18/18拆到两卡；embedding、视觉编码、projector/head在主卡。
这是一个模型和一个optimizer，batch2由两卡共同处理；跨卡张量拷贝保留autograd。
Qwen语言主干全量训练，视觉编码器/DINO冻结；无LoRA，无截断展开。
K=2，full unroll，eager attention，bf16主干，fp32候选/能量/loss。

GPU按预算保存activation（Stage I：16/19 GiB；Stage II：20/22.5 GiB），超出预算才卸载到CPU；参数视图复用驻留存储，避免权重重复搬运。
两方向独立前向/反向：训练器完成一个方向的backward并释放图后才构建另一个方向；累加梯度后只做一次optimizer step。
AdamW使用8-bit moments，这是显式优化器存储量化，数值与标准AdamW不保证逐位相同。
启用非重入activation checkpointing（use_reentrant=False）进行重计算；双卡K=2二阶梯度数值对照通过。完整展开、候选更新和参数训练范围保持一致。不因OOM自动改K或冻结语言主干。

## 运行、日志与停止

任务由已有supervisord管理，断开聊天后继续：

```bash
supervisorctl status energywam-validation energywam-stats energywam-gpu23
cat logs/gpu23_pipeline_status.json
tail -f logs/train_gpu23_bi_20261101.log
tail -f logs/runs/energywam_gpu23_bi_20261101/train_log.jsonl
```

流程入口：`scripts/run_gpu23_pipeline.py`。
Supervisor模板：`hpc/configs/supervisor_gpu23_pipeline.conf`，
已部署到 `/etc/supervisor/conf.d/energywam-gpu23.conf`。
依赖的stats/validation进程由同目录各自配置管理。
先等待验证和统计成功退出，再自动启动Stage I、val、Stage II、val。
任一错误立即停止流程并记录status，不反复OOM或静默改变方法。

首步完成即保存一次，其后每10步原子替换 `ckpt_latest.pt`，包含模型/optimizer/scheduler/RNG/统计/步数。
最终checkpoint按stage和step命名；仅保留滚动checkpoint和各阶段最终文件。
重新启动流程会从该阶段已有滚动checkpoint续训；CLI `--steps` 为目标总步数。
数据加载器的shuffle/游标尚未持久化，因此恢复不保证样本顺序逐步复现。

```bash
# 停止正式流程及其训练子进程
supervisorctl stop energywam-gpu23
# 如仍处准备阶段，可分别停止验证/统计
supervisorctl stop energywam-validation energywam-stats
# 从滚动checkpoint重新启动流程（stats/validation需已完成）
supervisorctl start energywam-gpu23
```

## 本次修复与验证范围

- 修复 `_bi_only` 在batch准备阶段丢失，原先导致每个独立方向再次跑两个方向、
  重复计算，并在分项backward时访问已释放的共享图。
- 新增语言层跨卡分配，checkpoint先CPU加载，防止全部权重/状态集中到单卡。
- 修复CLI续训步数按“目标总步数”执行；Trainer.fit仍按“额外步数”定义。
- 新增GPU二阶梯度数值对照、CPU activation offload对照、跨卡optimizer/restore实跑，
  和批量统计在完整/部分horizon、样本上限边界的数值等价检查。
- 49项相关测试通过（真实DINO测试通过本地路径注入；历史测试fixture默认仍是旧服务器路径）。
  小随机Qwen的梯度对照验证数学路径，不能替代预训练4B的真实显存/运行验证。

## 真实4B首步观测（工程验证，256窗口旧统计）

2026-11-01 北京时间12:21左右，首步成功：180.49秒（forward133.87 / backward46.29），loss3.726323，参数梯度范数0.181437，421个参数张量有梯度，非有限梯度0；主卡峰值22,888MB，CPU峰值约169GiB。正式训练器改为逐方向完成forward/backward，减少两张图同时驻留；该更改的梯度等价和断点恢复已有数值测试通过。此观测不是收敛或任务成功率结论。

## 当前启动记录

2026-11-01 北京时间12:24:50，后台流程已自动进入Stage I（200步）。
真实4B验证2步完成，共326.36秒，最后一步非有限参数梯度0。
150k训练窗口子集统计已完成，覆盖50任务；fingerprint `ad89d0de59bb62bf374612844a09e9687ef4b1e2cd9cca4c4342f870ef37f25a`。
实时状态见 `logs/gpu23_pipeline_status.json`，正式训练见 `logs/train_gpu23_bi_20261101.log`。

## Batch翻倍调整

2026-11-01：按用户要求，将Stage I和Stage II的batch均从1改为2，学习率及K=2完整展开保持原配置。当前是模型分层并行，两卡共同处理batch2。调整时正式训练尚未完成第一个optimizer step，无已完成的参数更新需要恢复；从相同预训练权重和固定150k统计重新启动。

## 2026-11-01 显存与吞吐调优

当前双卡模型仍采用batch2；增加有用的GPU activation驻留，而不是用无关tensor占显存。
开启非重入重计算，并按GPU 2/3分别设置16/19 GiB激活预算，超出部分按需卸载到CPU。
AdamW8bit的零moment在首个forward前初始化，使首步容量测试包含后续常驻optimizer内存；不提前执行参数更新。

调优记录位于 `logs/tuning_gpu23/`：

| 配置 | 完成情况 | 显存与速度 |
|---|---|---|
| batch2，重计算，完全关闭CPU卸载 | OOM | 一卡46.44 GiB已分配，无法继续 |
| batch2，重计算，12/12 GiB驻留预算 | 连续2步通过 | 热身后138.38秒/步；分配峰值39.29/35.92 GiB |
| batch4，重计算，8/8 GiB预算 | 首步未完成即达到时间上限并停止 | 观测45,074/42,870 MiB；未声称完整步吞吐或稳定性 |
| batch2，重计算，16/19 GiB预算，最长训练指令 | 完整步通过 | 154.93秒；分配峰值43.95/43.57 GiB，预留峰值44.92/44.70 GiB；非有限梯度0 |

最后一档提前分配optimizer states，并从完整训练清单中按Qwen tokenizer选取最长指令（46 tokens，episode20263）测试。
不同指令长度、预分配与冷/热状态下的耗时不能直接视为严格速度提升比例。
MiB/GiB为二进制单位；nvidia-smi还包含CUDA运行时开销，因此高于Torch已分配统计。
目标是接近显存上限并留出算子工作区余量，未声称找到了所有可能实现的理论最大batch或峰值。

小随机Qwen的双卡二阶梯度对照覆盖：完整求和、逐方向反向、CPU卸载、非重入重计算、
不对称GPU激活预算；AdamW8bit预分配与checkpoint恢复后继续一步均通过。真实4B另做容量和有限梯度检查。
正式日志新增每卡每步分配/预留峰值、batch_size、samples_per_second。

最新容量调优与完整压力测试见 [GPU23_CAPACITY_TUNING_20261101_CN.md](GPU23_CAPACITY_TUNING_20261101_CN.md)。Stage II独立预算20/22.5 GiB，真实最长指令完整步70.27秒，分配峰值43.75/43.69 GiB，梯度非有限数0。

# EnergyWAM / LTE-WAM 项目接手说明

> 更新日期：2026-10-31，Asia/Shanghai。交接对象：新 GPU 服务器 `/root/zzc/energyWAM/` 下的 GPT。
> 本文是当前状态交接，不是训练成功报告。旧 README、状态文档中有历史记录，应结合本文的日期和证据阅读。
> 不含密码、Token、私钥；认证信息应由用户在终端交互输入。

## 1. 接手先看：当前结论与授权边界

1. 项目代码已迁移到新服务器，150 个文件全部通过逐文件 SHA256 核对；本说明是随后补充的第 151 个交付文件。
2. 新服务器只有代码、配置、依赖版本参考、必要脚本和文档；**没有随代码迁移数据、模型权重、训练 checkpoint、统计 bundle、日志、虚拟环境或旧项目仓库**。
3. 用户先说“先别搞了，计划有变”，随后只授权代码迁移和本次交接整理。**暂停仍有效，不要自动恢复旧下载、安装环境、申请算力或启动训练。** 等待用户的新计划；目前可以先阅读代码和做只读检查。
4. 真实 4B 模型已经在原服务器加载，并进入第一个真实数据训练 step，但触发主机内存保护被终止。**没有已确认完成的 optimizer step，没有可交付的真实训练 checkpoint，没有闭环成功率。**
5. 原服务器本机 RoboCasa365 下载在暂停前为 **3.24%，任务 3/50、厨房资产 2/6、机器人资产就绪**。这不是全部下载完成，也不是新 GPU 服务器的本地进度。
6. 核心算法、CPU 数值/梯度/小尺寸真实 Qwen 结构测试已有大量验证；**不能把这些结果表述成真实 4B GPU 训练已经跑通**。
7. 最近新增的参数去重 CPU 卸载工具 `energywam/training/offload.py` 是未完成验证的工程尝试，不能默认启用或宣称已解决内存问题。

## 2. 机器、目录与交付范围

| 位置 | 内容与状态 |
|---|---|
| 原代码服务器 | `/data3/xyx_zhuzicheng/projects/hpc/energyWAM/`，原文件保留 |
| 新 GPU 服务器 | `cpod-1x51qyjx1lqg-s3.podtcp.compshare.cn`，SSH 端口 `24870`，用户 `root` |
| 新代码目录 | **`/root/zzc/energyWAM/`**，无重复的 `energyWAM/energyWAM/` 嵌套 |
| 原服务器新版数据根目录 | `/data7/zzc/energyWAM/`，它是数据/资产存储目录，不是代码目录 |
| 原服务器早期 OpenDrawer 副本 | `local_storage/datasets/robocasa365/OpenDrawer/20250816/lerobot`，相对于原代码根目录 |
| 旧超算存储 | `/home/bingxing2/home/scx8178/energywam_storage/`，与新 GPU 服务器完全不同 |

迁移前目标目录已存在但为空。目标没有 rsync，所以使用 SFTP 传输代码包；中断时可对同一已校验源包续传。先在目标旁临时目录校验整个包和各文件，再以不覆盖现有不同内容的方式放入目标。150/150 文件校验成功，临时包已清理。共 1,160,209 字节，不含本文。

代码包 SHA256（历史交付证据，不是当前整个目录的哈希）：

```text
27492488c4a4b8322e1ab4919a29fddb025f7e156748faa8e7927e11f5093cb4
```

迁移报告和精确源文件清单只在原服务器：

```text
logs/migration_20261031/report.json
logs/migration_20261031/source_manifest.json
```

新服务器未安装环境、未启动训练。旧服务器 `logs/`、`coordination/`、`local_storage/` 没有迁移；本文引用其中的路径仅表示证据来源，不代表新目录内存在那些文件。

## 3. 项目目标和算法总览

项目正式研究名称是 **Latent Transition Energy World Action Model（LTE-WAM）**；Python 包叫 `energywam`，主模型类叫 `QwenEnergyWAM`。

研究目标：用一个共享的标量能量函数，评价“当前条件 C、候选动作块 A、候选未来视觉潜变量 Z”是否相容，并通过对 A/Z 求能量梯度来生成结果。

```text
E_theta(C, A, Z) -> 每条样本一个标量能量
```

它借鉴 LDA 的动作—未来共同建模思路，以 StarVLA 的机器人数据/接口为工程参考，但当前方法不依赖一个独立 Diffusion/Flow 头输出动作，也不直接监督“正确能量标签”。监督的是经过若干次能量优化后的动作和未来结果。

### 3.1 条件编码：最重要的最新确认

师兄于 2026-10-31 明确要求：**当前 RGB 只过 Qwen 的视觉 ViT/原生 visual merger，语言只做 token embedding；这里不先跑一遍 Qwen 的语言 Transformer。**

```text
当前 RGB ─> Qwen ViT + native visual merger ─┐
指令文本 ─> Qwen token embedding ───────────┴─> T_VL（语言 Transformer 之前的 embedding）
当前 RGB ─> 冻结 DINOv3 ─> current-DINO projector ─> T_cur
机器人状态 q ─> state projector ─> T_state

C = [T_VL, T_cur, T_state]

候选未来 Z ─> future projector + time/spatial/camera/type encodings ─> T_future
候选动作 A ─> action projector + step/type encodings ─> T_action

X0 = [C, T_future, T_action, learnable Energy Token]
  ─> Qwen Language Transformer（每次能量计算一次）
  ─> 取最后 Energy Token 的 hidden state
  ─> LayerNorm / MLP Energy Head
  ─> 每个样本一个标量 E
```

对应 `energywam/backbones/qwen3vl_adapter.py`：

- `encode_condition()` / `_embed_vision_language()`：视觉和文本 embedding，准备 DeepStack 特征与 mRoPE 位置，不调用语言层。
- `run_sequence()`：拼接完整条件/候选/读出序列，直接调用 Qwen 的 `language_model`。
- 保留原生 image-placeholder `masked_scatter`、DeepStack、有效位置 mask 和 mRoPE；DeepStack 在完整序列的那次语言前向中注入。
- 不调用整个生成模型来提前获取条件 hidden states，不计算词表 `lm_head` logits，不使用 KV cache。
- “一次”指每次能量评价一次。候选更新后，下一次能量评价仍然要重新过语言层；不能把整个多步求解器误改为只计算一次 LLM。
- 原路径本来已符合这个边界；最近主要是提取明确的 helper、补充说明和 hook 计数回归测试，没有发现需要删除的前置语言前向。

### 3.2 因果序列顺序

默认顺序是 `[C, Z, A, E]`，沿用原生 causal attention：Future 可以读条件，Action 可以读条件和 Future，最后 Energy Token 可以读完整序列。

这不是 Future/Action token 之间的完全双向 attention。通过共享的最终标量及对两个候选的梯度，最终能量仍依赖二者。改成 `[C,A,Z,E]` 或双向 mask 属于结构消融，不能静默替换主方法。

### 3.3 内层优化与外层训练

联合更新在同一个候选点取两个梯度，然后同时更新：

```python
gA, gZ = torch.autograd.grad(E.sum(), [A, Z], create_graph=training)
A_next = A - alpha_A * gA
Z_next = Z - alpha_Z * gZ
```

- 用 `E.sum()`，不能换成 `E.mean()` 使候选梯度随 batch 大小变化。
- 内层只更新候选 A/Z，不执行参数 optimizer.step。
- 训练时 `create_graph=True`，默认 full unroll，不在候选链路 detach；外层 reconstruction loss 反传穿过展开过程，因此需要高阶梯度。
- 外层 trainer 每个 step 一次 backward，再对共享模型参数执行 AdamW 更新。
- 推理时每步重新 `detach().requires_grad_()` 候选，用 `autograd.grad`；不更新模型参数，不向参数 `.grad` 累积。
- 单方向条件优化固定的真实分支须保持固定，不对它应用候选约束或更新。
- E 是未校准的能量，不能直接解释为任务成功概率。

### 3.4 阶段一：双向条件能量学习

共享同一个能量网络，两种方向默认在同一 batch 计算并求和：

```text
真实未来 Z* 固定：从 A0 优化得到 A_K，计算 masked MSE(A_K, A*)。
专家动作 A* 固定：从 Z0 优化得到 Z_K，计算 masked MSE(Z_K, Z*)。
L_bi = lambda_action * L_A + lambda_future * L_Z。
```

代码：`training/stages.py::_run_bi()`。也支持按 global_step 确定性交替两种方向，但那是显式配置，不应混写实验条件。

### 3.5 阶段二：联合能量优化

从两个随机候选 `(A0,Z0)` 同时更新，最终：

```text
L_joint = beta_action * masked_MSE(A_K,A*)
        + beta_future * masked_MSE(Z_K,Z*)
```

两个模态各自按有效元素平均，不能因为 Z 元素多就直接对全部元素求和改变权重。默认不加 bi auxiliary，额外项是可选消融。

代码还提供 `action_given_future`、`future_given_action`、`action_only`、`direct_joint` 等控制实验。`action_only` 不能读取 future GT；未来占位块不可当作真实未来证据。

### 3.6 推理与机器人执行

推理只接收当前图像、指令、机器人状态以及明确的配置/随机种子，不读取动作或未来 GT。

1. 编码当前条件一次。
2. 在标准化空间采样 A0/Z0。
3. 进行 K 次能量梯度优化。
4. 返回动作块，并按正确统计及控制协议反标准化。
5. 将动作交给对应仿真/真实机器人控制器，重新观测，滚动执行。

步骤 1—4 有代码和测试；步骤 5 的真实 RoboCasa365 闭环端到端链路尚未验证。不可把离线 MSE 当作闭环成功率。

## 4. 数据、形状、标准化和冻结策略

### 4.1 当前主任务协议

| 项目 | 当前约定 |
|---|---|
| 主数据 | RoboCasa365 v1.0 target/human，PandaOmron |
| 首版相机 | `robot0_agentview_left`，V=1 |
| 动作 | 12 维，动作块 H=16，A 为 `[B,16,12]` |
| 状态 | 原始 16 维经 sin/cos 编码为 32 维 |
| 未来时刻 | 三帧，offset `[5,11,16]`，与 H=16 对齐 |
| DINO | DINOv3 ViT-S/16，224×224，只保留 patch tokens |
| 单帧未来特征 | 196 patches × 384 channels |
| 多帧未来 Z | `[B,3,1,196,384]` |
| Qwen | Qwen3-VL-4B-Instruct，hidden 2560，语言层 36 |
| Energy | `[B]`，末尾专用 token 读出，不做跨样本聚合 |

数据必须在同一 episode 内采样，明确 `a_t: obs_t -> obs_{t+1}` 的对齐，不能跨 episode 拼未来。启用完整 horizon 要求、合法元素 mask、camera 顺序和真实 state/action 定义。尾部窗口及 padding 不可污染损失和统计。

### 4.2 最新正式标准化基线

用户已确认 `paper_standardized_v1`：

```text
标准化值 = (原值 - mean_train) / (std_train + epsilon)
epsilon = 1e-6
A0 ~ N(0,I), Z0 ~ N(0,I)，均在对应的标准化空间。
```

动作按维、未来 DINO 按 channel 拟合训练集统计。必须绑定实际数据、训练 episode manifest、相机/几何、DINO 身份与预处理版本；验证/测试不能参与拟合。

旧 min-max/raw-DINO 设置在 `configs/legacy/` 保留，不得静默拿旧 checkpoint 或旧统计混用。缺统计应明确报错，不能随机生成“真实统计”绕过检查。tiny 的 synthetic bundle 只服务显式 tiny 测试。

### 4.3 参数与数值策略

- 主方法训练 Qwen language backbone；Qwen vision 和 DINO 冻结。
- 默认没有 LoRA；冻结语言主干、改 LoRA、截断展开、减少未来 token/分辨率均不能当作等价主方法静默替换。
- `lm_head` 不参与能量生成；若与输入 embedding 共享权重，不能为了冻结“未使用输出头”误冻结输入 embedding。
- 使用 eager attention 验证二阶梯度；现有 SDPA/FlashAttention 某些内核不支持所需 double backward，不应未经验证启用。
- 禁用 KV cache；不默认启用 gradient checkpointing、torch.compile 或 fused 高阶不兼容路径。
- 主配置通常 FP32；工程短试配置明确使用 BF16 Qwen、FP32 candidates/loss/energy head，不能称为同精度正式基线。
- 主配置 K=4；已尝试的工程短试是 K=2、batch=1、目标 10 steps，完整几何/full unroll 保留。

## 5. 代码地图与实际调用链

| 目录 / 文件 | 职责 |
|---|---|
| `energywam/config.py` | dataclass 配置、加载、几何和 offsets 校验 |
| `energywam/models/energy_wam.py` | 顶层模型，forward/predict、统计绑定、训练/推理接口 |
| `energywam/models/energy_core.py` | current-DINO/state/future/action projectors、语义编码、Energy token/head |
| `energywam/backbones/qwen3vl_adapter.py` | 真实 Qwen embedding、完整序列组装、原生多模态/位置适配 |
| `energywam/backbones/dino_v3.py` | 冻结 DINO，图像预处理、移除特殊 tokens、抽取 patch 特征 |
| `energywam/backbones/tiny_*` | 随机小模型测试后端，不能冒充 4B |
| `energywam/energy/solver.py` | 候选梯度迭代、单方向与联合模式、full unroll |
| `energywam/energy/encodings.py` | 时间、空间、相机、类型和 action step 编码 |
| `energywam/energy/losses.py` | 带合法 mask 的 reconstruction losses |
| `energywam/energy/init.py` | 标准高斯和显式可选初始化 |
| `energywam/energy/diagnostics.py` | 梯度、耦合、相容性等诊断 |
| `energywam/data/episode_dataset.py` | LeRobot episode/frame/video 读取 |
| `energywam/data/energy_dataset.py` | episode 内窗口、未来帧、相机/state/action、batch 与统计契约 |
| `energywam/data/standardization.py` | 版本化 z-score bundle 与 provenance 验证 |
| `energywam/data/stats_fit.py` | train-only 统计拟合 |
| `energywam/training/stages.py` | bi/joint/单方向/action-only 阶段 |
| `energywam/training/trainer.py` | 外层 backward、AdamW、warmup、日志和 checkpoint |
| `energywam/training/checkpoint.py` | RNG、优化器、scheduler、统计等 checkpoint 保存恢复 |
| `energywam/training/offload.py` | 最新未完成验证的参数去重 CPU saved-tensor 存储尝试 |
| `energywam/inference/predictor.py` | 推理包装 |
| `energywam/integration/starvla.py` | 显式 StarVLA 注册；不随包自动改 StarVLA |
| `scripts/train.py` | 通用训练入口、统计 fail-fast、数据加载和恢复 |
| `scripts/fit_normalization_stats.py` | 从训练 manifest 拟合标准化 bundle |
| `scripts/evaluate.py` | 离线指标，支持 held-out episode manifest |
| `scripts/serve.py` | 模型服务入口，仍需验证实际部署/闭环依赖 |
| `configs/` | 主方法、消融、tiny、历史设置和本地试跑配置 |
| `tests/` | 自定义 TESTS harness，含真实 HF 小 Qwen 结构测试 |
| `hpc/scripts/` | 数据下载/校验、划分、GPU 预检、资源监督及超算旧流程 |
| `hpc/tests/` | 下载安全/续传、配置和 episode split 检查 |
| `requirements.txt` | 源服务器实测版本参考，不是可直接搬来的虚拟环境 |

训练主链：

```text
scripts/train.py -> config -> resolve_training_stats（先检查统计，后加载大模型）
 -> QwenEnergyWAM -> build_dataloader
 -> model.forward -> _prepare_batch -> encode_condition
 -> StageRunner.run -> EnergyWAMCore.energy -> backbone.run_sequence
 -> refine / reconstruction loss -> trainer.backward -> optimizer.step
 -> checkpoint（同时记录统计和 RNG 等恢复信息）
```

推理主链：

```text
QwenEnergyWAM.predict -> 禁用外层 inference_mode 限制、启用候选梯度
 -> 仅准备允许的当前条件 -> encode_condition
 -> init_candidates -> refine（每步候选梯度）
 -> detached normalized_actions / 可选未来输出
```

## 6. 已有数据与新版下载的真实状态

### 6.1 原服务器已有旧版数据：不要再说“本机完全没有 RoboCasa”

| 原服务器路径 | 内容 |
|---|---|
| `/data6/xyx/datasets/Robocasa_kitchen/v0.1` | xuyuxuan 目录下已有 RoboCasa v0.1 HDF5，约 21 GB |
| `/data7/zzc/cdp/datasets/robocasa_v0.1_lerobot` | CDP 项目转换后的 v0.1，约 15 GB |

检查到旧版 30 个任务、1,651 条轨迹、577,597 帧，转换后 57 个 split 目录。它是 PandaMobile、旧版任务/控制协议；部分转换输出未提供本项目所需的 state feature。

这些不是当前官方 RoboCasa365 v1.0 target/human/PandaOmron 的完整替代品。已保留其他人的数据与 CDP 输出，没有为本项目覆盖/转换它们。扫描只覆盖可访问目录，不能断言他人不可读目录不存在副本。

### 6.2 原服务器新版存储布局

```text
/data7/zzc/energyWAM/
  datasets/robocasa365/v1.0/target/{atomic,composite}/<Task>/<Date>/lerobot/
  sim/robocasa365/robocasa/models/assets/
  sim/robosuite/robosuite/models/assets/
  downloads/                  # manifest、worker、固定源码包
  archives/                   # 未完成 .part 与本队列临时归档
  status/<entry_id>.json       # 单项安装/校验状态
  logs/full_robocasa365_status.json
  logs/full_robocasa365_worker.log
  logs/ROBOSUITE_ASSETS_READY.json
  logs/PAUSED_BY_USER.json
```

暂停前最后快照：**2026-10-31 17:23:27 +08:00**。

| 项目 | 已确认 |
|---|---|
| 压缩包等效下载量 | 2,352,600,565 / 72,544,565,510 bytes，3.24% |
| 任务就绪 | 3/50：OpenDrawer、OpenStandMixerHead、TurnOnElectricKettle |
| 厨房资产就绪 | 2/6：fixtures_lightwheel（实际目录 fixtures）、textures |
| 机器人资产 | 固定版本 Robosuite 源码/mesh 已校验解压 |
| 未完成项 | 其余任务、4 类厨房资产及末尾完整源码整合/总验收 |
| 当时部分下载 | objects_lightwheel、TurnOnMicrowave 的 `.part` 文件 |
| 当前是否活动 | 用户已暂停，检查时未发现本项目下载/试跑/监控进程在运行 |

**快照仍写 `stage=downloading` 只是最后一次写入，不能据此说仍在下载。以暂停标记、实际进程和时间戳为准。** 进度包含已复用或已清理压缩包的等效字节数，并非磁盘上所有压缩包之和；100% 下载后仍需解压、校验和整合。

本机启动时 `/data7` 共享磁盘约剩 584 GB；计划归档约 72.5 GB（67.56 GiB），数据/厨房资产预计解压约 85.6 GB，成功后清理本队列临时归档。脚本 200 GiB 保护上限不是个人账号配额。

### 6.3 OpenDrawer 已确认完整

- 官方 target/human，日期 20250816，机器人 PandaOmron，LeRobot v2.1。
- 514 episodes，137,431 frames，514 parquet，1,542 videos。
- 原压缩包 478,965,760 bytes，SHA256：

```text
f89d586e828126a568486e5c3a30c2fe9c91e43d8b18c760969bf850e9e6f005
```

- 原包在原代码服务器 `logs/dataset_download/OpenDrawer_target_human.tar.part`；虽名为 `.part`，已验证完整。
- `/data7/zzc/energyWAM/` 中复用了已解压核验的数据，没有再次从网络下载 OpenDrawer。
- 上述数据和压缩包**均未传到新 GPU 服务器**。

### 6.4 旧超算下载不要与本机进度混为一谈

旧超算账号为 `scx8178`，它是平台登录账号标识，未验证其实名归属，不能把它说成 xuyuxuan 的实名目录。

旧超算曾确认约 27.25%、任务 17/50、厨房资产 6/6、机器人资产就绪，随后连接失效。**当前完整度未重新核验，不能说超算全量完成，也不能把它等同于本机 3.24% 或新服务器的数据。**

早期网络链路依赖原服务器的 Box-only relay 和 SSH 反向端口；重连曾从登录节点 master-1 切到 master-2，导致转发端口落在错误节点，以及在错误节点查 PID 的误报。后来增加跨登录节点转发/查询工具，但当前没有保留可用认证连接，用户也已暂停。

超算还有历史 `MaxSubmitJobs=0` / `AssocMaxSubmitJobLimit` 阻碍；不是代码移到新 GPU 服务器后仍必须用 Slurm 的理由。

### 6.5 下载实现和恢复注意事项（供将来得到授权后使用）

- `robocasa_download_worker.py`：两路并发、Range 续传、长度和 Content-Range 检查、计算 SHA256、TAR/ZIP 安全解包、文件计数、原子安装。
- LeRobot v2.1 的 `chunks_size` 是 episode 目录分组，不是把多个 episode 合并为一个 parquet；不要重引入已修复的计数错误。
- ZIP 的顶层 README 要跳过；fixtures ZIP 根目录是 `fixtures`，objects_lightwheel 的根目录是 `lightwheel`。
- 不接受 HTML 错误页冒充数据；EDQUOT/ENOSPC 应暂停并保留断点，不能自动删别人的文件腾空间。
- SHA256 记录用于下载/搬运一致性，不代表每包都有官方发布的 SHA256。不能把自算 hash 说成官方验签。
- `full_robocasa_download_pipeline.py` 现支持 `--root`、`--python`、`--proxy`；默认路径仍是旧超算布局。
- `local_download_supervisor.py` 现有实现会尝试原机器的 7901/27904 代理；新服务器不能直接照搬这些 localhost 端口。
- 原服务器已测试能经本地代理下载 Box；新服务器是否可直连 Box/跳转主机应重新实测，不能照抄旧速度或 ETA。
- 新服务器恢复前应先确认数据放在哪、磁盘余量和是否已有副本，再检查进程/锁；不要重复启动旧机器或新机器的同一下载任务。
- 队列总完成以 50/50 任务、6/6 厨房资产、RoboCasa 源码整合、Robosuite receipt 都通过为准，之后仍需单独验证仿真器。

六类厨房资产：textures、generative_textures、fixtures_lightwheel、objaverse、aigen_objs、objects_lightwheel。完整 56 项 URL/大小/目标路径在附录 JSON；其中目标路径均相对于选定数据存储根目录。

## 7. 权重与固定源码

| 资源 | 原服务器位置 / 状态 |
|---|---|
| Qwen3-VL-4B-Instruct | `/data6/zzc/Qwen3-VL-4B-Instruct`，两个 safetensors 分片及 tokenizer/processor/config 完整；本次真实加载成功 |
| DINOv3 ViT-S/16 | `/data6/zzc/dinov3-vits16-pretrain-lvd1689m`，约 83 MB；用于真实统计拟合和模型加载 |
| 旧超算权重 | `energywam_storage/models/`，此前逐文件 hash 与原件一致，但当前连接不可用 |
| 新 GPU 服务器 | 仅代码，不含上述权重，不能认为路径已存在 |

固定 RoboCasa：v1.0.1，对应 commit：

```text
dc73a8ce18e4b8947e8df2ae2958245ca823f3b4
https://github.com/robocasa/robocasa
https://codeload.github.com/robocasa/robocasa/tar.gz/dc73a8ce18e4b8947e8df2ae2958245ca823f3b4
源码归档 SHA256: 5c24304a0c1c269bff8a617331d2c9c2c7c8edfff1b1102fca0bcd10dc28685f
```

固定 Robosuite：官方 master 当时的 commit，不是随意换 release：

```text
5ce6643f3092639d08f7b0f90ed1c6a84f50552c
https://github.com/ARISE-Initiative/robosuite
https://codeload.github.com/ARISE-Initiative/robosuite/tar.gz/5ce6643f3092639d08f7b0f90ed1c6a84f50552c
归档字节: 260653994
归档 SHA256: 24330cf5265effb1b5733c7ea2afda0e68899abe4b3c3de39e40b7db2b1b432e
```

原服务器源码包位于 `coordination/full_download_20261030/{robocasa_source,robosuite_source}.tar.gz`，未随代码迁移。模型/仿真资产下载完成不等于 MuJoCo/renderer/controller 已验证。

## 8. 统计、划分和真实试跑记录

### 8.1 OpenDrawer 训练划分与统计

按 `sha256(f'{seed}:{episode_index}')` 排序确定性分割，seed=20261029。514 episodes 分为：

```text
train=412, val=51, test=51，episode 不重叠。
train manifest SHA256: 58cd2291ce7e189742d3e56913a3b9cf5fefcfe3dc4ceda743c4981d359f07c6
val manifest SHA256:   7ade85b484e7f0f2348e2d3d9e1facceaa1895651d2cb415c03c734763183360
test manifest SHA256:  39fcbc80ccb974f154f9e9eb0b5dec820cd75775d54a14e21ee7956888f95abb
```

原服务器拟合了 **64 个训练窗口的 subset 统计**，不是完整训练集统计：

```text
logs/runs/local_gpu1_20261031/manifests/
logs/runs/local_gpu1_20261031/stats_subset.json
bundle fingerprint:
67f77291e324c9d2acf43c894d6d232683786e6aeb3fe0e8d661bc1f1b3b7cf5
动作维度 12；未来 channel 384；coverage=subset。
```

这些文件在未迁移的 logs 内；新服务器需要明确重新生成或由用户授权额外传递。虽然原 split report 写的是早期 data3 副本路径，实际统计和试跑配置后来指向 data7 的已校验副本；不能把这个旧路径当成新服务器数据路径。

### 8.2 2026-10-31 GPU 1 实际尝试

原服务器 GPU：RTX 4080 SUPER，实际报告总显存 32,760 MiB，UUID 为 `GPU-468e9427-c3c6-65ee-20db-0e93343a1861`。**新服务器卡号/型号/UUID 均需重查，不能沿用这个 UUID。**

配置 `configs/local_gpu1_20261031.yaml`：真实 OpenDrawer、左相机、H=16、三帧未来、batch=1、K=2 full unroll、BF16 Qwen、FP32 candidates、语言可训练、LoRA=false、阶段 bi、计划 10 steps。相对于正式基线是明确标注的工程短试，非正式科学结果。

为了在有限 GPU 显存下验证，使用 `save_on_cpu(pin_memory=True)` 存储反传张量，AdamW 的 `foreach=False` 避免大组 GPU 临时数组；没有用 LoRA、截断展开或缩小任务几何绕过方法。

记录：

| 时间 | 事件 |
|---|---|
| 17:18:47 | 资源检查、开始加载真实预训练模型 |
| 17:19:20 | 模型加载成功，总参数 4,488,412,545，可训练参数 4,051,468,289，CUDA allocated 9,158,459,904 bytes |
| 17:19:21 | 第一个 step 开始，episode 371、frame 0 |
| 17:20:17 | 本进程 RSS 57,601 MiB，触发 50 GiB 主机内存保护，SIGTERM 终止，exit=-15 |

监督脚本还设置主机可用内存低于 12 GiB、总试跑超过 30 分钟的保护。本次是**主机内存保护退出，不是已确认的 CUDA OOM**。普通 CPU saved-tensor 卸载可能重复拷贝多次用到的模型权重，这是后续优化动机，尚无完整内存 profile 证明它是唯一原因。

没有 `training_step_complete`；`result.json` 和计划 checkpoint `bi/local_gpu1_bi_step10.pt` 都不存在。不能声称已经跑了 10 步、完成训练、收敛或成功恢复。

原证据：`logs/runs/local_gpu1_20261031/{training.log,status.json,supervisor.log,stats_fit.log}`，没有传到新服务器，本文已摘录关键状态。

### 8.3 未完成的内存优化：务必标成实验性

新增 `energywam/training/offload.py::parameter_aware_cpu_offload`：

- 用 saved_tensors_hooks 将保存的张量放 CPU。
- 只针对模型参数持有的稳定 storage/view/version 去重，避免每次能量前向都再复制同一参数视图。
- 不对普通 activation 用裸地址去重，因为 CUDA allocator 地址可复用，错误缓存会污染梯度。
- 这意在改变存储，不改变求导图或训练目标；**该意图尚未充分验证**。

一次临时验证脚本在执行 `p.grad.clone()` 时遇到 `None` gradient（`AttributeError`），尚未走到完整对照断言。某些参数对候选梯度损失可能确实没有梯度，这不直接证明 offload 实现错误，但也绝不构成数值等价通过。

随后用户暂停。没有在真实 4B 上重试去重方案，也没有成功完成 CUDA/小 Qwen 的完整梯度对照。**接手不要把此文件当作已验收功能**。应先修正验证脚本处理 None，检查 native 与 offload 的 energy、候选梯度、高阶参数梯度、重复视图/参数版本、checkpoint 保存恢复，之后才能考虑资源受控的真实试跑。

`gpu_preflight.py` 已修正 `EnergyWAMTrainer(cfg, model)` 参数顺序，新增 opt-in checkpoint、低内存 AdamW 和去重卸载选项；`run_local_gpu_trial.py` 有 GPU 锁和资源保护。它们包含本轮后续变更，不在此前 26 项测试的整体认证范围内。

## 9. 已验证内容、测试证据与未验证边界

### 9.1 2026-10-31 针对 Qwen 单次语言前向的复核

| 测试模块 | 结果 | 范围 |
|---|---|---|
| `test_qwen_single_pass` | 3/3 | 实际 HF 小 Qwen 模块 hook，条件不进语言层、每次能量各层仅调用一次、复用条件、bi/joint/推理、候选和二阶梯度 |
| `test_qwen_structure` | 3/3 | 原生 image scatter、DeepStack、mRoPE、padding 与 double backward |
| `test_gradients` | 5/5 | 候选梯度、outer unroll、固定分支、训练/推理图边界 |
| `test_real_integration` | 15/15 | 模型生命周期、checkpoint、推理上下文与真实接口等 |

合计 26/26；主要使用随机小尺寸 Qwen，一部分集成测试读取原服务器的真实 DINO 权重。绝不是 pretrained 4B GPU 验证。另有独立检查确认提取 embedding helper 前后 embedding/energy 逐值一致、state_dict 键不变。

修正过测试夹具：`tests/real_fixtures.py` 重建 `ew.core` 后必须同步 `ew.stage_runner.core`，否则测试训练仍走旧核心。

### 9.2 其他历史记录

- 标准化主线历史 CPU 检查 165/165；随后阶段记录 166/166。
- 旧超算配置/episode split 检查 49/49，提交额度判断 5/5。
- 下载安全/归档布局/续传/配额回归 32/32；本机启动下载前再次通过。

这些来自不同阶段和套件，不应简单相加后声称“当前所有代码 X/X 全通过”。尤其后来新增的 offload 和本地监督脚本没有完整重跑全套验收。

### 9.3 尚未取得的结果

- 真实 4B full-unroll 的成功完整 optimizer step、稳定显存/主机内存和每步耗时。
- 真实数据 bi -> joint -> checkpoint -> resume 的完整训练链路。
- 全量训练集标准化统计、正式训练与 held-out 完整离线评估。
- RoboCasa365 MuJoCo/renderer/controller 闭环评测及任务成功率。
- 与 StarVLA/LDA/动作单模态等基线在一致机器人、数据和计算预算下的公平比较。
- Future 是否真正帮助 Action、相容性/耦合诊断及泛化结论。

论文仍是研究方案和待实验验证稿，不能填入没有真实证据的结果。

## 10. 新服务器环境与移植注意

源服务器参考：Linux x86_64、Python 3.10.22、torch 2.6.0+cu124、torchvision 0.21.0、transformers 4.57.0、tokenizers 0.22.1。具体参考版本见 `requirements.txt`。

依赖重点：

- Qwen adapter 依赖 transformers 4.57.0 的内部多模态 API；升级版本要重新验证，而不是只看 import 成功。
- PyTorch CUDA wheel、驱动和新 GPU 架构需实际匹配；`requirements.txt` 不是完整锁定环境，也不代表已经安装。
- pandas/pyarrow/av 等用于真实 LeRobot 读取；可选 LoRA 的 peft、仿真栈和完整 StarVLA 集成依赖需另行处理。
- 原环境只是借用 `/data3/xyx_zhuzicheng/miniconda3/envs/LDA/bin/python`，没有搬到新服务器。
- `../starVLA` 和 `../LDA-1B` 未复制。独立 EnergyWAM 不等于完整 StarVLA/仿真评测环境；部分兼容测试和 baseline 脚本需要这些仓库。
- `scripts/env.sh`、部分 configs、`tests/real_fixtures.py`、HPC 脚本有原机器绝对路径。先列出并按新计划显式覆盖，不能创建伪权重/假数据使路径检查通过。
- 原超算是 ARM64，与当前 GPU 服务器不是同一平台；不可照搬 ARM 环境或自动提交 Slurm 脚本。
- 旧 README/IMPLEMENTATION_STATUS/SOURCES_AND_ENV 部分“数据不存在”“未下载权重”“GPU 未跑”等描述是历史状态；以本文与实际文件为当前证据。

## 11. 给接手 GPT 的建议顺序

以下是将来的工作清单，**不是本次授权立即执行安装、下载或训练的命令**。

1. 阅读本文件、论文 `energywam/LTE_WAM_Paper_Draft_CN.md`、`docs/STANDARDIZED_BASELINE.md` 和实际模型/solver/stages 实现。
2. 向用户确认变更后的目标/范围；在此之前保持暂停。只读检查新服务器 GPU/驱动/CPU RAM/磁盘、已有环境、已有数据/权重。
3. 确认新数据存储目录及复用策略；用户允许后再下载/传递数据和权重。不要误把旧服务器 data7 当成本机挂载。
4. 构建独立环境，按新 GPU 匹配 CUDA，固定 transformers；不破坏用户已有环境和其他实验。
5. 调整路径，先跑必要 CPU/小 Qwen 回归。标明需要真实 DINO/StarVLA 的测试；缺依赖不能算通过。
6. 验证 offload 或其他**保持数学条件**的内存优化。比较 energy、一阶候选梯度和高阶参数梯度，而不只是“不报错”。
7. 固定 episode-disjoint split，先用标为 subset 的统计完成工程检查；正式结果另拟合 full-train 统计。
8. 得到用户训练授权且资源检查通过后，由助手实际启动有时间/内存边界的单卡真实单步检查；确认 optimizer step 完成和参数更新。
9. 再做短 bi -> joint -> save/resume -> held-out eval，保存 resolved config、种子、bundle fingerprint 和真实资源指标。
10. 最后做完整闭环与论文实验；每一阶段区分“计划、已启动、已完成、已核验”。

不能未经用户决定而把 LoRA、冻结语言、truncated unroll、换机器人数据、减少未来几何当作原方法。若正式目标受资源限制，应说明真实限制及可选方案。

### 11.1 工具入口示例（仅供未来授权后参考）

```bash
cd /root/zzc/energyWAM
# 以下假设已经另外配置好 Python；本次迁移并没有执行这些步骤。
python tests/run_all.py test_gradients
python tests/run_all.py test_qwen_single_pass
python -m unittest discover -s hpc/tests -p 'test_download*.py'
```

真实统计/训练/评估入口：

```text
python hpc/scripts/make_episode_manifests.py --help
python scripts/fit_normalization_stats.py --help
python scripts/train.py --help
python hpc/scripts/gpu_preflight.py --help
python scripts/evaluate.py --help
```

传入真实 train/val/test manifests；`--max-samples` 会生成 subset 统计，不是正式全量结果。`scripts/train.py` 支持 `--resume` 和 `--init-from`，阶段切换 optimizer policy 必须显式、并保持统计坐标一致。

用户偏好：需要实验时由助手检查资源并实际执行，不要只让用户复制启动命令；不停止或复制其他人的任务。耗时冗杂工作可委派本地 Claude CLI，但主 GPT 要独立核验。之前一轮 Claude CLI 在专项审查上耗时较长，最终测试由主 Agent 完成；不能把代理口头结论当成验证结果。

## 12. 接手后可直接向用户说明的话

> 我已阅读交接。新服务器现在只有经过文件校验的代码和说明，没有已迁移的数据、模型或环境。旧机器下载和试跑处于暂停；真实 4B 尚无已确认成功的训练 step。核心方法是 Qwen ViT/文本 embedding 准备条件，在完整 C/Z/A/E 序列上每次能量评价调用一次语言 Transformer，以双向条件训练再联合优化；最新 CPU 卸载去重仍需验证。我会先按你的新计划确认范围，再处理资源和数据，不自动恢复旧任务。

## 附录 A：完整 RoboCasa365 下载 manifest

下面 JSON 是官方注册表固定快照的 56 项配置，包含 50 个 target/human 任务与 6 类厨房资产，可从本说明独立恢复为 `downloads/full_robocasa365_manifest.json`。它只是 URL/版本/大小/布局配置，**不包含数据本体、模型或认证信息**。

只有获得新的下载授权后才创建队列。`relative_destination` 相对于用户选择的存储根目录；不要自动把它写到代码目录或某个不存在的旧超算 home 下。

```json
{
  "schema": "energywam_robocasa365_download_v1",
  "source": "official robocasa/robocasa main box_links registry snapshot 2026-10-30",
  "tasks": 50,
  "atomic": 18,
  "composite": 32,
  "asset_bundles": 6,
  "entries": [
    {
      "id": "dataset_OpenDrawer",
      "kind": "dataset",
      "key": "target/atomic/OpenDrawer/20250816/lerobot.tar",
      "url": "https://utexas.box.com/shared/static/mpztcjagk0fqs7e5360znjvhs08hnyjp.tar",
      "relative_destination": "datasets/robocasa365/v1.0/target/atomic/OpenDrawer/20250816/lerobot",
      "archive_bytes": 478965760,
      "range_supported": true
    },
    {
      "id": "asset_fixtures_lightwheel",
      "kind": "asset",
      "key": "fixtures_lightwheel",
      "url": "https://utexas.box.com/shared/static/idbncsadpnaz1jfl4i6m8qejawk7p9pi.zip",
      "relative_destination": "sim/robocasa365/robocasa/models/assets/fixtures",
      "archive_bytes": 529745465,
      "range_supported": true,
      "uncompressed_bytes": 1443420911,
      "expected_files": 6047,
      "archive_root": "fixtures"
    },
    {
      "id": "dataset_OpenStandMixerHead",
      "kind": "dataset",
      "key": "target/atomic/OpenStandMixerHead/20250818/lerobot.tar",
      "url": "https://utexas.box.com/shared/static/uf8tqixdm6elig7t3kgqnioxdnunxxqd.tar",
      "relative_destination": "datasets/robocasa365/v1.0/target/atomic/OpenStandMixerHead/20250818/lerobot",
      "archive_bytes": 289402880,
      "range_supported": true
    },
    {
      "id": "asset_textures",
      "kind": "asset",
      "key": "textures",
      "url": "https://utexas.box.com/shared/static/4i85ileasdvstmlln5sbvzptz7keuoy1.zip",
      "relative_destination": "sim/robocasa365/robocasa/models/assets/textures",
      "archive_bytes": 543180421,
      "range_supported": true,
      "uncompressed_bytes": 545204854,
      "expected_files": 225,
      "archive_root": "textures"
    },
    {
      "id": "dataset_TurnOnElectricKettle",
      "kind": "dataset",
      "key": "target/atomic/TurnOnElectricKettle/20250817/lerobot.tar",
      "url": "https://utexas.box.com/shared/static/ctirlg0mryjz511rb2pntdgmwir2m3vw.tar",
      "relative_destination": "datasets/robocasa365/v1.0/target/atomic/TurnOnElectricKettle/20250817/lerobot",
      "archive_bytes": 331950080,
      "range_supported": true
    },
    {
      "id": "asset_objects_lightwheel",
      "kind": "asset",
      "key": "objects_lightwheel",
      "url": "https://utexas.box.com/shared/static/vckqvvkh1z8t69k8qcpcmee6k66stii4.zip",
      "relative_destination": "sim/robocasa365/robocasa/models/assets/objects/lightwheel",
      "archive_bytes": 791939656,
      "range_supported": true,
      "uncompressed_bytes": 1453357859,
      "expected_files": 17393,
      "archive_root": "lightwheel"
    },
    {
      "id": "dataset_TurnOnMicrowave",
      "kind": "dataset",
      "key": "target/atomic/TurnOnMicrowave/20250813/lerobot.tar",
      "url": "https://utexas.box.com/shared/static/ta8tufw9408tzx3fxffwr5lh1hknp8fz.tar",
      "relative_destination": "datasets/robocasa365/v1.0/target/atomic/TurnOnMicrowave/20250813/lerobot",
      "archive_bytes": 341268480,
      "range_supported": true
    },
    {
      "id": "asset_generative_textures",
      "kind": "asset",
      "key": "generative_textures",
      "url": "https://utexas.box.com/shared/static/ebaad09k82tmfmlq6ohdkmrh8izl9vn5.zip",
      "relative_destination": "sim/robocasa365/robocasa/models/assets/generative_textures",
      "archive_bytes": 1184675536,
      "range_supported": true,
      "uncompressed_bytes": 1196546822,
      "expected_files": 900,
      "archive_root": "generative_textures"
    },
    {
      "id": "dataset_CloseToasterOvenDoor",
      "kind": "dataset",
      "key": "target/atomic/CloseToasterOvenDoor/20250818/lerobot.tar",
      "url": "https://utexas.box.com/shared/static/86caoedy7compqenfh5a8ry62xebruyh.tar",
      "relative_destination": "datasets/robocasa365/v1.0/target/atomic/CloseToasterOvenDoor/20250818/lerobot",
      "archive_bytes": 368977920,
      "range_supported": true
    },
    {
      "id": "asset_objaverse",
      "kind": "asset",
      "key": "objaverse",
      "url": "https://utexas.box.com/shared/static/03eionyo8fk3a9dsksq9jb8du5lqfw8h.zip",
      "relative_destination": "sim/robocasa365/robocasa/models/assets/objects/objaverse",
      "archive_bytes": 2163884721,
      "range_supported": true,
      "uncompressed_bytes": 6444097576,
      "expected_files": 39936,
      "archive_root": "objaverse"
    },
    {
      "id": "dataset_TurnOnSinkFaucet",
      "kind": "dataset",
      "key": "target/atomic/TurnOnSinkFaucet/20250812/lerobot.tar",
      "url": "https://utexas.box.com/shared/static/g0gamv85pvi4ab220szfhr72wg7tioap.tar",
      "relative_destination": "datasets/robocasa365/v1.0/target/atomic/TurnOnSinkFaucet/20250812/lerobot",
      "archive_bytes": 397762560,
      "range_supported": true
    },
    {
      "id": "asset_aigen_objs",
      "kind": "asset",
      "key": "aigen_objs",
      "url": "https://utexas.box.com/shared/static/nwi1vrn5pgbo95kushkasa3nx1i012ff.zip",
      "relative_destination": "sim/robocasa365/robocasa/models/assets/objects/aigen_objs",
      "archive_bytes": 5773223039,
      "range_supported": true,
      "uncompressed_bytes": 12980520150,
      "expected_files": 58733,
      "archive_root": "aigen_objs"
    },
    {
      "id": "dataset_TurnOffStove",
      "kind": "dataset",
      "key": "target/atomic/TurnOffStove/20250812/lerobot.tar",
      "url": "https://utexas.box.com/shared/static/xsvs8uazyfwoeqij298gsax3b400b4h8.tar",
      "relative_destination": "datasets/robocasa365/v1.0/target/atomic/TurnOffStove/20250812/lerobot",
      "archive_bytes": 423168000,
      "range_supported": true
    },
    {
      "id": "dataset_SlideDishwasherRack",
      "kind": "dataset",
      "key": "target/atomic/SlideDishwasherRack/20250820/lerobot.tar",
      "url": "https://utexas.box.com/shared/static/4mrm925oekfrlfyjlg9c8wflqktwla6y.tar",
      "relative_destination": "datasets/robocasa365/v1.0/target/atomic/SlideDishwasherRack/20250820/lerobot",
      "archive_bytes": 460021760,
      "range_supported": true
    },
    {
      "id": "dataset_NavigateKitchen",
      "kind": "dataset",
      "key": "target/atomic/NavigateKitchen/20250821/lerobot.tar",
      "url": "https://utexas.box.com/shared/static/z1td67netd0yyi9iiglws8hzxl9ueklj.tar",
      "relative_destination": "datasets/robocasa365/v1.0/target/atomic/NavigateKitchen/20250821/lerobot",
      "archive_bytes": 474685440,
      "range_supported": true
    },
    {
      "id": "dataset_CoffeeSetupMug",
      "kind": "dataset",
      "key": "target/atomic/CoffeeSetupMug/20250813/lerobot.tar",
      "url": "https://utexas.box.com/shared/static/rhrr67a994a16o25iprsfec07a1wmwm2.tar",
      "relative_destination": "datasets/robocasa365/v1.0/target/atomic/CoffeeSetupMug/20250813/lerobot",
      "archive_bytes": 476948480,
      "range_supported": true
    },
    {
      "id": "dataset_PickPlaceCounterToStove",
      "kind": "dataset",
      "key": "target/atomic/PickPlaceCounterToStove/20250818/lerobot.tar",
      "url": "https://utexas.box.com/shared/static/9zy06e4dcaxezsq35l6qpe39zi0uaqvh.tar",
      "relative_destination": "datasets/robocasa365/v1.0/target/atomic/PickPlaceCounterToStove/20250818/lerobot",
      "archive_bytes": 546488320,
      "range_supported": true
    },
    {
      "id": "dataset_PickPlaceCounterToCabinet",
      "kind": "dataset",
      "key": "target/atomic/PickPlaceCounterToCabinet/20250811/lerobot.tar",
      "url": "https://utexas.box.com/shared/static/ersnvqs6rcy6yge2fy5hmna7d954v9ty.tar",
      "relative_destination": "datasets/robocasa365/v1.0/target/atomic/PickPlaceCounterToCabinet/20250811/lerobot",
      "archive_bytes": 561387520,
      "range_supported": true
    },
    {
      "id": "dataset_PickPlaceToasterToCounter",
      "kind": "dataset",
      "key": "target/atomic/PickPlaceToasterToCounter/20250817/lerobot.tar",
      "url": "https://utexas.box.com/shared/static/4gn9j4llpxb2v8pk2r00im1a1zz7qqqm.tar",
      "relative_destination": "datasets/robocasa365/v1.0/target/atomic/PickPlaceToasterToCounter/20250817/lerobot",
      "archive_bytes": 612025344,
      "range_supported": true
    },
    {
      "id": "dataset_CloseFridge",
      "kind": "dataset",
      "key": "target/atomic/CloseFridge/20250816/lerobot.tar",
      "url": "https://utexas.box.com/shared/static/6yw8qwkwufg7di063v7e5zbcyhgn8nlf.tar",
      "relative_destination": "datasets/robocasa365/v1.0/target/atomic/CloseFridge/20250816/lerobot",
      "archive_bytes": 620166144,
      "range_supported": true
    },
    {
      "id": "dataset_PickPlaceDrawerToCounter",
      "kind": "dataset",
      "key": "target/atomic/PickPlaceDrawerToCounter/20250820/lerobot.tar",
      "url": "https://utexas.box.com/shared/static/xec45dl0meig2yl1v4cmi1hqxw3eu74l.tar",
      "relative_destination": "datasets/robocasa365/v1.0/target/atomic/PickPlaceDrawerToCounter/20250820/lerobot",
      "archive_bytes": 620431360,
      "range_supported": true
    },
    {
      "id": "dataset_CuttingToolSelection",
      "kind": "dataset",
      "key": "target/composite/CuttingToolSelection/20250814/lerobot.tar",
      "url": "https://utexas.box.com/shared/static/k05932wjpc1xjvg6wyfengklxj7fokq4.tar",
      "relative_destination": "datasets/robocasa365/v1.0/target/composite/CuttingToolSelection/20250814/lerobot",
      "archive_bytes": 636323840,
      "range_supported": true
    },
    {
      "id": "dataset_PickPlaceSinkToCounter",
      "kind": "dataset",
      "key": "target/atomic/PickPlaceSinkToCounter/20250813/lerobot.tar",
      "url": "https://utexas.box.com/shared/static/gtxm57dfse72c6zwbel05zhjritt1scq.tar",
      "relative_destination": "datasets/robocasa365/v1.0/target/atomic/PickPlaceSinkToCounter/20250813/lerobot",
      "archive_bytes": 683714560,
      "range_supported": true
    },
    {
      "id": "dataset_CloseBlenderLid",
      "kind": "dataset",
      "key": "target/atomic/CloseBlenderLid/20250822/lerobot.tar",
      "url": "https://utexas.box.com/shared/static/7anj5ckrlitmvwbexhh9elt9tof9v8sh.tar",
      "relative_destination": "datasets/robocasa365/v1.0/target/atomic/CloseBlenderLid/20250822/lerobot",
      "archive_bytes": 691466240,
      "range_supported": true
    },
    {
      "id": "dataset_StackBowlsCabinet",
      "kind": "dataset",
      "key": "target/composite/StackBowlsCabinet/20250815/lerobot.tar",
      "url": "https://utexas.box.com/shared/static/ifg8homjllollqodp7atye0mzq7bsazb.tar",
      "relative_destination": "datasets/robocasa365/v1.0/target/composite/StackBowlsCabinet/20250815/lerobot",
      "archive_bytes": 741949440,
      "range_supported": true
    },
    {
      "id": "dataset_RinseSinkBasin",
      "kind": "dataset",
      "key": "target/composite/RinseSinkBasin/20250816/lerobot.tar",
      "url": "https://utexas.box.com/shared/static/blk33oaca11933xgre1sxemmtluo46eo.tar",
      "relative_destination": "datasets/robocasa365/v1.0/target/composite/RinseSinkBasin/20250816/lerobot",
      "archive_bytes": 761405440,
      "range_supported": true
    },
    {
      "id": "dataset_PanTransfer",
      "kind": "dataset",
      "key": "target/composite/PanTransfer/20250817/lerobot.tar",
      "url": "https://utexas.box.com/shared/static/734jsqsgy5uu66gufuj6ltl0oyjybf28.tar",
      "relative_destination": "datasets/robocasa365/v1.0/target/composite/PanTransfer/20250817/lerobot",
      "archive_bytes": 778649600,
      "range_supported": true
    },
    {
      "id": "dataset_OpenCabinet",
      "kind": "dataset",
      "key": "target/atomic/OpenCabinet/20250813/lerobot.tar",
      "url": "https://utexas.box.com/shared/static/b15gbqq99i6kx8bogkdkq9lvurhbg9xw.tar",
      "relative_destination": "datasets/robocasa365/v1.0/target/atomic/OpenCabinet/20250813/lerobot",
      "archive_bytes": 787979264,
      "range_supported": true
    },
    {
      "id": "dataset_KettleBoiling",
      "kind": "dataset",
      "key": "target/composite/KettleBoiling/20250814/lerobot.tar",
      "url": "https://utexas.box.com/shared/static/r3rwnzdw6caab3vivwv6uknl9sr8ru6j.tar",
      "relative_destination": "datasets/robocasa365/v1.0/target/composite/KettleBoiling/20250814/lerobot",
      "archive_bytes": 833495040,
      "range_supported": true
    },
    {
      "id": "dataset_ScrubCuttingBoard",
      "kind": "dataset",
      "key": "target/composite/ScrubCuttingBoard/20250816/lerobot.tar",
      "url": "https://utexas.box.com/shared/static/yixaacen8dwnvlqmm9rmgf301nnwq6gk.tar",
      "relative_destination": "datasets/robocasa365/v1.0/target/composite/ScrubCuttingBoard/20250816/lerobot",
      "archive_bytes": 848977920,
      "range_supported": true
    },
    {
      "id": "dataset_WashLettuce",
      "kind": "dataset",
      "key": "target/composite/WashLettuce/20250814/lerobot.tar",
      "url": "https://utexas.box.com/shared/static/jmcowh1gtmshr52sc55p2ww265y97doh.tar",
      "relative_destination": "datasets/robocasa365/v1.0/target/composite/WashLettuce/20250814/lerobot",
      "archive_bytes": 899153920,
      "range_supported": true
    },
    {
      "id": "dataset_PrepareCoffee",
      "kind": "dataset",
      "key": "target/composite/PrepareCoffee/20250812/lerobot.tar",
      "url": "https://utexas.box.com/shared/static/6h5x8zxnzdd20h9kz7h85mz1bfo3u1lv.tar",
      "relative_destination": "datasets/robocasa365/v1.0/target/composite/PrepareCoffee/20250812/lerobot",
      "archive_bytes": 943380480,
      "range_supported": true
    },
    {
      "id": "dataset_WeighIngredients",
      "kind": "dataset",
      "key": "target/composite/WeighIngredients/20250812/lerobot.tar",
      "url": "https://utexas.box.com/shared/static/lbecgm6e5bk4icwylttsrkfxszh7d154.tar",
      "relative_destination": "datasets/robocasa365/v1.0/target/composite/WeighIngredients/20250812/lerobot",
      "archive_bytes": 1129082880,
      "range_supported": true
    },
    {
      "id": "dataset_PreSoakPan",
      "kind": "dataset",
      "key": "target/composite/PreSoakPan/20250809/lerobot.tar",
      "url": "https://utexas.box.com/shared/static/krqwe33yytnrse06xchr5e41sap41xfv.tar",
      "relative_destination": "datasets/robocasa365/v1.0/target/composite/PreSoakPan/20250809/lerobot",
      "archive_bytes": 1310842880,
      "range_supported": true
    },
    {
      "id": "dataset_BreadSelection",
      "kind": "dataset",
      "key": "target/composite/BreadSelection/20250815/lerobot.tar",
      "url": "https://utexas.box.com/shared/static/6o1ox753ry7wavwhc4fh34y3v1pczx3r.tar",
      "relative_destination": "datasets/robocasa365/v1.0/target/composite/BreadSelection/20250815/lerobot",
      "archive_bytes": 1386004480,
      "range_supported": true
    },
    {
      "id": "dataset_SetUpCuttingStation",
      "kind": "dataset",
      "key": "target/composite/SetUpCuttingStation/20250817/lerobot.tar",
      "url": "https://utexas.box.com/shared/static/pqckoqqbkthmoph3zm3vmmyq1st3owrh.tar",
      "relative_destination": "datasets/robocasa365/v1.0/target/composite/SetUpCuttingStation/20250817/lerobot",
      "archive_bytes": 1455892480,
      "range_supported": true
    },
    {
      "id": "dataset_CategorizeCondiments",
      "kind": "dataset",
      "key": "target/composite/CategorizeCondiments/20250814/lerobot.tar",
      "url": "https://utexas.box.com/shared/static/xuzdf1lfm9vzeoipi56iorzg2tt53y1a.tar",
      "relative_destination": "datasets/robocasa365/v1.0/target/composite/CategorizeCondiments/20250814/lerobot",
      "archive_bytes": 1548308480,
      "range_supported": true
    },
    {
      "id": "dataset_WaffleReheat",
      "kind": "dataset",
      "key": "target/composite/WaffleReheat/20250817/lerobot.tar",
      "url": "https://utexas.box.com/shared/static/d3bwmx0r6h4nyuv7qhizsyy2380cu230.tar",
      "relative_destination": "datasets/robocasa365/v1.0/target/composite/WaffleReheat/20250817/lerobot",
      "archive_bytes": 1616066560,
      "range_supported": true
    },
    {
      "id": "dataset_LoadDishwasher",
      "kind": "dataset",
      "key": "target/composite/LoadDishwasher/20250811/lerobot.tar",
      "url": "https://utexas.box.com/shared/static/k1qxg8rgjs0le1cnv98xylysh9t0dd8z.tar",
      "relative_destination": "datasets/robocasa365/v1.0/target/composite/LoadDishwasher/20250811/lerobot",
      "archive_bytes": 1658961920,
      "range_supported": true
    },
    {
      "id": "dataset_RecycleBottlesByType",
      "kind": "dataset",
      "key": "target/composite/RecycleBottlesByType/20250812/lerobot.tar",
      "url": "https://utexas.box.com/shared/static/iexgdqiq7tahatl4y98zwi5wu1ejj1sf.tar",
      "relative_destination": "datasets/robocasa365/v1.0/target/composite/RecycleBottlesByType/20250812/lerobot",
      "archive_bytes": 1732915200,
      "range_supported": true
    },
    {
      "id": "dataset_StirVegetables",
      "kind": "dataset",
      "key": "target/composite/StirVegetables/20250814/lerobot.tar",
      "url": "https://utexas.box.com/shared/static/f54ihxkmiiihnxnjvia54or0vqqorriw.tar",
      "relative_destination": "datasets/robocasa365/v1.0/target/composite/StirVegetables/20250814/lerobot",
      "archive_bytes": 1757286400,
      "range_supported": true
    },
    {
      "id": "dataset_SearingMeat",
      "kind": "dataset",
      "key": "target/composite/SearingMeat/20250812/lerobot.tar",
      "url": "https://utexas.box.com/shared/static/p2tngueixoz7v9jup2cr7905vbbbs77r.tar",
      "relative_destination": "datasets/robocasa365/v1.0/target/composite/SearingMeat/20250812/lerobot",
      "archive_bytes": 1771816960,
      "range_supported": true
    },
    {
      "id": "dataset_GatherTableware",
      "kind": "dataset",
      "key": "target/composite/GatherTableware/20250815/lerobot.tar",
      "url": "https://utexas.box.com/shared/static/aihrp20s9yoc73ngob1l6royp5gspldb.tar",
      "relative_destination": "datasets/robocasa365/v1.0/target/composite/GatherTableware/20250815/lerobot",
      "archive_bytes": 1802803200,
      "range_supported": true
    },
    {
      "id": "dataset_ArrangeTea",
      "kind": "dataset",
      "key": "target/composite/ArrangeTea/20250812/lerobot.tar",
      "url": "https://utexas.box.com/shared/static/1slqmm2stz2g0a3bavj6ums7ek64ht2m.tar",
      "relative_destination": "datasets/robocasa365/v1.0/target/composite/ArrangeTea/20250812/lerobot",
      "archive_bytes": 1880145920,
      "range_supported": true
    },
    {
      "id": "dataset_GetToastedBread",
      "kind": "dataset",
      "key": "target/composite/GetToastedBread/20250812/lerobot.tar",
      "url": "https://utexas.box.com/shared/static/zxh96llta6cdh9amc9wo2hbdu7u706q1.tar",
      "relative_destination": "datasets/robocasa365/v1.0/target/composite/GetToastedBread/20250812/lerobot",
      "archive_bytes": 1896581120,
      "range_supported": true
    },
    {
      "id": "dataset_DeliverStraw",
      "kind": "dataset",
      "key": "target/composite/DeliverStraw/20250813/lerobot.tar",
      "url": "https://utexas.box.com/shared/static/tza1gj7kvysu3b7v7pe3ccsi4by8qkey.tar",
      "relative_destination": "datasets/robocasa365/v1.0/target/composite/DeliverStraw/20250813/lerobot",
      "archive_bytes": 1924454400,
      "range_supported": true
    },
    {
      "id": "dataset_ArrangeBreadBasket",
      "kind": "dataset",
      "key": "target/composite/ArrangeBreadBasket/20250809/lerobot.tar",
      "url": "https://utexas.box.com/shared/static/7jgzm2brzvyknka6xge83ot21rwws9rd.tar",
      "relative_destination": "datasets/robocasa365/v1.0/target/composite/ArrangeBreadBasket/20250809/lerobot",
      "archive_bytes": 1972510720,
      "range_supported": true
    },
    {
      "id": "dataset_WashFruitColander",
      "kind": "dataset",
      "key": "target/composite/WashFruitColander/20250811/lerobot.tar",
      "url": "https://utexas.box.com/shared/static/8omrgv8mqo2p5s1hqjmas67p30etsiic.tar",
      "relative_destination": "datasets/robocasa365/v1.0/target/composite/WashFruitColander/20250811/lerobot",
      "archive_bytes": 1977344000,
      "range_supported": true
    },
    {
      "id": "dataset_StoreLeftoversInBowl",
      "kind": "dataset",
      "key": "target/composite/StoreLeftoversInBowl/20250813/lerobot.tar",
      "url": "https://utexas.box.com/shared/static/n69gzwudustsz0hc3txhqzhclnbpab6r.tar",
      "relative_destination": "datasets/robocasa365/v1.0/target/composite/StoreLeftoversInBowl/20250813/lerobot",
      "archive_bytes": 2009815040,
      "range_supported": true
    },
    {
      "id": "dataset_PortionHotDogs",
      "kind": "dataset",
      "key": "target/composite/PortionHotDogs/20250816/lerobot.tar",
      "url": "https://utexas.box.com/shared/static/vggfiwfep4n0jnm59a18qq3tcvmn8ccd.tar",
      "relative_destination": "datasets/robocasa365/v1.0/target/composite/PortionHotDogs/20250816/lerobot",
      "archive_bytes": 2022154240,
      "range_supported": true
    },
    {
      "id": "dataset_SteamInMicrowave",
      "kind": "dataset",
      "key": "target/composite/SteamInMicrowave/20250814/lerobot.tar",
      "url": "https://utexas.box.com/shared/static/orr1n70ald5dsep2kpdfnxo7jogcjqls.tar",
      "relative_destination": "datasets/robocasa365/v1.0/target/composite/SteamInMicrowave/20250814/lerobot",
      "archive_bytes": 2056683520,
      "range_supported": true
    },
    {
      "id": "dataset_GarnishPancake",
      "kind": "dataset",
      "key": "target/composite/GarnishPancake/20250815/lerobot.tar",
      "url": "https://utexas.box.com/shared/static/wshjpqv739wxu0zu47mtnlzehmm6ktoq.tar",
      "relative_destination": "datasets/robocasa365/v1.0/target/composite/GarnishPancake/20250815/lerobot",
      "archive_bytes": 2146775040,
      "range_supported": true
    },
    {
      "id": "dataset_SeparateFreezerRack",
      "kind": "dataset",
      "key": "target/composite/SeparateFreezerRack/20250815/lerobot.tar",
      "url": "https://utexas.box.com/shared/static/qf9mnd9kixmwt0hnciqtm8gpm3sqoy56.tar",
      "relative_destination": "datasets/robocasa365/v1.0/target/composite/SeparateFreezerRack/20250815/lerobot",
      "archive_bytes": 2225008640,
      "range_supported": true
    },
    {
      "id": "dataset_HeatKebabSandwich",
      "kind": "dataset",
      "key": "target/composite/HeatKebabSandwich/20250813/lerobot.tar",
      "url": "https://utexas.box.com/shared/static/wwot1hszlxuptxjvcwkpu4q2xpg2ogsx.tar",
      "relative_destination": "datasets/robocasa365/v1.0/target/composite/HeatKebabSandwich/20250813/lerobot",
      "archive_bytes": 2481725440,
      "range_supported": true
    },
    {
      "id": "dataset_MakeIceLemonade",
      "kind": "dataset",
      "key": "target/composite/MakeIceLemonade/20250813/lerobot.tar",
      "url": "https://utexas.box.com/shared/static/spj7rp3j0q8qu1ncj5lc8bosnebmlpwl.tar",
      "relative_destination": "datasets/robocasa365/v1.0/target/composite/MakeIceLemonade/20250813/lerobot",
      "archive_bytes": 2751324160,
      "range_supported": true
    },
    {
      "id": "dataset_PackIdenticalLunches",
      "kind": "dataset",
      "key": "target/composite/PackIdenticalLunches/20250815/lerobot.tar",
      "url": "https://utexas.box.com/shared/static/ic9q7o0s99boxmpqcl04iurj9dyt2lck.tar",
      "relative_destination": "datasets/robocasa365/v1.0/target/composite/PackIdenticalLunches/20250815/lerobot",
      "archive_bytes": 3433267200,
      "range_supported": true
    }
  ],
  "known_archive_bytes": 72544565510,
  "registry_sha256": {
    "box_links_ds.json": "f6544237349efdf2961657200e817f68c3d34139350c90559611ddd926451c8f",
    "box_links_assets.json": "918335d0585b1e9ed67072d68255db47f56a3815e9136f97ea8ced86c02f7c1d"
  },
  "uncompressed_asset_bytes": 24063148172,
  "robocasa_commit": "dc73a8ce18e4b8947e8df2ae2958245ca823f3b4",
  "source_archive_sha256": "5c24304a0c1c269bff8a617331d2c9c2c7c8edfff1b1102fca0bcd10dc28685f",
  "uncompressed_expected_bytes": 85621064844
}
```

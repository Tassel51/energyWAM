# EnergyWAM — Latent Transition Energy World Action Model

> **全量数据下载进行中（2026-10-30）**：用户已授权下载全部50个RoboCasa365 target/human任务与6类厨房资产，固定官方版本，约72.5GB压缩包、85.6GB解压量；另准备Robosuite机器人模型/mesh。已在超算后台启动两路可续传队列，OpenDrawer核验复用，新增OpenStandMixerHead已下载验证。其余仍在下载，不能视为全部完成。进度和路径见 [全量下载记录](hpc/docs/FULL_ROBOCASA365_DOWNLOAD.md)。


> **超算准备（2026-10-29）**：官方 OpenDrawer target/human 已完整部署到 `~/energywam_storage`（514 episodes、137431 frames）；真实 Qwen3-VL-4B 与 DINOv3 权重均逐文件 SHA256 核验。CPU 测试 166/166、HPC 配置/划分检查49项通过。GPU 提交被平台账号 `MaxSubmitJobs=0` 拒绝，尚未获得真实4B训练速度或显存结果；准备了最多等待6小时的自动提交流程，额度恢复后先15分钟单步验证，成功才接30分钟短训。详见 [HPC部署记录](hpc/docs/HPC_DEPLOYMENT.md)。


> **最新正式基线（用户已批准§4.8）**：paper_standardized_v1采用训练集统计标准化＋N(0,I)；旧配置在configs/legacy保留，旧checkpoint不静默迁移。最终独立165/165测试与CPU全链路6/6通过。真实RoboCasa365统计/预训练4B仍待验证。详见docs/STANDARDIZED_BASELINE.md与coordination/standardized_baseline_20261029/CODEX_FINAL_ACCEPTANCE.md；下方旧记录为历史阶段。


> **新版论文对照审查（2026-10-29）**：核心架构与两阶段训练符合；4项工程问题及诊断指标问题已修复，独立复跑115/115测试通过。归一化/初始化默认暂保留旧方案，待用户决定。逐节差异见 `docs/PAPER_ALIGNMENT_REVIEW_20261029.md`，最终核验见 `coordination/paper_audit_20261029/CODEX_FINAL_REVIEW.md`。


> **最终独立验收（2026-10-28）**：主 Agent 复跑 **95/95 测试通过**，日志
> `logs/codex-final-suite.log`。真实本地 AutoProcessor + 随机小 Qwen + 预训练 DINO
> 顶层 K=2 二阶反向及三种推理上下文通过，见 `logs/codex-final-native_processor.log`。
> 最新 CLI 已实跑 bi 12 步 → joint 12 步 → resume 至 step 24 → 离线评测，结果在
> `logs/codex-cli-*.log`、`logs/runs/codex_final_*`。预训练 4B / RoboCasa365 闭环仍未验证。
> 下文旧测试计数是各轮历史记录，最终计数以此处为准。


> 目标目录：`/data3/xyx_zhuzicheng/projects/hpc/energyWAM`
> 本项目是研究计划《Latent Transition Energy WAM》的可训练实现，复用 `../starVLA` 的
> 数据 / 训练 / 服务设施，但**不修改** `../starVLA`、`../LDA-1B` 与 `projects/LDA`。

---

## 0. 一句话

把 Qwen3-VL 改造成一个**标量 Energy 模型** `Eθ(C, Z, A)`，其中 `C` 是当前观测（视觉语言 + 当前 DINO + 机器人状态），
`Z` 是候选未来 DINO latent，`A` 是候选动作块。训练与推理都通过对 `(A, Z)` 求能量梯度并迭代优化完成
（Future→Action、Action→Future、联合），**不引入 Flow / Diffusion / 直接回归 / 重排序 / 打分头**，
也不用手写物理代价。

---

## 1. 研究计划 → 代码模块映射

| 研究计划（RESEARCH_PLAN.md） | 代码位置 |
|---|---|
| §3.2(1) 视觉语言条件 `T_VL` | `energywam/backbones/qwen3vl_adapter.py` `encode_condition()`（原生 `masked_scatter` 图像注入 + DeepStack + mRoPE） |
| §3.2(1) 当前 DINO `z_t → T_cur` | `energywam/backbones/dino_v3.py` + `energywam/models/energy_core.py` `build_current_dino_tokens()` |
| §3.2(2) 机器人状态 `q_t → T_state` | `energy_core.build_state_tokens()` |
| §3.2(3) Future latent `Z → T_future` | `energy_core.build_future_tokens()` + `energywam/energy/encodings.py`（时间/空间/camera/type 独立编码） |
| §3.2(4) Action `A → T_action` | `energy_core.build_action_tokens()`（step/type 编码，**无** Flow timestep） |
| §3.3 序列 `[C, T_future, T_action, e_E]` + Energy Head | `energy_core.energy()` / `EnergyHead` |
| §3.4 能量梯度迭代生成 | `energywam/energy/solver.py` `refine()` |
| §4 阶段一 双向条件训练 | `energywam/training/stages.py` `_run_bi()` / `_run_single()` |
| §4 阶段二 联合训练 | `stages._run_joint()` |
| §4 阶段三 联合推理 | `energywam/models/energy_wam.py` `predict()` / `energywam/inference/predictor.py` |

### 核心公式（与工程要求逐条对应）

```
gA, gZ = autograd.grad(E.sum(), [A, Z], create_graph=training)
A_{k+1} = A_k - alpha_A * gA
Z_{k+1} = Z_k - alpha_Z * gZ
```

* **同点同时更新**：两个梯度都在 `(A_k, Z_k)` 上取，然后一起更新。`joint_update_mode='alternating'`
  是**另一个显式配置**，绝不与同时更新混写。
* `E.sum()` 而非 `mean()`：每条样本的候选梯度不随 batch size 人为缩小；batch 平均放在外层损失里。
* 训练：`create_graph=True`，候选路径上**不 detach**，外层 loss 穿过整个 unroll 回传到 θ；
  内层**不做** `optimizer.step()`。
* 推理：每步 `detach().requires_grad_()` 释放历史图；用 `autograd.grad`（不是 `backward`），
  参数**不积累** `.grad`。

---

## 2. 数据流（真实 shape）

Qwen 条件编码与能量主干的边界（2026-10-31 明确）：

```text
当前 RGB → Qwen ViT + 原生 visual merger ─┐
指令文本 → Qwen token embedding ─────────┴→ T_VL（尚未经过语言 Transformer）
T_VL + 当前 DINO tokens + state tokens + future tokens + action tokens + e_E
  → Qwen language Transformer（每次能量计算一次）→ Energy Head → 标量能量
```

`encode_condition()` 通过 `_embed_vision_language()` 只做视觉和文本 embedding，
保留原生图像 token 替换、DeepStack 视觉特征和 mRoPE 位置；不会调用语言 Transformer。
`run_sequence()` 拼接完整序列后才调用语言 Transformer，DeepStack 在这次调用中注入。
训练和推理均复用这条路径；候选更新后的每次能量计算都需要重新经过语言 Transformer，
但复用当前观测的 embedding。“一次”指一次能量计算，不指整个多步优化只计算一次。

以 RoboCasa365 / PandaOmron 为目标任务（H=16, Da=12, Dq=32）：

| 张量 | shape | 说明 |
|---|---|---|
| 当前图像 | `B×V×3×224×224` | 首版 **V=1，仅 `robot0_agentview_left`** |
| language | `B` 条字符串 | |
| robot state | `B×32` | raw 16 维经 sincos 展开为 32（与 starVLA 训练契约一致） |
| action target `A*` | `B×16×12` | minmax 归一化到 `[-1, 1]` |
| action mask | `B×16×12` | 尾部无效元素置 0 |
| future RGB | `B×3×1×3×224×224` | offsets `[5, 11, 16]`，同一 episode |
| future target `Z*` | `B×3×1×196×384` | 冻结 DINOv3 的 **patch token**（去掉 CLS + 4 register） |
| future mask | `B×3×1×196` | 标记有效时间/视角/patch |
| future offsets | `B×3` long | 本样本实际使用的绝对帧号 |

**DINO**：`/data6/zzc/dinov3-vits16-pretrain-lvd1689m`，hidden 384 / patch 16 / register 4 / image 224。
224/16 = 14 → 196 patch。取 `last_hidden_state[:, 5:, :]`（丢 CLS + 4 register），并在加载时**断言**维度；
不匹配直接报错，**不静默换模型**。

**动作/未来对齐（待确认项）**：`示范动作 [a_t, …, a_{t+H-1}]`，未来偏移 `[5, 11, 16]`。
最后一个偏移 `16 = H` 指向**最后一个动作步之后的那一帧**。这一解释来自
`robocasa/scripts/dataset_scripts/convert_hdf5_lerobot.py` 的记录约定（`a_t` 对应 `obs_t → obs_{t+1}`），
但**未在本服务器的真实 RoboCasa365 数据上核验**（数据不在本机，见 §6）。已在
`energywam/data/future_offsets.py` 中集中实现并在测试中覆盖越界拒绝。

---

## 3. Qwen 多模态融合 / token 排序 / 位置编码 / mask

### 3.1 保留的原生路径

适配器 `Qwen3VLEnergyBackbone` **不重写 Transformers**，而是复用 `Qwen3VLModel` 的原生方法：

1. `get_input_embeddings()(input_ids)` → 词嵌入
2. `get_image_features(pixel_values, image_grid_thw)` → 图像特征 + **DeepStack** 特征列表
3. `get_placeholder_mask(...)` + `inputs_embeds.masked_scatter(image_mask, image_embeds)` → 原生图像 token 替换
4. `get_rope_index(input_ids, image_grid_thw, ...)` → **mRoPE 三轴 position_ids**
5. `language_model(inputs_embeds=…, position_ids=…, attention_mask=…, visual_pos_masks=…, deepstack_visual_embeds=…)`

DeepStack 只作用于 `visual_pos_masks` 为 True 的原生视觉位置；追加块用 False 填充，不会被误加。

### 3.2 追加块与位置编码

序列顺序（causal）：

```
[C_native_vl,  C_extra(dino+state),  T_future,  T_action,  e_E]
```

* **mRoPE 追加位置**：每样本取 `max(原生有效 mRoPE)`，`+1` 起连续编号，
  **三轴相同**（`_appended_position_ids`）。
* **独立语义编码**（加在 hidden space，不进 mRoPE）：
  future 的 **时间（用真实 offset）/ 空间 patch / camera / type**，
  action 的 **step / type**，以及 cond 的 dino/type、state/type、E/type。
  空间信息因此**不依赖** DINO token 的任意拼接顺序。
* **Energy token**：永远是序列**最后一列**，mask 恒为 True（`padding 只能屏蔽键，E 有效`）。

### 3.3 mask 语义

* attention：2D key padding mask（`create_causal_mask`），**只屏蔽 key**；
  被 padding 的 query 行允许产生垃圾，因为我们只读 E 的位置。
* 候选：`action_mask [B,H,Da]`、`future_mask [B,F,V,P]`，无效候选的 token 内容被置零，
  梯度被置零，refinement 中不移动。
* 损失：`masked_mse = sum((pred-target)²·mask) / num_valid`，
  **按有效元素数归一化**，20 万维 latent 误差不会和动作误差直接相加。

---

## 4. 完整 unroll 的梯度路径

```
theta ──► E_k = Eθ(C, Z_k, A_k)
              │
              ├─ autograd.grad(E_k.sum(), [A_k, Z_k], create_graph=True) ──► gA_k, gZ_k
              ▼
        A_{k+1} = A_k - α_A gA_k      Z_{k+1} = Z_k - α_Z gZ_k
              ▲                                  │
              └────────── k = 0 … K-1 ───────────┘
              ▼
        L = λ_A J_A(A_K, A*) + λ_Z J_Z(Z_K, Z*)   ──► 一次 backward，更新 θ
```

* **attention backend**：参考实现强制 `eager`。
  **实测证据**：在 torch 2.6.0+cu124 / transformers 4.57.0 上，用默认 `sdpa` 时
  二阶反向直接报 `derivative for aten::_scaled_dot_product_flash_attention_for_cpu_backward is not implemented`；
  强制 `eager` 后通过（见 `logs/test_results.json` 的 `test_qwen_structure`）。
  适配器在 `_enforce_attn_implementation()` 中对**外部传入的模型**也强制生效。
* 关闭：KV cache、gradient checkpointing、torch.compile（对本 framework）。
* dtype：候选 A/Z、能量累加、梯度、loss 均 **float32**；主干 dtype 可配（默认 float32，
  资源紧张时可改 bf16，但需重新跑二阶反向测试）。
* `unroll_mode='truncated'` 是**显式偏离**（默认关闭）：前 `K - truncation_length` 步 `no_grad`，
  名称明确，不用它冒充完整 unroll。

---

## 5. 训练阶段与消融

| stage | 内容 | 配置 |
|---|---|---|
| `bi` | Future→Action（固定 Z*）+ Action→Future（固定 A*），同一 Eθ | `configs/bi.yaml` |
| `joint` | 同时更新 A、Z，端点监督 | `configs/joint.yaml` |
| `direct_joint` | 从预训练 Qwen 直接 joint，跳过阶段一 | `configs/direct_joint.yaml` |
| `action_only` | 独立 `E(C,A)`，future 用**全 mask 的 learned placeholder** | `configs/action_only.yaml` |
| `action_given_future` | 单方向消融 | `configs/action_given_future.yaml` |
| `future_given_action` | 单方向消融 | `configs/future_given_action.yaml` |

**阶段切换**（bi → joint）：`EnergyWAMTrainer.switch_stage(...)`，
optimizer/scheduler 策略**显式**（`reset` / `continue`），并写入日志。
checkpoint 含 model / optimizer / scheduler / **stage / global_step / RNG / config / 归一化统计版本**。

---

## 6. 权重、数据与配置

### 权重

| 组件 | 路径 | 说明 |
|---|---|---|
| Qwen3-VL-4B | `/data6/zzc/Qwen3-VL-4B-Instruct` | 可配 `backbone.base_vlm` |
| DINOv3-ViT-S | `/data6/zzc/dinov3-vits16-pretrain-lvd1689m` | HF 格式，冻结 |

### 数据（**重要**）

* **RoboCasa365 数据不在本服务器上**。`playground/Datasets/robocasa365` 不存在，
  也未在 `/data6`、`/data3` 下检索到 OpenDrawer 等任务数据。
* `/data6/zzc/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim` 是 **GR1** 数据
  （`GR1ArmsAndWaistFourierHands`，state/action **44 维**，单 `ego_view` 相机 256×256，
  1000 episodes）。它**不是** RoboCasa365/PandaOmron，不能冒称，成功率也不可混用。
  本项目只用它做**数据管线实测**，并在输出里写明 `dataset_identity`。
* 实测记录见 `logs/data_probe_gr1_unified.*.json`。

### 配置

`configs/` 下：`tiny_cpu.yaml`（CPU 显式 tiny）、`real_smoke.yaml`、`bi.yaml`、`joint.yaml`、
`direct_joint.yaml`、`action_only.yaml`、`action_given_future.yaml`、`future_given_action.yaml`、
`eval.yaml`，以及对应的 `tiny_*.yaml`。全部可解析、可校验（见 `tests/test_compat.py`）。

> **2026-10-29 起**：正式配置默认 `data.normalization_profile=paper_standardized_v1`
> （论文 §4.8 标准化坐标 + N(0,I) 初始化，见 `docs/STANDARDIZED_BASELINE.md`）；
> 改动前的原配置逐字保存在 `configs/legacy/`，缺 `normalization_profile` 字段的旧 YAML
> 仍按 `legacy` 解释。新增 `configs/tiny_paper_standardized.yaml`（新版 tiny 验证）。

---

## 7. 如何启动

> 默认 shell 没有 `python`，统一用
> `/data3/xyx_zhuzicheng/miniconda3/envs/LDA/bin/python`（只借用，不改环境）。
> CPU 测试请设 `CUDA_VISIBLE_DEVICES=''` 并限制线程（`scripts/env.sh` 已设为 2）。

### 7.1 全部数学 / 集成测试（CPU，已实际运行）

```bash
cd /data3/xyx_zhuzicheng/projects/hpc/energyWAM
bash scripts/run_tiny_tests.sh
# 或单个模块
bash scripts/run_tiny_tests.sh test_gradients
```

### 7.2 tiny 端到端：bi 训练 → 保存 → 切 joint → 继续（CPU 过拟合演示）

> 这些脚本默认用 `configs/tiny_overfit.yaml`（`alpha=0.3, K=6, lr=3e-3`，未来几何缩到
> 与动作块同维），是**实测能清楚看到端点误差下降**的尺度。
> `configs/tiny_cpu.yaml` 保留工程要求的保守起点（`K=4, alpha=0.05`）作为 smoke 配置。

实测（CPU，2 线程，`logs/runs/tiny_overfit/`）：

```bash
STEPS=120 bash scripts/train_tiny_bi.sh
#   -> loss 2.128 -> 0.425 ; logs/runs/tiny_overfit/tiny_bi.pt

STEPS=120 bash scripts/train_tiny_joint_from_bi.sh logs/runs/tiny_overfit/tiny_bi.pt
#   -> switched: bi -> joint | optimizer: reset | scheduler: reset | step: 0
#      loss 3.283 -> 0.081 ; logs/runs/tiny_overfit/tiny_joint.pt

# 同阶段 resume（stage 必须一致，否则显式拒绝）
python scripts/train.py --config configs/tiny_overfit.yaml \
  --resume logs/runs/tiny_overfit/tiny_joint.pt --steps 40 --smoke \
  --set stage.stage=joint trainer.output_dir=logs/runs/tiny_resume
#   -> resumed at step 120, finished 40 steps, loss 2.877 -> 2.119
#      checkpoint: logs/runs/tiny_resume/tiny_overfit_step160.pt

STEPS=300 bash scripts/train_tiny_action_only.sh
bash scripts/train_tiny_single_direction.sh action_given_future
bash scripts/train_tiny_single_direction.sh future_given_action
```

### 7.3 真实 smoke / 完整训练（**需先确认 GPU**）

```bash
# 先确认有空闲显存；当前 9 张卡全被占用，未获批准前不要执行
nvidia-smi

export ENERGYWAM_ALLOW_GPU_TRAINING=1
export CUDA_VISIBLE_DEVICES=<一张有余量的卡>

bash scripts/real_smoke.sh                              # 2 步真实 4B + 冻结 DINO 二阶反向
bash scripts/train_bi.sh configs/bi.yaml                # 阶段一
bash scripts/train_joint.sh configs/joint.yaml logs/runs/energywam_bi/<ckpt>.pt
bash scripts/train_direct_joint.sh
bash scripts/train_action_only.sh
bash scripts/train_single_direction.sh action_given_future
```

所有真实训练脚本都带 `ENERGYWAM_ALLOW_GPU_TRAINING` 门禁，未设置时**拒绝启动**。

### 7.4 评测

```bash
# 离线动作 MSE（明确不是闭环成功率；报告里 is_closed_loop_success_rate: false）
CUDA_VISIBLE_DEVICES='' /data3/xyx_zhuzicheng/miniconda3/envs/LDA/bin/python \
  scripts/evaluate.py --config configs/tiny_overfit.yaml --synthetic 8 \
  --checkpoint logs/runs/tiny_overfit/tiny_joint.pt
#   -> logs/eval_report.json

# 闭环 RoboCasa365 成功率：沿用 starVLA 既有入口（EnergyWAM 不改写）
bash scripts/run_baseline_starvla.sh eval
```

### 7.5 数据探测

```bash
/data3/xyx_zhuzicheng/miniconda3/envs/LDA/bin/python scripts/data_probe.py \
  --root /data6/zzc/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/gr1_unified.PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_1000 \
  --horizon 16 --offsets 5 11 16
```

---

## 8. 与原研究计划的工程默认 / 必要偏差

1. **首版左视角 + state32**：starVLA 的 RoboCasa365 训练 registry 写 3 相机，评测只用左相机。
   本实现显式选择**左视角 + state32（sincos）**的匹配协议，**不继承**训练/评测错配。
2. **动作归一化 minmax**（`2*(x-min)/(max-min)-1`），state sincos，与
   `starVLA/dataloader/gr00t_lerobot/transform/state_action.py` 逐式一致
   （`energywam/data/normalization.py`）。
3. **attention backend 强制 eager**：实测 sdpa 不支持所需二阶反向（见 §4）。
4. **Qwen vision encoder 默认冻结**（`freeze_qwen_vision: true`），语言主干可训练。
   这是**工程默认**，不是论文定论。LoRA 是显式可选（`lora_enabled`），与主方法区分。
5. **Energy head 最后一层 bias 是规范自由度**：`E = W4·h + b4` 中 `b4` 只平移能量、
   不改变 `∇E`，而 refinement 只用 `∇E`。因此它的外层梯度恒为 0（已由有限差分确认）。
   这不是 bug，是方法的结构性个性。
6. **未来偏移 `[5,11,16]` 的帧对齐解释**（`t+16` 是最后一个动作步之后那帧）标为**待确认**，
   因为真实 RoboCasa365 数据不在本机，无法对实测样本核验。
7. **tiny overfit 测试用缩小的 future 几何**（3×1×2×4 = 24 维，与动作块同维），
   以便在几百步内清楚展示端点误差下降；solver/loss/梯度路径完全不变。原因写在
   `tests/test_overfit_tiny.py` 的 docstring。
8. **不复现 LDA 对照**：LDA-1B 仅作参考，本项目不声称同条件复现。
   Joint Flow WAM 基线在本仓库不存在，明确缺失。

---

## 9. 已实测资源需求

| 项目 | 数值 |
|---|---|
| tiny CPU 测试全套 | ~3 分钟，2 线程，无 GPU |
| tiny overfit（K=8, 300 步） | 每个 stage 约 30–65 秒 CPU |
| tiny 端到端 bi 200 步 | 见 `logs/` 的实际输出 |
| 真实 4B 全参数二阶训练 | **未执行**（9 张 GPU 全占用） |
| 真实 4B 二阶反向（tiny 随机配置） | 已通过，80 个参数拿到非零外层梯度 |

---

## 10. 目录

```
energyWAM/
├── README.md                    # 本文件
├── IMPLEMENTATION_STATUS.md     # 已完成 / 已运行 / 未运行
├── docs/
│   ├── ENGINEERING_REQUIREMENTS.md   # 原始工程要求（保留）
│   ├── RESEARCH_PLAN.md              # 原始研究计划（保留）
│   ├── ARCHITECTURE.md               # 架构与关键决策记录
│   └── SOURCES_AND_ENV.md            # 来源 / 环境 / 许可证
├── coordination/
│   ├── CLAUDE_TASK.md           # 任务书（保留）
│   ├── STATUS.md                # 进度
│   └── review_checklist.md      # 主 agent 验收清单（保留）
├── energywam/                   # 实现包（23 个模块，见 IMPLEMENTATION_STATUS.md）
├── configs/                     # 16 个可解析配置
├── scripts/                     # 16 个启动脚本
├── tests/                       # 11 个测试模块 / 48 个测试
└── logs/                        # 测试与训练日志（持久化）
```

## 11. 入口修复增量（2026-10-28）

主 agent 审查发现真实 CLI 入口仍有遗漏，本轮修复（测试：`tests/test_entrypoint_regressions.py`，
日志 `logs/codex-entrypoint-fix.log`）：

1. **`scripts/evaluate.py`**：真实数据分支改为经
   `energywam/data/energy_dataset.py::build_eval_examples` 构造 examples —— 与训练同一样本契约：
   `predict_action` 的 `image`/`lang`/`state` 单数键、target 用 `require_action_stats` 与训练同
   min-max 归一化、state 走 `sincos`、`data.data_mix` 根目录解析、相机顺序取 `data.camera_order`。
   离线评测接通 `require_future=False`：**不读取真实未来帧**、不需要 DINO。
   新增 `--max-samples`（默认 32），报告标 `subset`（离线子集，不冒称闭环 SR）。
   模型放 `cfg.trainer.device`（默认 CPU 配置即 CPU）。
2. **`scripts/serve.py`**：restore 后从 checkpoint `meta.norm_stats` 还原真实 action 统计
   （`norm_stats_from_meta`，即 `EnergyWAMTrainer.save` 序列化的逆），
   `EnergyWAMPredictor(action_stats=...)` 真正反归一化：`min=[2,4], max=[6,8]` 时 normalized 0
   映射 raw `[4,6]`。真实 backend 缺统计**拒绝启动**；tiny 可不带统计，但 metadata 显式标
   `action_units="normalized"` / `physical_eval_valid=false`。`--unnorm-key` 适配 metadata。
   模型放 `cfg.trainer.device`。
3. **trainer scheduler**：`trainer.warmup_steps` 接线默认 scheduler（warmup 后恒定的
   `LambdaLR`，工程选择：外层优化器不做衰减）；`warmup=0` 保持常数 LR，tiny 数值不变；
   传入自定义 scheduler 优先。`save`/`load`/`switch_stage` 的 scheduler resume/reset 现在
   **真恢复 / 真重置 LR 与状态**（resume 后下一步 LR 与不断训一致；switch reset 回初始 LR 日程），
   不只是计数器。缺 scheduler 状态时 `scheduler_policy='continue'` 显式报错。
4. **`sincos_state` 布局 `[sin(q), cos(q)]` 保持不变**（主 agent 已修复），数值测试
   （`test_review_regressions` / `test_entrypoint_regressions`）与本文档明确记录，不回退。

## 12. 标准化基线增量（2026-10-29，论文 §4.8 获批）

用户批准新版 §4.8 为正式基线：动作按**训练集**均值/标准差标准化、Future 按训练集**逐通道**
统计标准化、初始化 `N(0, I)`。已实现进完整训练/推理/恢复/服务/配置/统计管线（不是只改 YAML）。
详细说明、bundle/manifest 格式、准确命令、epsilon/统计定义、旧兼容范围与未验证项：
**`docs/STANDARDIZED_BASELINE.md`**。

要点：

* 坐标：`(x-mean)/(std+eps)`（population std，`eps=1e-6` 显式记录），逆变换精确含 eps；
  统计/累加 float64，候选/损失 float32；零方差维度有限可逆；mask 先屏蔽再计算。
* 优化变量就是 `tilde A / tilde Z`（solver/损失/`future_projector` 全在标准化坐标）；
  P_cur 仍 raw DINO；state sincos、相机协议不变。
* 统计为版本化 bundle（action/future mean+std+count、train episode manifest 哈希、dataset 身份、
  有序相机、H/offsets、DINO 标识/配置 hash、预处理版本、维度、coverage、fingerprint）；
  缺失/不匹配**明确拒绝**，绝不回退 raw/随机/测试集估计。
* 离线统计 CLI：`scripts/fit_normalization_stats.py`（流式 Welford/Chan、只读、拒 val/test、
  拒覆盖除非 `--overwrite`、`--max-samples` 标 `coverage=subset`、`--no-future` 只拟合动作）。
* checkpoint 自包含（persistent buffers + 内嵌 bundle）：外部 stats 移走仍可恢复/推理/服务；
  legacy↔standardized 互载、bundle 不一致 → 明确报错，无自动迁移；resume/bi→joint 校验同一统计。
* 服务按同一 bundle `normalized*(std+eps)+mean` 恢复物理动作；`future_latent` 默认仍 raw DINO，
  另给 `normalized_future_latent` 并标 space（legacy profile 不新增键，输出 schema 逐键不变）；
  手工 `future_target` 必须标 `future_target_space`。
* 验证：新增 34 项测试（手算统计/拒绝路径/N(0,I)、tiny 全链路 + K≥2 二阶反向、真实小 Qwen3-VL +
  真实 DINOv3 二阶反向、统计 CLI 在合成 LeRobot 数据（真实视频+真实 DINO）上实跑并手算核对）；
  CPU smoke 全链路：`bash scripts/smoke_standardized_cpu.sh`。
* 未测（诚实记录）：预训练 4B 未加载；真实 RoboCasa365 全量统计未拟合（数据不在本机）；
  式(40) 完整残差统计变体未实现（`current_dino_noise` 为固定尺度版本）。

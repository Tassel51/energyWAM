# 架构与关键决策记录

> **2026-11-04 新版补充（代码已改，尚未启动新训练）**：新实验使用 `text_dino_state`，删除 Qwen 视觉塔，保留文本 embedding、DINO 与状态条件，并加入 action/future 独立迭代 MLP、可选正值可学习步长。完整架构图、参数和训练验证安排见 [模型与实验方案](LTE_WAM_MODEL_AND_EXPERIMENT_SUMMARY_CN.md)。下文原生视觉注入 / DeepStack 的描述对应兼容模式 `qwen_vl`，不是新版配置路径。

> 配合 `README.md` 阅读。本文件记录「为什么这样做」以及被实测否决的方案。
> 日期：2026-10-28。

---

## 1. 模块分层

```
energywam/
├── config.py                 # 全部可配置字段 + 校验 + future offsets 规则
├── energy/
│   ├── types.py              # TokenGroup / ConditionBundle / EnergyReadout
│   ├── encodings.py          # future 时间/空间/camera/type、action step/type
│   ├── iteration.py          # 连续迭代 PE、独立 MLP、正值有界步长
│   ├── losses.py             # masked_mse / masked_cosine（按有效元素归一）
│   ├── solver.py             # 全 unroll refinement（E.sum，同点同时更新）
│   ├── solver_utils.py       # 候选梯度 clip / 候选 clamp（梯度行为已注明）
│   └── init.py               # A0 / Z0 初始化（normal / zeros / current_dino_noise）
├── backbones/
│   ├── tiny_transformer.py   # 显式 tiny：手写 causal attention（保证二阶反向）
│   ├── tiny_adapter.py       # tiny 的 encode_condition / run_sequence
│   ├── dino_v3.py            # 冻结 DINOv3，去 CLS+4register → 196×384
│   └── qwen3vl_adapter.py    # 真实 Qwen3-VL 适配（原生注入 + DeepStack + mRoPE）
├── models/
│   ├── energy_core.py        # projector / 编码 / Energy token / scalar head
│   └── energy_wam.py         # QwenEnergyWAM：forward / predict / batch 适配
├── data/
│   ├── future_offsets.py     # episode-safe 未来帧规划（唯一策略点）
│   ├── normalization.py      # 复用 starVLA min_max / mean_std 契约
│   ├── episode_dataset.py    # LeRobot episode 读取 + 真实样本探测
│   ├── cache.py              # DINO 特征 cache 契约（含完整 key）
│   └── tiny_data.py          # 确定性合成数据（仅显式 tiny）
├── training/
│   ├── stages.py             # bi / 单方向 / joint / direct_joint / action_only
│   ├── trainer.py            # 外层优化器（内层绝不 step）
│   └── checkpoint.py         # save/load/resume/switch + RNG
├── inference/predictor.py    # 联合推理 + 离线评测（明确非闭环 SR）
└── integration/starvla.py    # 显式 FRAMEWORK_REGISTRY 注册 + loss/predict 适配
```

**依赖方向**：`models` 依赖 `energy` 与 `backbones`；`backbones` 不依赖 `models`
（backbone 不反向持有 core，避免模块引用环；由 core 调用 backbone）。
`integration` 与 `inference` 是最外层，只有被显式调用时才导入。

---

## 2. 关键决策与被否决的方案

### 2.1 候选必须在语言层之前追加

**要求**：不要在 Qwen 算完所有语言层后才把 A/Z 拼到 hidden states 上再接 MLP。

**做法**：`Qwen3VLEnergyBackbone.run_sequence()` 亲自调用
`get_input_embeddings → get_image_features → masked_scatter → get_rope_index`，
拼上 `[C_extra, T_future, T_action, e_E]`，**然后**一次性调用
`language_model(inputs_embeds=…, position_ids=…, deepstack_visual_embeds=…)`。
候选因此真正参与 Qwen 的能量计算。

**被否决**：`model(**inputs)` 后拿 `hidden_states[-1]` 再 concat 候选 → 不符合目标架构。

### 2.2 attention backend 必须是 eager（实测结论）

**实测**（torch 2.6.0+cu124 / transformers 4.57.0 / CPU）：

| backend | 二阶反向 |
|---|---|
| 默认 sdpa | ✗ `derivative for aten::_scaled_dot_product_flash_attention_for_cpu_backward is not implemented` |
| eager | ✓ 80 个 HF 参数拿到非零外层梯度 |

证据：`logs/test_results.json` → `test_qwen_structure.qwen_native_path_and_second_order`。
适配器的 `_enforce_attn_implementation()` 对**外部传入的模型**也强制生效
（测试里先用 `Qwen3VLForConditionalGeneration(cfg)` 建模型，其 `_attn_implementation`
默认 sdpa，必须覆盖）。

**被否决**：用 sdpa + `torch.autograd.grad(..., create_graph=True)` —— 一步就崩。

### 2.3 mRoPE 追加位置

**要求**：追加位置在每样本原生有效 mRoPE 最大值之后，三轴相同连续编号。

**做法**（`_appended_position_ids`）：

```python
masked = pos.masked_fill(~mask.unsqueeze(0), INT64_MIN)   # pos: [3,B,S]
base = masked.amax(dim=(0, 2))                             # 每样本、跨三轴的最大有效值
start = base + 1
app = start[:, None] + arange(appended_len)                # 三轴相同
```

独立的 future 时间/空间/camera/type、action step/type 编码加在 **hidden space**，
不进 mRoPE，语义明确且与 Qwen 原生坐标不冲突。

### 2.4 padding 只屏蔽 key，E 永远有效

* attention：2D key padding mask（`create_causal_mask`）→ 只屏蔽 key。
* 被 padding 的 query 行允许产生垃圾 —— 我们只 gather E 位置的 hidden state。
* `build_energy_tokens()` 的 mask 恒为全 True。
* `test_shapes.energy_token_index_is_last_and_valid` 与
  `test_qwen_structure.qwen_padding_masks_keys_only` 覆盖。

### 2.5 Energy head 最后一层 bias = 规范自由度（数学性质）

`E = W₄·h + b₄`。refinement 用的是 `∇E`，而 `∇E = W₄·∇h` **与 `b₄` 无关**。
因此外层梯度 `∂L/∂b₄ ≡ 0`，PyTorch 报 `grad is None`（图上根本不存在这条边）。

这**不是 bug**，是标量能量模型的结构性个性：能量整体平移不改变能量下降方向。
已由 autograd 与中心差分双向确认
（`test_numerical.fd_outer_through_inner_parameters` 的 `[gauge]` 段）。
其余参数（`net.0/1`、projector、backbone）的外层梯度与有限差分吻合到 **1e-10**。

### 2.6 同点同时更新 vs alternating

`refine(joint_update_mode='simultaneous')`（默认/参考）：`gA_k, gZ_k` 都在
`(A_k, Z_k)` 上取，然后一起更新。测试
`simultaneous_uses_same_point` 用耦合二次能量构造出
`Z_same ≠ Z_alt`，确认两者确实不同；`alternating_is_a_separate_config_and_differs`
确认 alternating 是独立配置且轨迹不同。

### 2.7 action_only 的 future 占位

`build_future_placeholder_tokens()` 返回一个 **mask 全 False** 的 learned token 块。
结构上保持 `[C, T_future, T_action, E]` 几何不变，但 attention 里这些 key 全被屏蔽，
所以 `E(C, A)` **不可能**读到任何未来真值。

**观察**（不是证明）：`test_overfit_tiny.tiny_overfit_action_only_endpoint_error_drops` 的
history 里 future 误差逐步相同（1.222298264503479）。

**真正的不读真值证据**是输入扰动：`test_real_integration.
action_only_prediction_ignores_future_input_perturbation` 把 `future_target` /
`future_mask` / `future_offsets` / `action_target` 全部改动后，action-only 预测逐位不变；
`test_real_integration.real_predict_does_not_read_supervision` 确认联合推理在没有任何监督
字段时也能工作，且加入监督不改变输出。

---

## 3. 数值 / 优化尺度（短实验结论）

工程要求说「K、步长、初始化尺度是保守起点，用短实验检查数值、梯度与优化尺度」。
在 tiny CPU 上扫描的结果（单 batch 过拟合，300 步）：

| alpha | K | lr | 未来几何 | 结果 |
|---|---|---|---|---|
| 0.05 | 2 | 3e-3 | 4×8 (96 维) | action 1.15→1.07（**不明显**） |
| 0.3 | 4 | 3e-3 | 4×8 | action 1.15→0.90 |
| 0.5 | 4 | 3e-3 | 4×8 | action 1.15→0.78 |
| 0.3 | 12 | 3e-3 | 4×8 | action 1.14→0.39，future 1.07→0.81 |
| **0.3** | **8** | **3e-3** | **2×4 (24 维)** | **action 1.15→0.0012，future 1.04→0.0009** |
| 0.4 | 10 | 3e-3 | 2×4 | **发散**（future 1.04→1.92） |

结论：

1. `alpha*K ≈ 2.4` 稳定且有效；`alpha*K ≈ 4` 在这个玩具上会发散。
   所以正式配置仍用 `alpha=0.05, K=4` 作为保守起点，tiny 过拟合配置用 `alpha=0.3, K=6`。
2. 未来 latent 维度（96）远大于动作维度（24）时，几百步 CPU 内看不到明显下降。
   `configs/tiny_overfit.yaml` 把 future 几何缩到与动作同维以清楚展示优化本身
   （solver / loss / 梯度路径完全不变）。
3. 这些数字**不是**师兄已验证的超参，只是「数值与尺度可行性」的检查。

---

## 4. 时间对齐与 episode 边界

`energywam/data/future_offsets.py::plan_future_frames()` 是**唯一**策略点：

* 动作块覆盖 `[t, t+H-1]`；未来帧在 `t + offset`，offset ∈ `{5, 11, 16}`。
* 最后一个 offset `16 = H` 指向**最后一个动作步之后的那一帧**。
* 要求 offset 为正、严格递增、`≤ H`；否则拒绝。
* `require_full_horizon=True`（默认）时，动作与未来都必须落在 `[0, L)` 内才返回 valid；
  否则返回 `valid=False` 并给出越界原因。

**待确认**：`a_t ↔ obs_t..obs_{t+1}` 的记录约定来自
`robocasa/scripts/dataset_scripts/convert_hdf5_lerobot.py` 的 schema 注释，
但本服务器**没有 RoboCasa365 数据**，无法对真实样本核验。
`logs/data_probe_gr1_unified.*.json` 只证明 GR1 管线可读（见下）。

---

## 5. 数据可用性（实测，重要）

| 检查 | 结果 |
|---|---|
| `playground/Datasets/robocasa365` | **不存在** |
| `/data6`、`/data3` 下 `*robocasa365*` / `*OpenDrawer*` | **无匹配** |
| `/data6/zzc/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim` | 存在，**GR1** 数据 |

GR1 实测（`scripts/data_probe.py`）：

```
dataset_identity : gr1_unified.PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_1000
                   / GR1ArmsAndWaistFourierHands
episode_index    : 0      episode_length : 376    num_episodes : 1000
action_shape     : [376, 44]     state_shape : [376, 44]
cameras          : ["observation.images.ego_view"]   (1 个)
image_shape      : [256, 256, 3]
future_offsets_abs : [5, 11, 16]     future_valid : [1.0, 1.0, 1.0]
action shape     : [16, 44]      action_mask 全有效 44/44
action min/max   : -3.0 / 3.0
```

**结论**：GR1 只用于验证数据管线（episode-safe 未来帧、mask、offset、shape），
**不是** RoboCasa365/PandaOmron，不能冒称，成功率不可混用。
真实 RoboCasa365 训练/评测需先在 `data.data_root_dir` 放置对应 LeRobot 数据。

**首版协议选择**：starVLA 的 RoboCasa365 训练 registry 写 3 相机，评测只用左相机。
本实现显式选 **左视角 + state32（sincos）**，与评测一致，**不继承**训练侧的 3 相机错配。

---

## 6. Cache 契约

`energywam/data/cache.py` 定义 `CacheKey = (dataset_id, episode_index, frame_index,
camera, dino_checkpoint, preprocess_version, normalization_version)`，
并给出 `FeatureCache` 抽象 + `InMemoryFeatureCache` 参考实现。

**未实现**磁盘缓存 —— 按工程要求「cache 接口可仅定义可扩展契约无需实现缓存」。
真实训练可通过 batch 里传 `current_dino` 直接喂预计算特征（适配器因此把 DINO 设为
**惰性加载**，有 `current_dino` 时完全不碰权重）。

---

## 7. 与 StarVLA 的边界

* **不写** `../starVLA` 任何文件。框架通过
  `energywam/integration/starvla.py::register_with_starvla()` **显式注册**到
  `FRAMEWORK_REGISTRY`。
* `EnergyWAMFrameworkAdapter` duck-type `baseframework` 的
  `compute_loss('vla', batch)` / `predict_action(examples)` 两个接口。
* baseline 用 starVLA 既有入口（`scripts/run_baseline_starvla.sh` 只打印路径，不改写）。
* `tests/test_compat.py::starvla_frameworks_still_import` 回归验证：
  新增模块不会破坏既有 framework 的导入。
* LDA-1B 仅参考。**不声称**同条件复现 LDA / Joint Flow WAM 对照
  （本仓库也没有 Joint Flow WAM，明确缺失）。

---

## 8. 日志与研究诊断

`EnergyWAMTrainer` 每 `logging_frequency` 步往 `logs/runs/<run_id>/train_log.jsonl`
追加一条 JSON 记录（stage、global_step、各方向 loss、有效元素数）。
`solver.SolverState.as_diagnostics()` 提供能量轨迹 / 候选梯度范数 / 更新幅度。

**明确不作声称**：能量变化只是诊断量，**不是**跨任务成功概率或校准置信度；
端点监督也不保证全局能量排序，更不能据此说模型具备物理推理能力。
离线 action MSE 与闭环成功率是两件事，`scripts/evaluate.py` 的报告里
`is_closed_loop_success_rate: false` 写死。

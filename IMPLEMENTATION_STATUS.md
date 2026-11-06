# IMPLEMENTATION_STATUS

> **最新正式基线（用户已批准§4.8）**：paper_standardized_v1采用训练集统计标准化＋N(0,I)；旧配置在configs/legacy保留，旧checkpoint不静默迁移。最终独立165/165测试与CPU全链路6/6通过。真实RoboCasa365统计/预训练4B仍待验证。详见docs/STANDARDIZED_BASELINE.md与coordination/standardized_baseline_20261029/CODEX_FINAL_ACCEPTANCE.md；下方旧记录为历史阶段。


> **新版论文对照审查（2026-10-29）**：核心架构与两阶段训练符合；4项工程问题及诊断指标问题已修复，独立复跑115/115测试通过。归一化/初始化默认暂保留旧方案，待用户决定。逐节差异见 `docs/PAPER_ALIGNMENT_REVIEW_20261029.md`，最终核验见 `coordination/paper_audit_20261029/CODEX_FINAL_REVIEW.md`。


> **最终独立验收（2026-10-28）**：主 Agent 复跑 **95/95 测试通过**，日志
> `logs/codex-final-suite.log`。真实本地 AutoProcessor + 随机小 Qwen + 预训练 DINO
> 顶层 K=2 二阶反向及三种推理上下文通过，见 `logs/codex-final-native_processor.log`。
> 最新 CLI 已实跑 bi 12 步 → joint 12 步 → resume 至 step 24 → 离线评测，结果在
> `logs/codex-cli-*.log`、`logs/runs/codex_final_*`。预训练 4B / RoboCasa365 闭环仍未验证。
> 下文旧测试计数是各轮历史记录，最终计数以此处为准。


> 目标目录：`/data3/xyx_zhuzicheng/projects/hpc/energyWAM`
> 生成：2026-10-28。**已运行**与**未运行**严格分开，不把 skip 写成 pass。

---

## 1. 一句话状态

**tiny 路径端到端跑通**（双向训练 / 联合训练 / 单方向消融 / action_only / 联合推理 /
保存恢复 / bi→joint 切换 / resume / 离线评测，均在 CPU 实际执行）；
**真实 Qwen3-VL 结构与二阶反向路径已接好**，并在**随机初始化的极小 Qwen3VLConfig +
真实本地 AutoProcessor + 真实冻结 DINOv3** 上实测通过
（顶层 `QwenEnergyWAM`，原生图像注入 + DeepStack + mRoPE + 追加块，
head/projector/language 参数非零有限梯度）。

**尚未验证**（不能称完整实验证明成功）：
- **真实预训练 Qwen3-VL-4B 全参数二阶训练/推理未执行**（9 张 GPU 全被占用）。
- **RoboCasa365 数据不在本机**，真实任务训练与闭环成功率未执行。
- **真实数据的 a_t ↔ obs_t..obs_{t+1} 帧对齐仍标为待确认**。

---

## 2. 交付物清单

### 2.1 代码（`energywam/`，23 个模块）

| 模块 | 作用 |
|---|---|
| `config.py` | 全部可配置字段 + 校验 + future offsets 规则 |
| `energy/types.py` | TokenGroup / ConditionBundle / EnergyReadout |
| `energy/encodings.py` | future 时间/空间/camera/type、action step/type 独立编码 |
| `energy/losses.py` | masked_mse（按有效元素归一）、masked_cosine（默认关） |
| `energy/solver.py` | **全 unroll** refinement：`E.sum()`、同点同时更新、训练 `create_graph`、推理每步 detach |
| `energy/solver_utils.py` | 候选梯度 clip / 候选 clamp（梯度行为已注明） |
| `energy/init.py` | A0/Z0 初始化（normal / zeros / current_dino_noise） |
| `backbones/tiny_transformer.py` | 手写 causal attention（保证二阶反向） |
| `backbones/tiny_adapter.py` | tiny 的 encode_condition / run_sequence |
| `backbones/dino_v3.py` | 冻结 DINOv3，去 CLS+4register→196×384，维度断言 |
| `backbones/qwen3vl_adapter.py` | 真实 Qwen3-VL：原生注入 + DeepStack + mRoPE 追加 + eager 强制 |
| `models/energy_core.py` | projector / 编码 / Energy token / scalar head / action_only 路径 |
| `models/energy_wam.py` | `QwenEnergyWAM`：forward / predict / batch 适配 |
| `data/future_offsets.py` | episode-safe 未来帧规划（唯一策略点） |
| `data/normalization.py` | 复用 starVLA min_max / mean_std 契约（逐式一致） |
| `data/episode_dataset.py` | LeRobot episode 读取 + 真实样本探测 |
| `data/cache.py` | DINO 特征 cache 契约（完整 key：episode/frame/camera/ckpt/预处理/归一化版本） |
| `data/tiny_data.py` | 确定性合成数据（仅显式 tiny） |
| `training/stages.py` | bi / 单方向 / joint / direct_joint / action_only |
| `training/trainer.py` | 外层优化器（内层绝不 step）、日志 |
| `training/checkpoint.py` | save/load/resume/switch + RNG + 配置 + 归一化版本 |
| `inference/predictor.py` | 联合推理 + 离线评测（`is_closed_loop_success_rate: false`） |
| `integration/starvla.py` | 显式 `FRAMEWORK_REGISTRY` 注册 + loss/predict 适配 |

### 2.2 配置（`configs/`，16 个，全部可解析）

`tiny_cpu.yaml`（保守 smoke）、`tiny_overfit.yaml`（实测有效的尺度）、
`tiny_{bi→joint 相关, action_only, action_given_future, future_given_action, direct_joint}.yaml`、
`real_smoke.yaml`、`bi.yaml`、`joint.yaml`、`direct_joint.yaml`、
`action_only.yaml`、`action_given_future.yaml`、`future_given_action.yaml`、`eval.yaml`。

### 2.3 脚本（`scripts/`，16 个）

`env.sh`、`run_tiny_tests.sh`、`train_tiny_{bi,joint_from_bi,action_only,single_direction}.sh`、
`train_{bi,joint,direct_joint,action_only,single_direction}.sh`（**带 GPU 门禁，只交脚本不启动**）、
`real_smoke.sh`（GPU 门禁）、`eval_offline.sh`、`evaluate.py`、`train.py`、
`run_baseline_starvla.sh`、`data_probe.py`。

### 2.4 文档

`README.md`（中文）、`IMPLEMENTATION_STATUS.md`（本文件）、
`docs/ARCHITECTURE.md`、`docs/SOURCES_AND_ENV.md`，
以及原样保留的 `docs/ENGINEERING_REQUIREMENTS.md`、`docs/RESEARCH_PLAN.md`、
`coordination/CLAUDE_TASK.md`、`coordination/review_checklist.md`。

### 2.5 测试（`tests/`，11 个模块）

`harness.py`（pytest 未安装，自建收集器，**无 skip 语义**）、`common.py`、
`test_shapes` / `test_gradients` / `test_numerical` / `test_solver_analytic` /
`test_coupling` / `test_batch_masks` / `test_no_leakage` / `test_qwen_structure` /
`test_checkpoint_stage` / `test_overfit_tiny` / `test_compat`。

---

## 3. 已运行 — 测试（85 通过 / 0 失败，140 s）

命令：
```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /data3/xyx_zhuzicheng/miniconda3/envs/LDA/bin/python tests/run_all.py
```
日志：`logs/claude-final-tests.log`、`logs/test_results.json`、`logs/test_results_test_*.json`。

| 测试模块 | 数量 | 结果 |
|---|---|---|
| `test_mask_regression`（mask/NaN/mixed-dtype） | 8 | PASS |
| `test_review_regressions`（RNG/strict/offsets/harness/配置接线） | 10 | PASS |
| `test_real_integration`（顶层真实 Qwen+DINO 路径） | 15 | PASS |
| `test_no_leakage` | 7 | PASS |
| `test_solver_analytic` | 6 | PASS |
| `test_batch_masks` / `test_gradients` / `test_compat` | 5+5+5 | PASS |
| `test_checkpoint_stage` / `test_service_checkpoint` / `test_shapes` | 4+4+4 | PASS |
| `test_coupling` / `test_numerical` / `test_overfit_tiny` / `test_qwen_structure` | 3+3+3+3 | PASS |
| **合计** | **85** | **0 失败** |

### 3.1 数学与集成性质

| 检查 | 实测 |
|---|---|
| float64 有限差分（候选梯度） | max abs dev **1.662e-10**（tol 1e-5） |
| float64 有限差分（**外层穿过内层**的参数梯度） | max abs dev **1.248e-10**（tol 2e-4） |
| Energy head 末层 bias 规范自由度 | autograd=0，fd=0.000e+00 |
| 真实小 Qwen 顶层 K=2 二阶反向 | **80** 个 HF 参数非零外层梯度 |
| 真实 DINO CPU 提取（真实权重） | `[1, 196, 384]`，`strict=True` checkpoint 往返一致 |
| 固定 A*=5 在 action clamp[-1,1] 下 | 保持 5（不再被 clamp 改写） |
| 被 mask 的 action 维度改变能量 | **0**（修复前 1.056e-01） |
| masked target 含 NaN | 损失有限（修复前 nan） |
| assert_close 遇 NaN | **失败**（修复前误判通过） |

### 3.2 tiny 过拟合（alpha=0.3, K=8, lr=3e-3, 300 步，单 batch）

| stage | 动作端点 MSE | 未来端点 MSE |
|---|---|---|
| `bi` | 1.0614 → **0.0064**（-99.4%） | 1.1313 → **0.0021**（-99.8%） |
| `joint` | 1.1479 → **0.0012**（-99.9%） | 1.0371 → **0.0009**（-99.9%） |
| `action_only` | 0.9210 → **0.0236**（-97.4%） | 1.2223 → 1.2223（本次运行逐步相同） |

**措辞更正**：上面 action_only 的 future 误差「逐步相同」只是**本次测试的观察**，
本身不能证明「不读真值」。真正的不读真值结论来自**输入扰动测试**：
把 `future_target` / `future_mask` / `future_offsets` / `action_target` 全部大幅改动，
action-only 预测必须逐位不变（`test_real_integration.action_only_prediction_ignores_future_input_perturbation`），
联合推理则不得读取任何监督 future/action（`test_real_integration.real_predict_does_not_read_supervision`）。

## 4. 已运行 — 端到端脚本

| 命令 | 结果 | 日志 |
|---|---|---|
| `STEPS=300 bash scripts/train_tiny_bi.sh` | loss **2.128 → 0.019**，写 `logs/runs/tiny_overfit/tiny_bi.pt` | stdout |
| `STEPS=300 bash scripts/train_tiny_joint_from_bi.sh logs/runs/tiny_overfit/tiny_bi.pt` | `switched: bi -> joint \| optimizer: reset \| step: 0`，loss **3.268 → 0.007** | `logs/runs/tiny_overfit/train_log.jsonl` |
| `POLICY=continue … train_tiny_joint_from_bi.sh` | `optimizer: continue`，loss 3.268 → 0.355（50 步） | 同上 |
| `STEPS=300 bash scripts/train_tiny_action_only.sh` | loss **1.203 → 0.053** | stdout |
| `STEPS=200 bash scripts/train_tiny_single_direction.sh action_given_future` | loss **1.097 → 0.004** | stdout |
| `STEPS=200 bash scripts/train_tiny_single_direction.sh future_given_action` | loss **0.841 → 0.003** | stdout |
| `scripts/data_probe.py --root …/gr1_unified.PnPBottleToCabinet…` | 见 §5 | `logs/data_probe_gr1_unified.*.json` |

---

## 5. 已运行 — 真实数据管线探测

RoboCasa365 **不在本机**（`playground/Datasets/robocasa365` 不存在；`/data6`、`/data3`
下无 `*robocasa365*` / `*OpenDrawer*`）。用本机唯一的 LeRobot 语料做管线实测，
**明确标注不是 RoboCasa365**：

```
dataset_identity : gr1_unified.PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_1000
                   / GR1ArmsAndWaistFourierHands
episode_index 0, episode_length 376, num_episodes 1000
action [376,44] -> chunk [16,44]     state [376,44] -> [44]
cameras: ["observation.images.ego_view"] (1)     image [256,256,3]
future_offsets_abs [5, 11, 16]   future_valid [1, 1, 1]
action_mask 有效元素 44/44 × 16 步     action min/max -3.0 / 3.0
```

这证明：LeRobot 读取、episode-safe 未来帧、offset/mask、shape 契约**都能跑**。
它**不是** RoboCasa365/PandaOmron，动作/相机/状态/成功率不可混用。

---

## 6. 未运行（及原因）

| 项 | 原因 | 解锁条件 |
|---|---|---|
| **真实 Qwen3-VL-4B 全参数二阶训练** | 9 张 GPU 全部被占用（显存 22.8–40.6 / 24.5–46.1 GB，util 1–100%） | 主 agent / 用户确认有余量的卡后：`ENERGYWAM_ALLOW_GPU_TRAINING=1 CUDA_VISIBLE_DEVICES=<card> bash scripts/real_smoke.sh` |
| **真实 4B 推理验证（预训练权重）** | 同上 | 同上 |
| **RoboCasa365 训练 / 评测** | 数据不在本机 | 放置 `playground/Datasets/robocasa365` 后用 `configs/bi.yaml` + starVLA 既有评测入口 |
| **闭环成功率（SR）** | 需要仿真器 + 数据 + GPU | 用 `../starVLA/examples/simBenchmarks/Robocasa_365/eval_files/run_eval.sh`；EnergyWAM 不改写该协议 |
| **磁盘 DINO 特征缓存** | 工程要求允许「只定义契约」 | 按 `energywam/data/cache.py::FeatureCache` 子类化 |
| **LDA / Joint Flow WAM 同条件对照** | 本仓库无 Joint Flow WAM；LDA 仅参考 | —（明确缺失，不伪造） |

---

## 7. 工程默认与必要偏差（与原研究计划）

1. **首版左视角 + state32（sincos）**：starVLA 训练 registry 写 3 相机、评测只左相机；
   本实现选与评测一致的协议，不继承错配。
2. **attention backend 强制 eager**：实测 sdpa 不支持所需二阶反向
   （`derivative for aten::_scaled_dot_product_flash_attention_for_cpu_backward is not implemented`）。
3. **Qwen vision encoder 默认冻结**、语言主干可训练 —— 工程默认，不是论文定论。
   LoRA 是显式可选（`lora_enabled`），与主方法区分。
4. **Energy head 最后一层 bias 是规范自由度**，外层梯度恒为 0（有限差分确认）。
5. **未来偏移 `[5,11,16]` 的帧对齐解释**（`t+16` = 最后一个动作步之后那帧）**标为待确认** ——
   RoboCasa365 数据不在本机，无法对实测样本核验。
6. **tiny 过拟合用缩小的 future 几何**（24 维 = 动作块维度），以便几百步内清楚展示优化；
   solver / loss / 梯度路径完全不变。
7. **`alpha=0.05, K=4` 保守起点保留**；短实验证明 `alpha*K ≈ 2.4` 有效、`≈ 4` 在玩具上发散。
8. **不声称复现 LDA / Joint Flow WAM 对照**。

---

## 8. 资源限制与诚实边界

* 本次**未占用**任何已有实验的显存余量，**未启动**任何长训练，**未停止**任何其他进程。
* **未上传 / 未 push / 未远端安装 / 未提交 Slurm**；`../LDA-1B/tools/hpc/UPLOADS_PAUSED` 保持原状。
* **未下载**任何权重或数据集。
* **未修改** `../starVLA`、`../LDA-1B`、`projects/LDA` 的任何文件（`tests/test_compat.py` 回归）。
* 未读取 / 输出任何密钥。
* 结论表述：**「tiny 验证通过、真实 Qwen 结构与二阶反向已接好并在随机极小配置上验证、
  预训练 4B 路径待 GPU 资源验证」**，不能称完整实验证明成功。

---

## 9. 后续待办

1. GPU 有余量时跑 `scripts/real_smoke.sh`（2 步真实 4B 二阶反向），记录时间 / 峰值显存 / 参数梯度。
2. 放置 RoboCasa365 LeRobot 数据后跑 `configs/bi.yaml` 阶段一短训练，核对
   动作/相机/state 维度与 `data.camera_order`。
3. 对真实样本核验 `a_t ↔ obs_t..obs_{t+1}` 的帧对齐，把 §7-5 的「待确认」改成结论。
4. 如需 bf16 主干，重新跑 `test_qwen_structure` 的二阶反向验证（eager + bf16 组合未测）。
5. 磁盘 DINO cache 实现（按 `cache.py` 契约）。

---

## 10. 增量（2026-10-28）：真实 CLI 入口修复

主 agent 审查发现 `scripts/evaluate.py` / `scripts/serve.py` 真实分支仍有遗漏，定点修复并新增
`tests/test_entrypoint_regressions.py`（9 项，全部针对旧缺陷可红）。改动仅限：
`scripts/evaluate.py`、`scripts/serve.py`、`energywam/data/energy_dataset.py`、
`energywam/inference/predictor.py`（服务元数据/反归一化辅助）、
`energywam/training/trainer.py`、`energywam/training/checkpoint.py`、
`tests/test_entrypoint_regressions.py` 与文档/日志。未改 core/solver/其他测试。

* **evaluate**：examples 与训练同契约（`image`/`lang`/`state`、action stats 归一化 target、
  sincos state、`data_mix` 根解析、`data.camera_order`）；`require_future=False` 接通 ——
  离线评测不读真实未来帧、不需要 DINO；`--max-samples`（默认 32）+ 报告 `subset` 字段明确
  「离线子集，非闭环 SR」；模型放 `cfg.trainer.device`。
* **serve**：checkpoint `meta.norm_stats` → `NormStats`（`norm_stats_from_meta`）→
  `EnergyWAMPredictor(action_stats=...)`，返回的 `actions` 是真 raw；
  真实 backend 缺统计拒绝服务，tiny 无统计时 metadata 显式标 normalized / 无物理评测意义；
  `--unnorm-key` 与 metadata `available_unnorm_keys`/`default_unnorm_key` 对齐。
* **scheduler**：`trainer.warmup_steps` 驱动默认 warmup-后恒定 `LambdaLR`（工程选择：无衰减；
  `warmup=0` 常数 LR，tiny 数值不变；自定义 scheduler 优先）；resume 恢复 LR/状态、
  switch reset 回初始 LR 日程（不再只清计数）。
* **sincos**：`sincos_state` = `[sin(q), cos(q)]` 保持主 agent 修复后的布局，数值断言保留。

**已运行测试（本轮，CPU 2 线程，`logs/codex-entrypoint-fix.log`）**：

| 模块 | 结果 |
|---|---|
| `tests/test_entrypoint_regressions.py` | 9 通过 / 0 失败 |
| `tests/test_checkpoint_stage.py` | 4 通过 / 0 失败 |
| `tests/test_service_checkpoint.py` | 4 通过 / 0 失败 |
| `tests/test_review_regressions.py` | 10 通过 / 0 失败 |

**本轮未验证 / 已知限制**：

* `test_entrypoint_regressions` 全部在 **tiny 后端（mock 权重）+ CPU** 上；**不是**预训练
  4B 验证，也不含任何真实 DINO/RoboCasa365 数据读数。
* 真实数据 `evaluate.py` 分支未经端到端 CLI 跑通（本机无 RoboCasa365）；其正确性由
  `build_eval_examples` + 假 reader 测试覆盖（含「不读未来帧」计数断言）。
* `require_future=False` 仍按训练的合法起点采样（计划期要求未来偏移落在 episode 内），
  只是不读取/不使用未来像素；样本支持域与训练一致。
* 新 `scripts/serve.py` 入口组装（含真实统计反归一化）已用假模型输出 + tiny 真模型输出
  在 policy adapter 层验证；**未**再起一次 websocket loopback（既有
  `test_service_checkpoint.serve_loopback_request_response` 仍通过）。
* 长 overfit（`test_overfit_tiny`）按任务要求未重跑；主 agent 将独立跑一次全套。

## 11. 增量（2026-10-29）：论文 §4.8 标准化基线（用户已批准）

**范围**：仅 energyWAM；论文稿只读未改；旧 115 项测试科学条件未改。详细文档
`docs/STANDARDIZED_BASELINE.md`，报告 `coordination/standardized_baseline_20261029/REPORT.md`。

### 11.1 新增/修改

* 新增 `energywam/data/standardization.py`（唯一规范 API：`ZScoreStats` / `NormalizationBundle` /
  `StreamingMoments`（float64 Welford/Chan）/ `BundleFitter` / `load_bundle`/`save_bundle` / `StatsError`）、
  `energywam/data/stats_fit.py`（训练窗口流式拟合 + manifest 校验）、
  `scripts/fit_normalization_stats.py`（离线统计 CLI）、`scripts/smoke_standardized_cpu.sh`（CPU 全链路 smoke）。
* 接线：`config.py`（`normalization_profile`/`standardization_epsilon`/`norm_stats_bundle_path` +
  校验）、`normalization.py`（`zscore` 模式，旧模式不变）、`energy_dataset.py`（标准化输出 + 空间标记 +
  bundle 解析/校验）、`tiny_data.py`（合成 raw→标准化路径 + 合成 bundle）、`energy/init.py`
  （N(0,I) 基线 + `current_dino_noise` 先进标准化空间）、`models/energy_wam.py`（fail-fast stats、
  persistent buffers、future_target 空间契约、predict raw/normalized 双输出）、
  `training/checkpoint.py`（bundle 入 payload、坐标契约校验）、`training/trainer.py`、
  `inference/predictor.py`、`scripts/train|evaluate|serve.py`、`integration/starvla.py`（契约说明）。
* 配置：`configs/{bi,joint,direct_joint,action_only,action_given_future,future_given_action,real_smoke,eval}.yaml`
  切 `paper_standardized_v1`（新 run_id/output_dir，不覆盖旧结果）；原配置逐字存 `configs/legacy/`；
  新增 `configs/tiny_paper_standardized.yaml`（+ 合成 bundle 文件，明确 synthetic）。
* 文档：`docs/STANDARDIZED_BASELINE.md`（新）、`docs/PAPER_ALIGNMENT_REVIEW_20261029.md`（§4.8 行更新）、
  README §12、本文件 §11。

### 11.2 新增测试（34 项）

* `tests/test_standardization_stats.py`（16）：手算 population 均值/方差、mask/NaN 排除、逐通道计数、
  F,V,P 池化、零方差可逆、float64 稳定、round-trip 含 eps、NormStats 适配一致性、维度/provenance/
  fingerprint 拒绝、拒覆盖、manifest 拒 val/test/跨 split、N(0,I) 均值尺度（固定 seed/tol）、配置校验、
  `current_dino_noise` 标准化基座。
* `tests/test_standardized_pipeline.py`（12）：6 种 stage 训练步、K=2 全展开 head/projector/backbone
  梯度有限非零、`future_projector` 直接吃 tilde Z（pin 目标逐值比对）、无 double-normalization 契约、
  checkpoint 自包含（外部 stats 移走）/legacy↔new 拒绝/统计不一致拒绝/bi→joint、raw 动作反归一化含 eps、
  predict 输出契约与确定性、action_only 不读未来、`current_dino_noise` 顶层可达、
  真实小 HF Qwen3-VL + **真实 DINOv3** 二阶反向（预训练 4B 未测）、legacy predict 输出 schema 逐键不变守护、
  真实 dataset 输出 tilde A*/tilde Z* + 空间标记（逐值对照）。
* `tests/test_stats_cli.py`（6）：拒 val/test、拒覆盖、synthetic 标记、**真实 CLI 实跑**（合成 LeRobot
  on-disk 数据 + 真实冻结 DINOv3，动作统计与手算逐值核对、未来计数核对、--no-future、manifest 守卫）、
  几何不匹配/缺文件拒绝。

### 11.3 已运行证据

* `logs/standardized_20261029_*`（分轮测试 JSON/日志，不覆盖旧证据）。
* tiny 标准化短训练/恢复/阶段切换/评测/服务装配：`logs/runs/standardized_tiny*`、
  `logs/standardized_20261029_eval_tiny.json`。
* 统计 CLI 实跑日志：`logs/standardized_20261029_cli_synthetic_tiny.log` 及测试内 subprocess 记录。

### 11.4 未验证（不冒充）

* 预训练 Qwen3-VL-4B 未加载验证（GPU 约束）；真实 RoboCasa365 全量统计未拟合（数据不在本机）；
  GPU/长训练未跑。式(40) 完整残差统计变体、式(41) learned prior 未实现（非主方法）。


## 2026-10-29 超算真实数据/权重准备

- 用户授权EnergyWAM专属上传、下载、环境准备和短训练；保留其他项目暂停标记。
- 真实OpenDrawer官方包完整部署并核验：514 episodes、137431 frames、1542 videos、514 parquet。
- Qwen3-VL-4B两个完整分片、DINOv3及全部processor/tokenizer/config与本地原件逐文件SHA256一致。
- 新增hpc配置、episode划分、训练/恢复/离线评估runner、单步GPU预检、6小时互斥等待提交和持续监控工具。真实相机短名→原始特征名映射及入口数据路径检查已修复。
- 主Agent独立复跑：166/166 CPU测试；HPC配置/划分检查49/49；自动提交额度判断检查5/5。日志：logs/hpc_training_20261029_codex_tests.log、logs/hpc_training_20261029_hpc_tests.log。
- Slurm拒绝首次提交AssocMaxSubmitJobLimit；账号MaxSubmitJobs=0。真实GPU训练尚未运行，不报告训练耗时、显存、收敛、checkpoint恢复或闭环成功。自动等待仅在限额恢复后尝试1GPU/15min预检，成功后1GPU/30min短训，最多2作业，不修改平台限额，不静默改LoRA/冻结语言/截断展开。


## 2026-10-30 全量50任务/仿真资产下载

用户授权全量target/human 50任务及闭环资产。官方50任务（18 atomic+32 composite）与6类资产清单已固定到RoboCasa v1.0.1 commit dc73a8ce18e4b8947e8df2ae2958245ca823f3b4，56包72544565510 bytes，预计解压85621064844 bytes；ZIP目录/文件数/解压大小已逐包核对。独立32个下载/安全/续传/配额回归测试通过。

超算后台pipeline PID403805、worker PID403919，使用专属Box转发/持久SSH，2路并发；OpenDrawer实际核验复用，新OpenStandMixerHead从下载→核验→安装已完成，其余仍在队列。Robosuite按官方master固定5ce6643f3092639d08f7b0f90ed1c6a84f50552c，机器人XML/mesh独立上传和验证。详见coordination/full_download_20261030/CODEX_REVIEW.md与hpc/docs/FULL_ROBOCASA365_DOWNLOAD.md。

全量完成须进度tasks_ready=50、assets_ready=6、robosuite_assets_ready=true、stage=complete；目前尚未全部完成。没有启动新的GPU训练或闭环评估，没有修改其他项目。账号真实存储quota仍未知，200GiB是此下载流程自设保护上限；EDQUOT/ENOSPC会暂停并保留断点。


## 2026-10-31：Qwen embedding 与单次语言前向边界

按最新要求明确：条件编码只过 Qwen ViT/visual merger 与文本 embedding，拼接完整 C/Future/Action/Energy 序列后，每次能量计算只过一次语言 Transformer。审查确认原真实路径已符合；本次提取 `_embed_vision_language`、补充接口/README 说明和实际层调用次数回归测试，未改变数学路径或权重键名。修正测试夹具重建 core 后 stage_runner 仍引用旧 core 的问题。专项 3/3、Qwen 结构 3/3、梯度 5/5、真实路径集成 15/15，共 26/26 通过；另验证重构前后 embedding/energy 逐值一致。均为 CPU/小尺寸随机 Qwen 结构验证，不代表预训练 4B GPU 训练验证。详细记录：`coordination/qwen_single_pass_20261031/CODEX_REVIEW.md`。

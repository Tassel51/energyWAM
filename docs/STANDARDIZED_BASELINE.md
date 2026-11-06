# 标准化基线（论文 §4.8）实现说明

> 主Agent最终独立验收：165/165测试通过，CPU全链路6/6通过。统计仅train；训练严格只用bundle train列表；显式val/test还会与checkpoint train列表核对重叠，无法核验时不宣称heldout。smoke默认新建时间戳目录保留旧结果。证据见coordination/standardized_baseline_20261029/CODEX_FINAL_ACCEPTANCE.md。


**状态**：用户已于 2026-10-29 批准新版论文 §4.8 作为正式基线，已实现进完整训练 / 推理 / 恢复 / 服务 / 配置 / 统计管线。
论文稿 `energywam/LTE_WAM_Paper_Draft_CN.md` **只读未改动**。

正式配置（`configs/bi.yaml`、`joint`、`direct_joint`、`action_only`、`action_given_future`、
`future_given_action`、`real_smoke`、`eval`）默认 `data.normalization_profile=paper_standardized_v1`；
改动前的原配置逐字保留在 `configs/legacy/`。缺 `normalization_profile` 字段的旧 YAML / 旧 checkpoint
一律按 `legacy` 解释，语义与旧版本完全一致。

---

## 1. 坐标定义（工程实现）

对训练集统计 \(\mu,\sigma\)（**population** 标准差，除以 count 而非 count-1）：

```
tilde_x = (x - mean) / (std + epsilon)          # paper Eq. 37 / 39
x       = tilde_x * (std + epsilon) + mean      # 精确逆变换（epsilon 不丢）
```

* `epsilon = 1e-6`（`data.standardization_epsilon`，写入统计 bundle 与 checkpoint，逐处显式记录）。
* 统计与累加用 **float64**（Welford/Chan 流式并行方差）；候选 / 损失仍 float32。
* **零方差维度**：`std = 0` → 除数 = epsilon，有限且可逆（常数列映射为常数，不会被默默当 raw 列）。
* **mask 先屏蔽再计算**：padding / NaN / 无效元素不进入统计、不进入损失；有效元素逐维（action）/ 逐通道（future）计数。

### 计数口径（写入 bundle `provenance.element_counting`）

* **action**（每动作维 `Da`）：对每个合法训练窗口（与 `EnergyWAMEpisodeDataset` 同一 episode-safe 计划，
  `require_full_horizon` 一致）的 **有效元素**（`action_mask=1` 且有限）计数。一个被 k 个窗口覆盖的原始帧
  计 k 次——这正是训练损失看到的元素分布。
* **future**（每 DINO 通道 384）：对每个训练窗口的合法未来 **时间/视角/patch**（F,V,P）池化计数
  （`future_mask=1` 且有限），不逐 patch、不逐轨迹重新归一化。

### 变量空间

| 量 | 空间 | 说明 |
|---|---|---|
| `A* / A_K / tilde A_0` | 标准化（zscore） | 动作监督、动作候选、动作初始化（`N(0, I)`） |
| `Z* / Z_K / tilde Z_0` | 标准化（zscore） | future 监督、future 候选、future 初始化（`N(0, I)`） |
| `P_cur`（当前 DINO 条件） | **raw DINO** | 条件表征保持原始 DINO，不变 |
| state | sincos | 不变 |
| `predict` 的 `normalized_actions` | 模型空间（新版=zscore） | server 用同一 bundle 反归一化为物理动作 |
| `predict` 的 `future_latent` | **raw DINO**（默认语义不变） | paper profile 同时给 `normalized_future_latent` 并标注 space（legacy profile 不新增键，输出 schema 与旧版逐键一致） |

`future_projector` 直接接收 `tilde Z`（优化变量就是标准化坐标，不是在 raw 空间优化再在损失上除 σ）；
solver / losses / core 与 legacy profile 共用同一套实现。

### 初始化（§4.8）

* 基础 prior：`tilde A_0 ~ N(0, I)`、`tilde Z_0 ~ N(0, I)`（`init_action_std=1.0`、`init_future_std=1.0`）。
  配置校验强制此基线；若要按 §8 做同坐标先验消融，显式设 `init.allow_prior_ablation: true`（非 (0,1) 会被拒绝）。
* 可选 `initialization_future: current_dino_noise`：**raw** 当前 DINO 特征先用 **Future 统计** 变换到同一
  标准化空间，再扩展到 F 个时间槽并加配置噪声（`init_future_noise_std`）。不使用任何未来真值。
  **注意**：这不是论文式 (40) 的完整变体（未估计残差通道统计 `sigma_deltaZ`），只是固定尺度版本，已在代码与本文件记录。
* 主方法不包含 learned prior / rank loss / latent compression（式 41 为独立可选扩展，未实现）。

---

## 2. 统计 bundle 格式（`energywam_norm_bundle_v1`）

`scripts/fit_normalization_stats.py` 输出一个自描述 JSON（版本化、带内容指纹）：

```json
{
  "schema_version": "energywam_norm_bundle_v1",
  "profile": "paper_standardized_v1",
  "coordinate": "(x - mean) / (std + epsilon)",
  "inverse_coordinate": "x = normalized * (std + epsilon) + mean",
  "std_kind": "population",
  "epsilon": 1e-6,
  "action": {"mean": [...Da], "std": [...Da], "count": [...Da], "epsilon": 1e-6, ...},
  "future": {"mean": [...384], "std": [...384], "count": [...384], "epsilon": 1e-6, ...},
  "provenance": {
    "split": "train",
    "dataset_identity": "<name>/<robot>",
    "dataset_root": "...",
    "train_episodes": {"episode_indices": [...], "manifest_sha256": "...", "manifest_path": "...", "num_episodes": N},
    "camera_order": ["robot0_agentview_left"],
    "cameras_used": ["robot0_agentview_left"],
    "camera_index": 0,
    "num_cameras": 1,
    "action_horizon": 16,
    "future_offsets": [5, 11, 16],
    "num_future_frames": 3,
    "dino_preprocess_version": "dino_v3_vits16_224_center_norm_v1",
    "dino": {"model_path": "...", "config_hash": "...", "hidden_dim": 384, "num_patches": 196,
             "num_special_tokens": 5, "image_size": 224, "patch_size": 16},
    "dims": {"action_dim": 12, "future_dim": 384},
    "require_full_horizon": true,
    "element_counting": {"action": "...", "future": "..."},
    "coverage": {"kind": "full_train_split | subset | synthetic_train_split",
                 "num_windows": N, "max_samples": ..., "action_valid_elements": ..., "future_valid_elements": ...},
    "fitted_at": "..."
  },
  "fingerprint": "<sha256 of the canonical payload>"
}
```

* `action_only` / `--no-future` 拟合的 bundle `future: null`（只训练动作，不强制读未来图像）；需要 future
  统计的配置会明确拒绝它。
* `coverage.kind`：`subset`（`--max-samples`）与 `synthetic_train_split`（合成数据）**绝不冒充**完整训练集统计。
* `fingerprint` 覆盖统计与全部 provenance；resume / bi→joint / serve 用它校验同一坐标同一统计。
* **必要统计缺失 / 维度不符 / 相机顺序 / horizon / offsets / 预处理版本 / DINO 标识 / epsilon 不匹配 → 明确报错**，
  绝不回退到 raw、随机统计或 val/test 估计。

### 训练 episode manifest 格式（`energywam_episode_manifest_v1`）

```json
{
  "schema_version": "energywam_episode_manifest_v1",
  "split": "train",
  "dataset_identity": "robocasa365_open_drawer_target_human/PandaOmron",
  "dataset_root": "playground/Datasets/robocasa365/robocasa365_open_drawer_target_human",
  "episode_indices": [0, 1, 2, ...],
  "notes": "..."
}
```

校验：`split` 必须是 `train`（val/test 一概拒绝）；episode 必须存在于数据集；`dataset_identity` 与 reader 一致；
与 `--val-manifest` / `--test-manifest` 的 episode 交集为空（防跨 split 泄漏）。manifest 内容哈希写入 bundle。

> 说明：本实现**不复用** `meta/stats.json` 的动作统计（除非 legacy profile）——其 split 来源不明，
> 不默认相信；所有 paper 统计都从 manifest 指定的训练 episode 现场流式拟合。

---

## 3. 常用命令

```bash
# 环境（本机）
PY=/data3/xyx_zhuzicheng/miniconda3/envs/LDA/bin/python
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2   # CPU 验证时

# 1) 拟合训练集统计（显式 train manifest；只读；默认拒绝覆盖）
$PY scripts/fit_normalization_stats.py --config configs/bi.yaml \
    --split train \
    --episodes-manifest manifests/train_episodes.json \
    --val-manifest   manifests/val_episodes.json \
    --out stats/robocasa365_open_drawer_target_human_paper_standardized_v1.json

#    smoke 子集（bundle 标记 coverage=subset，不可当正式统计）：
#      加 --max-samples 64
#    action_only（不读未来图像）：
#      加 --no-future
#    tiny 合成 bundle（明确 synthetic）：
$PY scripts/fit_normalization_stats.py --config configs/tiny_paper_standardized.yaml \
    --split train --synthetic-tiny --out configs/tiny_paper_standardized.synthetic_bundle.json --overwrite

# 2) 训练（bi / joint / direct_joint / action_only / 单方向）
$PY scripts/train.py --config configs/bi.yaml
$PY scripts/train.py --config configs/action_only.yaml     # 只用 action 统计

# 3) 同 stage 恢复 / 阶段切换（bi -> joint）
$PY scripts/train.py --config configs/bi.yaml    --resume   logs/runs/energywam_bi_standardized_v1/ckpt.pt
$PY scripts/train.py --config configs/joint.yaml --init-from logs/runs/energywam_bi_standardized_v1/ckpt.pt \
    --stage-switch-policy reset

# 4) 离线评测（显式子集；不是闭环成功率）
$PY scripts/evaluate.py --config configs/eval.yaml \
    --checkpoint logs/runs/energywam_joint_standardized_v1/ckpt.pt --max-samples 32
#    显式留出集选择（val/test manifest）；无该参数时报告写明"未保证heldout"
#      加 --eval-episodes-manifest manifests/val_episodes.json

# 5) 服务（默认只绑 127.0.0.1；动作按 checkpoint 内嵌 bundle 反归一化为物理单位）
$PY scripts/serve.py --config configs/joint.yaml \
    --checkpoint logs/runs/energywam_joint_standardized_v1/ckpt.pt --port 10095

# 6) CPU 全链路 smoke（合成 tiny：fit→train→resume→switch→eval→serve，含外部 stats 移走自包含校验）
bash scripts/smoke_standardized_cpu.sh

# 测试
$PY tests/run_all.py                          # 全套
$PY tests/run_all.py test_standardization_stats
$PY tests/run_all.py test_standardized_pipeline
$PY tests/run_all.py test_stats_cli
```

---

## 4. checkpoint / 恢复 / 服务语义

* paper profile：统计作为 **persistent buffers**（`norm_action_mean/denom/count`、`norm_future_*`、`norm_epsilon`）
  + **完整 bundle payload**（`meta.normalization_bundle`，含 provenance 与 fingerprint）双写进 checkpoint；
  checkpoint **自包含**——原外部 stats 文件被移走后仍可推理 / 恢复 / 服务。
* legacy profile 不新增任何强制 buffer：`state_dict` keys 与旧版严格一致，旧 checkpoint 严格加载不变。
* `restore()` 在 `load_state_dict(strict=True)` **之前** 校验坐标契约：
  * legacy checkpoint ↔ paper 模型（或反向）→ 明确报错（不做任何自动迁移）；
  * bundle fingerprint 不一致（不同拟合）→ 明确报错；
  * 注册 buffers 与 bundle 不自洽 → 明确报错。
* `restore()` 还逐值校验 payload `model` 中的 `norm_*` 缓冲与内嵌 bundle 一致（wrapper 前缀按后缀定位）；
  resume / bi→joint 要求同一坐标同一统计；训练 loader 用**已恢复的** bundle，不会被磁盘上其他 stats 覆盖
  （若配置路径上存在**不同**的 bundle → 拒绝；路径失效 → 忽略，以内嵌 bundle 为准）。
* 推理 / 评测 / serve 优先用 checkpoint 内嵌统计；外部给定不一致 → 拒绝。
* raw 动作 = `normalized * (std + epsilon) + mean`（server `actions` 字段，`action_units=raw_dataset_units`）。

### 手工输入坐标契约（防 double normalization）

* 手工 `future_target` / `future_latent` **必须**带 `future_target_space: raw_dino | standardized`；
  paper profile 缺标记明确报错（拒绝猜测）；`raw_dino` 会用内嵌 future 统计标准化一次。
* 手工 `action_target` 可选带 `action_target_space: model_space | standardized`（缺省 = 已在模型空间，即示例契约）。
* legacy profile 保持 raw DINO 语义，并拒绝 `future_target_space='standardized'`。
* 诊断（`future_latent_metrics`）必须显式 `space=`（raw_dino 或 normalized_Z），cosine 空间不混写。

---

## 5. 旧兼容范围

| 对象 | 语义 |
|---|---|
| 旧 YAML（缺 `normalization_profile`） | `legacy`：min_max 动作 + raw DINO future + `N(0, 0.1²)` 初始化，逐字不变 |
| `configs/legacy/*.yaml` | 改动前正式配置的逐字副本 |
| 旧 tiny 配置 / 旧 115 项测试 | 科学条件不变（不改断言迁就新实现） |
| 旧 checkpoint（无 `normalization_profile` 字段） | 按 legacy 严格加载；与 paper 模型互载 → 明确报错 |
| `meta/stats.json`（StarVLA min_max 契约） | legacy profile 继续使用；paper profile 不使用 |

---

## 6. 资源限制与未验证项（诚实记录）

* **GPU / 全量训练未跑**：本机无 GPU 配额，全部验证在 CPU（tiny 短训练 + 小随机 HF Qwen3-VL + 真实 DINOv3 结构/二阶反向）。
* **预训练 Qwen3-VL-4B 未加载验证**（路径存在但按约束不加载）——4B 路径标记 **未测**。
* **真实 RoboCasa365 不在本机**：未做真实全量统计拟合；`stats/robocasa365_*.json` 需在有数据的机器上按 §3 命令
  生成。CLI 的真实路径已在一个**合成 LeRobot 数据集（真实视频 + 真实冻结 DINOv3）**上实跑并用手算统计核对，
  该证据明确标注 synthetic。
* 式 (40) 完整变体（估计 `sigma_deltaZ`）未实现；`current_dino_noise` 为固定尺度版本（见 §1）。
* 式 (41) learned prior、rank loss、latent compression 未实现（非主方法）。
* 已知记录但未改：`episode_dataset.build_sample` 未来帧取 `cams[v]`（非 `sel[v]`），`camera_index>0` 多相机
  时与当前帧选择不一致；`num_cameras=1`（当前全部配置）不受影响。统计 CLI 使用 `sel[v]`（配置相机顺序）。

---

## 7. 独立复核修复（2026-10-29 review fix）

主Agent独立复跑 149 项通过后的关键审查发现以下缺陷，已在本节记录的最小修复中闭合
（回归测试 `tests/test_standardized_contract_regressions.py`，修复前 11/11 失败、修复后全过，
日志 `logs/standardized_20261029_reviewfix_*`）。论文稿、旧实验、旧测试断言、科学坐标均未改动。

### 7.1 训练 split 接线（P0-1）

* **训练严格使用 bundle 记录的 `train_episodes.episode_indices` 索引**：
  `build_dataloader`（paper profile）把该列表传入 `EnergyWAMEpisodeDataset`，并校验
  `reader.dataset_identity == provenance.dataset_identity`；缺 episode / 空列表 / 重复 /
  非法索引 / 未记录列表一律**明确拒绝**，不存在"放开全部 episode"的 fallback。
  一个 bundle 若未记录训练 episode，任何真实训练都拒绝启动。
* **训练与评估分开**：eval（`build_eval_examples` / `scripts/evaluate.py`）**不**被限制到
  训练 episode；`--eval-episodes-manifest` 可显式选择留出集（val/test manifest，经
  `load_episode_manifest` 校验）。默认无显式选择时报告 `heldout.guaranteed_heldout=false`
  并写明"**未保证heldout**"，train MSE 不得冒称保留集结果。
* legacy profile 保持历史行为（全 episode 索引、旧统计来源）。

### 7.2 相机选择一致性（P0-2）

* `build_sample` 的当前帧与未来帧统一取 `sel[v]`（原未来帧误用 `cams[v]`，`camera_index>0`
  即使单相机也错位）。
* 统计拟合与训练使用同一选择：`camera_order[future_camera_index : +num_cameras]`
  （`data.future_camera_index` 现在真正接线到 dataset 与 stats fit 两侧）。
* bundle provenance 记录 `cameras_used` + `camera_index` + `num_cameras`；`validate_geometry`
  校验三者与 run 的实际选择一致（仅 `camera_order` 一致不够）。CLI 与训练读同一路相机已由
  数值/读取 key 测试核对（含单相机 `camera_index=1` 与多相机用例）。

### 7.3 action-only bundle（P0-3）

* `--no-future` 产物无 future 统计；当**不需要** future 标准化时跳过 Future 统计的 DINO
  身份/配置校验（此前会拿空 `model_path` 与真实 DINO 路径比较而误拒真实 action_only）。
  当前条件 DINO（P_cur）仍按正常方式加载/冻结，不受影响。带 future 统计的 bundle 照常校验。

### 7.4 来源保护（P0-4）

* **synthetic 统计默认拒绝进入真实（qwen3vl）路径**：`provenance.synthetic=true` 的 bundle
  在真实 backend 被拒；tiny 允许（显式测试模式）；注入非生产 backbone 的验证运行须显式设
  `data.allow_synthetic_stats: true`（专门开关，真实 4B 路径永不设置）。
* **DINO 配置身份整体校验**：除 `model_path` 与预处理版本外，`hidden_dim / num_patches /
  num_special_tokens / image_size / patch_size` 全部比对——同一路径下配置变更同样拒绝。
* **fingerprint 必需**：`from_payload`/`load_bundle` 拒绝缺 fingerprint 的 v1 payload
  （直接构造 `NormalizationBundle` 对象的测试 API 不受影响）。
* 训练数据身份在 §7.1 校验；无数据集的推理只用 self-contained bundle 的已验证身份。

### 7.5 checkpoint payload 缓冲校验（P1）

* `verify_normalization_contract` 在 `load_state_dict` **之前**校验 payload `model` 里的
  `norm_action_mean/denom/count`、`norm_future_*`、`norm_epsilon` 与内嵌 bundle 逐值一致
  （支持 wrapper 前缀，如 StarVLA adapter 的 `model.` 前缀，按后缀定位）；仅篡改 payload
  buffer 而不动 meta fingerprint 会被拒绝；正常自包含恢复不受影响；legacy checkpoint 行为不变。

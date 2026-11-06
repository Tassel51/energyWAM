# 论文稿—实现逐项核对（2026-10-29）

> **最终独立核验**：115/115测试通过。主Agent进一步修正AUC定义、诊断的自动微分上下文与空mask报告；式40目前仅实现固定尺度当前特征噪声先验，尚未估计残差通道统计。
> **2026-10-29 更新**：用户已批准新版 §4.8 为正式基线（动作标准化 + Future 逐通道标准化 + N(0,I) 初始化），
> 并已实现进完整训练/推理/恢复/服务/配置/统计管线；正式 YAML 切换为 `paper_standardized_v1`，
> 旧语义保留在 `configs/legacy/` 与缺省 `legacy` profile。实现范围、命令与未验证项见
> `docs/STANDARDIZED_BASELINE.md` 与 `coordination/standardized_baseline_20261029/REPORT.md`。
> 旧 115 项测试的科学条件未改动。最新细节与证据见 `coordination/paper_audit_20261029/CODEX_FINAL_REVIEW.md`。


对照稿件：`energywam/LTE_WAM_Paper_Draft_CN.md`（2026-10 新版，1051 行，**只读未改动**）。
实现：`energywam/`（tiny + 真实 Qwen3-VL 路径）。证据测试：`tests/`，日志 `logs/paper_audit_20261029_*`。

结论标记：**符合** / **确定错误(已修)** / **缺(核心)** / **缺(实验/可选)** / **待用户决定** / **待数据验证**。

> 声明：预训练 4B 权重未加载验证（本机无 GPU 配额）；RoboCasa365 数据不在本机。
> 现有验证 = 95 项 CPU 测试 + 真实小随机 Qwen3-VL 结构/二阶反向 + 真实 DINOv3 权重。
> 本审查不声称全符合；未获得的实验结果不填补。

---

## 1. 映射表

### §3 问题定义

| 稿件节/公式 | 实现（文件::函数） | 结论 | 证据 |
|---|---|---|---|
| 式(4) 动作块 [H,Da] | `config.py::DataConfig`（action_horizon/action_dim 从任务配置读取）；`episode_dataset.py::build_sample` | 符合 | test_review_regressions（偏移/掩码）；Da 不硬编码为 7 |
| 式(5) τ_i=round(iH/3) | `config.py::resolve_future_offsets` → H=16 得 [5,11,16] | 符合 | test_shapes / config 校验（正、严格递增、≤H） |
| 式(6)(7) Z*=[3,196,384] | `dino_v3.py::extract_patches`（去 CLS+4 register，仅 patch）；`energy_dataset.py::_future_dino_target` → [F,V,P,D] | 符合 | test_real_integration（真实 DINO 形状 196×384） |
| 式(8) 样本 (I,ℓ,q,A*,Z*) | `energy_dataset.py::EnergyWAMEpisodeDataset.__getitem__` | 符合（真实数据待验证） | fake-reader 测试；RoboCasa365 不在本机 |
| 式(9) E(C,A,Z)→R | `energy_core.py::EnergyWAMCore.energy`（每样本标量，无跨样本归一） | 符合 | test_shapes |
| 式(10) EBM 解释 | 无任何归一化密度/概率训练目标 | 符合（不声称已校准） | 代码中无 Z 估计/NLL |

### §4.2–4.4 条件与能量 Transformer

| 稿件节/公式 | 实现 | 结论 | 证据 |
|---|---|---|---|
| 式(11) Qwen 原生 VL 路径 | `qwen3vl_adapter.py::encode_condition`（`get_input_embeddings` + `get_image_features` masked_scatter + DeepStack + `get_rope_index` mRoPE + padding mask） | 符合 | test_qwen_structure（真实 HF Qwen3-VL 小配置走原生路径） |
| 式(12)(13) 当前 DINO→P_cur | `energy_core.py::build_current_dino_tokens`（`current_dino_projector`） | 符合 | test_qwen_structure / test_real_integration |
| 式(14)(15) 状态→P_state；C=[VL,T_cur,T_state] | `build_state_tokens`（N_q=`num_state_tokens`）；`run_sequence` 拼接顺序 [native VL, dino_cur, state] | 符合 | test_shapes |
| 式(16) Future tokens+时间嵌入 e_τi | `energy_core.py::build_future_tokens` + `encodings.py::future`（时间嵌入用**实际 offset** τ_i） | 符合 | test_review_regressions（非零起点仍用相对 offset） |
| 4.3.1 时间嵌入空间（384 或 d 二选一） | **投影后**加在 d 维（`encodings.py` 模块注释明确声明） | 符合（稿内两种合法选择之一） | encodings.py 文档串 |
| 式(17) Action tokens+步位置嵌入 | `build_action_tokens` + `encodings.py::action`（action_step 正弦编码+类型嵌入） | 符合 | test_shapes |
| 式(18) X_0=[C,Z,A,e_E] | `energy_core.py::energy` → groups [future, action, energy] 接在 C 后；因果注意力 | 符合 | test_qwen_structure |
| 式(19)(20) h_E=X_L[-1]，E=MLP_E | `qwen3vl_adapter.py::run_sequence`（e_index=最后一位）+ `energy_core.py::EnergyHead` | 符合 | test_qwen_structure（e_index 为序列末尾） |
| 4.4 位置编码/mRoPE 适配 | `_appended_position_ids`：追加块起点=各样本原生有效位置 max+1，三轴同号；另有 future-time/spatial/camera/type/action-step 独立语义编码；DeepStack visual_pos_masks 补零对齐 | 符合（工程适配，已测试） | test_qwen_structure::qwen_appended_positions_strictly_after_native |
| 4.4 结构消融（E-Token vs MeanPool vs 末 Action token；[C,Z,A,E] vs [C,A,Z,E]） | 未做 | **缺(实验/可选)** | — |
| 候选在语言层前拼接（不许后接 hidden） | `run_sequence` 在 `language_model(...)` 之前拼 inputs_embeds | 符合 | qwen3vl_adapter 文档串+测试 |
| 无不必要 vocab logits | 能量路径只调 `language_model`，从不调 `lm_head` | 符合 | 代码路径审查 |
| lm_head 闲置参数 | **确定错误(已修 F1)**：untied 闲置 lm_head 原被标可训练；tied（真实 4B `tie_word_embeddings: true`）时其 weight 即共享 embedding，冻结会误伤——现仅冻结 untied 情形 | 已修 | test_paper_regressions（tied/untied 双向） |
| 冻结 DINO / 语言可训 | `dino_v3.py::freeze`（requires_grad_(False)+eval）；`_apply_freezing`（tune_language_model 配置） | 符合 | test_real_integration |

### §4.5–4.7 求解器与两阶段训练

| 稿件节/公式 | 实现 | 结论 | 证据 |
|---|---|---|---|
| 式(21) A0~p_A, Z0~p_Z | `energy/init.py::init_candidates`（训练用全局 RNG，推理用局部种子生成器） | 符合（分布选择见 §4.8） | test_no_leakage（同种子确定性） |
| 式(22)–(24) 同点 E.sum() 双梯度、同时更新、独立 α_A/α_Z | `solver.py::_step`：一次 E，`autograd.grad(E.sum(), [A,Z])`，simultaneous 更新；α_A≠α_Z 可配 | 符合 | test_solver_analytic / test_gradients |
| 式(25) G^K 展开算子 | `solver.py::refine`（full unroll，K 步） | 符合 | test_gradients |
| 4.5 梯度裁剪/范围限制（可选稳定化） | `solver_utils.py`（默认全关） | 符合 | config 默认 None |
| 式(26)–(28) Future→Action（Z* 固定，J_A=MSE） | `stages.py::_run_bi`/`_run_single(direction='A')` + `losses.py::action_endpoint_loss`（按有效元素归一 MSE） | 符合 | test_no_leakage / test_gradients |
| 式(29)–(31) Action→Future（A* 固定，J_Z=MSE） | `stages.py::_run_bi`/`_run_single(direction='Z')` + `future_endpoint_loss` | 符合 | 同上 |
| 式(32) L_bi=λ_A L_A+λ_Z L_Z | `stages.py::_run_bi`（同 batch 两方向求和；可交替调度 `bi_both_directions_each_batch=False`） | 符合 | test_overfit_tiny |
| 4.6.2 余弦消融/辅助项 | `losses.py::masked_cosine`，`future_cosine_weight` 默认 0 | 符合（稿要求“分析前非必要”） | test_numerical |
| 式(33)(34) Stage II 联合 L_joint=β_A J_A+β_Z J_Z | `stages.py::_run_joint`（β 可配） | 符合 | test_overfit_tiny / test_checkpoint_stage |
| 式(35) 对展开轨迹求导（create_graph=True 二阶） | `solver.py`（训练 create_graph=True，全链路不 detach；推理仅一阶） | 符合 | test_qwen_structure（真实 Qwen 二阶反向到参数） |
| Algorithm 1 注：两组梯度同点求出后共同更新 | `joint_update_mode='simultaneous'` 为默认；`alternating` 是显式命名的不同配置 | 符合 | test_solver_analytic |
| fixed 分支不改（不 clamp/mask 目标） | `solver.py::_step`：仅更新分支做 clamp/mask；Z*/A* detach 后固定 | 符合 | test_mask_regression / test_batch_masks |
| action_only 不需未来真值 | `energy_core.py::energy_action_only`（future 全掩码 placeholder）；`stages.py::_run_action_only` 不读 future_target | 符合 | test_no_leakage::action_only_energy_has_no_future_dependency |
| joint 不泄漏 targets | `predict` 仅构造条件；改 targets 不改输出 | 符合 | test_no_leakage / test_real_integration::real_predict_does_not_read_supervision |
| masks 严格（NaN 不泄漏、按有效元素归一） | `losses.py::masked_mse/masked_cosine`（where 先置零）；`energy_core.py::_apply_value_mask` | 符合 | test_mask_regression / test_numerical |
| 训练 RNG / 推理种子 | 训练用全局 torch RNG（checkpoint 保存/恢复）；推理用局部 `torch.Generator(cfg.trainer.seed)` | 符合 | test_checkpoint_stage / test_no_leakage |
| resume / stage-switch 语义 | `checkpoint.py::restore`（resume 严格同 stage；switch 显式 optimizer/scheduler reset\|continue 并记日志） | 符合 | test_checkpoint_stage / test_entrypoint_regressions（scheduler 恢复 LR） |
| stats 持久化 | `checkpoint.py::norm_stats_to_meta/from_meta`；`trainer.save` 写入 meta | 符合 | test_training_stats_binding / test_service_checkpoint |

### §4.8 初始化先验

> **2026-10-29 用户已批准新版 §4.8 为正式基线**（动作按训练集均值/标准差标准化、Future 按训练集逐通道统计
> 标准化、初始化 N(0,I)），并已实现进完整管线（训练/推理/恢复/服务/配置/统计 CLI）。正式 YAML 默认
> `normalization_profile=paper_standardized_v1`；旧语义由 `configs/legacy/` 与缺省 `legacy` profile 保留。
> 实现细节与命令见 `docs/STANDARDIZED_BASELINE.md`。

| 稿件节/公式 | 实现 | 结论 | 证据 |
|---|---|---|---|
| 式(36) 归一化空间 N(0,I) | `paper_standardized_v1` profile：`init_*_std=1.0`（配置校验强制 N(0,I)，`allow_prior_ablation` 显式开关才允许同坐标先验消融）；legacy profile 保留旧 N(0, 0.1²) | 符合（新基线） | test_standardization_stats::gaussian_prior_mean_and_scale / config_enforces_paper_prior_and_profile |
| 式(37)(38) 动作标准化 (x-μ)/σ 及等价性注记 | `data/standardization.py`：`(x-mean)/(std+eps)` population std、float64 流式 Welford、mask 先屏蔽；逆变换精确含 eps；统计按训练窗口有效元素、逐动作维计数；`energy_dataset` 输出 tilde A*；solver 优化 tilde A | 符合（新基线） | test_standardization_stats（手算均值/方差/round-trip/零方差/float64）；test_standardized_pipeline（projector/损失坐标） |
| 式(39) Future 通道统计归一化 | 384 通道、训练样本合法 (F,V,P) 池化、只 train split 拟合（`scripts/fit_normalization_stats.py`，manifest 拒 val/test）；future target/loss/候选均在标准化坐标 | 符合（新基线） | test_stats_cli（真实 DINO + 合成 LeRobot 实跑，手算核对）；test_standardization_stats |
| 式(40) 当前 DINO 残差初始化 | `init.py` `initialization_future='current_dino_noise'`（可选）：raw 当前特征先用 Future 统计进标准化空间再扩展加噪；**非完整式(40)**（未估计 sigma_deltaZ），已记录；**确定错误(已修 F3)**：原只能手工传 current_dino，现顶层 forward/predict 自动复用 C 的 DINO 特征 | 部分（固定尺度，非残差统计） | test_standardization_stats::current_dino_noise… / test_standardized_pipeline::current_dino_noise… |
| 式(41) 可学习先验 | 未实现 | **缺(可选扩展，按任务要求不默认加入)** | — |
| 禁止用 A* 构造推理初值 | `predict` 只读条件 | 符合 | test_no_leakage |

### §4.9 耦合性分析 / §6.4 指标

| 稿件节/公式 | 实现 | 结论 | 证据 |
|---|---|---|---|
| 式(42)(43) 可分离能量退化情形 | `tests/test_coupling.py`（仅证明交叉梯度非零） | 部分（原仅定性） | — |
| 式(44) 归一化交叉梯度敏感性 S_{A←Z} | **新增** `energy/diagnostics.py::action_gradient_future_sensitivity`（及对称 S_{Z←A}） | 已补(本次) | test_diagnostics（闭式对照；可分离能量→精确 0） |
| 条件交换后 Action 优化轨迹 | **新增** `diagnostics.py::condition_swap_action_trajectory`（复用 solver 更新规则） | 已补(本次) | test_diagnostics |
| 式(45) 正/负配对 energy 排序、AUC、间隔 | **新增** `diagnostics.py::energy_ranking_metrics`（负样本**由调用方提供并标合法性**，拒绝自动打乱；无负样本/无合法性标记即报错） | 已补(本次) | test_diagnostics（手算对照+拒绝路径） |
| 式(46) 排序正则 L_rank | 未实现 | **缺(可选消融，按任务要求不默认加入)** | — |
| §6.4.2 Future masked MSE + patch cosine | **新增** `diagnostics.py::future_latent_metrics`（强制标注度量空间 raw/normalized） | 已补(本次) | test_diagnostics（手算 MSE/余弦） |
| §6.4.1 闭环成功率 | 无 | **缺(实验)**：需 RoboCasa365 仿真 | — |
| §6.4.3 仿真状态转移确认正负配对物理含义 | 无 | **缺(实验)**：需仿真环境；诊断工具只标数学排序 | — |
| §6.4.4 效率/稳定性报告 | 训练日志有 step/forward/backward 时间、参数梯度范数、RSS（`trainer.py`）；无 GPU 显存/推理时延基准 | 部分（**缺(实验)**：需 GPU/规模） | — |

### §4.10 / Algorithm 2 推理与滚动执行

| 稿件节/公式 | 实现 | 结论 | 证据 |
|---|---|---|---|
| 式(47) Denorm(A_K) | `inference/predictor.py::EnergyWAMPredictor`（用 checkpoint meta.norm_stats 反归一化；无统计时明确标 normalized） | 符合 | test_service_checkpoint / test_entrypoint_regressions |
| Algorithm 2 `enforce_valid_controls` | **缺**：输出前无合法性处理钩子（夹爪离散化/限幅规则依控制接口而定，数据接口未定） | **缺(核心待定项)** | 见开放问题 Q5 |
| 推理不包 no_grad | `predict`：`inference_mode(False)+enable_grad`，仅 `autograd.grad` 不留参数梯度 | 符合 | test_no_leakage（no_grad/inference_mode 下均通过） |
| 式(48) 滚动重规划 | 服务/评测接口存在（`serve`/`offline_eval`）；闭环执行依赖 RoboCasa365 | **缺(实验)** | — |

### §5 实现细节

| 稿件节 | 实现 | 结论 | 证据 |
|---|---|---|---|
| §5.1 维度表 / 式(49) N_seq=N_VL+196+N_q+588+H+1 | 与实现 token 计数一致（V=1 相机；多相机按 V 缩放） | 符合 | test_shapes；4B 配置 2560/36 已从磁盘 config.json 核对（权重未加载） |
| §5.2 同 episode 严格对齐（式(50)） | `future_offsets.py::plan_future_frames`（不跨 episode、末尾掩码/丢弃、require_full_horizon） | 符合 | test_review_regressions（非零起点相对偏移） |
| §5.2 动作/相机频率对齐 | **假设 1 动作/帧、同帧率取未来帧**；无数据可实测 | **待数据验证** | episode_dataset.py 文档串 |
| §5.2 当前/未来 DINO 预处理一致 | 同一 `FrozenDINOv3.preprocess`（确定性，无随机增强） | 符合 | test_real_integration（uint8/PIL 一致） |
| §5.2 归一化统计仅训练集 | `normalization.py` 复用 StarVLA/GR00T LeRobot 契约，统计来自 meta/stats.json，不在此重拟合 | 符合（真实统计待验证） | test_entrypoint_regressions |
| §5.2 DINO GT 离线提取/缓存 | `data/cache.py`：缓存键契约（episode/frame/camera+DINO 检查点+预处理/归一版本）+ 内存参考实现；磁盘实现可扩展 | 符合（按工程需求为可扩展契约） | cache.py |
| §5.3 记录 L_A、L_Z、‖∇_A E‖、‖∇_Z E‖、E(C,A_k,Z_k) | `solver.py::SolverState.as_diagnostics`（能量轨迹/梯度范数/更新范数全套） | 符合 | 各训练测试日志 |
| §5.4 接口 encode_condition/energy/optimize_energy/forward/predict_action | 全部具备（`optimize_energy` 实名为 `solver.refine(mode=...)`）；`integration/starvla.py::EnergyWAMFrameworkAdapter` | 符合（命名差异） | test_real_integration::starvla_adapter… |
| F4 action_only 数据路径 | **确定错误(已修)**：原 action_only 仍解码未来 RGB+跑 DINO；现 `require_future_for_stage` 默认跳过 | 已修 | test_paper_regressions::action_only_dataloader_skips… |
| F2 共享 DINO 设备迁移 | **确定错误(已修)**：`encode_images` 原 `self.to(device)` 把模型共享 DINO 拉回 CPU；现不搬 module，输出落请求设备 | 已修 | test_paper_regressions（spy .to 调用） |

### §6 其余实验（全部未跑）

§6.2 数据协议固定、§6.3 基线 B1–B7、§6.5 消融 A–F、§6.6 结果表、§6.7 定性图：**缺(实验)**（数据与 GPU 均不可用；表中数值保持 TBD，不编造）。

---

## 2. 本次修复清单（最小修复，均附可复现旧错误的回归测试）

| # | 缺陷（旧行为） | 修复 | 文件/位置 | 回归测试（旧代码下 FAIL） |
|---|---|---|---|---|
| F1 | untied 闲置 `lm_head` 被标可训练（浪费参数计数/梯度簿记）；且若误冻结会伤 tied 共享 embedding | `_freeze_unused_lm_head`：仅当 lm_head 权重**不是**共享 embedding 时冻结；tied（真实 4B）保持语言模型策略 | `backbones/qwen3vl_adapter.py`（`_apply_freezing` 内） | `test_paper_regressions.lm_head_frozen_when_untied_and_unused`（旧：FAIL）；`lm_head_tied_never_freezes_shared_embedding`（守护，双向通过） |
| F2 | `FrozenDINOv3.encode_images` 调 `self.to(device)`：dataset 以 `device="cpu"` 提取未来特征时把**共享**的模型 DINO 拉回 CPU，每 batch 设备来回搬运 | 删除 module 搬移；输入搬到 body 设备计算，输出落调用方请求设备（`extract_patches` 本就返回输入设备） | `backbones/dino_v3.py::encode_images` | `encode_images_never_moves_the_shared_module`、`shared_dino_dataset_extraction_does_not_touch_model_device`（旧：FAIL，spy 记录到 `to('cpu')`） |
| F3 | `initialization_future='current_dino_noise'` 在真实顶层 forward/predict **不可达**（两者都不产生 `current_dino` 键，只有手工构造 batch 才能用），配置合法却必炸 | `_init`/`_predict_impl` 回退复用条件编码已算出的 `extra_raw['dino']`；显式 batch 值仍优先 | `training/stages.py::_current_dino`、`models/energy_wam.py::_predict_impl` | `current_dino_noise_works_in_real_forward_and_predict`（旧：FAIL “requires current_dino features”）；`explicit_current_dino_still_wins`（守护） |
| F4 | `stage=action_only` 数据路径仍解码未来 RGB 并跑冻结 DINO（该阶段无法读未来，纯浪费且多读真实未来数据） | `require_future_for_stage()`：action_only 默认 `require_future=False`，显式参数仍优先 | `data/energy_dataset.py::build_dataloader` | `action_only_dataloader_skips_future_extraction`（旧：FAIL，计数器录到未来帧解码） |

验证方式：用 `coordination/paper_audit_20261029/before_source.tar.gz` 里的旧源码回放，7 项回归测试中 5 项按预期 FAIL（另 2 项为守护用例，新旧均应 PASS）；恢复修复后 7/7 PASS。日志：`logs/paper_audit_20261029_regressions_*.log`、`logs/paper_audit_20261029_full_suite.log`。

本次新增（不动训练目标）：`energywam/energy/diagnostics.py`（式44/条件交换/式45 排序/未来 MSE+余弦，全部标注“数学依赖≠物理正确/非校准置信度”，负样本强制调用方提供并标合法性）、`tests/test_diagnostics.py`（9 项）、`tests/test_paper_regressions.py`（7 项）。

**未改**：默认归一化坐标、初始化标准差、损失尺度、任何论文方法选择、论文稿本身。

---

## 3. 剩余需问用户的问题

1. **【最重要】§4.8 归一化/初始化坐标系**：~~待用户决定~~ → **已获用户批准（2026-10-29）并已实现为正式默认**：
   动作标准化 (x-μ_A)/(σ_A+ε) + Future 逐通道标准化 + N(0,I) 初始化（式 37–39），
   统计管线（train-only manifest 流式拟合 + 版本化 bundle + checkpoint 内嵌）已落地。
   旧语义保留在 `configs/legacy/` 与 `normalization_profile='legacy'`（缺省），旧 checkpoint 语义不变。
2. **Future 归一化统计口径**：~~待决定~~ → **已按主Agent接口 B 实现**：逐通道 384 维，对训练样本合法
   未来时间/视角/patch（F,V,P）池化；只 train split 拟合并冻结到版本化 bundle（含 manifest 哈希/coverage/fingerprint），
   评估/部署绝不拟合。
3. **式(16) 时间嵌入空间**：现选“投影后 d 维加法”。是否维持？（稿内两可，须论文中写明一种。）
4. **Gripper/特殊控制维度**（§4.6.1）：是否加分类/加权损失取决于数据接口——请给出 PandaOmron 动作定义（维数、夹爪维）。
5. **`enforce_valid_controls`（Algorithm 2）**：输出合法性处理的具体规则（夹爪二值化阈值、限幅范围、速度平滑）——请提供控制接口约束，否则此项保持缺失并在论文标注。
6. **式(46) 排序正则 / 式(41) 学习先验 / latent 压缩**：确认首轮论文均不启用（本次按任务要求未加入）。
7. **诊断负样本来源**（式45/§6.4.3）：负配对须由已知状态转移判定——是否计划从 RoboCasa365 仿真状态生成并人工复核？本地无数据前诊断只能用调用方显式给定的配对。

---

## 4. 缺失实验清单（不编造结果）

**核心（论文声称所需）**
- E1 闭环 RoboCasa365 成功率（表1、§6.4.1）——需仿真环境与数据。
- E2 未来潜变量质量（表2 Future Error、§6.4.2）——指标已有，需真实留出集。
- E3 转移相容性排序/AUC + 条件交换轨迹（表2、§6.4.3）——工具已有，需真实正/负配对。
- E4 训练策略消融 A（F2A/F2J/Direct Joint/Bi/Bi→Joint）。
- E5 预训练 4B 权重的真实验证（现仅小随机 Qwen 结构验证）。

**实验/可选**
- E6 初始化消融 C（同坐标系下尺度/协方差/条件先验对比；式38 等价性不作为独立改进）。
- E7 步数消融 D（K∈{1,2,4,8}）、效率表3（时延/显存）。
- E8 结构消融 E（Energy Token vs MeanPool vs 末 Action token；[C,Z,A,E] vs [C,A,Z,E]；DINO 冻结/微调）。
- E9 可选消融 F（式46 排序正则，须与主方法分开报告）。
- E10 泛化 RQ5（未见场景/物体/任务组合）。

---

## 5. 工程默认偏差（与论文显式数值/表述不同，均不改论文）

| 项 | 论文 | 现行实现 | 性质 |
|---|---|---|---|
| 动作归一化 | 式(37) 标准化 | **正式 YAML = 标准化 zscore（已实现）**；min_max [-1,1] 保留在 `configs/legacy/` 与 legacy profile | 已按用户批准切换（2026-10-29） |
| A0/Z0 初始化尺度 | 式(36) N(0,I) | **正式 YAML = N(0,I)（已实现，配置校验强制）**；旧 N(0, 0.1²) 保留在 legacy/tiny 旧配置 | 已按用户批准切换（2026-10-29） |
| Future 特征空间 | 式(39) 通道统计归一 | **正式 YAML = 384 通道标准化（已实现）**；raw DINO 保留在 legacy profile（P_cur 仍 raw DINO） | 已按用户批准切换（2026-10-29） |
| H、Da 默认值 | 不预设 | H=16、Da=12（PandaOmron 配置默认，随任务配置覆盖） | 工程默认，可覆盖 |
| N_q 状态 token 数 | N_q 未定 | num_state_tokens=1（可配） | 工程默认 |
| `optimize_energy` 命名 | §5.4 | `solver.refine(mode=…)` | 命名差异，功能等价 |
| 余弦辅助损失 | 消融/辅助（§4.6.2） | 存在但默认 weight=0 | 一致 |
| truncated unroll / alternating / Langevin | 非方法内容 | 均为显式命名、默认关闭 | 一致（deviation 已标注） |
| `enforce_valid_controls` | Algorithm 2 | 缺（控制接口未定） | 缺失待定（Q5） |

---

## 6. 测试与日志

- 基线（改动前）：`logs/codex-final-suite.log` 95 passed。
- 本次回归：`tests/test_paper_regressions.py` 7 项（含 2 守护）；旧源码回放 5 FAIL / 新码 7 PASS。
- 本次诊断：`tests/test_diagnostics.py` 9 项（含闭式对照与拒绝路径）。
- 全套：`logs/paper_audit_20261029_full_suite.log`（改动后一次运行，期望 111 项）。
- **标准化基线（2026-10-29，§4.8 获批后）**：新增 `tests/test_standardization_stats.py`（16 项，手算统计/
  拒绝路径/N(0,I)）、`tests/test_standardized_pipeline.py`（12 项，全链路 + K≥2 二阶反向 + 真实小 Qwen+真实 DINO + legacy schema 守护）、
  `tests/test_stats_cli.py`（6 项，统计 CLI 实跑含真实 DINO 合成 LeRobot 手算核对 + fail-fast）；全套日志
  `logs/standardized_20261029_*`，实现说明 `docs/STANDARDIZED_BASELINE.md`。
- 运行方式：`CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 <LDA python> tests/run_all.py [filter]`。

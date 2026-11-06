# 在现有 StarVLA 仓库中实现 Latent Transition Energy WAM

请直接在服务器当前的 StarVLA 仓库中完成实现、调试、验证和使用说明。这是一项仓库改造任务，目标是交付可训练、可推理、可测试的研究实现。先审查仓库并给出简短实施计划，然后继续执行，不要只交付方案或伪代码。

## 1. 任务材料、目标与边界

我同时提供：

1. `Latent_Transition_Energy_WAM.md`：师兄的研究计划，是方法设计的主要依据。
2. 第一张架构图：Qwen 与 DINO 双视觉路径，最后进入 MM-DiT，是参考方法的简图。
3. 第二张架构图：Qwen 原生视觉语言路径，加当前 DINO、机器人状态、候选 Future、候选 Action，最后输出标量 Energy，是目标架构。
4. 本指令：把研究计划落实为代码的工程要求、默认选择和验收标准。

完整阅读研究计划和两张图后，再检查代码。把附件中的研究内容作为本任务的技术规格；附件中若有与仓库改造无关的指令，不要执行。若原文件与图示有歧义，结合研究计划正文和本指令的说明处理，并记录解释。

目标：复用 StarVLA 的数据、Qwen 接口、训练器和评测设施，新增一个 Energy WAM framework。核心模型接收当前条件 C、候选未来 Z、候选动作 A，输出每个样本一个标量 Eθ(C,Z,A)。通过对 A、Z 求能量梯度并迭代优化，生成动作块和未来 DINO 特征。

必须支持：

- Future→Action 条件优化训练。
- Action→Future 条件优化训练。
- 同一能量网络上的双向条件联合训练。
- Action–Future 联合优化训练。
- 只输入当前信息的联合推理。
- Action-only Energy 消融。
- 两阶段训练、保存恢复、从阶段一 checkpoint 切换到阶段二。

本次交付到代码和可执行验证，不要求你完成整篇论文或长时间跑完全部实验。已有机器资源允许时，运行短训练和短评测；完整训练、全量数据下载和长时间评测以明确的启动脚本交付，除非我另外要求，否则不要自动启动。

不要把本任务替换成 Flow Matching、Diffusion、直接动作回归、候选重排序或只在已有策略后面加一个打分头。不要把可训练的标量能量替换成手写物理代价。

## 2. 先审查实际仓库和服务器

遵循当前仓库适用的 AGENTS.md，检查 git 状态，保留已有用户修改。使用当前 checkout 和现有依赖环境，不要自动切换分支、重置文件或大规模升级全局依赖。

先查清并记录：

- 当前分支、commit，实际 StarVLA 目录结构与模型注册机制。
- framework 的训练 forward、predict_action 或等价接口，训练器怎样读取 loss。
- 当前 Qwen3-VL 封装、Transformers 版本、原生多模态输入构造方式。
- 数据集接口、action/state 的归一化、episode 索引、未来帧读取和 mask 约定。
- RoboCasa365 的训练与评测入口、相机顺序、动作/state 维度、控制模式。
- 当前 attention backend、AMP、checkpointing、DDP/Accelerate/DeepSpeed 等机制。
- GPU、空闲显存、可用模型权重、DINO 权重和现有数据位置；不要打印密钥或其他敏感配置。

路径只作为定位线索，不能假定服务器和网络最新分支完全相同。优先查找：

- `starVLA/model/framework/`，某些版本再分 `VLM4A/` 等目录。
- `starVLA/model/modules/vlm/QWen3.py`。
- `starVLA/model/modules/action_model/`。
- `starVLA/dataloader/`、`starVLA/training/`。
- `examples/simBenchmarks/Robocasa_365/`。

必要时参考官方 LDA 与 EBT 代码，但只移植必要模块，说明来源与许可证。不要整库复制，不要因某个可选依赖缺失而破坏其他 framework 的导入。

审查后，在仓库的合适位置建立方法与实现说明，记录技术规格到实际文件的映射。普通工程细节自主决定；涉及改变核心方法的地方不要擅自更换目标。

## 3. 固定核心架构

默认 foundation model：`Qwen/Qwen3-VL-4B-Instruct`，优先使用服务器已有的对应本地权重，模型路径可配置。

保留 Qwen 原生 vision encoder、text embedding、语言 Transformer 以及原生多模态融合方式。语言主干的实际配置应从模型读取；该模型通常是 36 层、hidden size 2560，禁止在核心逻辑里硬编码这些数值。

DINO 使用冻结的 DINOv3-ViT-S，与研究计划对齐。224×224 输入时，目标是只取空间 patch token，形状通常为 196×384；实际读取 checkpoint 配置并断言，不要把 CLS/register token 混进未来监督。不要静默切换到 DINOv2、其他尺寸或随机初始化模型。

模块结构：

1. 当前 RGB + language → Qwen 原生视觉语言输入表示。
2. 当前 RGB → frozen DINO → current projector。
3. robot state → state projector。
4. candidate future Z → future time/spatial/type encoding → future projector。
5. candidate action A → action step/type encoding → action projector。
6. 以上 token 与可学习 Energy Token 拼接，输入 Qwen 的语言主干。
7. 取 Energy Token 的最终隐藏状态，经 scalar energy head 输出每样本一个实数。

目标输入顺序：

`[C, T_future, T_action, E_token]`

其中 `C = [Qwen原生视觉语言条件, current DINO tokens, robot state tokens]`。

第一版使用 causal attention。Energy Token 必须能看到全部有效 C、Z、A token。Future 可以看前面的 C；Action 可以看 C 与 Future。最终能量对 A、Z 都必须存在有效梯度。

不要在 Qwen 已经算完所有语言层之后才把 A/Z 拼到 hidden states 上，再简单接 MLP。这不符合候选参与 Qwen 能量计算的目标架构。

不需要在能量路径中生成词表 logits。绕过不需要的文本生成 head；如果词嵌入与 lm_head 共享参数，不要误删文本输入所需的权重。

## 4. 正确接入 Qwen 多模态路径

这是需要实际检查 Transformers 实现的部分，不能只假设把 `inputs_embeds` 拼起来就够了。

正确保留或适配：

- 原生图像 token 替换与特征注入。
- Qwen3-VL 的 DeepStack 等多层视觉融合路径，如果当前版本启用。
- image_grid_thw、mRoPE/position_ids、padding/attention mask。
- 变长语言、变长视觉输入、多视角输入。
- 在原生视觉语言序列后追加连续候选 token 的位置编码。
- Energy Token 的准确索引，避免误取 padding、文本结束符或其他样本的位置。

明确新增 token 的位置编码方案，避免未来帧、空间 patch、动作时间步与 Qwen 原生坐标冲突。空间信息不能完全依赖于 DINO 输出 token 的任意拼接顺序。

优先通过小范围 subclass/adapter 扩展原有接口，不要大范围重写 Transformers。若必须调用内部方法，集中封装并说明版本约束。

第一版先实现语义正确的完整前向。KV cache、条件前缀缓存和其他优化仅在验证正确后考虑。不要在训练时错误缓存或 detach 仍需训练的条件分支。

## 5. 数据契约与时间对齐

令 batch size 为 B、动作 horizon 为 H、动作维度为 Da、状态维度为 Dq、未来时间数 F=3、每图 patch 数 P、DINO 通道数 D。

建议逻辑数据契约如下，实际字段名服从仓库惯例：

| 字段 | 形状/含义 |
|---|---|
| 当前图像 | B×V×3×224×224 或仓库现有图像列表 |
| language | B 条指令 |
| state | B×Dq 或仓库既有时序格式 |
| action target A* | B×H×Da，使用既有归一化 |
| action mask | 可广播至 B×H×Da |
| future RGB | B×F×V×3×224×224 |
| future target Z* | 单视角为 B×F×P×D，多视角显式增加 V |
| future mask | 标记有效时间点、视角和 patch |
| future offsets | 本样本选取的三个实际轨迹偏移 |

示范动作定义为 `[a_t,...,a_(t+H-1)]`。必须检查环境/数据记录约定，解释第 H 个动作之后对应哪一帧，不要凭数组切片猜测。

三个未来偏移默认约为 `round(i*H/3), i=1,2,3`。H=16 时可选 `[5,11,16]`。要求偏移为正、单调、与动作覆盖范围对齐；H 太小导致重复时明确处理或拒绝配置。

未来帧来自同一 episode，禁止越界读到下一条轨迹。默认可以只采样有完整有效 horizon 的起点；若现有数据管线使用尾部 padding，必须同时处理 action/future mask，不能把补帧当作真实未来。给出实际采用的策略。

当前和未来图像采用一致且明确的 DINO 预处理。随机裁剪/几何增强若用于同一条转移，应保持必要的时序一致性，避免模型被迫预测不相关的视角变换。

动作维度、state 维度、相机数量从实际数据/配置读取，不要写死成 7 维动作、单相机或某台机器人的 state。连续 state 使用 projector，不能仅把它序列化为文字来替代目标分支。

DINO 目标在 no_grad 下提取并 detach。提供预计算/cache 的接口或清楚的扩展点；若实现缓存，键必须包括 episode/frame/camera、DINO checkpoint、预处理和归一化版本，避免错配。

真实未来 RGB 只用于训练监督，或用于条件训练中构造被固定的 Z*。部署的联合预测接口不能要求任何真实 future_image、A*、Z*。

## 6. Energy 模块与候选优化器

建议新增 `QwenEnergyWAM` framework（名字可按仓库规范调整），将以下逻辑解耦：

- 条件/候选编码与 scalar energy forward。
- energy refinement/solver。
- 双向、联合、action-only 的训练组织。
- 输入输出适配、评测和日志。

Energy 接口概念上为：

`energy(condition, future_candidate, action_candidate, masks) -> Tensor[B]`

每条样本独立输出一个能量。禁止沿 batch 做会使样本候选相互影响的归一化或聚合。

联合更新必须实现：

`gA, gZ = autograd.grad(E.sum(), [A, Z], create_graph=training)`

`A_next = A - alpha_A * gA`

`Z_next = Z - alpha_Z * gZ`

以上是接口示意，需结合仓库、mask 与可训练参数实现。使用 sum 是为了保持每样本的候选梯度不随 batch size 人为缩小。损失的 batch 平均放在外层进行。

第一版采用同一点 `(Ak,Zk)` 计算两个梯度，然后同时更新。若另外支持 alternating update，必须使用不同配置名并独立记录，不能与同时更新混写。

支持三种优化模式：

- `action_given_future`：固定 Z*，只更新 A。
- `future_given_action`：固定 A*，只更新 Z。
- `joint`：同时更新 A 与 Z。

Action-only 模式使用独立可用的 `E(C,A)` 路径，不能偷偷读取未来真值。可用明确的 learned placeholder 表示缺失模态，但必须正确配置 mask 并解释。

候选变量在归一化后的空间优化。初始化、K、alpha_A、alpha_Z、候选梯度 clipping、动作约束应可配置。默认不要用真值动作/未来的加噪版本作为联合部署的初始化。

可以提供 `normal` 初始化，和 `current DINO + noise` 的未来初始化选项；选项的均值、尺度与统计来源都要写清。任何 warm start 或候选多次重启属于额外选项，不是第一版必需条件。

Action 和 Future 使用各自步长。默认不引入额外 Langevin 噪声、对比损失或能量标签回归。

Action 约束依据归一化与机器人控制接口处理；训练中的 clamp 会影响梯度，需明确其行为。不要对所有 latent 特征强行按动作上下界截断。

冻结 DINO；其余训练范围可配置。主方法需要能训练 Qwen 语言主干与新增模块。优先实现全语言主干训练选项；若使用仓库现有 LoRA 能显著改善资源可行性，可以另提供 LoRA 配置，但必须明确区别。Qwen vision encoder 的冻结策略单独配置并记录。不能默认只训练一个 energy head 却宣称完整主方法已验证。

## 7. 保留展开式训练的正确梯度

训练的外层损失要能穿过候选更新，回传到能量网络参数。完整 unroll 是第一版参考实现。

必须做到：

- 候选需要梯度，训练内层使用 `create_graph=True`。
- 不在每一步随意 detach 候选、候选梯度或能量，使最终 loss 与 θ 断开。
- 内层不执行参数 optimizer.step；网络参数只在外层执行一次正常训练更新。
- 固定的 A*/Z* 是监督/条件，不作为内层可优化变量。
- 梯度只在有效 token/动作维度上更新，padding 不影响能量聚合或 loss。
- 不使用取整、numpy、argmin、离散采样等不可微操作切断候选到能量的路径。
- 候选优化器与外层参数优化器的步长和梯度 clipping 分开命名。

训练涉及混合二阶导数。某些 Flash/SDPA kernel、gradient checkpointing、分布式策略和低精度组合可能不支持所需 double backward。先在当前依赖上实际测通二阶梯度；必要时使用 eager 或支持高阶反向的 math attention，限制到新 framework，不要改坏其他模型。

默认参考实现关闭未经验证的 checkpointing/compile/高阶不兼容 kernel。候选 A/Z 的主副本、能量累加、梯度和 loss 尽量使用 float32。若主干使用 BF16/AMP，通过真实反向测试验证转换没有切断梯度。CPU tiny 测试采用 float32。

如果以后增加 truncated unroll、stop-gradient 近似、梯度归一化或其他省显存选项，默认关闭、名称明确，记录它们偏离完整方法的地方。不能用近似模式的测试冒充完整 unroll 测试。

Batch 内部多个能量项、两个条件方向是否共享条件计算图，应与图的释放策略一致。优先先求出总外层 loss 再执行一次 backward，避免多次 backward 导致图已释放或 DDP 重复标记问题。

## 8. 阶段一：双向条件训练

同一个 Eθ 学习两个方向。

Future→Action：

1. 用真实未来 RGB 构造 Z*，固定它。
2. 初始化 A0，展开 K_A 次 action refinement。
3. 得到 AK，计算 `LA = J_A(AK,A*)`。

Action→Future：

1. 固定真实归一化动作 A*。
2. 初始化 Z0，展开 K_Z 次 future refinement。
3. 得到 ZK，计算 `LZ = J_Z(ZK,Z*)`。

总损失：

`L_bi = lambda_A * LA + lambda_Z * LZ`

默认 J_A 使用有效动作元素上的 mean squared error；J_Z 使用有效时间/视角/patch/channel 上的 mean squared error。都按有效元素数归一化，不能直接把 20 多万维 latent 的误差求和与动作误差求和相加。若采用额外 cosine loss 等，默认关闭并解释。

两个方向的调度可选择每 batch 都计算或交替采样。提供明确配置，日志分别记录，不要把两个不同网络说成共享能量。

## 9. 阶段二：联合训练

从阶段一 checkpoint 加载模型，在同一个 framework 中继续训练：

1. 仅由允许的初始化方式产生 A0、Z0。
2. 同时展开 K 次联合能量更新。
3. 计算：

`L_joint = beta_A * J_A(AK,A*) + beta_Z * J_Z(ZK,Z*)`

4. 外层损失反向传播，更新网络参数。

默认联合阶段使用上述联合损失；若可选保留双向辅助损失，独立开关与权重，主配置明确是否启用。

支持直接从预训练 Qwen 初始化进行 joint 训练，作为 `direct_joint` 消融。阶段一、阶段二、direct_joint、action_only 的配置可共享基础配置，避免复制大量代码。

checkpoint 需包含模型、外层 optimizer/scheduler、训练阶段、global step、随机状态、归一化统计或其版本引用，以及必要配置。区分：

- resume 同阶段训练：恢复 optimizer/scheduler 等。
- 从 bi 阶段开始 joint：加载模型，按显式策略重置或继续 optimizer/scheduler，日志说明。

## 10. 推理与现有模型服务集成

联合推理只接收当前 RGB、language、robot state 和模型配置/统计信息。

执行：构造 C → 初始化 A/Z → K 次 energy refinement → 输出 AK、可选 ZK 与求解诊断。

模型 eval，网络参数不更新，但候选必须能求梯度。检查外层评测、模型服务器、predict_action 的 decorator 和上下文：

- 不得把整个 refinement 放进普通 no_grad。
- 外层若启用 inference_mode，仅增加 enable_grad 可能不够；应在合适边界退出 inference_mode，并在其中创建正常候选 tensor。
- 推理内层可在每一步 detach 后重新启用候选梯度，以释放历史图；这是推理行为，不要照搬到完整训练。
- 推理使用 `autograd.grad` 获取候选梯度，避免给参数积累 `.grad`。
- 输出给客户端前 detach，恢复现有 action unnormalization、shape 和控制模式约定。

与 StarVLA 的既有 predict_action 返回协议兼容。诊断信息不要破坏服务客户端。执行前 m 步再重规划的策略按现有评测框架配置，不要擅自改变任务评测协议。

ZK 是 latent，默认不实现 RGB/video decoder，也不要宣称直接生成了未来视频。

## 11. 配置与默认值

提供真实可解析且与当前仓库一致的配置，不要求机械使用下列字段名，但应覆盖这些概念：

```yaml
framework: QwenEnergyWAM
backbone: Qwen3-VL-4B-Instruct
dino: DINOv3-ViT-S
freeze_dino: true
freeze_qwen_vision: true       # 第一版工程默认，需说明而非称为论文定论
tune_language_model: true
attention_backend: eager      # 先测通二阶反向，再考虑其他 backend
num_future_frames: 3
future_offset_rule: uniform_within_action_horizon
stage: bi                     # bi / joint / direct_joint / action_only
unroll_mode: full
train_inner_steps: 4
eval_inner_steps: 4
alpha_action: 0.05
alpha_future: 0.05
action_loss: masked_mse
future_loss: masked_mse
lambda_action: 1.0
lambda_future: 1.0
beta_action: 1.0
beta_future: 1.0
initialization_action: normal
initialization_future: normal
init_action_std: 0.1
init_future_std: 0.1
```

这段只是字段概念示意，不是让你直接把一个不符合仓库 schema 的 YAML 塞进去。动作/state 维度、相机、归一化和 horizon 与实际任务配置保持一致。H 可优先使用已有任务的 16 步设置。

上述 K、步长、权重、初始化尺度是保守起点，不是师兄已验证的超参数。用短实验检查数值、梯度与优化尺度；若需要调整，记录理由。Smoke 配置可使用 K=1 或 2、极小 batch，正式基础配置保留可覆盖的 K。

至少提供：tiny CPU 配置、真实 Qwen smoke 配置、bi 基础配置、joint 基础配置、direct_joint 和 action_only 消融配置，以及评测启动配置。

## 12. 依次完成的实施阶段

### A. 仓库审查与数据链路

输出简短审查结果，确定实际修改文件。读取一条真实样本，打印脱敏后的 shape、episode/frame 索引、future offsets、mask、action/state 归一化契约。核实当前帧和未来帧。

### B. 小模型验证能量与训练机制

实现可注入的小型 Transformer/能量 backbone，用于 CPU 测试同一套 solver、loss 和阶段组织。Tiny 模式不需要下载 Qwen/DINO，使用确定性 mock encoder 与合成数据即可。Mock 必须只在显式测试模式使用，真实模式缺权重时明确报错。

### C. 接入真实 Qwen/DINO

完成实际 token 注入、能量前向和二阶反向，验证 A/Z 梯度以及参数梯度。不要把 tiny 成功等同于真实 Qwen 路径成功。

### D. 双向与联合训练

接入现有 trainer、日志、checkpoint 和阶段切换。完成短训练、保存恢复、从 bi 切换 joint。检查学习曲线与梯度。

### E. 推理服务和短闭环评测

兼容既有模型服务和仿真评测。核对相机、state、action 维度与模式，运行有资源支持的短评测。环境或资源不足时保留实现并报告尚未验证的具体环节。

请按依赖顺序推进，不要在 tiny 求解器都没有梯度时就启动长时间 GPU 训练。

## 13. 必须执行的有意义验证

这些检查针对本方法容易出错的数学和集成性质，不是只检查函数存在。

1. **形状与可变输入**：batch>1、不同语言长度、padding、多视角可用时的 shape，以及末尾 Energy Token 的索引正确。
2. **梯度存在**：候选 A/Z 的梯度为有限值且在非退化输入上非零。固定分支不被更新。
3. **外层可训练**：K≥2 完整 unroll 后，loss 对至少一个 Energy Head、Projector、启用训练的 backbone 参数有有限且非零的梯度；参数 optimizer.step 后发生实际变化。
4. **数值梯度核对**：tiny float64 配置上，以有限差分或 gradcheck 核对代表性候选/参数梯度，并覆盖外层经过内层的梯度。精度和容差写清。
5. **解析 toy solver**：用已知可解的耦合二次能量验证 action-only、future-only、joint 更新公式、同步更新和收敛方向。Toy 仅用于验证 solver，不替代真实网络。
6. **双向影响**：在非退化测试模型中固定 C/A 改变 Z，检查 action 梯度发生变化；固定 C/Z 改变 A，检查 future 梯度变化。共享标量不自动保证耦合，此检查也不等于证明学到了物理规律。
7. **batch 独立性**：同一样本单独计算与放入不同 batch 中计算，候选能量梯度在合理容差内一致。
8. **mask 与时间边界**：未来帧不跨 episode，尾部无效元素不参与损失；改变被 mask 的目标不改变损失。说明 padding 在 attention 中如何处理。
9. **禁止真值泄漏**：联合推理不接收真值；在相同 C 和随机种子下，修改 batch 中仅用于监督的 A*/future RGB，不应改变联合初始化和 refinement 输出。避免与允许固定真值的条件训练混淆。
10. **真实 backbone 的高阶反向**：若权重和 GPU 可用，至少完成 batch=1、小 K 的 Qwen+DINO 完整训练步并同步确认无隐藏 CUDA 异常，记录时间、峰值显存和参数梯度。
11. **推理上下文**：验证普通调用、外层 no_grad、如实际服务采用则包括 inference_mode 场景。推理前后参数不变化、不积累参数梯度，输出可正常序列化。
12. **checkpoint 与阶段切换**：保存加载后在固定种子/eval 下输出一致；同阶段恢复与 bi→joint 切换均运行通过。
13. **小样本拟合**：tiny 模式进行几十至几百步训练，确认端点预测误差能明显下降。真实数据/权重可用时再做真实少样本短训练，不能要求随机初始化时每一步能量一定下降。
14. **兼容性**：新增 optional dependency 不影响已有 framework 的导入与现有 smoke/config；执行修改范围内必要的回归检查。

测试如果失败，修复后再交付。不得把跳过的测试写成通过。缺 GPU/权重/数据/模拟器时，明确区分已执行通过、失败、未执行；继续完成不依赖缺失资源的部分。

## 14. 日志与研究诊断

至少记录：

- 阶段、global step、随机种子、模型/数据/预处理版本。
- 各方向 loss、joint action/future loss、mask 有效元素数。
- refinement 的能量轨迹、候选梯度范数、候选更新幅度。
- debug/训练可用真值时，每步 action/future 预测误差。
- nonfinite、动作越界或 clamp 比例，实际步长与迭代次数。
- 可训练/总参数、单训练步耗时、同步后的推理耗时、GPU 峰值显存。

日志中的 energy 变化是诊断，不能直接作为跨任务成功概率或校准置信度。端点监督并不自动保证全局能量排序，更不能先写成模型已具备物理推理。

## 15. RoboCasa365 与消融入口

优先复用本仓库已有 RoboCasa365 的一个简单任务，例如存在的 OpenDrawer。选择服务器已经有的数据，不要求额外采集。

明确区分 RoboCasa365/PandaOmron 与 LDA 论文中的 RoboCasa-GR1/tabletop；动作、相机、状态和成功率不可混用。检查当前 checkout 中训练与评测的相机数量/顺序是否一致，不能只因为接口能运行就视为评测有效。

交付以下实验的可执行入口：

- 仓库现成的普通 VLA baseline。
- action_only_energy。
- 单方向条件训练消融。
- direct_joint_energy。
- bi_then_joint_energy 主方法。
- 若仓库已有或易复用 Joint Flow WAM，给出匹配条件的对照入口；若没有，明确缺失，不要伪造一个“LDA baseline 已复现”的结论。

对照配置尽可能控制数据、骨干、输入视角、归一化、训练规模与评测协议。性能由实验决定；本次不编造成功率。

## 16. 最终交付物

1. 实际代码修改，按仓库规范注册新 framework。
2. 数据读取/未来帧与 mask 的必要改造。
3. scalar energy、三种 refinement、双阶段训练、联合推理实现。
4. 基础与消融配置、短 smoke 脚本、完整训练和评测命令。
5. checkpoint、恢复与阶段切换流程。
6. 关键数学性质和集成测试，以及真实执行结果。
7. 一份中文 README/方法实现说明，至少包括：
   - 研究计划如何对应代码模块。
   - 带真实 shape 的数据流和训练/推理差别。
   - Qwen 多模态融合、token 排序、位置编码与 mask 处理。
   - 完整 unroll 的梯度路径、支持的 attention backend 与版本。
   - 权重/数据放置与配置方法。
   - 如何启动 tiny、真实 smoke、bi、joint、resume、evaluate。
   - 实测资源需求、未验证项、工程默认值和研究假设。
8. 一份 implementation status，便于后续继续：完成什么、运行了什么、日志在哪、还缺什么。

不能只交空类、伪代码、长期 TODO、始终返回固定能量、随机动作或随机未来的占位实现。真实模式与 mock 模式必须有明确边界。可以保留确实依赖外部资源的后续事项，但要交代阻塞原因、准确命令和所需资源。

## 17. 执行授权和完成标准

你可以在当前仓库范围内完成必要的读取、编辑、测试和短验证，不需要每一个模块都向我确认。保留既有用户文件与结果；不要自动 push、提交 PR、重置仓库、修改原始数据或启动长时间训练。依赖变更优先在当前项目环境内进行并记录。

权重或数据缺失时，先查现有位置和配置，避免重复下载。确实没有时提供准确的资源说明；继续完成 tiny 验证、真实适配代码和启动脚本，不要把所有工作都停在等待下载上。

最终回复按以下内容汇报：

1. 架构与代码已经实现到什么程度。
2. 关键修改文件与作用。
3. 实际执行的测试、结果、日志位置。
4. 可直接复制的 smoke、阶段一、阶段二、恢复和评测命令。
5. 相对原研究计划的工程默认或必要偏差。
6. 尚未执行的验证和具体原因。

完成标准是：同一套实现的 tiny 路径能端到端完成双向训练、联合训练、联合推理与保存恢复；真实 Qwen/DINO 路径已经接好，在当前服务器资源可支持时完成真实训练步与推理验证；现有 StarVLA 流程仍可使用。若真实资源不可用，必须明确称为“tiny 验证通过、真实路径待验证”，不能称完整实验证明成功。

现在请开始阅读附件和审查当前仓库，然后继续执行改造。

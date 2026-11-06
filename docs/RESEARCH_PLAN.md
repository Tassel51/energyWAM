Latent Transition Energy World Action Model 研究计划
1. 研究目标
探索利用统一的 Latent Transition Energy 建模 World Action Model（WAM）中 Action 与 Future 的相互依赖关系，通过双向条件优化与联合能量生成，使 Future Modeling 更有效地服务于 Action Generation

整体思路是：复用预训练VLM/LLM → 将其改造为 State–Action–Future 标量 Energy Model（参考 EBT）→ 先学习 Future→Action、Action→Future 两种条件优化能力 → 再通过同一个 Energy 进行 Action–Future 联合生成 → 在 RoboCasa365 上验证操作性能与转移建模能力

2. 研究动机
现有 WAM 主要沿着三类路线建模动作与未来状态：

Future → Action： 先生成未来状态，再通过 Inverse Dynamics 预测动作；未来目标可能缺少可执行性约束，预测误差也会传递到动作生成
Action → Future： 根据动作预测环境变化，但仍需额外的决策过程将未来预测转化为动作
Action + Future： 使用 Joint Diffusion / Flow Matching 联合生成两者，缓解单向建模的割裂问题，但其生成式目标主要侧重拟合 Action–Future 的联合分布，缺少对两者转移兼容性的显式建模，Future Modeling 对 Action Generation 的约束作用仍有待加强
motivation：联合生成是否充分利用了 Action 与 Future 之间的相互依赖？Future Modeling 究竟在多大程度上帮助了 Action Generation？

为此，考虑使用一个共享的标量 Energy 函数描述当前状态、候选动作和未来状态的相容关系，把 Action Prediction、Future Prediction 和 Joint Generation 统一为同一 Energy Landscape 上的优化问题

3. 实验基础与模型架构
3.1 实验基础
架构参考： LDA-1B，参考其 Qwen3-VL 与 DINOv3 的双视觉编码设计，以及 State/Action Encoder 等模块。
Energy 参考： Energy-Based Transformers（EBT），重点参考标量 Energy 建模、梯度迭代优化及展开式训练。
基础模型： Qwen3-VL-4B-Instruct，保留其预训练视觉编码器和 Language Transformer，移除原有文本生成 Head，使用新增的 Energy Head 输出标量能量。
整体目标是复用 Qwen3-VL 的视觉语言理解与 Transformer 表征能力，构建统一的 State–Action–Future Energy Model，通过能量梯度直接生成 Action 与 Future Latent，无需额外的 Flow/Diffusion 生成网络。

3.2 输入表征与编码
给定当前 RGB 观测 
I
t
、语言指令 
ℓ
、机器人状态 
q
t
，以及长度为 
H
 的 Action Chunk：

A
t
=
[
a
t
,
a
t
+
1
,
…
,
a
t
+
H
−
1
]
∈
ℝ
H
×
D
a
其中 
D
a
 为动作维度。模型通过以下分支构造输入 Tokens。

（1）视觉语言条件编码

当前 RGB 图像与语言指令首先通过 Qwen3-VL 原生的 Vision Encoder 和文本 Embedding 模块，得到多模态输入表征：

T
VL
=
Embed
Qwen
⁡
(
I
t
,
ℓ
)
保留 Qwen 原有的视觉特征融合机制，将其作为后续 Language Transformer 的视觉语言条件。

同时，使用冻结的 DINOv3-ViT-S 提取当前图像的空间特征：

z
t
=
DINO
⁡
(
I
t
)
∈
ℝ
196
×
384
其中，196 对应输入分辨率为 
224
×
224
 时的 
14
×
14
 Patch Grid，384 为单个 Patch 的特征维度。

通过可训练的 Current Projector 将其映射到 Qwen 的 Hidden Space：

T
cur
=
P
cur
(
z
t
)
∈
ℝ
196
×
2560
（2）机器人状态编码

机器人关节状态、末端执行器状态等本体信息 
q
t
∈
ℝ
D
q
，经过 State Projector 映射为状态 Token：

T
state
=
P
state
(
q
t
)
∈
ℝ
N
q
×
2560
State Projector 可复用 LDA 的 State Encoder 结构，并适配 Qwen 的 Hidden Dimension。

最终将视觉语言特征、当前 DINO 特征和机器人状态组成条件序列：

C
t
=
[
T
VL
,
T
cur
,
T
state
]
（3）Future Latent 编码

在 Action Horizon 
H
 内均匀选取三个未来时间点：

τ
i
≈
i
H
3
,
i
=
1,2,3
对应的未来图像为：

I
t
+
τ
1
,
I
t
+
τ
2
,
I
t
+
τ
3
训练时，三张 Future RGB 图像通过与当前图像共享权重的冻结 DINOv3，构造 Future Latent Ground Truth：

Z
t
∗
=
[
DINO
⁡
(
I
t
+
τ
1
)
,
DINO
⁡
(
I
t
+
τ
2
)
,
DINO
⁡
(
I
t
+
τ
3
)
]
Z
t
∗
∈
ℝ
3
×
196
×
384
Future Latent 在训练和推理过程中作为连续优化变量，记为 
Z
k
，其维度与 
Z
t
∗
 一致。

将三个时间点的 Future Latent 加入对应的时间 Embedding，并沿 Token 维度拼接，再通过 Future Projector 映射至 Qwen Hidden Space：

T
future
=
P
future
(
Concat
i
=
1
3
⁡
(
Z
k
(
i
)
+
e
τ
i
)
)
T
future
∈
ℝ
588
×
2560
其中，
e
τ
i
 表示对应未来时间点的时间编码。

真实 Future RGB 仅用于训练时构造监督目标；推理时直接初始化并优化 Future Latent 
Z
k
，不需要未来 RGB 图像。

（4）Action 编码

对于候选 Action Chunk 
A
k
∈
ℝ
H
×
D
a
，使用可训练的 Action Projector 将每个时间步的连续动作映射为一个 Action Token：

T
action
=
P
action
(
A
k
)
∈
ℝ
H
×
2560
其中可加入 Action Step Positional Embedding，保留动作块内部的时间顺序。

Action Projector 参考 LDA 的 Action Encoder，但不再引入 Flow Matching Timestep。

3.3 Energy Transformer 架构
将上述条件、Future 和 Action Tokens 沿序列维度拼接，并在末尾添加一个可学习的 Energy Token：

X
0
=
[
C
t
,
 
T
future
,
 
T
action
,
 
e
E
]
其中：

e
E
∈
ℝ
2560
为新增的可学习 Energy Readout Token，用于聚合 State–Action–Future 的联合信息。

整个序列输入 Qwen3-VL-4B 原有的 36 层 Language Transformer：

X
L
=
Transformer
θ
⁡
(
X
0
)
第一版保留 Qwen 原生的 Causal Self-Attention，按照 
[
C
,
Z
,
A
,
E
]
 的顺序组织输入，使：

Condition Tokens： 表示当前视觉、语言和机器人状态。
Future Tokens： 可以关注前面的 Condition Tokens。
Action Tokens： 可以关注 Condition 和全部 Future Tokens。
Energy Token： 可以关注完整的 Condition、Future 和 Action Tokens。
由此，Action 的中间表征能够直接利用候选 Future 信息，而最终 Energy 同时依赖 Action 和 Future。

取最后一层 Transformer 中 Energy Token 对应的 Hidden State：

h
E
=
X
L
[
−
1
]
∈
ℝ
2560
经过新增的标量 Energy Head：

E
θ
(
C
t
,
Z
k
,
A
k
)
=
MLP
E
⁡
(
h
E
)
∈
ℝ
该 Energy 表示在当前条件下，候选 Action 与 Future Latent 之间的转移兼容性。

3.4 基于 Energy 的 Action–Future 生成
与传统 VLA 直接预测 Action、LDA 使用 MM-DiT 预测 Flow Velocity 不同，我们的模型仅输出标量 Energy，通过对输入候选变量求梯度完成生成。

首先初始化候选 Action 和 Future Latent：

A
0
∼
p
A
,
Z
0
∼
p
Z
每一步将 
(
C
t
,
Z
k
,
A
k
)
 输入共享 Energy Transformer，计算：

E
k
=
E
θ
(
C
t
,
Z
k
,
A
k
)
然后分别求 Energy 对 Action 和 Future Latent 的梯度：

g
A
=
∇
A
k
E
θ
(
C
t
,
Z
k
,
A
k
)
g
Z
=
∇
Z
k
E
θ
(
C
t
,
Z
k
,
A
k
)
沿能量下降方向迭代更新：

A
k
+
1
=
A
k
−
α
A
g
A
Z
k
+
1
=
Z
k
−
α
Z
g
Z
经过 
K
 次 Energy Optimization 后得到：

(
A
K
,
Z
K
)
其中 
A
K
 作为最终生成的 Action Chunk，
Z
K
 表示该动作对应的预期未来视觉状态。

网络训练采用两个阶段：

Bidirectional Conditional Energy Learning： 给定真实 Future 优化 Action，或给定真实 Action 优化 Future，利用两个方向的预测损失训练同一个 Energy Network。
Joint Energy Optimization： 同时从候选 Action 和 Future 出发，通过共享 Energy 联合优化，并使用真实 Action/Future 监督最终生成结果。
训练时，内层 Energy Optimization 更新候选变量 
(
A
k
,
Z
k
)
，外层通过最终预测损失更新网络参数 
θ
。外层梯度穿过展开的 Energy Optimization 过程，使网络学习能够引导 Action–Future 生成的 Energy Landscape。

最终将 Action Prediction、Future Prediction 和 Joint Generation 统一为同一 Energy Model 上的条件优化与联合优化过程。

4. 方法与训练流程
阶段一：Bidirectional Conditional Energy Learning
使用同一个 
E
θ
，分别训练两个条件方向，学习 Action 与 Future 的双向依赖关系。

Future → Action： 固定示范中的真实未来 
Z
∗
，从初始动作 
A
0
 出发优化：

A
k
+
1
=
A
k
−
α
A
∇
A
E
θ
(
c
,
A
k
,
Z
∗
)
,
ℒ
A
=
J
A
(
A
K
,
A
∗
)
.
Action → Future： 固定真实动作 
A
∗
，从初始未来 latent 
Z
0
 出发优化：

Z
k
+
1
=
Z
k
−
α
Z
∇
Z
E
θ
(
c
,
A
∗
,
Z
k
)
,
ℒ
Z
=
J
Z
(
Z
K
,
Z
∗
)
.
联合训练目标为：

ℒ
bi
=
λ
A
ℒ
A
+
λ
Z
ℒ
Z
.
这里监督的是经过多步 Energy Optimization 得到的 Action/Future，而不是让 Energy 标量直接回归某个数值标签；训练梯度通过展开的优化过程传回同一个网络。

阶段二：Joint Energy Optimization
在双向条件训练基础上，同时初始化 Action 和 Future，通过共享 Energy 联合更新：

A
k
+
1
=
A
k
−
α
A
∇
A
E
θ
(
c
,
A
k
,
Z
k
)
,
Z
k
+
1
=
Z
k
−
α
Z
∇
Z
E
θ
(
c
,
A
k
,
Z
k
)
.
ℒ
joint
=
β
A
J
A
(
A
K
,
A
∗
)
+
β
Z
J
Z
(
Z
K
,
Z
∗
)
.
目标是使 Action 与 Future 在同一次优化中相互约束，学习可用于联合生成的 State–Action–Future Energy Landscape。先双向训练、再联合训练是当前优先尝试的策略，后续与直接联合训练对比。

阶段三：Joint Inference
部署时仅输入当前观测、语言和机器人状态，初始化 
(
A
0
,
Z
0
)
，通过多步 Energy Gradient Optimization 得到 
(
A
K
,
Z
K
)
：执行 
A
K
，将 
Z
K
 作为对应的预期未来状态表示。生成过程直接由 Energy Model 完成，无需额外的 Flow/Diffusion Action Head。

5. 实验与评估
先以 RoboCasa365 为主要实验平台，选取物体搬运、关节机构交互、设备状态控制等不同类型的任务，使用已有示范数据构造对齐的 
(
c
,
A
∗
,
Z
∗
)
 训练样本，不额外采集数据；后续如需负样本，也仅从已有示范中构造

主要对照与消融：

常规 VLA / Joint WAM： 比较不显式进行 Energy Optimization 的动作生成，以及标准 Action–Future 联合生成方法
Action-only Energy： 仅对 Action 建模，检验引入 Future 联合建模是否真正改善动作生成
训练策略： 比较仅单方向、仅直接联合训练，以及 Bidirectional → Joint 两阶段训练
重点评估：

操作性能： RoboCasa365 任务成功率（SR），报告不同操作类型及整体表现。
Future Modeling： Future Latent 预测质量，以及预测结果对不同 Action 的响应。
转移一致性与泛化： Action–Future 相容性、未见场景和未见有效组合下的表现。
计算效率： Energy Optimization 步数、训练开销、推理延迟及稳定性。
6. 核心参考工作
Energy-Based Transformers are Scalable Learners and Thinkers：直接通过标量 Energy 和梯度优化生成预测
EBT-Policy: Energy Unlocks Emergent Physical Reasoning Capabilities：将 Energy Optimization 用于机器人动作生成；我们的重点进一步延伸到 Action–Future 的双向及联合生成
LDA-1B: Scaling Latent Dynamics Action Model via Universal Embodied Data Ingestion：Action 与 Future DINO Latent 联合建模的相关 WAM 工作
# LTE-WAM: Latent Transition Energy World Action Model
## Bidirectional Conditional Energy Learning for Joint Action–Future Optimization
### 基于双向条件能量学习与动作—未来联合优化的世界动作模型

> **论文初稿（Method + Experimental Protocol 完整版）**  
> **状态：研究方案与实验设计稿；尚未获得的实验结果一律留空，不代表已验证的结论。**  
> **基础工作：** LDA-1B；**能量学习参考：** EBT / EBT-Policy；**代码框架：** StarVLA；**主要评测：** RoboCasa365。  
> **作者：** 待填写。**版本：** 2026-10。

---

## Abstract

World Action Models (WAMs) aim to integrate action generation with future-state modeling, allowing robot policies to exploit predictive dynamics beyond conventional behavior cloning. Recent approaches, such as LDA-1B, jointly model action chunks and future visual latents using generative diffusion architectures. However, joint distribution modeling does not, by itself, ensure that action proposals are explicitly constrained by their corresponding future-state transitions. This paper proposes **Latent Transition Energy World Action Model (LTE-WAM)**, a unified energy-based formulation of state-conditioned action–future compatibility. LTE-WAM encodes current images and language with a pretrained Qwen3-VL backbone, extracts structured spatial representations using a frozen DINOv3 encoder, and evaluates candidate action chunks and future visual latents through a shared Transformer and scalar energy head. Instead of relying on a separate diffusion or flow-based generator, predictions are obtained by iterative energy minimization. To learn a useful joint optimization landscape, we introduce a two-stage training procedure: **bidirectional conditional energy learning**, which alternately optimizes actions given ground-truth future latents and future latents given expert actions; followed by **joint energy optimization**, which refines both variables simultaneously under reconstruction-based supervision. We further specify initialization priors and diagnostic tests for genuine action–future coupling. We outline a reproducible evaluation on RoboCasa365, including policy success, future-latent fidelity, transition compatibility, generalization, and computational efficiency. The empirical effectiveness of the proposed formulation remains to be established by the planned experiments.

**Keywords:** World Action Model; Energy-Based Model; Vision-Language-Action; Latent Dynamics; Robot Manipulation; Joint Optimization.

### 中文摘要

世界动作模型（World Action Model, WAM）试图统一机器人动作生成与未来状态建模，使策略不仅能够模仿专家行为，还能够利用环境转移信息辅助决策。以 LDA-1B 为代表的方法采用生成式架构联合建模动作块与未来视觉潜变量，但联合分布学习本身并不必然保证候选动作与预测未来之间的转移相容性被充分利用。本文提出 **Latent Transition Energy World Action Model（LTE-WAM）**，以共享的标量能量函数统一刻画当前状态、候选动作及未来视觉状态之间的相容关系。该模型复用预训练 Qwen3-VL 的视觉语言表征和冻结 DINOv3 的结构化视觉特征，通过同一 Transformer 对 Action–Future 候选组合进行能量评价，并以迭代梯度下降替代独立的 Diffusion/Flow 生成头。为学习可用于联合生成的能量景观，本文设计“**双向条件能量学习—联合能量优化**”两阶段训练策略：首先分别在固定真实未来和固定专家动作的条件下优化另一个变量，随后同时优化动作与未来潜变量，并对最终结果施加监督。此外，本文系统讨论初始化分布、耦合性检验和高阶梯度训练的实现条件，提出基于 RoboCasa365 的操作性能、未来预测、转移相容性与效率评测协议。本文目前为待实验验证的完整方法与实验稿，不报告尚未获得的实验结果。

**关键词：** 世界动作模型；能量模型；视觉语言动作模型；潜在动力学；双向条件学习；联合优化。

---

# 1. Introduction and Motivation

## 1.1 Background

机器人视觉语言动作模型（Vision-Language-Action, VLA）通常学习从当前观测、自然语言指令和机器人本体状态到动作序列的映射。设当前 RGB 观测为 \(I_t\)，语言指令为 \(\ell\)，本体状态为 \(q_t\)，长度为 \(H\) 的动作块为 \(A_t\)。传统行为克隆的主要目标可概括为

\[
A_t \sim p_\theta(A_t\mid I_t,\ell,q_t).
\tag{1}
\]

该形式能够利用专家演示学习策略，但其监督信号通常直接作用于动作，而未必要求模型显式描述动作之后的环境变化。

World Action Model 进一步引入未来状态建模。近期 LDA-1B [[1]](https://arxiv.org/abs/2602.12215) 利用结构化 DINO 视觉潜变量和多模态扩散 Transformer，联合学习动作生成、视觉预测与动力学任务，为动作和未来状态的联合表示提供了重要基础。然而，从联合建模到有效决策仍然存在一个值得研究的问题：**在给定当前状态时，所生成的动作是否与所预测的未来状态真正相容，以及未来建模是否实际改善动作生成？**

## 1.2 Limitations of Existing Formulations

现有 Action–Future 建模可以概括为三类：

1. **Future-to-Action：** 首先预测未来状态，再通过逆动力学推断动作。未来状态预测误差可能传递到动作，且目标未来不一定可执行。
2. **Action-to-Future：** 根据给定动作预测未来变化。该形式可以学习前向动力学，但还需要额外策略将预测未来转化为可执行动作。
3. **Joint Action–Future Generation：** 采用 Diffusion 或 Flow Matching 联合生成动作与未来。虽然两类变量可以共享信息，但标准生成损失并未单独要求候选动作与候选未来在一个显式可优化的标量相容性函数下相互约束。

需要指出：上述限制是本文的**研究动机与待检验假设**，不意味着 Diffusion/Flow 模型天然无法学习动作—未来依赖。本文关注的是能否通过不同的建模目标，更直接地利用这种依赖。

## 1.3 Motivation: From Joint Prediction to Transition Compatibility

考虑“抓住把手并打开抽屉”的任务。相同当前观测下，“抓住并拉动—抽屉打开”的组合通常合理；而“机械臂离开抽屉—抽屉自行打开”的组合与动作导致的状态转移不一致。仅有两个分别准确的预测器，并不能保证它们在候选层面保持一致。

为此，本文构造共享标量能量函数：

\[
E_\theta(C_t,A,Z)\in\mathbb R,
\tag{2}
\]

其中 \(C_t\) 为当前条件表示，\(A\) 为候选动作块，\(Z\) 为候选未来视觉潜变量。训练目标不是预先定义能量数值的“真值”，而是学习一个**能通过输入梯度将候选动作和未来优化到专家示范附近的能量景观**。

在该定义下，Action Prediction、Future Prediction 与 Joint Generation 可写成同一能量函数上的三种优化：

\[
\begin{aligned}
A^\dagger &= \arg\min_A E_\theta(C_t,A,Z^*),\\
Z^\dagger &= \arg\min_Z E_\theta(C_t,A^*,Z),\\
(A^\dagger,Z^\dagger)
  &= \arg\min_{A,Z} E_\theta(C_t,A,Z).
\end{aligned}
\tag{3}
\]

实际系统采用有限步、局部梯度下降近似上述最优化，而不声称能够找到全局最小值。

## 1.4 Proposed Contributions

本文拟提出并通过实验验证以下贡献：

1. **Unified State–Action–Future Energy Formulation.** 将动作预测、未来潜变量预测与二者联合生成统一为一个共享的、可微分的标量能量模型，不依赖额外的动作 Diffusion/Flow 生成网络。
2. **Bidirectional-to-Joint Energy Training.** 提出先双向条件优化、再联合优化的训练策略，以学习从 Future 到 Action 和从 Action 到 Future 的可微分优化能力。
3. **Explicit Compatibility Evaluation.** 通过不匹配配对、条件交换和交叉梯度敏感性分析，检验 Future 是否真正影响动作生成，而非仅作为独立辅助监督。
4. **Reproducible WAM Evaluation.** 依托 StarVLA 与 RoboCasa365，制定操作成功率、未来潜变量质量、转移一致性、泛化及推理开销的可复现实验协议。

> 上述条目描述的是**方法贡献及验证目标**，不应在实验完成前写成“已显著提升成功率”或“已证明物理推理能力”。

---

# 2. Related Work

## 2.1 Vision-Language-Action Policies

视觉语言动作模型利用大规模视觉语言预训练知识与机器人演示数据，将指令理解映射到连续或离散动作。Diffusion Policy [[5]](https://arxiv.org/abs/2303.04137) 等生成式策略将动作块采样转化为逐步去噪，具有表达多模态动作分布的优势。Flow Matching [[6]](https://arxiv.org/abs/2210.02747) 通过学习向量场构造从初始分布到目标动作的连续变换。本文与这些方法的主要区别在于：**优化方向并非由一个显式预测速度/噪声的 Action Head 给出，而是由单个标量能量对候选变量求导得到**。

## 2.2 World Action Models and Latent Visual Dynamics

LDA-1B [[1]](https://arxiv.org/abs/2602.12215) 通过冻结 DINOv3 表征未来视觉状态，在 MM-DiT 中联合建模未来潜变量和动作块，说明视觉动力学可以与策略学习共享表示。本文继承其“Qwen 视觉语言语义 + DINO 空间特征 + Action/Future 联合建模”的设计思路，但将生成主干改为 State–Action–Future Energy Transformer。

DINOv3 [[7]](https://arxiv.org/abs/2508.10104) 提供高质量密集视觉特征。采用视觉潜变量而非像素预测，可减少与任务无直接关系的像素重建负担。不过，DINO 潜变量并非天然的物理状态向量，对其预测仍需通过实际转移评测检验。

## 2.3 Energy-Based Transformers and Robotic Energy Policies

Energy-Based Transformers（EBT）[[2]](https://arxiv.org/abs/2507.02092) 通过对输入—候选输出组合赋予标量能量，并以梯度下降生成预测。EBT-Policy [[3]](https://arxiv.org/abs/2510.27545) 已将能量优化用于机器人动作策略。因此，**“使用 Energy 生成机器人动作”本身不是本文的新颖点**。本文拟研究的是将动作生成扩展为与未来状态预测耦合的共享转移能量，并通过两个方向的条件优化和联合优化显式评估 Future 对 Action 的作用。

## 2.4 Initialization Priors for Generative Policies

生成式机器人策略经常从标准高斯分布开始采样；LeaP [[8]](https://arxiv.org/abs/2606.17408) 进一步研究由机器人本体状态条件化的可学习高斯初始先验。本文将初始化先验视为影响有限步能量优化的独立设计变量，并在固定其他设置的条件下比较标准高斯、经验分布与状态相关 Future 初始化。可学习先验作为**可选扩展实验**，不构成基础模型的必要组件。

---

# 3. Problem Formulation

## 3.1 State, Action, and Future

在时间 \(t\)，给定单个当前 RGB 观测 \(I_t\)、指令 \(\ell\) 和机器人本体状态 \(q_t\)。定义长度为 \(H\) 的连续动作块：

\[
A_t^*=[a_t^*,a_{t+1}^*,\ldots,a_{t+H-1}^*]
\in\mathbb R^{H\times D_a},
\tag{4}
\]

其中 \(D_a\) 由选定机器人的实际控制接口决定。真实动作必须按相应控制定义归一化，不预设 PandaOmron 的动作维度一定为 7。

在该动作时间范围内，采样三个未来时刻：

\[
\tau_i=\operatorname{round}\!\left(\frac{iH}{3}\right),\quad i=1,2,3,
\tag{5}
\]

并从同一相机或经过明确对齐的视角获取未来图像 \(I_{t+\tau_i}\)。令 \(\phi_D\) 为冻结的 DINOv3-ViT-S 特征提取器，定义：

\[
Z_t^*=
\left[
\phi_D(I_{t+\tau_1}),
\phi_D(I_{t+\tau_2}),
\phi_D(I_{t+\tau_3})
\right].
\tag{6}
\]

当输入分辨率为 \(224\times224\)、Patch Size 为 16，且仅保留 patch tokens 时：

\[
\phi_D(I)\in\mathbb R^{196\times384},
\qquad
Z_t^*\in\mathbb R^{3\times196\times384}.
\tag{7}
\]

CLS、register tokens 是否包含在模型输出中，须由实现明确处理；上述形状仅适用于抽取 patch tokens 的配置。

一个训练样本为：

\[
\mathcal D=\{(I_t,\ell,q_t,A_t^*,Z_t^*)\}.
\tag{8}
\]

真实 Future RGB 只用于训练监督和离线评测；部署时不存在 \(I_{t+\tau_i}\) 或 \(Z_t^*\)。

## 3.2 Conditional Energy Objective

令 \(C_t=f_{\mathrm{cond}}(I_t,\ell,q_t)\) 为当前条件 token 序列。本文拟学习：

\[
E_\theta(C_t,A,Z):\quad
\mathcal C\times\mathcal A\times\mathcal Z\rightarrow\mathbb R.
\tag{9}
\]

较低能量被期望对应与条件相容、且能经优化到达示范附近的动作—未来组合。需要强调：**使用展开式终点损失并不自动获得经校准的能量概率或全局能量排序**。若形式上写出

\[
p_\theta(A,Z\mid C)\propto
\exp[-E_\theta(C,A,Z)/T],
\tag{10}
\]

该式仅提供一种 EBM 解释；除非进一步设计适当的概率估计目标并估计归一化常数，否则本文不声称已学习到规范化的条件密度。

---

# 4. Method

## 4.1 Overview

LTE-WAM 由四部分组成：

1. **视觉语言条件编码：** Qwen3-VL 保留预训练视觉与语言路径。
2. **空间及本体状态编码：** 冻结 DINOv3 提取当前视觉 patch 特征；State Projector 将本体状态映射至共享隐藏空间。
3. **候选变量编码：** Future Projector 和 Action Projector 将连续候选 \(Z,A\) 转化为 Tokens。
4. **共享能量建模与优化：** Qwen Language Transformer 汇聚上述 tokens，经 Energy Head 输出标量，随后利用 \(\nabla_A E,\nabla_Z E\) 迭代更新候选变量。

**模型与 LDA-1B 的关键区别：** LDA-1B 使用 MM-DiT 预测动作与 Future Latent 的生成方向；LTE-WAM 不保留这一额外生成主干，而是复用 Qwen Transformer 参数化共享 Energy 函数。

## 4.2 Multimodal Condition Representation

### 4.2.1 Qwen Vision–Language Tokens

当前 RGB 与语言指令通过预训练 Qwen3-VL 原生多模态路径得到：

\[
T_{\mathrm{VL}}=
\operatorname{Embed}_{\mathrm{Qwen}}(I_t,\ell)
\in\mathbb R^{N_{\mathrm{VL}}\times d},
\tag{11}
\]

其中 \(N_{\mathrm{VL}}\) 随图像处理与文本长度变化。采用 Qwen3-VL-4B-Instruct 时，语言主干隐藏维度为 \(d=2560\)，层数为 36 [[9]](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)。

### 4.2.2 Current DINO Features

相同当前 RGB 经冻结 DINOv3 提取 patch 特征：

\[
z_t=\phi_D(I_t)\in\mathbb R^{196\times384}.
\tag{12}
\]

使用可训练的 Current Projector：

\[
T_{\mathrm{cur}}=P_{\mathrm{cur}}(z_t)
\in\mathbb R^{196\times d}.
\tag{13}
\]

Qwen 视觉路径用于视觉语言语义条件；DINO 分支提供可与未来视觉目标直接对齐的密集空间表示。

### 4.2.3 Robot State Tokens

将机器人本体状态 \(q_t\in\mathbb R^{D_q}\) 经状态编码器映射为：

\[
T_{\mathrm{state}}=P_{\mathrm{state}}(q_t)
\in\mathbb R^{N_q\times d}.
\tag{14}
\]

状态编码器可参考 LDA-1B 的 State Encoder，但须适配当前状态字段及维度。最终条件为：

\[
\boxed{
C_t=[T_{\mathrm{VL}},T_{\mathrm{cur}},T_{\mathrm{state}}].
}
\tag{15}
\]

## 4.3 Future and Action Tokenization

### 4.3.1 Future Latent Tokens

对于候选 \(Z\in\mathbb R^{3\times196\times384}\)，为每个未来时间点加入时间嵌入 \(e_{\tau_i}\)：

\[
T_{\mathrm{future}}
=
P_{\mathrm{future}}\!\left(
\operatorname{Concat}_{i=1}^{3}
(Z^{(i)}+e_{\tau_i})
\right)
\in\mathbb R^{588\times d}.
\tag{16}
\]

时间嵌入应定义在投影前的 384 维特征空间；若采用投影后添加，则其维度须改为 \(d\)。两种实现应明确选择其一。

### 4.3.2 Action Tokens

对于候选动作块 \(A\in\mathbb R^{H\times D_a}\)，使用：

\[
T_{\mathrm{action}}=P_{\mathrm{action}}(A)
\in\mathbb R^{H\times d},
\tag{17}
\]

并加入动作步位置嵌入以区分时间顺序。所有动作变量的能量优化均在指定归一化空间中进行；输出机器人之前才进行反归一化和合法性处理。

## 4.4 State–Action–Future Energy Transformer

构造完整输入：

\[
\boxed{
X_0=
[C_t,T_{\mathrm{future}},T_{\mathrm{action}},e_E],
}
\tag{18}
\]

其中 \(e_E\in\mathbb R^d\) 是可学习 Energy Readout Token。经过 \(L\) 层 Qwen Language Transformer：

\[
X_L=\operatorname{Transformer}_\theta(X_0).
\tag{19}
\]

取末尾 Energy Token 对应的隐藏表示：

\[
h_E=X_L[-1],
\qquad
\boxed{
E_\theta(C_t,A,Z)=\operatorname{MLP}_{E}(h_E)\in\mathbb R.
}
\tag{20}
\]

**设计理由（写作提醒，论文中可强调解释）：** 对 \(X_L\) 做 Mean Pooling 再接 \(\operatorname{MLP}_{E}\) 也能得到标量能量，数学上同样成立。改用专门 Energy Token 的理由：
- (1) 提供可学习的专用全局读出位置，专职承担 \((C,A,Z)\mapsto E\)，而普通 token 各自还承担模态信息建模职责；
- (2) 汇聚权重经 attention 自适应学习，而非对所有 token 均匀平均（例如拉抽屉时重点关注夹爪与把手的位置关系、拉动方向、抽屉最终开合状态）；
- (3) 末尾读出与原生因果注意力自然匹配——取较早位置的 token（如某个 Future Token）作读出读不到后面的 Action，取最后一个 Action Token 读出则使其兼任动作建模与能量读出。以上是设计理由，不是 Energy Token 一定优于 Pooling 的理论证明；可做小型结构消融：Energy Token Readout vs. Last Action Token vs. Mean Pooling。

第一版使用原生因果注意力（Causal Attention），并采用 \([C,Z,A,E]\) 的顺序：Future Tokens 可读取条件，Action Tokens 可读取条件及前面的 Future Tokens，末尾 Energy Token 可读取整个序列。因此最终能量依赖于候选动作和候选未来。

**结构性消融：** 可比较 \([C,Z,A,E]\) 与 \([C,A,Z,E]\)，**论文有希望 Future Modeling 服务于 Action Generation 的动机**，或采用允许双向信息交换的专用 attention mask，以检验单向序列顺序是否引入不必要的 Future/Action 依赖不对称。

> **实现约束：** Qwen3-VL 的视觉 token、文本 token 与扩展连续 token 具有不同的位置编码需求。额外的 Action/Future/State/Energy Tokens 需要正确的 position IDs、attention mask 和多模态位置编码适配；不能仅拼接投影后的 tensor 就假定与原模型实现完全兼容。



## 4.5 Energy-Based Optimization

给定 \(C_t\)，从候选先验采样：

\[
A_0\sim p_A,\qquad Z_0\sim p_Z.
\tag{21}
\]

第 \(k\) 次迭代计算：

\[
E_k=E_\theta(C_t,A_k,Z_k),
\tag{22}
\]

\[
g_A^{(k)}=\nabla_{A_k}E_k,
\qquad
g_Z^{(k)}=\nabla_{Z_k}E_k.
\tag{23}
\]

对两个变量分别进行梯度下降：

\[
\boxed{
\begin{aligned}
A_{k+1} &= A_k-\alpha_A g_A^{(k)},\\
Z_{k+1} &= Z_k-\alpha_Z g_Z^{(k)},
\end{aligned}
}
\quad k=0,\ldots,K-1.
\tag{24}
\]

其中 \(\alpha_A,\alpha_Z>0\) 为各自的优化步长，不能因为变量共享 Energy 就默认采用同一数值。

记更新算子为：

\[
(A_K,Z_K)=
\mathcal G_\theta^K(C_t,A_0,Z_0).
\tag{25}
\]

在实际训练中，为控制优化不稳定，可实验性使用梯度范数裁剪、步长退火、候选变量范围限制等措施；这些属于**数值稳定化设计**，不改变模型的核心目标。

**为何同一能量能产生两种更新？** 因为能量是对 \((A,Z)\) 的联合标量函数，两种梯度分别沿动作和未来变量方向指出局部能量下降方向，而不是由两个独立网络分别给出。

## 4.6 Stage I: Bidirectional Conditional Energy Learning

直接同时从随机 \(A_0,Z_0\) 优化具有较高学习难度。为构建合理的初始能量景观，第一阶段采用共享参数的双向条件学习。

### 4.6.1 Future-to-Action Optimization

固定真实未来潜变量 \(Z^*\)，仅优化动作：

\[
A_{k+1}=
A_k-\alpha_A
\nabla_AE_\theta(C_t,A_k,Z^*).
\tag{26}
\]

经过 \(K_A\) 步后得到 \(A_{K_A}\)。动作监督损失为：

\[
\mathcal L_A
=
J_A(A_{K_A},A^*).
\tag{27}
\]

作为基础实现，采用归一化动作空间上的均方误差：

\[
J_A(\widehat A,A^*)
=
\frac{1}{HD_a}
\|\widehat A-A^*\|_F^2.
\tag{28}
\]

对于夹爪等具有离散/特殊控制属性的维度，是否采用额外分类或加权损失，需要根据数据接口确定。

### 4.6.2 Action-to-Future Optimization

固定真实专家动作 \(A^*\)，仅优化未来潜变量：

\[
Z_{k+1}=
Z_k-\alpha_Z
\nabla_ZE_\theta(C_t,A^*,Z_k).
\tag{29}
\]

经过 \(K_Z\) 步后：

\[
\mathcal L_Z=
J_Z(Z_{K_Z},Z^*).
\tag{30}
\]

采用归一化或通道尺度校正后的 patch-wise MSE 作为基础形式：

\[
J_Z(\widehat Z,Z^*)
=
\frac{1}{3N_PD_D}
\|\widehat Z-Z^*\|_F^2,
\quad N_P=196,\;D_D=384.
\tag{31}
\]

如 MSE 难以保持 DINO 特征方向，可将特征余弦距离作为消融指标或辅助项，但不应在结果分析之前将其视为必要设计。

### 4.6.3 Bidirectional Objective

两种条件优化共享同一个 \(E_\theta\)，共同训练：

\[
\boxed{
\mathcal L_{\mathrm{bi}}
=
\lambda_A\mathcal L_A+\lambda_Z\mathcal L_Z.
}
\tag{32}
\]

\(\lambda_A,\lambda_Z\) 为损失权重。两种任务可以在同一 batch 上计算并求和，也可以交替采样训练；需控制各自更新次数和采样比例。

此阶段并不让 Energy Head 直接回归某个“正确能量标签”，监督的是**经过能量优化后的候选结果**。

## 4.7 Stage II: Joint Action–Future Energy Optimization

在第一阶段训练参数基础上，同时初始化动作与未来：

\[
(A_0,Z_0)\sim p_A\times p_Z.
\tag{33}
\]

使用式（24）执行 \(K\) 次联合更新，得到 \((A_K,Z_K)\)。对最终输出施加：

\[
\boxed{
\mathcal L_{\mathrm{joint}}
=
\beta_AJ_A(A_K,A^*)
+
\beta_ZJ_Z(Z_K,Z^*).
}
\tag{34}
\]

其中 \(\beta_A,\beta_Z\) 控制动作监督和未来潜变量监督的权重。默认采用两项按元素平均的损失，避免因 Future 的特征数量远大于 Action 而仅产生元素总数上的权重偏置；仍需监测其梯度尺度。

第一阶段旨在学习两种条件生成能力，第二阶段则检验这两种能力能否在**共同未知的候选动作和未来**上形成互相约束。

### 4.7.1 Differentiating Through the Inner Optimization

这里存在**内层与外层两层优化**：

- **内层：** 固定当前网络参数 \(\theta\)，通过 \(-\nabla_AE_\theta\)、\(-\nabla_ZE_\theta\) 更新候选变量。
- **外层：** 使用 \(\mathcal L_{\mathrm{bi}}\) 或 \(\mathcal L_{\mathrm{joint}}\) 更新模型参数 \(\theta\)。

由于最终候选 \(A_K,Z_K\) 依赖 \(\theta\)，外层训练需对展开的优化轨迹求导。例如：

\[
\frac{\partial\mathcal L_{\mathrm{joint}}}{\partial\theta}
=
\frac{\partial\mathcal L}{\partial A_K}
\frac{\partial A_K}{\partial\theta}
+
\frac{\partial\mathcal L}{\partial Z_K}
\frac{\partial Z_K}{\partial\theta}
\;(+\text{其他显式依赖项}).
\tag{35}
\]

因更新步骤包含 \(\nabla_{A,Z}E_\theta\)，精确展开式反向传播涉及混合二阶导数。PyTorch 实现中，训练期 `autograd.grad(..., create_graph=True)` 是关键条件；推理阶段仅需要候选变量梯度，无需构造高阶反向图。

这一设计带来显著显存开销，且依赖 Attention 内核是否支持相应高阶自动微分。实际训练应先验证一个小规模 batch 的 backward-over-backward 正确性，再选择可支持高阶梯度的内核或经过说明的近似梯度方案。

## 4.8 Initialization Priors

初始分布会影响有限步优化的可达区域。基础方案在**明确归一化后的变量空间**中使用标准高斯：

\[
\widetilde A_0\sim\mathcal N(0,I),
\qquad
\widetilde Z_0\sim\mathcal N(0,I).
\tag{36}
\]

### 4.8.1 Action Prior

对训练集专家动作逐维统计 \(\mu_A,\sigma_A\)，采用：

\[
\widetilde A=
\frac{A-\mu_A}{\sigma_A+\varepsilon}.
\tag{37}
\]

若标准化空间中 \(\widetilde A_0\sim\mathcal N(0,I)\)，则原始动作空间的等价形式为：

\[
A_0=\mu_A+\sigma_A\odot\epsilon_A,
\qquad
\epsilon_A\sim\mathcal N(0,I).
\tag{38}
\]

故“标准化空间的标准高斯”和“物理空间用训练集均值方差采样”可能只是同一方案的两种表示，**不能作为独立改进相互比较**。有效的初始化消融应在同一坐标系下改变尺度、协方差结构或条件信息。

### 4.8.2 Future Latent Prior

未来特征分布与普通高斯并不必然一致。可以使用训练集 Future Latent 的通道统计量归一化：

\[
\widetilde Z=
\frac{Z-\mu_Z}{\sigma_Z+\varepsilon}.
\tag{39}
\]

作为可选扩展，可尝试当前视觉状态附近的残差初始化：

\[
Z_0=
\underbrace{[z_t,z_t,z_t]}_{Z_{\mathrm{base}}}
+s_Z\,\sigma_{\Delta Z}\odot\epsilon_Z,
\qquad
\Delta Z^*=Z^*-Z_{\mathrm{base}}.
\tag{40}
\]

该方式利用短期未来与当前视觉状态的结构相似性，但在大位移、遮挡和接触变化任务中可能存在不利偏置，须与标准化高斯公平比较。

### 4.8.3 Optional Learned Prior

参考 LeaP [[8]](https://arxiv.org/abs/2606.17408)，还可训练根据机器人状态输出动作初始化参数的轻量网络：

\[
p_\phi(A_0\mid q_t)=
\mathcal N\left(
\mu_\phi(q_t),
\operatorname{diag}(\sigma_\phi^2(q_t))
\right).
\tag{41}
\]

该模块是**独立的可选扩展**，应在基础 LTE-WAM 结果稳定后再加入。不能直接依赖当前样本真实动作 \(A^*\) 构造推理初值，因为推理阶段不可获得 \(A^*\)。

## 4.9 Action–Future Coupling Analysis

共享能量函数并不自动证明 Action 与 Future 存在实质性耦合。若：

\[
E_\theta(C,A,Z)
=
E_A(C,A)+E_Z(C,Z),
\tag{42}
\]

则：

\[
\nabla_A E_\theta(C,A,Z)=\nabla_AE_A(C,A),
\quad
\nabla_Z E_\theta(C,A,Z)=\nabla_ZE_Z(C,Z).
\tag{43}
\]

此时即使 \(\mathcal L_A,\mathcal L_Z\) 都较低，Action 优化仍然不依赖 Future。

为此，本文拟采用两类诊断：

**(i) Conditional Gradient Sensitivity.** 在固定 \(C,A\) 时交换相容/不相容的 Future：

\[
S_{\mathrm{A}\leftarrow\mathrm{Z}}
=
\mathbb E
\left[
\frac{
\|\nabla_AE(C,A,Z_1)-\nabla_AE(C,A,Z_2)\|_2
}{
\|\nabla_AE(C,A,Z_1)\|_2+\epsilon
}
\right].
\tag{44}
\]

该量衡量 Future 变化对 Action 梯度的影响，但**非零并不等于物理正确性**，仍须结合真实转移结果解释。

**(ii) Transition Compatibility Ranking.** 构造相匹配的 \((C,A^*,Z^*)\) 及同条件下不匹配的 \((C,A^*,Z^-)\)，比较：

\[
E_\theta(C,A^*,Z^*)
\quad\text{vs.}\quad
E_\theta(C,A^*,Z^-).
\tag{45}
\]

如需要进一步显式约束，可在**可选消融**中引入：

\[
\mathcal L_{\mathrm{rank}}
=
\max\bigl(
0,m+
E_\theta(C,A^*,Z^*)
-
E_\theta(C,A^*,Z^-)
\bigr).
\tag{46}
\]

此项不是基础两阶段算法的必要组成部分。负样本应从相同或相近条件下、由已知状态转移判定为不匹配的轨迹中构造；不同演示未来并不天然都是错误负样本，必须避免产生伪负例。

## 4.10 Joint Inference and Receding-Horizon Execution

部署时仅提供 \((I_t,\ell,q_t)\)，编码当前条件 \(C_t\)，采样 \(A_0,Z_0\)，通过式（24）得到 \((A_K,Z_K)\)。执行动作输出：

\[
\widehat A_t=\operatorname{Denorm}(A_K).
\tag{47}
\]

模型生成的 \(\widehat Z_t=Z_K\) 是预期未来的潜变量表示，不自动对应可观看的 RGB 图像。系统执行动作块的前 \(h_{\mathrm{exec}}\le H\) 个控制步后获取新观测，进行滚动重规划：

\[
\operatorname{Observe}\rightarrow
\operatorname{Optimize}(A,Z)\rightarrow
\operatorname{Execute}\rightarrow
\operatorname{Observe}.
\tag{48}
\]

推理时需要计算对 \(A,Z\) 的梯度，但通常不更新网络参数；因此不应把完整推理包裹在禁用自动微分的上下文中。

### Algorithm 1: Bidirectional-to-Joint Training

```text
Input: demonstration dataset D = {(C, A*, Z*)}
       energy model E_theta; step counts K_A, K_Z, K
       loss weights lambda_A, lambda_Z, beta_A, beta_Z

Stage I: Bidirectional Conditional Learning
  for minibatch (C, A*, Z*) in D:
      sample A0 ~ p_A
      A_hat = A0
      repeat K_A times:
          A_hat = A_hat - alpha_A * grad_A E_theta(C, A_hat, Z*)

      sample Z0 ~ p_Z
      Z_hat = Z0
      repeat K_Z times:
          Z_hat = Z_hat - alpha_Z * grad_Z E_theta(C, A*, Z_hat)

      L_bi = lambda_A * J_A(A_hat, A*)
           + lambda_Z * J_Z(Z_hat, Z*)
      update theta by differentiating through the inner steps

Stage II: Joint Energy Optimization
  initialize theta from Stage I
  for minibatch (C, A*, Z*) in D:
      sample A0 ~ p_A and Z0 ~ p_Z
      (A_hat, Z_hat) = (A0, Z0)
      repeat K times:
          E = E_theta(C, A_hat, Z_hat)
          A_hat = A_hat - alpha_A * grad_A E
          Z_hat = Z_hat - alpha_Z * grad_Z E

      L_joint = beta_A * J_A(A_hat, A*)
              + beta_Z * J_Z(Z_hat, Z*)
      update theta by differentiating through the inner steps
```

> Algorithm 1 的两种变量更新都使用同一步计算的能量梯度；实现时应先求出两组梯度，再共同更新，避免意外变成依次使用已更新变量的异步优化。

### Algorithm 2: Energy-Based Action–Future Inference

```text
Input: current RGB I_t, language ell, robot state q_t
       trained E_theta; initialization priors p_A, p_Z; K

C_t = encode_condition(I_t, ell, q_t)
sample A0 ~ p_A, Z0 ~ p_Z
for k = 0, ..., K-1:
    E_k = E_theta(C_t, A_k, Z_k)
    g_A = grad_A E_k
    g_Z = grad_Z E_k
    A_(k+1) = A_k - alpha_A * g_A
    Z_(k+1) = Z_k - alpha_Z * g_Z

A_hat = enforce_valid_controls(denormalize(A_K))
execute first h_exec actions of A_hat
observe environment again and replan
Output: A_hat, Z_K
```

---

# 5. Implementation Details

## 5.1 Backbone and Token Dimensions

主模型参考 Qwen3-VL-4B-Instruct [[9]](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)，保留预训练视觉编码器与 36 层、2560 hidden dimension 的语言 Transformer；将原始文本生成 Head 替换为标量 Energy Head。DINOv3-ViT-S [[7]](https://arxiv.org/abs/2508.10104) 冻结，用于当前与未来视觉特征。Projector 与 Energy Head 初始化后参与训练；Qwen 的冻结/解冻策略作为工程配置和消融因素记录。

| Module | Input | Output |
|---|---|---|
| Qwen native visual-language path | \(I_t,\ell\) | \(N_{\mathrm{VL}}\times2560\) |
| Frozen DINOv3, current | \(I_t\) | \(196\times384\) |
| Current Projector | \(196\times384\) | \(196\times2560\) |
| State Projector | \(q_t\) | \(N_q\times2560\) |
| Frozen DINOv3, future | three future RGB frames | \(3\times196\times384\) |
| Future Projector | candidate \(Z\) | \(588\times2560\) |
| Action Projector | candidate \(A\) | \(H\times2560\) |
| Energy Transformer + Head | \([C,Z,A,e_E]\) | scalar per sample |

因此，不计 Qwen padding 的序列长度为：

\[
N_{\mathrm{seq}}
=
N_{\mathrm{VL}}+196+N_q+588+H+1.
\tag{49}
\]

若 \(H=16\)，则 \(N_{\mathrm{seq}}=N_{\mathrm{VL}}+N_q+801\)。

## 5.2 Data Pipeline and Alignment

实现基于 StarVLA [[10]](https://github.com/starVLA/starVLA)，并在其训练数据路径中扩展未来 RGB 的时间索引。每条演示轨迹需要获得严格对齐的：

\[
(I_t,\ell,q_t,A_t^*,I_{t+\tau_1},I_{t+\tau_2},I_{t+\tau_3}).
\tag{50}
\]

预处理需验证：

- 动作控制频率与相机采样频率对齐，确保一个 Action Chunk 所对应的三张未来图像取自**同一条 episode**；
- 对末尾不足 \(H\) 的序列显式使用有效掩码或丢弃，不以跨 episode 帧补齐；
- 保持当前与未来 RGB 使用一致的 DINO 预处理、摄像机定义与空间尺度；
- 动作归一化统计仅从训练集计算并固定，评测集不参与统计拟合；
- DINO GT 可预先离线提取以减少训练开销，但必须保持数据增强与视觉特征目标的一致性。

## 5.3 Training and Numerical Considerations

完整展开式训练重复执行 Transformer 前向及输入梯度计算，成本显著高于一次性 Action Head。建议工程验证按以下顺序进行：首先仅用小模型或少量 token 测试梯度与二阶梯度；然后以短 \(K\) 训练小规模数据；最后扩大规模并报告实际显存、吞吐与推理时延。

动作与 Future Latent 可能具有不同的尺度，因此需记录：

\[
\mathcal L_A,\ \mathcal L_Z,\
\|\nabla_A E\|,\ \|\nabla_Z E\|,\
E(C,A_k,Z_k).
\tag{51}
\]

如训练中出现梯度爆炸、更新越界或能量单调性异常，应优先检查归一化、Attention 高阶梯度与步长设计。**低 Energy 不必然意味着真实可执行的动作**，最终仍需由机器人控制器约束和环境成功率评测判断。

## 5.4 StarVLA Integration

工程上建议新增独立框架类，复用 StarVLA 数据读取、训练调度和 RoboCasa365 评测接口，同时实现：

- `encode_condition(...)`：构造 \(C_t\)；
- `energy(C, A, Z)`：返回每样本标量能量；
- `optimize_energy(...)`：支持 Action-only、Future-only、Joint 三种模式；
- `forward(...)`：按训练阶段构造展开损失；
- `predict_action(...)`：执行联合优化并输出合法动作。

代码组织与接口需以实际安装的 StarVLA 版本为准；以上是拟实现的功能边界，不表示上游仓库已存在这些方法。

---

# 6. Experiments

## 6.1 Research Questions

实验围绕以下研究问题展开：

- **RQ1:** 与传统 VLA/Joint WAM 相比，共享 Transition Energy 是否提高 RoboCasa365 操作成功率？
- **RQ2:** Future 建模是否通过实际 Action–Future 耦合改善动作生成，而非仅提供辅助预测监督？
- **RQ3:** “双向条件学习 \(\rightarrow\) 联合优化”是否优于单阶段联合学习或单向条件学习？
- **RQ4:** 初始化分布、优化步数与未来潜变量维度如何影响性能、收敛和计算效率？
- **RQ5:** 在未见厨房场景、物体配置及任务组合上，模型的转移相容性是否保持稳定？

## 6.2 Benchmark and Data

以 RoboCasa365 [[4]](https://robocasa.ai/) 作为主要实验平台，从已有演示轨迹构造 \((C,A^*,Z^*)\)，原则上不新增人工示范。优先选择覆盖不同接触与动态特征的任务：

- **Object Manipulation:** 抓取、搬运、放置；
- **Articulation:** 抽屉、柜门、开关等关节或铰接机构操作；
- **Appliance/State Control:** 设备状态切换与组合操作。

采用固定的数据版本、机器人控制器、图像源、动作定义和 train/validation/test 拆分。对 StarVLA 的 PandaOmron RoboCasa365 配方与 LDA-1B 的 RoboCasa-GR1 tabletop 配方进行严格区分，避免跨机器人和跨协议直接比较不兼容的 checkpoint。

## 6.3 Baselines

| ID | Method | Joint Future? | Scalar Energy? | Purpose |
|---|---|---:|---:|---|
| B1 | StarVLA-OFT（相同训练数据） | No | No | 常规直接动作预测 |
| B2 | StarVLA-PI / Flow-based Policy | No or model-specific | No | 生成式动作策略 |
| B3 | LDA-1B（同控制接口下的公平复现） | Yes | No | Joint WAM 基线 |
| B4 | Action-only Energy | No | Yes | 隔离 Energy 生成本身的贡献 |
| B5 | Bidirectional Energy | Yes | Yes | 双向条件学习 |
| B6 | Direct Joint Energy | Yes | Yes | 不经过预训练阶段的联合优化 |
| B7 | Bidirectional \(\rightarrow\) Joint（LTE-WAM） | Yes | Yes | 本文完整方法 |

**公平性原则：** 除明确研究模型规模差异的实验外，应尽可能匹配预训练初始化、训练数据、任务集合、图像视角、动作空间、优化预算和评估种子。跨架构比较时需同时提供参数量和实际计算量，而不能将全部收益归因于 Energy。

## 6.4 Evaluation Metrics

### 6.4.1 Closed-Loop Manipulation Success

在每个任务上运行多次仿真 episode，以任务定义的环境成功条件计算：

\[
\mathrm{SR}
=
\frac{\#\text{successful episodes}}
{\#\text{evaluation episodes}}\times100\%.
\tag{52}
\]

报告总体和不同任务类别的成功率、固定场景/未见场景表现及相应置信区间。任务加权和 episode 加权平均须明确区分。

### 6.4.2 Future Latent Prediction Quality

在离线保留集上报告：

\[
\operatorname{MSE}_Z
=
\frac{1}{3N_PD_D}\|Z_K-Z^*\|_F^2,
\tag{53}
\]

以及 patch-wise cosine similarity。若采用特征标准化，应同时说明度量是在归一化空间还是原始 DINO 空间计算。

### 6.4.3 Transition Compatibility

在保持当前条件一致的情况下构造正/负 Action–Future 配对，计算能量排序准确率、AUC 或成对排序间隔，记录条件交换后 Action 优化轨迹的变化。同时结合实际仿真状态转移确认正负配对的物理含义。

### 6.4.4 Efficiency and Stability

报告推理优化步数 \(K\)、单次 Energy 评估成本、每个 action chunk 的墙钟延迟、峰值 GPU 显存、训练速度、显存溢出与数值失败率。对于滚动执行，额外报告实际控制周期是否满足实时性要求。

## 6.5 Ablation Studies

### A. Training Strategy

比较 Future-to-Action only、Action-to-Future only、Direct Joint、Bidirectional only、Bidirectional \(\rightarrow\) Joint，验证两阶段训练的必要性。

### B. Future Modeling Contribution

比较 Action-only Energy、Action+Future 双向训练、Action+Future 联合训练；还可固定 Action 分支能力，替换相容 Future 为打乱配对 Future，检验 Action 梯度与闭环成功率是否发生符合假设的变化。

### C. Initialization Prior

将 Action 统一置于同一归一化空间，比较标准高斯、不同标准差的高斯及状态条件高斯（可选）；Future 统一置于可比较的特征尺度，比较标准高斯、经验特征统计先验和 Current-DINO 残差初始化。

### D. Optimization Steps

比较 \(K\in\{1,2,4,8\}\) 等候选设置，记录 Action/Future 误差、Energy 轨迹、任务成功率与额外计算开销。该集合为**建议搜索范围**，非预先确定的最优超参数。

### E. Model and Tokenization

比较 Future 时间点数量、Future token 压缩方式、Attention Token 顺序、DINO 是否冻结及 Qwen 冻结/微调方式。

### F. Explicit Compatibility Regularization (Optional)

在基础方法上加入式（46）的能量排序正则，验证它能否缓解可分离 Energy 的问题；该实验应与原始 LTE-WAM 分开报告。

## 6.6 Results Tables (To Be Filled)

**Table 1. RoboCasa365 Task Success Rates (%).**

| Method | Object Manip. | Articulation | State Control | Overall |
|---|---:|---:|---:|---:|
| StarVLA-OFT | TBD | TBD | TBD | TBD |
| StarVLA-PI | TBD | TBD | TBD | TBD |
| LDA-1B (matched protocol) | TBD | TBD | TBD | TBD |
| Action-only Energy | TBD | TBD | TBD | TBD |
| Direct Joint Energy | TBD | TBD | TBD | TBD |
| LTE-WAM (ours) | TBD | TBD | TBD | TBD |

**Table 2. Training Strategy and Transition Modeling.**

| Variant | Action Error ↓ | Future Error ↓ | Compatibility AUC ↑ | SR ↑ |
|---|---:|---:|---:|---:|
| Future-to-Action only | TBD | — | TBD | TBD |
| Action-to-Future only | — | TBD | TBD | — |
| Bidirectional only | TBD | TBD | TBD | TBD |
| Direct Joint | TBD | TBD | TBD | TBD |
| Bidirectional → Joint | TBD | TBD | TBD | TBD |

**Table 3. Optimization Efficiency.**

| Variant | K | Inference Latency (ms) ↓ | Peak VRAM (GB) ↓ | SR ↑ |
|---|---:|---:|---:|---:|
| LTE-WAM | 1 | TBD | TBD | TBD |
| LTE-WAM | 2 | TBD | TBD | TBD |
| LTE-WAM | 4 | TBD | TBD | TBD |
| LTE-WAM | 8 | TBD | TBD | TBD |

## 6.7 Qualitative Analysis

拟展示以下定性分析图：

- **Figure 1:** LTE-WAM Overall Architecture（Qwen + DINO + Candidate A/Z + Shared Energy）。
- **Figure 2:** Bidirectional Conditional Learning and Joint Optimization。
- **Figure 3:** 在同一抽屉操作案例中，不同 \(k\) 的候选动作、Energy 与对应 Future Latent 相似度变化。
- **Figure 4:** 正确与错误 Action–Future 配对的 Energy 分布。
- **Figure 5:** 成功/失败案例与长时序闭环重规划过程。

定性 Future 可通过真实未来图像作对照，并绘制 DINO patch 特征差异或检索结果；除非额外训练可验证的解码器，不应将 \(Z_K\) 直接宣称为可视化 RGB 预测图。

---

# 7. Discussion

## 7.1 What Would Demonstrate Useful Future Modeling?

证明方法有效不能只观察 \(\mathcal L_Z\) 降低。更强的证据需要满足三层条件：

1. 给定相同 \(C,A\)，改变 Future 候选会在合理方向上改变动作优化梯度；
2. 相匹配的 Action–Future 在能量或排序评估中优于不匹配配对；
3. 在相同 Backbone/数据/预算下，联合能量优化比 Action-only Energy 带来统计上有支持的任务性能收益。

上述条件分别检验**数学依赖、转移相容性和下游效用**；单独满足其中任一条都不足以证明模型具有一般化的物理因果理解能力。

## 7.2 Challenges and Limitations

**High-Dimensional Future Search.** 若使用 3 张 \(196\times384\) DINO patch 特征，Future 含有 225,792 个连续变量，远多于典型动作块。直接在全维空间优化可能导致成本高、梯度尺度不均或偏离真实 DINO 流形。

**Unrolled Second-Order Differentiation.** 完整展开训练的计算图需要保存多次 Transformer 及其输入梯度，限制 batch size 和优化步数；加速 Attention 实现也未必兼容高阶求导。

**Energy Identifiability.** 仅以优化终点误差监督时，能量的绝对值、尺度与全局排序不一定可解释；未来是否真正对动作产生约束必须额外评测。

**Multimodal Expert Behaviors.** 对单个示范 \(A^*,Z^*\) 使用 MSE，可能压制同样有效的多种动作—未来结果；初始分布和多起点优化可能影响多模态覆盖。

**Inference Efficiency.** 相比一次性回归模型，多次对大型 Qwen Transformer 求输入梯度可能显著增加时延。必须在闭环任务与实际机器人控制频率下评估收益与成本，而不能仅以 K 较小推断更快。

**Physical Feasibility.** 低能量不构成碰撞安全、接触可达或动力学可行性的形式化保证。部署仍需运动学、控制约束与安全检查。

## 7.3 Potential Extensions

若基础方法已显示可靠收益，可进一步研究：压缩 Future Latent、学习 Action 条件初始化先验、多随机起点搜索、能量排序正则、交替/预条件能量优化，以及利用真实仿真状态计算结构化转移一致性。但这些不应在首轮论文中与核心两阶段方法混为一个无法归因的复杂系统。

---

# 8. Conclusion

本文提出 LTE-WAM，一种基于共享 State–Action–Future 标量能量的世界动作模型研究方案。与采用独立生成网络直接输出动作及未来潜变量的路线不同，LTE-WAM 将动作预测、未来预测和联合生成统一为能量景观上的迭代优化过程。方法通过 Qwen3-VL 与 DINOv3 构造视觉语言和空间特征，采用双向条件能量学习建立两类条件优化能力，再通过联合能量优化使 Action 和 Future 在同一候选更新过程中交互。本文还明确给出了初始化先验、展开式训练、转移相容性诊断及 RoboCasa365 评测协议。该方法是否在操作成功率、未来建模质量、转移一致性和推理效率上优于现有基线，需要由计划中的实验确定。

---

# References

[1] Lyu, J., et al. **LDA-1B: Scaling Latent Dynamics Action Model via Universal Embodied Data Ingestion.** RSS 2026 / arXiv:2602.12215. [https://arxiv.org/abs/2602.12215](https://arxiv.org/abs/2602.12215)  
Code: [https://github.com/jiangranlv/LDA-1B](https://github.com/jiangranlv/LDA-1B)

[2] Gladstone, A., et al. **Energy-Based Transformers are Scalable Learners and Thinkers.** ICLR 2026 / arXiv:2507.02092. [https://arxiv.org/abs/2507.02092](https://arxiv.org/abs/2507.02092)

[3] Davies, T., et al. **EBT-Policy: Energy Unlocks Emergent Physical Reasoning Capabilities.** arXiv:2510.27545, 2025. [https://arxiv.org/abs/2510.27545](https://arxiv.org/abs/2510.27545)

[4] Nasiriany, S., Nasiriany, S., Maddukuri, A., and Zhu, Y. **RoboCasa365: A Large-Scale Simulation Framework for Training and Benchmarking Generalist Robots.** ICLR 2026. [https://robocasa.ai/](https://robocasa.ai/)

[5] Chi, C., et al. **Diffusion Policy: Visuomotor Policy Learning via Action Diffusion.** International Journal of Robotics Research / arXiv:2303.04137. [https://arxiv.org/abs/2303.04137](https://arxiv.org/abs/2303.04137)

[6] Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M., and Le, M. **Flow Matching for Generative Modeling.** ICLR 2023 / arXiv:2210.02747. [https://arxiv.org/abs/2210.02747](https://arxiv.org/abs/2210.02747)

[7] Siméoni, O., et al. **DINOv3.** arXiv:2508.10104, 2025. [https://arxiv.org/abs/2508.10104](https://arxiv.org/abs/2508.10104)

[8] Dai, M., et al. **Where Should Action Generation Begin? A Learnable Source Prior for Generative Robot Policies.** arXiv:2606.17408, 2026. [https://arxiv.org/abs/2606.17408](https://arxiv.org/abs/2606.17408)

[9] Qwen Team. **Qwen3-VL-4B-Instruct Model Configuration.** [https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)

[10] StarVLA Contributors. **StarVLA: A Lego-like Codebase for Vision-Language-Action Model Developing.** [https://github.com/starVLA/starVLA](https://github.com/starVLA/starVLA)

---

# Appendix A. Symbols and Shapes

| Symbol | Description | Typical Shape |
|---|---|---|
| \(I_t\) | Current RGB | \(3\times224\times224\) |
| \(\ell\) | Language instruction | text |
| \(q_t\) | Robot state | \(D_q\) |
| \(A^*\) | Expert action chunk | \(H\times D_a\) |
| \(A_k\) | Candidate action at optimization step \(k\) | \(H\times D_a\) |
| \(z_t\) | Current DINO patch features | \(196\times384\) |
| \(Z^*\) | Future DINO GT (3 timestamps) | \(3\times196\times384\) |
| \(Z_k\) | Candidate future latent | \(3\times196\times384\) |
| \(C_t\) | Combined condition tokens | \((N_{\mathrm{VL}}+196+N_q)\times2560\) |
| \(e_E\) | Energy readout token | \(2560\) |
| \(E_\theta\) | Scalar energy | \(1\) |
| \(K\) | Number of energy optimization steps | scalar integer |

# Appendix B. Minimal Reproduction Checklist

- [ ] 选择并冻结实际使用的 RoboCasa365 数据版本、机器人、controller 和任务集合。
- [ ] 实现严格对齐的 \(I_t,\ell,q_t,A^*,I_{t+\tau_1:t+\tau_3}\) 数据抽样。
- [ ] 固定 DINO patch token 的预处理、数量与提取接口。
- [ ] 验证 Qwen multimodal token 构造、position IDs 与 mask 的正确性。
- [ ] 验证 \(E(C,A,Z)\) 输出形状为每样本标量，且对 \(A,Z\) 的梯度存在。
- [ ] 验证高阶梯度能从展开损失传回 Energy Head 和 Projector。
- [ ] 先完成 Future-to-Action / Action-to-Future 小样本过拟合测试。
- [ ] 再完成 Joint Energy Optimization 与收敛轨迹分析。
- [ ] 在统一实验协议下跑 Action-only、Direct Joint 与两阶段 LTE-WAM。
- [ ] 输出真实 Success Rate、Future Latent 指标、Compatibility 排序和推理成本。
- [ ] 在未见场景、不同初始噪声种子及不同 \(K\) 下重复实验。
- [ ] 仅在取得实验结果后撰写最终 Results、Abstract 数值与 Contributions 结论。

---

**写作边界说明：** 本文将原研究计划中的共享 Energy、Qwen+DINO 双视觉编码、双向训练、联合优化与 RoboCasa365 评测作为核心方案；将相容性排序正则、学习型初始化先验、Current-DINO 残差初始化及注意力顺序消融明确标记为建议的可选改进。所有数值、优势与理论解释均应接受实验检验；目前没有捏造任何实验结果。

# LTE-WAM 项目交接与异机复现指南

更新：2026-11-06，北京时间。本文以交接时的代码、冻结配置和实际日志为准。运行进度是快照，后续状态以日志为准。本文是当前入口；旧README、论文草稿和较早的训练计划中存在历史架构/参数，不能直接作为当前运行命令。

## 1. 接手后先了解什么

这是一个基于Qwen3-VL-4B语言主干的机器人标量能量模型研究实现，正在RoboCasa365 target/human数据上做Stage1双向条件预测。当前代码已真实运行4B、K=2完整展开求导、双卡层切分、有效batch16/native microbatch4；仍属于研究实验，不宣称已经获得机器人任务成功率或最终最优超参数。

**当前主要问题**：模型能否同时改善Action和Future的固定训练及验证误差；小样本上能拟合，不代表扩大数据后能泛化。Future损失权重9只是历史尺度补偿，正在与权重1对照。

接手建议按以下顺序：

1. 阅读本文第2～6节，确认架构、数据与当前参数。
2. 在自己的机器准备权重和相同数据视图，安装环境。
3. 用`prepare_remote_run.py`生成本机配置；它只准备和核验，不启动训练。
4. 先运行独立目录中的2步profile，确认能运行并监测CPU/GPU资源。
5. 再跑`tiny-weight1`或`tiny-weight9`的50更新实验；最后才扩大到三任务。
6. 不要直接运行源服务器的切换/清理/supervisor脚本，它们可能包含本机绝对路径和旧进程名称。

## 2. 仓库包含与不包含什么

包含：`energywam/`算法实现、`scripts/`、配置、测试、本文、历史实验说明、固定样本清单、轨迹划分ID、训练集标准化统计、小型环境/结果摘要。

不包含：原始数据、视频、parquet、仿真资产、Qwen/DINO模型权重、训练checkpoint、虚拟环境、完整日志目录、W&B个人配置/API Key、SSH私钥。

固定样本清单和标准化统计是复现所需的小型元数据，不含图像或完整示范轨迹。统计来自TRAIN；TEST不用于反复调参。

| 目录/文件 | 用途 |
|---|---|
| `energywam/models/energy_wam.py` | 数据准备、条件构建、模型顶层 |
| `energywam/models/energy_core.py` | Token投影、能量头、二次项、共享能量 |
| `energywam/backbones/qwen3vl_adapter.py` | Qwen条件与语言主干 |
| `energywam/backbones/dino_v3.py` | 冻结DINOv3 |
| `energywam/energy/solver.py` | 内层能量梯度更新、完整展开求导 |
| `energywam/energy/iteration.py` | 迭代进度编码、可学习有界步长 |
| `energywam/training/stages.py` | BI两个条件方向及其他阶段 |
| `energywam/training/shared_microbatch.py` | 单份模型、原生microbatch及梯度累积 |
| `energywam/training/parallel.py` | 语言层跨卡切分 |
| `energywam/training/run_state.py` | 采样位置、rolling latest、恢复时记录回退 |
| `energywam/training/best_checkpoint.py` | 独立best选择与保留 |
| `scripts/train_shared_sharded.py` | 当前两卡Stage1实际训练入口 |
| `scripts/run_iteration_search.py` | 有监测、空间检查、数值保护的队列入口 |
| `scripts/prepare_remote_run.py` | 为新机器生成路径配置与单实验队列 |
| `scripts/k_training_probe.py` | 固定窗口条件评估与冻结特征缓存 |
| `manifests/experiments/` | 本次交接的冻结样本清单 |
| `stats/trainmix_150k_paper_standardized_v1.json` | 固定TRAIN标准化统计 |
| `docs/reproducibility/` | 源环境、数据元信息指纹、进度与证据摘要 |
| `requirements.txt` | 当前非PyTorch训练依赖 |
| `requirements-dev.txt` | 加上测试依赖 |
| `requirements-wandb.txt` | 可选W&B采集环境依赖 |

## 3. 当前模型架构

### 3.1 输入与Token

当前条件`C`由三部分组成：

- 指令文本 → Qwen token embedding。
- 当前RGB → 冻结DINOv3 ViT-S/16 → 可训练current-DINO projector。
- 原始16维机器人状态 → `[sin(q), cos(q)]`32维 → 可训练state projector。

当前`condition_mode: text_dino_state`**移除了Qwen原生视觉ViT/merger路径**，不再同时使用两套当前RGB编码。构造条件时先做embedding/投影；随后条件仍会与候选一起进入语言Transformer，不能理解成条件永远不经过主干。

候选Future原始形状为`[B,3,1,196,384]`，Action为`[B,16,12]`。当前`dino_patch_pack_factor: 2`把空间2×2 patch拼接后投影，每帧196个patch变49个Token，**不把Future监督目标平均成49个patch**，原始MSE仍在196×384坐标计算。

序列顺序为：

```text
[C, Future tokens, Action tokens, Energy token]
                   ↓
       Qwen语言Transformer（36层，hidden=2560）
                   ↓
          Energy token的最终hidden state
                   ↓
     LayerNorm → Linear(2560,256) → GELU → Linear(256,1)
                   ↓
               每样本一个标量能量
```

原`lm_head`不参与词表logits计算。DINO冻结；语言主干、文本embedding、各投影、位置/类型编码、迭代MLP、Energy token和Energy head按当前配置训练。能量头最后一层权重/偏置为0初始化，不是冻结。

### 3.2 迭代条件

每次能量评估用`progress=k/K`；固定sin/cos PE（dim256，scale1000）经过两个独立可训练MLP，分别加到Action/Future tokens。不新增序列位置。它与Action内部16个时间步、Future偏移5/11/16是不同概念。

K=2时进度为0、0.5。每次循环重新从当前候选构造Token，不累加上一轮的迭代embedding。每次计算能量都会经过一次语言主干，K次不是只经过主干一次。

### 3.3 总能量与初始化

当前总能量：

\[
E=E_{NN}(C,A,Z,s)+\frac{\|A\|^2}{2\alpha_{A,0}}+\frac{\|Z-Z_{current}\|^2}{2\alpha_{Z,0}}.
\]

- `A`处于标准化动作坐标；二次项中心0对应训练动作均值，不一定是物理静止指令。
- `Z_current`仅由当前观测DINO生成，用同一套TRAIN Future统计标准化；绝不使用未来真值作中心。
- 分母是固定初始步长锚点，**不随可学习步长改变**。
- Action候选来自标准化`N(0,I)`，不是恒定均值向量。
- Future候选为当前观测标准化DINO，`init_future_noise_std=0`，复制到3个未来时刻。它不是上一次预测Future的热启动，也不是复制真实未来。

初始神经网络能量为0时，二次项使Action一步收缩到标准化0，Future保持当前画面。因此未训练基线本身已有先验优势，不得把这个优势算作训练收益。

## 4. 当前训练算法与batch含义

### 4.1 Stage1双向条件训练

同一组样本、同一份模型参数，依次计算：

1. 固定真实Future，K次修改Action候选，得到Action MSE。
2. 固定真实Action，K次修改Future候选，得到Future MSE。
3. 两方向梯度累积后，外层优化器只更新一次模型权重。

当前目标是`L=Action MSE + lambda_future * Future MSE`，主组三任务的`lambda_future=9`；小样本对照分别为1与9。权重乘在损失上，不等于实际梯度贡献正好变成9倍。9来自旧16样本基础MSE约0.9469/0.10514的比值，未证明最优。

训练采用`E.sum()`对候选求导，避免内层更新量随batch被额外除小；外层损失再按样本平均。内层`create_graph=True`、full unroll，外层MSE通过整个内层过程反向更新模型参数。推理用候选梯度但不更新模型权重。

\[
A_{k+1}=A_k-\alpha_A(\nabla_AE_{NN}+A_k/\alpha_{A,0}),
\]
\[
Z_{k+1}=Z_k-\alpha_Z(\nabla_ZE_{NN}+(Z_k-Z_{current})/\alpha_{Z,0}).
\]

能量没有独立的“真实能量标签”；模型学习怎样的能量梯度能改善最终候选MSE。加入迭代条件后，跨不同进度的能量值并非同一个静态函数值，不能直接把能量单调下降当成功标准。

### 4.2 当前超参数

| 参数 | 当前值 |
|---|---|
| Stage | `bi`，同batch双方向，Stage2未启动 |
| 训练/验证K | 2 / 2 |
| Action/Future初始内层步长 | 5 / 2000 |
| 步长 | 两个全局可学习标量，各内层迭代共享；有界为初值0.5～1.5倍 |
| 外层优化器 | AdamW8bit；步长参数组32bit状态 |
| 主干LR | 1e-5 |
| 投影等任务模块LR | 1e-4 |
| Energy head LR | 1e-3 |
| 步长参数LR | 1e-3 |
| warmup | 16次外层更新=256样本，之后保持恒定；不是余弦退火 |
| 外层梯度裁剪 | global norm 1 |
| 内层候选梯度裁剪/值clamp | 当前均关闭 |
| 计算精度 | highest FP32，TF32关闭；eager attention |
| gradient checkpointing | 开启，`use_reentrant=False` |
| 有效batch | 16 |
| 同时计算microbatch | 4 |
| 模型放置 | 每个实验1份模型切到2卡，语言层18/18；不是DDP |
| 激活卸载 | 保存张量可卸载CPU，去重开启；GPU预算10/14 GiB、host cache128 GiB |
| 标准化 | action/future zscore；状态sincos；TRAIN固定统计 |

`per_device_batch_size:16`在这个自定义入口里是**整份共享模型的有效batch16**，不能按普通DDP乘以两卡。当次先处理4个样本并反向，释放图、保留参数梯度，累计4组并覆盖两个方向后更新一次。一次update=一次外层optimizer step；K次是候选更新，不能混为一谈。

完整batch16输入/目标还会占CPU内存，但不会同时保留16个样本的全部展开图。CPU大头通常是高阶反向保存的中间结果与部分梯度暂存；模型和主要优化器状态仍在GPU。

## 5. 数据、动作监督与划分

### 5.1 外部数据与本机位置

使用RoboCasa365的50个target任务/human示范，在本机生成统一LeRobot视图。不是全部365任务，也不是已提交官方成功率评测。

| 资源 | 源服务器路径（仅说明，不在Git中） |
|---|---|
| 原始任务树 | `/cloud/zzc/energyWAM/datasets/robocasa365/v1.0/target` |
| 统一视图 | `/datasets/robocasa365_trainmix/robocasa365_target_50tasks` |
| 云盘真实数据根 | `/cloud/zzc/energyWAM/datasets` |
| Qwen目录 | `/datasets/models/Qwen3-VL-4B-Instruct`（权重实际引用平台`/model`） |
| DINO目录 | `/datasets/models/dinov3-vits16-pretrain-lvd1689m` |

`/datasets`下多个目录为软链接。跨机器复制软链接本身不会复制其目标，师兄必须下载/挂载自己的真实数据和完整权重目录。

外部来源：

- 数据与仿真：[RoboCasa官方仓库](https://github.com/robocasa/robocasa)、[官方文档](https://robocasa.ai/docs/build/html/introduction/overview.html)。具体50个任务/日期固定在`build_trainmix_view.py:EXPECTED`；较新数据修订不自动等价。
- Qwen：[Qwen/Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)，需要完整模型及tokenizer/processor/config文件。
- DINO：[facebook/dinov3-vits16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m)，需要完整HF目录；获取权限按模型页处理，token保留在各自机器。

当前离线示范训练不要求启动MuJoCo或装全套仿真资产；真实闭环评估需要另行安装官方仿真环境与执行接口。

### 5.2 数据视图与固定编号

```text
<DATA_PARENT>/robocasa365_target_50tasks/
  meta/info.json
  meta/modality.json
  meta/stats.json
  meta/episodes.jsonl
  data/chunk-000/episode_000000.parquet
  videos/chunk-000/observation.images.robot0_agentview_left/episode_000000.mp4
  ...（官方源还有另外两相机；当前模型只读left）
```

可从相同官方任务树生成新视图（输出必须为新目录，脚本不负责覆盖既有symlink）：

```bash
python -I scripts/build_trainmix_view.py \
  --src /YOUR_DATA/robocasa365/v1.0/target \
  --out-root /YOUR_DATA/robocasa365_trainmix \
  --out-name robocasa365_target_50tasks \
  --manifests /YOUR_OUTPUT/generated_global_manifests
```

不要只拿三个任务重新从0编号后继续套仓库中的样本清单。全局episode_index来自固定50任务顺序，必须与`docs/reproducibility/dataset_layout_fingerprints.json`中的`episodes.jsonl`等指纹匹配。新准备入口会检查，发现不一致就拒绝，而不是静默训练错样本。

原50任务按源任务内`source_episode_index %20`划分：0为TEST、1为VAL，其余TRAIN。总计TRAIN22,725、VAL1,285、TEST1,297条轨迹。轨迹是一次完整示范，不是一个图像帧；同一轨迹不会跨集合。当前实验只读取TRAIN/VAL。

### 5.3 窗口、监督和标准化

一个窗口是从某条轨迹的起点t生成一个训练样本：当前图像/文本/状态、从t开始的16步真实动作，以及t+5、t+11、t+16三帧的DINO特征。最后一帧是动作块末端之后的观测。必须满足完整horizon，不从别的轨迹补帧。

动作监督来自parquet的12维`action`控制指令，不是16维状态或绝对关节位置。按本数据`modality.json`，12维分段为：base_motion[0:4]、control_mode[4:5]、end_effector_position[5:8]、end_effector_rotation[8:11]、gripper_close[11:12]。具体物理执行尺度/坐标系必须结合控制器，不能把所有维度统称位置delta。原始16维是`observation.state`，当前统一做sincos32维。

Future监督由冻结DINO从真实未来图像在线提取并detach；当前帧特征也用相同预处理。MSE按全部有效元素平均；不能横向把Action0.6与Future0.1当同一物理单位。

标准化文件`stats/trainmix_150k_paper_standardized_v1.json`来自50任务TRAIN中的固定150k窗口统计，**不是三任务重新拟合、不是全量所有窗口**。指纹：

```text
ad89d0de59bb62bf374612844a09e9687ef4b1e2cd9cca4c4342f870ef37f25a
```

保持这套统计才能复核历史MSE；若另拟统计，要另建实验并记录坐标，不能直接比较原数值。

## 6. 正在进行的两类实验与历史证据

### 6.1 三任务完整轨迹池实验

冻结清单：`manifests/experiments/three_task_full_20261106/`。

| 任务 | TRAIN轨迹 | TRAIN窗口 | VAL轨迹 | VAL窗口 |
|---|---:|---:|---:|---:|
| OpenDrawer | 462 | 7,944 | 26 | 104 |
| PickPlaceCounterToCabinet | 450 | 7,573 | 26 | 104 |
| WashFruitColander | 455 | 29,251 | 26 | 104 |
| 合计 | 1,367 | 44,768 | 78 | 312 |

每条TRAIN轨迹按stride16取起点，并补上最后合法起点，打乱后完整访问一轮44,768窗口=2,798次更新。全部轨迹参与不等于遍历所有逐帧重叠起点。自然按窗口采样，洗水果约占65.3%，不是任务均衡。

GPU01的seed42继续运行；GPU23的seed43已按用户要求在**121次更新**安全暂停，完整latest121和best0保留。Git不带这些权重。最新进度快照在`docs/reproducibility/handoff_runtime_snapshot.json`，它不是实时状态。

每4,096样本/256更新做312固定VAL；每1,024样本/64更新检查12固定TRAIN。初始VAL Action约0.763883、Future约0.143725；判定学习应与自身初始和当前图像复制基线比较。Future初值低是短时当前特征复制优势，不是已经学会未来。

### 6.2 当前小样本权重1/9对照

冻结清单：`manifests/experiments/tiny_future_weight_20261106/`。复用旧OpenDrawer/OpenCabinet两条TRAIN轨迹的16窗口，另四条VAL轨迹16窗口；不是当前三个任务全集。

GPU23顺序跑权重1、9，两组seed42、相同新参数初始化和样本顺序，B16/native4/K2、warmup16，只有`loss.lambda_future`不同。每组50更新=800次样本访问=16样本反复50遍。每10更新记录固定TRAIN/VAL；先不自动延长预算。

主比较：同样第50次更新的**未加权Action/Future MSE**和各自曲线。不直接比较不同定义的总loss，也不拿两个不同加权best分数当统一赢家。两组依各自预声明验证目标保留latest/best；主结论用相同终点。50次可能不足分出优劣，不能把短预算无改善写成永远学不会。

旧B2实验“16样本反复50遍”是400更新；新B16是50更新。样本访问量相同，但更新次数不同，不能混淆。

### 6.3 重要历史结果与限制

- 旧双二次项/当前特征初始化/pack2，B2，16固定训练样本400更新：TRAIN Action0.946895→0.499622、Future0.105138→0.066373；VAL Action0.926307→0.847006、Future0.140071→0.164452。说明能拟合，Future验证变差；不是两方向都泛化成功。
- 旧同条件head LR对照：1e-2在20更新严重失稳；1e-3在160更新更快降低训练两误差；3e-4在160更新改善较小。但验证A/F有取舍，未证明1e-3普适最优。
- 十任务固定800TRAIN/100VAL，FP32、native4、相同4096样本预算：B16终点VAL约1.200696/0.144192，B64约1.222096/0.149266；B8末期回退；B32在104更新因固定TRAIN误差>10倍保护终止，不是OOM。单seed、有限LR复核，不是全局最优batch证明。
- 旧K=2/4/8/16/32推理诊断有Action边际收益递减，不能当作完成相同K训练对照；当前K2是实用基线。
- 旧native8成功、native16失败属于更早精度/状态设置，不能证明当前FP32最大同时batch就是8。当前native4有单独梯度等价检查依据。
- 迭代编码、双二次项、可学习步长各有实现/小样本证据，但未完成所有模块在当前三任务上的独立消融。

完整历史分析见`LTE_WAM_MODEL_AND_EXPERIMENT_SUMMARY_CN.md`及`reproducibility/experiment_evidence_summary.json`。源服务器完整日志仍在云盘，不随Git分发。

## 7. 环境安装与硬件预算

### 7.1 实测环境

`docs/reproducibility/source_environment_20261106.json`记录实际训练解释器的版本，而旧`requirements-legacy-20261031.txt`只是早期参考。

源环境：Python3.12.11、torch2.13.0+cu132、torchvision0.28.0+cu132、transformers4.57.0、bitsandbytes0.50.2；其他依赖以JSON和当前requirements为准。源PyTorch可能来自平台CUDA构建，**不保证在默认PyPI索引能安装相同带CUDA后缀wheel**。优先使用对应平台镜像/索引中的相同构建；若换版本，需重新跑求导、native梯度等价与GPU smoke，不能宣称逐位复现。

示例安装流程（先按目标机器平台安装正确PyTorch/torchvision配对）：

```bash
conda create -n energywam python=3.12 -y
conda activate energywam
# 在这里按目标平台/驱动安装兼容的torch和torchvision CUDA构建。
python -c 'import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.version.cuda, torch.cuda.is_available())'
python -m pip install -r requirements-dev.txt
# 可选W&B，离线训练不依赖它。
python -m pip install -r requirements-wandb.txt
```

当前入口直接从仓库根目录运行，不需要`pip install -e .`，仓库没有必须的打包安装步骤。标准离线示范训练不要求安装旧StarVLA仓库。

### 7.2 资源预算

- 当前配置按两张**各约48GiB** GPU实测，并非标准24GiB4090配置。不能只看“4090”型号推断容量。
- 一份模型两卡，峰值通常每卡约44～46GiB，余量有限。
- 进程CPU USS常见约65～80GiB；容器总内存还包含其他进程、共享内存和文件缓存。单任务建议至少128GiB可用RAM，140GiB是本次进程保护阈值，不是保证需求上限。
- 当前完整FP32 checkpoint约22.88GiB；单组latest/best/临时写入及余量保守约73GiB。不要只预留一份权重大小。
- 同一代码仓库的checkpoint写锁会串行大文件写入；跨两个不同clone没有共享该锁，需要自己预算并行临时写入。
- CPU卸载会降低GPU持续利用率，显存越满不代表越快。比较速度要用预热后每秒样本数，并记录验证/保存开销。

## 8. 在新机器生成并运行配置

### 8.1 clone并核验代码

```bash
git clone https://github.com/Tassel51/energyWAM.git
cd energyWAM
git rev-parse HEAD
```

如果仓库是私有的，由仓库拥有者为师兄账号开访问权限；各自在自己的机器配置Git认证，密钥不写到配置和仓库。

### 8.2 CPU检查，不占训练GPU

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
python -m pytest -q \
  tests/test_solver_analytic.py tests/test_iteration_aware.py \
  tests/test_native_microbatch.py tests/test_shared_microbatch.py \
  tests/test_best_checkpoint.py tests/test_run_state.py \
  tests/test_iteration_search_guard.py tests/test_checkpoint_space_budget.py \
  tests/test_resource_monitor.py tests/test_resource_failure_classification.py \
  tests/test_remote_run_preparation.py
```

部分真实小Qwen测试需要本地processor fixture，它们的依赖路径在`tests/real_fixtures.py`；若目标机器未提供模型缓存，应明确区分skip/缺资源和通过，不能只报总数。本文交接检查使用源机器环境，不等于目标机器已验证。

### 8.3 准备2步GPU smoke

下例只写本机配置并检查资产，不会自动启动训练。替换三个外部资源路径和输出目录：

```bash
python scripts/prepare_remote_run.py \
  --experiment tiny-weight1 \
  --data-root /YOUR_DATA/robocasa365_trainmix \
  --qwen /YOUR_MODELS/Qwen3-VL-4B-Instruct \
  --dino /YOUR_MODELS/dinov3-vits16-pretrain-lvd1689m \
  --output-dir /YOUR_OUTPUT/energywam_smoke \
  --physical-gpus 0,1 \
  --profile-steps 2
```

然后运行它打印的`Launch`命令。也可以手动：

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -u scripts/run_iteration_search.py \
  --jobs /YOUR_OUTPUT/energywam_smoke/jobs.json \
  --status /YOUR_OUTPUT/energywam_smoke/queue_status.json
```

profile跳过验证和完整断点，只验证前后向、资源与大致速度；不能把它当作完整保存恢复测试。它会写少量参数采样。后续正式训练必须换新输出目录。

### 8.4 跑完整小样本对照

先准备权重1，**不带profile参数**：

```bash
python scripts/prepare_remote_run.py \
  --experiment tiny-weight1 \
  --data-root /YOUR_DATA/robocasa365_trainmix \
  --qwen /YOUR_MODELS/Qwen3-VL-4B-Instruct \
  --dino /YOUR_MODELS/dinov3-vits16-pretrain-lvd1689m \
  --output-dir /YOUR_OUTPUT/tiny_weight1_seed42 \
  --physical-gpus 0,1
```

运行打印的Launch命令，完成50更新后，用`--experiment tiny-weight9`、新输出目录重复。若同时有两对空闲卡，可把第二组用`--physical-gpus 2,3`；空间预算必须覆盖两组。保持seed42和其他参数相同；不从第一组权重启动第二组。

准备器把模型、数据、统计和输出重定向到本机。检查`local_config.yaml`、`jobs.json`、`launch_plan.json`后运行。`--skip-path-check`只用于无资产时预生成配置，不能当作资产校验通过。

队列只在指定两卡无其他计算进程时启动；它使用当前Python解释器，不依赖源服务器的conda绝对路径。不要同时手动启动相同输出目录的另一个队列。

### 8.5 扩大到三任务

把上面的experiment改成`three-task`、使用新的输出目录；默认2798更新、44,768窗口一轮。其他默认参数与当前seed42主组相同。默认seed42，若做独立复核可`--seed 43`，必须新输出目录。

三任务TRAIN流式读取，只缓存312固定VAL和12固定TRAIN monitor，不把44,768样本全量DINO特征放RAM；小样本允许缓存，因为编码器冻结。

本机可查看当前脚本支持的GPU对：`0,1`或`2,3`。可见卡内部均映射为cuda0/1。更任意的物理编号需要显式扩展CLI并验证，不能只改`language_model_devices`；不要用默认torchrun/DDP方式启动这个自定义入口。

### 8.6 后台运行

目标机器不一定有supervisor，可以在tmux内运行队列命令，终端断开后训练继续。源服务器supervisor管理方式见第12节，只是本机状态，不应原样安装到别人的服务器。

直接运行`train_shared_sharded.py`也可以，但不会自动带上队列的CPU/GPU资源监测和固定MSE保护。复现时优先使用生成的队列命令。

## 9. 训练日志、验证、best与恢复

### 9.1 关键文件

| 输出文件 | 含义 |
|---|---|
| `console.log` | 加载、更新、报错 |
| `train_log.jsonl` | 每次更新损失、梯度、步长、实际各组LR |
| `timing_history.jsonl` | 含数据准备的端到端步耗时 |
| `speed_report.json` | 最新完成更新、最近最多100步速度 |
| `eval_XXXX_action.json/future.json` | 固定VAL条件评估，含每个窗口/任务 |
| `train_monitor/eval_XXXX_*.json` | 固定TRAIN样本评估 |
| `best_selection.json` | best步数、验证分量及预声明标准 |
| `ckpt_latest.pt`和sidecar | 最新完整可恢复断点 |
| `ckpt_best.pt`和sidecar | 验证选出的完整best，通常是软链接 |
| `resource_trace.jsonl/resource_summary.json` | 进程USS/RSS、GPU、容器与退出分类 |
| `initialization_step0.json` | 初始参数抽样哈希；不是完整权重哈希 |
| `source_initial.zip` | 每次训练开始保存的源代码/配置/样本清单快照 |
| `resume_history/` | 断点后活动记录回退时的归档，不删除旧证据 |

`global_step`在train_log中记录更新前编号（0-based）；`speed_report.completed_steps`和评估文件名是完成次数。比较时别差一。

固定验证不更新权重，恢复模型train/eval状态、requires_grad与RNG。它分别评估“给真实Future预测Action”“给真实Action预测Future”。**这不是联合生成或机器人成功率**；需要另做联合推理/仿真测试。

初始低Future误差应与“复制当前图像特征”基线比较；不能因为小于Action就说Future学好了。每batch样本不同，曲线抖动不等于没有学习，优先看固定TRAIN/VAL与逐任务指标。

### 9.2 best规则与保留政策

当前best最小化固定VAL的`lambda_action*A_MSE + lambda_future*F_MSE`，K2、同固定样本与初始化协议。两个方向单独保留，不能把最后一步或最低训练loss称为best。加权分数改善可能掩盖其中一个方向变差，结论需分开看。

每组保存完整latest和best，包含模型、优化器、调度器、RNG、配置与标准化统计。新latest写完、mmap可读校验后原子替换；best用独立generation和原子指针，旧generation随后清理。best=latest时可硬链接共享物理文件。拷贝软链接时需带上`.best_current/.best_snapshots`目标；只拷一个`ckpt_best.pt`链接会损坏。

定时保存按`checkpoint_interval_seconds=1800`在更新边界检查，另有final和best改善触发；启用秒间隔时`save_interval`并非同时额外每N步保存。所谓半小时不是强制中断一次耗时很长的update。

不得自动删除latest/best、数据或预训练模型。已淘汰短试验的删除也需保留配置/日志/报告/metadata及删除清单。历史用户已明确删除旧batch8/32/64主组权重，不能以历史通过报告推断它们仍可加载。

### 9.3 恢复与安全停止

使用生成队列：完整latest存在时自动添加`--resume`。数值失败被标记后会保持停止，避免无限重启；检查原因后才重新安排恢复。手动训练命令可用同配置、同manifest、同输出目录，加`--resume /.../ckpt_latest.pt`，steps表示目标总更新数，不是额外增加数。

恢复加载模型/优化器/调度器/RNG和准确采样位置。若日志比断点更新，活动记录会备份进`resume_history`并回退，避免把旧模型的后续评估误作恢复后结果。断点比已保留best更旧时拒绝同目录回退，应使用隔离目录。不要只用`load_state_dict(model)`冒充完整恢复。

训练器尚没有通用“任意时刻发信号就保存当前参数”的接口；直接kill会丢失上次latest之后的更新。源机暂停seed43采用等待新latest写后校验完成，再让其拥有的队列停止。队列安全接收TERM后只停止自己的子进程；不要用`pkill python`等全局命令。

现有恢复机制已在源服务器真实4B两步和隔离98→99重放验证；其中测试权重按用户授权已删除，只保留报告。这不等于目标机器和每个新配置都完成恢复测试。长训前至少对自己的首个完整断点做一次受控恢复检查。

## 10. 资源与数值保护

队列会检查空闲GPU、完整断点磁盘预算，记录5秒左右资源采样，并在固定TRAIN误差超过自身未训练基线10倍等情况下停止。非有限训练损失/参数梯度由训练器拒绝。梯度范数记录在外层裁剪前，巨大值不等于实际参数修改量，也不因clip=1就保证候选过程稳定。

退出时区分GPU分配失败、CPU分配失败、共享容器OOM相关SIGKILL、磁盘满、数值失败和未知错误。容器OOM事件包括同容器其他任务，只能结合时间/进程证据判定，不能凭一个历史计数甩给本实验。

`resource_summary`里的进程USS与cgroup总内存不是同一指标；累计卸载字节也不是当前占用。shared-memory/page-cache仍可能计入容器账单。

2026-11-06补充了非有限梯度报错分类；源机已运行的老监测进程可能仍加载老分类函数，未为此重启训练。新机器会使用修正后的代码。

## 11. W&B（可选）

训练和W&B采集独立，本地日志是权威记录。源账号密钥不会上传；新机器自行登录/配置，不默认继承共享机器账号。

仓库提供`configs/wandb_bridge.example.json`。复制为被Git忽略的`configs/wandb_bridge.local.json`，把entity/project、cache_dir、api_key_file、sources.config改成本机值；key文件放`.secrets/`且权限600，不写进JSON正文：

```bash
mkdir -p .secrets
chmod 700 .secrets
# 在本机安全编辑.secrets/wandb_api_key，然后：
chmod 600 .secrets/wandb_api_key
python -u scripts/watch_wandb.py --settings configs/wandb_bridge.local.json
```

先安装`requirements-wandb.txt`。sources里config指向本机生成的`local_config.yaml`。采集器在有首条训练日志后才创建run，按真实输出目录计算独立run ID；新实验要用新目录。相同输出目录用于恢复，不能把新配置混进旧曲线。

如使用代理，应填自己可访问的地址到进程环境。源机当前转发`127.0.0.1:17901`依赖现有SSH会话，在新机器通常无效。上传断开不等于训练停止，恢复网络可补传。

主要看：`comparison/train_fixed_action_mse`、`comparison/train_fixed_future_mse`、`comparison/val_fixed_action_mse`、`comparison/val_fixed_future_mse`，横轴`comparison/sample_visits`。原始每batch指标为`train/action_mse`、`train/future_mse`，未乘权重；`train/total_loss`已加权。总loss定义不同的组不可直接比数值。验证表里的每任务指标可从评估JSON汇总，采集器当前不是所有分任务条件指标都有独立panel。

## 12. 源服务器当前任务，仅供交接核对

| 服务 | 当前用途 |
|---|---|
| `energywam-three-task-42` | GPU01三任务主组，继续 |
| `energywam-three-task-43` | GPU23三任务第二种子，121步安全暂停 |
| `energywam-tiny-future-weight` | GPU23依次小样本权重1/9 |
| `energywam-wandb` | 独立日志上传，不执行训练 |
| `energywam-gpu23`等旧流程 | 已停止/历史；不要重启导致重复训练或Stage2 |

源机查看：

```bash
python scripts/observe_three_task_progress.py
python scripts/observe_tiny_future_weight_progress.py
supervisorctl status energywam-three-task-42 energywam-three-task-43 energywam-tiny-future-weight
```

本机路径：

```text
/cloud/zzc/energyWAM/experiments/three_task_full_20261106/
/cloud/zzc/energyWAM/experiments/tiny_future_weight_20261106/
```

这些观察/准备/切换脚本有本机绝对路径。师兄异机入口是第8节的`prepare_remote_run.py`和生成队列，不是`run_tiny_future_weight_comparison.py`；后者会尝试暂停源机指定supervisor服务。

## 13. 分工、版本与实验记录

共享Git只同步代码、配置与小型复现元数据；数据和权重分别部署。开始实验记录commit、配置文件/哈希、manifest哈希、统计指纹、数据版本、模型来源、seed、硬件、样本预算/更新预算和输出目录。环境/数据不一致就标明，不把它们当严格消融。

建议每个改动单独分支、每个实验独立配置和输出目录。师兄可优先跑与当前主组互补的单变量实验：损失权重、任务均衡采样、有效batch、Action二次项或head LR；不要在一次对照同时改变多个参数。早期先小样本检查能否拟合，再检查独立VAL，最终少数候选换训练seed复核。

Git不提供GPU、不自动同步checkpoint，也不自动让两台机器共同训练同一个模型。跨机器分布式训练不是本次交接目标。

本项目研究代码未在交接中另加开源许可证；仓库访问不等于任意再分发授权。第三方模型与数据按原项目许可获取和使用。

## 14. 常见问题

| 现象 | 优先检查 |
|---|---|
| 一开始Future MSE很低 | 当前特征复制基线；比较真实未来而非当前自身 |
| 前几十更新曲线抖动/不降 | 同一固定TRAIN/VAL曲线、目标比例、步长梯度；不凭不同batch断言失败 |
| GPU显存很满、功率不高 | CPU卸载、传输/数据准备、两卡串行层执行；比较稳态吞吐 |
| 无法启动，提示metadata不同 | 数据版本/50任务编号是否与冻结manifest一致，别强行跳过 |
| OOM | 分清GPU、进程USS、共享cgroup；不要直接增native batch |
| resume拒绝/日志回退 | 模型断点步、best步和活动日志是否一致；保留resume_history |
| 没有W&B曲线 | 是否已有train_log、sources配置、网络/代理、服务端实际上传 |
| 网页总loss空白 | 纵轴是否被固定0～1；实际总loss可能大于1 |
| 效果下降但没非有限值 | 数值有限不等于学习有效；分别看两个方向与各任务 |

## 15. 本次交接验证范围

- 已在源机器核验本机路径重定向准备器、数据元信息一致性与所选轨迹文件存在；未在师兄机器实际启动训练。
- 代码测试覆盖内层解析更新、完整高阶求导、迭代编码、步长、microbatch等价、保留/恢复记录、资源分类和新配置准备。
- 当前真实训练未因本次文档/Git整理而停止；GPU23已按此前明确指令换成权重对照。
- Git发布不含任何模型/数据/训练权重或账户密钥；上传前运行清单与凭据检查，提交/推送结果以实际Git记录为准。

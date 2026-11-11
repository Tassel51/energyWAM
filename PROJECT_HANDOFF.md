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

完整历史分析见`docs/LTE_WAM_MODEL_AND_EXPERIMENT_SUMMARY_CN.md`及`docs/reproducibility/experiment_evidence_summary.json`。源服务器完整日志仍在云盘，不随Git分发。

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

如使用代理，应填自己可访问的地址到进程环境。源机当前转发`127.0.0.1:17901`依赖现有SSH会话，在新机器通常无效。上传断开不等于训练停止，恢复网络可补传。 若SDK流在403后永久失败，仅网络恢复可能不足；当前采集器对仍在训练的来源每60秒核验服务端步数/状态，持续180秒无进展或服务端异常结束则重建上传会话，按服务端确认位置重放。SDK关闭阻塞会让采集器以75退出，由supervisor的unexpected策略重启；自行前台运行时需重新启动采集器。

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

## 2026-11-09 历史实验记录索引

用户要求为实验文档汇总此前所有研究。项目根目录`HISTORICAL_EXPERIMENTS.md`按16类研究记录目的、实际预算、TRAIN/VAL结果及限制，并汇总工程/恢复/数据诊断；涵盖旧版BI2300、早期Joint100、K/步长、先验/初始化、两轮batch、三任务、旧头LR、重放/低LR续训及最新单episode四档。逐目录清单和来源哈希在`docs/reproducibility/historical_experiment_inventory_20261109.{csv,json}`，135个证据目录包含测速/恢复/重试，不代表135个独立训练。未迁入的旧服务器日志仅按交接摘录，不声称全面重验；权重可用性须另查。此为实验结果索引，不取代本交接主文档。

## 2026-11-09 当前训练：单episode能量头LR

已按用户明确授权启动`energywam-single-episode-head-lr`，入口`scripts/run_single_episode_head_lr_study.py`；根目录`/cloud/zzc/energyWAM/experiments/single_episode_head_lr_20261109`。headLR5e-4后2e-3在物理GPU2/3串行，各fresh210update，与已有head1e-3结果比较，其他配置/数据不变。queue_status.json为逐组实际状态，comparison_report.json持续保留0步/固定评价；当前首次启动在初始化，勿当作已完成更新。完整latest/best按auxVAL A+9F保留。当前study不自动启动其他超参数研究。空间及上轮失败×2权重退休清单见preflight.json/retired_failed_overall_lr2_weights.json；旧组权重状态勿仅凭metadata判断。

## 2026-11-09 实验室远程执行已验证

代码仍在compshare维护；lab-via-pc经个人电脑双隧道可达user-SYS-420GP-TNR。实验室代码/data3/xyx_zhuzicheng/projects/energyWAM_remote/current；独立环境和输出在/data7/zzc/energyWAM/lab_execution_20261109。物理6/7 L40S必须按UUID选择，NVML/CUDA数字顺序不同；本机2/3策略未放宽。真实4B/B1/K2保存1->完整恢复2通过，latest2/best0完整，24项CPU检查通过；无正式训练提交。详细docs/LAB_REMOTE_RUN.md，本机logs/lab_remote_setup/deployment_validation_report.json和results/20261109T073143Z。管理：python scripts/lab_control.py status|logs|fetch|sync；新两步验证prepare-smoke --name再start-smoke，stop-smoke只停本次runner。dataset/model按SHA复用，标准化只做有记录的路径迁移、数值不变；可用数据视图仅3轨迹，不能直接提交完整三任务任务。

## 2026-11-09 用户授权实验室双队列（运行中，非完成结果）

用户明确指定物理GPU3/4跑Future损失权重1/9，GPU6/7跑内层步长研究。各组fresh官方4B、seed42、OpenDrawer3346的67TRAIN/16辅助VAL窗口、FP32、有效B16/native4、K2、210update（3360窗口访问、50.15等效epoch），评价每20update，完整latest/best；best按本组VAL A+lambda_F*F预声明。整体倍率固定原1倍/head1e-3，云端head-LR研究尚未判胜，不冒称最终最佳。远端torch2.6/cu124不同于云端，必须各硬件组自带对照，不能将跨主机差异归因超参数。

- loss根目录：`/data7/zzc/energyWAM/lab_execution_20261109/studies/future_weight_20261109`。weight9对照后weight1，串行；完整权重预算122.41GiB。32GB卡saved-tensor GPU缓存预算0/0，保留CPU异步卸载，未改数学配置。原18/18层分配容量检查在首个backward CUDA OOM，完成0update，原始日志归档failed_setup_attempts/oom18_18_*。正在独立验证16/20层分配（首卡含视觉编码，减少两层），须梯度一致性及2update容量通过后才自动正式启动，不能记成正式已训练。
- steps根目录：`/data6/zzc/energywam_lab_studies/inner_steps_20261109`。原alphaA5/F2000对照，再A2.5/F2000、A5/F1000，三组串行；保存预算168.17GiB；GPU缓存预算10/14。修改步长时同时补偿quadratic权重，保持有效先验precision .2/.0005，避免混淆先验强度与求解步长。共享真实4B梯度检查（native1 vs4，reference缓存10/14 vs candidate0/0，4个固定窗口）已通过；只覆盖初始抽样梯度，不冒称全部训练轨迹等价。2update B16/native4/FP32容量检查通过，16:15:37 CST启动的controller已进入正式training；已得到完整67窗口step0 TRAIN A0.6053476845/F0.1203727312，实际更新数须查最新日志。
- 兼容性准备失败保留：独立脚本缺sys.path及Torch2.6缺host_memory_stats，均在正式更新前发现；已修复导入和缺失统计兼容。该版本不可用全局pin缓存统计，记录missing，仍保留单context64GiB pin限额、每任务USS120GiB/主机available24GiB、单组12h保护。停止/失败不自动改超参数重跑。
- 独立tmux队列，不依赖SSH存活；代码按唯一release冻结。配置、arm_definition、数据manifest、source_manifest、启动命令、梯度检查和失败记录都在根目录。运行部署索引`logs/lab_remote_setup/ablation_deployment.json`；管理`python scripts/lab_ablation_control.py status`、`logs --study loss|steps`、`stop --study loss|steps`（只操作指定本次队列）。远端未复制W&B密钥，以原始本地日志为准。云端GPU2/3研究未停止，其他用户进程未操作。

补充16:29 CST实况：GPU3/4的16/20分配在梯度门槛backward时logicalGPU1 OOM（未做optimizer update），已保留梯度日志/资源；再次仅调整为17/19作最后一档容量尝试，不降低模型/精度/batch/K。前一次16/20失败归档目录名误为failed_setup_attempts/oom17_19_1794212892112140382，目录名不是实际配置，实际source/config/log明确为16/20。L40S正式对照此刻仍0/210：完成step0 TRAIN/VAL评价后在写入校验24,568,514,334字节完整pending checkpoint；进程D状态wait_on_buffer/rq_qos_wait，为/data6 ext4磁盘I/O等待，不能把capacity2步或step0评价算作正式更新。14项层分配/GPU政策/旧Torch统计兼容CPU测试通过。

16:34 CST启动检查结论（取代上文待定状态）：GPU3/4的17/19梯度门槛已双方向通过，相对L2 Action0.000250796/Future0.000457111；随后带完整optimizer的2步容量检查仍首次backward GPU0 OOM，完成0update，controller=blocked_capacity，未启动任何正式loss组，已释放3/4。本次不再尝试层分配，也不擅自改batch/精度/模型或改GPU。GPU6/7 controller=training，正式control仍0/210，另两组queued；真实2步capacity不计入正式预算。首个完整step0 pending checkpoint在/data6等待fsync，尚不是可恢复latest；10秒整盘写速11.6102MiB/s，D状态wait_on_buffer/rq_qos_wait。队列保留后台继续，不能描述为已完成正式更新。完整启动报告logs/lab_remote_setup/formal_ablation_startup_report.json；最新原始记录取回ablation_results/20261109T083430Z，各远端study根目录也保存deployment_startup_review.json。若要loss改排6/7，需要用户确认新的资源分配；尚未提交此替代任务。

16:39:44 CST只读复查：lab steps对照PID4178280已越过首个checkpoint磁盘等待，console新增[shared-sharded] begin update=0；ckpt_latest.pt与ckpt_best.pt及metadata均已出现（24,568,514,334字节）。正式第1次更新正在执行，尚无speed_report完成记录，已完成仍0/210。没有重启或改配置。

2026-11-09 16:41:47 CST实验室进程资源3秒采样：112逻辑CPU/56核，主机CPU11.6%，RAM345.764GiB/available273.132GiB，swap已用4.255GiB（不据此认定当前换页）。energyWAM训练PID4178280 CPU99.3%（100%=一逻辑核）/RSS48.392GiB/USS48.184GiB，训练及子进程合计CPU99.6%/USS48.263GiB；队列监督进程CPU21.8%。其他任务主要为TravelUAV仿真385.9%、VLN评估177.5%、GeMI3D训练115.6%、S2S两训练100.1/99.9%；各自RSS不能简单相加当独占内存。此前3/4第二套启动失败是CUDA显存不足，不是主机RAM不够。原始记录logs/lab_remote_setup/process_resource_audit_latest.json。未改进程/配置。

2026-11-09 16:43:37 CST用户明确要求恢复GPU3/4再试：已新鲜检查两卡空闲（<256MiB、无计算进程），将上一17/19容量失败整目录和状态归档failed_setup_attempts/capacity17_19_before_user_retry_20261109T084337Z，未删除或覆盖失败证据。固定原release/配置/seed/B16/native4/K2/FP32/缓存0,0/17,19层，不做配置变化，新tmux energywam_lab_loss_userretry_20261109T084337Z启动同样2update容量检查；既有同配置双方向梯度门槛通过不重复。只有容量通过才自动进入原权重9/1两组各210update；失败即停，不自动重试。尚无此次容量结果。回执logs/lab_remote_setup/loss_user_retry_latest.json及远端user_retry_20261109T084337Z.json。GPU6/7任务未动。

2026-11-09 16:46:08 CST用户授权GPU3/4同配置重试结果：容量子进程19883在约120秒后returncode1/CUDA out of memory，首个update反向重计算阶段失败，0/2容量更新、0正式更新，无新checkpoint。采样GPU3峰32030MiB、GPU4峰31407MiB；logicalGPU0=physical3，日志申请34MiB失败。退出后3/4显存20/19MiB释放，主机available271.962GiB，排除本次为主机RAM耗尽。6/7进程未操作且仍占用训练峰显存。保存logs/lab_remote_setup/loss_user_retry_result.json、远端capacity_profile/console.log/resource_summary.json/profile_status.json；controller blocked_capacity，不再自动重试或降低配置。

## 2026-11-09 16:52 CST 用户授权3/4改native2并继续

用户明确同意同时计算2样本，其余不变。仅loss研究的3份配置（两正式组及capacity）shared_microbatch_size4->2，元数据notes同步文字；脚本逐字段断言其他全部超参数一致。有效batch16，每方向累积8次后一次总更新；4B/FP32/K2/210updates/seed42/数据顺序与manifest/学习率/优化器/损失/17,19层分配/缓存0,0均不变，6/7未动。此前native4容量失败及原configs/plan/arm_definition归档failed_setup_attempts/native4_before_native2_20261109T085200Z。新tmux energywam_lab_loss_native2_20261109T085200Z先做真实4B双方向native1vs2梯度门槛（固定窗口0,66、strictFP32、SGDlr0不更新），再完整优化器2update容量检查，全部通过后才自动fresh开始weight9对照再weight1，各210updates，保留full latest/best；门槛阶段不算正式完成。没有授权进一步自动降低micro或改其他配置。实际阶段须查controller_status。回执logs/lab_remote_setup/loss_native2_preparation.json及远端native2_preparation_20261109T085200Z.json，部署索引已更新。

2026-11-09 16:56:31 CST native2实际结果：native1vs2双方向梯度检查通过，Action相对L2 .000293917794、Future .000925921787；只覆盖窗口0/66初始抽样，不声称全轨迹精确等价。随后完整B16/native2/FP32/K2/17,19分配容量测试PID36266首个backward CUDA OOM，returncode1，0/2容量更新、0正式更新，无新checkpoint。采样GPU3峰32046MiB、GPU4峰28767MiB；physical3=logical0申请12MiB失败。controller blocked_capacity，原配置现保留native2，未进一步改native1/精度/K/模型/学习率/层分配；6/7任务未操作。启动结果logs/lab_remote_setup/loss_native2_startup_result.json，所有失败原始日志/配置/资源及梯度报告保留远端，最新轻量原始记录经fetch取回ablation_results目录。下一步若继续需用户决定执行调整，不能把当前配置修改或梯度通过称为正式已跑。

## 2026-11-09 native1启动授权

用户明确要求GPU3/4 microbatch2->1，其余不变。实际准备时间及tmux见logs/lab_remote_setup/loss_native1_preparation.json；逐字段断言仅shared_microbatch_size及notes文字变化，B16、每方向16次累积、FP32/4B/K2/原学习率/210updates/相同manifest与seed/17,19层分配/缓存0,0不变。旧native2失败、配置、plan、arm_definition完整归档failed_setup_attempts/native2_before_native1_*。已有通过的native1-vs2真实梯度检查包含相同硬件配置的native1参考，不重复无更新probe；本次仍必须完整优化器2update容量通过才进入权重9对照再1各210正式更新，capacity不计入正式预算。6/7未改。当前仅启动capacity，不能标正式训练已完成。

2026-11-09 17:05:31 CST native1结果：容量PID48511在约60秒后returncode1，首次term.backward CUDA OOM，0/2容量更新、0正式更新，无新checkpoint；exception physical3/logical0进程31.23GiB，尝试96MiB失败。5秒资源采样GPU3/4峰27378/27481MiB，采样未捕捉瞬时分配峰，必须与exception31.23GiB区分，不能据采样27GiB声称显存仍足。主机available270.383GiB，非RAM耗尽。实际resolved配置native1，controller旧通用reason文字硬编码native4，不能将该文字当实际microbatch；以resolved_config/receipt为准。所有原始日志和配置保留，结果logs/lab_remote_setup/loss_native1_startup_result.json。6/7未动；3/4测试失败已自然退出，不继续重试或改精度/模型/K/层分配。既有配置证实常驻训练权重/梯度/Adam状态与高阶计算工作区均仍占GPU，尚无逐张量精确实测内存分解，不应把具体字节归因到某一类别。若继续同协议，需要进一步执行内存工程或改用大显存卡，未启动替代方案。

## 2026-11-09 CPU优化器状态卸载授权与部署

用户明确授权继续内存工程，要求GPU3/4可运行，保持数学协议。新增trainer.optimizer_state_cpu_offload（默认false，不影响现有云端/6,7）；仅3,4研究设true。OptimizerStateOffloader在前向/反向时将原AdamW8bit state张量同步复制CPU（保留dtype、嵌套标量和共享引用），原GPU step前按parameter.device搬回，之后finally卸载；仍使用原bnb kernels/8bit状态，未换CPUAdam/未改全局clip/梯度累积/学习率/FP32/B16/native1/K2。FP32参数没有额外mastercopy。load后重新预分配/卸载，checkpoint直接保存普通完整optimizer.state_dict，无侧车状态依赖。CPU单元16项通过，包括嵌套状态/别名/dtype、checkpoint roundtrip、paged状态拒绝及既有GPU政策/层分配/旧Torch兼容测试。

管理隧道一度Connection refused，用户已恢复后实测hostname正确；6/7后台未停止。新部署时间20261109T091636Z，tmux energywam_lab_loss_cpuoptimizer_20261109T091636Z，冻结代码/data3/xyx_zhuzicheng/projects/energyWAM_remote/releases/energywam_code_20261109T091542Z_632da7290132_release/code；旧native1失败与旧configs归档/data7/zzc/energyWAM/lab_execution_20261109/studies/future_weight_20261109/failed_setup_attempts/native1_before_optimizer_offload_20261109T091636Z。code仍compshare维护，独立release包含未提交改动。

3层自动门槛：①两GPU小型FP32参数原AdamW8bit vs CPUstateoffload，4次更新/全部state逐位比较并save2/resume至4（仅合成验证）；②真实4B/B16/native1/完整optimizer的2update容量；③真实相同B16/67TRAIN+16VAL完整save1->resume2，latest/best完整状态验证。全部通过后才fresh启动原weight9对照再weight1，各210updates。新门槛不计正式训练步数。当前仅提交门槛，不能记已通过或正式已启动。resume runner现按plan指定pair，原默认6/7不变；研究controller支持resume_gate。容量reason已去掉硬编码native4文字。

完整空间预检data7剩209.83GiB，正式两组latest/best/pending加独立resume验证与安全余量预算199.05GiB；未删除任何权重。部署回执logs/lab_remote_setup/optimizer_offload_deployment.json和远端optimizer_offload_deployment_*.json。真实GPU更新等价性/容量/恢复目前待门槛结果，不把CPU测试当GPU证明。

CPUstate容量检查中间证据：已真实完成第1次4B/B16/native1/K2更新（独立capacity，不计正式210），耗时123.305s，optimizer原state CPU/GPU往返8245862488字节=7.680GiB，param_grad_nonfinite=0.0；GPU allocated峰30.198/29.868GiB。第2步与真实resume仍待完成，不把单步通过称全面通过。offload/pinned_bytes是跨micro/direction累计请求量，不是当前主机驻留内存，不能解读为777GB实际占用。数据在logs/lab_remote_setup/optimizer_offload_capacity_progress.json。

17:27 CST真实resume验证中间进展：已完成第1次更新并写pending checkpoint；逐字段对比独立连续capacity第1步，loss、param_grad_norm、两方向endpoint MSE、更新后alphaA/F完全一致（差0），证据logs/lab_remote_setup/optimizer_offload_resume_trajectory_comparison.json。此为同CPU卸载实现的独立fresh重复，非4B无卸载基线对照；完整第2步resume尚待完成。

17:29:26 CST真实B16/native1恢复验证已完成第1步保存与固定评价，runner已自动启动target2新进程86293，通过--resume同目录ckpt_latest.pt继续。完整恢复的第二次更新及最终full-state检查尚未通过，不把进程启动当恢复验证完成。

## 2026-11-09 17:35 CST CPU状态卸载已跑通，GPU3/4正式队列启动

三层门槛均实际通过：①双GPU小型FP32参数原AdamW8bit vs CPUstate版4次更新，参数和全部state逐位一致，含save2/resume至4；②真实4B/B16/native1/FP32/K2完整optimizer的2次容量更新123.305/102.389秒，无非有限梯度，GPU allocated峰30.198/29.868GiB，进程USS峰33.416GiB；③同B16/native1/完整67TRAIN+16VAL save1->resume2验证通过，latest2/best2均24,569,187,110字节，包含model/optimizer/scheduler/rng/meta，恢复后第2次更新完成。恢复验证两步的loss、梯度范数、分方向训练batch endpoint MSE及更新后alpha，与连续capacity两步相同（差0）；这是记录指标一致，并非全4B参数逐位轨迹证明。原无卸载4B因OOM无法对照完整更新，原算法逐位证据限于小型GPU验证，需明确范围。

每次更新前后CPU/GPU搬运原优化器状态8,245,862,488字节=7.680GiB，保留原8-bit dtype与GPU update kernels，未改数学协议。4B FP32/B16/native1/K2、学习率、全局clip、数据/seed、17,19分配和210update预算不变。

正式weight9对照已fresh启动，PID94506；17:35:31 CST处于第0步固定评价，正式完成0/210，weight1仍queued。初始化sample hash1098bd8778560da832704b6f81bba2ab345ef4f51e80312827cd513295ae2546，与6/7对照相同（仅每参数64元素采样，不是全权重hash）。物理3/4按UUID授权落卡，初始参数numel分配cuda0=2162724995/cuda1=1917685504。两正式组在3/4串行各210，后续自动，不需要持续助手监视；6/7未停止/改动。

最新启动证据logs/lab_remote_setup/optimizer_offload_startup_report.json；含独立resume轻量原始日志/配置/metadata的取回目录logs/lab_remote_setup/ablation_results/20261109T093528Z（optimizer_resume/）；不复制大权重。主study仍/data7/zzc/energyWAM/lab_execution_20261109/studies/future_weight_20261109，独立resume输出outputs/optimizer_offload_b16_native1_resume_20261109T091636Z。正式latest/best随训练按预声明VAL准则保存，不能将验证step2计入正式预算。状态python scripts/lab_ablation_control.py status；日志logs --study loss；停止stop --study loss，仅本次队列。代码release在部署回执，后续sync不会改变当前固定release。

## 2026-11-09 GPU1/8上的K2/3/4研究（用户新授权）

用户明确要求立即启动K2/3/4，沿用GPU3/4全部配置和CPU卸载。GPU8已有fengtuo进程3721709，助手询问后用户明确回复“已获准共享GPU8”；不得把共享授权推广到其他卡或操作他人进程。物理GPU1 UUID468e9427-c3c6-65ee-20db-0e93343a1861 RTX4080SUPER，GPU8 UUID23d0e721-8534-c52f-9175-4296ef36c863 L40S；按完整GPU-UUID设置CUDA_VISIBLE_DEVICES，GPU1独占检查，GPU8允许共存，启动时两卡至少32000MiBfree。本任务GPU8 CUDA allocator限制31000MiB（不包含驱动全部开销，不等于对共享任务的硬隔离），GPU1保持原物理上限；共享争用会影响吞吐，记录co-tenant，不据此精确推断K速度收益。未停止3/4、6/7或他人任务。

部署20261109T095549Z，代码冻结/data3/xyx_zhuzicheng/projects/energyWAM_remote/releases/energywam_code_20261109T095430Z_b774d22735dc_release/code，tmux energywam_lab_k234_20261109T095549Z，study根/data8/xyx_zhuzicheng/energyWAM_studies/k234_20261109T095549Z。/data8新建UID1017自己的xyx_zhuzicheng/energyWAM_studies，实测可写剩3118.75GiB，正式3组latest/best/pending+独立完整resume验证预算244.82GiB；代码仍/data3项目，数据模型仍复用原路径，不复制大文件/密钥。deployment_preflight.json及本机logs/lab_remote_setup/k_study_deployment.json包含GPU原始占用、配置SHA、计划和共享授权。

正式各fresh seed42/相同4B FP32/有效B16/native1/K2参考设置/双方向loss A+9F/alphaA5,F2000/先验有效.2,.0005/17,19层/激活GPUcache0,0/原AdamW8bit stateCPU卸载；仅train_inner_steps与eval_inner_steps随组设2/3/4。每组210update=3360窗口访问=50.15等效epoch，固定67TRAIN/16auxVAL，每20update评价，best按本组ownK的VAL A+9F，单独记录共同推理K2，不混为同预算质量结论。按K2->K3->K4串行。时间保护每arm24h以容纳更大K，未增加210更新预算，也不自动扩展。batch数据访问预算一致，计算预算并不一致。

旧工具硬编码K2/4/8已修复：可显式probe budgets包含3，保留旧默认；collect保留所有K，guard可选ownK，best读取ownK；CPU测试覆盖K3独立iteration progress、K3评价/最佳/失稳guard、共享卡freefloor和逻辑allocator落卡等17项通过。新的通用功能在当前运行的旧release中不会生效，不修改其运行中代码。

启动门槛：新1/8上小型原bnb更新及CPUstate完整往返/恢复逐位检查；最大K4真实B16/native1两步容量；K3真实B16完整save1->resume2验证（67训练窗口采样不变，仅工程评价缩为2TRAIN/2VAL并以ownK3选best）。上述仅工程验证，不计正式210；正式评价仍67TRAIN/16VAL。当前处于门槛，三正式组都尚未完成，实际状态须查询，不能记已跑通。状态python scripts/lab_ablation_control.py status --study k，logs/stop同样--study k。完整记录沿用MEMORY/PROJECT_HANDOFF/HISTORICAL，不另建竞争主报告。

## K研究首次共享容量检查结果（2026-11-09T10:01:35.401947+00:00）

1/8小型原AdamW8bit/CPUstate更新与恢复逐位检查通过。K4真实容量PID130737在首个term.backward发生物理CUDA OOM，完成0/2容量更新、0正式更新，controller=blocked_capacity；K2/K3/K4三正式组均未启动，K3真实resume未执行，没有新4Bcheckpoint。GPU8上的原进程3721709运行中涨到20.57GiB，本任务当时23.62GiB（CUDAallocated22.88GiB），设备仅剩90.31MiB，申请96MiB失败。自设allocator上限30.27GiB尚未触及，因此不是quota造成，原任务的动态显存增长使共享物理空间不足。不能据此判定K4算法本身不稳定或独占时必然不适配。资源采样GPU8峰45228MiB是包含共享任务的整卡占用，不可当成全归本实验；GPU1峰31298MiB，进程USS峰54.571GiB，hostavailable约192GiB，非主机RAM耗尽。

原配置/源快照/command/资源trace和失败完整保留study/capacity_profile，结果本机logs/lab_remote_setup/k_study_shared_gpu_failure.json。3/4与6/7未改，他人进程未操作。助手已向用户询问下一步资源安排：等待GPU8无其他计算进程后自动启动（建议），或用户协调GPU8稳定至少32GiB可用后再共享；当前等待该选择，未自动重启失败测试或更改算法/卸载方式。共享授权本身已获得，不重复询问授权，待确认的是实际资源安排。现有全部K配置、评价代码、best保留和自动串行队列均准备完毕；若资源到位仍先重做容量与K3完整resume门槛，不能跳过。

K资源外部变化与实际重启：失败结果抓取后GPU8已变16MiB、45442MiBfree、无计算进程；再次nvidia-smi及ps确认原PID3721709不存在，退出原因未知，不声称正常完成或与本次共享无关。助手没有向其发送任何结束信号。用户此前已授权1/8运行且确认共享，现两卡实测空闲，缺失资源条件已由外部状态变化解除，未将尚未回答的资源选项当成批准。已明确告诉用户刚才问题不用再答，按现有启动授权继续。

20261109T100522Z将shared8首个OOM完整归档/data8/xyx_zhuzicheng/energyWAM_studies/k234_20261109T095549Z/failed_setup_attempts/shared8_oom_20261109T100522Z，新tmux energywam_lab_k234_exclusive_20261109T100522Z重新做原K4容量检查。启动条件改为1/8均无其他计算进程；GPU8原共享allocator31000MiB限额取消，只在卡空闲时启动。为匹配32GiB/46GiB异构卡，全K组统一语言层分配17/19->13/23，让8多承担4层，其他全部数学超参数和激活/Adam状态CPU卸载方式不变；不是K的额外模型因素。相同3/4数学设置仍成立，但设备放置与3/4不同，须使用1/8本地K2对照。旧配置已归档，最新resolved配置/arm_definition有13/23，实际部署索引ablation_deployment.json已更新。GPU8若重新有进程，队列启动前继续等待，不操作对方任务。正式仍0/210各组，不能把重启门槛算正式训练。回执logs/lab_remote_setup/k_study_exclusive_relaunch.json。前一条等待用户选择的状态由本条外部状态解除取代。

18:10:44 CST K4独占13/23层容量中间证据：真实完成第1次更新254.701s，param_grad_nonfinite0，GPU1/8本进程allocated峰23.970/36.099GiB，资源采样NVML整卡峰25248/37818MiB。显存已能容纳本次真实K4更新；第2步及K3恢复仍待验证，不把单步/工程更新计入正式预算。资源峰USS56.110GiB，非RAM瓶颈。证据logs/lab_remote_setup/k_study_exclusive_capacity_progress.json。

18:15:56 CST K4独占13/23两步容量已完整通过：254.701/211.000秒，两个param_grad_nonfinite均0，GPU allocated峰23.970/36.099GiB，CPUstate搬运每次8,245,862,488字节。上述是改变训练batch的工程容量记录，不以两步loss比较判定学习效果。K3真实B16/native1完整resume验证正在target1，已按K=3的2VAL A+9F保存step0 best/latest；criterion实际含K=3，确认评价/选择链路不再遗漏3。正式K2/3/4仍0/210且未启动，工程评价2TRAIN/2VAL不可当正式67TRAIN/16VAL结果。最新logs/lab_remote_setup/k_study_startup_report.json。

2026年11月09日 18:23按用户飞书截图格式撰写四项实验文档，已续入HISTORICAL_EXPERIMENTS.md第22节（目的/组表/固定设置/比较标准/阶段结论），保留查询快照logs/lab_remote_setup/four_study_report_snapshot_20261109.json。正文可用于替换1.2并续写1.3至1.5，不声称已编辑飞书。head半倍210 TRAIN A/F=.2790192951/.08718044325，基准.08811573361/.07539999085；加倍仍未完成。其余进度采用该快照，K正式仍0且恢复验证中。用户要求运行后无需持续监管，未因文档任务重启或改训练。

2026-11-09 18:37:49 CST按用户GPU截图只读检查：K队列未崩溃，controller PID143181、正式queue PID172031仍在；K4两步容量通过，K3真实B16/native1完整save1->resume2也通过，latest2/best0含model/optimizer/scheduler/rng/meta。验证进程正常退出释放1/8后，GPU8重新出现fengtuo任务，因此正式K2状态waiting_for_idle_gpus，K3/K4 queued，三组正式完成均0/210。当前政策要求1/8启动前无其他计算进程，显存free多不等于可启动；未自动切回共享、未重启任务、未停止他人进程。待8空闲后原队列自动启动。证据logs/lab_remote_setup/k_study_startup_report.json；用户要求跑起后无需持续监管，按需查询即可。

2026-11-09用户再次明确“不用空闲啊，和别人一起用就行”：已在不重启controller/不重复验证的情况下原子修改K study的GPU政策，允许GPU8共享，GPU1仍独占；按已测13/23分配K4峰值保留启动freefloor GPU1=26000MiB、GPU8=40000MiB（GPU8实测需求约37818MiB含context，区别于先前17/19共享32000门槛）。当次GPU8剩42944MiB满足门槛，现有正式queue每15秒重读策略会继续K2。未改模型/学习率/K/微批次/层分配/CPU卸载；没有再跑容量/resume，已通过记录继续有效。未操作他人进程。政策变更及原始GPU占用回执logs/lab_remote_setup/k_study_sharing_reenabled.json，远端sharing_reenabled_*.json；本条取代前文必须等待GPU8空闲的要求。

GPU8共享政策生效后，原正式queue已自动启动K2，PID188599，实际physical1/8，已完成模型13/23分配及67TRAIN/16VAL缓存初始化；当前正式完成仍0/210，第0步评价/初始化阶段。K3/K4自动排在后面，不需要重发命令。证明logs/lab_remote_setup/k_study_shared_formal_start.json。用户要求跑起后不持续监管，启动确认后结束观察，后续按需查询。

2026-11-09 20:02:56 CST按用户询问当前服务器任务只读检查：当前host cpod-1x51qyjx1lqg/compshare，energyWAM能量头LR0.5与2两组均210/210、returncode0，controller于19:59:31 CST complete，当前无energyWAM训练进程，GPU2/3各1MiB/0%空闲。GPU0/1为/workspace/xch的CDP2/Libero40双卡2step smoke，PID2758362/2758363各约30GiB，不能当成energyWAM训练；watch_wandb.py后台日志同步仍在。尝试查询lab三study时127.0.0.1:12399 Connection refused，当前实验室状态未知，不用旧日志冒充实时；隧道断开不等于远端tmux任务停止。未变更任何训练或他人进程。

## 能量头LR对照完成后的下一轮建议（2026-11-09）

只读核验三组同为210步的固定67TRAIN：head5e-4 A/F=.279019295057/.087180443247；head1e-3（既有控制）.088115733610/.075399990847；head2e-3 .157500841113/.085363779940。三组完整完成，head1e-3在本次TRAIN两个方向最好。裁剪前全参数grad_norm采样最大值分别79.462780/12.359374/8849.661329，均有clip1，不能把大裁剪前norm当成已发生非有限更新。最后固定16VAL分别.808400/.136369、.682835/.184489、.749488/.165326，验证未同步改善，不能认定全量训练最优；按预声明VAL A+9F的best步骤分别120/60/0，TRAIN末端排名与best选择分开。

对用户“后面该实验多少学习率”的建议：仍以原能量头1e-3为基线，优先细调7.5e-4，再试1.25e-3（±25%），其他LR、损失、K、数据、初始化/seed、210步预算全部固定。考虑2e-3大梯度尖峰及未超过基线，暂缓>2e-3或继续整体LR混调。此为实验建议，尚未获得新增训练启动指令，未提交/重启任何任务；研究目标为小样本拟合，不作泛化最优承诺。

## 能量头历史基线可比性审计及梯度峰值更正（用户2026-11-09质询）

已逐项核对三组resolved_config、source_initial.zip/study_source.zip、执行选项、sample/trainmonitor manifests、归一化fingerprint、初始化采样hash、所有step0固定窗口预测、完整210行train_log中的实际LR。除run_id/notes/output_dir外，唯一配置差异为energy_head_learning_rate(.001/.0005/.002)。其他LR全210步逐条相同，headLR逐条严格0.5/2倍，16步warmup均正常（首步head6.25e-5/3.125e-5/1.25e-4）。三组数据manifest与trainmonitor SHA相同、normfingerprint相同，step0所有固定窗口逐条预测一致、采样初始化hash一致、同cloud GPU2/3及同参数device划分、fresh无resume、B16/native4/FP32/K2/210等全部相同。所有已归档模型/数据/损失/solver/训练器/评价数学源文件字节一致；入口差异仅GPU授权/可见性检查和JSON path序列化，队列差异为GPU锁与授权检查，不改变参数更新。新增gpu_policy.py为授权模块。支持“能量头LR控制变量的历史基线比较”。但不是同批重跑基线；仅单seed、无独立完整逐run环境package锁定及全参数初始hash证据，不能称严格逐位复现/已排除运行间波动。建议下一轮加入同期1e-3重跑与候选档，未经用户新启动指令不跑。

必须更正此前梯度论据：speed_report.steps仅末100步（111到210），先前将其max当全程max错误。完整210条train_log的裁剪前全参数norm峰值：基线1e-3为27,148,763.809272（第44次更新）；5e-4为14,609.523183（第50次）；2e-3为16,790.794219（第26次）。此前12.359374/79.46278/8849.661329只对应末100步，不是全程峰值。明确撤回“2e-3全程峰值明显高于基线，因此更不稳定”的推断，也不能把高裁剪前norm当非有限更新；三组均有clip1。210步固定TRAIN A/F数值仍核验正确，1e-3是本次单seed的末端拟合最好，不代表稳定性或泛化最优。先前±25%细调可保留为候选，但原大梯度论据已撤回，应补同期控制验证重复性。

证据logs/lab_remote_setup/head_lr_comparability_audit.json，入口diff保存head_lr_comparability_diffs/。本审计只读实验记录，未启动训练、评价或改运行配置。

## 2026-11-09 用户授权能量头±25%细化，新轮已准备等待空间

用户接受上一条“补同期1e-3基线”建议并要求±25%继续跑；新研究三组按1e-3同期repeat->7.5e-4->1.25e-3串行，各fresh210update，共计划630update，B16/native4/FP32/K2、16warmup、clip1、A+9F、其余LR与67TRAIN/16VAL全部固定。源采用已完成head2的source_initial+study_source冻结快照；模型/数据/损失/solver/trainer/评价源码逐字节核验相同，仅队列增加每组启动前关键依赖/Python版本锁定检查，新controller负责整轮空间门槛和全JSONL梯度统计，不改训练数学。完整依赖名称版本（无credentials/URL）、GPU UUID/driver与源archive SHA均保存。配置差异assert、冻结源码身份、语法和三个配置实际loader检查通过。初始化仍是每参数64元素采样hash，不声称已新增全参数hash。

新根/cloud/zzc/energyWAM/experiments/single_episode_head_lr_refine_20261109T122513Z；冻结代码/root/zzc/energyWAM/logs/frozen_sources/head_refine_20261109T122513Z/code；tmux energywam_head_refine_20261109T122513Z；controller PID1889903正常phase=waiting_for_checkpoint_space，尚未启动GPU训练/没有重复跑旧任务。现cloud空闲49.64GiB，而三组full latest/best+pending+8GiB需要168.17GiB。controller只在整轮预算可用时继续，之后每组还为后续组预留快照。未删除任何权重、未动他人GPU0/1、实验室未改。

已准备具体删除提案/cloud/zzc/energyWAM/experiments/single_episode_head_lr_refine_20261109T122513Z/proposed_weight_retirement.json（proposal_only_not_deleted）：仅本用户已有三组globalLR0.6、headLR0.5、headLR2的latest/best，共6物理快照约137.287GiB，均210步完成，best步分别200/120/0。可保留所有配置/log/evaluation/metadata/source而移除权重，使cloud预计free186.92GiB可开新轮。head0.5的VAL best稍有参考价值，不能自动认定其完整权重已获退休授权；这些完成组的latest/best受AGENTS保留规则保护，仍需本次明确同意或用户扩容。未使用历史cleanup授权自行清除未来完成的头LR组。用户选择前队列继续等待，不冒险启动。部署索引logs/lab_remote_setup/head_lr_refinement_deployment.json。运行后按用户偏好不持续监管。

## 2026-11-09 20:32 CST 明确批准清理已执行，能量头细化正式启动

用户本轮明确回复“同意”及“同意清理这三组权重”。严格按proposed_weight_retirement.json清理globalLR0.6、headLR0.5、headLR2三组的latest/best共6个实际权重文件（约137.287GiB）；逐个核验run_id/210完成状态、物理路径在本组目录内、inode/size和checkpoint步数，无对应活跃train进程。删除前先落盘审批/逐操作清单，删除后验证214份保留的配置、原始日志、评价、source zip、manifest和checkpoint元数据SHA未变；每组新增weights_retired.json标明full_resume_weights_available=false。保留原globalLR1/head1e-3基线latest210/best60，未触及其他用户/数据/预训练模型。正式执行清单/cloud/zzc/energyWAM/experiments/single_episode_head_lr_refine_20261109T122513Z/approved_weight_retirement.json。旧三组metadata不能再被当作当前可恢复权重，只有历史指标仍可使用。

FUSE空闲计数初时未刷新，15秒后实测cloud free186.924GiB，已满足168.173GiB整轮空间门槛，现有controller自动继续，没有重新提交。新队列PID1945441，正式同期1e-3重复组PID1945608，物理GPU2/3 UUID分别db080f3e-1f31-ec8d-821e-326757e8d1e1/aec7e742-e07b-3549-9e8c-0197d2115e73，20:32:12 CST模型已初始化、固定0步评价进行中，正式完成0/210。采样初始化hash1098bd8778560da832704b6f81bba2ab345ef4f51e80312827cd513295ae2546与历史基线相同（每参数64值采样，非全权重hash），normfingerprint同ad89d0de...。每组启动前实际Python/关键依赖检查通过；环境完整manifest在study根，当前Torch2.13.0+cu132、transformers4.57.0、bnb0.50.2，所有三组将用同冻结代码/环境锁定检查。

顺序1e-3同期控制->7.5e-4->1.25e-3，各210update，总计划630，不自动扩展；FP32/B16/native4/K2、clip1、warmup16、A+9F及67TRAIN/16VAL均不变。完整latest/best与后续组/pending空间已预留。comparison_report的梯度最大值扫描全train_log.jsonl，不再用rolling speed_report末100步冒充全程。controller_status.free_bytes是最后一次等待空间时的旧采样，不是启动时/current free；当前空间证明见startup_verification.json与approved_manifest.free_after_backend_refresh_bytes，勿据旧字段误判未满足门槛。

启动证据logs/lab_remote_setup/head_lr_refinement_startup.json及/cloud/zzc/energyWAM/experiments/single_episode_head_lr_refine_20261109T122513Z/startup_verification.json；旧清理记录与新计划/源码/环境均保留。用户不要求持续监管，启动确认后结束观察，后续按需查进度。实验室队列未操作。

## 2026-11-09 21:08:06 +0800 学校服务器本地终端进度面板已部署

用户要求学校服务器上直接运行三套监视进度条。隧道恢复后，部署标准库只读脚本/data3/xyx_zhuzicheng/projects/energyWAM_remote/tools/monitor_studies.py和邻接study_dashboard.json，源脚本scripts/monitor_lab_studies.py；实际在服务器system python3 --once与--json运行通过，4项CPU测试通过（缺失/损坏JSON不虚构完成、真实JSONL更新计数、退出PID识别、固定MSE同一步/ownK配对）。无torch/rich等依赖，不分配GPU，不SSH，不训练/评价/发信号，不启动常驻助手监管。用户运行python3绝对脚本路径即默认每5秒清屏刷新，Ctrl-C只退面板，--once/--json单次快照。显示每组与总进度、运行/排队/完成/失败、PID存活、固定TRAIN/VAL ownK指标及对应评价步、trainlog时间、实际可用latest/best步数、GPU整卡占用（含共享），不把capacity/resume或计划步数计为完成，不混合不同步的A/F，不把只有metadata的已退休权重当可用。

实时抓取：loss权重9在100更新触发原有fixedTRAIN Future>10x baseline保护，非本次面板停止，随后队列自动切换lambda1。21:05:46快照固定TRAIN@100 A.721817/F736.224339，VAL@100 A.674263/F3.957800，保留latest95/best60，实际执行100不等于可恢复到100（尚有5步未存）；原进程94506退出，未重启。权重1同快照8/210。步长原对照210完成、Actionhalf20/210、Futurehalf排队；K2为70/210，K3/K4排队。当前数值以dashboard_latest_snapshot.json实际后续快照为准，不把此记录当常驻监视。

失败详细证据logs/lab_remote_setup/loss_weight9_exit_audit.json；面板部署及SHA logs/lab_remote_setup/dashboard_deployment.json，最新单次真实读数dashboard_latest_snapshot.json。本次仅读取/部署监视工具，未重启/停止任何训练或更改实验协议。


## 2026-11-09T22:14:20.626879+08:00 学校三套实验接入 W&B

用户明确授权把学校服务器loss/steps/K三套实验接入现有W&B项目。采用compshare独立只读日志镜像+W&B sidecar，未将凭据传到学校，未改变训练/评价/排队或停止任何训练。源scripts/sync_lab_wandb.py通过lab-via-pc读取8个正式arm的配置、train_log.jsonl、speed/resource/baselines、固定eval及latest/best metadata和远端权重存在性；不传权重、数据、console/环境/密钥。远端主机必须user-SYS-420GP-TNR。镜像logs/lab_wandb_mirror，输出目录仅镜像配置重写，original_config_yaml及真实远端目录保存在remote_source。每60秒自动抓取，断隧道时保留最后真实数据并记录error，恢复后自动补抓；不能把旧镜像称实时。

独立tmux energywam-lab-wandb运行scripts/run_lab_wandb_sidecar.sh：只对sidecar设置既有17901代理（含大小写），不改共享网络；watch_wandb.py使用configs/lab_wandb_bridge.json，每15秒上传，失败自动重启且按服务端ACK补传。已存在云端采集器未重启。设置status_file避免覆盖原云端bridge状态；远端checkpoint inventory只报告源机可用性，不伪造本地权重，summary标记真实storage host/folder。15项bridge测试通过，包括远端checkpoint存在性变化。首次8组配置/已有日志抓取成功，未开始的组保持排队，首次有train_log后自动创建run。

W&B分组lab-loss-20261109、lab-steps-20261109、lab-k-20261109；随机训练batch与固定TRAIN/VAL独立指标，probe含evalK，不插值指标；历史失败仍按真实resource标记failed。已存在完整原始日志仍在学校，镜像是副本。部署/最新服务端实际确认见logs/lab_remote_setup/wandb_mirror_status.json、wandb_server_verification.json；本地submitted不能冒充server ACK。

W&B接入时新日志同时确认loss权重1在20/210触发fixedTRAIN Action>10x step0保护、K2在80/210触发fixedTRAIN Future>10x保护；均resource returncode=-15/experiment_guard，助手未停止、恢复或重启。此前21:08进度快照已过时。W&B将保留失败run及全部已记录更新。

W&B服务端GraphQL已实际确认（不是本地提交计数）：weight9 run890265933165 failed/100、weight1 a32f898bdfb5 failed/20、步长对照be0214065d6a finished/210、Actionhalf 08bfb79b11a1 running/75、K2 8206f1af7acf failed/80；五个run均有19个probe固定评价summary字段。证据logs/lab_remote_setup/wandb_graphql_ack.json。Futurehalf/K3/K4当前无训练日志故未创建run，后续若队列实际启动并生成记录则自动上传，不能承诺其已运行。


## 2026-11-09 学校实验三组失稳：原始日志核查与可比性限制

用户询问loss1/9与K2为何都停止。按已有镜像原始train_log、固定67TRAIN评价和resource_summary核查，不能把W&B状态当原因：三组均为原有guard对子进程SIGTERM(-15)，不是本次助手停止；退出分类experiment_guard，无已记录非有限梯度，现有证据不是OOM退出。保护阈值为固定TRAIN A或F超过各自step0的10倍。

|组|实际更新|固定TRAIN触发值|step0|倍数|
|---|---:|---:|---:|---:|
|loss9|100|Future736.224339|0.120372730|约6116|
|loss1|20|Action19.534521|0.605347685|约32.27|
|K2|80|Future3.702631|0.120372730|约30.76|

loss1第20步Action评价先触发guard，Future第20步固定评价未生成；不得把step0 Future当20步或用batch Future28.16944替代固定评价。逐步训练：loss9第75次batch Future首次>1（5.14419），80为9.23631，92为99.00366，全程preclip梯度峰值3.703196128e9@92；loss1第13次preclip28977.958，16次batch Future6.25597，20次A9.21843/F28.16944，preclip峰值3.613530354e8@18；K2第79次batch Future2.16600、80次3.09880，preclip峰值1.360297766e6@79。固定评价与变化batch分开记录。

反证：同样K2/lambda9的GPU6/7步长原对照210完成，TRAIN A0.228943/F0.085120；其preclip也有尖峰21161@14，后来恢复。因此不能把单个大梯度当必然失败或归因K2/lambda9本身。失败三组统一native1+CPU optimizer state offload+GPU activation budget0，而完成对照native4+GPU optimizer state+activation10/14GiB；还有硬件及语言层分配差异(loss17/19、K13/23、步长默认18/18)。这些不能与实验因素一起归为严格同运行条件对照；loss9是当前loss研究的匹配control，K2是K研究匹配control，两者自身失败，不能借用另一运行方式的成功baseline宣称loss或K消融结论。

前三个K2/lambda9研究第一batch A0.379343/F0.0904417、初始preclip约0.071362几乎一致，但随后渐分离。符合数值敏感、microbatch累加/算子/分片/8bit状态量化等差异放大的假设，尚不能定位首因；未排除实现问题。CPU卸载本身按设计只移动原dtype状态，既有synthetic4update含resume bitwise gate通过，真实4B仅2update save/resume通过，不是native1/4/卸载开关全轨迹一致性证明。

现有max_grad_norm=1外层裁剪确已配置且代码在optimizer.step前调用；日志param_grad_norm记录裁剪前。内层candidate clipA/F均null、normalization none，可学习alphaFuture仅小幅变化(loss9末1984.576、K2末1989.136)，并非步长参数突然涨大。候选更新由step乘energy gradient决定，外层clip不直接限制内层候选修改与展开Jacobian；这是与日志一致的失稳机制，但初始扰动来源仍未确定，不能仅凭alpha数值大小认定原因。

下一项应先做同GPU/同初态/同样本/同候选种子的短程等价性定位，分别隔离microbatch1vs4及CPU状态卸载开关，逐更新比较候选梯度、裁剪前后参数梯度、optimizer moment和参数增量，再决定超参数搜索。此轮只读分析，没有新训练、评价、恢复、停止或改保护阈值。证据logs/lab_remote_setup/lab_instability_analysis.json，原始镜像logs/lab_wandb_mirror，既有optimizer gate scope见optimizer_offload_startup_report.json。


2026-11-09 补查microbatch与保护条件：B16下micro1累积16次、micro4累积4次，在样本/初态/候选随机数/按样本加权/裁剪时点一致且无跨样本运算时理论梯度等价；现代码按micro实际样本数/B加权、累积后clip1一次，inner E.sum()避免按micro均值缩放，未发现明显漏除B或提前optimizer.step的错误。浮点算子/累积次序/8bit状态等差异和未排除实现问题仍须定位，不能仅凭三个失败共用micro1断言因果。
现guard为单次固定TRAIN方向mean>10x初始即退出，不要求连续多次恶化；能标记当时严重异常，不能证明之后永远不恢复。单窗口分布补查：loss9@100 Future mean736.2243、median3.8767，42/67超过初始mean的10倍，最坏窗口贡献94.30%，去除最坏均值仍42.5768；loss1@20 Action mean19.5345、median1.5260，13/67超过10倍，去最坏仍12.7376；K2@80 Future mean3.70263、median1.16243，33/67超过10倍，去最坏仍3.28606。所以确有失稳并非单个坏点完全制造，但mean对极值敏感且保护无法区分可恢复短暂尖峰。改进建议将非有限/OOM等硬失败与有限误差尖峰的软告警分开，结合连续固定评价与median/P90和异常窗口比例；具体阈值/等待预算须事先约定，公平对照保持一致。此轮未改退出条件、未重启/续训。证据logs/lab_remote_setup/lab_guard_window_distribution.json。


2026-11-09 退出准则历史核对：近期OpenDrawer整体LR、能量头LR0.5/2、头LR±25%均在实际jobs JSON显式max_fixed_mse_ratio10、guard_min_sample_visits320、B16、固定TRAIN每320visits=20update评价；学校loss/steps/K同10倍核心规则、20update固定评价，并非专门对学校micro1加严。三任务full_20261106 jobs42/43同10倍，但guard_min_sample_visits256、固定TRAIN每1024visits（B16每64update），VAL每4096visits，触发机会不同。batch_grid_micro4_20261105四主arm同10倍、guard最少256visits、TRAIN每256visits，换算步数随B不同。故近期对应小样本使用相同核心早停，不可称全部历史实验的完整退出协议一模一样；更早jobs未显式ratio不等于无保护，run_iteration_search默认10/min_update20。smoke/容量/其他入口还需按实际入口/评估有无区分。未修改任何运行中准则。


2026-11-09 用户明确确认“三个失败run都删”：已从W&B删除学校loss9(890265933165)、loss1(a32f898bdfb5)、K2(8206f1af7acf)，删除前验证名称/failed状态并保存summary，delete_artifacts=False，GraphQL三项均null确认删除完成。清单logs/lab_remote_setup/wandb_deleted_runs.json同时是持久排除名单；sync_lab_wandb.py根据excluded_arm_names过滤上传来源，防自动重建；已从独立lab W&B设置移除三组并仅替换日志抓取worker，未停止/重启训练或原云端采集器。新镜像worker独立tmux energywam-lab-wandb-mirror-v2，原energywam-lab-wandb仍为上传器。原始训练/eval/config/checkpoint及日志镜像全部保留。

顺带只读查云端头LR细化：1e-3repeat执行92次后returncode1，runtime_or_signal_error，非10倍guard；console显示保存latest临时checkpoint metadata读取时JSONDecodeError(空内容)。原队列已自动进入headLR7.5e-4，22:30:57日志20次；1.25e-3后排队。此记录不是实时后续进度；本次未处理/重启该保存故障，须后续专项调查，不得混称模型失稳。

学校实时查询2026-11-09 22:33:31 CST：GPU3/4 loss队列complete（两组failed，不是训练完成）；GPU6/7 Actionhalf进程存活97/210，固定TRAIN@80 A0.476469565/F0.104321225，trainlog22:31:15，Futurehalf0 queued；GPU1/8 K2已退出80，K3 waiting_for_idle_gpus 0，K4 queued0。没有为查询/删除W&B而重新启动这些任务。最新dashboard_latest_snapshot.json。

2026-11-09 23:06 CST按用户询问剩余时间实查：学校Actionhalf133/210，近期纯更新47.47秒，估剩1–1.5h；含后续Futurehalf210步整个步长队列估5–6h。历史学校原对照210实际4.35h，估计包含评价/存盘开销。云端head7.5e-4为54/210，近期55.65秒/更新，当前组估2.5–3.5h，含后续1.25e-3共估7–8h。K3 waiting_for_idle_gpus0、K4queued0，无法给全部实验确定结束时间；估计前提无新保护/保存故障/资源等待。未改训练，证据eta_dashboard.json、eta_estimate.json。


## 2026-11-09 23:17 CST 用户授权：GPU6/7步长队列后自动运行K2/3/4

用户明确要求Future步长减半结束后，在学校同一对GPU6/7重做原K对照，沿用原大显存卡配置，不再做GPU1/8、3/4的小显存适配。已实际部署后台依赖队列，tmux `energywam_lab_k67_after_steps_20261109T151717Z`，wrapper PID538622，启动证明phase=waiting_for_preceding_study，当前新组均0/210（总0/630），未进行新GPU训练或评价。前序root /data6/zzc/energywam_lab_studies/inner_steps_20261109，最后arm episode3346_future_half_seed42及其controller完成后自动释放依赖。前序失败/中断则blocked_dependency，避免擅自处理异常。

新输出root `/data8/xyx_zhuzicheng/energyWAM_studies/k234_gpu67_20261109T151717Z`，可用约3027GiB，完整3组latest/best/pending预算180574435511B已核对；没有删除任何权重。代码 `/data3/xyx_zhuzicheng/projects/energyWAM_remote/releases/energywam_k67_native4_20261109T151717Z/code` 从GPU67原步长对照release逐文件复制；所有energywam/model/trainer/solver源字节完全一致，训练入口也保留原版；仅run_iteration_search.py和k_training_probe.py补K3评价/收集/ownK guard和显式probe budgets，再添加独立等待wrapper。source_manifest与两个diff、包版本、原始manifest SHA和arm_definition保存在study内。非K配置已计划/验证与原GPU67control完全相同：4B FP32/B16/native4、原默认18/18层分配、原AdamW8bit优化器状态留GPU、GPU激活cache10/14GiB，保留基线原有CPU激活卸载，不添加CPU optimizer state offload，不更换模型/样本/损失/LR/内层步长/裁剪。seed42、fresh官方权重；alphaA5/F2000（不继承前序减半值）、lossA+9F、LRlanguage1e-5/task1e-4/head1e-3/steps1e-3、warmup16、outerclip1。

顺序：前序Futurehalf完成 -> K4 native4全B16两更新容量profile（单独工程输出，不计正式进度） -> K2 -> K3 -> K4，各210update/3360visits/约50.15epoch，同样数据预算而计算预算不同；每20update固定67TRAIN/16VAL，评价ownK与共同K2，best按ownK的aux16VAL A+9F。原10倍guard、120GiB USS、系统余量24GiB和每arm12h限时保留。K4容量若失败则blocked_capacity，不擅自减microbatch/增卸载/改层分配。已有GPU67 gradient/storage gate沿用，只在其既有验证范围内；不冒充K4全程或新resume测试。

旧GPU1/8 K2失败记录/权重保留；通过原队列pair lock确认K3/K4均未开始、无训练进程后，仅撤空旧pending jobs；原controller/queue自然退出complete，未发进程停止信号。原jobs/status存于旧root/superseded_gpu67_20261109T151717Z，未启动K3/K4标记cancelled。绝不在两处重复提交。ablation_deployment.json原k组移入archived_groups，新k组指向GPU67；服务器本地dashboard已更新，管理CLI status/logs/stop --study k对应新root（stop可核验等待wrapper身份）。W&B镜像会自动发现新名episode3346_k2/3/4_gpu67_native4_seed42，不受原deleted_run名单影响；首次实际训练有日志才创建run，当前不声称已有曲线。

17项本地队列依赖/K3/guard/best/dashboard测试通过；学校项目环境Python3.10.22、torch2.6.0、transformers4.57.0、bnb0.50.2、numpy1.26.4，三份配置实际load通过。证据logs/lab_remote_setup/k67_preflight.json、k67_followup_deployment.json、k67_followup_startup.json、k67_final_queue_verification.json、k67_actual_code_validation.json。用户不要求持续助手监管，确认后台等待部署后结束观察；后续请求时再查。


## 2026-11-10 10:58 CST compshare能量头LR细化队列已结束

当前host cpod-1x51qyjx1lqg，GPU2/3各1MiB/0%，无train_shared_sharded.py训练进程，controller finished_with_failures明确标记队列含失败。1e-3repeat执行92/210后保存latest临时metadata JSONDecodeError退出(runtime_or_signal_error)，latest/best实际文件及meta均为60，未自动恢复。7.5e-4与1.25e-3均210/210、returncode0，末次日志分别01:48:33/05:40:03。固定67TRAIN@210：7.5e-4 A0.335072381/F0.097834878，1.25e-3 A0.382026096/F0.119413745；固定16VAL@210分别A0.677335889/F0.094159039与A0.767809754/F0.091125607。前者本轮TRAIN两方向更低；VAL Action前者更低、Future后者略低。原历史1e-3 TRAIN A0.0881157/F0.075400仍更低，但fresh repeat本轮中断，不能把本轮视为三组完整同预算比较。两完成组latest210，best分别200/160，均完整文件存在；本次只核查存在/size/meta，未做额外加载恢复测试。具体进程结束时间与checkpoint bytes见logs/lab_remote_setup/cloud_refinement_final_status_20261110.json。未重启/停止/评价/删除任何实验。


## 2026-11-10 能量头学习率五档结果与曲线归属确认

对照固定67TRAIN@210、K2的已完成结果，headLR5e-4 A0.2790193/F0.0871804；7.5e-4 A0.3350724/F0.0978349；历史1e-3 A0.0881157/F0.0754000；1.25e-3 A0.3820261/F0.1194137；2e-3 A0.1575008/F0.0853638。因此历史1e-3在本单episode/seed42/210update训练拟合主指标中两方向均最好，作为当前过拟合基准候选；尚不能宣称跨种子稳健最优，新的1e-3repeat因存盘错误在92停止，不能伪装完成重复验证。辅助16VAL@210显示7.5e-4为A0.6773359/F0.0941590，历史1e-3为A0.6828346/F0.1844891；1.25e-3为A0.7678098/F0.0911256。TRAIN赢家不是VAL每方向赢家，不能用TRAIN选择全量训练最优LR。

用户截图包含headscale0p5/0p75/1p25/2、lrscale1历史原基线、headscale1_repeat92中断，属于能量头LR对照相关；episode3346_action_half_seed42是学校Action内层步长减半，须从本图对照剔除。train/action_mse、train/future_mse是变化训练batch，结论按probe/train对应固定210评价；截图y轴裁到1会隐藏尖峰，不能据此判稳定性。精确结果logs/lab_remote_setup/head_lr_final_comparison_20261110.json。


2026-11-10 进一步核查headscale1_repeat在92步报错：console明确completed=92后调用checkpoint -> save_latest -> _save_latest_locked -> json.loads(ckpt_pending.pt.meta.json.read_text())，JSONDecodeError Expecting value line1 column1(char0)。代码流程已先完成torch.load(mmap=True)和full-resume必需key检查，失败发生在新pending权重的JSON旁文件验证阶段，未替换旧latest。随后except删除pending权重/sidecar并重新抛异常，无保存重试，训练exit1，父队列转下一组。保存history成功0、31、60；保留latest/best元数据均60，现有权重各24568519057B（可能别名共享物理空间），日志保留92实际更新。

不能从已删除的pending文件回溯其当时内容，JSONDecodeError char0只证明开头无有效JSON，不能直接断言空文件、网络故障或磁盘满。云盘实际findmnt为fuse.upfs；保存checkpoint.py旁文件普通write_text后立即回读，无旁文件原子发布/fsync/有限重试，属于需要排查的存储可见性/保存容错薄弱处，但未证实FUSE是唯一根因。此次故障不是10倍MSE保护、不是已记录OOM，也不证明1e-3超参数失稳。本次仅读取原始堆栈与冻结代码，未修代码、未重启/续训。


## 2026-11-10 用户授权复跑1e-3与清理不用权重；已修复保存链并启动fresh210步

用户明确要求复跑episode3346_headscale1_repeat_b16_micro4_seed42，允许删除以前无用的checkpoint。已实际清理5组：早期adaptive_shared的fit_K2_B2_A005、fit_K2_B2_Acandidate、fit_K4_B2和旧train_k248_sharded/k4短pilot，以及被fresh复跑替代的中断1e-3repeat(step60)。前4组云端只是指向本项目checkpoint_archive的symlink，已核对精确对应的实际归档文件后同时移除别名与目标；未越界到其他项目。共11个weight路径、5个物理快照87.36644GiB，其中云盘22.88GiB、项目本地归档64.48GiB；实际cloud free72.47->95.35GiB、local25.11->89.59GiB。105个原有小日志/config/eval/source/metadata SHA保持一致，逐组weights_retired.json标记不可恢复。保留历史1e-3基线、完成的7.5e-4/1.25e-3、三任务诊断及其他有参考价值快照，不动数据/Qwen/DINO/学校权重。删除清单logs/lab_remote_setup/checkpoint_retirement_20261110.json及cloud根同名；这是本次明确范围的已完成清理，不授权未来任意清理。旧中断run的W&B summary已实际更新checkpoint_retired/不可恢复，原曲线保留。

保存改动仅I/O：checkpoint_metadata.py以同目录唯一临时JSON写入、flush+fsync、原子replace、目录fsync，并有限重读验证；读取空/无效/旧metadata最多6次、最长约7.75秒后仍失败则抛出，永久损坏不伪装通过。权重torch.save后先fsync再发布JSON；save_latest在替换latest前按payload meta验证sidecar，并保留原失败清理/旧latest保护。未证明历史FUSE空读根因唯一，此为加固，不宣称永不失败。不改变训练数学、优化器格式或resume schema。

20项CPU测试在实际Py312训练环境通过，覆盖瞬时空JSON、永久错配、发布失败保留原JSON、pending sidecar损坏保留旧完整latest、checkpoint/best/save/load/RNG；CUDA隐藏，未用训练GPU。正式冻结源中的CPU实际云盘gate保存step1+best1、连续到2，另从best1恢复到2，模型逐位一致，latest2/best1/resume2均核验通过；只是tiny CPU gate，不能称真实4B2步resume测试。首次gate因旧训练快照不含tests而import失败（尚未建模型/写权重），改成直接tiny模块导入后通过，失败记录也保存在study。

新root `/cloud/zzc/energyWAM/experiments/single_episode_head_lr_storagefix_20261110T031718Z`，code `/root/zzc/energyWAM/logs/frozen_sources/head_repeat_storagefix_20261110T031718Z/code`，tmux `energywam_head_repeat_storagefix_20261110T031718Z`，Python /usr/local/miniconda3/envs/py312/bin/python，physical GPU2/3。新arm episode3346_headscale1_repeat_storagefix_b16_micro4_seed42，从fresh官方4B/seed42第0步开始，不从已删旧step60续训。原配置只改run_id/output_dir，所有其他超参数逐项相同：FP32/B16/native4/K2、A+9F、headLR1e-3、language1e-5/task1e-4/step1e-3、warmup16/clip1、原18/18层与activation预算、固定67TRAIN/16VAL、每20步评价、210update=3360visits≈50.15epoch、原10倍guard/140GiBUSS/12h。完整latest/best/pending预算82397577843B，cloud free已满足。移除旧job为其他并行energyWAM预留的102GB外部reserve（现在无其他云端energyWAM训练），不改变算法预算。

冻结源从原失败组源码复制，除checkpoint.py/run_state.py及新增metadata helper、CPU gate脚本、父队列storage gate/报告描述外全同；source_manifest/source_changes保留，model/solver/loss/trainer/entrypoint不变。父控制器启动前强制检查storage_gate_result.passed。初始64值/参数抽样hash1098bd8778560da832704b6f81bba2ab345ef4f51e80312827cd513295ae2546与旧组相同，stats fingerprint相同；第0步67TRAIN/16VAL、K0/K2两方向共332个预测MSE逐条差异0，未冒称full初始参数hash。正式4B第0步完整24568519057B checkpoint已通过save_latest真实payload/key/meta校验，并有latest0/best0；正在实际开始更新。后续真实完成数以startup_verification.json/train_log为准，不把0步保存当训练完成。

W&B已登记新独立来源，首个真实train_log后由现有bridge自动创建曲线；数据/命令/来源/环境/退出及checkpoint日志保存在新目录，旧失败日志保留。部署与验证logs/lab_remote_setup/head_lr_storagefix_deployment.json、head_lr_storagefix_startup.json，study/preflight_validation.json、step0_baseline_comparison.json、metadata_storage_gate/result.json。不修改运行中的学校队列；启动确认后按用户偏好不持续监管。

正式复跑启动最终确认：实际train_log已有2次参数更新，训练PID2540741存活于physical2/3；首个完整latest/best均step0、24568519057B并通过保存器全状态key及metadata一致性验证。第0步全部332个K0/K2 TRAIN/VAL方向记录与旧组prediction_mse逐条差0；这是预测检查而非fullweight hash。启动证据head_lr_storagefix_startup.json。


## 2026-11-10 11:33 CST 学校GPU6/7实际状态：Actionhalf完成，Futurehalf外部显存竞争OOM，K依赖阻塞

实际远程dashboard进程/日志确认，步骤研究controller complete但Futurehalf失败；新K67 wrapper为blocked_dependency且已退出，K2/K3/K4各0/210，K4容量profile亦未执行，不能称新K已跑。Actionhalf于00:38:32正常结束210/210，固定TRAIN@210 A0.1513523467/F0.06794667085，16VAL A0.8653279975/F0.2840123030，latest210/best60。原control210 TRAIN A0.2289433048/F0.08511979878，VAL A0.9455018081/F0.3077017516，latest210/best140；本轮Actionhalf两方向TRAIN/VAL终点评价均更低，但仅一seed/一episode，不替代泛化结论。

Futurehalf00:38:32启动，67TRAIN/16VAL step0评价及latest0/best0保存成功；00:46:13首个train_step内term.backward() CUDA OOM退出(returncode1, gpu_memory_allocation_failure)，实际完成更新0/210。报错逻辑GPU0即授权physical6：整卡44.39GiB，仅35.38MiB空闲；本训练进程31.92GiB，同时另PID774703占12.32GiB，申请34MiB失败。不能误归因Future步长参数失稳或把零步评价当学习结果；底层触发是共享显存竞争，后续若要排查为何同卡出现另一进程应另核验，未揣测他人意图/归属。当前11:33 GPU6用12639MiB、free32819MiB；GPU7用1748MiB、free43711MiB，仍有非本研究的其他CUDA进程。未停止他人进程，未改配置、恢复Future或重启K。

当初后续wrapper要求Futurehalf和其controller成功完成，故前序失败后按已部署逻辑blocked_dependency，现没有本研究GPU6/7训练在跑。日志/step0完整checkpoint仍保留。证据logs/lab_remote_setup/lab_status_20261110.json与future_half_failure_20261110.json（真实堆栈、resource_summary、GPU与apps快照）。本轮仅用户请求状态查询，不把此前授权改成自动调整显存配置/跳过失败前序。


## 2026-11-10 用户指定学校GPU0/7完成Future步长减半；独立补跑已部署，验证中

用户给出GPU0空闲、GPU7约1748MiB其他账号上下文的快照，明确要求在0/7跑完Futurehalf。已实时确认UUID0 GPU-199fb0a3-1013-d2e1-fece-6e075f7477a6、7 GPU-4de65a87-f820-da66-782e-0aab5d22e243，hostname user-SYS-420GP-TNR，CPUavailable约294GiB，data8约3027GiB。使用原环境Py3.10/torch2.6，不复制密钥/凭据，不改系统/他人进程。GPU7共享按用户给出占用后指定该卡解释为授权；GPU0保持独占计算进程检查，启动freefloor0=31000MiB/7=41000MiB；进程内UUID可见顺序7,0（大卡为逻辑cuda0）。

新root `/data8/xyx_zhuzicheng/energyWAM_studies/future_half_gpu07_20261110T034637Z`，code `/data3/xyx_zhuzicheng/projects/energyWAM_remote/releases/energywam_future07_20261110T034637Z/code`，tmux `energywam_future07_20261110T034637Z`，study名future07已接入原管理CLI和服务器本地dashboard，W&B按现有镜像机制首次真实train_log出现后自动建run。不覆盖旧6/7失败组，原latest0/best0保留；新组fresh同官方4B/seed42，仍只计划210update。正式配置的训练参数全部沿用旧Futurehalf：B16/native4、FP32、K2、alphaA5/F1000、有效Future先验系数不变、A+9F、原LR/clip1/warmup16/10倍guard、固定67TRAIN/16VAL每20update、完整latest/best且best按auxVAL A+9F。模型仍完整36层，不引入CPU optimizer state offload。

内存适配尝试均仅工程验证、正式0更新：1)GPU0作为主卡、18/18层、GPU激活cache0/8GiB，在真实4B梯度gate即RTX32GiB OOM，未跑formal；2)换GPU7主卡、18/18层、cache4/0GiB，梯度gate430张量通过（A relative6.98e-5/F4.83e-5、SGD lr0、4样本head_override .001，不是正式Adam训练），但真实B16/native4/Adam8bit容量首步backward在逻辑GPU1=物理0 OOM，peak物理0=32075MiB/7=43881MiB。两次原始日志/config/资源/源码快照均归档failed_setup_attempts/primary0_oom_*和primary7_18x18_oom_*，不当作已完成更新。

3)仅将2个语言层从物理0移到7：GPU7=20层、GPU0=16层，36层全保留；GPU激活cache改0/0、原CPU激活保存（参数仍GPU驻留，优化器状态也仍GPU）。为此只在原frozen cfg增加language_model_layer_counts字段、原trainer分片调用传该字段、parallel.py使用已有验证的显式counts选项；不更改optimizer/solver/loss/math/precision/batch/K。源diff4项为config.py/trainer.py/parallel.py以及共享卡检测用run_iteration_search.py，source_manifest和diff保留。正在重做梯度与2次真实B16容量门槛，当前不能声称formal已开始。首次relayout远端脚本有字符串换行转义SyntaxError（解析期，未做改动/训练），已改用原始字符串脚本scripts/relayout_future07_native4.py并成功部署。

这次是混合硬件迁移补跑，GPU顺序/层放置/激活储存有变，不能将与6/7的差异全部归因Future步长。梯度gate使用无更新SGD只验证局部梯度，不是8bit Adam轨迹完全一致证明；capacity以真实B16/native4/Adam8bit单独检验。禁止为了过容量自动降native4或增加CPU optimizer offload。整个运行保留step0及逐步A/F、实际LR、梯度/资源、最新/最佳权重和完整来源配置。16项本地GPU授权/dashboard/K协议测试通过。

后续K2/3/4仍在原GPU6/7，不迁到0/7；依赖已从原失败Futurehalf改为本次future07成功完成，并保留原plan/controller历史。验证重试时K依赖可能再次blocked，需在本次正式启动确认后核对/恢复其等待wrapper（不得提前启动K GPU训练）；K本身仍须等待6/7资源空闲。证据future07_preflight.json、future07_deployment.json、future07_primary0_oom_relayout.json、future07_primary7_capacity_failure.json、future07_20x16_relayout.json、future07_live_status.json、k67_dependency_moved_to_future07.json。


2026-11-10 Future07资源适配最新：19/17（GPU7/0，CPU激活cache0/0，优化器状态GPU）也在真实B16/native4容量首步backwardOOM；RTX物理0(logical1)CUDA31.47GiB、free138.25MiB、own31.13GiB，申请24MiB失败，实际正式0/210。20/16则在物理7OOM（own42.48GiB+其他上下文4x426MiB）；20/16 NVML峰0=26651MiB/7=45257MiB，CPU USS82.27GiB。均无正式optimizer更新，原始capacity/gradient/config/source/资源记录保留。19/17改为先容量再梯度，仅controller工作顺序变更；容量没通过所以19/17梯度复核尚未执行，不冒称通过。新controller可选gradient_gate_after_capacity默认False保持其他队列不变；实现已同步compshare维护代码。

已向用户异步提出资源选择，原因是用户此前要求尽量保持原配置，继续在0/7需改变此前保留GPU优化器状态的方式；当前没有擅自启用CPU optimizer state offload或降低native4。两项具体方案：(1)仍在0/7，B16/native4/FP32/K2/全部训练参数不变，仅将原AdamW8bit状态在前反向期间暂存CPU、更新前原dtype搬回GPU执行原kernel（既有代码短程bitwise gate已证明，实际新native4仍需验证）；(2)等待GPU6/7空闲，按原18/18/10-14激活cache/GPU optimizer state配置补跑。具体提案和完整配置已保存logs/lab_remote_setup/future07_resource_choice_proposal.json及future07_cpuoptimizer_proposal.yaml，标记awaiting_user_choice_not_launched，估计搬移8245862488B是既有真实4B测量而非本次新测。未收到选择前不做依赖训练启动。

K67调度依赖已经指向Future07；在容量失败后其等待wrapper可能再次blocked_dependency，用户决定后需按最终Future补跑root重连/恢复等待wrapper，仅等待不提前启动K训练；不要把K排队0说成运行。之前父队列阶段complete只代表结束，失败保留明确。CLI logs对blocked_capacity现能读profile错误日志。此轮本机cloud1e-3复跑未改动，学校其他用户进程未触碰。


## 2026-11-10 14:33 CST 用户选择等待GPU6/7空闲，原配置Futurehalf已实际排队

用户对资源选择明确回答“等待6/7空闲，保持原配置”。没有启用CPU optimizer offload或降低native4。原0/7所有工程尝试均无正式训练更新，controller已自然结束；已保存原状态/排队文件到user_decision_wait_GPU67，清空其pending full jobs、标记cancelled，仅移出active index并保留全部日志/配置/资源/源码，未发送进程停止信号或碰其他任务。CPUoffload提案仅本次未选择，仍保留未执行草案，不视为用户永久禁止未来方案。

已实际部署新的原配置补跑root `/data8/xyx_zhuzicheng/energyWAM_studies/future_half_original_gpu67_20261110T063335Z`，source `/data3/xyx_zhuzicheng/projects/energyWAM_remote/releases/energywam_future67_original_20261110T063335Z/code`，tmux `energywam_future67_original_20261110T063335Z`。所有Python源逐字节复制原GPU67步骤研究code，未带入0/7的显式分层/CPU优化器/激活cache改动；原Futurehalf YAML仅run_id/output不同，目标环境load+完整非身份字段比较通过。原默认18/18分层、activation GPU预算10/14GiB、AdamW8bit状态驻GPU、4B FP32/B16/native4/K2、Futurealpha1000、beta/先验/LR/warmup16/clip1/10倍guard、67TRAIN16VAL/每20update评价均原样。新arm episode3346_future_half_gpu67_original_seed42，fresh同官方权重seed42，210update。完整latest/best/pending预算约76.74GiB已在data8确认充足，不删权重。先沿用同GPU67已通过native4/storage门槛，待空闲后再执行新2update真实容量profile，之后fresh正式210；工程更新不计正式进度。

实际启动进程：controller1812375、容量queue1812377，queue phase=waiting_for_idle_gpus physical6/7；当前没有本次GPU训练，正式0/210。复用原不共享启动策略，GPU6/7均无其他compute进程才会自动启动，不以GPUutil0%代替空闲，不停止其他用户。用户已选择等待，无需再要求确认同一资源安排，后台tmux不依赖SSH存活。

K67原三组配置/源/资源要求完全不变，调度依赖从Future07改指新Future67成功完成，旧plan/controller/after_study状态存dependency_history/wait_GPU67_*。等待wrapper1812394已实际存活于waiting_for_preceding_study，K正式仍0/630；Future成功后才执行K4容量门槛及K2->K3->K4各210；不将Future减半步长带入K对照。active ablation_deployment.json归档future07、新增future67并同步K dependency；本地服务器dashboard新增Future67原配置补跑，CLI增加--study future67，W&B首次真实train_log后自动注册新run（当前不称已有曲线）。

证据logs/lab_remote_setup/future67_retry_preflight.json、future67_original_retry_deployment.json、study/configuration_verification.json、source_manifest.json。0/7历史失败记录和CPUoffload未执行提案保留。本轮未启动任何其他训练或改变云端正在运行的复跑。按用户偏好，确认等待队列部署后不持续助手监管，后续按需查进度。


## 2026-11-10 14:46 CST 用户要求停止seed42并启动seed43，权重仅保存云盘

用户先授权headLR×1换seed复核，随后明确立即停止GPU2/3现有seed42，最后明确不要把新权重存本机磁盘，允许清理以前checkpoint腾云盘。已向核验身份的旧controller2540586发送SIGTERM，仅其自有queue2540591/train2540741退出，三PID均已不存在，controller14:44:35 CST标interrupted。实际train_log197/210；停止时latest186/best60，非数值guard或OOM。没有为保存额外尾步再训练。

本次授权清理范围仅旧storagefix重复验证这一组的latest186/best60及best物理快照，共3路径/2物理文件45.76243GiB；62个非权重文件SHA未变，日志、评价、配置、源码和meta全部保留，weights_retired.json标不可恢复。历史已完成210步×1原基线和其他实验权重/数据/预训练模型未删。清单cloud experiments/checkpoint_retirement_seed43_switch_20261110.json及logs/lab_remote_setup同名，实际cloud空闲49.57->约95.33GiB，满足新组latest/best/pending+8GiB=76.65GiB预算。未在项目本地创建新训练权重。

新root `/cloud/zzc/energyWAM/experiments/single_episode_head_lr_seed43_20261110T064618Z`，arm `episode3346_headscale1_b16_micro4_seed43`，tmux `energywam_head_lr_seed43_20261110T064618Z`，GPU2/3。直接复用storagefix已冻结源码 `/root/zzc/energyWAM/logs/frozen_sources/head_repeat_storagefix_20261110T031718Z/code`，数学和I/O源码均不改。YAML与旧seed42除run_id/output/notes外唯一差异trainer.seed42->43，完整结构比对通过。fresh官方Qwen3VL4B/DINO，随机任务模块初始化/训练候选随机流/采样次序用seed43，不从机器人训练checkpoint续训；同67TRAIN16VAL窗口清单，固定评价seed2026及末端辅助2027保持不变，仍B16/native4/FP32/K2、head1e-3/language1e-5/task1e-4/steps1e-3、warmup16、outerclip1、A+9F、原GPU分层/激活卸载。210updates/3360visits≈50.15epoch，固定评价20update，guard原10倍，best按16VAL A+9F，全状态latest/best。此试验验证×1跨seed表现，不直接证明跨seed各LR排名。

新输出云盘上CPU tiny保存1/恢复至2 gate通过，模型逐位一致，非真实4B GPU恢复测试。14:46:40 CST controller429083、queue429086启动，实际训练PID429255加载预训练模型，正式更新尚0，不能把加载或step0评价当训练完成。W&B已由controller注册原project独立来源，首次真实train_log后创建；当前未声称曲线已上传。准备脚本scripts/prepare_head_lr_seed43.py有防重复receipt，源代码/命令/完整config/manifest/环境/清理及gate记录已保存，启动后日志继续自动记录。


## 2026-11-10 18:13 CST seed43查询：80步因Future固定训练集10倍保护停止

实时核查seed43实际80/210，训练PID429255已不存在，云端GPU2/3均1MiB/0%。最后逐update日志16:12:17 CST，controller16:13:54 CST结束finished_with_failures。queue明确Fixed train future MSE at update80 exceeded10x其step0；SIGTERM -15来自已部署保护，非OOM/存盘报错/手动停止。固定67TRAIN A/F：0步0.60534770/0.12037273；20步0.60552320/0.28604783；40步0.60553770/0.14909368；60步1.01258919/0.91346051；80步0.60538480/1.71115927（Future初始14.2155倍，阈值1.20372729）。16VAL@80 A0.66561145/F2.62666807。最新训练batch@80 A0.69713652/F0.87796336，与固定TRAIN评价不同。完整latest61/best0文件及meta均存在，各24568518993B；仅核查存在/大小/step，未新增加载或恢复测试。80实际更新不等于能从80恢复，未保存的62–80不能用现有权重恢复。

这说明headLR1e-3在seed43下未通过现有稳定性准则，不能把seed42的好结果称跨seed稳健；不证明每seed均失败，也不能从提前停止推断若继续永远无法恢复。未重启/放宽保护/额外训练。真实状态快照logs/lab_remote_setup/head_lr_seed43_latest_status.json，所有原始日志与checkpoint云盘保留。


## 2026-11-10 18:28 CST seed43允许有限MSE超阈值继续：独立完整断点续训已启动

用户明确同意“出现这种情况别退出继续往后训练”。本次仅seed43诊断job设置fixed_mse_guard_mode=warn：固定TRAIN10倍阈值保留作为持久告警，有限大梯度仍按原全局clip1；非有限训练loss/参数梯度、固定评价NaN/Inf、OOM、原USS140GiB/12h资源限制仍会停止。默认其他job仍stop，学校队列未改。scripts/run_iteration_search.py增加显式mode判定和fixed_mse_warnings.jsonl去重告警，6项guard测试通过。冻结旧K2 guard不支持evaluation_k参数，首次CPU预检在训练启动前抛TypeError；已适配原接口，记录preflight_attempt1_failure.json；修复后冻结代码stop/warn/nonfinite验证通过。

root `/cloud/zzc/energyWAM/experiments/single_episode_seed43_warn_continue_20261110T102649Z`，arm `episode3346_headscale1_seed43_warn_resume61`，code `/root/zzc/energyWAM/logs/frozen_sources/seed43_warn_20261110T102649Z/code`，tmux `energywam_seed43_warn_20261110T102649Z`，physicalGPU2/3。源码复制seed43原冻结版本，仅scripts/run_iteration_search.py不同，核心训练/优化器/I/O/采样/评价源码均相同。YAML只改run_id/output，仍seed43/head1e-3/B16/native4/K2/FP32/A+9F/warmup16/原LR及GPU卸载。恢复原seed43完整latest61（模型/Adam/scheduler/RNG全状态），目标global210，新增149次更新；62–80重放不冒充首次新增训练，原run失败80及原日志保持不变。新目录继承原0–61逐步日志及<=61固定评价，continuation_provenance明确来源；未复制原62–80日志为新结果。实际恢复后会重新评价61。

best继承原auxVAL A+9F选择的step0，独立generation用硬链接共享完整物理快照，后续轮换只动新目录。新latest初始是指向原latest61的symlink，滚动保存os.replace替换别名，不写源checkpoint；原失败组latest61/best0均保留。torch.load CPU mmap检查两份payload全状态必需keys、seed43、step61/0和metadata一致通过，未称GPU恢复已完成；实际startup证据另记。新code source_manifest/source_changes及source_resume.zip、配置/数据清单/环境/命令已保存。

云盘空间按此前用户允许清理旧checkpoint授权，仅退休已完成headscale1p25的latest210/best160及物理快照45.76243GiB；73个非权重文件hash未变。原1e-3基线、0.75组、seed43恢复来源保留，不动数据/预训练权重；1.25现在不可从原checkpoint恢复。清单checkpoint_retirement_seed43_warn_20261110.json保存在cloud实验根及本地logs/lab_remote_setup。cloud实际空闲恢复约95.32GiB，保守预留新latest/best/pending+8GiB，所有新权重只写cloud。

启动前父controller/storagegate检查通过，18:28 CST已提交tmux；后续确认具体PID/恢复step/首次新增更新再写startup，不把继承61条日志当本次已执行61更新。W&B新独立来源由controller注册，同项目，曲线0–61为明确继承历史，其后为真实重放/续训。本次只授权到210，不自动延长或附加seed。


18:33 CST续训启动实证：训练PID2666512/GPU2/3已完成第62次更新，开始63；本次新增1步、61步继承。第62步Action/Future MSE、param_grad_norm、更新后alphaA/alphaF与原seed43第62步五项逐值精确相同，实证fullresume首步复现。恢复61固定TRAIN Future已超过10倍，但队列写fixed_mse_warnings.jsonl后继续，未退出，告警模式实际通过。首步106.406秒，期间CUDA allocator打印96MiB分配OOM重试警告，但未抛终止异常，最终完成更新；不可将警告写成训练OOM失败。新W&B bcd72c072283已创建，0–61为继承历史。证据seed43_warning_startup.json及新run/startup_verification.json。确认启动后不持续助手监管。


## 2026-11-10 21:05 CST 用户要求停止续训并完成短重放诊断

已向核验身份的seed43 warn controller2666265发送TERM，实际停止189步（最后记录），latest171/best0保留；原始seed43 latest61/best0也保留。旧queue/train2666347/2666512退出、GPU2/3空闲，controller状态interrupted。停止记录logs/lab_remote_setup/seed43_stop_for_diagnostics.json。

独立4步诊断root `/cloud/zzc/energyWAM/experiments/seed43_diagnostic_replay_20261110T130428Z`，output `/cloud/zzc/energyWAM/experiments/seed43_diagnostic_replay_20261110T130428Z/seed43_replay61to65_window_module_trace`，code `/root/zzc/energyWAM/logs/frozen_sources/seed43_diagnostics_20261110T130428Z/code`，tmux `energywam_diag_seed43_20261110T130428Z`。从原seed43完整61重放62–65，覆盖64梯度峰；不更改原训练数学、优化器、batch16/native4、seed43、K2、clip1或数据顺序。采用既有profile诊断模式跳过固定评价和新大checkpoint保存，保留源完整61用于完整重放；只保存小型final_parameter_samples，不伪装成可恢复的最终权重。诊断不计主210预算。原61是已异常状态，不能据此声称定位训练最早发病时刻。

新scripts/diagnostic_replay_observer.py通过外部observer记录真实训练窗口每次内层更新MSE/修改RMS/候选梯度/energy、候选外层adjoint、每个microbatch每个参数incoming gradient和累积梯度、全局裁剪前后各模块norm、CPU副本计算每个参数完整实际更新norm/相对norm（非采样估计）、输入范围/finite/mask。所有诊断只detach读，不增加backward或optimizer更新。scripts/run_seed43_diagnostic_replay.py先检查完整model/optimizer张量keys形状dtype及首尾64值、完整scheduler/RNG，再四步逐项对照原loss/梯度/alpha；任一不精确复现则报错停止诊断，避免将观测引入的轨迹差异误当根因。

CPU tiny observer测试通过：同初始与batch，两组loss、gradnorm和最终全参数逐值一致，16个逐window内层记录、clip后norm<=1。原frozen核心源码逐字节复制，只增加observer/runner/audit辅助，source_manifest/diagnostic_source.zip/完整plan/config/data/environment/command均保留。preparation首个PID检查因旧controller/proc空cmd条目拒绝，尚未启动，无新训练；核验原训练已退出后正常启动，记录preparation_attempt1.json。诊断资源预算45分钟/USS180GiB（包含观测用CPU参数副本），GPU仅2/3；不影响0/1或学校。W&B登记独立diagnostic组。现在只是启动，完成/结果须按audit_status/replay_comparisons与实际trace确认。


## 2026-11-10 21:13 CST seed43短重放已完成：窗口、内层、模块实证

停止旧续训189步、保留latest171/best0后，从原完整61重放62–65，GPU2/3，21:05:02–21:13:29 CST正常完成returncode0；4步A/F、原gradnorm、两alpha五项均精确复现，源checkpoint未变。GPU2/3结束后各1MiB，无本实验进程。未追加训练或新完整checkpoint。

64步原preclip74783008.850448，postclip0.999999978。主要incoming参数梯度来自Future方向episode3346起始帧0/84/72/168四窗口组；任务模块preclip71974699，主干20301242，能量头28592.65。最大参数梯度action_mlp.2.weight5506万、action_projector.2.weight3038万；Future使用Action条件，因此参数名字含Action不表示Action loss主导。任务模块占平方梯度总norm约92.63%，不等于唯一根因模块。64步Future frame84 MSE .094163→5.602920→4.076109，最后候选反传到一次更新后候选的损失梯度RMS放大23487倍；frame264 .084723→40.575641→.122778，中间偏离后返回且并非最大参数梯度组。

全局clip有效，64步真实参数更新相对范数主干.013254%、任务.151873%、能量头.292640%；不能据此保证内层映射稳定。128条终点候选梯度缩放解析核对最大相对误差1.772e-7，已记录目标/DINO/mask无非有限值，未完成全数据或高阶有限差分审计。窗口ID由candidate seed+冻结采样+manifest精确还原，原始null trace保留。61已异常，无法定位最早起因；无逐窗口完整参数梯度，不宣称唯一坏帧或唯一bug。

详细报告与原始证据：`/cloud/zzc/energyWAM/experiments/seed43_diagnostic_replay_20261110T130428Z/seed43_replay61to65_window_module_trace/diagnostic_report.md`，同目录diagnostic_summary.json、diagnostics/*.jsonl、replay_comparisons.json。建议后续做高阶导数有限差分及Future步长单变量（保持二次先验有效系数固定），尚未启动。


## 2026-11-10 21:26 CST 新seed44对照已启动（仅seed变化）

用户要求再换一个不同于42/43的seed，并明确“之前的什么checkpoint都可以删了”。选择seed44，在compshare物理GPU2/3从官方Qwen3VL4B/DINO重新初始化；不从seed43异常checkpoint续训。root `/cloud/zzc/energyWAM/experiments/single_episode_head_lr_seed44_20261110T132500Z`，arm `episode3346_headscale1_b16_micro4_seed44`，冻结源码 `/root/zzc/energyWAM/logs/frozen_sources/head_repeat_storagefix_20261110T031718Z/code`，tmux `energywam_head_lr_seed44_20261110T132500Z`。controller235788/queue235790/trainer235955已启动，当前仍加载/初始化，未把启动当作已完成更新。

YAML与seed42 storagefix、seed43 original逐结构对照，除run_id/output/notes外唯一差异trainer.seed=44；源码直接复用同冻结版本无改动。67TRAIN/16VAL固定清单相同，B16/native4/FP32/K2、A+9F、head1e-3/task1e-4/language1e-5/steps1e-3、warmup16、alphaA5/F2000、clip1、原GPU分层及卸载均不变。目标210updates/3360visits≈50.15epoch，固定评价每20update、seed2026、末端辅助2027，best按16VAL A+9F。seed同时改变任务模块初始化、训练样本次序和候选随机流。严格比较原42/43协议，沿用fixedTRAIN10x的stop，不沿用seed43后期单独授权的warn续训；已向用户明确说明停止规则。

云盘空间原72.42GiB，预算76.65GiB（3份约22.88GiB完整快照+8GiB）。本次仅删除已中断seed43 warn续训的latest171，释放约22.88GiB，68个非权重文件hash未变；warn best0、原seed43latest61/best0、seed42原胜出基线等保留。latest171现在不可恢复，记录latest171_retired_for_seed44.json；旧停止时“保留latest171”只描述当时状态。清单`checkpoint_retirement_seed44_start_20261110.json`在cloud实验根和logs/lab_remote_setup。unlink后首次free统计滞后，后续确认95.30GiB。禁止将新权重写本机盘。

CPU tiny实际云盘save1/resume2 gate通过，最终模型逐位一致，不等同真实4B恢复测试。第一次gate因冻结源码未附带tiny_cpu.yaml报FileNotFoundError，在正式训练前失败；改用当前项目fixture绝对路径后通过，两次命令与日志保留。environment_manifest已重新采集，关键包版本与原43一致，旧参考manifest另存。W&B桥已注册独立来源，曲线以真实记录为准。部署receipt logs/lab_remote_setup/head_lr_seed44_deployment.json。


2026-11-10 21:30 CST seed44启动实证：PID235955/GPU2/3完成1/210并开始第2次更新，首步86.915秒；Action MSE=0.4537394344806671、Future MSE=0.09197258204221725。CUDA allocator有分配OOM重试警告但首步完成，未当作终止性OOM。完整latest0/best0及meta已保存，实际4B新断点尚未做额外恢复试验。数据清单/标准化统计与seed43相同，54个Python归档hash完全相同，execution_options仅3项身份路径变化，初始化抽样hash按新seed变化。证据新run/startup_verification.json及seed43_control_comparison.json。确认正常开始后不继续助手持续监管。


## 2026-11-11 11:47 CST seed44失败核查与seed45启动

用户询问seed44是否失稳，并授权尝试seed45。seed44实际完成20/210，2026-11-10 21:49:06 CST controller结束finished_with_failures；原因fixedTRAIN Future@20=1.426836533超过step0=.120372729的10倍（约11.85倍），Action从.605347700到.759320940。preclip梯度最大49065294.872565；20步最后batch A1.561512589/F3.122419596，不等同fixedTRAIN。退出-15是队列guard发TERM，不是OOM终止。仅留latest0/best0，无20步可恢复权重；failure_summary.json与logs/lab_remote_setup/head_lr_seed44_final_status.json保留证据。

按用户既有旧checkpoint清理授权及失败短试验例外，仅退休seed44 step0的latest/best两个别名和best generation，共一份22.88GiB物理快照，保留全部非权重文件并hash核对一致。seed44现在无法从checkpoint恢复；其初始化抽样/seed/source保留。manifest `/cloud/zzc/energyWAM/experiments/checkpoint_retirement_seed45_start_20261111.json`，删除后cloud空闲95.27GiB；原seed43完整61和seed42基线均未删。

seed45 root `/cloud/zzc/energyWAM/experiments/single_episode_head_lr_seed45_20261111T034652Z`，arm `episode3346_headscale1_b16_micro4_seed45`，tmux `energywam_head_lr_seed45_20261111T034652Z`，冻结code `/root/zzc/energyWAM/logs/frozen_sources/head_repeat_storagefix_20261110T031718Z/code`。compshare GPU2/3 fresh官方4B/DINO，不从失败组恢复；210updates/3360visits≈50.15epoch。与原42/43/44除seed及身份路径/notes外YAML相同，B16/native4、K2、A+9F、LR1e-5/1e-4/1e-3/1e-3、warmup16、alpha5/2000、FP32、clip1、分层与卸载不改；67TRAIN/16VAL固定清单，评价seed2026和末端2027，20update评价，原fixedTRAIN10x stop保持。seed同时控制初始化、采样和训练候选随机流。

关键包版本实测与seed44一致，环境manifest刷新；冻结数学和checkpoint代码直接复用，沿用同云盘/同冻结源码已通过的seed44 CPU tiny save1/resume2 gate，未声称本次重测或真实4B恢复新验证。空间预算latest+best+pending+8GiB=76.65GiB通过。新权重仅存云盘，best按16VAL A+9F，全状态latest/best。11:47启动，刚提交不算已完成训练更新；实际进度见queue/train_log。W&B桥自动登记独立来源。receipt logs/lab_remote_setup/head_lr_seed45_deployment.json。


## 2026-11-11 11:51 CST 学校Futurehalf完成，K对照容量阻塞（实时只读核查）

通过lab-via-pc实际连到user-SYS-420GP-TNR，实时dashboard/进程/原始日志核对。最新Future67原配置补跑于2026-11-10 22:01:33 CST controller complete，正式210/210，GPU6/7，无该研究训练进程存活。实验为Future初始步长2000→1000，保持有效二次先验系数.0005，Action5，B16/native4/FP32/K2、seed42及原GPU卸载；不是K对照，也不是本机seed实验。

下表均为各组210update固定67TRAIN和16VAL、evalK2结果，不是变化训练batch，也不是best checkpoint指标。

| arm | 更新 | TRAIN Action | TRAIN Future | VAL Action | VAL Future | 实际可用权重 |
| --- | --- | --- | --- | --- | --- | --- |
| episode3346_steps_control_seed42 | 210/210 | 0.228943305 | 0.085119799 | 0.945501808 | 0.307701752 | latest210/best140 |
| episode3346_action_half_seed42 | 210/210 | 0.151352347 | 0.067946671 | 0.865327997 | 0.284012303 | latest210/best60 |
| episode3346_future_half_gpu67_original_seed42 | 210/210 | 0.083474936 | 0.078191998 | 0.594276000 | 0.116746288 | latest210/best100 |

本轮Futurehalf训练Action最低，Actionhalf训练Future最低；Futurehalf末端VAL两项优于另外两组，但仅单seed/单episode，不证明泛化稳健；其VAL Future .116746仍高于未训练.090122。保留latest210/best100，全状态权重存在性/metadata已查，未额外加载恢复或跑新评价。源目录/data8/xyx_zhuzicheng/energyWAM_studies/future_half_original_gpu67_20261110T063335Z。

后续K67 wrapper于22:01:48自动启动capacity_k4_gpu67_native4，22:06:28结束blocked_capacity。第一步term.backward出现CUDA OOM，logical0=physical6，44.39GiB容量、free54.69MiB、本进程44.19GiB，申请24MiB失败；NVML物理6/7峰45403/44534MiB。此次日志未显示原Futurehalf那种另一PID12GiB竞争，不把旧失败原因套到新K失败。K2/K3/K4正式均0/210，没有损失对照结果；队列现在已退出，不是继续等待会自行恢复。没有因查询重启、改变microbatch/卸载或终止他人任务。

旧Futurehalf首步共享显存OOM/0更新的历史保持，不与新成功补跑混淆。当前GPU6空闲18MiB，GPU7有其他任务12642MiB，本任务未占用。证据logs/lab_remote_setup/lab_status_20261111.json及lab_future_k_details_20261111.txt；最新镜像时间可参考但本次结果来自实际SSH。


2026-11-11 学校三组210步最终可学习步长实查：从远端train_log最后一条global_step209（第210次更新）的diag/alpha_*_after_update读取，不误用更新前值，且latest metadata step210一致。基线Action5→5.027140617/Future2000→1943.319946；Actionhalf A2.5→2.535452843/F2000→1951.613281；Futurehalf A5→5.013432503/F1000→986.013733。变化仍接近各自初始化，不能称收敛到跨初始化共同最优值。原始JSON logs/lab_remote_setup/lab_final_learned_steps_20261111.json。本次只读无训练/恢复/修改。


2026-11-11 用户要求将学校Actionhalf/Futurehalf两组同步W&B。Actionhalf已有run08bfb79b11a1/本地记录210；Future67因实际配置名configs/future_half.yaml不同于arm名，旧sync_lab_wandb.py只按arm名找yaml导致漏注册。已修复远端采集从full_jobs映射配置路径，默认旧路径兼容；非标准配置文件名fixture测试通过。Future原始arm_definition中的resolved_config_yaml先恢复本地镜像配置（源hash记录future67_wandb_registration_fix.json），不改训练配置；只替换只读mirror进程到tmux energywam-lab-wandb-mirror-v3。学校/云端训练未停止或重启。

Futurehalf成功创建run590bca5ec829，https://wandb.ai/zhuzc-zhejiang-university/lte-wam/runs/590bca5ec829，已向SDK提交210步和48个fixed TRAIN/VAL评价文件。此时网络17901代理间歇超时/TLS EOF，SDK仍在finish补传文件；尚未完成服务端全量核验，不能冒称210步已全部ACK。Actionhalf链接https://wandb.ai/zhuzc-zhejiang-university/lte-wam/runs/08bfb79b11a1。已询问用户恢复个人电脑代理隧道，用户要求重发命令。服务器侧转发端口已知17901，个人电脑实际HTTP/mixed代理端口没有记录，不能默认7890。两组本地原始数据完整，后台采集器保留重试；等待网络恢复后需核对服务端history/summary，必要时只重建上传会话，不重训、不重建已授权删除的失败run。


2026-11-11 12:04 CST W&B服务端实查纠正：现有17901通道间歇恢复，GraphQL实际返回Future67 run590bca5ec829 finished、last_uploaded_update210、source_training_completed=true；Actionhalf run08bfb79b11a1却crashed、last_uploaded_update111、completed=false，虽然本地collector_finished/last_step210。不能把本地finish当服务端完整证据。已仅向lab上传器2949926发TERM，由原supervisor重启为770431，利用已有reconcile_submitted_state从服务端ACK补传，未触碰训练或其他人的进程。恢复记录steps_wandb_recovery_20261111.json，服务端回执steps_wandb_server_check_20261111_tunnel.json。增加只读scripts/verify_lab_step_wandb.py准备逐点核验训练和固定评价，首次执行网络TLS EOF，尚未获得history完整性通过结论。

用户截图确认Clash Verge端口7897；PC执行-R17901:127.0.0.1:7897时Connection closed by117.50.181.105:24870。服务器现有17901/12399仍监听，sshd允许转发；listener当时27个未认证连接，effective MaxStartups10:30:100，可能被未认证连接限流，但无该次连接日志证明，不能断言唯一原因。未更改sshd/防火墙/共享代理配置，未关闭旧SSH。下一步可由PC用ClearAllForwardings=yes和-vv单次普通SSH诊断，避免误把此错误归因Clash端口或先盲目改网络。


## 2026-11-11 12:12 CST 学校两组W&B完整同步完成，服务端逐点核验通过

用户PC已成功建立-R127.0.0.1:17902:127.0.0.1:7897（Clash端口7897）。17902实际W&B认证GraphQL200、网页200、Google204通过；偶发TLS错误仍可能重试。仅学校上传器切换17902，scripts/run_lab_wandb_sidecar.sh支持ENERGYWAM_WANDB_PROXY覆盖和--upload-only以复用独立mirror-v3；云端训练、学校训练、其他用户进程/SSH配置均未改。

Actionhalf旧run08bfb79b11a1服务端此前crashed/111，本地finished/210为SDK提交而非服务端确认。为避免公用Api登录重试拖延，停止本项目lab uploader后，用scripts/repair_action_half_wandb.py基于新鲜独立服务端ACK111，SDK resume=must继续原run并调用既有poll上传112–210及缺失评价；未新建替代run，未重训。脚本原始状态/修复记录在action_half_wandb_isolated_repair.json。修复结束后恢复tmux energywam-lab-wandb-17902的正常学校上传器，mirror-v3保留。

最终独立GraphQL history检查：Actionhalf08bfb79b11a1与Futurehalf590bca5ec829均finished、last_uploaded_update210、source_training_completed=true。scripts/verify_lab_step_wandb.py逐项对照原始镜像：每组210次更新的A/F共420点，加固定TRAIN/VAL evalK0/K2共96点，总516点/组；缺失0，数值不匹配0（容差1e-8*max(1,abs(value))）。服务端history的null轴值已正确跳过；第一次验证因null轴TypeError修复后重查，不把异常当通过。证据logs/lab_remote_setup/steps_wandb_history_verified_20261111.json，以及wandb_server_history_08bfb79b11a1.json / wandb_server_history_590bca5ec829.json。

链接：https://wandb.ai/zhuzc-zhejiang-university/lte-wam/runs/08bfb79b11a1 （Action减半），https://wandb.ai/zhuzc-zhejiang-university/lte-wam/runs/590bca5ec829 （Future减半）。此次用户要求的两组同步已完成，不再称待补传/待核验。后续新数据自动上传依赖该PC窗口；已经上传的数据不受其关闭影响。


## 2026-11-11 12:20 CST 双步长减半已在学校GPU6/7后台排队

用户明确授权“两者都half，盯着学校6/7，都空了就跑”。已实际部署，不是仅给建议：root `/data8/xyx_zhuzicheng/energyWAM_studies/both_half_gpu67_20261111T041907Z`，arm `episode3346_both_half_gpu67_seed42`，tmux `energywam_both_half67_20261111T041907Z`，controller PID3800276，容量queue PID3800278；实时启动状态waiting_for_idle_gpus，正式0/210、容量0/2。预检GPU6仅18MiB无compute进程，GPU7有其他进程3564589约12616MiB，因此没有占用GPU开始训练。两张卡都无compute进程后自动2步独立capacity profile，通过后fresh正式210，工程步数不算正式预算。后台在学校tmux内，SSH断开不停止等待/训练；不启用sharing，不结束他人任务，不因本次排队重启此前blocked_capacity的K队列。

协议：Action初始alpha2.5/Future1000，可学习范围分别1.25–3.75/500–1500；framework quadratic_action_weight=.5/future_weight=.5，除以初始alpha后有效先验仍.2/.0005。对已完成Futurehalf配置逐结构比较，仅Actionalpha5→2.5和维持先验所需weight1→.5，以及run_id/output不同。4B/Qwen+DINO/seed42 fresh、B16/native4/FP32/K2、lossA+9F、原LR/warmup16/clip1、18/18层、GPU optimizer状态、激活GPU cache10/14GiB与原CPU激活卸载不变。固定67TRAIN/16VAL清单逐字节复制并hash，20update评价、seed2026及末端2027、原10倍MSE stop保持；210update/3360visits≈50.15epoch。

直接复用已完成Future67冻结源码 `/data3/xyx_zhuzicheng/projects/energyWAM_remote/releases/energywam_future67_original_20261110T063335Z/code`，本实验保存全部Python源码zip及sha manifest，无模型/solver/trainer改动。远端load_config、声明变化/先验系数验证通过；Python3.10.22/torch2.6.0/transformers4.57.0/bnb0.50.2/numpy1.26.4。复用已通过原native4/storage gate，仅新做2步容量；若失败则记录blocked，不自动降microbatch或改卸载。完整latest/best按aux16VAL A+9F选择，pending预算76.65GiB；data8空闲约2981GiB，未清理任何学校权重。

源脚本scripts/deploy_both_half67.py已维护在compshare；receipt logs/lab_remote_setup/both_half67_deployment.json和预检both_half67_preflight_20261111.json。首次只读预检因tmux无活动server返回1而中止，改为允许“无tmux会话”后重查通过；未产生训练尝试。部署root在提交前已写本地receipt防重试重复；远端controller root flock防并发。plan/jobs/full config/manifests/gpu policy/source snapshot/commands/arm_definition均在实验root。

已将bothhalf67加入既有服务器dashboard与本地ablation_deployment索引，修复后的W&B镜像会读取正确full_jobs配置，首个真实日志后自动创建run；现在没有声称已有曲线或完成步数。确认后台真实等待后不需助手持续轮询，用户可按需查看。


## 2026-11-11 16:24 CST seed45完整结束核查：210步正常退出，拟合仍有限

本机compshare GPU2/3的episode3346_headscale1_b16_micro4_seed45已正常完成210/210（逐update global_step0–209连续核验、3360样本访问），controller/queue均complete、returncode0，结束时间2026-11-11 15:39:27 CST；train_log最后修改15:35:20 CST，之后完成末端评价/保存。查询时原训练PID589018已不存在，GPU2/3各1MiB/0%；原controller PID588844为已退出的Z（zombie）条目，不在执行训练，身份已存审计。原固定67TRAIN任一方向MSE超过初始10倍的stop规则未变，此次未触发，不是失稳保护或OOM导致停止。

固定67TRAIN evalK2：初始A0.6053476999/F0.1203727293，210步A0.4263393132/F0.1168348105，分别下降约29.57%/2.94%。固定16VAL初始A0.6641318994/F0.0901216990，210步A1.5687278602/F0.1219278334，两项均恶化；不能把正常跑完当作训练稳定或充分过拟合。固定TRAIN Future最高为40步0.4536907772（初始约3.77倍），低于10倍阈值1.2037272929；同40步VAL Future3.6338690761，但该规则按TRAIN判断。变化batch Future峰4.0252242088@48，不与固定TRAIN混淆。

仍出现巨大裁剪前梯度尖峰：update34为354726571.233105（约3.55亿），全210条param_grad_nonfinite均0；配置原有global clip1，未额外测量本run实际clip后范数，不能借用seed43短重放测量冒充本run证据。末步batch A0.5244400501/F0.1265787780，preclip10.602014；末次更新后alphaA5.0410490036/F1943.1478271484。

云盘完整latest210/best0及metadata均存在，各24568519057B；best按16VAL A+9F（K2）选择，best_history仅0步，说明后续评价无综合VAL优于初始值。只核查文件/大小/metadata，不声称新做4B恢复测试；未重启、加训、改变协议或删除权重。原始文件/root为/cloud/zzc/energyWAM/experiments/single_episode_head_lr_seed45_20261111T034652Z。本次证据logs/lab_remote_setup/head_lr_seed45_final_status.json及run/completion_audit_20261111.json，保存真实查询时间、源码证据文件hash、全固定评价时序、末步/峰值及checkpoint metadata。


## 2026-11-11 17:02 CST seed45 W&B显示不全：上传恢复已配置，等待PC代理隧道

用户截图显示seed45曲线不完整。训练原始日志仍210/210正常结束，不因网页曲线短而认定训练中止。实读run_id8973f4a3f421本地上传状态last_step190、最近记录server_ack179、saved_step178，无collector_finished；这三个数是历史本地上传记录，不是本次服务端实时查询结果。17902 TCP Connection refused；17901可连接但W&B认证请求超时；无代理直连HTTP403。服务端history本次尚未读成功，不声称最终缺失范围已确认。截图横轴0–600/纵轴上限2会压缩和截断显示，与漏传分别处理。

仅给主configs/wandb_bridge.json新增proxy_url=http://127.0.0.1:17902，账户/project/sources不变；原配置备份cloud_wandb_proxy_before_*.json。scripts/watch_wandb.py支持每配置显式覆盖四个HTTP/HTTPS代理环境变量、保留NO_PROXY，首次连通性检查前生效；无字段不改变环境。17项bridge测试通过。仅supervisorctl restart energywam-wandb，旧2678363已停止、新3724864运行，当前configured_proxy_unavailable等待网络；学校mirror/uploader、所有训练进程/配置、系统SSH/代理配置未动。用户已获PowerShell单行-R17902:127.0.0.1:7897命令并待回复。

准备scripts/verify_wandb_source.py只读独立核验：seed45每步A/F/preclip梯度630点，加固定TRAIN/VAL evalK0/K2共96点，总726点；真实本地日志210连续更新。完整/缺失/数值不匹配三种无网络fixture通过。后台tmux energywam-seed45-wandb-verify最多等待6小时（非训练监控），通过后保存logs/lab_remote_setup/seed45_wandb_verified_20261111.json；目前ProxyError，不能声称已补传或校验成功。主上传器网络恢复后按服务端ACK继续原run，不创建替代run，不重训。恢复记录seed45_wandb_proxy_recovery_20261111.json。链接https://wandb.ai/zhuzc-zhejiang-university/lte-wam/runs/8973f4a3f421。


## 2026-11-11 17:23:33 CST seed45 W&B原run补传完成：服务端726点逐项核验通过

用户恢复17902隧道后TCP/HTTP CONNECT恢复，服务器侧TLS仍间歇超时/EOF；用户PC经Clash7897对api.wandb.ai根路径收到404，证明该次PC代理TLS和HTTP正常，非API鉴权结论。独立服务端history终于确认原run8973f4a3f421为crashed、ACK179：实际缺180–210的A/F/preclip梯度93点和固定评价24点，共117点，已存在数值无冲突。证据seed45_wandb_pre_repair_20261111_last_server.json。此前“local190/历史ACK179”与本次实时服务端证明分开记录。

仅停止主energywam-wandb（PID3724864），不动训练/学校上传器。scripts/repair_wandb_source.py检查新鲜独立报告、源正式完成、同source/run、无指标冲突、主writer已退出及互斥锁，resume=must继续同run，从179后重传且重新提交固定评价；未创建替代run/更改训练数据。source主记录最后210及checkpoint210已正常保存。补传SDK一度file_stream POST和storage PUT遭EOF、持续重试，无明确HTTP4xx/5xx或fatal。期间曾请用户换代理节点，但随后自动重试成功，已明确告知无需再切换，不能将成功归因未获确认的节点切换。

最终服务端state=finished、last_uploaded_update210、source_training_completed=true。独立对照210更新A/F/梯度共630点和固定67TRAIN/16VAL evalK0/K2共96点，总726点；缺失0、数值不匹配0（容差1e-8*max(1,abs(value)))。最终证据seed45_wandb_verified_20261111.json；SDK receipt seed45_wandb_isolated_repair_20261111.json仅标提交完成，由独立核验提供完整性证据。verify脚本现在额外保存last_server回执，防后续网络失败覆盖最近真实ACK。

repair/helper+原bridge共29项无网络测试通过。restore_wandb_after_repair.py在确认隔离worker退出后仅恢复主service，returncode0；当前主PID3961028运行、新代理17902，自动核验/补传任务已完成。restore receipt为seed45_wandb_isolated_repair_20261111_uploader_restored.json。训练没有重启，权重未改。链接https://wandb.ai/zhuzc-zhejiang-university/lte-wam/runs/8973f4a3f421。网页若仍截断，可刷新并调整横轴0–210、纵轴自动；坐标范围是显示问题，与本次已解决的数据漏传分别说明。


## 2026-11-11 18:29 CST 服务器停用前GitHub备份快照准备

用户明确授权将当前项目全部代码更新到Tassel51/energyWAM/main，并询问ModelScope需保留的权重/数据。当前工作区所有源码/脚本/配置/测试/文档纳入备份；个人W&B运行配置继续忽略，Git新增lab配置example，不上传密钥。既有部署key可通过GitHub SSH443直连认证，旧17901/17902 HTTP转发本次Git连接不可靠；只在发布命令中覆盖SSH transport，不改系统SSH/账号配置。

另归档被忽略的4套冻结源码、91份源码ZIP（89个不同内容）及独立辅助源码：source_archives/manifest.json共347个源引用、345个独立目标，全部SHA256校验。原始实验文本记录分包备份4733文件（cloud3682+lab_remote_setup1051），545983646B原始内容压为14包62331983B，逐成员字节/hash及归档整体hash校验通过；103个软链接不跟随，排除映射在manifest，密钥/签名URL扫描无命中。采集18:24:13–18:25:49 CST是逐文件采集，无观察到变化，不冒称全局原子快照。四份自定义trainmix meta文件及现有manifests/stats也保留。

新export_historical_sources.py/export_experiment_records.py可重建归档，Git中保留恢复说明。近期17个CPU测试模块88 passed；完整suite尝试190 passed/10 failed/1 skipped因maxfail停止，8旧DINO路径、1缺外部starVLA、1旧cfg=None fixture，详见docs/reproducibility/github_backup_validation_20261111.json；未为备份修改训练数学或启动真实GPU实验。

大文件本次仅盘点，未上传ModelScope、未删除任何权重。现存29独立权重437.15GiB，其中本机checkpoint_archive208.34GiB、cloud228.81GiB；部分cloud链接实际指向本机。优先seed42 latest210/best60、seed43 latest61/best0、旧400步model-only共99.23GiB；数据原始树57.27GiB，trainmix视图与之重叠，学校只有3任务1.016GiB而非全50任务。完整位置/元数据/学校副本核验及优先级见server_retirement_artifacts_20261111.md/.json。推送是否完成以之后的远端SHA核验记录为准，不能将此准备记录称作大文件已备份。


2026-11-11 GitHub备份实际完成核验：完整代码/历史源码/4733份文本实验记录已推送Tassel51/energyWAM的main；远端refs/heads/main与本地备份提交ce56582a1fccf9593908db2fe2279a995868d740完全一致。收据docs/reproducibility/github_backup_receipt_20261111.json。大权重和原始数据尚未上传ModelScope，本机checkpoint_archive208.34GiB等风险仍须按迁移清单处理；不能把GitHub备份成功解释为全部模型数据已异地保存。

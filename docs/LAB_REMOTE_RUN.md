# 实验室远程运行配置

2026-11-11迁移说明：下文保留11月9日建立连接与验证的原始记录。之后的步长对照及双步长减半部署以根目录 `PROJECT_HANDOFF.md`、`HISTORICAL_EXPERIMENTS.md` 和实际队列为准。大文件是否仍在维护机/学校、是否已有副本，见 `docs/reproducibility/server_retirement_artifacts_20261111.md`。Git中个人W&B配置用example模板代替；历史部署receipt与冻结代码按归档manifest恢复，不要将旧“当前”状态直接当作正在运行。

## 当前可用配置（2026年11月9日）

已通过个人电脑建立的两条回环SSH转发，从compshare-gpu免交互登录`lab-via-pc`，实际主机`user-SYS-420GP-TNR`、账号`xyx_zhuzicheng`。个人电脑无需开放SSH服务或传出私钥。原lab-proxy仍未授权云端专用密钥，当前连接依赖个人电脑保持两条隧道；隧道断开后后台tmux任务继续，管理/取回结果需重建连接。

| 内容 | 实际位置 |
|---|---|
| 维护代码 | compshare-gpu：`/root/zzc/energyWAM` |
| 实验室代码指针 | `/data3/xyx_zhuzicheng/projects/energyWAM_remote/current`，指向已核验的唯一release |
| 独立训练环境 | `/data7/zzc/energyWAM/lab_execution_20261109/environments/energywam_py310_v2` |
| 原始数据（只读复用） | `/data7/zzc/energyWAM/datasets/robocasa365/v1.0/target/atomic/OpenDrawer/20250816/lerobot` |
| 当前3轨迹兼容视图 | `/data7/zzc/energyWAM/lab_execution_20261109/assets/robocasa365_target_50tasks`，不具备完整50任务数据 |
| Qwen权重（只读复用） | `/data6/zzc/Qwen3-VL-4B-Instruct`，全部safetensors SHA256与维护机一致 |
| DINO权重 | 复用`/data6/zzc/dinov3-vits16-pretrain-lvd1689m/model.safetensors`；精确源配置放在`/data7/zzc/energyWAM/lab_execution_20261109/assets/dino_source_config` |
| 真实4B验证输出 | `/data7/zzc/energyWAM/lab_execution_20261109/outputs/real4b_resume_smoke_relocated_stats` |
| 原始失败尝试（保留） | 同outputs下`real4b_resume_smoke`，0步，被标准化路径身份校验拦截 |
| 部署/环境记录 | `/data7/zzc/energyWAM/lab_execution_20261109/setup`及维护机`logs/lab_remote_setup` |

**GPU：**物理6/7，均为L40S。NVML与CUDA数字枚举不同，必须按UUID选择：6=`GPU-9a1246f3-2243-0b77-9411-7de2c9e1ea9f`，7=`GPU-4de65a87-f820-da66-782e-0aab5d22e243`。UUID小矩阵计算已通过，并用PID回查nvidia-smi确认实际落卡；旧数字编号试探的结果不作为正确落卡证明。新增host-scoped策略适配，compshare默认只允许2/3的政策保持不变。

**环境：**独立复制旧LDA环境（原件不修改），Python3.10.22、torch2.6.0+cu124、transformers4.57.0，新环境补装bitsandbytes0.50.2/pytest9.2.0，依赖实际版本见environment_validation.json。初次离线克隆因缺包缓存失败，保留setup日志，随后在独立v2目录正常创建。远端24项CPU检查通过；未安装系统CUDA/nvcc。与维护机torch版本不同，因此小验证不作为训练质量公平对照。

**标准化：**未重新拟合。DINO权重SHA256一致后，仅迁移统计文件的DINO路径来源信息，原Action/Future均值、方差、计数逐值一致；新旧fingerprint及迁移证据写assets/stats_relocation_receipt.json，原统计文件和原模型目录不修改。

**真实4B验证状态：通过，2026-11-09 15:30 CST结束。**预算仅2个optimizer update，B1/micro1/K2、2个TRAIN/2个VAL窗口，先保存step1再完整恢复到step2；保留完整latest/best和源码/config，best预声明VAL Action+9Future。已实际完成step1保存并恢复到step2，完整latest=2、best=0；均含model/optimizer/scheduler/RNG/meta。与原B16对照不同，不解释为训练效果实验，验证后不追加完整训练。

在**compshare-gpu项目目录**使用：

```bash
python scripts/lab_control.py status       # 实际状态与完成步数
python scripts/lab_control.py logs         # 最近50行
python scripts/lab_control.py fetch        # 取回日志/配置/评价/断点元数据，不搬23GiB权重
python scripts/lab_control.py sync         # 当前未提交代码生成新release并更新current，不改正在运行的固定release
python scripts/lab_control.py stop-smoke   # 仅向核实UID及命令身份的本次runner发TERM
```

首次启动验证用`python scripts/lab_control.py start-smoke`；本次已启动，重复调用会拒绝覆盖/重复提交。要再次做同样的两步验证，先执行`python scripts/lab_control.py prepare-smoke --name UNIQUE_NEW_NAME`，再`start-smoke`；会拒绝覆盖结果或替换运行中计划。此命令仍只准备B1的两步验证。正式实验的参数/资源预算另行确认。

完整checkpoint保留在远端输出目录，可按需单独scp取回并校验；自动fetch默认排除.pt，避免占用维护机正在运行实验的保存空间。个人电脑可直接用其原有`lab-jump`别名查看相同远端路径。

---

以下为历史连接排查与个人电脑中转备用流程；以本节和实际run_status为当前状态。


检查日期：2026-11-09。维护机为compshare-gpu，实际hostname为`cpod-1x51qyjx1lqg`，代码目录`/root/zzc/energyWAM`。用户希望今后保持代码在此维护，实验通过lab-jump运行，远端用户名`xyx_zhuzicheng`。

## 历史：地址确认后跳板认证失败，准备个人电脑中转

用户已提供：lab-proxy为`tmp_user1@42.194.179.254:32841`，lab-jump为`xyx_zhuzicheng@10.72.234.155:23698`，通过lab-proxy访问。GPU主机应为`user-SYS-420GP-TNR`。公钥已装在lab-jump，未装在lab-proxy。

已在compshare-gpu的`/root/.ssh/config`实际配置两段别名及专用IdentityFile，禁止agent转发。实测跳板网络可达，但`ssh lab-proxy`返回`Permission denied (publickey)`，尚不能访问内网GPU主机。没有索取密码或个人电脑私钥。首次接触的跳板主机公钥保存在独立`known_hosts_energywam_lab`中（TOFU，非独立渠道核验），指纹`SHA256:OfJO9dkc8VK06vORIEi9b9sHEUNbN2ue1u0yPDdgws8`。

用户已允许使用空闲卡，本次tiny CUDA验证优先GPU6/7（L40S），执行时重新检查UUID、计算进程与显存；不启动完整训练。代码目录按用户偏好准备在`/data3/xyx_zhuzicheng/projects/energyWAM_remote/releases/<唯一标识>/code`，实际落地后以sync_receipt为准。home/data3约46GB余量仅是用户提供快照，未在本会话远程重验；不能用于安排完整latest/best/pending训练快照。

### 初次检查历史

compshare-gpu上不存在`/root/.ssh/config`，SSH_AUTH_SOCK未设置。实际执行：

```bash
ssh -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=yes -l xyx_zhuzicheng lab-jump 'hostname; id -un'
```

返回255：`Could not resolve hostname lab-jump: Name or service not known`。这表示尚未到认证阶段；不能据此认定密码错误、远端离线或已有免密权限。

需要用户从**个人电脑**提供lab-jump的HostName、Port、User、ProxyJump/ProxyCommand（如有），以及它是否仅为跳板、实际GPU节点如何连接。不要复制私钥。地址确认后，再验证compshare-gpu到远端的网络、主机指纹及独立认证。

用户已授权使用compshare-gpu专用密钥：私钥`/root/.ssh/id_ed25519_energywam_lab`（权限600），公钥同名加`.pub`（权限644），未修改其他密钥或系统配置。工具check/probe默认在该文件存在时显式使用它和IdentitiesOnly=yes；可通过`--identity`指定其他经授权的本机专用密钥。私钥不进入项目、日志和同步包。

公钥指纹：`SHA256:AFIUWGr/2/0h8B/spuxSTYOEO76y5OGb9S7v6ZTKyn4`。

## 用户待提供的非敏感信息格式

```text
Host lab-jump
    HostName <实际IP或域名>
    Port <实际SSH端口>
    User xyx_zhuzicheng
    ProxyJump <若无则写“无”>

lab-jump角色：GPU主机 / 登录或跳板机 / 待确认
若另有计算节点：节点连接方式，或Slurm分区/账户/申请规则
访问要求：公网直连 / 校园网或VPN / 待确认
SSH主机公钥指纹：若能从可信管理员处确认，请一并提供
```

不需要提供个人电脑IdentityFile内容、密码、令牌。若需要独立ProxyJump主机，请同时给出其HostName/Port/User；跳板与目标是否都允许这枚公钥须分别核实。

将专用**公钥整行**追加到目标账号的`~/.ssh/authorized_keys`，或交管理员配置；不要覆盖原文件。若有跳板节点，则按实际认证链配置。具体身份与主机指纹确认前不自动接受未知SSH主机密钥。

## 已准备并验证

- `scripts/lab_remote.py bundle`：打包当前工作树，包含未提交和未跟踪的代码；排除.git、环境、缓存、符号链接、W&B/认证配置、二进制及超过4MiB文件。包含当前单episode的窗口manifest，不包含数据和模型。
- 同步包内`SOURCE_MANIFEST.json`记录逐文件SHA256；包外`logs/lab_remote_setup/bundle_report.json`记录git提交、工作树摘要、包哈希与文件数。初次打包469文件，逐文件校验通过。
- `scripts/lab_remote.py check`：严格免交互SSH连接检查，保留结果与错误。
- `scripts/lab_remote.py probe`：连通后只读记录主机、GPU/驱动、nvcc、磁盘、环境管理器、Slurm/PBS/LSF工具和已知历史资产目录。不会运行训练或修改环境；检查到调度工具后仍需确认实际GPU分配规则。
- `scripts/lab_remote.py export-probe`：生成独立`logs/lab_remote_setup/lab_probe.py`，供个人电脑中转上传后执行；不依赖项目Python环境。
- `logs/lab_remote_setup/remote_requirements.json`：当前3条episode及实际源文件映射、文件哈希、模型大小、标准化身份、资源参考和待适配项。

在compshare-gpu项目目录执行：

```bash
python scripts/lab_remote.py bundle
python scripts/lab_remote.py check
python scripts/lab_remote.py probe
```

有真实地址时可显式指定（HostKey和认证必须先具备）：

```bash
python scripts/lab_remote.py check --host REAL_HOST --port PORT --user xyx_zhuzicheng
python scripts/lab_remote.py probe --host REAL_HOST --port PORT --user xyx_zhuzicheng
```

有跳板时可添加`--proxy-jump user@JUMP_HOST:PORT`；跳板自身的专用密钥认证配置也需独立核实，不假设目标的`-i`自动适用于每一跳。

## 当前小样本所需资产

| 作用 | 混合视图episode | OpenDrawer原始episode | 帧数 |
|---|---:|---:|---:|
| TRAIN | 3346 | 324 | 281 |
| 辅助VAL | 3423 | 401 | 213 |
| 辅助VAL | 3483 | 461 | 214 |

只用左相机视频和对应parquet，当前6个源文件约0.45MiB；不含构建兼容数据视图所需元数据。不能把原始324误作全局324。先检查远端原始OpenDrawer数据与50任务视图，必要时再按清单传最小子集；标准化继续使用原TRAIN统计，不在验证轨迹重拟合。

当前本机Qwen目录约8.89GB，DINO目录约86.4MB。先确认远端已有版本及完整tokenizer/processor/权重，未授权盲目重复搬运。完整checkpoint单份约22.88GiB；一组latest/best/pending加8GiB余量约76.65GiB，此外需模型、环境和数据空间。

## 若compshare-gpu无法直连：个人电脑中转

已提供可执行中转工具`scripts/lab_pc_relay.py`，运行于个人电脑，仅使用该电脑已有SSH配置；不读取或传输私钥，不打开agent转发。先在**个人电脑**执行：

```bash
scp compshare-gpu:/root/zzc/energyWAM/scripts/lab_pc_relay.py .
python3 lab_pc_relay.py sync
```

sync会从compshare生成当前工作树包、只读检查目标主机身份/磁盘/模型历史路径，将包通过电脑转存到新建远端release，验证包及逐文件SHA256，拒绝覆盖已有release，并把probe和sync回执传回compshare。不会复制模型/数据或安装训练依赖。

确认探针中旧环境Python确实存在后，执行已授权的小型CUDA验证：

```bash
python3 lab_pc_relay.py gpu-smoke --gpu 6,7 \
  --remote-python /data3/xyx_zhuzicheng/miniconda3/envs/LDA/bin/python
```

该Python路径是历史环境候选，不保证当前仍存在；不存在则停止，需根据probe实际环境路径调整。smoke在独立validation目录建立无pip的venv，系统包只读复用，不改旧环境；每卡只做256×256矩阵乘法并验证结果，记录真实hostname/UUID/torch版本/进程和显存。发现所选GPU已有计算进程或显存占用超过256MiB则拒绝运行。结果放在`energyWAM_remote/validation/<唯一标识>/`，并回传compshare的`logs/lab_remote_setup/pc_relay_gpu_smoke_result.json`。这不是真实4B训练验证，也未建立可正式训练的完整独立依赖环境。

脚本的暂存、逐文件校验和拒绝覆盖已通过本地临时目录测试；远端GPU验证截至本记录仍未执行。收到回执后，继续选择非紧张存储、核对完整模型与数据，并按明确预算准备真实模型2步验证。

个人电脑只作传输与提交端，代码仍在compshare-gpu维护；无需向任一服务器复制个人电脑私钥，也无需agent转发。

在compshare-gpu先执行`python scripts/lab_remote.py bundle`及`export-probe`。包的确切文件名在`logs/lab_remote_setup/bundle_report.json`。随后在**个人电脑**执行以下模板（替换包名和已有SSH别名；这些步骤尚未实际执行）：

```bash
# 1. 从维护机取经过校验的代码包和独立只读探针。
scp compshare-gpu:/root/zzc/energyWAM/logs/lab_remote_setup/energywam_code_<实际包名>.tar.gz .
scp compshare-gpu:/root/zzc/energyWAM/logs/lab_remote_setup/lab_probe.py .

# 2. 只读检查登录/跳板机；不在登录节点启动GPU计算。
scp lab_probe.py lab-jump:energywam_lab_probe.py
ssh lab-jump 'python3 ~/energywam_lab_probe.py' > lab_probe_result.json

# 3. 将探针结果带回维护机，继续适配。不要回传私钥。
scp lab_probe_result.json compshare-gpu:/root/zzc/energyWAM/logs/lab_remote_setup/pc_relay_probe_result.json
```

若仅有内网计算节点，后续在个人电脑通过其已有ProxyJump配置连接计算节点，或在lab-jump提交Slurm作业。远端代码包放入新建唯一release目录，先核对包哈希与逐文件清单，再解包；后台使用实际调度器，非调度主机才考虑独立tmux/session。正式提交及日志/停止/结果拉取命令待节点和路径确认后生成，不能把这些模板当成已经部署成功。

结果取回同样走“计算节点/共享输出→个人电脑→compshare-gpu”，先同步日志/配置/评价；checkpoint按需要单独传输并核对SHA256，不重复传整份模型或数据。SSH断开不应依赖个人电脑保持连接维持训练。

## 已核实的本机运行信息

| 项目 | 当前实际位置或限制 |
|---|---|
| 训练入口 | `scripts/train_shared_sharded.py` |
| 实验队列 | `scripts/run_iteration_search.py` |
| 当前能量头研究 | `scripts/run_single_episode_head_lr_study.py`，其中云盘路径是本机专用，不能直接照搬远端 |
| 通用移植准备器 | `scripts/prepare_remote_run.py`，现支持three-task/tiny-weight1/tiny-weight9；尚不直接支持当前67窗口研究 |
| 非PyTorch依赖 | `requirements.txt`；Python3.12环境实测。PyTorch/torchvision CUDA组合需按远端驱动选择，不能盲装本机cu132 |
| Qwen/DINO | `/datasets/models`指向`/cloud/zzc/energyWAM/datasets/models` |
| 数据视图 | `/datasets/robocasa365_trainmix`指向`/cloud/zzc/energyWAM/datasets/robocasa365_trainmix` |
| 标准化 | `stats/trainmix_150k_paper_standardized_v1.json` |
| 本机结果 | `/cloud/zzc/energyWAM/experiments/` |
| 当前GPU策略 | 只允许本机物理2/3；入口只接收0/1或2/3两卡组合，不能未经适配用于单卡、其他编号或Slurm GPU可见设备 |

## 等连通后完成的步骤

1. 确认跳板/计算节点身份和GPU分配方式，优先用实际调度器；有卡并不代表登录节点可直接训练。
2. 仅检查用户可访问的已有Qwen/DINO和数据目录，核对版本与窗口编号；不盲目传输整份数据或预训练模型。
3. 选择用户有写权限的新release/环境/output目录，不覆盖旧实验；创建项目独立环境。
4. 同步经过哈希核验的工作树包；按实际GPU拓扑/分配方式适配入口，不在compshare-gpu放宽本机禁用GPU0/1的规则。
5. 先做远端CUDA小计算，再做有预算的小规模真实模型验证；保存主机名、GPU UUID、命令、源码身份、配置和日志。
6. 确定调度任务ID或独立后台会话后，补齐启动、日志、状态、停止本次任务和结果取回命令。当前这些远端命令尚未生成，因为工作目录、调度器和GPU节点未确认，不能填写虚构参数。

远端代码/数据/环境的实际可用状态尚待个人电脑回执；GPU6/7已获本次空闲验证授权，仍需启动前复查。截至本次检查，助手尚未实际同步远端或执行CUDA验证，所有计划路径不冒充实际落地结果。compshare-gpu原训练未因这次配置请求停止。

## Windows PowerShell 命令

用户电脑为Windows PowerShell、Conda base。使用`python --version`确认解释器，再用`python lab_pc_relay.py sync`。避免Windows的python3别名未执行真实解释器；PowerShell不使用Bash反斜杠续行。CUDA命令写为单行：`python lab_pc_relay.py gpu-smoke --gpu "6,7" --remote-python /data3/xyx_zhuzicheng/miniconda3/envs/LDA/bin/python`，仅在sync成功且远端Python路径确认后执行。2026-11-09截图报错时compshare尚未收到probe/sync/GPU任何回执，不能称同步完成。

## PC上传断线处理与已收到的远端事实

2026-11-09收到只读probe回执，主机/用户符合预期；驱动580.126.20，6/7为L40S且当次无计算进程。/data7约270GB空闲且旧项目数据路径可写，/data6约209GB且旧Qwen/DINO路径存在；这些不是完整性验证。代码上传时Windows scp连接被reset，尚未部署成功。最新版lab_pc_relay.py改用带保活的二进制SSH流上传，瞬时传输错误最多重试3次，并原子落盘/核验哈希；不会自动重试可能创建环境或运算的远端Python。请在PowerShell重新下载脚本后运行sync，勿把旧pending目录当已验证release。

2026-11-09再次中转失败：PC到compshare的SSH24870连接超时，命令未执行，与lab认证不同。已更新relay：内嵌只读probe，不再每次从source获取；只读SSH/代码打包允许最多3次瞬时网络重试，有副作用/计算不重试；持久PC cache保存已验证包，可显式--reuse-cached-bundle（可能非最新）继续；回传失败保留本地回执且不中断后续lab步骤。读重试上限、变更不重试、回传失败保留结果及嵌入probe语法已验证。仍需用户更新PC脚本；网络本身可达性不能由代码保证，尚无sync或GPU完成回执。

## 无需Windows SSH服务的个人电脑双隧道

本机未装Tailscale，实际测试100.69.16.101:22超时。可通过个人电脑现有SSH认证建立两段转发，不需登录电脑或复制其私钥。PowerShell窗口1：`ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -L "127.0.0.1:12398:10.72.234.155:23698" lab-proxy`。窗口2：`ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -R "127.0.0.1:12399:127.0.0.1:12398" compshare-gpu`。两窗口保持连接；端口只监听回环。compshare已准备alias lab-via-pc，连接本机12399并用专用密钥认证真正lab-jump账号。转发依赖服务器允许TCP forwarding，尚未实际建通；成功后由助手`ssh lab-via-pc`验证hostname/GPU，不能据配置存在就称连接成功。

## 2026-11-09 授权正式消融队列

当前部署索引：`logs/lab_remote_setup/ablation_deployment.json`。GPU3/4：Future损失权重9对照与1，各210update；GPU6/7：原步长、Action初值半倍、Future初值半倍，各210update。两套并行、组内串行；统一4B/FP32/B16/native4/K2/67TRAIN+16VAL。先进行梯度与容量门槛，门槛失败不启动正式组。详细当前状态必须查询，不把计划当完成。

```bash
python scripts/lab_ablation_control.py status
python scripts/lab_ablation_control.py logs --study loss
python scripts/lab_ablation_control.py logs --study steps
python scripts/lab_ablation_control.py stop --study loss
python scripts/lab_ablation_control.py stop --study steps
```

结果根目录：loss `/data7/zzc/energyWAM/lab_execution_20261109/studies/future_weight_20261109`；steps `/data6/zzc/energywam_lab_studies/inner_steps_20261109`。每组目录含resolved_config、初始化证据、source_initial.zip、样本manifest、train_log/speed_report、分方向固定评价、资源trace、latest/best及元数据。`comparison_report.json`为汇总；`failed_setup_attempts`保留失败证据。每组按自身辅助VAL A+lambda_F*F选择best，不用跨权重总loss排名。

SSH中断后tmux继续运行。代码冻结到唯一release，后续sync不会改已运行代码。远端没有复制W&B认证；要取回原始记录，可用SSH或rsync选择日志/config/json/source zip，避免默认传23GiB级checkpoint。机器路径和GPU UUID授权均封装在plan及gpu_policy中。

### 2026-11-09 native2授权变更

GPU3/4的Future权重对照改为`shared_microbatch_size: 2`，有效batch仍16，每方向分8次计算。其余数学配置、样本顺序、210update预算均不变，两正式组使用同一执行设置；GPU6/7仍native4。此前native4容量失败已归档，不计入正式训练步数。新native1-vs2梯度门槛和2update完整容量门槛通过后，自动开始weight9对照再weight1；实际运行阶段查询controller_status，不能将门槛测试算入210正式更新。回执`logs/lab_remote_setup/loss_native2_preparation.json`。

### AdamW8bit状态CPU卸载（2026-11-09）

GPU3/4研究启用`trainer.optimizer_state_cpu_offload: true`；默认false，6/7与云端不受影响。保留原AdamW8bit的量化状态、更新kernel和dtype，仅在前向/反向期间将状态存放CPU，step前按参数所属GPU恢复，step后同步卸载。完整checkpoint沿用普通optimizer.state_dict，load后重新卸载，无外部状态文件依赖。原FP32/B16/native1/K2、全局梯度裁剪、学习率/损失及数据协议不变。

新增状态搬运计数`offload/optimizer_state_restore_bytes`与`offload/optimizer_state_cpu_bytes`，记录每次更新的搬运字节数。现有`offload/pinned_bytes`是多个micro/direction累计请求量，并非实时驻留RAM。

启动门槛：小型双GPU原AdamW8bit与CPU存储版4步逐位更新/state/save-resume比较；真实4B两步容量；真实B16/native1完整save1->resume2及latest/best内容检查。全部通过后才进入原210update正式队列。原始门槛与失败证据保留，部署索引`logs/lab_remote_setup/optimizer_offload_deployment.json`，实际阶段查询`python scripts/lab_ablation_control.py status`。正式loss输出路径不变，独立resume输出路径记录在该索引smokeplan.output。

上述CPU状态卸载三层门槛已于2026-11-09通过，正式GPU3/4队列已启动。17:35 CST为weight9第0步评价、0/210；实际进度必须查询。详细启动证据logs/lab_remote_setup/optimizer_offload_startup_report.json，独立resume latest2/best2完整。

## GPU1/8上的K2/3/4对照

用户授权GPU8共享（当时已有fengtuo任务），独立study路径见`logs/lab_remote_setup/k_study_deployment.json`，输出位于/data8/xyx_zhuzicheng/energyWAM_studies，正式各210update，顺序K2->K3->K4。沿用3/4weight9配置和CPU卸载，仅训练/评价K改变；分别记录ownK和共同推理K2，best按ownK的16VAL A+9F。K3评价、选择与guard已补齐测试。

```bash
python scripts/lab_ablation_control.py status --study k
python scripts/lab_ablation_control.py logs --study k
python scripts/lab_ablation_control.py stop --study k
```

共享卡空闲显存会在运行中变化；首个K4容量检查因GPU8共存任务涨至20.57GiB而物理显存耗尽，完成0更新，未启动正式组。失败证据见logs/lab_remote_setup/k_study_shared_gpu_failure.json及study/capacity_profile。是否改为等待GPU8独占空闲，或由用户协调稳定的共享额度，需以最新用户决定与实际controller状态为准，不能把部署完成写成正式训练完成。

K队列随后复查发现GPU8已无计算进程（原PID退出原因未知，助手没有发送停止信号），按已有1/8授权继续。最新资源模式为两卡均空闲才启动，不再共享；数据根不变。为适配RTX32GiB/L40S46GiB，三个K组统一13/23语言层分配，原CPU卸载算法及数学参数不变；GPU8空闲时移除原共享allocator限额。首个共享OOM已归档failed_setup_attempts/shared8_oom_*，实际重启回执logs/lab_remote_setup/k_study_exclusive_relaunch.json。K4容量与K3完整恢复仍须实际通过，查询状态为准。

GPU8共享政策生效后，原正式queue已自动启动K2，PID188599，实际physical1/8，已完成模型13/23分配及67TRAIN/16VAL缓存初始化；当前正式完成仍0/210，第0步评价/初始化阶段。K3/K4自动排在后面，不需要重发命令。证明logs/lab_remote_setup/k_study_shared_formal_start.json。用户要求跑起后不持续监管，启动确认后结束观察，后续按需查询。

## 学校服务器本地进度面板

在学校服务器任意目录运行（标准库脚本，无需激活训练环境）：

```bash
python3 /data3/xyx_zhuzicheng/projects/energyWAM_remote/tools/monitor_studies.py
```

默认每5秒刷新三套队列。`--interval 2`改为2秒；`--once`单次显示；`--json`单次结构化输出。Ctrl-C只关闭面板，不结束训练。显示正式更新计数、固定TRAIN/VAL评价步数及ownK、日志时间、进程状态、checkpoint步和GPU占用；异常或缺失数据明确标记，旧PID不等于进程存活，GPU整卡统计可能含共享任务。


## 学校实验 W&B 自动同步

compshare的tmux `energywam-lab-wandb`独立运行只读sidecar，学校训练不依赖它。60秒取回正式arm小日志，15秒上传；学校侧无W&B凭据。个人电脑两SSH隧道需保持连通，否则W&B停留在最后已取回数据，恢复后自动补传，不影响训练。

```bash
cat logs/lab_remote_setup/wandb_mirror_status.json
cat logs/lab_wandb_bridge_status.json
tail -n 30 logs/lab_remote_setup/wandb_upload.log
cat logs/lab_remote_setup/wandb_server_verification.json
```

W&B每组独立run，分组lab-loss-20261109、lab-steps-20261109、lab-k-20261109。`train/action_mse`、`train/future_mse`为变化batch；固定评价使用`probe/train/*_mse_evalK*`与`probe/val/*_mse_evalK*`，按probe/update看评价步；未运行组不伪造曲线。checkpoint可恢复性指学校源目录，模型文件没有复制到镜像。


## GPU6/7 K234 后续队列（2026-11-09）

当前K研究已迁至GPU6/7，等待步长研究的Futurehalf与其controller完成，再做K4容量检查，然后fresh K2/K3/K4各210。旧1/8待运行组已撤下，新root `/data8/xyx_zhuzicheng/energyWAM_studies/k234_gpu67_20261109T151717Z`，source `/data3/xyx_zhuzicheng/projects/energyWAM_remote/releases/energywam_k67_native4_20261109T151717Z/code`，tmux `energywam_lab_k67_after_steps_20261109T151717Z`。沿用GPU67原native4/18-18层/GPU优化器状态/激活cache10-14配置，模型训练代码相同；容量失败不自动改配置。

```bash
python scripts/lab_ablation_control.py status --study k
python scripts/lab_ablation_control.py logs --study k
# 仅用户决定停止新队列时使用：
python scripts/lab_ablation_control.py stop --study k
```

学校本地进度面板原命令不变；等待状态与正式0/630分开显示。部署回执logs/lab_remote_setup/k67_followup_deployment.json。


## Futurehalf GPU0/7补跑

用户2026-11-10指定0/7；输出 `/data8/xyx_zhuzicheng/energyWAM_studies/future_half_gpu07_20261110T034637Z`，tmux `energywam_future07_20261110T034637Z`。逻辑CUDA顺序7,0，当前容量目标GPU7主卡20层/GPU0 16层、native4/B16/FP32/K2，GPU激活cache0/0，参数与Adam8bit状态留GPU。原先18/18两次OOM的工程记录留在failed_setup_attempts；只有梯度与真实batch容量通过后才formal，不能把profile计为210步进度。

```bash
python scripts/lab_ablation_control.py status --study future07
python scripts/lab_ablation_control.py logs --study future07
# 仅用户决定停止本次运行时使用
python scripts/lab_ablation_control.py stop --study future07
```

原学校本地dashboard命令不变，现多一项Future步长减半GPU0/7补跑。K队列仍使用GPU6/7，依赖该补跑成功结束。


## Futurehalf最终资源选择：等待GPU6/7原配置（2026-11-10）

0/7工程队列已取消、记录归档。当前原配置重试root `/data8/xyx_zhuzicheng/energyWAM_studies/future_half_original_gpu67_20261110T063335Z`，tmux `energywam_future67_original_20261110T063335Z`。保持native4/B16、默认18/18、GPU activation预算10/14、GPU优化器状态，等待6/7均无其他compute进程后自动容量检查与210步正式运行；无需再提交。K67依赖该新Future67成功完成，之后仍原K2/3/4配置。

```bash
python scripts/lab_ablation_control.py status --study future67
python scripts/lab_ablation_control.py logs --study future67
```

服务器本地dashboard命令不变。部署索引future67_original_retry_deployment.json，source/config逐项与原Future67一致。

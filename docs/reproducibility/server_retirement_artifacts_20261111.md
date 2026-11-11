# 服务器停用前：大文件备份清单

检查时间：2026-11-11T18:19:20.264153+08:00。本次仅盘点；**未上传 ModelScope，也未删除任何权重**。GiB按2³⁰字节计算；权重按设备号+inode去重，软链接/硬链接不重复计数。完整路径、别名、metadata和学校实时检查见[JSON清单](server_retirement_artifacts_20261111.json)。

**优先保存自己训练产生、无法直接下载恢复的结果。** GitHub中的代码和报告不能代替checkpoint与原始实验记录。

| 优先级 | 内容 | 现存断点 | 独立大小 | 建议 |
| --- | --- | --- | ---: | --- |
| P1 | 云端seed42整体LR×1基线 | latest210 / best60 | 45.76GiB | 必须优先备份；210步拟合结果与验证最好断点分别保留 |
| P1 | 云端seed43原实验 | latest61 / best0 | 45.76GiB | 保留异常短重放的确切起点；61步曾由历史诊断精确恢复 |
| P1 | 历史400步过拟合 | model_reference，step400 | 7.71GiB | 仅模型权重，优化器已退休，不能完整续训；实际文件在本机盘 |
| P2 | 三任务完整训练seed42 | latest680 / best256 | 45.76GiB | 后续失稳定位、短重放和恢复对照 |
| P2 | 能量头LR×0.75 | latest210 / best200 | 45.76GiB | 已完成学习率对照的可加载结果 |
| P2 | 新seed45 | latest210 / best0 | 45.76GiB | 跨seed对照；跑完但拟合/验证表现并不理想 |
| P3 | 其余旧实验 | 逐项见JSON | 约200.63GiB | 存储和时间允许时也备份；本次不自动清理 |

前三项合计 **99.23GiB**。现存全部大权重共29份独立物理文件、**437.15GiB**：云盘10份228.81GiB，本机`checkpoint_archive`19份208.34GiB。完整断点单份约22.88GiB；已有latest/best的保留政策不因优先级表自动改变。

**特别注意真实保存位置：** 部分`/cloud/.../experiments`中的权重只是指向`/root/zzc/energyWAM/checkpoint_archive`的软链接。只保留云盘上的链接，服务器停用后仍会丢失实际文件。迁移应复制真实权重，并保留别名映射和对应metadata。

云端主要实验根目录：

- `single_episode_lr_20261108/episode3346_lrscale1_b16_micro4_seed42`
- `single_episode_head_lr_seed43_20261110T064618Z/episode3346_headscale1_b16_micro4_seed43`
- `three_task_full_20261106/three_task_b16_micro4_seed42`
- `single_episode_head_lr_refine_20261109T122513Z/episode3346_headscale0p75_b16_micro4_seed42`
- `single_episode_head_lr_seed45_20261111T034652Z/episode3346_headscale1_b16_micro4_seed45`

以上均位于`/cloud/zzc/energyWAM/experiments/`。400步模型真实位置是`/root/zzc/energyWAM/checkpoint_archive/iteration_aware_v1_20261104/ablation_prior1_currentinit1_seed42/model_reference.pt`。

**原始记录和自定义数据定义也必须保存。** 实验根目录除去上述大权重后，共3818个独立文件、约0.519GiB，包含逐步日志、固定评价、配置、源代码快照、诊断和删除清单；不能只留下W&B链接。项目`manifests/`约9.43MiB、`stats/`约0.69MiB，另有trainmix自定义`meta/`约5.52MiB，这些文件决定原任务/episode映射、训练划分和归一化。

| 数据/预训练资产 | 本机真实位置与大小 | 学校目前实存情况 | 备份优先级 |
| --- | --- | --- | --- |
| RoboCasa原始解压树 | `/cloud/zzc/energyWAM/datasets/robocasa365`，61,494,471,178B≈57.27GiB | `/data7/zzc/energyWAM/datasets/robocasa365`仅3任务、1536个Parquet、4608个MP4；共10,776文件/1.016GiB | 50任务全量未在学校核验为完整副本；要继续全量实验则迁移完整数据，或保留精确上游来源再下载 |
| trainmix50任务视图 | `/datasets/robocasa365_trainmix/robocasa365_target_50tasks`；25307episode、14957899帧；链接指向原始树，展开约41GiB | 成功小样本实验使用3episode视图3346/3423/3483，保留全量metadata但只有3条实际轨迹 | 保存映射和meta；备份payload须解引用链接；不要把41GiB与原始树57.27GiB相加当独立数据 |
| Qwen3-VL-4B-Instruct | `/datasets/models/...`实际指向`/model/ModelScope/Qwen/Qwen3-VL-4B-Instruct`，约8.28GiB | `/data6/zzc/Qwen3-VL-4B-Instruct`实存；共同主要文件大小一致，本机额外73B的ModelScope configuration.json | 可从`Qwen/Qwen3-VL-4B-Instruct`重下载，先记录精确版本/配置；本次未重新哈希整个8GB模型 |
| DINOv3 ViT-S/16 | `/cloud/zzc/energyWAM/datasets/models/dinov3-vits16-pretrain-lvd1689m`，约82.43MiB | 成功实验实际用`/data7/zzc/energyWAM/lab_execution_20261109/assets/dino_source_config` | 本轮实际model.safetensors SHA256与学校一致；官方预训练可重下载，重复上传优先级低 |
| 仿真资产 | `/cloud/zzc/energyWAM/datasets/sim`，约22.46GiB | 本轮未检查学校全部仿真资产 | 若继续仿真评估需保留/重下载；不是当前小样本离线训练独有成果 |

学校原始3任务是 **OpenDrawer、OpenStandMixerHead、TurnOnElectricKettle**。本轮对小样本实际使用的3个Parquet和3个左相机视频逐文件SHA256校验，6/6匹配既有receipt；没有将这一结果扩大为全数据哈希验证。公开数据的本机下载源记录是`ryanhoangt/robocasa365_datasets_split_target_source_human`；原始压缩包此前经授权删除，现在保存的是解压数据。

**学校已有独立实验权重：** 基线latest210/best140、Actionhalf latest210/best60在`/data6/zzc/energywam_lab_studies/inner_steps_20261109/`；Futurehalf latest210/best100在`/data8/xyx_zhuzicheng/energyWAM_studies/future_half_original_gpu67_20261110T063335Z/`。三组共约137.29GiB，建议后续独立异地备份。它们是不同实验的结果，**不等于云端seed42/43/45断点已有学校副本**。学校其他断点也列在JSON中；断点step不能当作当前训练完成步数。

这次只核查权重文件存在、大小、真实目标和metadata，没有加载大模型或新增恢复测试；此前已删除的权重不能由残留metadata恢复。ModelScope的大文件上传应作为下一项明确任务，指定存储仓库后上传并核对远端完整性；当前GitHub代码推送不代表这里的大文件已备份成功。

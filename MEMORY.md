# LTE-WAM 项目持久记忆

更新：2026-11-05。用户明确要求长期记住的checkpoint保留规则。

## 训练权重的保留与删除

- 每组训练保留最新完整断点 `ckpt_latest.pt` 和验证最好的完整断点 `ckpt_best.pt`；两者相同时可以共用同一份物理文件。
- “最好”按开训前确定的验证指标选择，分别记录Action和Future误差，不能把最低训练loss或最后一步自动当作最好。
- 新文件写完、校验可读后，才原子替换对应旧版本；其他历史版本再删除。最新和最好默认都应可完整续训。
- 删除权重前说明是否会丢失续训、复现推理或评估能力；不要把训练成果描述成“没有价值的缓存”。
- 不擅自删除最新/最好、数据集或Qwen/DINO预训练权重。用户后续明确授权的清理以当次具体范围为准。
- 清理权重时保留配置、样本清单、源码快照、日志、曲线与评估结果，并记录删除清单及实际释放空间。
- 2026-11-05的整批云盘清理是一次性授权，不代表以后可以再次清空训练成果。历史清单：`/cloud/zzc/energyWAM/experiments/checkpoint_cleanup_20261105.json`。

## 实现状态

`scripts/train_shared_sharded.py --keep-best` 已接入完整latest与独立best保留，写后读取校验和原子best指针。2026-11-05真实4B测试已从第1步完整续训至第2步，验证latest为2、best按验证指标保留在0（并非自动取最后一步）。证据：`/cloud/zzc/energyWAM/experiments/batch_grid_micro4_20261105/fp32_full_resume_retention_smoke/resume_smoke_result.json`。这不表示其他入口或所有batch配置均已验证；新长训仍须显式启用、实测恢复并预算latest/best/写入临时空间。FP32新配置单份完整断点约22.88GiB，不能沿用旧BF16+主权重30.4GiB估计。

## 2026-11-05 补充清理授权

用户明确回复“允许清理恢复测试和明确淘汰的短试验权重”。仅限已通过的恢复测试和有结果证明已淘汰的短试验；四种batch主对照及最终优胜组仍保留完整latest/best，不扩大为再次整批清空。删除前核对无活跃使用、保留配置/日志/评估/metadata，写删除清单并复查实际空间。

已删除`batch_grid_micro4_20261105/fp32_full_resume_retention_smoke`的latest(step2)、best(step0)两个物理快照，约45.76GiB；该两步测试已通过，但现在不能再从其权重续训。原通过结果保留，删除清单为该目录的`retired_full_snapshots.json`。延迟结算后复查，实际空闲增加约45.76GiB，达到约182.49GiB；主对照权重未删除。

本轮batch-grid另由`energywam-checkpoint-audit`只读进程归档轮换metadata并记录观察清单，脚本为`scripts/watch_checkpoint_retention.py`。当前只覆盖指定实验根目录内`batch*_micro4_fp32*`目录；其他目录或命名不自动覆盖。它不删除或修改权重，也不替代最新/最好快照本身。

## 2026-11-05 后续三任务训练目标

用户要求：确定同时计算样本数及有效batch后，继续使用“Future从当前观测的标准化DINO特征初始化＋Action/Future双二次项”的Stage 1方法，扩大到三个任务的完整TRAIN轨迹，不再仅使用少量固定窗口。保存完整latest和按预先确定验证指标选择的best；不启动Stage 2。沿用方法不等于默认续用已过拟合的小样本权重，须记录真实权重起点、采样覆盖、训练预算和验证协议。

排队目标为`configs/experiments/three_task_full_followup_target_20261105.json`；当前batch对照及必要复核完成后，必须激活并实际执行这一后续目标，不因旧goal工具目标文字尚未包含它而遗漏。当前仅登记并审计元数据，尚未启动三任务训练。

## 2026-11-06 恢复记录回退修复

`train_shared_sharded.py`恢复训练时现调用`reconcile_resume_outputs`：断点之后的日志和评估先校验备份到`resume_history`，再回退活动记录，避免复用旧模型的后续评估；权重不变。保留best比恢复点更新时拒绝在同目录回退，须另开目录。W&B记录恢复标记并重放丢失更新区间，旧服务端历史不删除；这部分逻辑有单测，尚未在线实测回退重放。数值保护失败结束标记已在线核实。

真实4B已用原batch32的完整latest98，在隔离目录恢复到99，模型/优化器全部张量结构与首尾样本、完整RNG及调度器检查通过，两方向loss与步长均精确复现原99；源断点身份不变、未写新大断点，不计入主对照预算。证据：`/cloud/zzc/energyWAM/experiments/batch_grid_micro4_20261105/guarded_batch32_resume_audit/resume_audit_result.json`与`restored_state_check.json`。这是恢复能力验证，不改变该主试验在104次更新因数值保护失败的结论。

## 2026-11-06 三任务训练优先级更新

用户明确选择有效batch=16、同时计算4个样本，授权四张GPU执行三任务完整TRAIN实验；不再以前述剩余LR筛选完成作为启动前置。拟每两卡一份模型，seed42/43复核；每份模型四个microbatch累计16个样本才更新一次。当前未启动，仍需完整数据采样/验证准备和持久存储预算；云盘实测约68GiB空闲不足一组latest+best+pending及安全余量约73GiB。不得因此擅删受保护主对照权重。

## 2026-11-06 用户授权原始压缩包清理

用户明确同意核对后删除下载原始压缩包；已核对tar内177599个文件及6个资源zip内文件的解压路径/大小，两个根README差异单独保留，删除7个原始压缩包。解压数据、预训练权重、实验latest/best均保留。清单：`/cloud/zzc/energyWAM/experiments/archive_cleanup_20261106.json`；README备份同级`archive_readme_preserved_20261106`。这是本次压缩包清理授权，不是未来数据删除的通用许可。

## 2026-11-06 用户明确删除三个batch主实验权重

用户明确授权删除batch8、batch32、batch64主实验全部权重（latest/best及实际快照），保留实验结果记录。已执行并逐文件哈希核对配置/日志/评估/源码/metadata未变；这些组不能再从原断点恢复或加载重新评价。batch16和batch8学习率×0.3短实验权重保留。清单：`/cloud/zzc/energyWAM/experiments/batch_grid_micro4_20261105/user_authorized_main_weights_cleanup_20261106.json`。这是上述三组的明确例外，不扩大到其他训练。

## 2026-11-06 三任务训练已启动

用户要求开始训练后，已启动supervisor `energywam-three-task-42/43`，物理GPU01/23各一份共享模型，seed42/43；有效batch16/native4/K2，Future当前观测初始化+双二次项。全部1367条TRAIN轨迹按stride16并补末端，44768窗口，2798次更新（一个窗口遍历）；另78VAL轨迹312固定窗口，每4096样本评价，best依据Action MSE+9×Future MSE，两方向分报；未使用TEST。全TRAIN流式提取、不全量RAM缓存，仅固定评价缓存；初始warmup256样本、其他超参数复用batch16。根目录`/cloud/zzc/energyWAM/experiments/three_task_full_20261106`，计划`training_plan.json`，实时状态`status_42.json/status_43.json`。本条写入时仍在step0基线评价，不能称已有训练成果。预计2–3天起，待实际流式吞吐更新；保留完整latest/best，受资源/数值保护监测。

## 2026-11-06 W&B新转发端口

原7901断开后已发现服务器现有SSH转发17901，认证API200，通过修改仅energywam-wandb supervisor代理并重启上传器恢复。三任务seed42/43独立run为730001df8bb8/8eb062210ec1，服务端核实上传；训练未重启。新端口仍依赖现有SSH转发生命周期，若失联先重新检查服务器现有可用转发，不直接要求用户恢复。

## 2026-11-06 用户授权暂停seed43并运行小样本权重对照

用户要求腾两卡运行16样本反复50遍的Future权重1/9对照。`energywam-tiny-future-weight`控制器已启动：等待seed43的下一份完整latest写后校验/历史提交，再仅向energywam-three-task-43发送TERM，保留全部latest/best，不影响seed42；随后GPU23依次跑权重1、9，各50次更新（有效batch16/native4/K2），复用旧OpenDrawer/OpenCabinet两轨迹16TRAIN+四轨迹16VAL，seed42相同新初始化和数据顺序。不是三任务完整数据实验；warmup16次等其他当前设置一致。主比较为相同更新次数的未加权A/F误差，各组按自身预声明验证目标保留完整latest/best，不能比较不同权重的总loss。根目录`/cloud/zzc/energyWAM/experiments/tiny_future_weight_20261106`，控制/队列状态分别controller_status.json/queue_status.json，安全暂停证据seed43_pause_record.json，结束自动写comparison_report.json。写入本条时控制器尚等待下一次安全断点，须查实际状态，不能把排队当已训练。预计2–3小时加切换等待；不自动延长50次预算。

三任务seed43实际已在121次更新安全暂停，latest完整121、best0均保留，已完成更新未丢失；训练PID2144185已退出。权重1小样本组已由GPU23实际启动（初始PID2216947），权重9串行排队，seed42三任务长训仍在GPU01运行。见tiny_future_weight_20261106/seed43_pause_record.json及queue_status.json。

## 2026-11-06 GitHub交接发布

用户明确要求详细交接并上传https://github.com/Tassel51/energyWAM，已完成main首次推送，首发commit52cb548；远端SHA与本地核对相同。主文档`项目交接文档.md`513行，更新README/requirements，旧README及依赖作为明确历史文件保留；复制固定样本清单和小型复现统计，不传数据/权重/运行日志/个人W&B配置或密钥。53项测试通过，异机准备器在源机验证tiny和3task数据路径/编号，未在对方机器实测训练或干净安装。仓库写入使用项目专用deploy key（私钥在Git目录之外），Git本地SSH设置经现有17901代理，不改变全局Git身份/登录；以后网络失联先检查转发。发布不影响现有训练。

## 2026-11-06 交接文档位置偏好

用户要求主交接文档位于项目根目录、使用中文名。已将旧`docs/PROJECT_HANDOFF_CN.md`移动为根目录`项目交接文档.md`，同步README及依赖说明；今后直接维护这一原件，不另生成多个最新版。

## 2026-11-06 交接文档最终文件名

用户随后要求仍位于根目录，但改英文文件名。主文档现为`PROJECT_HANDOFF.md`，内容保持中文；替代上一条中文文件名偏好。只维护这一份主交接文档，README和requirements引用已同步。

# GPU 2、3 吞吐优化与当前训练方案

当前选择依据是相同第20步断点、相同训练样本形状、有效batch2下的热身后端到端吞吐。
正式训练使用 `configs/gpu23_bi_20261101.yaml` 和 `configs/gpu23_joint_20261101.yaml`，
由 `scripts/run_gpu23_pipeline.py` 管理；`scripts/train.py` 按execution_mode转入新的双卡执行入口。

## 任务基准与数据范围

- RoboCasa365 v1.0 target/human，50个任务，PandaOmron；含atomic/composite厨房任务。
- 数据池：25,307条轨迹，14,957,899帧，合并视图 `/datasets/robocasa365_trainmix/robocasa365_target_50tasks`。
- 本地固定轨迹划分：train22,725 / val1,285 / test1,297；每个划分覆盖50任务，互不重叠。
  规则为每个任务source_episode_index模20：0为test、1为val，其余train；这是项目本地episode划分。
- 统计：训练清单中150,000窗口子集估计动作和未来DINO通道z-score；覆盖50任务，未声称完整训练集统计。
  训练采样池仍是整个train划分，按有效窗口打乱采样。
- 动作：H=16、Da=12；状态raw16经sincos为32；单左相机robot0_agentview_left；未来偏移5/11/16。

## 模型和两阶段首轮

Qwen3-VL-4B语言主干全量训练，Qwen视觉编码器和DINOv3-ViT-S/16冻结；投影层及Energy Head训练。
DINO目标为196个patch×384维；bf16主干，fp32候选/能量/loss，标准化空间N(0,I)初始化。
K=2完整展开，alpha_A=alpha_Z=0.05，eager attention，非重入重计算；内层不做optimizer.step。

| 阶段 | 目标 | 首轮预算 | 学习率 / warmup |
|---|---|---:|---|
| I：bi | 固定Z*优化A；固定A*优化Z；两方向共享E_theta，损失相加 | 总200步，第20步断点续训，剩180步 | 1e-5 / 20步 |
| II：joint | 从阶段I权重初始化，同时优化A、Z | 100步，optimizer/scheduler reset | 5e-6 / 10步 |

AdamW8bit、weight_decay=0、梯度裁剪1；有效global batch=2。
200+100是工程短训和学习信号检查，不是50任务最终收敛预算，不会遍历整个训练集一个epoch。
完成首轮后应根据验证曲线确定长训预算。

## 新执行方式

每张GPU放完整且同步的模型副本，分别只可见自己的物理GPU（rank0→GPU2、rank1→GPU3）。

- BI：两rank读取相同两个样本；rank0做Future→Action、rank1做Action→Future；各自microbatch1累积两个样本。
  第一微批完成的外层参数梯度无损暂存CPU；第二微批结束合并，再对两方向梯度求SUM。
- Joint：每rank处理同一global batch中的一个样本；外层梯度求和后等价于两样本平均。
- 全部内层候选梯度和展开图在本rank完整计算，手工同步的是外层参数梯度；未安装DDP autograd hooks。
- 同步采用256MiB桶，共约31个collective；有效batch、损失、参数范围和K保持不变。
- GPU激活预算每rank3GiB；异步D2H与页锁定H2D，单rank页锁定请求预算128GiB、D2H在途上限1GiB。
  预算是上限而非预分配；实测页锁定allocator峰值约95GiB/rank（含梯度暂存）。
  闲置页锁定缓存超过120GiB时回收，活跃saved tensors仍由allocator持有。

## 实测对比

同一step20断点、同一smoke样本流、有效global batch2；计时包含数据加载和同步，未含周期checkpoint写盘。
首次页锁定内存注册属于一次性预热，应与稳态分开报告。

| 执行方案 | 热身后端到端秒/步 |
|---|---:|
| 原按语言层分卡、pageable CPU卸载 | 约117 |
| 按层分卡、32GiB异步页锁定预算 | 约100 |
| 两方向副本并行、32GiB页锁定预算 | 约82 |
| 副本并行、64GiB页锁定＋分桶 | 49.96 / 47.63 |
| 副本并行、128GiB页锁定＋分桶（选用） | 33.76 / 30.59；均值32.17 |

选用方案吞吐约0.062样本/秒，相对原约0.017提高约3.6倍。
最长训练指令（46tokens，episode20263）稳态34.75秒/步，非有限梯度0，pageable激活保存字节数0。
联合阶段最长指令稳态22.09秒/步，非有限梯度0，pageable激活保存字节数0。
全量随机样本、数据解码、系统负载与checkpoint写盘会影响正式训练速度；不同执行方式的rank RNG流分别记录。
压测loss不用于宣称模型质量提升。

## 验证和恢复

- 29项CPU回归与2项GPU回归通过：完整二阶梯度、按方向求和/按样本平均、微批拆分、Qwen不同长度padding、
  异步copy、归一化、AdamW8bit、checkpoint和每rank RNG恢复等。
- 真实4B多配置完整步通过有限梯度检查；128GiB方案连续3步，最长指令连续2步，joint连续2步。
- 当前Torch2.13.0+cu132的原生Flash/Efficient Attention仍不支持需要的double backward；保留eager。
- NUMA内存策略探测被容器权限拒绝；未更改宿主机设置。
- 第20步的模型/optimizer/scheduler断点保留为 `ckpt_speed_anchor_step20.pt`（硬链接）；性能实验未覆盖正式权重。
- 新模式记录两rank RNG状态；新任务首次完成更新即存滚动断点，其后每10步保存。
  当前恢复不保存DataLoader iterator游标，样本顺序不保证跨重启逐步复现。

## 评测范围

每阶段后用val清单做32样本离线动作MSE检查；小子集不代表全部50任务的均衡成绩。
任务闭环成功率尚需另行运行RoboCasa仿真rollout、多seed和逐任务汇总，遵守所选官方评测协议。
当前训练吞吐benchmark不是闭环成功率评测，也不是完整365任务实验。

## 查看运行

```bash
supervisorctl status energywam-gpu23
cat logs/gpu23_pipeline_status.json
tail -f logs/train_gpu23_bi_20261101.log
tail -f logs/runs/energywam_gpu23_bi_20261101/train_log.jsonl
```

rank1日志在各stage输出目录的rank1子目录；日志中的cuda:0指该worker可见的单张物理卡。
各候选原始记录见 `logs/speed_trials/`、`logs/tuning_gpu23/speed_*/` 和 `logs/speed_*`。

# 历史实验记录

整理日期：2026年11月9日。依据保留的原始训练日志、固定评价、终止记录和交接文档整理；目录中的日期是实验批次标识，部分实验跨日完成。

**阅读口径：**A/F 分别指未加权的 Action / Future MSE，越低越好；TRAIN 是固定训练窗口评价，VAL 是独立轨迹上的固定评价。不同数据池、模型版本和预测 K 的数值不能直接排名。update 指外层参数更新；容量测试、只读评价及恢复检查不计为新的完整训练实验。当前没有已验证的机器人闭环成功率。

## 1. 早期真实模型试跑与训练流程验证（10月31日—11月1日）

**目的：**确认真实 Qwen3-VL-4B、DINO、双向条件训练及联合训练能够运行。

| 实验 | 实际完成情况 | 结果与结论 |
|---|---|---|
| 原服务器 OpenDrawer 试跑 | 计划10步，未确认完成一次参数更新 | 首步触发50 GiB主机内存保护；不是已确认的CUDA OOM。原始日志未迁入，依据旧交接文档 |
| 新服务器真实4B验证 | 2步 | 完成前向、反向和参数更新，仅证明流程可运行 |
| 早期 BI 短训练 | 累计200步 | 末次batch A/F=2.00414/2.00994；不是固定评价结果 |
| 早期 Joint 短训练 | 100步 | 末次batch A/F=1.17823/1.94099；未形成闭环成功率结论 |

**结论：**早期已经试跑过Joint训练，但后续新版主线主要停留在Stage 1；不能将旧Joint短试视为新版Stage 2验证通过。

## 2. 旧版50任务 BI 训练与内层迭代诊断（11月1—2日）

**设置：**旧架构，K=2，Action/Future内层步长均为0.05。训练实际到2300步；旧final_report只写到2000，按后续日志和断点记录修正。

| 检查 | 结果 |
|---|---|
| 1000步，50样本离线联合推理Action MSE | 1.73770 |
| 2000步，同口径Action MSE | 1.58339 |
| 固定2300步模型，增加推理K至32 | Action改善逐渐饱和，Future基本不动 |

只读内层诊断中，固定验证记录的Action MSE从K=0的2.00245降至K=2的1.57540、K=8的1.46104、K=32的1.45335；Future始终约2.02704。诊断包含两个候选初始化种子，不是两次独立训练。

**结论：**旧模型主要学到了Action方向；单纯增加推理迭代次数没有解决Future学习不足。这些结果不等同于后续加入迭代条件、二次先验的新架构。

## 3. 旧版训练K=2/4/8对照及补跑（11月2—3日）

**设置：**20个训练窗口、20个验证窗口，有效batch=2，相同初始化。

| 实验阶段 | 实际完成 | 结论 |
|---|---|---|
| 首轮及重复尝试 | K2完成100步；K4/K8未完成有效长对照 | K2末端VAL A/F=1.92968/2.02743；K4遇主机内存保护，K8尝试存在显存占用干扰 |
| 单模型双卡切分补跑 | K2、K4各2步；K8完成1步 | K8首步约814.55秒，随后暂停转做执行优化 |

**结论：**没有完成三组同预算质量比较，不能据此认定K2最佳。旧汇总曾把K8写为0步，实际speed_report记录1步；这里采用后者。

## 4. 旧版Future内层步长与小样本拟合（11月3日）

**目的：**检查Future梯度修改量是否过小，以及增大内层步长能否帮助学习。内层步长与外层模型学习率是两类参数。

| 对照 | 实际预算 | 结果 |
|---|---|---|
| Future步长50/100/200/500/1000 | 五组各20步 | 末端VAL Future分别约2.02744/2.02741/2.00308/2.00864/2.02235 |
| 步长200续训 | 到40步 | VAL Future=1.98969；不能认定全局最优 |
| 步长500续训尝试 | 仅确认20步训练及评价记录 | 续训命令超时；另有handoff声称保存30步，记录不一致，未计作完成40步 |
| 自适应执行优化后的K2，Future步长0.05与500 | 各40步 | 预测K2的VAL Future约2.02742与1.94889；Action基本未改善 |
| 同路线K4短试 | 日志30步，最后固定评价20步 | 未形成同预算优胜结论 |
| Future步长200，训练K2/K4过拟合检查 | 各200步；10条训练轨迹、80窗口 | 按各自K评价，TRAIN A/F：K2为2.28639/1.81827，K4为2.27128/1.66621；无独立验证集 |

**结论：**较大的Future步长出现改善信号，但旧版仍未实现充分拟合；K4收益需结合更高计算成本判断。

## 5. 新版迭代条件与数值设置短筛选（11月4日批次）

**目的：**排查Future难学，尝试迭代条件、可学习步长、FP32主权重、任务模块学习率、梯度RMS归一化、token尺度与二次先验。多数使用旧OpenDrawer/OpenCabinet的16个TRAIN窗口、16个VAL窗口，K=2；不同设置的初始基线不同。

| 实际试验 | 更新数 | 末端TRAIN A/F | 主要观察 |
|---|---:|---|---|
| 迭代条件＋可学习步长初试（iter_learned） | 20 | 2.28618 / 1.98635 | 尚无充分拟合证据 |
| BF16短试复跑 | 40 | 1.99069 / 1.89180 | Action基本未学 |
| FP32主权重短试 | 40 | 1.99514 / 1.94733 | 仅主权重改动未解决问题 |
| 步长尺度校准 | 40 | 1.98805 / 1.98087 | 改善有限 |
| 提高任务模块LR | 40 | 1.02407 / 1.97482 | 主要改善Action |
| 提高任务模块LR＋Future步长2000 | 40 | 1.16984 / 1.89390 | Future有所改善，仍较弱 |
| RMS归一化0.25 / 0.5 | 各40 | 2.10689/2.09630；2.04704/2.29229 | 未表现出有效双向拟合 |
| RMS归一化0.5＋token尺度调整 | 40 | 0.96483 / 2.28829 | Action改善，Future仍差 |
| 双二次先验残差形式 | 40 | 0.94687 / 0.10514 | 接近先验基线，不能算充分训练收益 |
| 双先验＋pack2初试 | 40 | 0.94622 / 0.10514 | 短预算内基本未学习 |
| 双先验＋token尺度，头LR=1e-3 | 160 | 0.82746 / 0.08729 | 两个训练方向均有改善 |
| 同路线头LR=3e-4 | 160 | 0.94053 / 0.10412 | 学习较慢 |
| 同路线头LR=1e-2 | 20，停止 | 7.11869 / 12792.83691 | 严重失稳 |

首轮tinyfit_bf16/master/master_calibrated目录还保留了初始化或失败日志，但没有完整训练结果；有效数值采用后续r2记录，不能把两者算作重复成功实验。

**结论：**逐步形成“当前DINO初始化＋双二次先验＋可学习步长”的后续参考配置；多项设置共同变化的短试不能当成严格单因素因果结论。

## 6. 二次先验 × Future初始化消融（11月4日批次）

**设置：**16 TRAIN / 16 VAL窗口，batch=2，K=2；正常组400 update，即800次窗口访问、约50遍。

| 二次先验 | Future初始化 | 实际update | TRAIN A/F | VAL A/F |
|---|---|---:|---|---|
| 无 | 高斯 | 40，保护停止 | 1.99233 / 37.49624 | 1.96071 / 16.09852 |
| 无 | 当前特征 | 400 | 1.22072 / 0.09366 | 1.32147 / 2538.84074 |
| Action＋Future | 高斯 | 400 | 0.53453 / 0.06781 | 1.12136 / 0.21780 |
| **Action＋Future** | **当前特征** | **400** | **0.49962 / 0.06637** | **0.84701 / 0.16445** |

双先验＋当前初始化的第0步TRAIN为0.94689/0.10514，VAL为0.92631/0.14007。终点TRAIN两方向改善，但VAL Future仍比初始差。

**结论：**双先验＋当前初始化是较好的拟合参考；先验在第0步就能降低误差，必须与真实训练收益分开。结果没有证明Future泛化通过。

## 7. 仅Future先验与可学习步长消融（11月4—5日批次）

| 对照 | 实际update | TRAIN A/F | VAL A/F |
|---|---:|---|---|
| 双先验＋可学习步长，参考组 | 400 | 0.49962 / 0.06637 | 0.84701 / 0.16445 |
| 仅Future先验＋可学习步长 | 400 | 0.59128 / 0.18455 | 1.00128 / 0.44437 |
| 无先验＋固定步长5/2000 | 40步评价触发保护 | Future=10.69345 | 未形成完整终点评价 |
| 无先验，头LR降至1e-4，固定步长 | 80 | 1.65218 / 0.10517 | 1.63587 / 0.14017 |
| 无先验，头LR降至1e-4，可学习步长 | 80 | 1.31766 / 0.10528 | 1.27790 / 0.14048 |

固定步长失败组的活动训练日志还写到42步，保护依据是40步固定评价；两者不能混成“第42步评价”。

**结论：**仅Future先验没有优于双先验；低头LR下，可学习步长对Action有帮助，但Future基本不变。三个候选初始化种子的只读复核支持Action差异，但不等于三个独立训练种子。

## 8. 扩大到10条训练轨迹的两种子检查（11月4日批次）

**设置：**10条TRAIN轨迹80窗口，另10条VAL轨迹80窗口；双先验、当前初始化、batch=2；seed42/43各200 update，约5遍训练池。

| 种子 | TRAIN A/F | VAL A/F |
|---|---|---|
| 42 | 1.28120 / 0.23407 | 1.21229 / 0.24415 |
| 43 | 1.30129 / 0.22316 | 1.21564 / 0.24336 |

初始VAL为1.21835/0.26238。**结论：**Future验证有改善，Action验证改善不足1%；没有达到双方向充分学习的要求。

## 9. 小样本有效batch=2/4/8对照（11月5日）

**设置：**旧16窗口训练池，microbatch=1，按相同1600次窗口访问比较；正常完成组相当于100遍训练池。

| 有效batch | 实际update / 访问数 | TRAIN A/F | 原16窗口VAL A/F |
|---|---|---|---|
| 2 | 800 / 1600 | 0.12168 / 0.03835 | 0.89324 / 0.16649 |
| 4 | 400 / 1600 | 0.14806 / 0.04929 | 1.00074 / 0.16908 |
| 8 | 20 / 160，保护停止 | 0.98124 / 1.70767 | 0.95597 / 2.46063 |

补做batch2/8、头LR=3e-4、各320次访问的配对，两组均完成但基本未学习。另增加16条独立验证轨迹、64窗口：batch2 A/F=0.77132/0.17544，batch4=0.85578/0.19378；保持当前特征的Future基线为0.14596。

**结论：**该小样本配置下batch2优于batch4，但Future仍未超过简单验证基线；不能直接外推为更大数据集上的最佳batch。

## 10. 10任务有效batch=8/16/32/64对照（11月5—6日）

**设置：**800个TRAIN窗口、20个训练监测窗口、100个快速VAL窗口；同时计算4样本，FP32；每组计划4096次窗口访问。

| 有效batch | 实际update / 访问数 | 最后完整VAL的update | VAL A/F | 结果 |
|---|---|---:|---|---|
| 8 | 512 / 4096 | 512 | 1.28245 / 0.17130 | 完成，但后期回退 |
| 16 | 256 / 4096 | 256 | 1.20070 / 0.14419 | 完成，终点两项较好 |
| 32 | 104 / 3328 | 96 | 6.95539 / 0.24084 | 104步固定TRAIN Action超限停止 |
| 64 | 64 / 4096 | 64 | 1.22210 / 0.14927 | 完成，终点弱于batch16 |

共同初始VAL A/F=1.28889/0.17129。batch8的best384为1.17126/0.14698，与batch16的best仍有Action/Future权衡。

补做batch8、整体LR×0.3、1024次访问：VAL=1.24183/0.15545，原LR同预算为1.21963/0.14934。其余计划中的7组LR复核、完整训练种子复核及400窗口确认评价未完成。

**结论：**batch16是当时较实用的后续基准，随后按用户选择用于三任务训练；没有证明全局最优batch。

## 11. 三任务完整训练池实验（11月6—7日）

**设置：**OpenDrawer、PickPlaceCounterToCabinet、WashFruitColander；1367条TRAIN轨迹、44768窗口，计划2798 update；VAL为78条轨迹312窗口。batch16、microbatch4、K2、双先验、Future当前初始化、损失权重1/9。

| 种子 | 实际进度 | 结果 |
|---|---|---|
| 42 | 704 / 2798 | 固定TRAIN Action超限，保护停止 |
| 43 | 121 / 2798 | 按用户要求安全暂停，腾卡运行小样本对照；不能算数值失败 |

seed42的正式VAL A/F：第0步0.76388/0.14373，第256步0.72158/0.12245，第512步0.82452/0.12451。

异常时间线：约402步出现梯度预警；532–600步主要Action失稳；690/694步batch Future分别达到167.83865/188.97705；704步固定TRAIN Action达到17.09571，为初始27.31倍，触发保护。704步固定TRAIN Future未完成记录。原latest680、best256保留。

**结论：**原配置无法稳定完成一个窗口遍历；失败由数值保护触发，未见对应OOM证据。

## 12. Future损失权重1/9对照（11月6日）

**设置：**旧OpenDrawer/OpenCabinet的16 TRAIN / 16 VAL窗口，batch16、microbatch4、K2；两组各50 update，约50遍窗口池。

| Future权重 | 50步TRAIN A/F | 50步VAL A/F |
|---|---|---|
| 1 | 0.94679 / 0.11057 | 0.92633 / 0.14011 |
| 9 | 0.94689 / 0.10516 | 0.92631 / 0.14008 |

初始TRAIN为0.94689/0.10514。**结论：**50步内两组均没有有效双向学习，不能认定权重9最优。权重9最初来自初始误差尺度比约9，并不保证梯度平衡。权重9组随后续到200步，计入下一项研究，不是新初始化的第二份独立实验。

## 13. 旧16窗口能量头学习率四档对照（11月6—7日）

**设置：**其他模块LR固定，仅比较能量头1e-4/3e-4/1e-3/3e-3；旧16 TRAIN / 16 VAL窗口，batch16、microbatch4、K2、Future权重9；先50步，未失败组延长到200步。

| 能量头LR | 实际update | TRAIN A/F | VAL A/F |
|---|---:|---|---|
| 1e-4 | 200 | 0.53018 / 0.09712 | 1.21382 / 0.20546 |
| 3e-4 | 50，失败 | 0.93548 / 14.17461 | 50步未完成记录 |
| 1e-3 | 200 | 0.45129 / 0.10286 | 0.93001 / 0.15626 |
| 3e-3 | 20，失败 | 1.38228 / 1.16488 | 1.21287 / 71.46554 |

**结论：**没有满足预设训练改善与验证退化限制的优胜组，因此未启动原计划的三任务确认。旧汇总中的pending是状态遗漏，不表示仍在训练。该数据池与当前单episode67窗口不同，不能代替当前计划中的能量头对照。

## 14. 三任务失稳重放与内层裁剪（11月7日）

**设置：**从三任务完整680步断点恢复，保持相同训练样本顺序，各更新12次至692；对照原设置与候选梯度逐元素裁剪（Action=1、Future=0.00025）。

| 观察 | 原设置 | 候选裁剪 |
|---|---:|---:|
| 第690步batch Future MSE | 167.83865 | 0.15408 |
| 第692步固定TRAIN12 A/F | 1.39828 / 2.17153 | 1.17143 / 0.14618 |
| 第692步固定VAL24 A/F | 2.75800 / 2.75218 | 2.18582 / 0.21883 |

裁剪组自身初始TRAIN为0.72335/0.10216、VAL为1.26274/0.16719，末端均恶化。裁剪会立即改变推理，因此两组初始评价不同。

**结论：**原爆炸可复现；内层裁剪能压制尖峰，但没有恢复有效学习，不能称为已验证修复。

## 15. 三任务best256续训：整体学习率×0.3/×0.1（11月7—8日）

**设置：**从同一best256恢复模型、Adam、调度及RNG；统一缩小四类外层学习率。最初续至512，后授权延长到2048，最终按用户要求切换到单episode研究。

| 倍率 | 实际停止位置 | 新增update | 最后正式VAL位置 | VAL A/F |
|---|---:|---:|---:|---|
| ×0.3 | 1337 | 1081 | 1280 | 0.53284 / 0.11105 |
| ×0.1 | 1172 | 916 | 1152 | 0.61351 / 0.11563 |

两组均为安全暂停，未完成2048目标。共同681–704区间的最大batch Future：原LR为188.97705，×0.3为0.16106，×0.1为0.16321。

**结论：**降低整体LR显著改善了这段训练的稳定性；×0.3有较好的验证结果。上表两组终点预算不同，不据此宣布严格同预算优胜。该结果也不能直接外推到从零训练的单episode实验。

## 16. OpenDrawer单episode四档整体学习率（11月8—9日）

**设置：**episode3346，281帧、67训练窗口；另外两条轨迹16个辅助VAL窗口。fresh相同预训练初始化，batch16、microbatch4、K2；每组210 update，约50.15 epoch。先比较×1/×0.3，再按拟合结果选择×0.6/×2。

| 整体LR倍率 | 实际update | TRAIN A/F | 末端辅助VAL A/F |
|---|---:|---|---|
| 初始 | 0 | 0.60535 / 0.12037 | 0.66413 / 0.09012 |
| ×0.3 | 210 | 0.29348 / 0.11925 | 0.72428 / 0.09281 |
| ×0.6 | 210 | 0.59091 / 0.12009 | 0.65762 / 0.09200 |
| **×1** | **210** | **0.08812 / 0.07540** | **0.68283 / 0.18449** |
| ×2 | 20，保护停止 | 1.23481 / 4.62289 | 20步未完成记录 |

**结论：**×1训练拟合最好，Action/Future较初始下降85.4%/37.4%，但辅助验证未改善；×2快速失稳。选择×1作为后续单episode拟合基准，不代表全量训练最优。已完成三组210步、失败一组20步，实际总计650次更新，不能写成840次全部完成。

## 17. 容量、速度与数值正确性验证

这些验证用于选择可运行的执行方式，不能替代训练效果实验。

| 验证系列 | 实际内容与结论 |
|---|---|
| 11月1日双卡容量/速度试验 | 检查BI/Joint、单卡副本/模型切分、CPU卸载与缓存；部分大batch触发主机内存预算。另有“GPU被占用”的无效首轮单独保留，不能用来判断绝对容量 |
| 11月3日缓存复用 | 同类短测中更新耗时约138.5→30.0→22.2秒；最后一组同时增加GPU缓存预算，收益不能全归因于单一改动 |
| pack1/pack2与卸载/重算对照 | pack1约17.23秒/update，pack2约5.83秒；另测关闭卸载、关闭重算及组合。pack2改变token组织，是架构消融，不是无损等价加速 |
| BF16 native microbatch梯度检查 | native2相对逐样本Action梯度差约26.25%；FP32复核降至约0.0084%。不能把容量通过等同于梯度一致 |
| 同时计算1/2/4样本 | 有效batch4、各8步；约11.14/6.01/4.71秒/update。这里只验证当时配置的短测吞吐 |
| 同时计算4/8/16样本 | 有效batch16：4样本约15.76秒/update；8样本约40.52秒且CPU内存更高；16样本独立复测未完成首步即CUDA OOM |
| FP32 native4与有效batch累积检查 | 最高精度FP32通过所测Action/Future梯度门槛；B8/16/32/64累积检查通过，后续采用FP32 native4 |
| CPU保存张量去重及缓存预算 | 全优化器短测约88.7→65.4秒/update，同时改变去重与预算，不能单独归因；梯度复核通过所测样本 |
| 双卡组硬件放置对照 | 同B16配置，GPU01/23约3.389/4.297秒/样本；证明速度比较受放置影响，未唯一定位NUMA原因 |
| 完整保存与恢复 | 完成1→2、batch2的800→801、batch4的400→401、batch32的98→99、三任务680→681隔离检查；所测恢复结果通过，额外步不计入主实验预算 |
| 固定评价复现 | batch8 latest512抽样复算误差一致；候选种子复查、Future末端重新加载复算等均保留原始记录 |

## 18. 数据与失稳原因专项核查（11月7—8日）

对三任务已训练的11264个窗口、1350条episode检查动作/状态有限性、窗口边界和任务映射，未发现所覆盖范围的明显结构错误。另CPU重算64个窗口的DINO特征：690/694步简单“复制当前特征”基线Future误差约0.14078/0.06882，而模型分别为167.84/188.98。

普通固定样本在晚期模型中也被内层更新推坏；结合低LR重放，支持“模型更新尺度与内层求解失稳”的解释。相关CPU测试和真实4B局部梯度/恢复检查通过，但未排除所有晚期高阶导、卸载或优化器精度问题。

**结论：**没有找到足以解释爆炸的明确坏数据或唯一计算bug；不能把“未发现”写成“已证明没有bug”。

## 19. 当前累计结论与尚未执行项

- 已证明模型能降低固定小样本的两方向训练误差；尚未证明稳定泛化或机器人闭环成功率。
- Future当前特征初始化和双先验可作为后续基准，但初始先验收益必须与训练收益区分。
- 当前执行基准为FP32、microbatch4、有效batch16、K2；来自数值、容量和有限效果证据，不是全局最优结论。
- 三任务续训降低LR有效，单episode从零拟合则×1最好；数据与起点不同，不能直接互换结论。
- 当前67窗口数据上的能量头×0.5/×1/×2对照尚未启动；旧16窗口四档研究已完成，两者必须分开记录。
- 当前数据上的新损失权重、内层步长/裁剪、K4及batch8对照均属后续计划；不得列为已完成。

## 20. 来源索引与记录保留

原始目录以 `/cloud/zzc/energyWAM/experiments/` 为根：

| 本文项目 | 主要来源 |
|---|---|
| 1—2 | 项目`PROJECT_HANDOFF_2026-10-31.md`、`logs/runs/`、云盘`runs/energywam_formal_bi_20261101/`、`inner_diag_20261102/`、`inner_diag_k32_20261102/` |
| 3—4 | `train_k248_20261102/`、`train_k248_sharded_20261103/`、`adaptive_shared_20261103/`、`future_step_sweep_20261103/`、`overfit_k24_alpha200_20261103/` |
| 5—9 | `iteration_aware_v1_20261104/`，包括逐组日志、固定评价、消融/配对报告及扩展验证 |
| 10、17 | `native_capacity_20261105/`、`batch_grid_micro4_20261105/`及项目`logs/`中的早期容量/速度报告 |
| 11 | `three_task_full_20261106/` |
| 12—13 | `tiny_future_weight_20261106/`、`head_lr_tiny16_20261106/` |
| 14、18 | `stability_diagnosis_20261107/`，详细分析见其`forensic_review_20261108/report_zh.md` |
| 15 | `lr256_comparison_20261107/`、`single_episode_lr_20261108/old_training_pause.json` |
| 16 | `single_episode_lr_20261108/comparison_report.json`及逐组终止/评价记录 |

逐目录清单：[historical_experiment_inventory_20261109.csv](docs/reproducibility/historical_experiment_inventory_20261109.csv)。机器可读证据与SHA256：[historical_experiment_inventory_20261109.json](docs/reproducibility/historical_experiment_inventory_20261109.json)。共索引135个有训练日志或运行报告的证据目录；包含恢复、测速和重试归档，**不等于135项独立训练实验**。只读诊断与汇总报告另列来源哈希。

本文覆盖本机/云盘现存证据及已迁移交接记录；旧服务器未迁入的原始材料无法逐项重验。部分权重已按历史授权退休，配置、日志、评价与删除清单仍保留；本文记录实验结果，不承诺每项历史权重仍可恢复。

## 2026年11月9日后续启动记录

用户已授权启动当前67窗口上的能量头LR对照：仅新增5e-4、2e-3两组，沿用已有1e-3为参考；各210update，在GPU2/3串行。首次启动时间11:49 CST，启动时仍在初始化，尚无新组效果结论。此前第19节“尚未启动”是本次授权前状态，以本条取代。运行记录见`single_episode_head_lr_20261109/`，不把新计划记为已完成结果。

## 21. 实验室远程运行与恢复验证（11月9日）

在user-SYS-420GP-TNR的物理6/7 L40S上，独立环境、真实Qwen4B、B1/micro1/K2、2TRAIN+2VAL，完成step1保存并完整恢复至step2；latest2/best0含完整状态，24项远端CPU检查通过。CUDA/NVML编号不同，已改UUID并核实PID落卡。旧DINO绝对路径导致的0步失败保留记录，采用数值不变的标准化路径迁移解决；模型和6个实际数据文件按SHA复用。属于工程可运行/恢复验证，没有正式训练或质量优胜结论，原B16远端容量尚未验证。证据见docs/LAB_REMOTE_RUN.md及logs/lab_remote_setup/deployment_validation_report.json；原始输出/data7/zzc/energyWAM/lab_execution_20261109/outputs/real4b_resume_smoke_relocated_stats。

## 2026-11-09 用户授权实验室双队列（运行中，非完成结果）

用户明确指定物理GPU3/4跑Future损失权重1/9，GPU6/7跑内层步长研究。各组fresh官方4B、seed42、OpenDrawer3346的67TRAIN/16辅助VAL窗口、FP32、有效B16/native4、K2、210update（3360窗口访问、50.15等效epoch），评价每20update，完整latest/best；best按本组VAL A+lambda_F*F预声明。整体倍率固定原1倍/head1e-3，云端head-LR研究尚未判胜，不冒称最终最佳。远端torch2.6/cu124不同于云端，必须各硬件组自带对照，不能将跨主机差异归因超参数。

- loss根目录：`/data7/zzc/energyWAM/lab_execution_20261109/studies/future_weight_20261109`。weight9对照后weight1，串行；完整权重预算122.41GiB。32GB卡saved-tensor GPU缓存预算0/0，保留CPU异步卸载，未改数学配置。原18/18层分配容量检查在首个backward CUDA OOM，完成0update，原始日志归档failed_setup_attempts/oom18_18_*。正在独立验证16/20层分配（首卡含视觉编码，减少两层），须梯度一致性及2update容量通过后才自动正式启动，不能记成正式已训练。
- steps根目录：`/data6/zzc/energywam_lab_studies/inner_steps_20261109`。原alphaA5/F2000对照，再A2.5/F2000、A5/F1000，三组串行；保存预算168.17GiB；GPU缓存预算10/14。修改步长时同时补偿quadratic权重，保持有效先验precision .2/.0005，避免混淆先验强度与求解步长。共享真实4B梯度检查（native1 vs4，reference缓存10/14 vs candidate0/0，4个固定窗口）已通过；只覆盖初始抽样梯度，不冒称全部训练轨迹等价。2update B16/native4/FP32容量检查通过，16:15:37 CST启动的controller已进入正式training；已得到完整67窗口step0 TRAIN A0.6053476845/F0.1203727312，实际更新数须查最新日志。
- 兼容性准备失败保留：独立脚本缺sys.path及Torch2.6缺host_memory_stats，均在正式更新前发现；已修复导入和缺失统计兼容。该版本不可用全局pin缓存统计，记录missing，仍保留单context64GiB pin限额、每任务USS120GiB/主机available24GiB、单组12h保护。停止/失败不自动改超参数重跑。
- 独立tmux队列，不依赖SSH存活；代码按唯一release冻结。配置、arm_definition、数据manifest、source_manifest、启动命令、梯度检查和失败记录都在根目录。运行部署索引`logs/lab_remote_setup/ablation_deployment.json`；管理`python scripts/lab_ablation_control.py status`、`logs --study loss|steps`、`stop --study loss|steps`（只操作指定本次队列）。远端未复制W&B密钥，以原始本地日志为准。云端GPU2/3研究未停止，其他用户进程未操作。

补充16:29 CST实况：GPU3/4的16/20分配在梯度门槛backward时logicalGPU1 OOM（未做optimizer update），已保留梯度日志/资源；再次仅调整为17/19作最后一档容量尝试，不降低模型/精度/batch/K。前一次16/20失败归档目录名误为failed_setup_attempts/oom17_19_1794212892112140382，目录名不是实际配置，实际source/config/log明确为16/20。L40S正式对照此刻仍0/210：完成step0 TRAIN/VAL评价后在写入校验24,568,514,334字节完整pending checkpoint；进程D状态wait_on_buffer/rq_qos_wait，为/data6 ext4磁盘I/O等待，不能把capacity2步或step0评价算作正式更新。14项层分配/GPU政策/旧Torch统计兼容CPU测试通过。

16:34 CST启动检查结论（取代上文待定状态）：GPU3/4的17/19梯度门槛已双方向通过，相对L2 Action0.000250796/Future0.000457111；随后带完整optimizer的2步容量检查仍首次backward GPU0 OOM，完成0update，controller=blocked_capacity，未启动任何正式loss组，已释放3/4。本次不再尝试层分配，也不擅自改batch/精度/模型或改GPU。GPU6/7 controller=training，正式control仍0/210，另两组queued；真实2步capacity不计入正式预算。首个完整step0 pending checkpoint在/data6等待fsync，尚不是可恢复latest；10秒整盘写速11.6102MiB/s，D状态wait_on_buffer/rq_qos_wait。队列保留后台继续，不能描述为已完成正式更新。完整启动报告logs/lab_remote_setup/formal_ablation_startup_report.json；最新原始记录取回ablation_results/20261109T083430Z，各远端study根目录也保存deployment_startup_review.json。若要loss改排6/7，需要用户确认新的资源分配；尚未提交此替代任务。

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

18:15:56 CST K4独占13/23两步容量已完整通过：254.701/211.000秒，两个param_grad_nonfinite均0，GPU allocated峰23.970/36.099GiB，CPUstate搬运每次8,245,862,488字节。上述是改变训练batch的工程容量记录，不以两步loss比较判定学习效果。K3真实B16/native1完整resume验证正在target1，已按K=3的2VAL A+9F保存step0 best/latest；criterion实际含K=3，确认评价/选择链路不再遗漏3。正式K2/3/4仍0/210且未启动，工程评价2TRAIN/2VAL不可当正式67TRAIN/16VAL结果。最新logs/lab_remote_setup/k_study_startup_report.json。

## 22. 当前四项单episode对照：实验文档稿

以下四项属于同一轮 OpenDrawer 单 episode 过拟合研究。**进度查询：2026年11月09日 18:23—18:25（北京时间）；未完成组尚无最终结论。**

**共同设置：**Qwen3-VL-4B、FP32、seed=42；训练 episode 3346，共67个窗口，辅助验证集为另外两条轨迹的16个窗口。每组预算210 update，即3360次窗口访问，约50.15个等效epoch；窗口有重叠，仍只有一条训练轨迹的信息量。有效batch固定为16，外层梯度裁剪阈值为1，warmup为16次更新。各组从相同预训练权重重新初始化，分别记录自己的第0步基线。

### 1.2 能量头学习率对照（ongoing）

**目的：**在整体学习率倍率×1的基础上，单独调整能量头更新速度，检查能否改善Action／Future拟合并减少梯度尖峰。

| 组别 | 能量头学习率 | 实际进度 | 210步固定TRAIN Action／Future MSE |
|---|---:|---|---|
| 减半 | 5e-4 | 已完成210/210 | 0.27902／0.08718 |
| 基准 | 1e-3 | 使用上一轮已完成的210步结果 | 0.08812／0.07540 |
| 加倍 | 2e-3 | 运行中，131/210 | 待完成 |

**固定设置：**K=2，损失为`Action MSE + 9 × Future MSE`；语言主干、任务模块和可学习步长参数的外层学习率分别固定为1e-5、1e-4、1e-3。有效batch=16、microbatch=4，在compshare的GPU2、3上串行运行。

**比较标准：**每20次更新评价全部67个固定TRAIN窗口和16个辅助VAL窗口，分别报告Action／Future MSE，并记录梯度范数、裁剪及实际学习率。以同为210步的固定TRAIN拟合为主，VAL单独报告；两个方向有取舍时不强行判定整体优胜者。

**阶段结论：**减半组在本次固定TRAIN上的两个指标均未超过既有基准；加倍组尚未完成，暂不对三档学习率作最终排序。

### 1.3 Action／Future损失权重对照（ongoing）

**目的：**检查Future损失是否需要9倍权重，以及降低其权重后，Action拟合和Future学习能否更好地兼顾。

| 组别 | 训练损失 | 实际进度 |
|---|---|---|
| 原权重对照 | Action MSE + 9 × Future MSE | 运行中，23/210 |
| 等权重 | Action MSE + Future MSE | 排队中，原权重组结束后自动启动 |

**固定设置：**K=2，初始内层步长Action=5、Future=2000；语言主干、任务模块、能量头和可学习步长参数的外层学习率分别为1e-5、1e-4、1e-3、1e-3。有效batch=16、microbatch=1，在实验室GPU3、4上串行，启用激活和原AdamW8bit状态的CPU卸载。

**比较标准：**每20次更新评价固定67个TRAIN窗口和16个辅助VAL窗口，比较未加权的Action／Future MSE、相对各自第0步的改善和稳定性。两组总loss的权重不同，不能直接用总loss大小判胜。

**当前结论：**尚未完成同预算对照，暂不判断权重1或9更好。

### 1.4 内层步长设置对照（ongoing）

**目的：**分别减小Action或Future的初始求解步长，检查能否改善拟合和稳定性；这里的内层步长不同于模型参数的外层学习率。

| 组别 | Action初始步长 | Future初始步长 | 实际进度 |
|---|---:|---:|---|
| 原设置对照 | 5 | 2000 | 运行中，86/210 |
| Action减半 | 2.5 | 2000 | 排队中 |
| Future减半 | 5 | 1000 | 排队中 |

**固定设置：**K=2，损失为`Action MSE + 9 × Future MSE`，外层学习率保持原倍率×1，能量头学习率为1e-3。有效batch=16、microbatch=4，在实验室GPU6、7上按表格顺序串行。

步长仍可学习，允许范围为各自初值的0.5～1.5倍。因此本实验比较的是“初值及对应可学习范围”这一套设置。修改初值时补偿二次先验系数，使Action／Future的有效先验强度分别保持0.2／0.0005，避免同时改变先验强度。

**比较标准：**除固定TRAIN／VAL的Action／Future MSE外，记录实际步长、是否贴近允许边界、候选修改量和梯度范数，不能仅凭两个步长数值的大小判断优劣。

**当前结论：**两个减半组尚未开始，暂不判断哪个方向应缩小步长。

### 1.5 内层迭代次数K对照（准备与验证中）

**目的：**比较增加内层求解次数是否改善拟合，以及收益是否值得额外计算成本。

| 组别 | 训练K | 配套评价K | 正式训练状态 |
|---|---:|---|---|
| 基准 | 2 | K=2 | 尚未开始 |
| 增加一次 | 3 | K=3，同时记录共同推理K=2 | 尚未开始 |
| 增加两次 | 4 | K=4，同时记录共同推理K=2 | 尚未开始 |

**固定设置：**沿用1.3原权重组的数学配置，即4B、FP32、有效batch=16、microbatch=1、损失`Action MSE + 9 × Future MSE`、初始步长5／2000及原外层学习率；激活与优化器状态CPU卸载方式相同。在实验室GPU1、8上按K=2→3→4串行，各210 update。

为适配1号32 GiB与8号46 GiB显卡，三个K组统一采用13／23层分配，让8号卡多承担部分模型；这属于设备放置调整。每组启动前要求两卡空闲，使用本机同一对卡上的K=2作对照。

**比较标准：**每20次更新评价67个固定TRAIN窗口和16个辅助VAL窗口。分别报告各自K下的拟合，以及共同推理K=2时的结果，同时记录单次更新时间和显存峰值。相同210次更新代表相同样本访问预算，不代表相同计算预算。

**准备进展：**K=3的评价、best选择和保护逻辑已补齐；K=4真实两步容量检查通过，GPU1／8的本进程分配峰值约24.0／36.1 GiB。K=3完整保存恢复仍在验证中，三组正式训练均为0/210；工程验证步数不计入正式预算。最初共享GPU8的尝试因共存任务显存增长而OOM，记录已保留，未作为模型效果结论。

**记录与比较口径：**四项均保留原始日志、配置、数据清单、代码快照及完整latest／best checkpoint。best按预先声明的辅助VAL加权误差选择：损失权重组使用各自权重，K组使用各自评价K；best选择与固定TRAIN拟合排名分开。不同机器、microbatch及卸载设置的结果优先在各自研究内部比较，当前均不代表机器人闭环成功率或全量训练的最终最佳配置。


2026-11-09 18:37:49 CST按用户GPU截图只读检查：K队列未崩溃，controller PID143181、正式queue PID172031仍在；K4两步容量通过，K3真实B16/native1完整save1->resume2也通过，latest2/best0含model/optimizer/scheduler/rng/meta。验证进程正常退出释放1/8后，GPU8重新出现fengtuo任务，因此正式K2状态waiting_for_idle_gpus，K3/K4 queued，三组正式完成均0/210。当前政策要求1/8启动前无其他计算进程，显存free多不等于可启动；未自动切回共享、未重启任务、未停止他人进程。待8空闲后原队列自动启动。证据logs/lab_remote_setup/k_study_startup_report.json；用户要求跑起后无需持续监管，按需查询即可。

2026-11-09用户再次明确“不用空闲啊，和别人一起用就行”：已在不重启controller/不重复验证的情况下原子修改K study的GPU政策，允许GPU8共享，GPU1仍独占；按已测13/23分配K4峰值保留启动freefloor GPU1=26000MiB、GPU8=40000MiB（GPU8实测需求约37818MiB含context，区别于先前17/19共享32000门槛）。当次GPU8剩42944MiB满足门槛，现有正式queue每15秒重读策略会继续K2。未改模型/学习率/K/微批次/层分配/CPU卸载；没有再跑容量/resume，已通过记录继续有效。未操作他人进程。政策变更及原始GPU占用回执logs/lab_remote_setup/k_study_sharing_reenabled.json，远端sharing_reenabled_*.json；本条取代前文必须等待GPU8空闲的要求。

GPU8共享政策生效后，原正式queue已自动启动K2，PID188599，实际physical1/8，已完成模型13/23分配及67TRAIN/16VAL缓存初始化；当前正式完成仍0/210，第0步评价/初始化阶段。K3/K4自动排在后面，不需要重发命令。证明logs/lab_remote_setup/k_study_shared_formal_start.json。用户要求跑起后不持续监管，启动确认后结束观察，后续按需查询。

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


## 2026-11-10 能量头LR非单峰排序只读核查

用户质疑headLR ×1、×2、×0.5、×0.75、×1.25的非单调结果。实读五组resolved_config，除run_id/output_dir/notes外只差energy_head_learning_rate；sample_manifest JSON一致，初始参数抽样hash一致（不是full参数hash）。source_initial.zip中energywam数学代码一致，部分组仅gpu_policy.py不同；storagefix复跑另有checkpoint.py/run_state.py/metadata helper的保存链差异。结果仍按固定67TRAIN同210步两方向分报，不能要求学习率响应单峰。

核查时storagefix复跑有192个实际update，与历史×1前192步的逐update Action MSE、Future MSE、裁剪前param_grad_norm、更新后alphaA/alphaF五项全部精确相同；旧存盘失败repeat前92步这五项也一致。新复跑固定TRAIN至180步各评价均与历史值一致，未把尚未到达的210步标完成。这支持当前seed42/环境下轨迹可复现，不能把历史×1优势直接归因偶然随机波动，也不等于跨seed稳健最优或排除共同实现bug。

×0.75固定TRAIN Action@20/100/120为0.605365/0.604671/0.613498，初始0.605348，至140后才明显下降；×1.25 @80/100/120为0.605144/0.603316/0.604939，长期平台后末210为0.382026。裁剪前总梯度最大：×1为27,148,763.809272@44，×0.75为17,953,501.024412@49，×1.25为122,289,297.958642@74。所有组已有全参数clip1；最大尖峰不是最终效果单独判据（×1也尖峰但后来恢复更好）。基于实现推断：能量头改变内层候选梯度，再经展开反传影响其他模块；尖峰与全局裁剪/Adam历史造成不同优化路径，210步又尚非收敛，因此终点无单峰保证。未证明量化、数值非确定性或单一bug为根因。建议先完成现有×1复跑，再用相同协议跨seed复核候选，未额外启动实验。

原始逐组配置差异、源码差异、固定评价时序、梯度峰和重复一致性证据：logs/lab_remote_setup/head_lr_nonmonotonic_audit_20261110.json。此次仅只读训练资料并记录分析，未更改训练参数/进程/权重。


## 2026-11-10 14:46 CST 用户要求停止seed42并启动seed43，权重仅保存云盘

用户先授权headLR×1换seed复核，随后明确立即停止GPU2/3现有seed42，最后明确不要把新权重存本机磁盘，允许清理以前checkpoint腾云盘。已向核验身份的旧controller2540586发送SIGTERM，仅其自有queue2540591/train2540741退出，三PID均已不存在，controller14:44:35 CST标interrupted。实际train_log197/210；停止时latest186/best60，非数值guard或OOM。没有为保存额外尾步再训练。

本次授权清理范围仅旧storagefix重复验证这一组的latest186/best60及best物理快照，共3路径/2物理文件45.76243GiB；62个非权重文件SHA未变，日志、评价、配置、源码和meta全部保留，weights_retired.json标不可恢复。历史已完成210步×1原基线和其他实验权重/数据/预训练模型未删。清单cloud experiments/checkpoint_retirement_seed43_switch_20261110.json及logs/lab_remote_setup同名，实际cloud空闲49.57->约95.33GiB，满足新组latest/best/pending+8GiB=76.65GiB预算。未在项目本地创建新训练权重。

新root `/cloud/zzc/energyWAM/experiments/single_episode_head_lr_seed43_20261110T064618Z`，arm `episode3346_headscale1_b16_micro4_seed43`，tmux `energywam_head_lr_seed43_20261110T064618Z`，GPU2/3。直接复用storagefix已冻结源码 `/root/zzc/energyWAM/logs/frozen_sources/head_repeat_storagefix_20261110T031718Z/code`，数学和I/O源码均不改。YAML与旧seed42除run_id/output/notes外唯一差异trainer.seed42->43，完整结构比对通过。fresh官方Qwen3VL4B/DINO，随机任务模块初始化/训练候选随机流/采样次序用seed43，不从机器人训练checkpoint续训；同67TRAIN16VAL窗口清单，固定评价seed2026及末端辅助2027保持不变，仍B16/native4/FP32/K2、head1e-3/language1e-5/task1e-4/steps1e-3、warmup16、outerclip1、A+9F、原GPU分层/激活卸载。210updates/3360visits≈50.15epoch，固定评价20update，guard原10倍，best按16VAL A+9F，全状态latest/best。此试验验证×1跨seed表现，不直接证明跨seed各LR排名。

新输出云盘上CPU tiny保存1/恢复至2 gate通过，模型逐位一致，非真实4B GPU恢复测试。14:46:40 CST controller429083、queue429086启动，实际训练PID429255加载预训练模型，正式更新尚0，不能把加载或step0评价当训练完成。W&B已由controller注册原project独立来源，首次真实train_log后创建；当前未声称曲线已上传。准备脚本scripts/prepare_head_lr_seed43.py有防重复receipt，源代码/命令/完整config/manifest/环境/清理及gate记录已保存，启动后日志继续自动记录。


## 2026-11-10 18:13 CST seed43查询：80步因Future固定训练集10倍保护停止

实时核查seed43实际80/210，训练PID429255已不存在，云端GPU2/3均1MiB/0%。最后逐update日志16:12:17 CST，controller16:13:54 CST结束finished_with_failures。queue明确Fixed train future MSE at update80 exceeded10x其step0；SIGTERM -15来自已部署保护，非OOM/存盘报错/手动停止。固定67TRAIN A/F：0步0.60534770/0.12037273；20步0.60552320/0.28604783；40步0.60553770/0.14909368；60步1.01258919/0.91346051；80步0.60538480/1.71115927（Future初始14.2155倍，阈值1.20372729）。16VAL@80 A0.66561145/F2.62666807。最新训练batch@80 A0.69713652/F0.87796336，与固定TRAIN评价不同。完整latest61/best0文件及meta均存在，各24568518993B；仅核查存在/大小/step，未新增加载或恢复测试。80实际更新不等于能从80恢复，未保存的62–80不能用现有权重恢复。

这说明headLR1e-3在seed43下未通过现有稳定性准则，不能把seed42的好结果称跨seed稳健；不证明每seed均失败，也不能从提前停止推断若继续永远无法恢复。未重启/放宽保护/额外训练。真实状态快照logs/lab_remote_setup/head_lr_seed43_latest_status.json，所有原始日志与checkpoint云盘保留。


2026-11-10 seed敏感性解释进一步只读核查：trainer.seed同时控制新模块初始化、FixedSampleBudgetSampler顺序、sample-aligned候选随机流；不是只更换数据顺序。固定评价seed2026未改变，Future当前DINO初始化噪声std0，不能归因直接改变Future初始噪声；Action随机候选起点会变。step0预测一致不代表新模块权重相同（能量头零初始化可遮蔽内部初值差异）。seed43@80 alphaF2002.23755较初始2000只增加约0.112%，不能称可学习步长自身暴涨。固定67TRAIN Future@80一层后平均0.527912、二层后1.711159，说明本状态下第二次内层迭代平均放大预测误差，不证明K1整体训练更好或唯一根因。@80中位数0.635232，最大24.91729，剔除最大后平均1.359551仍超过原1.203727保护阈值；21/67窗口超过“初始全体平均10倍”阈值，此非逐窗口自身初始比值。@60平均0.913461/中位0.300022，表明异常不止终点单一坏窗口。

仅提出诊断方案未执行：先从保留61完整断点复现62–80；再把init/data/candidate随机流独立开关，分别在42基准上只换一项43，保持固定评价与其他超参。若候选随机流与不同样本顺序交互，需要按窗口+访问编号固定候选来区分随机流主效应和配对效应，不能把全seed变化归因单一因素。优先核查内层候选更新及各模块/方向梯度、参数实际更新相对范数、数据mask/标准化/特征异常，再单独测试外层LR或Future步长；8bit优化器仅待证假设，不能因存在量化就定罪。两seed不足以估计失败概率，guard是停止规则不是异常根因。具体数值原始记录见seed43目录seed_sensitivity_diagnostics.json。


## 2026-11-10 18:28 CST seed43允许有限MSE超阈值继续：独立完整断点续训已启动

用户明确同意“出现这种情况别退出继续往后训练”。本次仅seed43诊断job设置fixed_mse_guard_mode=warn：固定TRAIN10倍阈值保留作为持久告警，有限大梯度仍按原全局clip1；非有限训练loss/参数梯度、固定评价NaN/Inf、OOM、原USS140GiB/12h资源限制仍会停止。默认其他job仍stop，学校队列未改。scripts/run_iteration_search.py增加显式mode判定和fixed_mse_warnings.jsonl去重告警，6项guard测试通过。冻结旧K2 guard不支持evaluation_k参数，首次CPU预检在训练启动前抛TypeError；已适配原接口，记录preflight_attempt1_failure.json；修复后冻结代码stop/warn/nonfinite验证通过。

root `/cloud/zzc/energyWAM/experiments/single_episode_seed43_warn_continue_20261110T102649Z`，arm `episode3346_headscale1_seed43_warn_resume61`，code `/root/zzc/energyWAM/logs/frozen_sources/seed43_warn_20261110T102649Z/code`，tmux `energywam_seed43_warn_20261110T102649Z`，physicalGPU2/3。源码复制seed43原冻结版本，仅scripts/run_iteration_search.py不同，核心训练/优化器/I/O/采样/评价源码均相同。YAML只改run_id/output，仍seed43/head1e-3/B16/native4/K2/FP32/A+9F/warmup16/原LR及GPU卸载。恢复原seed43完整latest61（模型/Adam/scheduler/RNG全状态），目标global210，新增149次更新；62–80重放不冒充首次新增训练，原run失败80及原日志保持不变。新目录继承原0–61逐步日志及<=61固定评价，continuation_provenance明确来源；未复制原62–80日志为新结果。实际恢复后会重新评价61。

best继承原auxVAL A+9F选择的step0，独立generation用硬链接共享完整物理快照，后续轮换只动新目录。新latest初始是指向原latest61的symlink，滚动保存os.replace替换别名，不写源checkpoint；原失败组latest61/best0均保留。torch.load CPU mmap检查两份payload全状态必需keys、seed43、step61/0和metadata一致通过，未称GPU恢复已完成；实际startup证据另记。新code source_manifest/source_changes及source_resume.zip、配置/数据清单/环境/命令已保存。

云盘空间按此前用户允许清理旧checkpoint授权，仅退休已完成headscale1p25的latest210/best160及物理快照45.76243GiB；73个非权重文件hash未变。原1e-3基线、0.75组、seed43恢复来源保留，不动数据/预训练权重；1.25现在不可从原checkpoint恢复。清单checkpoint_retirement_seed43_warn_20261110.json保存在cloud实验根及本地logs/lab_remote_setup。cloud实际空闲恢复约95.32GiB，保守预留新latest/best/pending+8GiB，所有新权重只写cloud。

启动前父controller/storagegate检查通过，18:28 CST已提交tmux；后续确认具体PID/恢复step/首次新增更新再写startup，不把继承61条日志当本次已执行61更新。W&B新独立来源由controller注册，同项目，曲线0–61为明确继承历史，其后为真实重放/续训。本次只授权到210，不自动延长或附加seed。


18:33 CST续训启动实证：训练PID2666512/GPU2/3已完成第62次更新，开始63；本次新增1步、61步继承。第62步Action/Future MSE、param_grad_norm、更新后alphaA/alphaF与原seed43第62步五项逐值精确相同，实证fullresume首步复现。恢复61固定TRAIN Future已超过10倍，但队列写fixed_mse_warnings.jsonl后继续，未退出，告警模式实际通过。首步106.406秒，期间CUDA allocator打印96MiB分配OOM重试警告，但未抛终止异常，最终完成更新；不可将警告写成训练OOM失败。新W&B bcd72c072283已创建，0–61为继承历史。证据seed43_warning_startup.json及新run/startup_verification.json。确认启动后不持续助手监管。


## 2026-11-10 20:04 CST seed43续训稳定性再诊断与建议（未启动新实验）

实际全局143/210、PID2666512存活，最后逐步日志20:04:23 CST。固定67TRAIN@100 A0.605364/F0.120840，@120 A0.604777/F1.360455，@140 A0.600715/F0.937144；初始A0.605348/F0.120373。继续后Future曾回落到初始附近，但再恶化，Action至140约改善0.77%，不能称已恢复有效学习，也不能断言一次尖峰永久损坏模型。Future@140内层一次后15.600886、二次后0.937144，提示不止第二步放大，整个内层路径可能剧烈偏离再部分拉回。@140能量头Action梯度0.0490745、已乘9的Future梯度51959.5078；这是head梯度证据，不代表所有模块都按此比例。全局clip1已存在，尚缺实际裁剪后norm、各模块实际参数更新相对范数及分模块贡献。

建议先在原61完整断点、相同batch/RNG下短重放并补观察量，定位首个异常窗口/内层迭代/模块，核查mean-sum/mask/归一化/样本种子/先验等实现并对tiny双精度展开做有限差分验证；autograd anomaly只适合NaN/反向错误调试，不能自动诊断有限大梯度。优先稳定对照：在同seed43 fresh和原配置下只将Futurealpha2000->1000，并保持有效二次先验系数/归一化不变，避免alpha改变导致prior_precision=1/alpha一并变化；与学校seed42/mixed硬件结果不能当严格同seed对照。再分别测试Future损失权重9->1或外层整体LR×0.3等，不一起改；内层更新RMS限幅需按健康基线实际更新分布定阈值，并验证是否大面积饱和造成无梯度。历史三任务内层clip已压尖峰但未恢复学习，不能当作新万能方案。稳定须同时满足固定TRAIN A/F下降和多seed，而非只看norm或不退出。当前用户在问建议，未授权新增具体对照，未停止/改动当前续训。快照logs/lab_remote_setup/seed43_stability_next_diagnosis.json。


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

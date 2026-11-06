# 来源、环境与许可证记录

> 本文件记录 EnergyWAM 的代码来源、外部依赖、服务器环境与许可证边界。
> 生成日期：2026-10-28。

---

## 1. 运行环境（实测）

| 项 | 值 | 备注 |
|---|---|---|
| Python | `/data3/xyx_zhuzicheng/miniconda3/envs/LDA/bin/python`（3.10.22） | **只借用，不修改该环境** |
| torch | 2.6.0+cu124 | |
| transformers | 4.57.0 | Qwen3-VL 实现在 `transformers/models/qwen3_vl/` |
| 平台 | linux x86_64 | 超算 ARM64 目标环境需另行适配 |
| 默认 shell `python` | 不存在 | 所有脚本显式用上面的绝对路径 |
| CPU 测试 | `CUDA_VISIBLE_DEVICES=''`，`OMP_NUM_THREADS=2`，`MKL_NUM_THREADS=2` | 由 `scripts/env.sh` 统一设置 |

**新依赖**：本项目**未安装任何新依赖**。只用 torch / numpy / PIL / omegaconf / pandas /
torchvision / transformers，均为当前环境已有的传递依赖。可选依赖（`peft` 用于 LoRA、
`imageio` 作为视频解码后备）通过**惰性导入**使用，缺失只影响对应可选功能。

---

## 2. 权重与数据（只读，未下载任何内容）

| 资源 | 路径 | 用途 | 是否被本项目修改 |
|---|---|---|---|
| Qwen3-VL-4B-Instruct | `/data6/zzc/Qwen3-VL-4B-Instruct` | 语言主干 + 视觉编码 | 否（只读） |
| DINOv3-ViT-S | `/data6/zzc/dinov3-vits16-pretrain-lvd1689m` | 冻结未来/当前视觉特征 | 否（只读） |
| GR1 仿真数据 | `/data6/zzc/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim` | **数据管线实测**（非 RoboCasa365） | 否（只读） |
| LDA-robocasa | `/data6/zzc/LDA-robocasa` | 未使用 | 否 |
| RoboCasa365 数据 | **不存在** | — | — |

* 本次**未下载**任何权重、数据集或仿真资产。
* **未上传**任何内容到远端；`../LDA-1B/tools/hpc/UPLOADS_PAUSED` 保持原状。
* **未提交** git（`energyWAM` 目前不是 git 仓库），未 push。
* 未记录任何账号 / 密码 / Token；本文件也不包含任何凭据。

---

## 3. GPU 状态（2026-10-28 实测）

```
index  memory.used / total   util
0      27278 / 32760 MiB      1%
1      28462 / 32760 MiB     31%
2      22827 / 24564 MiB     91%
3      27206 / 32760 MiB      2%
4      31521 / 32760 MiB     93%
5      27998 / 46068 MiB    100%
6      34841 / 46068 MiB     28%
7      40200 / 46068 MiB    100%
8      40551 / 46068 MiB     84%
```

9 张卡全部有活动任务。**本次未执行真实 4B 全参数二阶训练**，也未占用任何已有实验的显存余量。
真实训练脚本带 `ENERGYWAM_ALLOW_GPU_TRAINING` 门禁，须由主 agent / 用户审核资源后显式开启。

---

## 4. 代码来源与许可证

### 4.1 本项目原创

`energywam/`、`configs/`、`scripts/`、`tests/`、`README.md`、
`docs/ARCHITECTURE.md`、`docs/SOURCES_AND_ENV.md`、`IMPLEMENTATION_STATUS.md`
均为本次实现，随本目录交付。

### 4.2 参考 / 对齐的既有实现（只读，未拷贝）

| 来源 | 用途 | 说明 |
|---|---|---|
| `../starVLA`（源码快照，**无 .git**） | 框架注册、数据归一化契约、训练/评测入口 | **只读引用**。未修改任何文件。不能编造其 commit。 |
| `../starVLA/starVLA/dataloader/gr00t_lerobot/transform/state_action.py` | `min_max` / `mean_std` 归一化公式 | 公式语义对齐，代码在 `energywam/data/normalization.py` 重写 |
| `../starVLA/starVLA/model/tools.py` `FRAMEWORK_REGISTRY` | 框架注册协议 | 通过 `energywam/integration/starvla.py` **显式注册**，不改源文件 |
| `../starVLA/examples/simBenchmarks/Robocasa_365/` | 任务契约（action12 / state16→32 / 相机顺序） | 只读；首版显式选「左视角 + state32」 |
| `../LDA-1B`（LDA 论文实现） | 双视觉编码（Qwen + DINO）设计参考 | **仅参考**。未整库拷贝，未声称同条件复现 LDA 基线 |
| Energy-Based Transformers (EBT) | 标量 Energy + 梯度迭代 + 展开式训练范式 | 方法级参考，无代码移植 |

### 4.3 许可证

* `../starVLA` 为 MIT License（见 `../starVLA/LICENSE`）；本项目只读引用其接口，未复制其源码文件。
* `../LDA-1B` 有独立 LICENSE；本项目未复制其代码。
* Qwen3-VL / DINOv3 权重各自的许可证由模型发布方约束；本项目只读加载，不重新分发。
* **未整库复制 LDA**，未因可选依赖缺失破坏其他 framework 的导入
  （`tests/test_compat.py` 显式回归）。

---

## 5. 外部 API 版本约束

Qwen3-VL 适配器依赖以下 transformers 4.57.0 内部接口，集中在
`energywam/backbones/qwen3vl_adapter.py`（若升级 transformers 需重新验证）：

* `Qwen3VLModel.get_input_embeddings()`
* `Qwen3VLModel.get_image_features(pixel_values, image_grid_thw)` → `(image_embeds, deepstack_image_embeds)`
* `Qwen3VLModel.get_placeholder_mask(...)`
* `Qwen3VLModel.get_rope_index(input_ids, image_grid_thw, video_grid_thw, attention_mask)` → `[3, B, S]`
* `Qwen3VLTextModel.forward(inputs_embeds, position_ids, attention_mask, visual_pos_masks, deepstack_visual_embeds)`

配置构造注意：`Qwen3VLConfig(vision_config=..., text_config=...)` 在 4.57.0 需要传
**dict**（传 config 实例会静默丢失属性）；`Qwen3VLTextConfig` 需要
`rope_scaling={"type": "mrope", "mrope_section": [...,...,...]}`，三段之和等于 `head_dim`。

---

## 6. 附件

工程要求提到的两张架构图（参考方法简图、目标架构图）在
`energyWAM/docs/` 下**未找到图形文件**。其文字描述已完整保留在
`docs/ENGINEERING_REQUIREMENTS.md` §1 中；如后续补图，请放入 `docs/attachments/`。

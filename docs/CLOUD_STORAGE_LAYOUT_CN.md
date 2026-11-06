# 云盘数据与训练存储布局

日期：2026-11-01。

云盘挂载点：`/cloud`，500 GB；本项目使用独立目录 `/cloud/zzc/energyWAM/`，不改动已有 `/cloud/xyx/`。

## 数据迁移

`/datasets/` 下这些目录复制到 `/cloud/zzc/energyWAM/datasets/` 的同名目录：

- `robocasa365_archives`：原始示范下载压缩包。
- `robocasa365_assets_archives`：仿真资产压缩包。
- `sim`：解压后的仿真资产。
- `sources`：固定版本源码包。
- `models`：本项目模型目录；已有 Qwen 绝对软链接仍指向共享 `/model`，不移动共享权重。
- `robocasa365`：实际训练示范数据。
- `robocasa365_trainmix`：多任务合并视图。

每个目录先 rsync 复制，再使用 rsync checksum dry-run 校验内容和布局；校验通过后，原目录切换为指向云盘的软链接，再删除本地旧副本。已有 `/datasets/...` 配置路径继续有效。

迁移脚本：`scripts/migrate_datasets_to_cloud.py`。

```bash
cat logs/cloud_migration_status.json
tail -f logs/cloud_migration.log
df -h / /cloud
```

状态为 `copying` 或 `verifying_checksums` 时，本地副本仍保留；`complete` 才表示全部列出的目录迁移完成。

## 正式训练输出

工作区路径：

- `logs/runs/energywam_formal_bi_20261101`
- `logs/runs/energywam_formal_joint_20261101`

以上均为软链接，实际存放在：

- `/cloud/zzc/energyWAM/runs/energywam_formal_bi_20261101`
- `/cloud/zzc/energyWAM/runs/energywam_formal_joint_20261101`

每 100 步保存一个 `ckpt_latest.pt`，先写临时文件并 fsync，再原子替换旧断点。写入期间需要容纳新旧两份；Joint 首个断点保存成功后删除 BI 断点，不保留大量历史快照。第一份正式断点保存后会清理旧短训权重。

训练进程由 Supervisor 管理，异常退出后自动从滚动断点恢复；恢复参数、优化器、调度器、步数、随机状态和确定性的样本位置。流水线会等待 `/cloud` 挂载，避免写回本地盘。宿主机重启后恢复还依赖 Supervisor 服务正常启动。

## 验证范围

已验证云盘基础写入、读取校验、文件原子重命名、软链接和目录 fsync 接口可用。基础 16 MiB 测试不代表大规模随机读取的稳态性能；数据迁移后需观察真实训练的数据耗时。

系统盘仅保留项目代码、小型日志和统计文件等；共享 `/model`、其他项目和系统缓存不在本次迁移范围内。

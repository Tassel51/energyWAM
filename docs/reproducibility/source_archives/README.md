# 历史源码备份

`manifest.json` 逐文件记录原路径、备份路径、字节数和SHA256。该目录保存普通Git忽略规则会漏掉的冻结代码和源码ZIP；数据、权重、虚拟环境、密钥不在其中。

- `frozen_sources/`：维护机原 `logs/frozen_sources/` 的目录结构；恢复时可复制回该位置，再按新机器修正绝对路径。
- `bundles/`：实验源码ZIP，以内容SHA256命名并去重；对应实验和原文件名见manifest。解压到新目录即可检查当时源代码，不会覆盖当前代码。
- `local_auxiliary/`：原 `logs/` 下独立辅助源码，保留相对路径。
- `experiment_auxiliary/`：原云盘实验目录里的独立取证/报告脚本，保留相对路径。

生成器为 `scripts/export_historical_sources.py`。生成后按SHA256再次核对每份副本。代码归档不包含这些历史实验的模型权重；具体权重状态见相邻的服务器迁移清单。

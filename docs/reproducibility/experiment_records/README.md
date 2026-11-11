# 原始实验记录备份

来源：`cloud_experiments` = `/cloud/zzc/energyWAM/experiments`；`lab_remote_setup` = `/root/zzc/energyWAM/logs/lab_remote_setup`。采集区间（UTC）：2026-11-11T10:24:13.939032+00:00 至 2026-11-11T10:25:49.913512+00:00。

保留 4733 个原始文本文件，原始总大小 545,983,646 bytes，压缩后 62,331,983 bytes，分为 14 个归档，每个小于45MiB。全部归档已逐成员恢复并核对SHA256与字节数。

这是逐文件采集，不是活跃文件系统的原子快照。每个文件双读核验，发生变化时重读并在manifest中记录；持续变化者明确标为最后读取版本。没有重启训练、删除源记录、截断文件或改写日志内容。

排除项 103 个，详见manifest的excluded。任何命中凭据或签名URL的记录均只列路径和原因，原件仍留在服务器，需要另行私密备份。模型权重、数据集、大型二进制和SDK调试目录不属于本归档。

学校部署记录仅选择JSON/YAML/Markdown，不笼统打包本机其他logs。各压缩包用两个恢复根前缀区分来源：`cloud_experiments/` 对应实验目录，`lab_remote_setup/` 对应项目 `logs/lab_remote_setup/`。在新目录解压全部包即可合并恢复：

```bash
mkdir -p restored_records
for archive in experiment-records-*.tar.gz; do
  tar -xzf "$archive" -C restored_records
done
```

`manifest.json` 逐文件记录原路径、字节数、SHA256、采集时间、变化检查和所属归档。

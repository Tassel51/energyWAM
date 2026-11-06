# 全量 RoboCasa365 数据与资产下载

当前为后台下载，不能视为全部完成。任务范围为官方50个target/human任务（18 atomic+32 composite）与6类厨房资产；另含固定版本RoboCasa源码和官方master版Robosuite机器人模型/mesh。

远端根目录：`/home/bingxing2/home/scx8178/energywam_storage`。

- 数据：`datasets/robocasa365/v1.0/target/{atomic,composite}/<Task>/<Date>/lerobot`。
- 厨房场景/对象/纹理：`sim/robocasa365/robocasa/models/assets`。
- 机器人及底层模型：`sim/robosuite/robosuite/models/assets`。
- 总进度：`logs/full_robocasa365_status.json`；明细：`status/*.json`。
- 最终数据/厨房资产标记：`logs/FULL_ROBOCASA365_READY.json`；另检查`logs/ROBOSUITE_ASSETS_READY.json`。

56个数据/厨房资产包合计约72.5GB，解压后约85.6GB，另有少量源码及机器人后端。OpenDrawer已有副本核验后复用，其他包逐一续传、核验、解压。两路并发，不需要GPU。专用连接会在网络中断后自动重连，长期下载的登录续期使用已授权凭据且仅保留在进程内存；重连失败时保留断点并在进度中报告。验证后清理本流程临时压缩包。账号真实存储限额未知；200GiB为脚本保护上限，不代表平台分配额度；磁盘/配额错误会暂停保留断点。

在本机服务器查看进度（每分钟刷新）：

```bash
watch -n 60 'cat /data3/xyx_zhuzicheng/projects/hpc/energyWAM/logs/full_download_status.json'
```

`tasks_ready`应最终为50，`assets_ready`为6，`robosuite_assets_ready`为true，`stage`为complete。Ctrl+C只退出监控，不停止后台下载。下载完成不代表MuJoCo仿真环境或闭环评估已通过；本次未改动训练环境或启动GPU作业。

#!/usr/bin/env python3
"""Display the local download status mirror without contacting the cluster."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import sys
import time


LOCAL_STATUS = Path('/data7/zzc/energyWAM/logs/full_robocasa365_status.json')
DEFAULT_STATUS = (LOCAL_STATUS if LOCAL_STATUS.exists() else
                  Path(__file__).resolve().parents[2] / "logs/full_download_status.json")
STAGES = {
    "downloading": "下载中",
    "verifying_results": "校验结果",
    "integrating_pinned_source": "安装指定版本源码",
    "waiting_for_source_archive": "等待源码包",
    "waiting_for_robosuite_assets": "等待机器人资产",
    "complete": "完成",
    "blocked_storage": "存储不足，已暂停",
    "failed": "失败",
    "stopped_unexpectedly": "进程意外停止",
    "connection_unavailable": "连接中断，正在恢复",
    "extracting": "解压中",
}


def size(value):
    value = float(value or 0)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(value) < 1024 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024


def bar(done, total, width):
    if not total:
        return "[" + "?" * width + "] 未知"
    fraction = max(0, min(1, done / total))
    filled = int(fraction * width)
    return f"[{'#' * filled}{'-' * (width - filled)}] {fraction:6.1%}"


def render(status, previous=None):
    disconnected = status.get("stage") == "connection_unavailable"
    data = previous if disconnected and previous else status
    width = max(10, min(40, shutil.get_terminal_size((80, 24)).columns - 30))
    lines = ["RoboCasa365 下载进度", f"阶段：{STAGES.get(status.get('stage'), status.get('stage', '等待状态'))}"]
    if disconnected and previous:
        lines.append("以下为连接中断前的进度，恢复后自动更新。")
    done = data.get("archive_progress_bytes", 0)
    total = data.get("archive_total_bytes", 0)
    lines += [
        "压缩包下载 " + bar(done, total, width),
        f"{size(done)} / {size(total) if total else '未知'}",
        f"任务就绪 {data.get('tasks_ready', '?')}/{data.get('tasks_total', 50)}"
        f"  厨房资产 {data.get('assets_ready', '?')}/{data.get('assets_total', 6)}"
        f"  机器人资产：{'就绪' if data.get('robosuite_assets_ready') else '待确认'}",
    ]
    if not disconnected:
        speed = data.get("recent_bytes_per_second")
        lines.append(f"最近采样速度：{size(speed) + '/s' if speed is not None else '未知'}")
    for item in data.get("active", []):
        label = item.get("id", "未知项目").replace("dataset_", "任务 ", 1).replace("asset_", "资产 ", 1)
        state = item.get("state", "未知")
        lines += [f"  {label}（{STAGES.get(state, state)}）",
                  "  " + bar(item.get("bytes", 0), item.get("total", 0), width)]
    stamp = data.get("monitor_updated_at") or data.get("time")
    if stamp:
        try:
            age = max(0, (datetime.now().astimezone() - datetime.fromisoformat(stamp)).total_seconds())
            lines.append(f"状态更新：{stamp}（{int(age)} 秒前{'，可能已滞后' if age > 180 else ''}）")
        except (ValueError, TypeError):
            lines.append(f"状态更新：{stamp}")
    errors = status.get("errors", [])
    if status.get("error"):
        errors = [*errors, status["error"]]
    if errors:
        lines.append("最近错误：")
        lines.extend("  " + (f"{e.get('id', '')}: {e.get('error', e)}" if isinstance(e, dict) else str(e)) for e in errors[-3:])
    else:
        lines.append("最近错误：无记录")
    refresh = 60 if 'monitor_updated_at' in data else 5
    lines += [f"下载 100% 后仍需解压、校验；状态源约每 {refresh} 秒刷新。",
              "Ctrl+C 仅退出此监控，不停止后台下载。"]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--interval", type=float, default=2)
    parser.add_argument("--once", action="store_true", help="打印一次后退出")
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be positive")
    previous = None
    last_output = None
    tty = sys.stdout.isatty() and not args.once
    try:
        while True:
            try:
                status = json.loads(args.status.read_text())
                if not isinstance(status, dict):
                    raise ValueError("状态应为 JSON 对象")
                output = render(status, previous)
                if "archive_total_bytes" in status:
                    previous = status
            except (OSError, ValueError) as exc:
                output = f"等待状态文件：{args.status}\n{exc}"
                if previous:
                    output += "\n上次成功读取：\n" + render(previous)
            if tty:
                sys.stdout.write("\033[H\033[2J" + output + "\n")
                sys.stdout.flush()
            elif output != last_output:
                print(output, flush=True)
            last_output = output
            if args.once:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n已退出监控，后台下载继续。")


if __name__ == "__main__":
    main()

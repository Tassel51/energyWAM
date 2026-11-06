"""Bounded singleton supervisor for the local, resumable RoboCasa365 queue."""
import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    args = p.parse_args()
    root = args.root.resolve()
    logs = root / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    lock = (logs / 'local_download_supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    pipeline = Path(__file__).with_name('full_robocasa_download_pipeline.py')
    started = time.monotonic()
    for attempt in range(24):
        if time.monotonic() - started > 72 * 3600:
            print('72-hour retry budget exhausted; partial files preserved', flush=True)
            return 1
        # Both are existing local proxies. A new attempt alternates routes;
        # no remote SSH connection or cloud login is needed for this queue.
        proxy = f'http://127.0.0.1:{(7901, 27904)[attempt % 2]}'
        print(json.dumps(dict(time=datetime.now().astimezone().isoformat(),
                              supervisor_pid=os.getpid(), attempt=attempt + 1,
                              proxy=proxy)), flush=True)
        result = subprocess.run([sys.executable, '-u', str(pipeline),
                                 '--root', str(root), '--python', sys.executable,
                                 '--proxy', proxy], stdin=subprocess.DEVNULL)
        status_path = logs / 'full_robocasa365_status.json'
        try:
            status = json.loads(status_path.read_text())
        except (OSError, ValueError):
            return result.returncode or 1
        if result.returncode == 0 and status.get('stage') == 'complete':
            return 0
        errors = status.get('errors') or []
        transient = ('urlerror', 'timeout', 'timed out', 'connection',
                     'ssl', 'truncated body', 'incomplete', 'http error 429',
                     'http error 500', 'http error 502', 'http error 503',
                     'http error 504')
        if status.get('stage') != 'failed' or not errors or not all(
            any(term in str(e.get('error', '')).lower() for term in transient)
            for e in errors
        ):
            print('Stopped for a non-network error; existing data preserved', flush=True)
            return result.returncode or 1
        print('Network failure: preserving partials; retry in 300 seconds', flush=True)
        time.sleep(300)
    return 1


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except BlockingIOError:
        print('Existing local supervisor preserved; no duplicate started', flush=True)

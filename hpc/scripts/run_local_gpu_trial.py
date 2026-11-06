"""Run the authorized ten-step local trial with a GPU lock and resource guard."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--gpu', required=True)
    p.add_argument('--deduplicate-parameter-offload', action='store_true')
    args = p.parse_args()
    root = Path(__file__).resolve().parents[2]
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock = Path('/tmp/energywam-' + args.gpu + '.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    memory = subprocess.check_output([
        'nvidia-smi', '-i', args.gpu, '--query-gpu=memory.free',
        '--format=csv,noheader,nounits'], text=True)
    if int(memory.strip()) < 30000:
        raise RuntimeError('Requested GPU no longer has 30000 MiB free; no training started')
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=args.gpu, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
               ENERGYWAM_ALLOW_GPU_TRAINING='1', PYTHONUNBUFFERED='1')
    command = [sys.executable, '-u', str(root / 'hpc/scripts/gpu_preflight.py'),
               '--config', args.config, '--steps', '10', '--offload-saved-tensors',
               '--low-memory-adamw', '--checkpoint', 'local_gpu1_bi_step10.pt',
               '--out', str(out / 'result.json')]
    if args.deduplicate_parameter_offload:
        command.append('--deduplicate-parameter-offload')
    log = out / 'training.log'
    with log.open('x') as output:
        child = subprocess.Popen(command, cwd=root, env=env, stdin=subprocess.DEVNULL,
                                 stdout=output, stderr=subprocess.STDOUT)
    started = time.monotonic()
    reason = None
    while True:
        code = child.poll()
        latest = {}
        for line in log.read_text(errors='replace').splitlines()[-100:]:
            try:
                event = json.loads(line)
                if isinstance(event, dict) and 'stage' in event:
                    latest = event
            except ValueError:
                pass
        rss_kib = 0
        try:
            for line in Path(f'/proc/{child.pid}/status').read_text().splitlines():
                if line.startswith('VmRSS:'):
                    rss_kib = int(line.split()[1])
        except FileNotFoundError:
            pass
        available_kib = next(int(line.split()[1]) for line in
                             Path('/proc/meminfo').read_text().splitlines()
                             if line.startswith('MemAvailable:'))
        if code is None and not reason:
            if rss_kib > 50 * 1024**2 or available_kib < 12 * 1024**2:
                reason = 'Host memory guard: preserving other users workloads'
            elif time.monotonic() - started > 30 * 60:
                reason = '30-minute engineering-trial limit reached'
            if reason:
                child.terminate()
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    child.kill()
                code = child.wait()
        stage = ('complete' if code == 0 else 'failed') if code is not None else latest.get('stage', 'starting')
        data = dict(stage=stage, time=datetime.now().astimezone().isoformat(),
                    supervisor_pid=os.getpid(), training_pid=child.pid, gpu_uuid=args.gpu,
                    elapsed_seconds=round(time.monotonic() - started),
                    rss_mib=round(rss_kib / 1024), exit_code=code, error=reason,
                    latest_event=latest, log=str(log))
        status = out / 'status.json'
        temporary = status.with_suffix('.tmp')
        temporary.write_text(json.dumps(data, indent=2))
        temporary.replace(status)
        if code is not None:
            print(json.dumps(data), flush=True)
            return code
        time.sleep(5)


if __name__ == '__main__':
    raise SystemExit(main())

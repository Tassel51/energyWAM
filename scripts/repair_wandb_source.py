"""Repair one completed source in its existing W&B run, using verified history gaps."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import watch_wandb as bridge


def require_uploader_stopped(pid, proc_root=Path('/proc')):
    """A stopped Supervisor program may leave a zombie; neither can write."""
    if pid <= 0:
        raise ValueError('A positive paused uploader PID is required')
    proc = Path(proc_root) / str(pid)
    try:
        status = (proc / 'status').read_text()
        command = (proc / 'cmdline').read_bytes()
    except FileNotFoundError:
        if not proc.exists():
            return 'exited'
        # Recheck a disappearing /proc entry rather than treating unreadable as stopped.
        try:
            status = (proc / 'status').read_text()
        except FileNotFoundError:
            if not proc.exists():
                return 'exited'
            raise RuntimeError('Cannot establish paused uploader state')
        command = b''
    state = next((line.split()[1] for line in status.splitlines()
                  if line.startswith('State:')), None)
    if state == 'Z':
        return 'zombie'
    # Even a reused live PID is rejected: it is not evidence the requested writer stopped.
    if command or state != 'Z':
        raise RuntimeError('Paused uploader PID is still alive; stop its supervisor first')


def recovery_plan(folder, cfg, report, resource, rows, now=None):
    now = time.time() if now is None else now
    age = now - float(report['checked_unix'])
    if not 0 <= age < 1800:
        raise ValueError('Refresh independent server verification before repair')
    rid = hashlib.sha256(str(folder.resolve()).encode()).hexdigest()[:12]
    if report.get('run_id') != rid or Path(report['source_folder']).resolve() != folder.resolve():
        raise ValueError('Verification belongs to a different source or run')
    if report.get('state') not in ('crashed', 'failed', 'finished'):
        raise ValueError('Existing remote run must be terminal before isolated repair')
    if not isinstance(report.get('mismatches'), list) or report['mismatches']:
        raise ValueError('Mismatched existing values require investigation, not replay')
    if not isinstance(report.get('missing'), list):
        raise ValueError('Verification must include explicit missing metric points')
    target = int(cfg['trainer']['max_train_steps'])
    if target <= 0 or sorted(rows) != list(range(1, target + 1)):
        raise ValueError('Authoritative local update history is not complete and contiguous')
    if report.get('expected_updates') != target:
        raise ValueError('Verification target does not match authoritative source')
    if (resource.get('returncode') != 0 or resource.get('exit_classification') != 'completed'
            or int(resource.get('last_sample', {}).get('completed_updates', -1)) != target):
        raise ValueError('Source experiment did not formally complete its configured budget')
    ack = report.get('server_last_uploaded_update')
    if not isinstance(ack, (int, float)) or isinstance(ack, bool) or int(ack) != ack:
        raise ValueError('Verification must contain a numeric integral server ACK')
    ack = int(ack)
    if not 0 <= ack <= target:
        raise ValueError('Server ACK is outside the actual source update range')
    missing_train = []
    for key in report['missing']:
        if not isinstance(key, (list, tuple)) or len(key) != 3:
            raise ValueError('Malformed missing metric key')
        if key[0] == 'train':
            step = int(key[1])
            if not 1 <= step <= target:
                raise ValueError('Missing training update is outside source budget')
            missing_train.append(step)
    replay_after = min(ack, min(missing_train) - 1) if missing_train else ack
    return dict(run_id=rid, target_updates=target, verified_server_ack=ack,
                replay_after_update=replay_after,
                missing_metric_points=len(report['missing']))


def replay_state(original, replay_after):
    state = bridge.reconcile_submitted_state(original, replay_after)
    # Replaying probes is intentional even when only probes (rather than train rows) are missing.
    state['probe_files'] = {}
    state['validation_files'] = {}
    for key in ['saved_step', 'last_speed_step', 'collector_finished', 'collector_exit_code']:
        state.pop(key, None)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settings', required=True)
    parser.add_argument('--source-config', required=True)
    parser.add_argument('--verification', required=True)
    parser.add_argument('--receipt', required=True)
    parser.add_argument('--paused-uploader-pid', type=int, required=True)
    args = parser.parse_args()
    settings = json.loads((ROOT / args.settings).read_text())
    if settings.get('mode', 'online') != 'online':
        raise ValueError('Server repair requires online settings')
    config = ROOT / args.source_config
    cfg = yaml.safe_load(config.read_text())
    folder = ROOT / cfg['trainer']['output_dir']
    rid = hashlib.sha256(str(folder.resolve()).encode()).hexdigest()[:12]
    state_path = ROOT / 'logs/wandb_bridge' / f'{rid}_online.json'
    state_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = state_path.with_name(f'{rid}_repair.lock')
    receipt = ROOT / args.receipt
    with lock_path.open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another repair for this source is already running') from None
        paused = require_uploader_stopped(args.paused_uploader_pid)
        report = json.loads((ROOT / args.verification).read_text())
        resource = json.loads((folder / 'resource_summary.json').read_text())
        rows = bridge.read_rows(folder / 'train_log.jsonl')
        plan = recovery_plan(folder, cfg, report, resource, rows)
        original = json.loads(state_path.read_text()) if state_path.exists() else {}
        state = replay_state(original, plan['replay_after_update'])
        record = dict(started_unix=time.time(), pid=os.getpid(), source_folder=str(folder),
                      source_config=str(config), verification=str(ROOT / args.verification),
                      paused_uploader_pid=args.paused_uploader_pid, paused_uploader_state=paused,
                      original_local_state=original, training_restarted=False,
                      needs_server_verification=True, phase='starting', **plan)
        bridge.atomic_json(receipt, record)
        os.environ['WANDB_API_KEY'] = Path(settings['api_key_file']).read_text().strip()
        os.environ['WANDB_BASE_URL'] = settings.get('base_url', 'https://api.wandb.ai')
        bridge.apply_proxy_settings(settings)
        require_uploader_stopped(args.paused_uploader_pid)
        import wandb
        try:
            run = wandb.init(entity=settings['entity'], project=settings['project'], id=rid,
                             resume='must', mode='online', name=cfg['run_id'], dir=settings['cache_dir'],
                             settings=wandb.Settings(init_timeout=60, x_disable_stats=True,
                                                     disable_code=True, save_code=False, console='off'))
            for axis in ['train/update', 'validation/update', 'checkpoint/update', 'probe/update']:
                run.define_metric(axis)
            for prefix in ['train/*', 'performance/*', 'memory/*', 'diagnostics/*', 'solver/*']:
                run.define_metric(prefix, step_metric='train/update')
            run.define_metric('probe/*', step_metric='probe/update')
            run.define_metric('validation/*', step_metric='validation/update')
            run.define_metric('checkpoint/*', step_metric='checkpoint/update')
            run.define_metric('comparison/sample_visits')
            run.define_metric('comparison/*', step_metric='comparison/sample_visits')
            item = dict(run=run, cfg=cfg, folder=folder, state=state, state_path=state_path,
                        url=run.url, source_config=str(config))
            bridge.poll(item)
            if not item.get('finished'):
                raise RuntimeError('Source completion prerequisites did not close the SDK run')
            record.update(ended_unix=time.time(), phase='sdk_finished',
                          submitted_through=item['state'].get('last_step'))
            bridge.atomic_json(receipt, record)
            print(json.dumps({'run_id': rid, 'phase': 'sdk_finished',
                              'submitted_through': item['state'].get('last_step'),
                              'needs_server_verification': True}), flush=True)
        except Exception as error:
            record.update(ended_unix=time.time(), phase='failed', error_type=type(error).__name__)
            bridge.atomic_json(receipt, record)
            # Do not print SDK exception text: it may embed signed storage URLs.
            raise SystemExit(f'Repair failed: {type(error).__name__}; see sanitized receipt') from None


if __name__ == '__main__':
    main()

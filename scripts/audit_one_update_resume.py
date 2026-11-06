"""Isolated full-state resume replay; never rewrite the source checkpoint."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
AUDIT_OUTPUT = None


def save(path, data):
    temporary = path.with_suffix('.pending.json')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def compare_state(actual, expected, stats, full=False):
    import numpy as np
    import torch
    if isinstance(expected, torch.Tensor):
        if actual.shape != expected.shape or actual.dtype != expected.dtype:
            raise AssertionError('Restored tensor shape/dtype differs')
        a, b = actual.detach().reshape(-1), expected.detach().reshape(-1)
        if not full and a.numel() > 128:
            a, b = torch.cat([a[:64], a[-64:]]), torch.cat([b[:64], b[-64:]])
        if not torch.equal(a.cpu(), b.cpu()):
            raise AssertionError('Restored tensor values differ')
        stats['tensors'] += 1
        stats['values_compared'] += a.numel()
    elif isinstance(expected, np.ndarray):
        if not np.array_equal(actual, expected):
            raise AssertionError('Restored NumPy state differs')
    elif isinstance(expected, dict):
        if actual.keys() != expected.keys():
            raise AssertionError('Restored state keys differ')
        for key in expected:
            compare_state(actual[key], expected[key], stats, full)
    elif isinstance(expected, (list, tuple)):
        if len(actual) != len(expected):
            raise AssertionError('Restored state length differs')
        for a, b in zip(actual, expected):
            compare_state(a, b, stats, full)
    elif actual != expected:
        raise AssertionError('Restored scalar state differs')


def worker(args, checkpoint, step):
    import torch
    from energywam.training.shared_microbatch import SharedMicrobatchTrainer
    from energywam.training.checkpoint import _rng_state
    import scripts.train_shared_sharded as entry
    original = SharedMicrobatchTrainer.load

    def audited_load(trainer, path, *positional, **kwargs):
        meta = original(trainer, path, *positional, **kwargs)
        payload = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        checks = {}
        for name, actual, expected, full in [
            ('model', trainer.model.state_dict(), payload['model'], False),
            ('optimizer', trainer.optimizer.state_dict(), payload['optimizer'], False),
            ('scheduler', trainer.scheduler.state_dict(), payload['scheduler'], True),
            ('rng', _rng_state(), payload['rng'], True),
        ]:
            stats = dict(tensors=0, values_compared=0)
            compare_state(actual, expected, stats, full)
            checks[name] = stats
        if trainer.global_step != step:
            raise AssertionError('Restored update counter differs')
        save(args.output / 'restored_state_check.json', dict(
            passed=True, restored_step=step, checks=checks,
            scope='All tensor keys/shapes/dtypes and first/last64 values checked; RNG fully checked. Not a full4B parameter hash.'))
        return meta

    SharedMicrobatchTrainer.load = audited_load
    sys.argv = ['train_shared_sharded.py', '--config', str(args.config_output),
                '--steps', str(step + 1), '--resume', str(checkpoint),
                '--fixed-window-manifest', str(args.output / 'sample_manifest.json'),
                '--physical-gpus', args.physical_gpus, '--profile']
    entry.main()


def main():
    global AUDIT_OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config-output', type=Path, required=True)
    parser.add_argument('--physical-gpus', choices=['0,1', '2,3'], default='0,1')
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    AUDIT_OUTPUT = args.output
    checkpoint = args.source_run / 'ckpt_latest.pt'
    meta = json.loads(checkpoint.with_name(checkpoint.name + '.meta.json').read_text())
    step = int(meta['meta']['global_step'])
    if args.worker:
        worker(args, checkpoint, step)
        return
    os.environ['CUDA_VISIBLE_DEVICES'] = ''  # The parent is CPU-only.
    import psutil
    import yaml
    from scripts.run_iteration_search import gpu_idle
    from energywam.training.resource_monitor import ResourceMonitor
    if args.output.exists():
        raise RuntimeError('Audit output exists; inspect its process/result, never duplicate blindly')
    resource = json.loads((args.source_run / 'resource_summary.json').read_text())
    if resource.get('returncode') is None or psutil.pid_exists(resource['pid']):
        raise RuntimeError('Audit requires a terminal source run and stable checkpoint')
    if not gpu_idle(args.physical_gpus):
        raise RuntimeError('Assigned audit GPU pair is busy')
    args.output.mkdir(parents=True)
    cfg = json.loads((args.source_run / 'resolved_config.json').read_text())
    cfg['run_id'] = args.output.name
    cfg['trainer'].update(output_dir=str(args.output), max_train_steps=step + 1)
    args.config_output.write_text(yaml.safe_dump(cfg, sort_keys=False))
    import zipfile
    with zipfile.ZipFile(args.output / 'source_resume_audit.zip', 'w',
                         compression=zipfile.ZIP_DEFLATED) as archive:
        for source in (ROOT / 'energywam').rglob('*.py'):
            archive.write(source, str(source.relative_to(ROOT)))
        for name in ['train_shared_sharded.py', 'audit_one_update_resume.py']:
            archive.write(ROOT / 'scripts' / name, 'scripts/' + name)
        archive.write(args.config_output, 'experiment_config.yaml')
    for name in ['train_log.jsonl', 'timing_history.jsonl', 'speed_report.json',
                 'best_selection.json', 'checkpoint_history.jsonl', 'best_history.jsonl',
                 'final_report.json', 'sample_manifest.json', 'source_initial.zip']:
        source = args.source_run / name
        if source.exists():
            shutil.copy2(source, args.output / name)
    for folder in [Path('.'), Path('train_monitor')]:
        for source in (args.source_run / folder).glob('eval_*.json'):
            target = args.output / folder / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    before = checkpoint.stat()
    identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.physical_gpus,
               ENERGYWAM_TRIM_BETWEEN_DIRECTIONS='0', ENERGYWAM_SAMPLE_ALIGNED_RNG='1',
               ENERGYWAM_FIXED_DATA_CACHE='1', ENERGYWAM_ONLY_DIRECTION='',
               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1',
               PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True', TOKENIZERS_PARALLELISM='false')
    command = [sys.executable, '-u', str(Path(__file__).resolve()), *sys.argv[1:], '--worker']
    status = args.output / 'audit_status.json'
    with (args.output / 'console.log').open('a') as log:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        monitor = ResourceMonitor(child.pid, args.physical_gpus.split(','), args.output)
        monitor.start()
        save(status, dict(phase='running', pid=child.pid, source_step=step, target_step=step + 1))
        start = time.time()
        reason = None
        while child.poll() is None:
            if time.time() - start > 1200:
                reason = 'resume audit time budget exceeded'
            if monitor.summary.get('last_sample', {}).get('process_uss_gib', 0) > 150:
                reason = 'resume audit USS budget exceeded'
            if reason:
                import signal
                os.killpg(child.pid, signal.SIGTERM)
                child.wait(timeout=55)
                break
            time.sleep(2)
        monitor.finish(child.returncode, reason)
    after = checkpoint.stat()
    if identity != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise AssertionError('Source checkpoint identity changed')
    if child.returncode:
        save(status, dict(phase='failed', pid=child.pid, returncode=child.returncode, reason=reason))
        raise RuntimeError('Resume audit failed; inspect preserved console/resource evidence')
    original = {int(x['global_step_completed']): x for x in map(json.loads,
                (args.source_run / 'timing_history.jsonl').read_text().splitlines())}
    replay = json.loads((args.output / 'timing_history.jsonl').read_text().splitlines()[-1])
    import math
    comparisons = {}
    for key in ['bi/action_loss', 'bi/future_loss', 'diag/alpha_action', 'diag/alpha_future']:
        expected, actual = original[step + 1][key], replay[key]
        comparisons[key] = dict(original=expected, replay=actual, absolute_difference=abs(actual - expected))
        if not math.isclose(actual, expected, rel_tol=1e-5, abs_tol=1e-6):
            raise AssertionError('Replayed loss/step size does not match original next update')
    marker = json.loads((args.output / 'resume_history/latest.json').read_text())
    if marker['restored_step'] != step or replay['global_step_completed'] != step + 1:
        raise AssertionError('Resume/output rewind counters differ')
    if any(args.output.glob('ckpt*.pt')):
        raise AssertionError('Profile audit unexpectedly wrote a large checkpoint')
    save(args.output / 'resume_audit_result.json', dict(
        passed=True, restored_step=step, replayed_step=step + 1,
        comparisons=comparisons, source_checkpoint_identity_unchanged=True,
        archived_tail=marker['archive_dir'], new_full_checkpoints_written=False,
        counts_toward_main_budget=False, scope='One isolated full-state optimizer update and output-tail rewind; source run untouched.'))
    save(status, dict(phase='complete', pid=child.pid, returncode=0, passed=True))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        if AUDIT_OUTPUT is not None and AUDIT_OUTPUT.exists():
            save(AUDIT_OUTPUT / ('worker_error.json' if '--worker' in sys.argv else 'audit_error.json'),
                 dict(phase='failed', error_type=type(error).__name__, reason=str(error)))
        raise

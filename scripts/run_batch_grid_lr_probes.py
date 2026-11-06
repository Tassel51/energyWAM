"""Durable serial LR probes after main trials and the placement control.

The existing queue enforces GPU idleness, numerical/resource guards and full
latest/best disk reservation. Serial execution avoids reserving two new runs
when storage permits only one. No weights are removed by this controller.
"""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

from scripts.run_batch_grid_hardware_control import ROOT, R, main_arms_ready, read, write, summarize


def validated_jobs(root=R, config_root=ROOT):
    plan = read(root / 'matched_lr_probes_pending_plan.json')
    audit = read(root / 'matched_lr_prelaunch_config_audit.json')
    expected = {(b, f) for b in [8, 16, 32, 64] for f in [.3, 3.]}
    jobs = plan['jobs']
    if not audit.get('passed') or len(jobs) != len(expected):
        raise RuntimeError('Expected eight audited finite LR probes')
    proofs = {row['name']: row for row in audit['configurations']}
    seen = set()
    for job in jobs:
        key = (job['effective_batch'], job['outer_lr_factor'])
        if key not in expected or key in seen:
            raise RuntimeError('Duplicate or unexpected batch/LR probe')
        seen.add(key)
        proof = proofs.get(job['name'], {})
        path = config_root / job['config']
        if (hashlib.sha256(path.read_bytes()).hexdigest() != proof.get('config_sha256')
                or proof.get('batch') != key[0] or proof.get('factor') != key[1]
                or proof.get('sample_budget') != 1024
                or proof.get('warmup_sample_visits') != 256
                or job['steps'] * job['effective_batch'] != 1024
                or not job.get('keep_best') or job.get('profile')
                or job.get('only_direction')
                or job['physical_gpus'] not in ['0,1', '2,3']
                or type(job.get('snapshot_bytes_estimate')) is not int
                or job['snapshot_bytes_estimate'] <= 0):
            raise RuntimeError(f"Unverified LR probe configuration: {job['name']}")
    return jobs


def prerequisites_ready(root=R):
    if not main_arms_ready(root):
        return False
    status = read(root / 'hardware_control_controller_status.json')
    if status['phase'] == 'failed':
        raise RuntimeError('Hardware control failed; inspect before LR probes')
    if status['phase'] != 'complete':
        return False
    # Recheck real trial outputs, not just the controller's phase string.
    summarize(root, read(root / 'hardware_control_pending_plan.json')['jobs'])
    return True


def main():
    status_path = R / 'lr_probe_controller_status.json'
    jobs = validated_jobs()
    while not prerequisites_ready():
        write(status_path, dict(phase='waiting_for_main_and_hardware_control',
                                updated_unix=time.time()))
        time.sleep(30)
    job_path = R / 'matched_lr_serial_jobs.json'
    queue_status = R / 'matched_lr_serial_status.json'
    write(job_path, jobs)
    with (R / 'matched_lr_serial_queue.log').open('a') as log:
        child = subprocess.Popen([sys.executable, '-u', 'scripts/run_iteration_search.py',
                                  '--jobs', str(job_path), '--status', str(queue_status)],
                                 cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        while child.poll() is None:
            state = read(queue_status) if queue_status.exists() else {}
            waiting = [dict(name=n, **row) for n, row in state.items()
                       if row.get('phase', '').startswith('waiting_')]
            write(status_path, dict(phase='serial_lr_queue_running', queue_pid=child.pid,
                                    queue_status=str(queue_status), waiting=waiting,
                                    completed_or_failed=[n for n, row in state.items()
                                                         if row.get('phase') in ['complete', 'failed']],
                                    updated_unix=time.time()))
            time.sleep(30)
        if child.returncode:
            raise RuntimeError(f'LR queue exited {child.returncode}')
    state = read(queue_status)
    if any(state.get(j['name'], {}).get('phase') not in ['complete', 'failed'] for j in jobs):
        raise RuntimeError('LR queue exited with nonterminal jobs')
    subprocess.run([sys.executable, 'scripts/report_batch_grid.py', '--lr-probes'],
                   cwd=ROOT, check=True)
    write(status_path, dict(phase='finite_lr_probes_terminal_review_required',
                            queue_status=str(queue_status),
                            failed_jobs=[n for n, row in state.items() if row['phase'] == 'failed'],
                            note='Not final selection: selected LRs need common-budget training and finalist seed confirmation.',
                            updated_unix=time.time()))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        write(R / 'lr_probe_controller_status.json',
              dict(phase='failed', reason=str(error), updated_unix=time.time()))
        raise

"""Run the predeclared hardware control only after all four main arms finish."""
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
R = Path('/cloud/zzc/energyWAM/experiments/batch_grid_micro4_20261105')


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    temporary = path.with_suffix('.pending.json')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main_arms_ready(root):
    """Require terminal main trials; only reviewed numerical guards are exceptions.

    An accepted guarded failure remains a failed, incomplete-budget result.
    Unknown controller failures, live trials and OOM cases cannot pass this gate.
    """
    status = read(root / 'main_controller_status.json')
    failed_controller = status['phase'] == 'preflight_or_main_failed'
    if status['phase'] not in ['main_four_arms_complete', 'preflight_or_main_failed']:
        return False
    accepted_path = root / 'accepted_main_guard_failures.json'
    accepted = read(accepted_path).get('runs', {}) if accepted_path.exists() else {}
    if failed_controller and not any(
            status.get('reason', '').startswith(f'Queue arm failed or incomplete: {name}:')
            for name in accepted):
        raise RuntimeError('Main controller failed for an unreviewed reason; inspect first')
    observed_failures = []
    for round_number, batches in [(1, [8, 16]), (2, [32, 64])]:
        for pair, batch in zip(['01', '23'], batches):
            name = f'batch{batch}_micro4_fp32_seed42'
            state = read(root / f'main_round{round_number}_status_{pair}.json')[name]
            if (state.get('phase') == 'complete' and state.get('returncode') == 0
                    and state.get('completed_updates') == 4096 // batch):
                continue
            proof = accepted.get(name, {})
            if (state.get('phase') == 'failed' and state.get('early_stopped') is True
                    and proof and state.get('returncode') == -15
                    and 0 < state.get('completed_updates', 0) <= 4096 // batch
                    and state.get('reason', '').startswith('Fixed train ')
                    and 'MSE' in state['reason'] and 'untrained prediction baseline' in state['reason']
                    and all(state.get(k) == proof.get(k) for k in
                            ['reason', 'returncode', 'completed_updates', 'pid'])):
                resource = read(root / name / 'resource_summary.json')
                events = resource['last_sample']['memory_events']
                before = resource['memory_events_before']
                if (resource.get('exit_classification') != 'experiment_guard'
                        or resource.get('returncode') != -15
                        or any(events[k] - before.get(k, 0) for k in ['oom', 'oom_kill'])):
                    raise RuntimeError('Reviewed numerical guard does not match resource evidence')
                import psutil
                if psutil.pid_exists(state['pid']):
                    raise RuntimeError('Reviewed failed process still exists; inspect handle')
                observed_failures.append(name)
            else:
                raise RuntimeError(f'Main arm incomplete: {name}')
    if failed_controller and not any(
            status['reason'].startswith(f'Queue arm failed or incomplete: {name}:')
            for name in observed_failures):
        raise RuntimeError('Controller failure does not match reviewed terminal arms')
    return True


def summarize(root, jobs):
    import yaml
    rows = []
    for job in jobs:
        cfg = yaml.safe_load((ROOT / job['config']).read_text())
        folder = Path(cfg['trainer']['output_dir'])
        state = read(root / f"hardware_control_status_{job['physical_gpus'].replace(',', '')}.json")[job['name']]
        if (state.get('phase') != 'complete' or state.get('returncode') != 0
                or state.get('completed_updates') != 6):
            raise RuntimeError(f"Hardware control incomplete: {job['name']}")
        timing = [json.loads(line) for line in (folder / 'timing_history.jsonl').read_text().splitlines() if line.strip()]
        unique = {int(t['global_step_completed']): t for t in timing}
        if set(unique) != set(range(1, 7)):
            raise RuntimeError('Expected exactly six observed optimizer updates')
        initialization = read(folder / 'initialization_step0.json')
        rows.append(dict(name=job['name'], physical_gpus=job['physical_gpus'],
                         steady_median_seconds_per_sample=statistics.median(
                             unique[i]['end_to_end_seconds'] / 16 for i in range(3, 7)),
                         sampled_initial_parameter_sha256=initialization['sampled_parameter_sha256'],
                         resource_summary=str(folder / 'resource_summary.json')))
    if len({r['sampled_initial_parameter_sha256'] for r in rows}) != 1:
        raise RuntimeError('Hardware-control initializations differ')
    return dict(phase='complete', rows=rows,
                scope='B16/native4/K2, same fresh initialization/order, six updates per GPU pair; first two excluded. Short descriptive placement control, not batch quality or a long-run throughput guarantee.')


def validate_jobs(jobs, config_root=ROOT):
    if len(jobs) != 2 or any(not j.get('profile') or j['steps'] != 6
                             or j['effective_batch'] != 16 for j in jobs):
        raise RuntimeError('Unexpected hardware-control budget')
    if {j['physical_gpus'] for j in jobs} != {'0,1', '2,3'}:
        raise RuntimeError('Hardware controls require distinct GPU pairs')
    import yaml
    configs = [yaml.safe_load((config_root / job['config']).read_text()) for job in jobs]
    if len({c['trainer']['output_dir'] for c in configs}) != 2:
        raise RuntimeError('Hardware controls require separate output directories')
    for cfg in configs:
        if (cfg['trainer']['shared_microbatch_size'] != 4
                or cfg['trainer']['per_device_batch_size'] != 16
                or cfg['backbone']['dtype'] not in ['float32', 'fp32']):
            raise RuntimeError('Hardware-control math/execution changed')
        cfg['trainer'].pop('output_dir')
        cfg.pop('run_id', None)  # Display identity, not a model/data parameter.
    if configs[0] != configs[1]:
        raise RuntimeError('Hardware controls differ beyond run/output identity')


def main():
    status_path = R / 'hardware_control_controller_status.json'
    plan = read(R / 'hardware_control_pending_plan.json')
    jobs = plan['jobs']
    validate_jobs(jobs)
    while not main_arms_ready(R):
        write(status_path, dict(phase='waiting_for_four_main_arms', updated_unix=time.time()))
        time.sleep(30)
    approval_path = R / 'accepted_main_guard_failures.json'
    terminal_failures = list(read(approval_path).get('runs', {})) if approval_path.exists() else []
    # Profile mode forbids large checkpoint writes. GPU-idle checks and resource
    # guards remain in the existing queue; no existing process is signalled.
    children = []
    for job in jobs:
        pair = job['physical_gpus'].replace(',', '')
        path = R / f'hardware_control_jobs_{pair}.json'
        write(path, [job])
        log = (R / f'hardware_control_queue_{pair}.log').open('a')
        command = [sys.executable, '-u', 'scripts/run_iteration_search.py',
                   '--jobs', str(path), '--status', str(R / f'hardware_control_status_{pair}.json')]
        child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                 env=dict(os.environ))
        children.append((child, log))
    write(status_path, dict(phase='hardware_control_running',
                            queue_pids=[c.pid for c, _ in children],
                            reviewed_terminal_main_failures=terminal_failures,
                            updated_unix=time.time()))
    for child, log in children:
        code = child.wait()
        log.close()
        if code:
            raise RuntimeError(f'Hardware-control queue exited {code}')
    result = summarize(R, jobs)
    result['reviewed_terminal_main_failures'] = terminal_failures
    result['updated_unix'] = time.time()
    write(R / 'hardware_control_results.json', result)
    write(status_path, result)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        write(R / 'hardware_control_controller_status.json',
              dict(phase='failed', reason=str(error), updated_unix=time.time()))
        raise

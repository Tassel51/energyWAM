"""Bounded, serial head-LR study on GPU23; leaves the GPU01 job untouched."""
import argparse
import copy
import hashlib
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = Path('/cloud/zzc/energyWAM/experiments/head_lr_tiny16_20261106')
REFERENCE = Path('/cloud/zzc/energyWAM/experiments/tiny_future_weight_20261106')
RATES = [('lr0001', 1e-4), ('lr0003', 3e-4), ('lr001', 1e-3), ('lr003', 3e-3)]
SNAPSHOT = 24568519377
CHILD = None
STOP = False


def stop(*_):
    global STOP
    STOP = True
    if CHILD is not None and CHILD.poll() is None:
        CHILD.send_signal(signal.SIGTERM)


def run_queue(queue, status):
    global CHILD
    CHILD = subprocess.Popen([sys.executable, '-u', str(ROOT/'scripts/run_iteration_search.py'),
                              '--jobs', str(queue), '--status', str(status)], cwd=ROOT)
    code = CHILD.wait()
    CHILD = None
    if STOP:
        raise SystemExit('Controller stopped; owned queue was asked to stop its child')
    if code:
        raise RuntimeError(f'Queue exited {code}')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    path = Path(path)
    temp = path.with_suffix('.pending.json')
    temp.write_text(json.dumps(data, indent=2))
    if path.exists():
        temp.chmod(path.stat().st_mode & 0o777)
    temp.replace(path)


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    base_path = ROOT/'configs/experiments/tiny16_futureweight9_b16_micro4_seed42.yaml'
    base = yaml.safe_load(base_path.read_text())
    for filename in ['train_val_manifest.json', 'train_monitor_manifest.json']:
        source = REFERENCE/filename
        dest = OUT/filename
        if dest.exists() and dest.read_bytes() != source.read_bytes():
            raise ValueError('Existing study manifest changed')
        if not dest.exists():
            dest.write_bytes(source.read_bytes())
    jobs = []
    for tag, rate in RATES:
        cfg = copy.deepcopy(base)
        baseline = rate == 1e-3
        name = base['run_id'] if baseline else f'head_{tag}_tiny16_b16_micro4_seed42'
        config = base_path if baseline else ROOT/f'configs/experiments/{name}.yaml'
        if not baseline:
            cfg['run_id'] = name
            cfg['trainer'].update(output_dir=str(OUT/name), max_train_steps=200,
                                  energy_head_learning_rate=rate)
            cfg['notes'] = ('Head LR only; fresh seed42, fixed16 TRAIN/16 VAL, '
                            'FP32/K2/native4/effective16, current-DINO init, dual prior, '
                            'learnable solver steps, Future weight9. Screen50 then200 updates.')
            if config.exists() and yaml.safe_load(config.read_text()) != cfg:
                raise ValueError('Existing LR config differs')
            config.write_text(yaml.safe_dump(cfg, sort_keys=False))
        else:
            # The schedule has no max-step dependence. Preserve every LR and
            # source snapshot; only extend this arm's declared training budget.
            if base['trainer']['energy_head_learning_rate'] != rate or base['trainer']['warmup_steps'] != 16:
                raise ValueError('Unexpected baseline LR/schedule')
            cfg['trainer']['max_train_steps'] = 200
            config.write_text(yaml.safe_dump(cfg, sort_keys=False))
        job = dict(name=name, config=str(config.relative_to(ROOT)), physical_gpus='2,3',
                   steps=200, manifest=str(OUT/'train_val_manifest.json'),
                   train_monitor_manifest=str(OUT/'train_monitor_manifest.json'),
                   sample_eval_interval=160, train_monitor_sample_interval=160,
                   keep_best=True, monitor_resources=True, effective_batch=16,
                   guard_min_sample_visits=160, max_fixed_mse_ratio=10,
                   max_process_uss_gib=140, max_wall_seconds=8*3600,
                   snapshot_bytes_estimate=SNAPSHOT,
                   external_checkpoint_reserve_bytes=SNAPSHOT)
        jobs.append(job)
    write(OUT/'arms.json', jobs)
    write(OUT/'plan.json', dict(rates=[r for _, r in RATES], train_windows=16,
        val_windows=16, train_episodes=2, val_episodes=4, seed=42,
        screening_updates=50, main_updates=200, visits_per_main_arm=3200,
        fixed_evaluation_every_updates=10,
        initialization='Fresh identical initialization for every LR; 1e-3 already completed50 and resumes its own full state, never another arm.',
        baseline_schedule='Linear warmup16 then constant; independent of max steps; all other LRs unchanged.',
        best='Within each run minimize fixed VAL Action+9*Future. Keep full latest/best.',
        comparison='At matched200 compare separate TRAIN/VAL MSE and full curves; require both TRAIN errors improve >1% and no VAL direction worsens >5% vs its baseline. Rank eligible arms by mean relative VAL MSE; report tradeoffs and no-winner explicitly.',
        early_stop='Nonfinite, TRAIN direction >10x initial after160 visits, USS>140GiB or8h per invocation; no automatic rerun of failed arm.',
        baseline_reference=str(REFERENCE/base['run_id']),
        manifest_sha256=hashlib.sha256((OUT/'train_val_manifest.json').read_bytes()).hexdigest(),
        storage='Full latest/best/pending plus concurrent main pending; wait rather than delete retained new arms.',
        followup='If an eligible lower/higher LR beats1e-3, prepare fresh three-task4096-visit confirmation on GPU23; small-pool result alone is not final optimum.'))
    return jobs


def summarize(jobs):
    from scripts.run_iteration_search import collect
    arms = []
    for job in jobs:
        cfg = yaml.safe_load((ROOT/job['config']).read_text())
        folder = Path(cfg['trainer']['output_dir'])
        points = collect(folder)
        row = dict(name=job['name'], rate=cfg['trainer']['energy_head_learning_rate'],
                   folder=str(folder), points=points)
        ini = folder/'initialization_step0.json'
        if ini.exists():
            row['initialization_sample_hash'] = read(ini)['sampled_parameter_sha256']
        speed = folder/'speed_report.json'
        row['completed_updates'] = read(speed)['completed_steps'] if speed.exists() else 0
        arms.append(row)
    hashes = [a['initialization_sample_hash'] for a in arms if 'initialization_sample_hash' in a]
    result = dict(arms=arms, initial_sample_hashes_equal=len(hashes)==4 and len(set(hashes))==1,
                  outcome='pending_matched200_review', generated_unix=time.time())
    eligible = []
    for arm in arms:
        lookup = {(p['update'],p['split'],p['direction']):p['mse'] for p in arm['points'] if p['k']==2}
        keys = [(s,d) for s in ['train','val'] for d in ['action','future']]
        if not all((u,s,d) in lookup for u in [0,200] for s,d in keys):
            continue
        ratios = {f'{s}_{d}':lookup[200,s,d]/lookup[0,s,d] for s,d in keys}
        arm['ratios_at200'] = ratios
        arm['eligible'] = (all(ratios[f'train_{d}'] < .99 for d in ['action','future'])
                           and all(ratios[f'val_{d}'] <= 1.05 for d in ['action','future']))
        if arm['eligible']:
            arm['mean_relative_val_mse'] = (ratios['val_action']+ratios['val_future'])/2
            eligible.append(arm)
    if eligible and result['initial_sample_hashes_equal']:
        winner = min(eligible, key=lambda a:a['mean_relative_val_mse'])
        result.update(outcome='small_pool_candidate_requires_three_task_confirmation',
                      candidate_rate=winner['rate'], candidate_name=winner['name'])
    elif all(a['completed_updates']>=200 for a in arms):
        result['outcome'] = 'no_eligible_winner'
    write(OUT/'comparison_report.json', result)
    return result


def confirm_on_three_tasks(result, jobs):
    if result.get('candidate_rate') in (None, 1e-3):
        return 'not_needed_or_no_candidate'
    # Require an actual improvement over the existing1e-3 arm, without hiding
    # a direction tradeoff behind a combined score.
    winner = next(a for a in result['arms'] if a['name']==result['candidate_name'])
    baseline = next(a for a in result['arms'] if a['rate']==1e-3)
    if 'ratios_at200' not in baseline:
        return 'baseline_incomplete_review_required'
    w, b = winner['ratios_at200'], baseline['ratios_at200']
    if ((w['val_action']+w['val_future']) >= .99*(b['val_action']+b['val_future'])
        or any(w[f'val_{d}'] > 1.05*b[f'val_{d}'] for d in ['action','future'])):
        return 'no_clear_validation_advantage'
    original = read('/cloud/zzc/energyWAM/experiments/three_task_full_20261106/jobs_42.json')[0]
    cfg = yaml.safe_load((ROOT/original['config']).read_text())
    name = f"three_task_confirm_head{result['candidate_rate']:g}_seed42"
    cfg['run_id'] = name
    cfg['trainer'].update(energy_head_learning_rate=result['candidate_rate'],
                          output_dir=str(OUT/name), max_train_steps=256)
    config = ROOT/f'configs/experiments/{name}.yaml'
    config.write_text(yaml.safe_dump(cfg,sort_keys=False))
    job = dict(original, name=name, config=str(config.relative_to(ROOT)), physical_gpus='2,3',
               steps=256, external_checkpoint_reserve_bytes=SNAPSHOT, max_wall_seconds=12*3600)
    write(OUT/'confirmation_jobs.json',[job])
    # Collector reloads only source metadata, leaving credentials untouched.
    bridge = ROOT/'configs/wandb_bridge.json'
    settings = read(bridge)
    if not any(s['config']==job['config'] for s in settings['sources']):
        settings['sources'].append(dict(config=job['config'],group='head-lr-three-task-confirmation',priority=120))
        write(bridge,settings)
    write(OUT/'controller_status.json',dict(phase='three_task_confirmation',candidate_rate=result['candidate_rate']))
    run_queue(OUT/'confirmation_jobs.json',OUT/'confirmation_status.json')
    from scripts.run_iteration_search import collect
    write(OUT/'confirmation_report.json',dict(candidate_rate=result['candidate_rate'],
          candidate=collect(Path(cfg['trainer']['output_dir'])),
          baseline=collect(Path(yaml.safe_load((ROOT/original['config']).read_text())['trainer']['output_dir'])),
          compare_at_update=256,interpretation='Compare matched fixed312-window VAL; no robot success-rate claim.'))
    return 'confirmation_terminal_review_report'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    signal.signal(signal.SIGTERM,stop)
    jobs = prepare()
    if args.prepare_only:
        return
    # A separately named queue per phase permits continuation without treating
    # a completed50 arm as a completed200 arm. Checkpoints are fully resumed.
    failed = set()
    for steps in [50, 200]:
        phase = []
        for job in jobs:
            if job['name'] in failed:
                continue
            cfg = yaml.safe_load((ROOT/job['config']).read_text())
            folder = Path(cfg['trainer']['output_dir'])
            completed = read(folder/'speed_report.json')['completed_steps'] if (folder/'speed_report.json').exists() else 0
            if completed >= steps:
                continue
            item = dict(job, steps=steps)
            phase.append(item)
        queue = OUT/f'jobs_{steps}.json'
        status = OUT/f'status_{steps}.json'
        write(queue, phase)
        write(OUT/'controller_status.json',dict(phase=f'running_to{steps}',updated_unix=time.time()))
        if phase:
            run_queue(queue,status)
            states = read(status)
            failed.update(n for n,s in states.items() if s['phase']=='failed')
        summarize(jobs)
    result = summarize(jobs)
    confirmation = confirm_on_three_tasks(result,jobs)
    write(OUT/'controller_status.json',dict(phase='finished_review_report',outcome=result['outcome'],
                                           confirmation=confirmation,failed_arms=sorted(failed),updated_unix=time.time()))


if __name__ == '__main__':
    main()

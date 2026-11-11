"""Bounded, serial head-only LR study; reuse the completed single-episode control."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import zipfile
from datetime import datetime
from zoneinfo import ZoneInfo

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_iteration_search import collect

OUT = Path('/cloud/zzc/energyWAM/experiments/single_episode_head_lr_20261109')
OLD = OUT.parent / 'single_episode_lr_20261108'
BASE = OLD / 'episode3346_lrscale1_b16_micro4_seed42'
SNAPSHOT = 24569214417
CHILD = None
STOP = False


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    pending = path.with_suffix('.pending.json')
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    pending.replace(path)


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def stop(*_):
    global STOP
    STOP = True
    if CHILD is not None and CHILD.poll() is None:
        CHILD.send_signal(signal.SIGTERM)


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    baseline_cfg = yaml.safe_load((ROOT / 'configs/experiments/episode3346_lrscale1_b16_micro4_seed42.yaml').read_text())
    baseline_definition = read(BASE / 'arm_definition.json')
    assert baseline_cfg == baseline_definition['hyperparameters']
    assert read(BASE / 'final_report.json')['global_step'] == 210
    for name in ['train_val_manifest.json', 'train_monitor_manifest.json']:
        source = (OLD / name).read_bytes()
        target = OUT / name
        if target.exists():
            assert target.read_bytes() == source
        else:
            target.write_bytes(source)
    jobs = []
    for scale, tag in [(0.5, '0p5'), (2.0, '2')]:
        name = f'episode3346_headscale{tag}_b16_micro4_seed42'
        folder = OUT / name
        folder.mkdir(exist_ok=True)
        cfg = copy.deepcopy(baseline_cfg)
        cfg['run_id'] = name
        cfg['notes'] = 'Head-only LR comparison against single_episode_lr_20261108 scale1. Fresh identical seed42/pretrained initialization; only head LR changed. Same67TRAIN/16auxVAL, B16/native4/K2, lambdaFuture9, outerclip1. No robot checkpoint resume.'
        cfg['trainer']['output_dir'] = str(folder)
        cfg['trainer']['energy_head_learning_rate'] *= scale
        # This assertion prevents accidental changes to other experimental factors.
        restored = copy.deepcopy(cfg)
        for key in ['run_id', 'notes']:
            restored[key] = baseline_cfg[key]
        for key in ['output_dir', 'energy_head_learning_rate']:
            restored['trainer'][key] = baseline_cfg['trainer'][key]
        assert restored == baseline_cfg
        config = ROOT / f'configs/experiments/{name}.yaml'
        if config.exists():
            assert yaml.safe_load(config.read_text()) == cfg
        else:
            config.write_text(yaml.safe_dump(cfg, sort_keys=False))
        job = copy.deepcopy(baseline_definition['job'])
        job.update(name=name, config=str(config.relative_to(ROOT)), physical_gpus='2,3',
                   manifest=str(OUT / 'train_val_manifest.json'),
                   train_monitor_manifest=str(OUT / 'train_monitor_manifest.json'))
        job.pop('initial_checkpoint', None)
        assert job['steps'] == 210 and job['keep_best'] and job['effective_batch'] == 16
        definition = dict(head_scale=scale, job=job, hyperparameters=cfg,
                          initialization='Fresh pretrained Qwen plus seed42 new modules, same as control; not resumed from trained control.',
                          manifest_sha256=hashlib.sha256((OUT / 'train_val_manifest.json').read_bytes()).hexdigest(),
                          baseline_folder=str(BASE))
        write(folder / 'arm_definition.json', definition)
        archive = folder / 'study_source.zip'
        if not archive.exists():
            with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as z:
                for p in list((ROOT / 'energywam').rglob('*.py')) + [ROOT / 'scripts' / n for n in ['train_shared_sharded.py', 'k_training_probe.py', 'run_iteration_search.py', Path(__file__).name]]:
                    z.write(p, str(p.relative_to(ROOT)))
                z.write(config, 'experiment_config.yaml')
        jobs.append(job)
    write(OUT / 'jobs.json', jobs)
    plan = dict(created_at=now(), status='prepared', authorization='User2026-11-09: 好的你启动吧; two new head-only arms, reuse existing head1e-3 control.',
                independent_variable='energy_head_learning_rate only', head_rates=[0.0005, 0.001, 0.002],
                new_arms=[0.5, 2.0], existing_control=str(BASE), steps_per_new_arm=210,
                train_windows=67, auxiliary_val_windows=16, equivalent_epochs_per_arm=210*16/67,
                selection='Compare all67 fixed TRAIN Action/Future separately at210updates against own step0/control; report stability and auxiliary VAL separately; no winner if directions trade off.',
                best_checkpoint='Predeclared minimum auxiliary16VAL Action MSE+9*Future MSE, full-resume latest/best; distinct from TRAIN fit ranking.',
                physical_gpus='2,3', execution='serial queue, no automatic further hyperparameter trials',
                retention='Keep full latest/best for both new arms; no automatic weight retirement in this study.',
                space_budget=dict(snapshot_bytes=SNAPSHOT, new_retained_snapshots=4, pending_snapshots=1, safety_bytes=8*2**30, required_bytes=5*SNAPSHOT+8*2**30),
                baseline_metrics=read(OLD/'comparison_report.json')['arms'][1]['train_mse'])
    write(OUT / 'plan.json', plan)
    bridge = ROOT / 'configs/wandb_bridge.json'
    settings = read(bridge)
    for job in jobs:
        if not any(s['config'] == job['config'] for s in settings['sources']):
            settings['sources'].append(dict(config=job['config'], group='single-episode3346-head-lr-20261109', priority=210))
    write(bridge, settings)
    bridge.chmod(0o600)
    return jobs


def summarize(jobs):
    statuses = read(OUT/'queue_status.json') if (OUT/'queue_status.json').exists() else {}
    arms = []
    for scale, folder, name in [(1., BASE, 'existing_control')] + [(read(OUT/j['name']/'arm_definition.json')['head_scale'], OUT/j['name'], j['name']) for j in jobs]:
        points = collect(folder)
        logs = read(folder/'speed_report.json') if (folder/'speed_report.json').exists() else {}
        initial = read(folder/'initialization_step0.json') if (folder/'initialization_step0.json').exists() else {}
        arms.append(dict(name=name, head_scale=scale, folder=str(folder),
                         completed_updates=logs.get('completed_steps', 0), points=points,
                         initialization_sample_hash=initial.get('sampled_parameter_sha256'),
                         queue_status=statuses.get(name, dict(phase='existing_complete' if name=='existing_control' else 'queued'))))
    reference = arms[0]['initialization_sample_hash']
    for a in arms[1:]:
        if a['initialization_sample_hash'] and a['initialization_sample_hash'] != reference:
            raise RuntimeError('Initialization sampled hash differs from control; stop unfair comparison')
    result = dict(updated_at=now(), arms=arms, criterion=read(OUT/'plan.json')['selection'],
                  initialization_check='sampled parameter hash only, not a full parameter equivalence proof')
    write(OUT/'comparison_report.json', result)
    return result


def main():
    global CHILD
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    jobs = prepare()
    if args.prepare_only:
        summarize(jobs)
        return
    # Existing queue provides GPU-pair locking, full-state retention, per-arm
    # disk checks, numerical/resource guards and no duplicate failed-job retry.
    command = [sys.executable, '-u', str(ROOT/'scripts/run_iteration_search.py'),
               '--jobs', str(OUT/'jobs.json'), '--status', str(OUT/'queue_status.json')]
    write(OUT/'controller_status.json', dict(phase='starting', started_at=now(), command=command))
    CHILD = subprocess.Popen(command, cwd=ROOT)
    while CHILD.poll() is None and not STOP:
        summarize(jobs)
        time.sleep(20)
    if STOP:
        CHILD.wait(timeout=100)
    result = summarize(jobs)
    states = [a['queue_status'].get('phase') for a in result['arms'][1:]]
    phase = 'interrupted' if STOP else ('complete' if all(x in ['complete','failed'] for x in states) and CHILD.returncode == 0 else 'failed')
    write(OUT/'controller_status.json', dict(phase=phase, ended_at=now(), arm_phases=states,
                                           returncode=CHILD.returncode, result=str(OUT/'comparison_report.json')))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        stop()
        if CHILD is not None:
            CHILD.wait(timeout=100)
        OUT.mkdir(parents=True, exist_ok=True)
        write(OUT/'controller_status.json', dict(phase='failed', reason=repr(error), at=now()))
        raise

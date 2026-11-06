#!/usr/bin/env python
"""Supervised, resumable bi -> held-out validation -> joint training on GPUs 2,3.

No algorithm fallback on failure. --steps remains a total stage target on resume.
Run through the checked-in supervisor template, or with CUDA_VISIBLE_DEVICES=2,3.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
STATUS = ROOT / 'logs/gpu23_pipeline_status.json'


def status(phase, **extra):
    record = {'phase': phase, 'updated_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **extra}
    tmp = STATUS.with_suffix('.tmp')
    tmp.write_text(json.dumps(record, indent=2))
    tmp.replace(STATUS)
    print(json.dumps(record), flush=True)


def wait_for(program, artifact, timeout=7200):
    start = time.monotonic()
    while True:
        state = subprocess.run(['supervisorctl', 'status', program], capture_output=True, text=True).stdout
        if artifact.exists() and 'EXITED' in state:
            return
        if any(s in state for s in ['FATAL', 'BACKOFF', 'STOPPED', 'ERROR']) or ('EXITED' in state and not artifact.exists()):
            raise RuntimeError(f'{program} failed before producing {artifact}: {state.strip()}')
        if time.monotonic() - start > timeout:
            raise TimeoutError(f'{program} did not complete within {timeout}s')
        time.sleep(5)


def run(args, logfile):
    print(f'Running: {args} -> {logfile}', flush=True)
    with (ROOT / logfile).open('a') as stream:
        subprocess.run([sys.executable, '-u', *args], cwd=ROOT, env=os.environ.copy(), stdout=stream, stderr=subprocess.STDOUT, check=True)


def train_stage(stage, init_from=None):
    run_id = f'energywam_gpu23_{stage}_20261101'
    cfg = f'configs/gpu23_{stage}_20261101.yaml'
    folder = ROOT / 'logs/runs' / run_id
    import yaml
    target = int(yaml.safe_load((ROOT / cfg).read_text())['trainer']['max_train_steps'])
    final = folder / f'{run_id}_step{target}.pt'
    report = folder / 'final_report.json'
    if report.exists() and final.exists() and json.loads(report.read_text()).get('global_step') == target:
        return final
    args = ['scripts/train.py', '--config', cfg]
    latest = folder / 'ckpt_latest.pt'
    if latest.exists():
        args += ['--resume', str(latest)]
    elif init_from:
        args += ['--init-from', str(init_from), '--stage-switch-policy', 'reset']
    status(f'training_{stage}', target_steps=target, config=cfg, log=f'logs/train_gpu23_{stage}_20261101.log')
    run(args, f'logs/train_gpu23_{stage}_20261101.log')
    if not final.exists():
        raise RuntimeError(f'missing expected final checkpoint {final}')
    return final


def evaluate(stage, checkpoint):
    output = f'logs/runs/energywam_gpu23_{stage}_20261101/val_offline_32.json'
    status(f'validating_{stage}', checkpoint=str(checkpoint), output=output)
    run(['scripts/evaluate.py', '--config', f'configs/gpu23_{stage}_20261101.yaml', '--checkpoint', str(checkpoint), '--eval-episodes-manifest', 'manifests/val_episodes.json', '--mode', 'joint', '--max-samples', '32', '--out', output], f'logs/eval_gpu23_{stage}_20261101.log')


def main():
    os.chdir(ROOT)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '2,3':
        raise RuntimeError('this launcher requires CUDA_VISIBLE_DEVICES=2,3')
    status('waiting_for_4b_validation')
    report = ROOT / 'logs/runs/energywam_gpu23_validation_20261101/final_report.json'
    wait_for('energywam-validation', report)
    result = json.loads(report.read_text())
    if result.get('global_step') != 2 or result['last_loss']['param_grad_nonfinite'] != 0:
        raise RuntimeError('4B validation did not complete two finite-gradient steps')
    status('waiting_for_train_statistics')
    bundle_path = ROOT / 'stats/trainmix_150k_paper_standardized_v1.json'
    wait_for('energywam-stats', bundle_path)
    from energywam.data.standardization import load_bundle
    bundle = load_bundle(bundle_path)
    coverage = bundle.provenance['coverage']
    if coverage['num_windows'] != 150000 or coverage['kind'] != 'subset':
        raise RuntimeError(f'unexpected statistics coverage {coverage}')
    status('prerequisites_passed', stats_fingerprint=bundle.fingerprint, coverage=coverage)
    checkpoint = train_stage('bi')
    evaluate('bi', checkpoint)
    checkpoint = train_stage('joint', init_from=checkpoint)
    evaluate('joint', checkpoint)
    status('complete', checkpoint=str(checkpoint), note='offline validation only; closed-loop success rate remains to be evaluated')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        status('failed', error=str(exc))
        raise

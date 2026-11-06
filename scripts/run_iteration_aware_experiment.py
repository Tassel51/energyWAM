#!/usr/bin/env python
"""Prepare/inspect a bounded experiment. Launch only with explicit --run.

Does not stop existing jobs, register a supervisor job, or launch Stage 2.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = Path('/cloud/zzc/energyWAM/experiments/overfit_k24_alpha200_20261103/sample_manifest.json')


def selected_gpus_idle(pair):
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'],text=True)
    selected = {uuid.strip() for row in raw.splitlines() for index,uuid in [row.split(',')] if index.strip() in pair.split(',')}
    apps = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    if len(selected)!=2 or any(row.split(',')[0].strip() in selected for row in apps.splitlines()):
        raise RuntimeError('所选 GPU 有进程或不可用；不停止、不挤占现有任务。')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant',choices=['noiter_fixed','iter_fixed','iter_learned'],default='iter_learned')
    p.add_argument('--physical-gpus',choices=['0,1','2,3'],default='2,3')
    p.add_argument('--steps',type=int,default=20,help='absolute target updates; 20 preflight then resume to 200')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--run',action='store_true',help='without this flag only print the plan')
    a=p.parse_args()
    if not 1<=a.steps<=400:p.error('此预实验限 1–400 次更新')
    sys.path.insert(0,str(ROOT))
    from energywam.config import load_config,to_plain_dict
    config=ROOT/f'configs/experiments/iteration_v1_{a.variant}.yaml'
    cfg=load_config(config);out=Path(cfg.trainer.output_dir)
    manifest=json.loads(MANIFEST.read_text())
    train=[w for w in manifest['windows'] if w['split']=='train']
    if len(train)!=80 or len({w['episode_index'] for w in train})!=10:
        raise RuntimeError('expected frozen 80-window /10-episode protocol')
    checkpoint=out/'ckpt_latest.pt'
    if a.resume and not checkpoint.exists():raise FileNotFoundError(checkpoint)
    if not a.resume and out.exists() and any(out.iterdir()):
        raise RuntimeError('输出目录已有结果，请使用 --resume；不同实验必须另用目录')
    command=[sys.executable,'-u',str(ROOT/'scripts/train_shared_sharded.py'),'--config',str(config),
        '--steps',str(a.steps),'--physical-gpus',a.physical_gpus,
        '--fixed-window-manifest',str(out/'sample_manifest.json'),'--probe-eval-interval','20']
    if a.resume:command+=['--resume',str(checkpoint)]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.physical_gpus,ENERGYWAM_SAMPLE_ALIGNED_RNG='1',
        ENERGYWAM_TRIM_BETWEEN_DIRECTIONS='0',ENERGYWAM_ONLY_DIRECTION='',
        ENERGYWAM_PROBE_MAX_K='2',ENERGYWAM_PROBE_SPLITS='train',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    print(json.dumps(dict(status='will_run' if a.run else 'dry_run_only',variant=a.variant,
        physical_gpus=a.physical_gpus,train_samples=80,episodes=10,global_batch=2,microbatch=1,
        target_updates=a.steps,target_sample_visits=a.steps*2,train_K=2,
        alpha_initial=[cfg.solver.alpha_action,cfg.solver.alpha_future],learned_steps=cfg.solver.learnable_step_sizes,
        output=str(out),command=shlex.join(command)),indent=2,ensure_ascii=False))
    if not a.run:return
    selected_gpus_idle(a.physical_gpus)
    snapshot=MANIFEST.read_bytes()
    if a.resume and (out/'sample_manifest.json').read_bytes()!=snapshot:
        raise RuntimeError('恢复时数据清单不一致')
    out.mkdir(parents=True,exist_ok=True)
    if not a.resume:
        (out/'sample_manifest.json').write_bytes(snapshot)
        # Preserve source/config provenance alongside each run.
        source_hashes={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest()
            for folder in ['energywam','scripts'] for f in (ROOT/folder).rglob('*.py')}
        (out/'experiment_provenance.json').write_text(json.dumps(dict(
            config=to_plain_dict(cfg),manifest_sha256=hashlib.sha256(snapshot).hexdigest(),source_sha256=source_hashes),indent=2))
    else:
        saved=json.loads((out/'experiment_provenance.json').read_text())['config']
        if saved!=to_plain_dict(cfg):raise RuntimeError('恢复配置已改变；请使用原配置或建立新实验')
    with (out/'console.log').open('a') as log:
        subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)


if __name__=='__main__':main()

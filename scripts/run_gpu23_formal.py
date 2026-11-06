#!/usr/bin/env python
"""Automatically resumable two-stage run; one rolling checkpoint across stages."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import yaml
ROOT=Path(__file__).resolve().parents[1]
STATUS=ROOT/'logs/gpu23_formal_status.json'

def status(phase,**kw):
    record=dict(phase=phase,updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw)
    tmp=STATUS.with_suffix('.tmp');tmp.write_text(json.dumps(record,indent=2));tmp.replace(STATUS)
    print(json.dumps(record),flush=True)

def retire_superseded():
    bi=ROOT/'logs/runs/energywam_formal_bi_20261101/ckpt_latest.pt'
    joint=ROOT/'logs/runs/energywam_formal_joint_20261101/ckpt_latest.pt'
    old=[]
    if bi.exists() or joint.exists():
        old.extend([ROOT/'logs/runs/energywam_gpu23_bi_20261101/energywam_gpu23_bi_20261101_step200.pt',
                    ROOT/'logs/runs/energywam_gpu23_joint_20261101/energywam_gpu23_joint_20261101_step100.pt'])
    if joint.exists():old.append(bi)
    for path in old:
        if path.exists():
            path.unlink();Path(str(path)+'.meta.json').unlink(missing_ok=True)
            print(f'Removed superseded checkpoint: {path}',flush=True)


def run(args,log):
    with (ROOT/log).open('a') as f:
        process=subprocess.Popen([sys.executable,'-u',*args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        while process.poll() is None:
            retire_superseded()
            time.sleep(5)
        retire_superseded()
        if process.returncode:raise subprocess.CalledProcessError(process.returncode,args)

def main():
    os.chdir(ROOT)
    # Wait for the user-mounted storage; never silently fall back to the local disk.
    while not os.path.ismount('/cloud'):
        status('waiting_for_cloud_mount')
        time.sleep(5)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='2,3':raise RuntimeError('requires physical GPUs 2,3')
    previous=None
    # Joint is deliberately disabled pending BI diagnostic experiments.
    for stage in ['bi']:
        if stage == 'bi' and Path('logs/runs/energywam_formal_joint_20261101/ckpt_latest.pt').exists():
            continue
        cfgpath=f'configs/gpu23_formal_{stage}_20261101.yaml'
        cfg=yaml.safe_load(Path(cfgpath).read_text());folder=Path(cfg['trainer']['output_dir']);folder.mkdir(parents=True,exist_ok=True)
        checkpoint=folder/'ckpt_latest.pt';target=cfg['trainer']['max_train_steps']
        # Chunk-end reports are metadata; a complete checkpoint is the source of resume state.
        for end in range(1000,target+1,1000):
            validation=folder/f'val_step{end}.json'
            if validation.exists() and checkpoint.exists():continue
            args=['scripts/train.py','--config',cfgpath,'--steps',str(end)]
            if checkpoint.exists():args+=['--resume',str(checkpoint)]
            elif previous:args+=['--init-from',str(previous),'--stage-switch-policy','reset']
            status('training',stage=stage,chunk_target=end,stage_target=target,checkpoint=str(checkpoint))
            run(args,f'logs/train_formal_{stage}.log')
            if not checkpoint.exists():raise RuntimeError('training did not save rolling checkpoint')
            # The first successfully saved Joint checkpoint replaces the BI checkpoint.
            if previous and previous!=checkpoint:
                previous.unlink(missing_ok=True);Path(str(previous)+'.meta.json').unlink(missing_ok=True)
                previous=None
            status('validating',stage=stage,step=end)
            run(['scripts/evaluate.py','--config',cfgpath,'--checkpoint',str(checkpoint),
                 '--eval-episodes-manifest','manifests/val_episodes.json','--one-per-task',
                 '--mode','joint','--out',str(validation)],f'logs/eval_formal_{stage}.log')
        previous=checkpoint
    status('bi_complete_joint_disabled',checkpoint=str(previous),note='Stage 2 disabled pending BI diagnostics; offline metrics are not success rate.')

if __name__=='__main__':
    try:main()
    except Exception as exc:
        status('failed',error=str(exc));raise

#!/usr/bin/env python
"""Protect the stopped BI checkpoint, diagnose once, and upload small reports."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
OUT=Path('/cloud/zzc/energyWAM/experiments/inner_diag_20261102')


def status(phase,**extra):
    record=dict(phase=phase,updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**extra)
    tmp=OUT/'status.tmp';tmp.write_text(json.dumps(record,indent=2));tmp.replace(OUT/'status.json')
    print(json.dumps(record),flush=True)


def main():
    os.chdir(ROOT);OUT.mkdir(parents=True,exist_ok=True)
    state=subprocess.run(['supervisorctl','status','energywam-gpu23'],capture_output=True,text=True).stdout
    if not any(word in state for word in ['STOPPED','EXITED']):raise RuntimeError('Old training is still active; refusing GPU conflict')
    reference=OUT/'reference.json'
    if not reference.exists():
        import torch,gc
        status('verifying_handoff_checkpoint')
        original=ROOT/'logs/runs/energywam_formal_bi_20261101/ckpt_latest.pt'
        original_meta=Path(str(original)+'.meta.json')
        payload=torch.load(original,map_location='cpu',weights_only=False)
        step=payload['meta']['global_step']
        assert step==2300 and payload['meta']['stage']=='bi'
        assert payload['model'] and payload['optimizer'] and payload['scheduler']
        assert len(payload['distributed_rng_states'])==2
        fingerprint=payload['meta']['normalization_bundle']['fingerprint'] if 'fingerprint' in payload['meta']['normalization_bundle'] else None
        del payload;gc.collect()
        target=OUT/'reference'/f'bi_step{step}.pt';target.parent.mkdir(exist_ok=True)
        actual=original.resolve();meta_actual=original_meta.resolve()
        if actual!=target:
            if target.exists():raise RuntimeError('Reference destination already exists unexpectedly')
            actual.rename(target);actual.symlink_to(target)
            target_meta=Path(str(target)+'.meta.json');meta_actual.rename(target_meta);meta_actual.symlink_to(target_meta)
        with target.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
        record={'checkpoint_step':step,'checkpoint':str(target),'sha256':digest,
                'stage':'bi','normalization_fingerprint':fingerprint,
                'verified_model_optimizer_scheduler_and_two_rank_rng':True,
                'training_log':str((original.parent/'train_log.jsonl').resolve())}
        reference.write_text(json.dumps(record,indent=2))
        shutil.copy2(ROOT/'configs/gpu23_formal_bi_20261101.yaml',OUT/'original_bi_config.yaml')
        shutil.copy2(original.parent/'resolved_config.json',OUT/'original_resolved_config.json')
        shutil.copy2('/etc/supervisor/conf.d/energywam-gpu23.conf',OUT/'paused_training_supervisor.conf')
    record=json.load(open(reference))
    if not(OUT/'report.json').exists():
        status('diagnosing',checkpoint_step=record['checkpoint_step'])
        subprocess.run([sys.executable,'-u','scripts/run_inner_diagnostic.py','--experiment',args.experiment,'--checkpoint',record['checkpoint'],'--output',str(OUT)],check=True)
    if not(OUT/'wandb.json').exists():
        status('uploading_reports')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',HTTP_PROXY='http://127.0.0.1:7901',HTTPS_PROXY='http://127.0.0.1:7901',http_proxy='http://127.0.0.1:7901',https_proxy='http://127.0.0.1:7901')
        subprocess.run([str(ROOT/'.venv-wandb/bin/python'),'-u','scripts/upload_inner_diagnostic.py','--output',str(OUT)],env=env,check=True)
    status('complete',checkpoint_step=record['checkpoint_step'],wandb=json.load(open(OUT/'wandb.json')))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default=str(OUT))
    parser.add_argument('--experiment',default='configs/experiments/inner_diag_20261102.json')
    args=parser.parse_args();OUT=Path(args.output)
    try:main()
    except Exception as exc:
        status('failed',error=str(exc));raise

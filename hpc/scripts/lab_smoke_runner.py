#!/usr/bin/env python3
"""Bounded real4B one-update/save then resume-to-two validation, lab only."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time

CHILD = None


def stop(*_):
    if CHILD is not None and CHILD.poll() is None:
        os.killpg(CHILD.pid, signal.SIGTERM)
    raise KeyboardInterrupt


def write(path, value):
    p=Path(path);temp=p.with_suffix('.pending.json')
    temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(p)


def main():
    global CHILD
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True)
    plan=parser.parse_args().plan;cfg=json.loads(plan.read_text())
    assert socket.gethostname()==cfg['hostname']
    out=Path(cfg['output']);out.mkdir(exist_ok=False)
    shutil.copy2(plan,out/'launch_plan.json');shutil.copy2(__file__,out/'runner_source.py')
    state=dict(phase='preflight',hostname=socket.gethostname(),owner_uid=os.getuid(),runner_pid=os.getpid(),started_at=datetime.now(timezone.utc).isoformat(),commands=[])
    write(out/'run_status.json',state)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    pair=cfg.get('physical_gpus','6,7')
    lock=open(out.parent/('gpu'+pair.replace(',','')+'_validation.lock'),'a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        assert shutil.disk_usage(out).free>3*24569214417+8*2**30,'Insufficient full-resume snapshot budget'
        rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
        policy=json.loads(Path(cfg['gpu_policy']).read_text())
        shared=set(policy.get('authorized_sharing',{}).get('indices',[]))
        exclusive={u for i,u in zip(pair.split(','),cfg['gpu_uuids']) if i not in shared}
        assert not any(line.split(',')[0].strip() in exclusive for line in rows.splitlines()),'Exclusive GPU no longer idle'
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(cfg['gpu_uuids']),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',ENERGYWAM_TRIM_BETWEEN_DIRECTIONS='0',ENERGYWAM_SAMPLE_ALIGNED_RNG='1',ENERGYWAM_FIXED_DATA_CACHE='1',ENERGYWAM_PROBE_MAX_K='2',ENERGYWAM_PROBE_SPLITS='train,val',PYTHONPATH=cfg['code'],PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
        if cfg.get('probe_ks'):
            env['ENERGYWAM_PROBE_KS']=','.join(map(str,cfg['probe_ks']))
            env['ENERGYWAM_PROBE_MAX_K']=str(max(cfg['probe_ks']))
        sys.path.insert(0,cfg['code'])
        from energywam.training.resource_monitor import ResourceMonitor
        for target in [1,2]:
            command=[cfg['python'],'-u',str(Path(cfg['code'])/'scripts/train_shared_sharded.py'),'--config',cfg['config'],'--steps',str(target),'--physical-gpus',pair,'--execution-gpu-policy',cfg['gpu_policy'],'--fixed-window-manifest',cfg['manifest'],'--probe-eval-interval','1','--keep-best']
            if cfg.get('train_monitor_manifest'):
                command+=['--train-monitor-manifest',cfg['train_monitor_manifest'],'--train-monitor-sample-interval','320']
            if target==2:command+=['--resume',str(out/'ckpt_latest.pt')]
            state.update(phase='running',target_update=target);state['commands'].append(command)
            with (out/'console.log').open('a') as log:
                CHILD=subprocess.Popen(command,cwd=cfg['code'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                state['child_pid']=CHILD.pid;write(out/'run_status.json',state)
                resource_dir=out/f'resources_stage{target}';resource_dir.mkdir()
                monitor=ResourceMonitor(CHILD.pid,pair.split(','),resource_dir);monitor.start();begin=time.monotonic()
                reason=None
                while CHILD.poll() is None:
                    if time.monotonic()-begin>cfg.get('stage_timeout_seconds',1200):reason='Bounded smoke stage exceeded1200seconds'
                    if monitor.summary.get('last_sample',{}).get('process_uss_gib',0)>100:reason='Smoke process USS exceeded100GiB'
                    if reason:
                        os.killpg(CHILD.pid,signal.SIGTERM);CHILD.wait(timeout=60);break
                    time.sleep(5)
                monitor.finish(CHILD.returncode,reason)
                if CHILD.returncode or reason:raise RuntimeError(reason or f'Stage{target} exited{CHILD.returncode}')
            assert json.loads((out/'final_report.json').read_text())['global_step']==target
        import torch
        checkpoints={}
        for role in ['latest','best']:
            p=out/f'ckpt_{role}.pt';payload=torch.load(p,map_location='cpu',mmap=True,weights_only=False)
            required=['model','optimizer','scheduler','rng','meta']
            assert all(k in payload for k in required),(role,list(payload))
            checkpoints[role]=dict(path=str(p),bytes=p.stat().st_size,step=payload['meta']['global_step'],full_state_keys=required)
            del payload
        assert checkpoints['latest']['step']==2
        write(out/'resume_validation_result.json',dict(passed=True,total_optimizer_updates=2,actual_resume_from_step=1,hostname=socket.gethostname(),gpu_uuids=cfg['gpu_uuids'],checkpoints=checkpoints,scope=cfg.get('scope','B1/micro1/K2 engineering validation only; not a quality comparison or formal experiment')))
        state.update(phase='complete',ended_at=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        if CHILD is not None and CHILD.poll() is None:
            os.killpg(CHILD.pid,signal.SIGTERM);CHILD.wait(timeout=60)
        state.update(phase='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',reason=repr(error),ended_at=datetime.now(timezone.utc).isoformat())
        write(out/'run_status.json',state)
        raise
    write(out/'run_status.json',state)


if __name__=='__main__':main()

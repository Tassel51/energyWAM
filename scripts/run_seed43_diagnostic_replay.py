"""Bounded four-update full-state replay; source checkpoint never modified."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def write(path,value):
    path=Path(path);temp=path.with_suffix('.pending.json')
    temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def worker(plan):
    from scripts.diagnostic_replay_observer import ReplayObserver
    from scripts.audit_one_update_resume import compare_state
    from energywam.training.shared_microbatch import SharedMicrobatchTrainer
    from energywam.training.checkpoint import _rng_state
    import torch
    import scripts.train_shared_sharded as entry
    output=Path(plan['output']);old_load=SharedMicrobatchTrainer.load;old_step=SharedMicrobatchTrainer.train_step
    original={int(v['global_step'])+1:v for line in (Path(plan['source_run'])/'train_log.jsonl').read_text().splitlines()
              if 'global_step' in (v:=json.loads(line)) and not v.get('event')}
    observer=None;comparisons=[]

    def load(trainer,*args,**kwargs):
        result=old_load(trainer,*args,**kwargs)
        payload=torch.load(plan['checkpoint'],map_location='cpu',mmap=True,weights_only=False)
        checks={}
        for name,actual,expected,full in [('model',trainer.model.state_dict(),payload['model'],False),
              ('optimizer',trainer.optimizer.state_dict(),payload['optimizer'],False),
              ('scheduler',trainer.scheduler.state_dict(),payload['scheduler'],True),('rng',_rng_state(),payload['rng'],True)]:
            stats=dict(tensors=0,values_compared=0);compare_state(actual,expected,stats,full);checks[name]=stats
        assert trainer.global_step==61
        write(output/'restored_state_check.json',dict(passed=True,step=61,checks=checks,
             scope='All keys/shapes/dtypes; tensor first/last64 values; full scheduler/RNG. Not a full tensor-content hash.'))
        return result

    def step(trainer,batch):
        nonlocal observer
        if observer is None:observer=ReplayObserver(trainer,output/'diagnostics').install()
        row=old_step(trainer,batch);number=trainer.global_step
        keys=['bi/action_action_endpoint_mse','bi/future_future_endpoint_mse','param_grad_norm','diag/alpha_action_after_update','diag/alpha_future_after_update']
        comp={k:dict(original=original[number][k],replay=row[k],exact_equal=original[number][k]==row[k]) for k in keys}
        comparisons.append(dict(update=number,metrics=comp));write(output/'replay_comparisons.json',comparisons)
        print('[diagnostic-replay] '+json.dumps(dict(update=number,exact_equal=all(x['exact_equal'] for x in comp.values()))),flush=True)
        if not all(x['exact_equal'] for x in comp.values()):
            raise RuntimeError('Observed replay diverged from original: stop and inspect instead of attributing altered trajectory')
        return row

    SharedMicrobatchTrainer.load=load;SharedMicrobatchTrainer.train_step=step
    sys.argv=['train_shared_sharded.py','--config',plan['config'],'--steps','65','--resume',plan['checkpoint'],
              '--physical-gpus','2,3','--fixed-window-manifest',str(output/'sample_manifest.json'),'--profile']
    try:entry.main()
    finally:
        if observer is not None:observer.remove()
    write(output/'replay_result.json',dict(passed=True,replayed_updates=[62,63,64,65],all_logged_metrics_exact=True,
          full_state_resume_verified=True,source_checkpoint_untouched=True,new_full_snapshots_written=False,
          scope='Four optimizer updates for diagnosis; not a new210-update training arm. Initial full61 checkpoint retained.'))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--worker',action='store_true');args=parser.parse_args()
    plan=json.loads(args.plan.read_text());output=Path(plan['output'])
    if args.worker:
        worker(plan);return
    import fcntl
    from scripts.run_iteration_search import gpu_idle
    from energywam.training.resource_monitor import ResourceMonitor
    lock=(ROOT/'logs/gpu_pair_2_3.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert gpu_idle('2,3'),'GPU2/3 busy; do not touch other tasks'
    statepath=output/'audit_status.json';assert not statepath.exists(),'Do not duplicate a prior audit'
    source=Path(plan['checkpoint']);before=source.stat();identity=(before.st_ino,before.st_size,before.st_mtime_ns)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='2,3',ENERGYWAM_TRIM_BETWEEN_DIRECTIONS='0',ENERGYWAM_SAMPLE_ALIGNED_RNG='1',
             ENERGYWAM_FIXED_DATA_CACHE='1',ENERGYWAM_ONLY_DIRECTION='',ENERGYWAM_PROBE_MAX_K='2',ENERGYWAM_PROBE_SPLITS='train,val',
             OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',BLIS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
    cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--plan',str(args.plan),'--worker']
    with (output/'console.log').open('a') as stream:
        child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        monitor=ResourceMonitor(child.pid,['2','3'],output);monitor.start()
        state=dict(phase='running',pid=child.pid,command=cmd,started_at=datetime.now(timezone.utc).isoformat(),source_step=61,target_step=65)
        write(statepath,state);started=time.time();reason=None
        def stop(*_):
            nonlocal reason
            reason='User interrupted diagnostic supervisor'
        signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
        while child.poll() is None:
            if time.time()-started>2700:reason='45minute diagnostic time budget exceeded'
            if monitor.summary.get('last_sample',{}).get('process_uss_gib',0)>180:reason='180GiB diagnostic USS limit exceeded'
            if reason:os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=60);break
            time.sleep(3)
        monitor.finish(child.returncode,reason)
    after=source.stat();unchanged=identity==(after.st_ino,after.st_size,after.st_mtime_ns)
    state.update(phase='complete' if child.returncode==0 and unchanged else 'failed',returncode=child.returncode,reason=reason,
                 source_checkpoint_identity_unchanged=unchanged,ended_at=datetime.now(timezone.utc).isoformat())
    write(statepath,state)


if __name__=='__main__':main()

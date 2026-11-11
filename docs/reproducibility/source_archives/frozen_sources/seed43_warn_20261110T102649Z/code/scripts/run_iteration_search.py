#!/usr/bin/env python
"""Durable bounded experiment queue; records results and never interrupts other jobs."""
import json,os,subprocess,sys,time,statistics,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=Path('/cloud/zzc/energyWAM/experiments/iteration_aware_v1_20261104')
PYTHON=sys.executable  # Use the caller's environment on other machines.
JOBS=OUT/'search_jobs.json'
STOP=False

def write(path,data):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2));temp.replace(path)

def gpu_idle(pair):
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
    wanted={uuid.strip() for row in rows.splitlines() for index,uuid in [row.split(',')] if index.strip() in pair.split(',')}
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    return not any(row.split(',')[0].strip() in wanted for row in apps.splitlines())

def checkpoint_space_required(folder,snapshot_bytes,*,keep_best=False,profile=False):
    """Additional bytes for retained roles plus one serialized pending write."""
    if profile:return 2*2**30
    if type(snapshot_bytes) is not int or snapshot_bytes<=0:
        raise ValueError('positive measured/estimated snapshot size required')
    paths=[Path(folder)/'ckpt_latest.pt']
    if keep_best:paths.append(Path(folder)/'ckpt_best.pt')
    retained={}
    for path in paths:
        if path.exists():
            value=path.stat();retained[(value.st_dev,value.st_ino)]=value.st_size
    size=max([snapshot_bytes,*retained.values()])
    roles=2 if keep_best else 1
    return max(0,roles*size-sum(retained.values()))+size+4*2**30


def collect(folder):
    points={}
    for file in sorted(list(folder.glob('eval_*.json'))+list((folder/'train_monitor').glob('eval_*.json'))):
        try:
            report=json.loads(file.read_text())
            for row in report['rows']:
                if row['k'] not in [0,2]:continue
                key=(row['update'],row['split'],row['direction'],row['k'])
                points.setdefault(key,[]).append(row['prediction_mse'])
        except (OSError,ValueError,KeyError):continue
    return [dict(update=u,split=s,direction=d,k=k,mse=statistics.mean(v),samples=len(v)) for (u,s,d,k),v in sorted(points.items())]


def fixed_mse_failure(points, ratio=10., min_update=20, *, effective_batch=None, min_sample_visits=None):
    """Reject a grossly worse fixed training prediction; never use test data."""
    if min_sample_visits is not None:
        if not effective_batch or effective_batch<1:raise ValueError('effective batch required for sample-based guard')
        import math
        min_update=math.ceil(min_sample_visits/effective_batch)
    for direction in ('action','future'):
        rows=[p for p in points if p['split']=='train' and p['direction']==direction and p['k']==2]
        initial=next((p for p in rows if p['update']==0),None)
        later=[p for p in rows if p['update']>=min_update]
        if initial is None or not later:continue
        latest=max(later,key=lambda p:p['update'])
        if latest['mse']>ratio*max(initial['mse'],1e-12):
            return f"Fixed train {direction} MSE at update {latest['update']} exceeded {ratio:g}x its untrained prediction baseline"
    return None

def stop(*_):
    global STOP;STOP=True

def fixed_mse_guard_decision(points, job):
    """Explicit opt-in warning mode; nonfinite evaluations still fail."""
    import math
    mode = job.get('fixed_mse_guard_mode', 'stop')
    if mode not in ('stop', 'warn'):
        raise ValueError('fixed_mse_guard_mode must be stop or warn')
    if mode == 'warn':
        for point in points:
            if not math.isfinite(point['mse']):
                return f"Nonfinite fixed evaluation MSE at update {point['update']}", None
    reason = fixed_mse_failure(points, job.get('max_fixed_mse_ratio', 10.),
        effective_batch=job.get('effective_batch'), min_sample_visits=job.get('guard_min_sample_visits'))
    return (None, reason) if mode == 'warn' else (reason, None)

def main():
    import argparse
    global JOBS
    parser=argparse.ArgumentParser();parser.add_argument('--jobs',type=Path,default=JOBS)
    parser.add_argument('--status',type=Path,default=OUT/'search_status.json')
    args=parser.parse_args();JOBS=args.jobs
    import yaml
    signal.signal(signal.SIGTERM,stop)
    state={}
    if (args.status).exists():state=json.loads((args.status).read_text())
    while not STOP:
        jobs=json.loads(JOBS.read_text());pending=[]
        for job in jobs:
            name=job['name'];previous=state.get(name,{})
            if previous.get('phase') in ['complete','failed']:continue
            if job.get('depends_on'):
                dependency=state.get(job['depends_on'],{})
                if dependency.get('phase')=='failed':
                    state[name]=dict(phase='failed',reason='Prerequisite experiment failed; continuation not launched')
                    continue
                if dependency.get('phase')!='complete':continue
            pending.append(job)
        if not pending:
            write(args.status,state);return
        job=pending[0];name=job['name'];pair=job.get('physical_gpus','2,3')
        # Explicit local user restriction overrides historical queue placement.
        policy_path=ROOT/'configs/execution_gpu_policy.json'
        if policy_path.exists():
            allowed=json.loads(policy_path.read_text())['allowed_physical_pairs']
            if pair not in allowed:
                if len(allowed)!=1:raise RuntimeError('Requested GPU pair prohibited by local user policy')
                requested=pair;pair=allowed[0];job=dict(job,physical_gpus=pair)
                audit_folder=Path(yaml.safe_load((ROOT/job['config']).read_text())['trainer']['output_dir'])
                audit_folder.mkdir(parents=True,exist_ok=True)
                write(audit_folder/'gpu_placement_override.json',dict(requested=requested,actual=pair,policy=str(policy_path)))
                definition=audit_folder/'arm_definition.json'
                if definition.exists():
                    contents=json.loads(definition.read_text());contents['job']['physical_gpus']=pair;write(definition,contents)
        # Reserve one temporary snapshot, using the real existing checkpoint
        # size when resuming; do not deadlock an otherwise recoverable run at
        # an arbitrary larger disk threshold.
        import shutil
        import yaml
        disk_cfg=yaml.safe_load((ROOT/job['config']).read_text())
        disk_folder=Path(disk_cfg['trainer']['output_dir'])
        existing=disk_folder/'ckpt_latest.pt'
        snapshot_bytes=existing.stat().st_size if existing.exists() else job.get('snapshot_bytes_estimate',32*2**30)
        required_bytes=checkpoint_space_required(disk_folder,snapshot_bytes,
            keep_best=job.get('keep_best',False),profile=job.get('profile',False))
        # Reserve space for a different concurrently running job's pending save.
        required_bytes += int(job.get('external_checkpoint_reserve_bytes',0))
        space_path=disk_folder
        while not space_path.exists():space_path=space_path.parent
        free_bytes=shutil.disk_usage(space_path).free
        if free_bytes < required_bytes:
            state[name]=dict(state.get(name,{}), phase='waiting_for_checkpoint_space',physical_gpus=pair,
                output_filesystem_path=str(space_path),required_gib=required_bytes/2**30,free_gib=free_bytes/2**30)
            write(args.status,state);time.sleep(15);continue
        import fcntl
        gpu_lock_path=ROOT/'logs'/('gpu_pair_'+pair.replace(',','_')+'.lock')
        gpu_lock=gpu_lock_path.open('a+b')
        try:fcntl.flock(gpu_lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            gpu_lock.close()
            state[name]=dict(phase='waiting_for_gpu_pair_lock',physical_gpus=pair)
            write(args.status,state);time.sleep(15);continue
        if not gpu_idle(pair):
            gpu_lock.close()
            state[name]=dict(phase='waiting_for_idle_gpus',physical_gpus=pair)
            write(args.status,state);time.sleep(15);continue
        only_direction=job.get('only_direction','')
        if only_direction not in ('','action','future'):raise ValueError('invalid explicit diagnostic direction')
        cfgpath=ROOT/job['config'];cfg=yaml.safe_load(cfgpath.read_text());folder=Path(cfg['trainer']['output_dir']);folder.mkdir(parents=True,exist_ok=True)
        manifest=Path(job['manifest']);snapshot=folder/'sample_manifest.json'
        if snapshot.exists() and snapshot.read_bytes()!=manifest.read_bytes():raise RuntimeError('manifest mismatch')
        if not snapshot.exists():snapshot.write_bytes(manifest.read_bytes())
        if job.get('runtime_environment_manifest'):
            import importlib.metadata
            expected=json.loads(Path(job['runtime_environment_manifest']).read_text())
            actual={k:importlib.metadata.version(k) for k in expected['critical_versions']}
            if actual!=expected['critical_versions'] or sys.executable!=expected['python'] or sys.version!=expected['python_version']:
                gpu_lock.close()
                state[name]=dict(phase='failed',reason='Runtime environment differs from contemporaneous study lock')
                write(args.status,state);continue
            write(folder/'runtime_environment.json',dict(python=sys.executable,python_version=sys.version,critical_versions=actual,checked_unix=time.time()))
        command=[PYTHON,'-u','scripts/train_shared_sharded.py','--config',str(cfgpath),'--steps',str(job['steps']),
            '--physical-gpus',pair,'--fixed-window-manifest',str(snapshot),'--probe-eval-interval',str(job.get('eval_interval',20))]
        if job.get('profile',False):command+=['--profile']
        if job.get('keep_best',False):command+=['--keep-best']
        if job.get('resume_config_lrs',False):command+=['--resume-config-lrs']
        for key,flag in [('sample_eval_interval','--sample-eval-interval'),('train_monitor_manifest','--train-monitor-manifest'),('train_monitor_sample_interval','--train-monitor-sample-interval'),('alternate_final_eval_seed','--alternate-final-eval-seed')]:
            if key in job:command += [flag,str(job[key])]
        checkpoint=folder/'ckpt_latest.pt'
        if checkpoint.exists():command+=['--resume',str(checkpoint)]
        elif (folder/'train_log.jsonl').exists():
            gpu_lock.close()
            state[name]=dict(phase='failed',reason='training logs exist without a recoverable checkpoint; refusing duplicate overwrite')
            write(args.status,state);continue
        elif job.get('initial_checkpoint'):
            # Explicit fork into a new output directory; the source is read-only.
            command += ['--resume',str(job['initial_checkpoint'])]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=pair,ENERGYWAM_TRIM_BETWEEN_DIRECTIONS='0',ENERGYWAM_SAMPLE_ALIGNED_RNG='1',
            ENERGYWAM_FIXED_DATA_CACHE='1',ENERGYWAM_ONLY_DIRECTION=only_direction,ENERGYWAM_PROBE_MAX_K='2',ENERGYWAM_PROBE_SPLITS='train,val',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',BLIS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
        with (folder/'console.log').open('a') as log:
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            monitor=None
            if job.get('monitor_resources',False):
                sys.path.insert(0,str(ROOT))
                from energywam.training.resource_monitor import ResourceMonitor
                monitor=ResourceMonitor(child.pid,pair.split(','),folder)
                monitor.start()
            state[name]=dict(phase='running',pid=child.pid,physical_gpus=pair,config=str(cfgpath),target_updates=job['steps'],only_direction=only_direction)
            while child.poll() is None and not STOP:
                state[name]['evaluations']=collect(folder)
                rejection,warning=fixed_mse_guard_decision(state[name]['evaluations'],job)
                if warning and warning not in state[name].setdefault('fixed_mse_warnings',[]):
                    state[name]['fixed_mse_warnings'].append(warning)
                    with (folder/'fixed_mse_warnings.jsonl').open('a') as alerts:
                        alerts.write(json.dumps(dict(at_unix=time.time(),mode='warn',reason=warning,evaluations=state[name]['evaluations']))+'\n')
                    print('[fixed-mse-warning] '+warning,flush=True)
                if job.get('resume_guard_ratio'):
                    for direction in ('action','future'):
                        rows=[r for r in state[name]['evaluations'] if r['split']=='train' and r['direction']==direction and r['k']==2]
                        if len(rows)>1:
                            first=min(rows,key=lambda r:r['update']);last=max(rows,key=lambda r:r['update'])
                            if last['mse']>job['resume_guard_ratio']*max(first['mse'],1e-12):
                                rejection=f"Fixed TRAIN {direction} exceeded {job['resume_guard_ratio']}x resumed baseline at {last['update']}"
                if monitor and job.get('max_process_uss_gib') and monitor.summary.get('last_sample',{}).get('process_uss_gib',0)>job['max_process_uss_gib']:
                    rejection=f"Owned process USS exceeded {job['max_process_uss_gib']} GiB experiment budget"
                if monitor and job.get('max_wall_seconds') and time.time()-monitor.started>job['max_wall_seconds']:
                    rejection=f"Experiment exceeded {job['max_wall_seconds']} second time budget"
                if rejection:
                    os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=55)
                    state[name].update(phase='failed',reason=rejection,early_stopped=True)
                    write(args.status,state);break
                speed=folder/'speed_report.json'
                if speed.exists():
                    try:state[name]['completed_updates']=json.loads(speed.read_text())['completed_steps']
                    except (OSError,ValueError):pass
                write(args.status,state);time.sleep(15)
            if STOP:
                # Only the child session created by this controller is signalled.
                os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=55)
                if monitor:monitor.finish(child.returncode,'controller interrupted')
                state[name]['phase']='interrupted';write(args.status,state);return
            if monitor:
                monitor.finish(child.returncode,state[name].get('reason'))
                state[name]['resource_summary']=str(folder/'resource_summary.json')
            if (folder/'speed_report.json').exists():
                try:state[name]['completed_updates']=json.loads((folder/'speed_report.json').read_text())['completed_steps']
                except (OSError,ValueError,KeyError):pass
            state[name].update(phase='complete' if child.returncode==0 else 'failed',returncode=child.returncode,evaluations=collect(folder))
            write(args.status,state)
        gpu_lock.close()
if __name__=='__main__':main()

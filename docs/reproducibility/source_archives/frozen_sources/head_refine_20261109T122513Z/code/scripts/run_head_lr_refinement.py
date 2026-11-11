"""Run an explicitly prepared head-LR refinement, waiting for full disk budget."""
import argparse
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
CHILD=None
STOP=False


def read(path, default=None):
    try:return json.loads(Path(path).read_text())
    except (OSError,ValueError):return default


def write(path,value):
    path=Path(path);temp=path.with_suffix('.pending.json')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n');temp.replace(path)


def stop(*_):
    global STOP
    STOP=True
    if CHILD is not None and CHILD.poll() is None:CHILD.send_signal(signal.SIGTERM)


def summarize(plan):
    from scripts.run_iteration_search import collect
    root=Path(plan['root']);states=read(root/'queue_status.json',{});arms=[]
    for job in plan['jobs']:
        folder=Path(job['folder']);speed=read(folder/'speed_report.json',{});initial=read(folder/'initialization_step0.json',{})
        logs=[]
        if (folder/'train_log.jsonl').exists():
            for line in (folder/'train_log.jsonl').read_text().splitlines():
                try:row=json.loads(line)
                except ValueError:continue
                if 'optimizer_lr/energy_head' in row:logs.append(row)
        arms.append(dict(name=job['name'],head_lr=job['head_lr'],completed_updates=speed.get('completed_steps',0),target_updates=210,queue=states.get(job['name'],{'phase':'queued'}),initialization_sample_hash=initial.get('sampled_parameter_sha256'),points=collect(folder),full_train_log_records=len(logs),preclip_gradient_norm_max_full_log=max((r.get('param_grad_norm',0) for r in logs),default=None)))
    hashes={a['initialization_sample_hash'] for a in arms if a['initialization_sample_hash']}
    if len(hashes)>1:raise RuntimeError('Current-study initialization hashes differ')
    report=dict(updated_at=datetime.now(timezone.utc).isoformat(),arms=arms,comparison='Contemporaneous1e-3controlthen7.5e-4/1.25e-3;matched210updates;separatefixedTRAIN/VAL A/F;gradientmaxusesfullJSONLnotrolling100stepreport',initialization_scope='64valuesperparameterhash,notfullweightidentity')
    write(root/'comparison_report.json',report)
    return report


def main():
    global CHILD
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);args=p.parse_args();plan=read(args.plan);root=Path(plan['root'])
    lock=(root/'controller.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    state=dict(pid=os.getpid(),phase='preflight',started_at=datetime.now(timezone.utc).isoformat(),plan=str(args.plan),code=str(ROOT))
    try:
        while not STOP:
            free=shutil.disk_usage(root).free
            if free>=plan['required_free_bytes']:break
            state.update(phase='waiting_for_checkpoint_space',free_bytes=free,required_bytes=plan['required_free_bytes'])
            write(root/'controller_status.json',state);summarize(plan);time.sleep(20)
        if STOP:raise KeyboardInterrupt
        # Register only actual runnable studies with the existing log bridge.
        bridge=Path(plan['wandb_bridge']);settings=read(bridge,{})
        if settings.get('sources') is not None:
            for job in plan['jobs']:
                config=job['config']
                if not any(s.get('config')==config for s in settings['sources']):settings['sources'].append(dict(config=config,group=plan['name'],priority=220))
            write(bridge,settings);bridge.chmod(0o600)
        command=[sys.executable,'-u',str(ROOT/'scripts/run_iteration_search.py'),'--jobs',str(root/'jobs.json'),'--status',str(root/'queue_status.json')]
        state.update(phase='training',command=command)
        with (root/'queue_console.log').open('a') as stream:
            CHILD=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            state['child_pid']=CHILD.pid;write(root/'controller_status.json',state)
            while CHILD.poll() is None and not STOP:summarize(plan);time.sleep(20)
            if STOP and CHILD.poll() is None:CHILD.send_signal(signal.SIGTERM)
            CHILD.wait(timeout=100)
        result=summarize(plan)
        phases=[a['queue']['phase'] for a in result['arms']]
        state.update(phase='interrupted' if STOP else 'complete' if all(s=='complete' for s in phases) else 'finished_with_failures',arm_phases=phases,returncode=CHILD.returncode)
    except BaseException as error:
        if CHILD is not None and CHILD.poll() is None:CHILD.send_signal(signal.SIGTERM);CHILD.wait(timeout=100)
        state.update(phase='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',reason=repr(error))
        write(root/'controller_status.json',state)
        if not isinstance(error,KeyboardInterrupt):raise
    state['ended_at']=datetime.now(timezone.utc).isoformat();write(root/'controller_status.json',state)


if __name__=='__main__':main()

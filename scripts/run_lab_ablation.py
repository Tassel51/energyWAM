"""Durable host-scoped lab ablation: gates first, then bounded serial arms."""
import argparse
import fcntl
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
CHILD=None
STOP=False


def write(path,data):
    path=Path(path);temp=path.with_suffix('.pending.json')
    temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(path)


def read(path,default=None):
    for _ in range(3):
        try:return json.loads(Path(path).read_text())
        except (OSError,ValueError):time.sleep(.1)
    return default


def stop(*_):
    global STOP
    STOP=True
    if CHILD is not None and CHILD.poll() is None:CHILD.send_signal(signal.SIGTERM)


def summarize(plan):
    from scripts.run_iteration_search import collect
    queue=Path(plan['root'])/'full_status.json'
    statuses=read(queue,{})
    previous=read(Path(plan['root'])/'comparison_report.json',{})
    prior_steps={a['name']:a['completed_updates'] for a in previous.get('arms',[])}
    arms=[]
    for arm in plan['arms']:
        folder=Path(arm['folder']);speed=folder/'speed_report.json'
        speed_data=read(speed,{})
        completed=speed_data.get('completed_steps',prior_steps.get(arm['name'],0))
        initial=folder/'initialization_step0.json'
        arms.append(dict(name=arm['name'],factors=arm['factors'],completed_updates=completed,target_updates=210,equivalent_epochs=completed*16/67,points=collect(folder),status=statuses.get(arm['name'],{'phase':'queued'}),initialization_sample_hash=read(initial,{}).get('sampled_parameter_sha256'),speed_read_missing=not bool(speed_data)))
    hashes={a['initialization_sample_hash'] for a in arms if a['initialization_sample_hash']}
    if len(hashes)>1:raise RuntimeError('Within-study parameter sample initialization hashes differ')
    write(Path(plan['root'])/'comparison_report.json',dict(updated_at=datetime.now(timezone.utc).isoformat(),purpose=plan['purpose'],arms=arms,comparison='Separate fixed67TRAIN and16VAL Action/Future MSE; own0step baseline; no cross-weight total-loss ranking; failed arms not matched-budget winners',initialization_check='sampled parameters, not full state hash'))


def execute(command,log,state_file,state,*,gpu_pair=None):
    global CHILD
    from energywam.training.resource_monitor import ResourceMonitor
    with Path(log).open('a') as stream:
        env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',ENERGYWAM_TRIM_BETWEEN_DIRECTIONS='0')
        CHILD=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        state['child_pid']=CHILD.pid;state.setdefault('commands',[]).append(command);write(state_file,state)
        monitor=None
        if gpu_pair:
            folder=Path(log).parent/(Path(log).stem+'_resources');folder.mkdir(exist_ok=True)
            monitor=ResourceMonitor(CHILD.pid,gpu_pair.split(','),folder);monitor.start()
        start=time.monotonic()
        while CHILD.poll() is None and not STOP:
            if monitor:
                sample=monitor.summary.get('last_sample',{})
                if sample.get('process_uss_gib',0)>120 or sample.get('system_memory_available_gib',999)<24 or time.monotonic()-start>state.get('gate_timeout_seconds',1200):
                    os.killpg(CHILD.pid,signal.SIGTERM);CHILD.wait(timeout=60)
                    monitor.finish(CHILD.returncode,'Gradient gate resource/time guard')
                    raise RuntimeError('Gradient gate resource/time guard')
            time.sleep(5)
        if STOP and CHILD.poll() is None:CHILD.send_signal(signal.SIGTERM)
        CHILD.wait(timeout=100)
        if monitor:monitor.finish(CHILD.returncode)
        return CHILD.returncode


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True)
    args=parser.parse_args();plan=json.loads(args.plan.read_text());root=Path(plan['root'])
    assert socket.gethostname()==plan['hostname']
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
    os.environ['PYTHONPATH']=str(ROOT)
    lock=(root/'controller.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    state=dict(phase='preflight',pid=os.getpid(),hostname=socket.gethostname(),started_at=datetime.now(timezone.utc).isoformat(),plan=str(args.plan),gate_timeout_seconds=plan.get('gate_timeout_seconds',1200))
    state_file=root/'controller_status.json'
    if state_file.exists():
        import shutil
        history=root/'controller_history';history.mkdir(exist_ok=True)
        shutil.copy2(state_file,history/(str(time.time_ns())+'.json'))
    try:
        if plan.get('gradient_gate_command') and not plan.get('gradient_gate_after_capacity',False):
            gate=Path(plan['gradient_gate_result'])
            if not gate.is_file() or not json.loads(gate.read_text()).get('passed'):
                state['phase']='gradient_gate'
                result=execute(plan['gradient_gate_command'],root/'gradient_gate.log',state_file,state,gpu_pair=plan['gpu_pair'])
                if STOP:raise KeyboardInterrupt
                if result or not gate.is_file() or not json.loads(gate.read_text()).get('passed'):raise RuntimeError('Real4B gradient/storage equivalence gate did not pass')
        state['phase']='capacity_profile'
        result=execute([sys.executable,'-u',str(ROOT/'scripts/run_iteration_search.py'),'--jobs',plan['profile_jobs'],'--status',str(root/'profile_status.json')],root/'profile_controller.log',state_file,state)
        if STOP:raise KeyboardInterrupt
        status=read(root/'profile_status.json',{})
        if not status:raise RuntimeError('Capacity queue exited before writing status; inspect profile_controller.log')
        if result or not all(s['phase']=='complete' for s in status.values()):
            state.update(phase='blocked_capacity',reason='Requested resolved configuration did not pass capacity profile; no full arm launched')
            write(state_file,state);return
        if plan.get('gradient_gate_after_capacity',False):
            state['phase']='gradient_gate'
            result=execute(plan['gradient_gate_command'],root/'gradient_gate.log',state_file,state,gpu_pair=plan['gpu_pair'])
            if STOP:raise KeyboardInterrupt
            if result or not read(plan['gradient_gate_result'],{}).get('passed'):raise RuntimeError('Gradient/storage equivalence gate did not pass')
        gate=Path(plan['gradient_gate_result'])
        while not STOP and not (gate.is_file() and json.loads(gate.read_text()).get('passed')):
            state['phase']='waiting_for_gradient_gate';write(state_file,state);time.sleep(15)
        if STOP:raise KeyboardInterrupt
        if plan.get('resume_gate_command') and not read(plan.get('resume_gate_result',''),{}).get('passed'):
            state['phase']='resume_gate'
            result=execute(plan['resume_gate_command'],root/'resume_gate.log',state_file,state,gpu_pair=plan['gpu_pair'])
            proof=read(plan['resume_gate_result'],{})
            if STOP:raise KeyboardInterrupt
            if result or not proof.get('passed'):raise RuntimeError('Full model CPU-optimizer-state resume gate did not pass')
        import shutil
        if shutil.disk_usage(root).free<plan['whole_study_checkpoint_budget_bytes']:raise RuntimeError('Whole-study full latest/best/pending budget no longer available')
        state['phase']='training';command=[sys.executable,'-u',str(ROOT/'scripts/run_iteration_search.py'),'--jobs',plan['full_jobs'],'--status',str(root/'full_status.json')]
        global CHILD
        with (root/'full_controller.log').open('a') as log:
            CHILD=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            state['child_pid']=CHILD.pid;state.setdefault('commands',[]).append(command);write(state_file,state)
            while CHILD.poll() is None and not STOP:
                summarize(plan);time.sleep(20)
            if STOP and CHILD.poll() is None:CHILD.send_signal(signal.SIGTERM)
            CHILD.wait(timeout=100)
        summarize(plan)
        state.update(phase='interrupted' if STOP else ('complete' if CHILD.returncode==0 else 'failed'),ended_at=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        if CHILD is not None and CHILD.poll() is None:
            CHILD.send_signal(signal.SIGTERM);CHILD.wait(timeout=100)
        state.update(phase='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',reason=repr(error),ended_at=datetime.now(timezone.utc).isoformat())
        write(state_file,state);raise
    write(state_file,state)


if __name__=='__main__':main()

#!/usr/bin/env python3
"""Start one predeclared study only after a specified preceding arm completes."""
import argparse
from datetime import datetime,timezone
import fcntl
import json
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

STOP=False
CHILD=None


def read(path):
    try:return json.loads(Path(path).read_text())
    except (OSError,ValueError):return {}


def write(path,value):
    path=Path(path);temp=path.with_suffix('.pending.json')
    temp.write_text(json.dumps(value,indent=2));temp.replace(path)


def dependency_state(root,arm):
    root=Path(root);controller=read(root/'controller_status.json')
    status=read(root/'full_status.json').get(arm,{})
    if status.get('phase')=='complete' and controller.get('phase')=='complete':return 'ready'
    if status.get('phase') in ['failed','interrupted','cancelled']:return 'blocked_dependency'
    if controller.get('phase') in ['failed','interrupted','blocked_capacity']:return 'blocked_dependency'
    return 'waiting_for_preceding_study'


def stop(*_):
    global STOP
    STOP=True
    if CHILD is not None and CHILD.poll() is None:CHILD.send_signal(signal.SIGTERM)


def main():
    global CHILD
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True);args=parser.parse_args()
    plan=read(args.plan);root=Path(plan['root']);assert socket.gethostname()==plan['hostname']
    lock=(root/'after_study.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=dict(pid=__import__('os').getpid(),hostname=socket.gethostname(),plan=str(args.plan),started_at=datetime.now(timezone.utc).isoformat(),dependency=plan['after_study'])
    state_path=root/'after_study_status.json'
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    while not STOP:
        dependency=plan['after_study'];phase=dependency_state(dependency['root'],dependency['arm'])
        state.update(phase=phase,updated_at=datetime.now(timezone.utc).isoformat());write(state_path,state)
        if phase=='ready':break
        write(root/'controller_status.json',state)
        if phase=='blocked_dependency':return
        time.sleep(20)
    if STOP:
        state['phase']='interrupted';write(state_path,state);write(root/'controller_status.json',state);return
    # The standard controller performs exclusive GPU checks, K4 capacity, and disk gates.
    command=[sys.executable,'-u',str(Path(__file__).with_name('run_lab_ablation.py')),'--plan',str(args.plan)]
    state.update(phase='launching_gates',command=command);write(state_path,state)
    with (root/'after_study_child.log').open('a') as stream:
        CHILD=subprocess.Popen(command,cwd=Path(__file__).resolve().parents[1],stdout=stream,stderr=subprocess.STDOUT)
        state['child_pid']=CHILD.pid;write(state_path,state)
        while CHILD.poll() is None:
            if STOP:CHILD.send_signal(signal.SIGTERM);break
            time.sleep(5)
        CHILD.wait(timeout=120)
    final=read(root/'controller_status.json')
    state.update(phase=final.get('phase','failed'),returncode=CHILD.returncode,ended_at=datetime.now(timezone.utc).isoformat())
    write(state_path,state)

if __name__=='__main__':main()

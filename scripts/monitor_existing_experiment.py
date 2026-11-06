"""Attach resource evidence and finite safety bounds to an existing owned job."""
import argparse, os, signal, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from energywam.training.resource_monitor import ResourceMonitor
import psutil

def main():
    p=argparse.ArgumentParser();p.add_argument('--pid',type=int,required=True)
    p.add_argument('--expected-command',required=True);p.add_argument('--gpus',required=True)
    p.add_argument('--folder',required=True);p.add_argument('--max-uss',type=float,default=170)
    p.add_argument('--max-seconds',type=float,default=3600);a=p.parse_args()
    process=psutil.Process(a.pid)
    if a.expected_command not in process.cmdline():raise RuntimeError('process identity mismatch')
    created=process.create_time();folder=Path(a.folder);folder.mkdir(parents=True,exist_ok=True)
    monitor=ResourceMonitor(a.pid,a.gpus.split(','),folder);monitor.start();reason=None
    while process.is_running() and process.status()!=psutil.STATUS_ZOMBIE:
        if process.create_time()!=created:raise RuntimeError('PID identity changed')
        if time.time()-created>a.max_seconds:reason='owned experiment time budget'
        if monitor.summary.get('last_sample',{}).get('process_uss_gib',0)>a.max_uss:reason='owned experiment CPU USS budget'
        if reason:
            os.kill(a.pid,signal.SIGTERM);break
        time.sleep(5)
    monitor.finish(None,reason)

if __name__=='__main__':main()

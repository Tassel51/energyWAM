#!/usr/bin/env python3
"""Read/control only the two explicitly authorized lab hyperparameter studies."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import lab_pc_relay as relay


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['status','logs','stop'])
    p.add_argument('--study',choices=['loss','steps','k','future07','future67']);p.add_argument('--host',default='lab-via-pc')
    a=p.parse_args()
    if a.action=='stop' and not a.study:p.error('Specify exactly one --study to stop')
    plans=json.loads((ROOT/'logs/lab_remote_setup/ablation_deployment.json').read_text())['groups']
    plans=[x for x in plans if not a.study or x['name']==a.study]
    code='PLANS='+repr(plans)+'\nACTION='+repr(a.action)+'\n'+'''
from pathlib import Path
import json,os,signal,socket
assert socket.gethostname()=='user-SYS-420GP-TNR'
def read(p):
    try:return json.loads(p.read_text())
    except (OSError,ValueError):return {}
result=[]
for plan in PLANS:
    root=Path(plan['root']);state=read(root/'controller_status.json')
    row=dict(study=plan['name'],gpu_pair=plan['gpu_pair'],root=str(root),controller=state)
    if ACTION=='stop':
        assert state.get('phase') not in ['complete','failed','blocked_capacity','interrupted','blocked_dependency'], 'Study already stopped'
        pid=state['pid'];proc=Path('/proc')/str(pid)
        assert proc.stat().st_uid==os.getuid()
        cmd=(proc/'cmdline').read_bytes()
        expected=b'run_lab_after_study.py' if state.get('phase')=='waiting_for_preceding_study' else b'run_lab_ablation.py'
        assert expected in cmd and str(root/'plan.json').encode() in cmd,'PID identity changed'
        os.kill(pid,signal.SIGTERM);row['stop_requested_pid']=pid
    elif ACTION=='logs':
        phase=state.get('phase');log=root/('gradient_gate.log' if phase=='gradient_gate' else 'resume_gate.log' if phase=='resume_gate' else 'profile_controller.log' if phase in ['capacity_profile','blocked_capacity'] else 'full_controller.log')
        row['tail']='\\n'.join(log.read_text(errors='replace').splitlines()[-35:]) if log.is_file() else ''
        if phase=='resume_gate':
            resume_path=read(root/'plan.json').get('resume_gate_result')
            if resume_path:
                f=Path(resume_path).parent/'console.log'
                row['active_console_tail']='\\n'.join(f.read_text(errors='replace').splitlines()[-20:]) if f.is_file() else ''

        queue=read(root/('profile_status.json' if phase in ['capacity_profile','blocked_capacity'] else 'full_status.json'))
        for name,s in queue.items():
            if s.get('phase')=='running':
                folder=root/('capacity_profile' if name.startswith('capacity_') else name)
                f=folder/'console.log';row['active_console_tail']='\\n'.join(f.read_text(errors='replace').splitlines()[-20:]) if f.is_file() else ''
    else:
        row['controller']={k:v for k,v in state.items() if k!='commands'}
        row['profile_status']={k:{a:b for a,b in v.items() if a!='evaluations'} for k,v in read(root/'profile_status.json').items()}
        report=read(root/'comparison_report.json');row['arms']=[]
        for arm in report.get('arms',[]):
            row['arms'].append({k:v for k,v in arm.items() if k not in ['points','status']})
            row['arms'][-1]['phase']=arm.get('status',{}).get('phase')
            evaluation_k=arm.get('factors',{}).get('train_k',2)
            fixed=[p for p in arm.get('points',[]) if p['k']==evaluation_k and p['split']=='train']
            last=max([p['update'] for p in fixed],default=None)
            row['arms'][-1]['last_fixed_train']=[p for p in fixed if p['update']==last]
    result.append(row)
print(json.dumps(result))
'''
    print(json.dumps(relay.remote_python(a.host,code,retry_safe=a.action!='stop'),indent=2))


if __name__=='__main__':main()

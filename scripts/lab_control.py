#!/usr/bin/env python3
"""Control the explicitly authorized bounded lab validation from compshare."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import re

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import lab_pc_relay as relay

STATE=ROOT/'logs/lab_remote_setup'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['status','logs','prepare-smoke','start-smoke','stop-smoke','fetch','sync'])
    p.add_argument('--host',default='lab-via-pc')
    p.add_argument('--name',help='Unique name for a NEW bounded two-update smoke; never overwrites results')
    a=p.parse_args();plan=json.loads((STATE/'real_smoke_plan.json').read_text())
    prefix='PLAN='+repr(plan)+'\n'+'''from pathlib import Path
import json,os,signal,socket,subprocess
assert socket.gethostname()==PLAN['hostname']
out=Path(PLAN['output'])
assert out.parent==Path('/data7/zzc/energyWAM/lab_execution_20261109/outputs')
status=json.loads((out/'run_status.json').read_text()) if (out/'run_status.json').is_file() else {'phase':'not_started'}
'''
    if a.action=='prepare-smoke':
        import yaml
        if not a.name or not re.fullmatch(r'[A-Za-z0-9_-]{1,70}',a.name):p.error('prepare-smoke requires --name with1..70letters/digits/underscore/hyphen')
        config=yaml.safe_load((ROOT/'configs/remote/lab_real4b_smoke_relocated.yaml').read_text())
        new=dict(plan);new['output']=str(Path(plan['output']).parent/a.name)
        new['config']='/data7/zzc/energyWAM/lab_execution_20261109/setup/validation_plans/'+a.name+'/config.yaml'
        new['tmux_session']='energywam_lab_'+a.name
        latest=STATE/'latest_code_release.json'
        if latest.is_file():new['code']=json.loads(latest.read_text())['code_dir']
        config['run_id']=a.name;config['trainer']['output_dir']=new['output']
        config['data']['norm_stats_bundle_path']='/data7/zzc/energyWAM/lab_execution_20261109/assets/relocated_stats.json'
        assert config['trainer']['per_device_batch_size']==1 and config['trainer']['max_train_steps']==2
        code=prefix+'NEW='+repr(new)+'\nCONFIG='+repr(yaml.safe_dump(config,sort_keys=False))+'\n'+'''
assert status['phase'] in ['complete','failed','interrupted','not_started'],'Existing validation active; refusing to replace its plan'
assert not Path(NEW['output']).exists(),'Output already exists; choose a fresh name'
folder=Path(NEW['config']).parent;folder.mkdir(parents=True,exist_ok=False)
(folder/'config.yaml').write_text(CONFIG)
(folder/'plan.json').write_text(json.dumps(NEW,indent=2)+'\\n')
active=Path('/data7/zzc/energyWAM/lab_execution_20261109/setup/smoke_plan.json')
pending=active.with_suffix('.pending.json');pending.write_text(json.dumps(NEW,indent=2)+'\\n');pending.replace(active)
print(json.dumps({'prepared_only':True,'plan':NEW}))
'''
        receipt=relay.remote_python(a.host,code)
        history=STATE/'smoke_plans';history.mkdir(exist_ok=True)
        (history/(Path(plan['output']).name+'.json')).write_text(json.dumps(plan,indent=2)+'\n')
        (STATE/'real_smoke_plan.json').write_text(json.dumps(new,indent=2)+'\n')
        print(json.dumps(receipt,indent=2));return
    if a.action=='sync':
        from scripts.lab_remote import bundle
        bundle(STATE)
        report=json.loads((STATE/'bundle_report.json').read_text())
        name=Path(report['archive']).name.removesuffix('.tar.gz')+'_release'
        stage=relay.remote_python(a.host,'RELEASE='+repr(name)+'\nPROJECT_PARENT="/data3/xyx_zhuzicheng/projects"\n'+relay.REMOTE_STAGE)
        relay.upload_archive(a.host,report['archive'],stage['release']+'/source.tar.gz')
        receipt=relay.remote_python(a.host,'RELEASE='+repr(stage['release'])+'\nDIGEST='+repr(report['archive_sha256'])+'\n'+relay.REMOTE_VERIFY)
        code='TARGET='+repr(receipt['code_dir'])+'\n'+'''
from pathlib import Path
import json
root=Path('/data3/xyx_zhuzicheng/projects/energyWAM_remote');link=root/'current'
assert not link.exists() or link.is_symlink(),'Refusing to replace a real project directory'
tmp=root/'current.pending';assert not tmp.exists() and not tmp.is_symlink()
tmp.symlink_to(TARGET);tmp.replace(link)
print(json.dumps({'current':str(link),'resolved':str(link.resolve())}))
'''
        relay.remote_python(a.host,code)
        (STATE/'latest_code_release.json').write_text(json.dumps(receipt,indent=2)+'\n')
        print(json.dumps(receipt,indent=2));return
    if a.action=='status':
        code=prefix+'''
result=dict(hostname=socket.gethostname(),status=status)
for name in ['speed_report.json','resume_validation_result.json']:
    p=out/name
    if p.is_file():
        d=json.loads(p.read_text());result[name]=dict(completed_steps=d['completed_steps']) if name=='speed_report.json' else d
print(json.dumps(result,indent=2))
'''
    elif a.action=='logs':
        code=prefix+'''
p=out/'console.log'
if not p.is_file():p=Path('/data7/zzc/energyWAM/lab_execution_20261109/setup/real_smoke_controller.log')
print(json.dumps(dict(log_path=str(p),tail='\\n'.join(p.read_text(errors='replace').splitlines()[-50:]) if p.is_file() else 'not yet created'),indent=2))
'''
    elif a.action=='start-smoke':
        code=prefix+'''
import shlex
assert not out.exists(),'Existing validation output; refusing duplicate submission or overwrite'
assert subprocess.run(['tmux','has-session','-t',PLAN['tmux_session']],capture_output=True).returncode!=0,'Validation session already exists'
command=shlex.join([PLAN['python'],'/data7/zzc/energyWAM/lab_execution_20261109/setup/lab_smoke_runner.py','--plan','/data7/zzc/energyWAM/lab_execution_20261109/setup/smoke_plan.json'])+' > /data7/zzc/energyWAM/lab_execution_20261109/setup/real_smoke_controller.log 2>&1'
subprocess.run(['tmux','new-session','-d','-s',PLAN['tmux_session'],command],check=True)
print(json.dumps(dict(launched=True,session=PLAN['tmux_session'],budget='2 total updates only')))
'''
    elif a.action=='stop-smoke':
        code=prefix+'''
assert status.get('phase')=='running','No running owned smoke to stop'
pid=status['runner_pid'];proc=Path('/proc')/str(pid)
assert proc.stat().st_uid==os.getuid(),'Process belongs to another user'
cmd=(proc/'cmdline').read_bytes()
assert b'/lab_smoke_runner.py' in cmd and b'/setup/smoke_plan.json' in cmd,'PID identity changed; refusing signal'
os.kill(pid,signal.SIGTERM)
print(json.dumps(dict(stopped_owned_runner_pid=pid)))
'''
    else:
        code=prefix+'''
import tarfile,sys
assert out.is_dir(),'No smoke output yet'
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as tar:
    for f in sorted(out.rglob('*')):
        if f.is_file() and not f.is_symlink() and f.suffix!='.pt' and '.best_snapshots' not in f.parts and f.stat().st_size<=8*2**20:
            tar.add(f,arcname=str(f.relative_to(out)),recursive=False)
'''
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        folder=STATE/'results'/stamp;folder.mkdir(parents=True,exist_ok=False)
        archive=folder/'lab_validation_logs.tar.gz'
        with archive.open('wb') as stream:
            result=subprocess.run(['ssh',*relay.SSH_OPTIONS,a.host,'python3 -c '+shlex.quote(code)],stdout=stream,stderr=subprocess.PIPE,timeout=180)
        if result.returncode:raise RuntimeError(result.stderr.decode(errors='replace'))
        with tarfile.open(archive) as tar:
            for m in tar.getmembers():
                assert m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
                target=folder/m.name;target.parent.mkdir(parents=True,exist_ok=True)
                with target.open('xb') as f:f.write(tar.extractfile(m).read())
        print('Fetched logs/configs/evaluations/checkpoint metadata; full weights retained remotely:',folder);return
    print(json.dumps(relay.remote_python(a.host,code,retry_safe=a.action in ['status','logs']),indent=2))


if __name__=='__main__':main()

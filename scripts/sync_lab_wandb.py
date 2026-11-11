#!/usr/bin/env python3
"""Pull only allowlisted lab metrics to a local W&B sidecar; no remote secrets."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.watch_wandb import atomic_json

REMOTE=r'''
import json,socket
from pathlib import Path
assert socket.gethostname()=='user-SYS-420GP-TNR'
out=[]
for group in GROUPS:
 root=Path(group['root'])
 jobs_path=Path(group.get('full_jobs',str(root/'full_jobs.json')))
 jobs=json.loads(jobs_path.read_text()) if jobs_path.is_file() else []
 job_configs={job['name']:Path(job['config']) for job in jobs if 'name' in job and 'config' in job}
 for arm in group['arms']:
  folder=Path(arm['folder']);files={}
  config=job_configs.get(arm['name'],root/'configs'/(arm['name']+'.yaml'))
  paths=[config]
  paths += [folder/n for n in ['train_log.jsonl','speed_report.json','resource_summary.json','dataset_baselines.json','arm_definition.json','ckpt_latest.pt.meta.json','ckpt_best.pt.meta.json']]
  paths += list(folder.glob('eval_*.json'))+list((folder/'train_monitor').glob('eval_*.json'))
  for path in paths:
   if path.is_file() and path.stat().st_size<32*1024*1024:
    try:files['config.yaml' if path==config else str(path.relative_to(folder))]=path.read_text()
    except OSError:pass
  out.append(dict(study=group['name'],name=arm['name'],folder=str(folder),files=files,
   inventory=dict(hostname=socket.gethostname(),folder=str(folder),latest_exists=(folder/'ckpt_latest.pt').is_file(),best_exists=(folder/'ckpt_best.pt').is_file())))
print(json.dumps(out))
'''

def sync_once(host):
 groups=json.loads((ROOT/'logs/lab_remote_setup/ablation_deployment.json').read_text())['groups']
 code='GROUPS='+repr(groups)+'\n'+REMOTE
 result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ForwardAgent=no','-o','ConnectTimeout=10',host,'python3','-'],input=code,text=True,capture_output=True,timeout=90)
 if result.returncode:raise RuntimeError(result.stderr[-600:])
 rows=json.loads(result.stdout);sources=[]
 tombstone_path=ROOT/'logs/lab_remote_setup/wandb_deleted_runs.json'
 tombstones=json.loads(tombstone_path.read_text()) if tombstone_path.exists() else {}
 excluded=set(tombstones.get('excluded_arm_names',[]))
 for row in rows:
  folder=ROOT/'logs/lab_wandb_mirror'/row['study']/row['name'];folder.mkdir(parents=True,exist_ok=True)
  for name,content in row['files'].items():
   relative=Path(name)
   if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe mirror path')
   path=folder/relative;path.parent.mkdir(parents=True,exist_ok=True)
   if name=='config.yaml':
    cfg=yaml.safe_load(content)
    cfg['remote_source']={'hostname':row['inventory']['hostname'],'output_dir':row['folder'],'original_config_yaml':content}
    cfg['trainer']['output_dir']=str(folder)
    content=yaml.safe_dump(cfg,sort_keys=False)
   temp=path.with_suffix(path.suffix+'.pending');temp.write_text(content);temp.replace(path)
  atomic_json(folder/'remote_checkpoint_inventory.json',row['inventory'])
  if row['name'] not in excluded and (folder/'config.yaml').exists():sources.append(dict(config=str(folder/'config.yaml'),group='lab-'+row['study']+'-20261109',priority=250,validation='Fixed 67 TRAIN / 16 auxiliary VAL; own K and common K2 separately; not success rate'))
 settings=json.loads((ROOT/'configs/wandb_bridge.json').read_text())
 settings.update(sources=sources,status_file='logs/lab_wandb_bridge_status.json',cache_dir=str(ROOT/'logs/lab_wandb_cache'),max_source_starts_per_cycle=8,poll_seconds=15)
 atomic_json(ROOT/'configs/lab_wandb_bridge.json',settings)
 atomic_json(ROOT/'logs/lab_remote_setup/wandb_mirror_status.json',dict(last_success_unix=time.time(),host=host,sources=len(sources),arms=[dict(name=r['name'],remote_folder=r['folder'],files=len(r['files'])) for r in rows]))
 print(json.dumps(dict(event='mirror_updated',sources=len(sources),at=time.time())),flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--once',action='store_true');p.add_argument('--host',default='lab-via-pc');p.add_argument('--interval',type=int,default=60);a=p.parse_args()
 while True:
  try:sync_once(a.host)
  except Exception as error:
   atomic_json(ROOT/'logs/lab_remote_setup/wandb_mirror_error.json',dict(at=time.time(),error=str(error),metrics_not_fabricated=True))
   print(json.dumps(dict(event='mirror_error',error=str(error))),flush=True)
   if a.once:raise
  if a.once:return
  time.sleep(max(15,a.interval))
if __name__=='__main__':main()

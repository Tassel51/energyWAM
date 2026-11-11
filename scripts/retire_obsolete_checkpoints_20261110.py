"""Explicit 2026-11-10 authorized, bounded retirement; never delete logs/assets."""
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

ROOT=Path('/cloud/zzc/energyWAM/experiments')
SELECTED=[
 'adaptive_shared_20261103/fit_K2_B2_A005',
 'adaptive_shared_20261103/fit_K2_B2_Acandidate',
 'adaptive_shared_20261103/fit_K4_B2',
 'train_k248_sharded_20261103/k4',
 'single_episode_head_lr_refine_20261109T122513Z/episode3346_headscale1_repeat_b16_micro4_seed42',
]
MANIFEST=Path('/root/zzc/energyWAM/logs/lab_remote_setup/checkpoint_retirement_20261110.json')


def record(data):
 for target in [MANIFEST,ROOT/'checkpoint_retirement_20261110.json']:
  temp=target.with_suffix('.pending.json');temp.write_text(json.dumps(data,indent=2));temp.replace(target)


def main():
 assert not MANIFEST.exists(),'Retirement already recorded; do not execute twice'
 active=[]
 for proc in Path('/proc').iterdir():
  if not proc.name.isdigit():continue
  try:args=(proc/'cmdline').read_bytes().split(b'\0')
  except OSError:continue
  if any(b'train_shared_sharded.py' in arg for arg in args):
   assert not any(any(Path(folder).name.encode() in arg for arg in args) for folder in SELECTED),'Selected run active'
   active.append(int(proc.name))
 data=dict(authorized='User2026-11-10: rerun this experiment; previous unused checkpoints may be deleted',created_utc=datetime.now(timezone.utc).isoformat(),scope='Only listed project trials; no pretrained weights/datasets',selected=SELECTED,reason='Superseded old short pilots and interrupted repeat now replaced by a fresh rerun; retain original baseline/new completed LR trials/three-task diagnostic snapshots',free_before_bytes=shutil.disk_usage(ROOT).free,local_free_before_bytes=shutil.disk_usage('/root/zzc/energyWAM').free,files=[],preserved_files={},executed=False)
 for name in SELECTED:
  folder=ROOT/name;assert folder.is_dir()
  assert (folder/'train_log.jsonl').is_file(),'No actual training log evidence'
  for file in folder.rglob('*'):
   if not file.is_file():continue
   if file.suffix=='.pt':
    assert file.name in ['ckpt_latest.pt','ckpt_best.pt','checkpoint.pt','ckpt_pending.pt'],file
    if folder.resolve() not in file.resolve().parents:
     archive=Path('/root/zzc/energyWAM/checkpoint_archive')/name/file.name
     assert file.resolve()==archive and not archive.is_symlink(),'Unrecognized external weight target'
     ast=archive.stat()
     data['files'].append(dict(path=str(archive),bytes=ast.st_size,device=ast.st_dev,inode=ast.st_ino,link_inode=ast.st_ino,symlink=False,resolved=str(archive)))
     for extra in archive.parent.glob('*.json'):
      data['preserved_files'][str(extra)]=hashlib.sha256(extra.read_bytes()).hexdigest()
    st=file.stat();lst=file.lstat()
    data['files'].append(dict(path=str(file),bytes=st.st_size,device=st.st_dev,inode=st.st_ino,link_inode=lst.st_ino,symlink=file.is_symlink(),resolved=str(file.resolve())))
   elif file.stat().st_size<=8*2**20:
    data['preserved_files'][str(file)]=hashlib.sha256(file.read_bytes()).hexdigest()
  meta=folder/'ckpt_latest.pt.meta.json';d=json.loads(meta.read_text());assert isinstance(d.get('meta',{}).get('global_step'),int)
 unique={(f['device'],f['inode']):f['bytes'] for f in data['files']};data['expected_physical_bytes']=sum(unique.values());record(data)
 for f in sorted(data['files'],key=lambda x:not x['symlink']):
  path=Path(f['path']);lst=path.lstat();assert lst.st_ino==f['link_inode']
  if not f['symlink']:
   st=path.stat();assert (st.st_dev,st.st_ino,st.st_size)==(f['device'],f['inode'],f['bytes'])
  path.unlink()
 for path,sha in data['preserved_files'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha
 for name in SELECTED:
  folder=ROOT/name;(folder/'weights_retired.json').write_text(json.dumps(dict(at_utc=datetime.now(timezone.utc).isoformat(),authorization=data['authorized'],full_resume_available=False,manifest=str(ROOT/'checkpoint_retirement_20261110.json'),logs_configs_evaluations_preserved=True),indent=2))
 data['executed']=True;data['all_selected_weight_paths_removed']=all(not Path(f['path']).exists() and not Path(f['path']).is_symlink() for f in data['files'])
 for attempt in range(20):
  data['free_after_bytes']=shutil.disk_usage(ROOT).free
  data['local_free_after_bytes']=shutil.disk_usage('/root/zzc/energyWAM').free
  gained=data['free_after_bytes']-data['free_before_bytes']+data['local_free_after_bytes']-data['local_free_before_bytes']
  if gained>=int(data['expected_physical_bytes']*.9):break
  time.sleep(1)
 record(data)
 print(json.dumps(dict(files=len(data['files']),physical_retired_gib=data['expected_physical_bytes']/2**30,free_after_gib=data['free_after_bytes']/2**30,local_free_after_gib=data['local_free_after_bytes']/2**30,preserved_files=len(data['preserved_files'])),indent=2))

if __name__=='__main__':main()

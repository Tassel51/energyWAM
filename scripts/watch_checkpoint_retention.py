"""Read-only checkpoint lifecycle audit; never delete or modify weight files.

Archive metadata while a version still exists; observe atomic role replacements
and generation removal. Shared disk deltas are recorded without attributing all
concurrent cloud usage to an individual deletion.
"""
import argparse, datetime, fcntl, json, os, shutil, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DEFAULT=Path('/cloud/zzc/energyWAM/experiments/batch_grid_micro4_20261105')

def write_json(path,value):
    pending=path.with_suffix('.pending.json');pending.write_text(json.dumps(value,indent=2));pending.replace(path)

def observe_folder(folder,state):
    current={}
    for role in ['latest','best']:
        weight=folder/f'ckpt_{role}.pt';meta=folder/f'ckpt_{role}.pt.meta.json'
        if not weight.exists() or not meta.exists():continue
        stat=weight.stat();old=state.get(role)
        if old and (stat.st_dev,stat.st_ino,stat.st_size)==(old['device'],old['inode'],old['bytes']) and Path(old['metadata_archive']).exists():
            current[role]=dict(old,links=stat.st_nlink);continue
        content=meta.read_bytes();metadata=json.loads(content);step=metadata['meta']['global_step']
        key=f'step{step:08d}_inode{stat.st_ino}'
        archive=folder/'retained_checkpoint_metadata'/key;archive.mkdir(parents=True,exist_ok=True)
        target=archive/'checkpoint.pt.meta.json'
        if not target.exists():target.write_bytes(content)
        elif target.read_bytes()!=content:raise ValueError('same snapshot identity has different metadata')
        if role=='best' and (folder/'best_selection.json').exists():
            selection=(folder/'best_selection.json').read_bytes();json.loads(selection)
            if not (archive/'selection.json').exists():(archive/'selection.json').write_bytes(selection)
        current[role]=dict(step=step,device=stat.st_dev,inode=stat.st_ino,bytes=stat.st_size,links=stat.st_nlink,resolved_path=str(weight.resolve()),metadata_archive=str(target))
    events=[]
    for role,new in current.items():
        old=state.get(role)
        if old and (old['device'],old['inode'],old['step'])!=(new['device'],new['inode'],new['step']):
            still_selected=any((v['device'],v['inode'])==(old['device'],old['inode']) for v in current.values())
            old_path=Path(old['resolved_path'])
            old_generation_weight_missing=role=='best' and not old_path.exists()
            events.append(dict(event='checkpoint_role_replaced',run=folder.name,role=role,previous=old,current=new,previous_inode_still_selected=still_selected,previous_best_generation_weight_missing=old_generation_weight_missing,free_observed_bytes=shutil.disk_usage(folder).free,space_scope='Shared cloud free space at observation; not a causal per-file freed-byte measurement',observed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    return current,events

def tick(root):
    path=root/'checkpoint_retention_observer_state.json'
    state=json.loads(path.read_text()) if path.exists() else {}
    events=[]
    for folder in sorted(root.glob('batch*_micro4_fp32*')):
        if not folder.is_dir():continue
        current,changes=observe_folder(folder,state.get(folder.name,{}))
        if current:state[folder.name]=current
        events.extend(changes)
    if events:
        with (root/'checkpoint_retention_observations.jsonl').open('a') as f:
            for event in events:f.write(json.dumps(event)+'\n')
            f.flush();os.fsync(f.fileno())
    write_json(path,state)
    write_json(root/'checkpoint_retention_observer_status.json',dict(updated_unix=time.time(),runs_observed=len(state),changes_this_tick=len(events),weight_files_modified=False))

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=DEFAULT);p.add_argument('--once',action='store_true');a=p.parse_args()
    lock=ROOT/'logs/.checkpoint_write.lock';lock.parent.mkdir(exist_ok=True)
    while True:
        with lock.open('a+b') as f:
            try:fcntl.flock(f.fileno(),fcntl.LOCK_SH|fcntl.LOCK_NB)
            except BlockingIOError:pass # Leave a writer completely undisturbed.
            else:tick(a.root)
        if a.once:return
        time.sleep(15)

if __name__=='__main__':main()

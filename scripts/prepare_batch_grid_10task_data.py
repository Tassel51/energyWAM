"""Freeze trajectory-disjoint ten-task data for the batch8/16/32/64 goal."""
import json,random,hashlib,sys,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from energywam.config import load_config
from energywam.data.energy_dataset import LeRobotEpisodeReader,resolve_dataset_root
R=Path('/cloud/zzc/energyWAM/experiments/batch_grid_micro4_20261105');R.mkdir(exist_ok=True)
plan=json.loads((ROOT/'configs/experiments/batch8_16_32_64_plan_20261105.json').read_text());cfg=load_config(ROOT/'configs/experiments/native_capacity_micro4_effective16_seed42.yaml')
reader=LeRobotEpisodeReader(resolve_dataset_root(cfg));train=set(json.loads((ROOT/'manifests/train_episodes.json').read_text())['episode_indices']);val=set(json.loads((ROOT/'manifests/val_episodes.json').read_text())['episode_indices']);assert train.isdisjoint(val)
rng=random.Random(20261105);training=[];quick=[];confirm=[];monitor=[];task_counts={}
def windows(ep,task,split,count,start):
 return [dict(window_id=start+j,split=split,task=task,episode_index=ep['episode_index'],frame_index=round(j*(ep['length']-17)/max(count-1,1)),source_dataset=ep['source_dataset']) for j in range(count)]
for index,task in enumerate(plan['train_tasks']):
 pool=[e for e in reader.episodes if e['length']>=32 and e.get('source_dataset','').split('/')[-2]==task]
 tr=sorted((e for e in pool if e['episode_index'] in train),key=lambda e:e['episode_index']);va=sorted((e for e in pool if e['episode_index'] in val),key=lambda e:e['episode_index']);assert len(tr)>=10 and len(va)>=15
 picked=rng.sample(tr,10);held=rng.sample(va,15)
 for ep in picked:training+=windows(ep,task,'train',8,len(training))
 # Two different train episodes per task, detached fixed-monitor manifest.
 monitor.extend([dict(training[index*80]),dict(training[index*80+8])])
 for ep in held[:5]:quick+=windows(ep,task,'val',2,100000+len(quick))
 for ep in held[5:]:confirm+=windows(ep,task,'val',4,200000+len(confirm))
 task_counts[task]=dict(eligible_train_episodes=len(tr),eligible_val_episodes=len(va),selected_train=[e['episode_index'] for e in picked],selected_quick_val=[e['episode_index'] for e in held[:5]],selected_confirmation_val=[e['episode_index'] for e in held[5:]])
tr={w['episode_index'] for w in training};va={w['episode_index'] for w in quick};co={w['episode_index'] for w in confirm};assert len(training)==800 and len(quick)==100 and len(confirm)==400 and len(monitor)==20
assert len(tr)==100 and len(va)==50 and len(co)==100 and tr.isdisjoint(va|co) and va.isdisjoint(co) and tr<=train and va|co<=val
manifests={'train_quick_val_manifest.json':training+quick,'confirmation_manifest.json':confirm,'train_monitor_manifest.json':monitor}
hashes={}
for name,rows in manifests.items():
 data=dict(dataset_identity='robocasa365_target_50tasks/PandaOmron',split_seed=20261105,windows=rows,note='Fixed trajectory split; TRAIN and VAL only; no test data or statistics fitting.');p=R/name
 if p.exists():assert json.loads(p.read_text())==data,'Refusing to change frozen data'
 else:p.write_text(json.dumps(data,indent=2))
 hashes[name]=hashlib.sha256(p.read_bytes()).hexdigest()
audit=dict(status='frozen',train_windows=800,quick_val_windows=100,confirmation_windows=400,monitor_windows=20,episode_counts=[len(tr),len(va),len(co)],trajectory_disjoint=True,test_used=False,normalization_refit=False,files_sha256=hashes,tasks=task_counts);(R/'data_audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps({k:v for k,v in audit.items() if k!='tasks'},indent=2))

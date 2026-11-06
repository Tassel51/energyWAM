"""Freeze full-trajectory-pool stride16 Stage1 training and held-out evaluation."""
import json,sys,math,hashlib,collections
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from energywam.config import load_config
from energywam.data.energy_dataset import LeRobotEpisodeReader,resolve_dataset_root
R=Path('/cloud/zzc/energyWAM/experiments/three_task_full_20261106');R.mkdir(exist_ok=True)
cfg=load_config(ROOT/'configs/experiments/batch16_micro4_fp32_seed42.yaml');reader=LeRobotEpisodeReader(resolve_dataset_root(cfg))
train=set(json.loads((ROOT/'manifests/train_episodes.json').read_text())['episode_indices']);val=set(json.loads((ROOT/'manifests/val_episodes.json').read_text())['episode_indices']);assert train.isdisjoint(val)
tasks=['OpenDrawer','PickPlaceCounterToCabinet','WashFruitColander'];tr=[];va=[];monitor=[];counts={}
for task in tasks:
 eps=[e for e in reader.episodes if e['source_dataset'].split('/')[-2]==task]
 te=[e for e in eps if e['episode_index'] in train];ve=[e for e in eps if e['episode_index'] in val]
 for split,pool,target in [('train',te,tr),('val',ve,va)]:
  for e in pool:
   end=e['length']-17;assert end>=0
   starts=sorted(set([*range(0,end+1,16),end])) if split=='train' else [round(end*j/3) for j in range(4)]
   for t in starts:target.append(dict(window_id=len(tr)+len(va),split=split,task=task,episode_index=e['episode_index'],frame_index=t,source_dataset=e['source_dataset']))
 counts[task]=dict(train_episodes=len(te),val_episodes=len(ve),train_windows=sum(w['task']==task for w in tr),val_windows=sum(w['task']==task for w in va))
 for e in te[:4]:monitor.append(dict(split='train',task=task,episode_index=e['episode_index'],frame_index=(e['length']-17)//2,source_dataset=e['source_dataset'],window_id=1000000+len(monitor)))
assert len({w['episode_index'] for w in tr})==1367 and len({w['episode_index'] for w in va})==78
# IDs must be unique across both splits, independent of construction order.
for i,w in enumerate(tr+va):w['window_id']=i
files={'train_val_manifest.json':dict(dataset_identity=reader.dataset_identity,stream_training=True,windows=tr+va),'train_monitor_manifest.json':dict(dataset_identity=reader.dataset_identity,windows=monitor)}
for n,d in files.items():
 p=R/n
 if p.exists():assert json.loads(p.read_text())==d
 else:p.write_text(json.dumps(d,indent=2))
steps=math.ceil(len(tr)/16)
for seed,pair in [(42,'0,1'),(43,'2,3')]:
 c=yaml.safe_load((ROOT/'configs/experiments/batch16_micro4_fp32_seed42.yaml').read_text());name=f'three_task_b16_micro4_seed{seed}'
 c['run_id']=name;c['notes']='All1367TRAIN trajectories; stride16 plus terminal start; one full shuffled window pass; streaming TRAIN; fresh initialization; Stage1 only.'
 c['trainer'].update(seed=seed,max_train_steps=steps,output_dir=str(R/name),save_interval=100,checkpoint_interval_seconds=1800)
 cp=ROOT/f'configs/experiments/{name}.yaml';cp.write_text(yaml.safe_dump(c,sort_keys=False))
 job=dict(name=name,config=str(cp.relative_to(ROOT)),physical_gpus=pair,steps=steps,manifest=str(R/'train_val_manifest.json'),train_monitor_manifest=str(R/'train_monitor_manifest.json'),train_monitor_sample_interval=1024,sample_eval_interval=4096,keep_best=True,monitor_resources=True,effective_batch=16,guard_min_sample_visits=256,max_process_uss_gib=140,max_fixed_mse_ratio=10,snapshot_bytes_estimate=24568519249)
 (R/f'jobs_{seed}.json').write_text(json.dumps([job],indent=2))
audit=dict(status='prepared_not_launched',tasks=counts,train_windows=len(tr),val_windows=len(va),updates=steps,sample_visits=steps*16,training_protocol='One shuffled pass over stride16 windows plus endpoint per full TRAIN trajectory; at most15 repeated windows to complete final batch. Natural window proportions, not task-balanced.',initialization='Fresh pretrained Qwen/DINO and new trainable modules; no toy warmstart',normalization='Frozen50taskTRAIN-only150k stats; no VAL/TEST fit',best_criterion='validation Action MSE +9*Future MSE, both conditional directions K2; separate metrics and per-task rows retained',validation_cadence=4096,weights='Full resumable latest/best; atomically verified rotation',files_sha256={n:hashlib.sha256((R/n).read_bytes()).hexdigest() for n in files},preliminary_hours_per_seed=[len(tr)*3.4/3600,len(tr)*5/3600],caveat='Historical cached throughput only; streaming decode/DINO and evaluation add time; replace with measured ETA.')
(R/'training_plan.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit,indent=2))

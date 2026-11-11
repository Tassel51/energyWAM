"""Read-only audit of recorded instability windows and their raw action/state targets."""
import json,sys,collections
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,'/root/zzc/energyWAM')
from energywam.config import load_config
from energywam.data.energy_dataset import resolve_dataset_root
from energywam.data.episode_dataset import LeRobotEpisodeReader
from energywam.training.run_state import FixedSampleBudgetSampler
R=Path('/cloud/zzc/energyWAM/experiments/stability_diagnosis_20261107');B=R/'replay680_baseline'
S=Path('/cloud/zzc/energyWAM/experiments/three_task_full_20261106/three_task_b16_micro4_seed42')
cfg=load_config('configs/experiments/three_task_b16_micro4_seed42.yaml');reader=LeRobotEpisodeReader(resolve_dataset_root(cfg))
m=json.load(open(B/'sample_manifest.json'));train=[w for w in m['windows'] if w['split']=='train'];order=list(FixedSampleBudgetSampler(len(train),42,680*16,692*16));batch={u:[train[i] for i in order[(u-681)*16:(u-680)*16]] for u in range(681,693)}
rows=[]
for split,folder,step in [('val',B,680),('val',B,692),('train',B/'train_monitor',680),('train',B/'train_monitor',688),('train',B/'train_monitor',692)]:
 for direction in ['action','future']:
  p=folder/f'eval_{step:04d}_{direction}.json'
  for w in json.load(open(p))['rows']:
   if w['k']==2:rows.append(w)
allw={ (w['episode_index'],w['frame_index']):w for ws in batch.values() for w in ws}
allw.update({(w['episode_index'],w['frame_index']):w for w in rows})
stats=json.load(open('stats/trainmix_150k_paper_standardized_v1.json'))['action'];mean=np.array(stats['mean']);std=np.array(stats['std']);cache={};checks={}
for key,w in allw.items():
 ep,t=key
 if ep not in cache:cache[ep]=(reader.read_actions(ep),reader.read_states(ep))
 a,q=cache[ep];chunk=a[t:t+16];z=(chunk-mean)/(std+stats['epsilon'])
 checks[key]={'episode_index':ep,'frame_index':t,'task':w['task'],'raw_action_shape':list(chunk.shape),'raw_action_absmax':float(np.abs(chunk).max()),'standardized_action_absmax':float(np.abs(z).max()),'standardized_action_zero_mse':float(np.square(z).mean()),'action_finite':bool(np.isfinite(chunk).all()),'state_finite':bool(np.isfinite(q[t]).all()),'full_horizon':len(chunk)==16,'future_in_bounds':t+16<len(a)}
bybatch={str(u):{'task_counts':dict(collections.Counter(w['task'] for w in ws)),'max_standardized_action_abs':max(checks[(w['episode_index'],w['frame_index'])]['standardized_action_absmax'] for w in ws),'mean_zero_action_mse':float(np.mean([checks[(w['episode_index'],w['frame_index'])]['standardized_action_zero_mse'] for w in ws])),'windows':ws} for u,ws in batch.items()}
per_task={}
for step in [680,688,692]:
 for split in ['train','val']:
  for d in ['action','future']:
   selected=[w for w in rows if w['update']==step and w['split']==split and w['direction']==d]
   if selected:
    per_task[f'{step}/{split}/{d}']={task:{'mse':float(np.mean([w['prediction_mse'] for w in selected if w['task']==task])),'count':sum(w['task']==task for w in selected)} for task in sorted({w['task'] for w in selected})}
worst=[]
for split in ['train','val']:
 for d in ['action','future']:
  selected=sorted([w for w in rows if w['split']==split and w['direction']==d],key=lambda x:x['prediction_mse'],reverse=True)[:4]
  for w in selected:
   key=(w['episode_index'],w['frame_index']);ident=lambda r:(r['episode_index'],r['frame_index'])==key
   historical=[]
   for step in [0,256,512]:
    folder=S if split=='val' else S/'train_monitor';p=folder/f'eval_{step:04d}_{d}.json'
    if p.exists():
     for r in json.load(open(p))['rows']:
      if r['k']==2 and ident(r):historical.append({'update':step,'mse':r['prediction_mse']})
   worst.append({'split':split,'direction':d,'update':w['update'],'task':w['task'],'episode_index':key[0],'frame_index':key[1],'initial_candidate_mse':next((r['prediction_mse'] for r in json.load(open((B if split=='val' else B/'train_monitor')/f'eval_{w["update"]:04d}_{d}.json'))['rows'] if r['k']==0 and ident(r)),None),'mse':w['prediction_mse'],'inner_steps':w.get('inner_steps'),'earlier_same_window':historical,'raw_check':checks[key]})
out={'scope':'Read-only raw action/state audit for replay681..692 and36 fixed monitor/VAL windows; no new GPU evaluation, no comprehensive RGB/Future latent integrity audit.','unique_windows':len(checks),'unique_episodes':len(cache),'all_action_state_finite':all(w['action_finite'] and w['state_finite'] for w in checks.values()),'all_full_horizon_and_future_in_bounds':all(w['full_horizon'] and w['future_in_bounds'] for w in checks.values()),'standardized_action_absmax':max(w['standardized_action_absmax'] for w in checks.values()),'by_batch':bybatch,'fixed_per_task':per_task,'worst_windows':worst,'raw_checks':list(checks.values())}
(R/'task_sample_audit.json').write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k not in ['by_batch','worst_windows','raw_checks']},indent=2));print('worst',json.dumps(worst[:4],indent=2));print('batches',json.dumps({k:{a:b for a,b in v.items() if a!='windows'} for k,v in bybatch.items()},indent=2))

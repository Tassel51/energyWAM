"""Read-only CPU audit of paired K training; never loads a model onto GPUs."""
import json,math,statistics
from pathlib import Path
import torch
from safetensors import safe_open
ROOT=Path('/cloud/zzc/energyWAM/experiments/overfit_k24_alpha200_20261103')
BASE=Path('/datasets/models/Qwen3-VL-4B-Instruct')
index=json.loads((BASE/'model.safetensors.index.json').read_text())['weight_map']
report={};logs={}
for k in [2,4]:
 folder=ROOT/f'k{k}'
 data={int(r['global_step']):r for r in map(json.loads,(folder/'train_log.jsonl').read_text().splitlines())}
 logs[k]=data
 checkpoint=torch.load(folder/'ckpt_latest.pt',map_location='cpu',mmap=True,weights_only=False)
 changed=total=0;tensors=[]
 for name,tensor in checkpoint['model'].items():
  if not name.startswith('backbone.model.model.language_model.'):continue
  original=name.removeprefix('backbone.model.')
  if original not in index:continue
  shape=tensor.shape
  slices=(slice(0,4),slice(0,1024)) if tensor.ndim==2 else (slice(0,4096),)
  with safe_open(str(BASE/index[original]),framework='pt',device='cpu') as f:
   before=f.get_slice(original)[slices]
  after=tensor[slices];different=int((after!=before).sum());n=after.numel()
  changed+=different;total+=n
  if len(tensors)<4:tensors.append(dict(name=name,dtype=str(tensor.dtype),changed=different,sampled=n,max_abs_change=float((after.float()-before.float()).abs().max())))
 evaluations=[]
 for file in sorted(folder.glob('eval_*.json')):
  rows=json.loads(file.read_text())['rows']
  for ek in [0,2,4]:
   subset=[r['prediction_mse'] for r in rows if r['k']==ek]
   if subset:evaluations.append(dict(file=file.name,eval_K=ek,mse=statistics.mean(subset)))
 first=data[min(data)]
 report[str(k)]=dict(latest_logged_step=max(data),checkpoint_step=checkpoint['meta']['global_step'],
  actual_action_K=sorted({r['diag/action_solver_steps'] for r in data.values()}),
  actual_future_K=sorted({r['diag/future_solver_steps'] for r in data.values()}),
  sampled_language_weight_change=dict(changed=changed,total=total,fraction=changed/total,examples=tensors),
  first_update_candidate_step_rms=dict(action=first['diag/action_action_update_norm_mean']/math.sqrt(16*12),
     future=first['diag/future_future_update_norm_mean']/math.sqrt(3*196*384)),
  optimizer_step_counts=sorted({s['step'] for s in checkpoint['optimizer']['state'].values()}),evaluations=evaluations)
 del checkpoint
common=sorted(logs[2].keys()&logs[4].keys())
report['paired_training_logs']=dict(common_updates=len(common),
 mean_abs_total_loss_difference=statistics.mean(abs(logs[2][i]['loss']-logs[4][i]['loss']) for i in common),
 exact_same_total_losses=sum(logs[2][i]['loss']==logs[4][i]['loss'] for i in common),
 first_gradient_norms=[logs[k][0]['param_grad_norm'] for k in [2,4]])
report['limitations']='Weight changes sampled against pretrained language weights only, not full checkpoint equality; snapshots can be from different update counts. No training settings changed.'
(ROOT/'overlap_audit.json').write_text(json.dumps(report,indent=2))
for k in ['2','4']:
 r=report[k];print('K',k,'checkpoint',r['checkpoint_step'],'actual K',r['actual_action_K'],r['actual_future_K'],
 'changed language weight sample fraction',r['sampled_language_weight_change']['fraction'],
 'candidate initial step RMS',r['first_update_candidate_step_rms'])
print(report['paired_training_logs'])

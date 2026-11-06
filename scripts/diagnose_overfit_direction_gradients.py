"""Read-only fresh-initialization gradient-scale diagnostic; no optimizer steps."""
import os,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
if os.environ.get('CUDA_VISIBLE_DEVICES')!='0,1':raise RuntimeError('diagnostic uses physical GPUs0/1 only')
import torch
from energywam.config import load_config
from energywam.models.energy_wam import QwenEnergyWAM
from energywam.training.shared_microbatch import SharedMicrobatchTrainer
from energywam.training.run_state import FixedSampleBudgetSampler
from energywam.data.energy_dataset import collate_energy_batch
from k_training_probe import fixed_dataset
import train as T
OUT=Path('/cloud/zzc/energyWAM/experiments/overfit_k24_alpha200_20261103/direction_gradient_diagnostic');OUT.mkdir(exist_ok=True)
cfg=load_config(ROOT/'configs/experiments/overfit_k2_alpha200_20261103.yaml');cfg.trainer.output_dir=str(OUT)
torch.manual_seed(cfg.trainer.seed)
bundle=T.resolve_training_stats(cfg);model=QwenEnergyWAM(cfg,stats_bundle=bundle);trainer=SharedMicrobatchTrainer(cfg,model)
manifest=json.loads((OUT.parent/'sample_manifest.json').read_text());dataset,_=fixed_dataset(cfg,model,manifest,'train')
indices=list(FixedSampleBudgetSampler(len(dataset),cfg.trainer.seed,consumed=0,total=2))
records=[]
def direction_grad(batch,direction,alpha,position):
 cfg.solver.alpha_future=alpha;model.train();trainer.optimizer.zero_grad(set_to_none=True)
 role=0 if direction=='action' else 1
 torch.manual_seed(cfg.trainer.seed+role+position*100003)
 start=time.monotonic();result=model(dict(batch,_bi_only=direction,_global_step=0));result['action_loss'].backward()
 scalar={k:float(v.detach()) for k,v in result.items() if isinstance(v,torch.Tensor) and v.numel()==1}
 normsq=0.;vector=[];group_norms={}
 for name,p in model.named_parameters():
  if p.grad is None:continue
  n=float(torch.linalg.vector_norm(p.grad,dtype=torch.float32));normsq+=n*n
  group='language' if 'language_model' in name else 'other_trainable'
  group_norms[group]=group_norms.get(group,0.)+n*n
  vector.append(p.grad.detach().reshape(-1)[:256].float().cpu().clone())
 vector=torch.cat(vector)
 row=dict(sample_index=indices[position],position=position,direction=direction,alpha_future=alpha,
     loss=scalar.get(f'bi/{direction}_loss'),total_parameter_grad_norm=normsq**.5,
     subgroup_grad_norms={g:v**.5 for g,v in group_norms.items()},seconds=time.monotonic()-start,
     update_norm=scalar.get(f'diag/{direction}_{direction}_update_norm_mean'))
 del result;trainer.optimizer.zero_grad(set_to_none=True)
 return row,vector
for position,index in enumerate(indices):
 batch=collate_energy_batch([dataset[index]])
 action,av=direction_grad(batch,'action',200.,position);records.append(action)
 for alpha in [.05,200.]:
  future,fv=direction_grad(batch,'future',alpha,position)
  future['sampled_gradient_cosine_with_action']=float(torch.dot(av,fv)/(av.norm()*fv.norm()).clamp_min(1e-30))
  future['gradient_norm_ratio_to_action']=future['total_parameter_grad_norm']/action['total_parameter_grad_norm']
  records.append(future)
 (OUT/'report.json').write_text(json.dumps(dict(weight_updates=0,initialization='fresh seed42, same as K comparison',full_gradient_cosine=False,gradient_cosine_scope='up to first256 elements per tensor, diagnostic sample only',records=records),indent=2))
 print(json.dumps(records[-3:]),flush=True)
print('complete: no optimizer steps or existing checkpoints changed',flush=True)

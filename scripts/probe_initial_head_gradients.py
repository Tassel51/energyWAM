"""Read-only initial residual-head gradient comparison on fixed training windows."""
import os,sys,json,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--manifest',required=True);p.add_argument('--output',required=True);p.add_argument('--limit',type=int,default=8);p.add_argument('--checkpoint');p.add_argument('--full-head',action='store_true');a=p.parse_args()
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='0':raise RuntimeError('this probe explicitly requires isolated physical GPU 0')
 import torch
 from energywam.config import load_config
 from energywam.models.energy_wam import QwenEnergyWAM
 from energywam.data.energy_dataset import collate_energy_batch
 from k_training_probe import fixed_dataset
 import train as T
 cfg=load_config(a.config);torch.manual_seed(cfg.trainer.seed);cfg.loss.lambda_action=cfg.loss.lambda_future=1.
 cfg.trainer.gradient_checkpointing=True
 cfg.trainer.saved_tensor_gpu_budgets_gib=[14.]
 cfg.trainer.language_model_devices=[0]
 model=QwenEnergyWAM(cfg,stats_bundle=T.resolve_training_stats(cfg)).to('cuda:0').train().requires_grad_(False)
 checkpoint_step=0
 if a.checkpoint:
  from energywam.training.checkpoint import restore
  payload=torch.load(a.checkpoint,map_location='cpu',mmap=True,weights_only=False)
  meta=restore(payload,model=model,optimizer=None,scheduler=None,mode='resume',restore_rng=False)
  checkpoint_step=meta.global_step;del payload
 weight=model.core.energy_head.net[-1].weight
 parameters=list(model.core.energy_head.parameters()) if a.full_head else [weight]
 for parameter in parameters:parameter.requires_grad_(True)
 last_index=next(i for i,parameter in enumerate(parameters) if parameter is weight)
 ds,windows=fixed_dataset(cfg,model,json.loads(Path(a.manifest).read_text()),'train')
 rows=[];grads={'action':[],'future':[]};out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
 selected=torch.linspace(0,len(windows)-1,min(a.limit,len(windows))).long().tolist()
 for i in selected:
  row=dict(windows[i])
  for direction in grads:
   torch.manual_seed(2026+windows[i]['window_id']*1009)
   batch=collate_energy_batch([ds[i]]);batch['_bi_only']=direction
   loss=model(batch)['action_loss'];derivatives=torch.autograd.grad(loss,parameters,allow_unused=True)
   g=torch.cat([(torch.zeros_like(parameter) if derivative is None else derivative).float().cpu().flatten() for parameter,derivative in zip(parameters,derivatives)])
   grads[direction].append(g)
   last=derivatives[last_index]
   row[direction]={'mse':float(loss.detach()),'head_gradient_norm':float(g.norm()),'last_weight_gradient_norm':0. if last is None else float(last.norm())};del loss,g,derivatives,last
  rows.append(row)
  report={'weight_updates':0,'checkpoint_step':checkpoint_step,'source_checkpoint':a.checkpoint,'scope':('all energy-head parameters' if a.full_head else 'last energy-head weight only')+'; unweighted directional training loss; not full-backbone gradients','rows':rows}
  ga=torch.stack(grads['action']);gz=torch.stack(grads['future'])
  report['rms_norm_ratio_action_over_future']=float(ga.square().sum(-1).mean().sqrt()/gz.square().sum(-1).mean().sqrt().clamp_min(1e-30))
  report['mean_gradient_cosine']=float(torch.nn.functional.cosine_similarity(ga.mean(0),gz.mean(0),dim=0))
  out.write_text(json.dumps(report,indent=2));print(json.dumps(row),flush=True)
if __name__=='__main__':main()

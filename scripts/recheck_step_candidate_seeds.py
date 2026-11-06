"""Read-only action robustness check, no optimizer, no training updates."""
import pathlib,os,json,sys,gc,time,argparse
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='2,3'
os.environ.update(ENERGYWAM_PROBE_MAX_K='2',ENERGYWAM_PROBE_SPLITS='train,val',ENERGYWAM_FIXED_DATA_CACHE='1')
import torch
from types import SimpleNamespace
from energywam.config import load_config
from energywam.models.energy_wam import QwenEnergyWAM
from energywam.training.checkpoint import restore
from k_training_probe import evaluate
import train as T
R=pathlib.Path('/cloud/zzc/energyWAM/experiments/iteration_aware_v1_20261104')
class NoOptimizer:
 def zero_grad(self,**kwargs):pass
results={}
parser=argparse.ArgumentParser();parser.add_argument('--comparison',choices=['step','prior'],default='step')
parser.add_argument('--direction',choices=['action','future'],default='action');parser.add_argument('--only');args=parser.parse_args()
step=80 if args.comparison=='step' else 400
names=([f'no_prior_currentinit_{label}_headlr0001_diag_seed42' for label in ['fixed','learned']]
       if args.comparison=='step' else ['ablation_prior1_currentinit1_seed42','future_only_currentinit_learned_seed42'])
if args.only:
 if args.only not in names:raise ValueError('requested model is outside the paired comparison')
 names=[args.only]
prefix=args.comparison if args.direction=='action' else f'{args.comparison}_future_replay'
for name in names:
 folder=R/name
 for _ in range(120):
  checkpoint=folder/'model_reference.pt'
  if not checkpoint.exists():checkpoint=folder/'ckpt_latest.pt'
  final=folder/'final_report.json'
  if (checkpoint.exists() and pathlib.Path(str(checkpoint)+'.meta.json').exists()
      and final.exists() and json.loads(final.read_text())['global_step']==step):break
  time.sleep(15)
 else:raise RuntimeError('bounded model reference not published; no restart attempted')
 cfg=load_config(f'configs/experiments/{name}.yaml');torch.manual_seed(42)
 bundle=T.resolve_training_stats(cfg);model=QwenEnergyWAM(cfg,stats_bundle=bundle)
 model.to(torch.device(cfg.trainer.device))
 from energywam.training.parallel import shard_language_model
 shard_language_model(model,cfg.trainer.language_model_devices,torch.device(cfg.trainer.device))
 payload=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False)
 meta=restore(payload,model=model,optimizer=None,scheduler=None,mode='resume',restore_rng=False);assert meta.global_step==step
 del payload
 trainer=SimpleNamespace(model=model,cfg=cfg,optimizer=NoOptimizer(),global_step=meta.global_step)
 model.requires_grad_(False);model.eval();results[name]={}
 for seed in ([2026,2027,2028] if args.direction=='action' else [2026]):
  out=R/f'{prefix}_candidate_seed_recheck'/name/f'seed{seed}';out.mkdir(parents=True,exist_ok=True)
  evaluate(trainer,R/'tinyfit_manifest.json',out,0 if args.direction=='action' else 1,evaluation_seed=seed)
  data=json.loads((out/f'eval_{step:04d}_{args.direction}.json').read_text());assert data['weight_updates_during_eval']==0
  scores={s:sum(x['prediction_mse'] for x in data['rows'] if x['split']==s and x['k']==2)/16 for s in ['train','val']}
  if seed==2026:
   reference=json.loads((folder/f'eval_{step:04d}_{args.direction}.json').read_text())
   a=[x['prediction_mse'] for x in reference['rows'] if x['k']==2];b=[x['prediction_mse'] for x in data['rows'] if x['k']==2]
   assert len(a)==len(b)==32
   difference=max(abs(x-y) for x,y in zip(a,b));assert difference<1e-5,f'checkpoint evaluation mismatch {difference}'
   scores['original_evaluation_max_abs_difference']=difference
  results[name][seed]=scores
  (R/f'{prefix}_candidate_seed_recheck_summary.json').write_text(json.dumps({'weight_updates':0,'checkpoint_step':step,'direction':args.direction,'scope':('Action-only paired candidate initialization seeds; not new training seeds or episodes.' if args.direction=='action' else 'Exact saved-weight Future replay; exact-current initialization is deterministic.'),'results':results},indent=2))
  print(name,seed,scores,flush=True)
 del trainer,model,bundle;gc.collect();torch.cuda.empty_cache()

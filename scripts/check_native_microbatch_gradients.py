"""Actual4B native2 versus sequential1 full outer gradients, SGD lr0."""
import argparse,json,os,sys,math,gc,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--manifest',required=True);p.add_argument('--output',required=True)
 p.add_argument('--physical-gpus',choices=['0,1','2,3'],default='2,3');p.add_argument('--precision',choices=['configured','float32','float16'],default='configured');p.add_argument('--directions',default='action,future');p.add_argument('--native-microbatch',type=int,choices=[2,4],default=2);p.add_argument('--sample-indices');p.add_argument('--math-mode',choices=['default','strict','tf32'],default='default');p.add_argument('--repeat-reference',action='store_true');p.add_argument('--record-all',action='store_true');p.add_argument('--saved-gpu-budget',type=float);p.add_argument('--no-async-offload',action='store_true');p.add_argument('--dedup-cpu',action='store_true');a=p.parse_args()
 assert os.environ.get('CUDA_VISIBLE_DEVICES')==a.physical_gpus
 os.environ['ENERGYWAM_SAMPLE_ALIGNED_RNG']='1'
 import torch
 if a.math_mode=='strict':
  torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
  torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
  torch.backends.cuda.matmul.allow_tf32=False
  torch.backends.cudnn.allow_tf32=False
 from energywam.config import load_config
 from energywam.models.energy_wam import QwenEnergyWAM
 from energywam.training.shared_microbatch import SharedMicrobatchTrainer
 from energywam.data.energy_dataset import collate_energy_batch
 from k_training_probe import fixed_dataset
 import train as T
 cfg=load_config(a.config);cfg.trainer.fp32_master_weights=False;cfg.trainer.preallocate_optimizer_state=False;cfg.trainer.max_grad_norm=0.;cfg.trainer.directional_head_diagnostics=False
 if a.saved_gpu_budget is not None:cfg.trainer.saved_tensor_gpu_budgets_gib=[a.saved_gpu_budget,a.saved_gpu_budget]
 if a.no_async_offload:cfg.trainer.saved_tensor_async_offload=False
 cfg.trainer.saved_tensor_deduplicate_cpu=a.dedup_cpu
 cfg.trainer.output_dir=str(Path(a.output).parent/'native_gradient_probe_runtime')
 torch.manual_seed(cfg.trainer.seed);model=QwenEnergyWAM(cfg,stats_bundle=T.resolve_training_stats(cfg))
 before_ids={id(p) for p in model.parameters()}
 trainer=SharedMicrobatchTrainer(cfg,model)
 missing=len({id(p) for p in model.parameters()}-before_ids)
 if a.precision=='float32':
  model.backbone._mm.language_model.float();torch.backends.cuda.matmul.allow_tf32=False
 if a.precision=='float16':model.backbone._mm.language_model.half()
 # Placement/dispatch may replace Parameter objects. Build no-op SGD afterwards.
 trainer.optimizer=torch.optim.SGD(model.parameters(),lr=0.);trainer.scheduler=None
 assert {id(p) for p in model.parameters()}=={id(p) for g in trainer.optimizer.param_groups for p in g['params']}
 gen=torch.Generator(device='cuda:0').manual_seed(919)
 with torch.no_grad():
  head=model.core.energy_head.net[-1].weight;head.copy_(torch.randn(head.shape,device=head.device,dtype=head.dtype,generator=gen)*.001)
 ds,windows=fixed_dataset(cfg,model,json.loads(Path(a.manifest).read_text()),'train')
 indices=list(map(int,a.sample_indices.split(','))) if a.sample_indices else [0,len(windows)//2]
 if len(indices)!=a.native_microbatch:raise ValueError('provide exactly native-microbatch sample indices')
 batch=collate_energy_batch([ds[i] for i in indices])
 def weight_sample_hash():
  h=hashlib.sha256()
  for name,parameter in model.named_parameters():h.update(name.encode());h.update(parameter.detach().reshape(-1)[:64].float().cpu().numpy().tobytes())
  return h.hexdigest()
 initial_hash=weight_sample_hash()
 report=dict(weight_changes=0,optimizer='SGD lr0 created after final placement; no master/Adam state for correctness only; capacity tested separately',legacy_preplacement_missing_parameter_count=missing,language_dtype=str(next(model.backbone._mm.language_model.parameters()).dtype),physical_gpus=a.physical_gpus,K=2,effective_batch=len(indices),native_microbatch=a.native_microbatch,math_mode=a.math_mode,reference_math=('highest_fp32' if a.math_mode=='tf32' else a.math_mode),sample_indices=indices,saved_gpu_budgets=cfg.trainer.saved_tensor_gpu_budgets_gib,async_offload=cfg.trainer.saved_tensor_async_offload,head_override_std=.001,initial_sampled_hash=initial_hash,comparisons=[],passed=False)
 path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True)
 directions=a.directions.split(',');assert all(d in ['action','future'] for d in directions)
 for direction in directions:
  os.environ['ENERGYWAM_ONLY_DIRECTION']=direction;reference={};reference_loss=None
  for pass_index,micro in enumerate(([1,1,a.native_microbatch] if a.repeat_reference else [1,a.native_microbatch])):
   if a.math_mode=='tf32':
    if a.precision!='float32':raise ValueError('TF32 probe requires FP32 model')
    torch.set_float32_matmul_precision('highest' if micro==1 else 'high')
    torch.backends.cuda.matmul.allow_tf32=(micro!=1)
   cfg.trainer.shared_microbatch_size=micro;trainer.global_step=0;model.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
   row=trainer.train_step(batch);value=row['loss'];gradients={n:p.grad for n,p in model.named_parameters() if p.grad is not None}
   assert all(torch.isfinite(g).all() for g in gradients.values()),'nonfinite gradient in native microbatch check'
   assert weight_sample_hash()==initial_hash,'no-op probe changed model weights'
   if pass_index==0:reference={n:g.detach().cpu() for n,g in gradients.items()};reference_loss=value
   else:
    assert reference.keys()==gradients.keys();error=norm=norm_new=dot=0.;groups={}
    for n,g in gradients.items():
     ref=reference[n].to(g.device).float();delta=g.float()-ref;d=float(delta.square().sum());v=float(ref.square().sum());error+=d;norm+=v;norm_new+=float(g.float().square().sum());dot+=float((g.float()*ref).sum())
     key='language' if 'language_model' in n else n.split('.')[1];pair=groups.setdefault(key,[0.,0.]);pair[0]+=d;pair[1]+=v
    relative=math.sqrt(error/max(norm,1e-30));bygroup={k:math.sqrt(v[0]/max(v[1],1e-30)) for k,v in groups.items()}
    ok=relative<.02 and abs(value-reference_loss)<max(1e-6,.01*abs(reference_loss)) and all(bygroup[k]<.05 for k,v in groups.items() if math.sqrt(v[1])>1e-8)
    report['comparisons'].append(dict(direction=direction,comparison=('reference_repeat' if micro==1 else f'native{micro}'),loss_micro1=reference_loss,loss_micro2=value,gradient_cosine=dot/max(math.sqrt(norm*norm_new),1e-30),reference_gradient_l2=math.sqrt(norm),absolute_gradient_difference_l2=math.sqrt(error),relative_gradient_l2=relative,group_relative_l2=bygroup,gradient_tensors=len(gradients),passed=ok));path.write_text(json.dumps(report,indent=2));print(json.dumps(report['comparisons'][-1]),flush=True)
    if not ok and not a.record_all:raise RuntimeError('native microbatch gradient mismatch; do not select requested native batch')
   del gradients
  del reference;model.zero_grad(set_to_none=True)
 report['passed']=all(x['passed'] for x in report['comparisons']);path.write_text(json.dumps(report,indent=2))
 if not report['passed']:raise RuntimeError('native microbatch gradient mismatch; predeclared gate failed')
if __name__=='__main__':main()

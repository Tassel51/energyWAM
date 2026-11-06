"""Actual36-layer 4B gradient comparison, storage-only offload on/off, no updates."""
import argparse,json,os,sys,gc,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--manifest',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='2,3':raise RuntimeError('explicit GPU23 isolation required')
 import torch
 from energywam.config import load_config
 from energywam.models.energy_wam import QwenEnergyWAM
 from energywam.training.parallel import shard_language_model
 from energywam.data.energy_dataset import collate_energy_batch
 from k_training_probe import fixed_dataset
 import train as T
 cfg=load_config(a.config);torch.manual_seed(cfg.trainer.seed)
 model=QwenEnergyWAM(cfg,stats_bundle=T.resolve_training_stats(cfg)).to('cuda:0')
 shard_language_model(model,[0,1],torch.device('cuda:0'));model.train()
 generator=torch.Generator(device='cuda:0').manual_seed(919)
 with torch.no_grad():
  head=model.core.energy_head.net[-1].weight
  head.copy_(torch.randn(head.shape,device=head.device,dtype=head.dtype,generator=generator)*1e-3)
 manifest=json.loads(Path(a.manifest).read_text());ds,windows=fixed_dataset(cfg,model,manifest,'train')
 report=dict(weight_updates=0,backbone_layers=len(model.backbone._mm.language_model.layers),physical_gpus='2,3',pack_factor=cfg.framework.dino_patch_pack_factor,
    raw_future_shape=[model.core.num_future_frames,model.core.num_cameras,model.core.future_patches,model.core.future_dim],head_override_std=.001,
    note='Fresh model, nonzero diagnostic head override before both passes; not a training or quality score. Same candidate seed and theta, full outer gradients; first and middle training windows.',comparisons=[],passed=False)
 output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True)
 try:
  for index in sorted({0,len(windows)//2}):
   for role,direction in enumerate(['action','future']):
    reference={};reference_loss=None
    for offload in [True,False]:
     cfg.trainer.offload_saved_tensors=offload;model.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
     torch.manual_seed(2026+windows[index]['window_id']*1009+role)
     batch=collate_energy_batch([ds[index]]);batch['_bi_only']=direction
     result=model(batch);loss=result['action_loss'];loss.backward()
     value=float(loss.detach());gradients={name:p.grad for name,p in model.named_parameters() if p.grad is not None}
     if any(not torch.isfinite(g).all() for g in gradients.values()):raise FloatingPointError('nonfinite gradient')
     if offload:
      reference={name:g.detach().cpu() for name,g in gradients.items()};reference_loss=value
     else:
      assert reference.keys()==gradients.keys(),'parameter gradient presence mismatch'
      difference=norm=0.;max_abs=0.;groups={}
      for name,g in gradients.items():
       ref=reference[name].to(g.device);delta=g.float()-ref.float()
       d=float(delta.square().sum());n=float(ref.float().square().sum());difference+=d;norm+=n
       max_abs=max(max_abs,float(delta.abs().max()))
       group='language' if 'language_model' in name else name.split('.')[1]
       group_values=groups.setdefault(group,[0.,0.]);group_values[0]+=d;group_values[1]+=n
      relative=math.sqrt(difference/max(norm,1e-30))
      group_relative={name:math.sqrt(v[0]/max(v[1],1e-30)) for name,v in groups.items()}
      passed=relative<.01 and abs(value-reference_loss)<=max(1e-6,abs(reference_loss)*1e-5)
      passed &= all(group_relative[name]<.02 for name,v in groups.items() if math.sqrt(v[1])>1e-8)
      row=dict(window_id=windows[index]['window_id'],task=windows[index]['task'],direction=direction,loss_offload=reference_loss,loss_reference=value,
          gradient_parameters=len(gradients),relative_gradient_l2=relative,max_absolute_gradient_error=max_abs,group_relative_l2=group_relative,passed=bool(passed))
      report['comparisons'].append(row);output.write_text(json.dumps(report,indent=2));print(json.dumps(row),flush=True)
      if not passed:raise RuntimeError('offload/reference gradient mismatch; review report')
     del result,loss,batch,gradients
    del reference;model.zero_grad(set_to_none=True)
  report['passed']=True
 except Exception as error:
  report['error_type']=type(error).__name__;output.write_text(json.dumps(report,indent=2));raise
 output.write_text(json.dumps(report,indent=2))
if __name__=='__main__':main()

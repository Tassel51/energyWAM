"""Real 4B, real-data CUDA engineering trial; checkpoint export is opt-in."""
import argparse,json,time,sys,contextlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch
from energywam.config import load_config
from energywam.models.energy_wam import QwenEnergyWAM
from energywam.training.trainer import EnergyWAMTrainer
sys.path.insert(0,str(ROOT/'scripts'))
from train import build_dataloader,resolve_training_stats

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--set',nargs='*',default=[]);p.add_argument('--offload-saved-tensors',action='store_true');p.add_argument('--deduplicate-parameter-offload',action='store_true');p.add_argument('--low-memory-adamw',action='store_true',help='Use the same AdamW with foreach=False to avoid large GPU temporaries');p.add_argument('--steps',type=int,default=1);p.add_argument('--checkpoint',default=None,help='Optional checkpoint filename after all steps succeed');p.add_argument('--out',required=True);a=p.parse_args()
 if a.steps<1:p.error('--steps must be positive')
 cfg=load_config(a.config,overrides=a.set);torch.manual_seed(cfg.trainer.seed)
 def log(stage,**kw):print(json.dumps(dict(stage=stage,time=time.strftime('%Y-%m-%dT%H:%M:%S%z'),**kw)),flush=True)
 log('resource_check',gpu=torch.cuda.get_device_name(),total_gpu_bytes=torch.cuda.get_device_properties(0).total_memory,torch=torch.__version__,save_on_cpu=a.offload_saved_tensors,geometry=dict(H=cfg.data.action_horizon,Da=cfg.data.action_dim,F=cfg.data.num_future_frames),unroll=cfg.solver.unroll_mode,K=cfg.solver.train_inner_steps,lora=cfg.backbone.lora_enabled,tune_language=cfg.backbone.tune_language_model)
 bundle=resolve_training_stats(cfg);log('loading_pretrained_model')
 model=QwenEnergyWAM(cfg,stats_bundle=bundle).to(cfg.trainer.device)
 optimizer=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=cfg.trainer.learning_rate,weight_decay=cfg.trainer.weight_decay,foreach=False) if a.low_memory_adamw else None
 trainer=EnergyWAMTrainer(cfg,model,optimizer=optimizer);trainer.export_config();loader=build_dataloader(cfg,model,smoke=True)
 log('model_loaded',parameters=sum(p.numel() for p in model.parameters()),trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),cuda_allocated=torch.cuda.memory_allocated())
 it=iter(loader);results=[]
 for step in range(a.steps):
  batch=next(it);log('training_step_started',step=step,episodes=batch.get('episode_index'),frames=batch.get('frame_index'))
  torch.cuda.synchronize();t=time.monotonic()
  from energywam.training.offload import parameter_aware_cpu_offload
  storage=parameter_aware_cpu_offload(model) if a.deduplicate_parameter_offload else torch.autograd.graph.save_on_cpu(pin_memory=True) if a.offload_saved_tensors else contextlib.nullcontext()
  with storage:
   metrics=trainer.train_step(batch)
  torch.cuda.synchronize();metrics['synchronized_step_seconds']=time.monotonic()-t;results.append(metrics);log('training_step_complete',**metrics)
 checkpoint=str(trainer.save(a.checkpoint)) if a.checkpoint else None
 report=dict(stage='complete',steps=results,save_on_cpu=a.offload_saved_tensors,deduplicate_parameter_offload=a.deduplicate_parameter_offload,low_memory_adamw=a.low_memory_adamw,checkpoint=checkpoint,description='real-data full4B bounded engineering trial; subset statistics; no closed-loop success-rate claim')
 Path(a.out).write_text(json.dumps(report,indent=2));log('complete',report=a.out)
if __name__=='__main__':main()

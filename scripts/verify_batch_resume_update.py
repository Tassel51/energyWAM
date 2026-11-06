"""One isolated real optimizer update after full resume; source stays unchanged."""
import argparse,os,sys,json,time,subprocess,signal,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
R=Path('/cloud/zzc/energyWAM/experiments/iteration_aware_v1_20261104')
def worker(name,expected):
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0,1'
 import torch
 from energywam.config import load_config
 from energywam.models.energy_wam import QwenEnergyWAM
 from energywam.training.shared_microbatch import SharedMicrobatchTrainer
 from energywam.training.run_state import FixedSampleBudgetSampler
 from energywam.data.energy_dataset import collate_energy_batch
 from k_training_probe import fixed_dataset
 import train as T
 source=R/name/'ckpt_latest.pt';before=(source.stat().st_size,source.stat().st_mtime_ns)
 out=R/f'{name}_resume_update_verification';out.mkdir(exist_ok=True)
 cfg=load_config(ROOT/f'configs/experiments/{name}.yaml');cfg.trainer.output_dir=str(out)
 torch.manual_seed(cfg.trainer.seed);model=QwenEnergyWAM(cfg,stats_bundle=T.resolve_training_stats(cfg))
 trainer=SharedMicrobatchTrainer(cfg,model);start=time.monotonic();meta=trainer.load(source)
 assert trainer.global_step==expected and meta.optimizer_policy==meta.scheduler_policy=='continue'
 def sample_hash():
  h=hashlib.sha256()
  for n,p in model.named_parameters():h.update(n.encode());h.update(p.detach().reshape(-1)[:64].float().cpu().numpy().tobytes())
  return h.hexdigest()
 loaded_hash=sample_hash();load_seconds=time.monotonic()-start
 ds,_=fixed_dataset(cfg,model,json.loads((R/'tinyfit_manifest.json').read_text()),'train');b=cfg.trainer.per_device_batch_size
 order=list(FixedSampleBudgetSampler(len(ds),cfg.trainer.seed,consumed=expected*b,total=(expected+1)*b))
 batch=collate_energy_batch([ds[i] for i in order]);row=trainer.train_step(batch)
 assert trainer.global_step==expected+1 and row.get('param_grad_nonfinite',0)==0 and torch.isfinite(torch.tensor(row['loss']))
 changed_hash=sample_hash();assert loaded_hash!=changed_hash
 after=(source.stat().st_size,source.stat().st_mtime_ns);assert before==after
 report=dict(passed=True,source_checkpoint=str(source),restored_step=expected,verified_next_step=trainer.global_step,source_checkpoint_unchanged=True,source_size=before[0],source_mtime_ns=before[1],optimizer_policy=meta.optimizer_policy,scheduler_policy=meta.scheduler_policy,load_seconds=load_seconds,loaded_parameter_sample_hash=loaded_hash,updated_parameter_sample_hash=changed_hash,loss=row['loss'],parameter_gradient_nonfinite=row.get('param_grad_nonfinite'),next_sample_positions=[expected*b,(expected+1)*b],dataset_indices=order,scope='Isolated full model/AdamW8bit/FP32-master/scheduler/RNG resume and one update. No source rewrite or new checkpoint. This update is excluded from quality comparisons.')
 (out/'result.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
def main(a):
 if a.worker:return worker(a.name,a.step)
 from run_iteration_search import gpu_idle
 from energywam.training.resource_monitor import ResourceMonitor
 if a.wait_confirmation:
  for _ in range(1440):
   confirmation=R/'batch_expanded_validation'/a.name/'resource_summary.json'
   try:ready=json.loads(confirmation.read_text()).get('exit_classification')=='completed'
   except (OSError,ValueError):ready=False
   if ready and gpu_idle('0,1'):break
   time.sleep(15)
  else:raise RuntimeError('Confirmation not terminal within wait budget; no work interrupted')
 assert gpu_idle('0,1'),'do not interrupt an occupied GPU pair'
 out=R/f'{a.name}_resume_update_verification';out.mkdir(exist_ok=True)
 with (out/'console.log').open('a') as log:
  child=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),a.name,'--step',str(a.step),'--worker'],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',ENERGYWAM_SAMPLE_ALIGNED_RNG='1',ENERGYWAM_FIXED_DATA_CACHE='1',ENERGYWAM_TRIM_BETWEEN_DIRECTIONS='0'),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  monitor=ResourceMonitor(child.pid,['0','1'],out);monitor.start();started=time.monotonic();guard=None
  while child.poll() is None:
   if time.monotonic()-started>600:guard='10minute full-resume verification budget'
   if monitor.summary.get('last_sample',{}).get('process_uss_gib',0)>80:guard='80GiB owned USS full-resume verification budget'
   if guard:os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=55);break
   time.sleep(2)
  monitor.finish(child.returncode,guard)
  if child.returncode:raise RuntimeError(f'Resume verification failed; see {out}/console.log')
 print((out/'result.json').read_text())
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('--step',type=int,required=True);p.add_argument('--worker',action='store_true');p.add_argument('--wait-confirmation',action='store_true');main(p.parse_args())

"""Read-only expanded validation after the finite batch and native-gradient jobs."""
import os,sys,json,time,subprocess,signal,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
R=Path('/cloud/zzc/energyWAM/experiments/iteration_aware_v1_20261104')
def read(p):
 try:return json.loads(p.read_text())
 except (OSError,ValueError):return {}
def worker(name):
 os.environ.update(ENERGYWAM_PROBE_MAX_K='2',ENERGYWAM_PROBE_SPLITS='val',ENERGYWAM_FIXED_DATA_CACHE='1')
 import torch
 from types import SimpleNamespace
 from energywam.config import load_config
 from energywam.models.energy_wam import QwenEnergyWAM
 from energywam.training.checkpoint import restore
 from energywam.training.parallel import shard_language_model
 from k_training_probe import evaluate,dataset_baselines
 import train as T
 cfg=load_config(ROOT/f'configs/experiments/{name}.yaml');torch.manual_seed(42)
 model=QwenEnergyWAM(cfg,stats_bundle=T.resolve_training_stats(cfg));device=torch.device(cfg.trainer.device)
 model.to(device);shard_language_model(model,cfg.trainer.language_model_devices,device)
 payload=torch.load(R/name/'ckpt_latest.pt',map_location='cpu',mmap=True,weights_only=False)
 meta=restore(payload,model=model,optimizer=None,scheduler=None,mode='resume',restore_rng=False)
 expected=800 if name.startswith('batch2_') else 400
 assert meta.global_step==expected;del payload
 class NoOptimizer:
  def zero_grad(self,**kw):pass
 trainer=SimpleNamespace(cfg=cfg,model=model,optimizer=NoOptimizer(),global_step=meta.global_step)
 model.requires_grad_(False);model.eval();out=R/'batch_expanded_validation'/name;out.mkdir(parents=True,exist_ok=True)
 baselines=dataset_baselines(cfg,model,R/'batch_confirmation_manifest.json',out)
 summary=dict(baselines=baselines,weight_updates=0,checkpoint_step=meta.global_step,scope='64 fixed windows from16 new validation episodes of the same2 tasks; no test data, no new training seed.',metrics={})
 for rank,direction in enumerate(['action','future']):
  evaluate(trainer,R/'batch_confirmation_manifest.json',out,rank,evaluation_seed=2026)
  report=read(out/f'eval_{meta.global_step:04d}_{direction}.json');assert report['weight_updates_during_eval']==0
  rows=[x for x in report['rows'] if x['k']==2];assert len(rows)==64
  summary['metrics'][direction]=sum(x['prediction_mse'] for x in rows)/len(rows)
  (out/'summary.json').write_text(json.dumps(summary,indent=2))
 print(summary,flush=True)
def controller():
 from run_iteration_search import gpu_idle
 from energywam.training.resource_monitor import ResourceMonitor
 # Native diagnostic owns first opportunity on a free pair; wait for that finite work.
 for _ in range(1440):
  plan=read(R/'batch_size_comparison_plan.json');p=plan.get('native_microbatch_preflight',{});status=p.get('status','')
  native_done=status in ['native2_not_selected_pending_review','complete_native1_selected_BF16_native2_gate_failed']
  if status=='corrected_BF16_gradient_passed_profiles_queued':
   native_done=all(any(read(R/f'search_status_{pair}.json').get(j['name'],{}).get('phase') in ['complete','failed'] for pair in ['01','23']) for j in p['profile_jobs'])
  states={}
  for pair,b in [('01',2),('23',4)]:
   n=f'batch{b}_tiny16_samples1600_seed42';states[n]=read(R/f'search_status_{pair}.json').get(n,{})
  if native_done and states['batch2_tiny16_samples1600_seed42'].get('phase') in ['complete','failed'] and gpu_idle('0,1'):break
  time.sleep(15)
 else:raise RuntimeError('Confirmation wait budget exhausted; no jobs interrupted')
 for name in states:
  pair='01' if name.startswith('batch2_') else '23'
  for _ in range(1440):
   state=read(R/f'search_status_{pair}.json').get(name,{})
   if state.get('phase') in ['complete','failed'] and gpu_idle('0,1'):break
   time.sleep(15)
  else:raise RuntimeError('Quality trial or GPU remained busy; no training interrupted')
  if state.get('phase')!='complete':continue
  out=R/'batch_expanded_validation'/name;out.mkdir(parents=True,exist_ok=True)
  if read(out/'summary.json').get('metrics',{}).keys()>={'action','future'} and read(out/'resource_summary.json').get('exit_classification')=='completed':continue
  with (out/'console.log').open('a') as log:
   child=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--worker',name],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false'),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   monitor=ResourceMonitor(child.pid,['0','1'],out);monitor.start();start=time.time();guard=None
   while child.poll() is None:
    if time.time()-start>1800:guard='Read-only confirmation30minute budget';os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=55);break
    time.sleep(2)
   monitor.finish(child.returncode,guard)
   if child.returncode:raise RuntimeError(f'{name} confirmation failed; inspect {out}/console.log')
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--worker');args=parser.parse_args()
 if args.worker:worker(args.worker)
 else:controller()

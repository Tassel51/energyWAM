"""Bounded next phase: owned trials finish, preserve reference, same-pair probes."""
import json,time,subprocess,statistics,os
from pathlib import Path
from run_iteration_search import gpu_idle
R=Path('/cloud/zzc/energyWAM/experiments/iteration_aware_v1_20261104');PY='/usr/local/miniconda3/envs/py312/bin/python'
def service():
 return subprocess.run(['supervisorctl','status','energywam-iteration-search23'],capture_output=True,text=True).stdout

def enqueue(jobs):
 while 'RUNNING' in service():time.sleep(5)
 if 'EXITED' not in service():raise RuntimeError('Controller not in expected completed state; no automatic override')
 p=R/'search_jobs_23.json';queue=json.loads(p.read_text())
 for job in jobs:
  if not any(x['name']==job['name'] for x in queue):queue.append(job)
 tmp=p.with_suffix('.next.tmp');tmp.write_text(json.dumps(queue,indent=2));tmp.replace(p)
 subprocess.run(['supervisorctl','start','energywam-iteration-search23'],check=True)

def main():
 parents={'01':'episode10_prior_tokenscale_seed43','23':'episode10_prior_tokenscale_seed42'}
 while True:
  states={pair:json.loads((R/f'search_status_{pair}.json').read_text()).get(name,{}) for pair,name in parents.items()}
  if any(s.get('phase')=='failed' for s in states.values()):raise RuntimeError('Parent failed; requires inspection')
  finished=all(s.get('phase')=='complete' for s in states.values())
  finished &= all((R/name/'final_report.json').exists() and json.loads((R/name/'final_report.json').read_text())['global_step']>=200 for name in parents.values())
  if finished and gpu_idle('0,1') and gpu_idle('2,3'):break
  time.sleep(15)
 subprocess.run([PY,'scripts/retire_completed_tinyfit_reference.py'],check=True)
 jobs=json.loads((R/'pack_profile_prepared_jobs.json').read_text());enqueue(jobs)
 while True:
  state=json.loads((R/'search_status_23.json').read_text());rows=[state.get(j['name'],{}) for j in jobs]
  if any(x.get('phase')=='failed' for x in rows):raise RuntimeError('Performance probe failed; no next training launched')
  if all(x.get('phase')=='complete' for x in rows):break
  time.sleep(15)
 reports=[];hashes=[]
 for job in jobs:
  folder=R/job['name'];initial=json.loads((folder/'initialization_step0.json').read_text());hashes.append(initial['common_parameter_sampled_sha256'])
  data=[json.loads(line) for line in (folder/'timing_history.jsonl').read_text().splitlines()]
  steady=[x for x in data if x['global_step_completed']>=3]
  reports.append(dict(name=job['name'],median_update_seconds=statistics.median(x['end_to_end_seconds'] for x in steady),
    process_lifetime_peak_rss_gib=max(x['process_rss_peak_gib'] for x in data),
    gpu0_peak_allocated_gib=max(x['gpu0_peak_allocated_gib'] for x in data),
    gpu1_peak_allocated_gib=max(x['gpu1_peak_allocated_gib'] for x in data)))
 assert hashes[0]==hashes[1],'common initial parameter sample mismatch'
 ratio=reports[0]['median_update_seconds']/reports[1]['median_update_seconds']
 result=dict(common_parameter_sampled_sha256=hashes[0],full_parameter_hash=False,skip_warmup_updates=2,physical_gpus='2,3',reports=reports,measured_speed_ratio=ratio,
   note='Architecture tokenization ablation, not equivalent model. Same original future dimensions. RSS includes startup; GPU numbers are per-process torch allocated, not total driver memory. Eight-update profiles do not assess final quality.')
 (R/'pack_profile_comparison.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
 if ratio<1.2:print('Insufficient speed evidence; no automatic quality trial',flush=True);return
 while not gpu_idle('2,3'):time.sleep(15)
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='2,3',ENERGYWAM_FIXED_DATA_CACHE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
 subprocess.run([PY,'-u','scripts/check_pack_offload_gradients.py','--config','configs/experiments/iteration_tinyfit_master_prior_pack2_r2.yaml','--manifest',str(R/'tinyfit_manifest.json'),'--output',str(R/'pack_offload_gradient_comparison.json')],env=env,check=True)
 assert json.loads((R/'pack_offload_gradient_comparison.json').read_text())['passed']
 job=dict(name='tinyfit_master_prior_pack2_r2',config='configs/experiments/iteration_tinyfit_master_prior_pack2_r2.yaml',manifest=str(R/'tinyfit_manifest.json'),steps=40,eval_interval=10,physical_gpus='2,3')
 enqueue([job])
 (R/'pack2_prepared_plan.json').write_text(json.dumps(dict(status='quality_trial_queued',job=job,measured_speed_ratio=ratio,success_claim=False),indent=2))
if __name__=='__main__':main()

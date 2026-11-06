"""Continue authorized batch study after bounded controls, touching owned jobs only."""
import os,sys,json,time,subprocess,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from run_iteration_search import gpu_idle
from energywam.training.resource_monitor import ResourceMonitor
R=Path('/cloud/zzc/energyWAM/experiments/iteration_aware_v1_20261104');PY='/usr/local/miniconda3/envs/py312/bin/python'
def read(p):
 try:return json.loads(p.read_text())
 except (OSError,ValueError):return {}
def terminal(name):
 s=read(R/'search_status_23.json').get(name,{})
 if s.get('phase') not in ['complete','failed']:return False
 try:os.kill(s['pid'],0)
 except ProcessLookupError:return True
 return False
def wait_for(check,label):
 for _ in range(720):
  if check():return
  print('Waiting:',label,flush=True);time.sleep(15)
 raise RuntimeError('Observation deadline; no process was restarted')
def update(**values):
 p=R/'batch_size_comparison_plan.json';plan=read(p);plan['native_microbatch_preflight'].update(values);p.write_text(json.dumps(plan,indent=2));return plan
control='batch2_tiny16_headlr0003_samples320_seed42'
wait_for(lambda:terminal(control),'bounded B2 recheck')
if read(R/'search_status_23.json')[control]['phase']=='complete':
 wait_for(lambda:(R/control/'model_reference.pt.meta.json').exists() and not (R/control/'ckpt_latest.pt').exists(),'bounded B2 model preservation')
# These two bounded controls are not long-run finalists. Preserve weights, free optimizer space.
s=Path('scripts/preserve_completed_iteration_pilot.py').read_text().replace("[('iter_learned',20)]","[('batch8_tiny16_headlr0003_samples320_seed42',40)]")
s=s.replace('Completed20-update pilot, supervisor confirmed EXITED; preserve exact model and all logs; reclaim optimizer for user-authorized batch4 comparison','Completed bounded B8 recheck; exact model/logs retained, optimizer retired for batch4 rolling headroom')
script=R/'preserve_bounded_batch8_reference.py';script.write_text(s)
if read(R/'search_status_23.json').get('batch8_tiny16_headlr0003_samples320_seed42',{}).get('phase')=='complete':subprocess.run([PY,str(script)],cwd=ROOT,check=True)
wait_for(lambda:gpu_idle('2,3'),'GPU23 idle for gradient check')
plan=update(status='actual4B_gradient_check_running')
folder=R/'native_microbatch_gradient_probe';folder.mkdir(exist_ok=True)
command=[PY,'-u','scripts/check_native_microbatch_gradients.py','--config',plan['native_microbatch_preflight']['profile_jobs'][0]['config'],'--manifest',str(R/'tinyfit_manifest.json'),'--output',str(R/'native_microbatch_gradient_check.json')]
env=dict(os.environ,CUDA_VISIBLE_DEVICES='2,3',ENERGYWAM_SAMPLE_ALIGNED_RNG='1',ENERGYWAM_FIXED_DATA_CACHE='1',ENERGYWAM_TRIM_BETWEEN_DIRECTIONS='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
with (folder/'console.log').open('a') as log:
 child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 monitor=ResourceMonitor(child.pid,[2,3],folder);monitor.start();code=child.wait();monitor.finish(code)
passed=code==0 and read(R/'native_microbatch_gradient_check.json').get('passed',False)
plan=update(status='gradient_passed_profiles_queued' if passed else 'gradient_failed_native2_not_selected',gradient_returncode=code)
if passed:
 jobs_path=R/'search_jobs_23.json';jobs=read(jobs_path)
 for job in plan['native_microbatch_preflight']['profile_jobs']:
  if not any(x['name']==job['name'] for x in jobs):jobs.append(job)
 jobs_path.write_text(json.dumps(jobs,indent=2));subprocess.run(['supervisorctl','start','energywam-iteration-search23'],check=True)
 wait_for(lambda:all(terminal(j['name']) for j in plan['native_microbatch_preflight']['profile_jobs']),'native capacity profiles')
 update(status='profiles_terminal_review_pending')
wait_for(lambda:gpu_idle('2,3'),'GPU23 idle for batch4')
plan=read(R/'batch_size_comparison_plan.json');job=plan['batch4_candidate']['job'];q=R/'search_jobs_23.json';jobs=read(q)
if not any(x['name']==job['name'] for x in jobs):jobs.append(job)
q.write_text(json.dumps(jobs,indent=2));plan['batch4_candidate']['status']='queued';(R/'batch_size_comparison_plan.json').write_text(json.dumps(plan,indent=2))
(R/'batch4_progress_plan.json').write_text(json.dumps(dict(jobs=[job],sample_budget_each=1600),indent=2))
subprocess.run(['supervisorctl','start','energywam-iteration-search23'],check=True)
with (ROOT/'logs/batch4_progress_observer.log').open('a') as log:subprocess.Popen([PY,'-u','scripts/observe_batch_progress.py','--plan','batch4_progress_plan.json'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print('Batch4 paired trial started; no Stage2 queued.',flush=True)

"""Finite diagnostic retry on a terminal quality pair, never interrupt training."""
import os,sys,json,time,subprocess,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from run_iteration_search import gpu_idle
from energywam.training.resource_monitor import ResourceMonitor
R=Path('/cloud/zzc/energyWAM/experiments/iteration_aware_v1_20261104');PY='/usr/local/miniconda3/envs/py312/bin/python'
def read(p):
 try:return json.loads(p.read_text())
 except (OSError,ValueError):return {}
def update(**values):
 p=R/'batch_size_comparison_plan.json';plan=read(p);plan['native_microbatch_preflight'].update(values);p.write_text(json.dumps(plan,indent=2));return plan
selected=None
for _ in range(960):
 for pair,name in [('0,1','batch2_tiny16_samples1600_seed42'),('2,3','batch4_tiny16_samples1600_seed42')]:
  s=read(R/f"search_status_{pair.replace(',','')}.json").get(name,{})
  if s.get('phase') not in ['complete','failed']:continue
  try:os.kill(s['pid'],0)
  except ProcessLookupError:
   if gpu_idle(pair):selected=pair;break
 if selected:break
 print('Waiting for a terminal, idle quality GPU pair',flush=True);time.sleep(15)
else:raise RuntimeError('No free owned pair within diagnostic observation budget')
plan=update(status='corrected_probe_running',recheck_physical_gpus=selected)
base_config=plan['native_microbatch_preflight']['profile_jobs'][0]['config']
def run(precision,directions):
 folder=R/f'native_microbatch_recheck_{precision}';folder.mkdir(exist_ok=True);out=R/f'native_microbatch_gradient_recheck_{precision}.json'
 cmd=[PY,'-u','scripts/check_native_microbatch_gradients.py','--config',base_config,'--manifest',str(R/'tinyfit_manifest.json'),'--output',str(out),'--physical-gpus',selected,'--precision',precision,'--directions',directions]
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=selected,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',ENERGYWAM_FIXED_DATA_CACHE='1',ENERGYWAM_TRIM_BETWEEN_DIRECTIONS='0')
 guard=None
 with (folder/'console.log').open('a') as log:
  child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  monitor=ResourceMonitor(child.pid,selected.split(','),folder);monitor.start()
  started=time.time()
  while child.poll() is None:
   if monitor.summary.get('last_sample',{}).get('process_uss_gib',0)>80:guard='diagnostic process USS exceeded80GiB working-set budget'
   if time.time()-started>1200:guard='diagnostic20minute observation limit'
   if guard:os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=55);break
   time.sleep(1)
  monitor.finish(child.returncode,guard)
 return child.returncode,read(out)
code,report=run('configured','action,future')
if code==0 and report.get('passed'):
 plan=update(status='corrected_BF16_gradient_passed_profiles_queued',corrected_probe_report='native_microbatch_gradient_recheck_configured.json')
 jobs_path=R/f"search_jobs_{selected.replace(',','')}.json";jobs=read(jobs_path)
 for template in plan['native_microbatch_preflight']['profile_jobs']:
  job=dict(template,physical_gpus=selected)
  if not any(x['name']==job['name'] for x in jobs):jobs.append(job)
 jobs_path.write_text(json.dumps(jobs,indent=2))
 subprocess.run(['supervisorctl','start',f"energywam-iteration-search{selected.replace(',','')}"],check=True)
else:
 update(status='corrected_BF16_failed_FP32_action_diagnostic',corrected_probe_report='native_microbatch_gradient_recheck_configured.json')
 fp32code,fp32=run('float32','action')
 update(status='native2_not_selected_pending_review',fp32_diagnostic_returncode=fp32code,fp32_action_diagnostic_passed=bool(fp32.get('passed')),fp32_probe_report='native_microbatch_gradient_recheck_float32.json')
 print('No capacity profile selected from a failed BF16 equivalence gate.',flush=True)

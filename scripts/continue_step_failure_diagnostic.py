"""Bounded continuation: archive completed80 control before paired80 learned arm."""
import pathlib,json,time,subprocess,os
R=pathlib.Path('/cloud/zzc/energyWAM/experiments/iteration_aware_v1_20261104')
name='no_prior_currentinit_fixed_headlr0001_diag_seed42'
for _ in range(180):
 s=json.loads((R/'search_status_23.json').read_text()).get(name,{})
 if s.get('phase')=='complete':
  try:os.kill(s['pid'],0)
  except ProcessLookupError:break
 if s.get('phase')=='failed':raise RuntimeError('bounded fixed diagnostic failed; inspect before continuation')
 print(json.dumps({'waiting_for':name,'phase':s.get('phase'),'completed_updates':s.get('completed_updates')}),flush=True)
 time.sleep(30)
else:raise RuntimeError('diagnostic observation deadline; do not restart training')
py='/usr/local/miniconda3/envs/py312/bin/python'
subprocess.run([py,'scripts/preserve_finished_step_diagnostic.py','--name',name],check=True)
p=R/'future_only_and_learned_steps_plan.json';plan=json.loads(p.read_text());job=plan['bounded_failure_diagnostic']['jobs'][1]
pjobs=R/'search_jobs_23.json';jobs=json.loads(pjobs.read_text())
if not any(x['name']==job['name'] for x in jobs):jobs.append(job)
pjobs.write_text(json.dumps(jobs,indent=2));plan['bounded_failure_diagnostic']['status']='learned_arm_queued_after_fixed80';p.write_text(json.dumps(plan,indent=2))
subprocess.run(['supervisorctl','start','energywam-iteration-search23'],check=True)
print('paired learned80 started',flush=True)

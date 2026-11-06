"""Bounded paired follow-up, no restart of a failed trial."""
from pathlib import Path
import json,time,subprocess,os
R=Path('/cloud/zzc/energyWAM/experiments/iteration_aware_v1_20261104');name='batch8_tiny16_headlr0003_samples320_seed42'
for _ in range(240):
 s=json.loads((R/'search_status_23.json').read_text()).get(name,{})
 if s.get('phase') in ['complete','failed']:
  try:os.kill(s['pid'],0)
  except ProcessLookupError:break
 print(json.dumps({'waiting_for':name,'phase':s.get('phase'),'updates':s.get('completed_updates')}),flush=True);time.sleep(15)
else:raise RuntimeError('observation deadline; no training restarted')
p=R/'batch_size_comparison_plan.json';plan=json.loads(p.read_text());job=plan['failure_recheck']['jobs'][1]
q=R/'search_jobs_23.json';jobs=json.loads(q.read_text())
if not any(x['name']==job['name'] for x in jobs):jobs.append(job)
q.write_text(json.dumps(jobs,indent=2));plan['failure_recheck']['status']='batch2_queued_after_batch8_terminal';p.write_text(json.dumps(plan,indent=2))
subprocess.run(['supervisorctl','start','energywam-iteration-search23'],check=True)

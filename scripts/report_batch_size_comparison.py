"""Matched sample-budget results; resource scope and failures stay explicit."""
import json,statistics
from pathlib import Path
R=Path('/cloud/zzc/energyWAM/experiments/iteration_aware_v1_20261104')
def read(path):
 try:return json.loads(path.read_text())
 except (OSError,ValueError):return {}
def lines(path):
 if not path.exists():return []
 out=[]
 for line in path.read_text().splitlines():
  try:out.append(json.loads(line))
  except ValueError:pass
 return out
def main():
 plan=read(R/'batch_size_comparison_plan.json');runs=[]
 jobs=[dict(j,comparison_group='initial_headlr0001') for j in plan['jobs']]
 if plan.get('batch4_candidate',{}).get('job'):
  jobs.append(dict(plan['batch4_candidate']['job'],comparison_group='initial_headlr0001'))
 jobs += [dict(j,comparison_group='lower_headlr00003') for j in plan.get('failure_recheck',{}).get('jobs',[])]
 for job in jobs:
  f=R/job['name'];cfg=read(f/'resolved_config.json');batch=cfg.get('trainer',{}).get('per_device_batch_size',job.get('effective_batch',2 if job['physical_gpus']=='0,1' else 8))
  points={};progress=read(f/'progress_summary.json').get('evaluation_checkpoints',{})
  for direction in ['action','future']:
   for file in sorted(f.glob(f'eval_*_{direction}.json')):
    report=read(file)
    if not report:continue
    assert report['weight_updates_during_eval']==0
    rows=report['rows'];step=int(file.name.split('_')[1]);sample=step*batch
    point=points.setdefault(sample,dict(sample_visits=sample,optimizer_updates=step))
    for split in ['train','val']:
     values=[x['prediction_mse'] for x in rows if x['k']==2 and x['split']==split]
     if values:point[f'{split}_{direction}']=statistics.mean(values)
    point.update(progress.get(str(step),{}))
  points={s:p for s,p in points.items() if all(k in p for k in ['train_action','train_future','val_action','val_future'])}
  history={int(row['global_step']):row for row in lines(f/'timing_history.jsonl')}
  steady=[v['step_seconds']/batch for k,v in history.items() if (k+1)*batch>32 and 'step_seconds' in v]
  resource=read(f/'resource_summary.json');state=read(R/f"search_status_{job['physical_gpus'].replace(',','')}.json").get(job['name'],{})
  trace=lines(f/'resource_trace.jsonl');cpu=[v['process_cpu_percent_one_core'] for v in trace if 'process_cpu_percent_one_core' in v]
  init=read(f/'initialization_step0.json');last=max(points,default=0);baseline=points.get(0,{})
  improvement={key:1-points[last][key]/baseline[key] for key in ['train_action','train_future','val_action','val_future'] if key in baseline and baseline[key]>0} if points else {}
  runs.append(dict(name=job['name'],comparison_group=job['comparison_group'],head_learning_rate=cfg.get('trainer',{}).get('energy_head_learning_rate'),effective_batch=batch,microbatch=1,physical_gpus=job['physical_gpus'],phase=state.get('phase'),failure_reason=state.get('reason'),
   completed_sample_visits=(max(history,default=-1)+1)*batch,matched_sample_trajectory=points,latest_improvement_over_own_initial=improvement,
   steady_median_seconds_per_sample=statistics.median(steady) if steady else None,
   mean_process_cpu_cores=statistics.mean(cpu)/100 if cpu else None,resource_summary=resource,
   initial_parameter_sample_hash=init.get('sampled_parameter_sha256'),stats_fingerprint=init.get('stats_fingerprint')))
 groups={}
 for group in {run['comparison_group'] for run in runs}:
  members=[run for run in runs if run['comparison_group']==group]
  pairwise=[]
  for i,left in enumerate(members):
   for right in members[i+1:]:pairwise.append(dict(left=left['name'],right=right['name'],common_sample_checkpoints=sorted(set(left['matched_sample_trajectory'])&set(right['matched_sample_trajectory']))))
  groups[group]=dict(runs=[run['name'] for run in members],common_sample_checkpoints=sorted(set.intersection(*(set(run['matched_sample_trajectory']) for run in members))),pairwise=pairwise)
 common=groups.get('initial_headlr0001',{}).get('common_sample_checkpoints',[])
 result=dict(status='running' if any(x['phase'] not in ['complete','failed'] for x in runs) else 'terminal',runs=runs,comparisons=groups,common_sample_checkpoints=common,
 interpretation='Sample visits are the primary comparison axis, not update count. Wall-time points come from an external5s observer; initial eval may predate observer. CPU process memory and shared-container memory/cache are distinct. GPU totals refer to assigned cards. No global optimum or success-rate claim.')
 (R/'batch_size_comparison_results.json').write_text(json.dumps(result,indent=2));print(json.dumps({'status':result['status'],'common_sample_checkpoints':common,'runs':[{k:x[k] for k in ['name','phase','completed_sample_visits','steady_median_seconds_per_sample']} for x in runs]},indent=2))
if __name__=='__main__':main()

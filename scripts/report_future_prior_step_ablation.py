"""Read-only paired reports; do not select by a single best validation point."""
import json
import statistics
from pathlib import Path
R=Path('/cloud/zzc/energyWAM/experiments/iteration_aware_v1_20261104')
PAIRS={
 'future_only_vs_both':['ablation_prior1_currentinit1_seed42','future_only_currentinit_learned_seed42'],
 'learned_vs_fixed_without_prior':['ablation_prior0_currentinit1_seed42','no_prior_currentinit_fixed_seed42'],
 'lower_head_lr_bounded_diagnostic':['no_prior_currentinit_learned_headlr0001_diag_seed42',
                                    'no_prior_currentinit_fixed_headlr0001_diag_seed42'],
}
def collect(name):
 folder=R/name;points={};distribution={}
 for direction in ['action','future']:
  for path in sorted(folder.glob(f'eval_*_{direction}.json')):
   report=json.loads(path.read_text())
   assert report['weight_updates_during_eval']==0
   for split in ['train','val']:
    rows=[x for x in report['rows'] if x['k']==2 and x['split']==split]
    if not rows:continue
    step=rows[0]['update'];key=f'{split}_{direction}'
    points.setdefault(step,{})[key]=sum(x['prediction_mse'] for x in rows)/len(rows)
    episodes={}
    for row in rows:episodes.setdefault(row['episode_index'],[]).append(row['prediction_mse'])
    distribution.setdefault(step,{})[key]=dict(
        median=statistics.median(row['prediction_mse'] for row in rows),
        maximum=max(row['prediction_mse'] for row in rows),
        episode_means={ep:statistics.mean(values) for ep,values in episodes.items()},
        worst_windows=[{key:row[key] for key in ['episode_index','frame_index','prediction_mse']}
                       for row in sorted(rows,key=lambda row:row['prediction_mse'],reverse=True)[:3]])
 log=folder/'train_log.jsonl';updates={}
 if log.exists():
  for line in log.read_text().splitlines():
   row=json.loads(line);updates[int(row['global_step'])]=row
 steady=[row['step_seconds'] for step,row in updates.items() if step>=2 and 'step_seconds' in row]
 baseline=points.get(0,{})
 improvements={step:{key:1-value/baseline[key] for key,value in values.items()
                     if key in baseline and baseline[key]>0} for step,values in points.items()}
 latest_steps=sorted(s for s,v in points.items() if len(v)==4 and s>0)[-3:]
 recent={key:statistics.median(points[s][key] for s in latest_steps) for key in baseline} if latest_steps else {}
 alphas={key:value for key,value in updates[max(updates)].items() if key in
         ['diag/alpha_action_after_update','diag/alpha_future_after_update','diag/alpha_action','diag/alpha_future']} if updates else {}
 state={}
 for pair in ['01','23']:
  status=R/f'search_status_{pair}.json'
  if status.exists():state.update(json.loads(status.read_text()).get(name,{}))
 return dict(name=name,complete=(folder/'final_report.json').exists(),trajectory=points,
             per_window_distribution=distribution,
             process_phase=state.get('phase'),early_stop_reason=state.get('reason'),
             improvement_over_own_initial=improvements,recent_three_check_median=recent,
             completed_updates=max(updates,default=-1)+1,step_size_diagnostics=alphas,
             steady_median_update_seconds=statistics.median(steady) if steady else None)
def main():
 result={}
 for label,names in PAIRS.items():
  runs=[collect(n) for n in names]
  common=sorted(step for step in set(runs[0]['trajectory'])&set(runs[1]['trajectory'])
                if len(runs[0]['trajectory'][step])==4 and len(runs[1]['trajectory'][step])==4)
  result[label]=dict(runs=runs,common_update_points=common,
   interpretation='Conditional single-seed comparison; report training and validation separately. Positive learning requires beating own untrained K2 baseline, not merely random K0. Prior ablation has different initial action baseline by construction.')
 target=R/'future_only_and_learned_steps_results.json';target.write_text(json.dumps(result,indent=2))
 print(json.dumps({k:{'complete':[r['complete'] for r in v['runs']],'latest_common':max(v['common_update_points'],default=None)} for k,v in result.items()},indent=2))
if __name__=='__main__':main()

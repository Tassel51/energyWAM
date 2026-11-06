"""Read-only, sample-matched report for the current ten-task FP32 batch grid."""
import argparse
import json
import math
import statistics
from pathlib import Path

DEFAULT=Path('/cloud/zzc/energyWAM/experiments/batch_grid_micro4_20261105')

def read(path,default=None):
    try:return json.loads(Path(path).read_text())
    except (OSError,ValueError):return default

def jsonlines(path):
    try:lines=Path(path).read_text().splitlines()
    except OSError:return []
    rows=[]
    for line in lines:
        try:rows.append(json.loads(line))
        except ValueError:continue
    return rows

def sampled_resource_means(folder):
    """Periodic-sample means, including loading, validation and checkpoint I/O."""
    rows=jsonlines(folder/'resource_trace.jsonl')
    def metrics(items,keys):
        out={}
        for key in keys:
            values=[x[key] for x in items if isinstance(x.get(key),(int,float)) and math.isfinite(x[key])]
            if values:out[key]=dict(mean=statistics.mean(values),valid_samples=len(values))
        return out
    gpus={}
    for row in rows:
        for gpu in row.get('gpus',[]):gpus.setdefault(str(int(gpu['index'])),[]).append(gpu)
    return dict(periodic_samples=len(rows),
                process=metrics(rows,['process_cpu_percent_one_core','process_rss_gib','process_uss_gib']),
                gpus={index:metrics(items,['memory_used_mib','utilization_percent','power_w']) for index,items in gpus.items()},
                scope='Arithmetic means of periodic resource samples over the observed run, including loading/validation/checkpoint I/O. Not kernel timing, energy measurement or a final throughput comparison; CPU100%=one logical core.')

def training_stability(folder,batch):
    rows={}
    for row in jsonlines(folder/'train_log.jsonl'):
        if 'global_step_completed' in row:step=int(row['global_step_completed'])
        elif isinstance(row.get('global_step'),(int,float)) and math.isfinite(row['global_step']):step=int(round(row['global_step']))+1
        else:continue
        rows[step]=row
    norms=[(step,row['param_grad_norm']) for step,row in rows.items() if isinstance(row.get('param_grad_norm'),(int,float)) and math.isfinite(row['param_grad_norm'])]
    peaks=sorted(norms,key=lambda x:x[1],reverse=True)[:8]
    return dict(logged_updates=len(rows),parameter_gradient_norm_before_clip_median=statistics.median([v for _,v in norms]) if norms else None,
                largest_pre_clip_gradient_events=[dict(update=step,sample_visits=step*batch,gradient_norm=value,train_action_mse=rows[step].get('bi/action_loss'),train_future_mse=rows[step].get('bi/future_loss')) for step,value in peaks],
                scope='Finite pre-clipping gradient norms, not update sizes or an automatic instability verdict. Read alongside fixed-window errors and exit/guard records; clipping can limit an individual update without guaranteeing stable learning.')

def resource_sessions(folder,resource):
    sessions=[]
    for row in jsonlines(folder/'resource_trace.jsonl'):
        if not all(isinstance(row.get(k),(int,float)) and math.isfinite(row[k]) for k in ['unix','elapsed_seconds']):continue
        start=row['unix']-row['elapsed_seconds']
        matches=[s for s in sessions if abs(s['start_unix']-start)<.01]
        if matches:matches[0]['end_unix']=max(matches[0]['end_unix'],row['unix'])
        else:sessions.append(dict(start_unix=start,end_unix=row['unix']))
    if resource.get('started_unix') is not None:
        start=resource['started_unix'];end=resource.get('ended_unix',resource.get('last_sample',{}).get('unix',start))
        matches=[s for s in sessions if abs(s['start_unix']-start)<.01]
        if matches:matches[0]['end_unix']=max(matches[0]['end_unix'],end)
        else:sessions.append(dict(start_unix=start,end_unix=end))
    return sorted(sessions,key=lambda s:s['start_unix'])

def annotate_times(points,timings,sessions,resource):
    for point in points.values():
        count=int(point['updates'])
        times=[timings[s]['end_to_end_seconds'] for s in range(1,count+1) if s in timings and math.isfinite(timings[s]['end_to_end_seconds'])]
        point['cumulative_unique_logged_step_seconds']=sum(times) if len(times)==count else None
        point['active_session_wall_seconds_to_evaluation']=None
        point['wall_seconds_from_first_session_to_evaluation']=None
        written=point['evaluation_written_unix']
        if sessions and written>=sessions[0]['start_unix']:
            point['wall_seconds_from_first_session_to_evaluation']=written-sessions[0]['start_unix']
            intervals=[dict(s) for s in sessions]
            if resource.get('returncode') is None:
                intervals[-1]['end_unix']=max(intervals[-1]['end_unix'],written)
            point['active_session_wall_seconds_to_evaluation']=sum(max(0,min(s['end_unix'],written)-s['start_unix']) for s in intervals)

def paired_episode_bootstrap(left,right,replicates=10000,seed=20261105):
    """Uncertainty over held-out episodes; NOT over training seeds.

    Pair identical windows; resample whole episodes within each task. The
    frozen quick/confirmation protocols balance episodes and windows, so the
    task/episode average equals the reported window mean. Reject unbalanced
    inputs rather than silently change the estimand.
    """
    import numpy as np
    if replicates<100:raise ValueError('too few bootstrap replicates')
    if left['weight_updates_during_eval'] or right['weight_updates_during_eval']:raise ValueError('evaluation changed weights')
    if left['evaluation_seed']!=right['evaluation_seed']:raise ValueError('candidate evaluation seeds differ')
    def indexed(report):
        rows=[x for x in report['rows'] if x['split']=='val' and x['k']==2]
        out={x['window_id']:x for x in rows}
        if not out or len(out)!=len(rows):raise ValueError('empty or duplicated validation windows')
        if len({x.get('update') for x in rows})!=1 or len({x['direction'] for x in rows})!=1:raise ValueError('mixed checkpoint steps or directions')
        return out
    a,b=indexed(left),indexed(right)
    if a.keys()!=b.keys():raise ValueError('validation windows differ')
    episodes={}
    for wid,x in a.items():
        y=b[wid]
        if (x['episode_index'],x['frame_index'],x['task'],x['direction'])!=(y['episode_index'],y['frame_index'],y['task'],y['direction']):raise ValueError('window identity/direction differs')
        delta=float(y['prediction_mse'])-float(x['prediction_mse'])
        if not math.isfinite(delta):raise ValueError('nonfinite paired metric')
        episodes.setdefault((x['task'],x['episode_index']),[]).append(delta)
    tasks=sorted({t for t,e in episodes})
    groups=[np.array([statistics.mean(v) for (task,ep),v in sorted(episodes.items()) if task==t]) for t in tasks]
    if len({len(v) for v in episodes.values()})!=1 or len({len(g) for g in groups})!=1:raise ValueError('balanced fixed-window protocol required')
    if any(len(g)<2 for g in groups):raise ValueError('at least two independent episodes per task required')
    rng=np.random.default_rng(seed)
    means=sum(g[rng.integers(0,len(g),size=(replicates,len(g)))].mean(axis=1) for g in groups)/len(groups)
    return dict(difference_right_minus_left=float(np.mean([g.mean() for g in groups])),episode_bootstrap_95_interval=list(map(float,np.quantile(means,[.025,.975]))),tasks=len(tasks),episodes=len(episodes),windows=len(a),replicates=replicates,bootstrap_seed=seed,scope='Paired, task-stratified episode uncertainty for these fixed trained models/candidate seed; does not measure training-seed uncertainty or robot success')

def curve(folder,batch,split,expected_windows):
    """Accept only paired directions on identical fixed windows and eval seed."""
    points={}
    for file in sorted(folder.glob('eval_*_action.json')):
        left=read(file);right=read(file.with_name(file.name.replace('_action.json','_future.json')))
        if not left or not right:continue # An evaluation can still be in progress.
        parts=[]
        for report,direction in [(left,'action'),(right,'future')]:
            if report['weight_updates_during_eval']!=0:raise ValueError('evaluation changed weights')
            rows=[x for x in report['rows'] if x['split']==split and x['k']==2]
            if len(rows)!=expected_windows:raise ValueError(f'wrong {split} window count: {file}')
            if any(x['direction']!=direction or not math.isfinite(x['prediction_mse']) for x in rows):raise ValueError('invalid metric row')
            ids=[(x['window_id'],x['episode_index'],x['frame_index'],x['task']) for x in rows]
            if len(set(ids))!=len(ids):raise ValueError('duplicate fixed window')
            steps={x['update'] for x in rows}
            if len(steps)!=1:raise ValueError('mixed model checkpoints')
            by_task={}
            for x in rows:by_task.setdefault(x['task'],[]).append(x['prediction_mse'])
            parts.append(dict(ids=ids,step=next(iter(steps)),seed=report['evaluation_seed'],mean=statistics.mean(x['prediction_mse'] for x in rows),per_task={t:statistics.mean(v) for t,v in by_task.items()}))
        a,z=parts
        if (a['ids'],a['step'],a['seed'])!=(z['ids'],z['step'],z['seed']):raise ValueError('Action/Future evaluation protocol differs')
        visits=a['step']*batch
        written=max(file.stat().st_mtime,file.with_name(file.name.replace('_action.json','_future.json')).stat().st_mtime)
        points[visits]=dict(sample_visits=visits,updates=a['step'],action_mse=a['mean'],future_mse=z['mean'],per_task_action=a['per_task'],per_task_future=z['per_task'],window_ids=a['ids'],evaluation_seed=a['seed'],evaluation_written_unix=written)
    return points

def completed_budget_comparison(arms):
    completed=[a for a in arms if a.get('execution_status')=='completed' and 4096 in a['validation']]
    common=sorted(set.intersection(*(set(a['validation']) for a in completed))) if completed else []
    tail=[v for v in common if v>0][-3:]
    return dict(batches=[a['batch'] for a in completed],common_sample_visits=common,
                tail_medians=[dict(batch=a['batch'],sample_points=tail,
                    action_median=statistics.median(a['validation'][v]['action_mse'] for v in tail),
                    future_median=statistics.median(a['validation'][v]['future_mse'] for v in tail))
                    for a in completed if tail],
                scope='Only normally completed4096-budget arms; failed arms remain in the main table with actual counts. This is metric availability, not stability approval or final selection.')


def main_comparison_status(arms,common):
    if all(a.get('execution_status') in ['completed','failed','terminated_incomplete'] for a in arms):
        if any(a['execution_status']!='completed' for a in arms):
            return 'main_trials_terminal_with_failures_no_final_recommendation'
    return 'main_common_budget_available' if 4096 in common else 'incomplete_no_final_recommendation'


def build_report(root):
    arms=[]
    for batch in [8,16,32,64]:
        name=f'batch{batch}_micro4_fp32_seed42';folder=root/name
        val=curve(folder,batch,'val',100)
        train=curve(folder/'train_monitor',batch,'train',20)
        timings={int(x['global_step_completed']):x for x in jsonlines(folder/'timing_history.jsonl') if 'global_step_completed' in x}
        steady=[x['end_to_end_seconds']/batch for step,x in timings.items() if step*batch>64 and math.isfinite(x['end_to_end_seconds'])]
        overlaps=[e for e in jsonlines(root/'measurement_overlap_events.jsonl') if e.get('other_training_run')==name]
        affected=set()
        for event in overlaps:
            affected.update(range(event['other_step_before']+1,event['other_step_after']+2))
        without_overlap=[x['end_to_end_seconds']/batch for step,x in timings.items() if step*batch>64 and step not in affected and math.isfinite(x['end_to_end_seconds'])]
        resource=read(folder/'resource_summary.json',{})
        sessions=resource_sessions(folder,resource)
        annotate_times(val,timings,sessions,resource);annotate_times(train,timings,sessions,resource)
        init=read(folder/'initialization_step0.json',{})
        arm=dict(name=name,batch=batch,native_microbatch=4,completed_updates=max(timings,default=0),completed_sample_visits=max(timings,default=0)*batch,validation=val,train_monitor=train,steady_median_seconds_per_sample=statistics.median(steady) if steady else None,steady_timing_updates=len(steady),resource_peaks=resource.get('peaks',{}),gpu_peaks=resource.get('gpu_peaks',{}),exit_classification=resource.get('exit_classification'),best_selection=read(folder/'best_selection.json'),initialization_sample_hash=init.get('sampled_parameter_sha256'),stats_fingerprint=init.get('stats_fingerprint'))
        code=resource.get('returncode')
        arm['execution_status']=('running' if resource.get('pid') else 'not_started') if code is None else ('completed' if code==0 and arm['completed_sample_visits']>=4096 else ('failed' if code!=0 else 'terminated_incomplete'))
        arm.update(returncode=code,guard_reason=resource.get('guard_reason'),
                   target_sample_visits=4096,target_budget_reached=arm['completed_sample_visits']>=4096,
                   last_full_validation_sample_visits=max(val,default=None),
                   latest_checkpoint_step=read(folder/'ckpt_latest.pt.meta.json',{}).get('meta',{}).get('global_step'))
        arm['sampled_resource_means']=sampled_resource_means(folder)
        arm['runtime_clock']=dict(sessions=sessions,scope='Evaluation write mtimes and recorded resource sessions. Active-session wall time includes setup/data/validation/checkpoint work, excludes between-process gaps, and has resource-sampling boundary uncertainty. First-session wall includes pauses; unique logged step sums exclude setup/eval/save/replayed lost work. No interpolated MSE at unobserved times.')
        arm['training_stability']=training_stability(folder,batch)
        arm['measurement_overlap_context']=dict(events=overlaps,conservatively_affected_updates=sorted(affected),steady_updates_after_exclusion=len(without_overlap),steady_median_seconds_per_sample_excluding_recorded_overlaps=statistics.median(without_overlap) if without_overlap else None,scope='Raw timing median remains above; this additional sensitivity calculation excludes only explicitly recorded external-work overlaps, including a neighboring in-flight update.')
        baselines=read(folder/'dataset_baselines.json')
        arm['simple_baselines']=baselines
        arm['validation_vs_simple_baselines']={}
        if baselines and val:
            reference=baselines['val']
            action=reference['action']['training_mean_mse']
            future=reference['future']['current_visual_persistence_mse']
            if reference['action']['samples']!=100 or reference['future']['samples']!=100:
                raise ValueError('simple baselines use a different validation sample count')
            if 0 in val and not (math.isclose(val[0]['action_mse'],reference['action']['zero_mse'],abs_tol=1e-6) and math.isclose(val[0]['future_mse'],future,abs_tol=1e-6)):
                raise ValueError('untrained prior and simple-baseline coordinates differ')
            for visits,point in val.items():
                arm['validation_vs_simple_baselines'][visits]=dict(
                    action_difference_from_training_mean=point['action_mse']-action,
                    future_difference_from_current_visual=point['future_mse']-future,
                    scope='Negative difference beats this simple baseline; not a significance test or training-seed confirmation.')
        arms.append(arm)
    available=[x for x in arms if x['validation']]
    # Missing arms must not silently disappear from the final comparison.
    common=sorted(set.intersection(*(set(x['validation']) for x in arms)))
    pairwise=[]
    for i,left in enumerate(available):
        for right in available[i+1:]:
            matches=sorted(set(left['validation'])&set(right['validation']))
            for visits in matches:
                a,z=left['validation'][visits],right['validation'][visits]
                if (a['window_ids'],a['evaluation_seed'])!=(z['window_ids'],z['evaluation_seed']):raise ValueError('cross-arm validation protocol differs')
            pairwise.append(dict(batches=[left['batch'],right['batch']],common_sample_visits=matches))
    if len({a['initialization_sample_hash'] for a in available if a['initialization_sample_hash']})>1:raise ValueError('initializations differ')
    if len({a['stats_fingerprint'] for a in available if a['stats_fingerprint']})>1:raise ValueError('normalization differs')
    # Last-three medians use the same positive sample checkpoints in every arm.
    positive=[v for v in common if v>0]
    tail=positive[-3:]
    matched=[]
    for arm in arms:
        if not tail:continue
        matched.append(dict(batch=arm['batch'],sample_points=tail,action_median=statistics.median(arm['validation'][v]['action_mse'] for v in tail),future_median=statistics.median(arm['validation'][v]['future_mse'] for v in tail)))
    for arm in arms:
        # Identity rows are checked above; avoid repeating hundreds of IDs in a rollup.
        for series in ['validation','train_monitor']:
            for point in arm[series].values():point.pop('window_ids')
    hardware=read(root/'hardware_control_results.json') or read(root/'hardware_control_controller_status.json') or {'phase':'not_executed'}
    return dict(status=main_comparison_status(arms,common),common_sample_visits=common,pairwise=pairwise,last_common_positive_points=tail,matched_tail_medians=matched,arms=arms,full_budget_arm_comparison=completed_budget_comparison(arms),hardware_placement_control=hardware,scope='Fixed10task100window conditional validation; independent Action/Future MSE. No matched-LR or training-seed confirmation claim. Resource peaks are process/assignedGPU; shared cgroup is separate. No robot success rate or global-best claim.')

def build_lr_probe_report(root):
    """Report bounded LR probes without selecting a long-run winner or deleting weights."""
    plan=read(root/'matched_lr_probes_pending_plan.json')
    if not plan:raise ValueError('missing matched LR probe plan')
    groups=[]
    lr_keys=['learning_rate','task_learning_rate','energy_head_learning_rate','step_size_learning_rate']
    for batch in [8,16,32,64]:
        jobs=[dict(name=f'batch{batch}_micro4_fp32_seed42',outer_lr_factor=1.)]
        jobs += [j for j in plan['jobs'] if j['effective_batch']==batch]
        if sorted(j['outer_lr_factor'] for j in jobs)!=[.3,1.,3.]:raise ValueError('each batch requires the same three LR factors')
        candidates=[]
        for job in jobs:
            folder=root/job['name'];points=curve(folder,batch,'val',100)
            init=read(folder/'initialization_step0.json',{});config=read(folder/'resolved_config.json')
            point=points.get(1024)
            if point and (not config or not init.get('sampled_parameter_sha256') or not init.get('stats_fingerprint')):
                raise ValueError('completed LR probe lacks initialization/config evidence')
            candidates.append(dict(name=job['name'],outer_lr_factor=job['outer_lr_factor'],point=point,initialization=init,config=config))
        ready=[c for c in candidates if c['point']]
        baseline=candidates[0]
        for candidate in ready:
            if not baseline['point']:continue
            left,right=baseline['point'],candidate['point']
            if (left['window_ids'],left['evaluation_seed'])!=(right['window_ids'],right['evaluation_seed']):raise ValueError('LR probe validation protocol differs')
            for key in ['sampled_parameter_sha256','stats_fingerprint']:
                if candidate['initialization'][key]!=baseline['initialization'][key]:raise ValueError('LR probe initialization/statistics differ')
            ref,cfg=baseline['config'],candidate['config']
            for key in ['backbone','data','solver','init','loss','stage','framework']:
                if cfg.get(key)!=ref.get(key):raise ValueError('LR probe changes algorithm/data as well as LR')
            for item in [ref['trainer'],cfg['trainer']]:
                if (item['per_device_batch_size'],item['shared_microbatch_size'],item['seed'],item['warmup_steps']*batch)!=(batch,4,42,256):raise ValueError('LR probe batch/seed/warmup differs')
            for key in lr_keys:
                if not math.isclose(cfg['trainer'][key],ref['trainer'][key]*candidate['outer_lr_factor'],rel_tol=1e-10):raise ValueError('actual LR factor differs from plan')
        complete=len(ready)==3
        undominated=[]
        for candidate in ready:
            point=candidate['point']
            dominated=any(other['point']['action_mse']<=point['action_mse'] and other['point']['future_mse']<=point['future_mse'] and (other['point']['action_mse']<point['action_mse'] or other['point']['future_mse']<point['future_mse']) for other in ready)
            if not dominated:undominated.append(candidate['outer_lr_factor'])
        rows=[]
        for candidate in candidates:
            point=candidate['point']
            metric=None if point is None else {k:point[k] for k in ['updates','sample_visits','action_mse','future_mse']}
            rows.append(dict(name=candidate['name'],outer_lr_factor=candidate['outer_lr_factor'],validation_at1024=metric))
        groups.append(dict(batch=batch,status='three_factors_available' if complete else 'incomplete',candidates=rows,pareto_factors_at1024=undominated if complete else None))
    complete_batches=[g['batch'] for g in groups if g['status']=='three_factors_available']
    return dict(status='short_lr_comparisons_available' if len(complete_batches)==4 else 'incomplete_no_lr_selection',complete_batches=complete_batches,groups=groups,scope='Equal1024 sample visits, fixed quick validation. Pareto means neither metric is worse at this point, not statistical superiority or a final LR/batch choice. Slower learning can need longer budgets. Selected LR still needs the common full budget, training-seed and independent confirmation. No checkpoint is deleted by this report.')

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=DEFAULT);p.add_argument('--lr-probes',action='store_true');a=p.parse_args()
    report=build_lr_probe_report(a.root) if a.lr_probes else build_report(a.root)
    target=a.root/('matched_lr_comparison_report.json' if a.lr_probes else 'main_comparison_report.json');temporary=target.with_suffix('.pending.json')
    temporary.write_text(json.dumps(report,indent=2));temporary.replace(target)
    keys=['status','complete_batches'] if a.lr_probes else ['status','common_sample_visits','matched_tail_medians']
    print(json.dumps({k:report[k] for k in keys},indent=2))

if __name__=='__main__':main()

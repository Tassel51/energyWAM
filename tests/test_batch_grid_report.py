import json
import os
import pytest
from scripts.report_batch_grid import build_report,curve

def make_eval(folder,batch,visits=1024):
    folder.mkdir(parents=True,exist_ok=True)
    for direction,mse in [('action',.8),('future',.1)]:
        rows=[dict(split='val',k=2,direction=direction,prediction_mse=mse,window_id=i,episode_index=i//2,frame_index=i%2,task=f'task{i//10}',update=visits//batch) for i in range(100)]
        (folder/f'eval_{visits//batch:04d}_{direction}.json').write_text(json.dumps(dict(rows=rows,weight_updates_during_eval=0,evaluation_seed=2026)))

def test_missing_arm_cannot_produce_a_four_batch_recommendation(tmp_path):
    for b in [8,16,32]:make_eval(tmp_path/f'batch{b}_micro4_fp32_seed42',b)
    report=build_report(tmp_path)
    assert report['status']=='incomplete_no_final_recommendation'
    assert report['common_sample_visits']==[]
    assert all(p['common_sample_visits']==[1024] for p in report['pairwise'])


def test_terminal_guard_failure_is_reported_without_fabricating_full_budget_metrics(tmp_path):
    for batch in [8,16,32,64]:
        folder=tmp_path/f'batch{batch}_micro4_fp32_seed42'
        make_eval(folder,batch,3072 if batch==32 else 4096)
        updates=104 if batch==32 else 4096//batch
        (folder/'timing_history.jsonl').write_text(json.dumps({
            'global_step_completed':updates,'end_to_end_seconds':1.})+'\n')
        (folder/'resource_summary.json').write_text(json.dumps({
            'returncode':-15 if batch==32 else 0,
            'guard_reason':'fixed MSE guard' if batch==32 else None}))
    report=build_report(tmp_path)
    assert report['status']=='main_trials_terminal_with_failures_no_final_recommendation'
    failed=report['arms'][2]
    assert failed['execution_status']=='failed'
    assert failed['completed_sample_visits']==3328 and not failed['target_budget_reached']
    assert failed['last_full_validation_sample_visits']==3072
    assert 4096 not in failed['validation']
    assert report['full_budget_arm_comparison']['batches']==[8,16,64]
    active=tmp_path/'batch64_micro4_fp32_seed42'/'resource_summary.json'
    active.write_text(json.dumps({'pid':123,'returncode':None}))
    assert build_report(tmp_path)['status']=='incomplete_no_final_recommendation'


def test_recorded_overlap_keeps_raw_timing_and_reports_sensitivity(tmp_path):
    folder=tmp_path/'batch16_micro4_fp32_seed42';folder.mkdir()
    timings=[dict(global_step_completed=s,end_to_end_seconds=t) for s,t in [(5,10.),(6,100.),(7,100.),(8,10.)]]
    (folder/'timing_history.jsonl').write_text('\n'.join(map(json.dumps,timings)))
    (tmp_path/'measurement_overlap_events.jsonl').write_text(json.dumps(dict(other_training_run=folder.name,other_step_before=5,other_step_after=6))+'\n')
    arm=build_report(tmp_path)['arms'][1]
    assert arm['steady_median_seconds_per_sample']==55/16
    context=arm['measurement_overlap_context']
    assert context['conservatively_affected_updates']==[6,7]
    assert context['steady_median_seconds_per_sample_excluding_recorded_overlaps']==10/16


def test_validation_clock_excludes_resume_gap_and_keeps_step_time_separate(tmp_path):
    folder=tmp_path/'batch8_micro4_fp32_seed42';make_eval(folder,8)
    for path in folder.glob('eval_*.json'):os.utime(path,(2200.,2200.))
    rows=[dict(global_step_completed=s,end_to_end_seconds=2.) for s in range(1,129)]
    (folder/'timing_history.jsonl').write_text('\n'.join(map(json.dumps,rows)))
    sessions=[dict(unix=1100.,elapsed_seconds=100.),dict(unix=1300.,elapsed_seconds=300.),dict(unix=2100.,elapsed_seconds=100.),dict(unix=2300.,elapsed_seconds=300.)]
    (folder/'resource_trace.jsonl').write_text('\n'.join(map(json.dumps,sessions)))
    point=build_report(tmp_path)['arms'][0]['validation'][1024]
    assert point['cumulative_unique_logged_step_seconds']==256.
    assert point['active_session_wall_seconds_to_evaluation']==500.
    assert point['wall_seconds_from_first_session_to_evaluation']==1200.
    (folder/'timing_history.jsonl').write_text('\n'.join(map(json.dumps,rows[1:])))
    assert build_report(tmp_path)['arms'][0]['validation'][1024]['cumulative_unique_logged_step_seconds'] is None

def test_sample_axis_matches_different_update_counts_and_keeps_both_metrics(tmp_path):
    for b in [8,16,32,64]:make_eval(tmp_path/f'batch{b}_micro4_fp32_seed42',b,4096)
    report=build_report(tmp_path)
    assert report['common_sample_visits']==[4096]
    assert {a['validation'][4096]['updates'] for a in report['arms']}=={512,256,128,64}
    assert len(report['matched_tail_medians'])==4
    assert all(x['action_median']==.8 and x['future_median']==.1 for x in report['matched_tail_medians'])
    assert 'matched-LR or training-seed confirmation' in report['scope']

def test_mismatched_fixed_windows_are_rejected(tmp_path):
    make_eval(tmp_path,8)
    file=tmp_path/'eval_0128_future.json';report=json.loads(file.read_text());report['rows'][1]['episode_index']=999
    file.write_text(json.dumps(report))
    with pytest.raises(ValueError,match='protocol differs'):curve(tmp_path,8,'val',100)

def test_paired_bootstrap_resamples_whole_episodes_and_keeps_constant_shift():
    from scripts.report_batch_grid import paired_episode_bootstrap
    rows=[dict(split='val',k=2,direction='future',task=t,episode_index=e,frame_index=f,window_id=100*t+10*e+f,prediction_mse=1.) for t in [0,1] for e in [1,2] for f in [0,1]]
    a=dict(rows=rows,weight_updates_during_eval=0,evaluation_seed=2026)
    b=dict(a,rows=[dict(x,prediction_mse=.75) for x in rows])
    report=paired_episode_bootstrap(a,b,replicates=100)
    assert report['difference_right_minus_left']==-.25
    assert report['episode_bootstrap_95_interval']==[-.25,-.25]
    assert report['episodes']==4 and report['windows']==8
    assert 'training-seed uncertainty' in report['scope']

def test_paired_bootstrap_rejects_mismatched_seed_and_unbalanced_frames():
    from scripts.report_batch_grid import paired_episode_bootstrap
    rows=[dict(split='val',k=2,direction='action',task='a',episode_index=e,frame_index=f,window_id=10*e+f,prediction_mse=1.) for e in [1,2] for f in [0,1]]
    a=dict(rows=rows,weight_updates_during_eval=0,evaluation_seed=2026)
    with pytest.raises(ValueError,match='seeds differ'):paired_episode_bootstrap(a,dict(a,evaluation_seed=2027),100)
    unbalanced=dict(a,rows=rows[:-1])
    with pytest.raises(ValueError,match='balanced'):paired_episode_bootstrap(unbalanced,unbalanced,100)


def test_many_correlated_windows_do_not_act_like_independent_episodes():
    from scripts.report_batch_grid import paired_episode_bootstrap
    rows=[dict(split='val',k=2,direction='future',task=t,episode_index=e,frame_index=f,window_id=1000*t+100*e+f,prediction_mse=1.) for t in [0,1] for e in [1,2] for f in range(20)]
    a=dict(rows=rows,weight_updates_during_eval=0,evaluation_seed=2026)
    b=dict(a,rows=[dict(x,prediction_mse=.5 if x['episode_index']==1 else 1.5) for x in rows])
    result=paired_episode_bootstrap(a,b,replicates=1000)
    assert result['episodes']==4 and result['windows']==80
    assert result['episode_bootstrap_95_interval']==[-.5,.5]


def test_lr_probe_report_requires_all_factors_and_checks_actual_protocol(tmp_path):
    from scripts.report_batch_grid import build_lr_probe_report
    jobs=[]
    for batch in [8,16,32,64]:
        for label,factor in [('03',.3),('3',3.)]:
            jobs.append(dict(name=f'batch{batch}_probe{label}',effective_batch=batch,outer_lr_factor=factor))
    (tmp_path/'matched_lr_probes_pending_plan.json').write_text(json.dumps(dict(jobs=jobs)))
    for name,factor in [('batch8_micro4_fp32_seed42',1.),('batch8_probe03',.3),('batch8_probe3',3.)]:
        folder=tmp_path/name;make_eval(folder,8)
        (folder/'initialization_step0.json').write_text(json.dumps(dict(sampled_parameter_sha256='same',stats_fingerprint='train-only')))
        trainer=dict(per_device_batch_size=8,shared_microbatch_size=4,seed=42,warmup_steps=32)
        trainer.update({key:factor*value for key,value in [('learning_rate',1e-5),('task_learning_rate',1e-4),('energy_head_learning_rate',1e-3),('step_size_learning_rate',1e-3)]})
        (folder/'resolved_config.json').write_text(json.dumps(dict(trainer=trainer)))
    result=build_lr_probe_report(tmp_path)
    assert result['status']=='incomplete_no_lr_selection' and result['complete_batches']==[8]
    assert result['groups'][0]['pareto_factors_at1024']==[1.,.3,3.]
    path=tmp_path/'batch8_probe3'/'resolved_config.json';cfg=json.loads(path.read_text())
    cfg['trainer']['energy_head_learning_rate']=1e-3;path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError,match='actual LR factor'):build_lr_probe_report(tmp_path)
    cfg['trainer']['energy_head_learning_rate']=3e-3;path.write_text(json.dumps(cfg))
    path=tmp_path/'batch8_probe3'/'initialization_step0.json';init=json.loads(path.read_text());init['sampled_parameter_sha256']='different';path.write_text(json.dumps(init))
    with pytest.raises(ValueError,match='initialization/statistics differ'):build_lr_probe_report(tmp_path)

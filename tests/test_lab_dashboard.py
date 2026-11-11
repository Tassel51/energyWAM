import json
from scripts.monitor_lab_studies import arm_info,fixed_metrics,bar,process_info


def test_completed_from_full_log_not_target_or_eval(tmp_path):
    (tmp_path/'train_log.jsonl').write_text(json.dumps({'global_step':6,'optimizer_lr/energy_head':.001})+'\n')
    (tmp_path/'speed_report.json').write_text('{broken')
    arm=arm_info(dict(name='trial',folder=str(tmp_path)),{'trial':dict(phase='failed',target_updates=210)})
    assert arm['completed']==7 and arm['target']==210
    assert arm['latest'] is None


def test_new_initialization_is_zero_not_completed_evaluation(tmp_path):
    (tmp_path/'initialization_step0.json').write_text('{"global_step":0}')
    info=arm_info(dict(name='trial',folder=str(tmp_path)),{'trial':dict(phase='running',pid=999999999)})
    assert info['completed']==0 and info['phase']=='process_exited_or_pid_changed'
    assert not process_info(999999999,['train_shared_sharded.py'])['alive']


def test_fixed_metrics_pairs_common_step_and_own_k(tmp_path):
    for step,direction in [(20,'action'),(20,'future'),(40,'action')]:
        rows=[dict(split='train',k=k,prediction_mse=m) for k,m in [(2,9.),(3,.25)]]
        (tmp_path/f'eval_{step:04d}_{direction}.json').write_text(json.dumps(dict(rows=rows)))
    value=fixed_metrics(tmp_path,'train',3)
    assert value['update']==20 and value['action']==value['future']==.25
    assert value['action_samples']==1


def test_missing_data_not_marked_complete(tmp_path):
    info=arm_info(dict(name='trial',folder=str(tmp_path)),{'trial':dict(phase='complete',target_updates=210)})
    assert info['completed'] is None
    assert '未记录' in bar(None,210)

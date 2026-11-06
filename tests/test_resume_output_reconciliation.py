import json
import pytest
from energywam.training.run_state import reconcile_resume_outputs


def test_resume_archives_later_metrics_and_partial_evaluation_without_touching_weights(tmp_path):
    log = tmp_path / 'train_log.jsonl'
    original = '\n'.join([json.dumps({'global_step':97,'loss':1}),
                          json.dumps({'global_step':98,'loss':2}),
                          json.dumps({'event':'restore','restored_step':98})]) + '\n'
    log.write_text(original)
    (tmp_path/'timing_history.jsonl').write_text(json.dumps({'global_step_completed':104})+'\n')
    (tmp_path/'speed_report.json').write_text(json.dumps({
        'completed_steps':104,'steps':[{'global_step_completed':98},{'global_step_completed':104}]}))
    (tmp_path/'best_selection.json').write_text(json.dumps({'global_step':32}))
    (tmp_path/'final_report.json').write_text(json.dumps({'global_step':104}))
    (tmp_path/'ckpt_latest.pt').write_bytes(b'weight sentinel')
    monitor=tmp_path/'train_monitor';monitor.mkdir()
    (monitor/'eval_0104_action.json').write_text(json.dumps({'rows':[]}))
    (monitor/'eval_0098_future.json').write_text('partial')
    (monitor/'eval_0096_action.json').write_text(json.dumps({'rows':[{'update':96}]}))
    result=reconcile_resume_outputs(tmp_path,98)
    archive=tmp_path/'resume_history'/f"{result['generation']}_from98"
    assert (archive/'train_log.jsonl').read_text()==original
    assert [json.loads(x) for x in log.read_text().splitlines()]==[
        {'global_step':97,'loss':1},{'event':'restore','restored_step':98}]
    speed=json.loads((tmp_path/'speed_report.json').read_text())
    assert speed=={'completed_steps':98,'steps':[{'global_step_completed':98}]}
    assert not (monitor/'eval_0104_action.json').exists()
    assert not (monitor/'eval_0098_future.json').exists()
    assert (monitor/'eval_0096_action.json').exists()
    assert (archive/'train_monitor/eval_0104_action.json').exists()
    assert not (tmp_path/'final_report.json').exists()
    assert (tmp_path/'ckpt_latest.pt').read_bytes()==b'weight sentinel'
    assert not reconcile_resume_outputs(tmp_path,98)['changed']


def test_resume_older_than_retained_best_is_rejected_before_modifying_records(tmp_path):
    (tmp_path/'best_selection.json').write_text(json.dumps({'global_step':100}))
    report=tmp_path/'eval_0104_action.json';report.write_text('original')
    with pytest.raises(ValueError,match='retained best'):
        reconcile_resume_outputs(tmp_path,98)
    assert report.read_text()=='original'
    assert not (tmp_path/'resume_history').exists()

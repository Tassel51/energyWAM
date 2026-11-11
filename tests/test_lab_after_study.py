import json
from scripts.run_lab_after_study import dependency_state


def test_requires_final_arm_and_preceding_controller_completion(tmp_path):
    assert dependency_state(tmp_path,'future_half')=='waiting_for_preceding_study'
    (tmp_path/'full_status.json').write_text(json.dumps({'future_half':{'phase':'complete'}}))
    (tmp_path/'controller_status.json').write_text(json.dumps({'phase':'training'}))
    assert dependency_state(tmp_path,'future_half')=='waiting_for_preceding_study'
    (tmp_path/'controller_status.json').write_text(json.dumps({'phase':'complete'}))
    assert dependency_state(tmp_path,'future_half')=='ready'


def test_failed_prerequisite_does_not_launch_even_when_queue_completed(tmp_path):
    (tmp_path/'full_status.json').write_text(json.dumps({'future_half':{'phase':'failed'}}))
    (tmp_path/'controller_status.json').write_text(json.dumps({'phase':'complete'}))
    assert dependency_state(tmp_path,'future_half')=='blocked_dependency'

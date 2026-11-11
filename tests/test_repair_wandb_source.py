import hashlib
import pytest

from scripts.repair_wandb_source import recovery_plan, replay_state, require_uploader_stopped


def inputs(folder):
    cfg = {'trainer': {'max_train_steps': 3}}
    report = {'checked_unix': 100, 'run_id': hashlib.sha256(str(folder.resolve()).encode()).hexdigest()[:12],
              'source_folder': str(folder), 'state': 'crashed', 'mismatches': [],
              'missing': [], 'expected_updates': 3, 'server_last_uploaded_update': 3}
    resource = {'returncode': 0, 'exit_classification': 'completed',
                'last_sample': {'completed_updates': 3}}
    rows = {1: {}, 2: {}, 3: {}}
    return cfg, report, resource, rows


def test_history_gap_overrides_later_summary_ack(tmp_path):
    cfg, report, resource, rows = inputs(tmp_path)
    report['missing'] = [['train', 2, 'train/future_mse']]
    plan = recovery_plan(tmp_path, cfg, report, resource, rows, now=110)
    assert plan['replay_after_update'] == 1
    report['server_last_uploaded_update'] = 0
    assert recovery_plan(tmp_path, cfg, report, resource, rows, now=110)['replay_after_update'] == 0


def test_probe_only_gap_retains_ack_but_clears_all_probe_markers(tmp_path):
    cfg, report, resource, rows = inputs(tmp_path)
    report['missing'] = [['probe', 0, 'probe/train/action_mse_evalK2']]
    plan = recovery_plan(tmp_path, cfg, report, resource, rows, now=110)
    assert plan['replay_after_update'] == 3
    original = {'last_step': 3, 'probe_files': {'eval.json': 'hash'},
                'collector_finished': True, 'saved_step': 3}
    state = replay_state(original, 3)
    assert state['probe_files'] == {} and 'collector_finished' not in state
    assert 'saved_step' not in state
    assert original['probe_files'] == {'eval.json': 'hash'}


@pytest.mark.parametrize('field,value', [
    ('checked_unix', -2000), ('checked_unix', 120),
    ('run_id', 'other'), ('state', 'running'),
    ('mismatches', [['train', 2, 'train/action_mse']]),
    ('server_last_uploaded_update', 4), ('server_last_uploaded_update', None),
    ('expected_updates', 4),
])
def test_rejects_untrusted_or_stale_repair_evidence(tmp_path, field, value):
    cfg, report, resource, rows = inputs(tmp_path)
    report[field] = value
    with pytest.raises(ValueError):
        recovery_plan(tmp_path, cfg, report, resource, rows, now=110)


def test_refuses_failed_training_and_incomplete_source_history(tmp_path):
    cfg, report, resource, rows = inputs(tmp_path)
    resource['returncode'] = -15
    with pytest.raises(ValueError):
        recovery_plan(tmp_path, cfg, report, resource, rows, now=110)
    resource['returncode'] = 0
    del rows[2]
    with pytest.raises(ValueError):
        recovery_plan(tmp_path, cfg, report, resource, rows, now=110)


def test_paused_uploader_must_be_absent_or_zombie(tmp_path):
    assert require_uploader_stopped(123, tmp_path) == 'exited'
    proc = tmp_path / '123'
    proc.mkdir()
    (proc / 'status').write_text('State:\tS (sleeping)\n')
    (proc / 'cmdline').write_bytes(b'python\0scripts/watch_wandb.py\0')
    with pytest.raises(RuntimeError):
        require_uploader_stopped(123, tmp_path)
    (proc / 'status').write_text('State:\tZ (zombie)\n')
    (proc / 'cmdline').write_bytes(b'')
    assert require_uploader_stopped(123, tmp_path) == 'zombie'

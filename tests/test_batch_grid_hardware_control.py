import json
import pytest
from scripts.run_batch_grid_hardware_control import main_arms_ready, validate_jobs


def save(path, value):
    path.write_text(json.dumps(value))


def test_hardware_gate_waits_for_main_and_rejects_incomplete_arm(tmp_path):
    controller = tmp_path / 'main_controller_status.json'
    save(controller, {'phase': 'main_round_running'})
    assert not main_arms_ready(tmp_path)
    save(controller, {'phase': 'main_four_arms_complete'})
    for round_number, batches in [(1, [8, 16]), (2, [32, 64])]:
        for pair, batch in zip(['01', '23'], batches):
            save(tmp_path / f'main_round{round_number}_status_{pair}.json',
                 {f'batch{batch}_micro4_fp32_seed42': {
                     'phase': 'complete', 'returncode': 0,
                     'completed_updates': 4096 // batch}})
    assert main_arms_ready(tmp_path)
    save(tmp_path / 'main_round2_status_23.json', {
        'batch64_micro4_fp32_seed42': {'phase': 'complete', 'returncode': 0,
                                      'completed_updates': 63}})
    with pytest.raises(RuntimeError, match='incomplete'):
        main_arms_ready(tmp_path)
    save(controller, {'phase': 'preflight_or_main_failed'})
    with pytest.raises(RuntimeError, match='failed'):
        main_arms_ready(tmp_path)


def test_hardware_jobs_allow_run_identity_but_reject_training_differences(tmp_path):
    import yaml
    jobs = []
    for pair in ['0,1', '2,3']:
        name = pair.replace(',', '')
        cfg = {'run_id': name, 'backbone': {'dtype': 'float32'}, 'trainer': {
            'shared_microbatch_size': 4, 'per_device_batch_size': 16,
            'output_dir': str(tmp_path / name), 'seed': 42}}
        (tmp_path / f'{name}.yaml').write_text(yaml.safe_dump(cfg))
        jobs.append(dict(profile=True, steps=6, effective_batch=16,
                         physical_gpus=pair, config=f'{name}.yaml'))
    validate_jobs(jobs, tmp_path)
    cfg['trainer']['seed'] = 43
    (tmp_path / '23.yaml').write_text(yaml.safe_dump(cfg))
    with pytest.raises(RuntimeError, match='differ'):
        validate_jobs(jobs, tmp_path)


def test_only_reviewed_terminal_numerical_failure_can_pass_after_other_arms_finish(tmp_path):
    name = 'batch32_micro4_fp32_seed42'
    reason = 'Fixed train action MSE at update 104 exceeded 10x its untrained prediction baseline'
    for round_number, batches in [(1, [8, 16]), (2, [32, 64])]:
        for pair, batch in zip(['01', '23'], batches):
            save(tmp_path / f'main_round{round_number}_status_{pair}.json', {
                f'batch{batch}_micro4_fp32_seed42': {
                    'phase': 'complete', 'returncode': 0, 'completed_updates': 4096 // batch}})
    failure = dict(phase='failed', early_stopped=True, reason=reason,
                   returncode=-15, completed_updates=104, pid=99999999)
    save(tmp_path / 'main_round2_status_01.json', {name: failure})
    save(tmp_path / 'main_controller_status.json', {
        'phase': 'preflight_or_main_failed',
        'reason': f'Queue arm failed or incomplete: {name}: failed updates=104'})
    with pytest.raises(RuntimeError, match='unreviewed'):
        main_arms_ready(tmp_path)
    save(tmp_path / 'accepted_main_guard_failures.json', {'runs': {name: failure}})
    folder = tmp_path / name
    folder.mkdir()
    resource = dict(exit_classification='experiment_guard', returncode=-15,
                    memory_events_before={'oom': 0, 'oom_kill': 0},
                    last_sample={'memory_events': {'oom': 0, 'oom_kill': 0}})
    save(folder / 'resource_summary.json', resource)
    assert main_arms_ready(tmp_path)
    resource['last_sample']['memory_events']['oom_kill'] = 1
    save(folder / 'resource_summary.json', resource)
    with pytest.raises(RuntimeError, match='resource evidence'):
        main_arms_ready(tmp_path)
    # A reviewed failed arm cannot bypass the other running main trial.
    save(tmp_path / 'main_controller_status.json', {'phase': 'main_round_running'})
    assert not main_arms_ready(tmp_path)

import hashlib
import json

import pytest

from scripts import run_batch_grid_lr_probes as controller


def save(path, value):
    path.write_text(json.dumps(value))


def test_lr_queue_requires_all_audited_configs_and_full_retention(tmp_path):
    jobs, proofs = [], []
    for batch in [8, 16, 32, 64]:
        for factor in [.3, 3.]:
            name = f'{batch}_{factor}'
            path = tmp_path / (name + '.yaml')
            path.write_text('same audited config')
            jobs.append(dict(name=name, config=path.name, effective_batch=batch,
                             outer_lr_factor=factor, steps=1024 // batch,
                             keep_best=True, physical_gpus='0,1',
                             snapshot_bytes_estimate=100))
            proofs.append(dict(name=name, batch=batch, factor=factor,
                               sample_budget=1024, warmup_sample_visits=256,
                               config_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    save(tmp_path / 'matched_lr_probes_pending_plan.json', dict(jobs=jobs))
    save(tmp_path / 'matched_lr_prelaunch_config_audit.json',
         dict(passed=True, configurations=proofs))
    assert len(controller.validated_jobs(tmp_path, tmp_path)) == 8
    (tmp_path / jobs[0]['config']).write_text('unreviewed change')
    with pytest.raises(RuntimeError, match='Unverified'):
        controller.validated_jobs(tmp_path, tmp_path)
    (tmp_path / jobs[0]['config']).write_text('same audited config')
    jobs[0]['keep_best'] = False
    save(tmp_path / 'matched_lr_probes_pending_plan.json', dict(jobs=jobs))
    with pytest.raises(RuntimeError, match='Unverified'):
        controller.validated_jobs(tmp_path, tmp_path)


def test_lr_queue_waits_for_real_hardware_control_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(controller, 'main_arms_ready', lambda _: False)
    assert not controller.prerequisites_ready(tmp_path)
    monkeypatch.setattr(controller, 'main_arms_ready', lambda _: True)
    status = tmp_path / 'hardware_control_controller_status.json'
    save(status, dict(phase='hardware_control_running'))
    assert not controller.prerequisites_ready(tmp_path)
    save(status, dict(phase='failed'))
    with pytest.raises(RuntimeError, match='Hardware control failed'):
        controller.prerequisites_ready(tmp_path)
    save(status, dict(phase='complete'))
    save(tmp_path / 'hardware_control_pending_plan.json', dict(jobs=['real_outputs']))
    def rejected_outputs(root, jobs):
        assert root == tmp_path and jobs == ['real_outputs']
        raise RuntimeError('real trial output missing')
    monkeypatch.setattr(controller, 'summarize', rejected_outputs)
    with pytest.raises(RuntimeError, match='real trial output missing'):
        controller.prerequisites_ready(tmp_path)

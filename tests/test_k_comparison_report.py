import importlib.util
import json
from pathlib import Path


def test_incomplete_or_failed_group_is_not_a_winner(tmp_path):
    spec=importlib.util.spec_from_file_location('k_compare',Path(__file__).parents[1]/'scripts/run_k_training_comparison.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.OUT=tmp_path
    for k in [2,4,8]:(tmp_path/f'k{k}').mkdir()
    module.summarize({'2':{'phase':'running'}})
    report=(tmp_path/'RESULTS_CN.md').read_text()
    assert '未完成训练' in report and '本轮结论' not in report
    module.summarize({str(k):{'phase':'failed'} for k in [2,4,8]})
    report=(tmp_path/'RESULTS_CN.md').read_text()
    assert '无法给出可信的配置选择' in report
    assert all(g['updates']==0 for g in json.loads((tmp_path/'comparison.json').read_text())['groups'].values())


def test_probe_restores_rng_and_parameter_flags_on_failure(tmp_path, monkeypatch):
    import sys
    import torch
    from types import SimpleNamespace
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'scripts'))
    import k_training_probe as probe
    model=torch.nn.Sequential(torch.nn.Linear(2,2),torch.nn.Linear(2,1))
    model[0].requires_grad_(False)
    model.train()
    flags=[p.requires_grad for p in model.parameters()]
    rng=torch.get_rng_state().clone()
    def fail(*args):
        torch.rand(5)
        raise ValueError('intentional validation data failure')
    monkeypatch.setattr(probe,'fixed_dataset',fail)
    manifest=tmp_path/'manifest.json';manifest.write_text('{}')
    trainer=SimpleNamespace(model=model,optimizer=torch.optim.SGD(model.parameters(),lr=.1),cfg=None)
    import pytest
    with pytest.raises(ValueError,match='intentional'):
        probe.evaluate(trainer,manifest,tmp_path,0)
    assert model.training
    assert flags==[p.requires_grad for p in model.parameters()]
    assert torch.equal(rng,torch.get_rng_state())


def test_cleanup_reaps_workers_after_launcher_already_exited():
    import subprocess
    import sys
    import os
    import signal
    import time
    spec=importlib.util.spec_from_file_location('k_compare',Path(__file__).parents[1]/'scripts/run_k_training_comparison.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    # Launcher exits while its worker ignores TERM and remains in the session.
    code="import subprocess,sys; p=subprocess.Popen([sys.executable,'-c','import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); print(\"ready\",flush=True); time.sleep(60)']); print(p.pid,flush=True)"
    process=subprocess.Popen([sys.executable,'-c',code],stdout=subprocess.PIPE,text=True,start_new_session=True)
    child=int(process.stdout.readline());assert process.stdout.readline().strip()=='ready'
    process.wait(timeout=5)
    try:
        module.terminate_session(process)
        path=Path(f'/proc/{child}/stat')
        if path.exists():assert path.read_text().split()[2]=='Z'
    finally:
        try:os.kill(child,signal.SIGKILL)
        except ProcessLookupError:pass
        process.stdout.close()

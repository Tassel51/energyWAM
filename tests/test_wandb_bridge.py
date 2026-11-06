import importlib.util
import json
from pathlib import Path

spec=importlib.util.spec_from_file_location('wandb_bridge',Path(__file__).parents[1]/'scripts/watch_wandb.py')
bridge=importlib.util.module_from_spec(spec);spec.loader.exec_module(bridge)


def test_reconcile_server_gap_replays_validation_and_preserves_source_state():
    state={'last_step':400,'last_speed_step':400,'saved_step':384,'probe_files':{'eval.json':'hash'},'validation_files':{'val.json':'hash'}}
    recovered=bridge.reconcile_submitted_state(state,122)
    assert recovered['last_step']==122
    assert recovered['probe_files']=={} and recovered['validation_files']=={}
    assert 'saved_step' not in recovered and 'last_speed_step' not in recovered
    assert state['last_step']==400 and state['probe_files']=={'eval.json':'hash'}
    assert bridge.reconcile_submitted_state(state,400)==state


def test_source_completion_requires_success_budget_and_saved_latest(tmp_path):
    cfg={'trainer':{'max_train_steps':512}}
    state={'last_step':512,'saved_step':512}
    resource={'returncode':0,'exit_classification':'completed','last_sample':{'completed_updates':512}}
    file=tmp_path/'resource_summary.json';file.write_text(json.dumps(resource))
    assert bridge.source_training_complete(tmp_path,cfg,state)
    assert not bridge.source_training_complete(tmp_path,cfg,dict(state,last_step=511))
    assert not bridge.source_training_complete(tmp_path,cfg,dict(state,saved_step=511))
    resource['returncode']=1;file.write_text(json.dumps(resource))
    assert not bridge.source_training_complete(tmp_path,cfg,state)
    resource['returncode']=0;resource['last_sample']['completed_updates']=400;file.write_text(json.dumps(resource))
    assert not bridge.source_training_complete(tmp_path,cfg,state)


def test_guard_failure_finishes_as_failed_only_after_last_training_row_submitted(tmp_path):
    cfg={'trainer':{'max_train_steps':128}}
    state={'last_step':103,'saved_step':98}
    resource=dict(returncode=-15,exit_classification='experiment_guard',
                  guard_reason='Fixed train action MSE exceeded 10x baseline',
                  ended_unix=200,started_unix=100,last_sample={'completed_updates':104})
    (tmp_path/'resource_summary.json').write_text(json.dumps(resource))
    assert bridge.source_terminal_status(tmp_path,cfg,state) is None
    terminal=bridge.source_terminal_status(tmp_path,cfg,dict(state,last_step=104))
    assert terminal['exit_code']==1 and terminal['status']=='failed'
    assert terminal['reason']==resource['guard_reason']
    resource['returncode']=None
    (tmp_path/'resource_summary.json').write_text(json.dumps(resource))
    assert bridge.source_terminal_status(tmp_path,cfg,dict(state,last_step=104)) is None


def test_finished_source_reopens_only_when_a_new_recorded_session_exists(tmp_path):
    item={'finished':True,'folder':tmp_path,
          'state':{'collector_terminal_started_unix':100}}
    file=tmp_path/'resource_summary.json'
    file.write_text(json.dumps({'started_unix':100,'returncode':-15}))
    assert not bridge.source_session_changed(item)
    file.write_text(json.dumps({'started_unix':300,'returncode':None}))
    assert bridge.source_session_changed(item)


def test_resume_marker_replays_uncheckpointed_tail_instead_of_skipping_new_rows(tmp_path):
    folder=tmp_path/'resume_history';folder.mkdir()
    marker={'generation':123,'restored_step':98}
    (folder/'latest.json').write_text(json.dumps(marker))
    original={'last_step':104,'last_speed_step':104,'probe_files':{'old':'digest'}}
    state,observed=bridge.reconcile_resume_marker(tmp_path,original)
    assert observed==marker and state['last_step']==98
    assert state['probe_files']=={} and 'last_speed_step' not in state
    assert original['last_step']==104
    again,observed=bridge.reconcile_resume_marker(tmp_path,state)
    assert again==state and observed is None


def test_completed_updates_and_duplicate_resume_rows(tmp_path):
    file=tmp_path/'log.jsonl'
    file.write_text('\n'.join([json.dumps({'global_step':0,'loss':4}),
                              json.dumps({'global_step':0,'loss':3}),
                              json.dumps({'global_step_completed':2,'loss':2}),
                              json.dumps({'event':'resume'}),'incomplete']))
    rows=bridge.read_rows(file)
    assert list(rows)==[1,2]
    assert rows[1]['loss']==3


def test_distinct_losses_and_memory_semantics():
    cfg={'trainer':{'learning_rate':1e-5,'warmup_steps':100}}
    row={'bi/action_loss':1.5,'bi/future_loss':2,'loss':3.5,
         'offload/pinned_bytes':400e9,'cuda_max_alloc_mb':40960,'param_grad_norm':float('nan')}
    metrics=bridge.training_metrics(row,50,cfg)
    assert metrics['train/action_mse']==1.5 and metrics['train/future_mse']==2
    assert metrics['memory/rank0_gpu_peak_allocated_gib']==40
    assert metrics['train/scheduled_lr']==5e-6
    assert 'offload/pinned_bytes' not in metrics
    assert 'diagnostics/parameter_grad_norm' not in metrics


def test_account_setup_isolated_credentials(tmp_path,monkeypatch):
    import types,sys
    spec=importlib.util.spec_from_file_location('wandb_setup',Path(__file__).parents[1]/'scripts/setup_wandb_account.py')
    setup=importlib.util.module_from_spec(spec);spec.loader.exec_module(setup)
    monkeypatch.setattr(setup,'ROOT',tmp_path)
    (tmp_path/'configs').mkdir()
    config=tmp_path/'configs/wandb_bridge.json'
    config.write_text(json.dumps({'mode':'offline','entity':'my-team'}))
    netrc=tmp_path/'.netrc';netrc.write_text('shared-account-sentinel')
    monkeypatch.setattr(Path,'home',lambda:tmp_path)
    monkeypatch.setattr(sys,'stdin',types.SimpleNamespace(isatty=lambda:True))
    monkeypatch.setattr('builtins.input',lambda prompt:'')
    monkeypatch.setattr(setup.getpass,'getpass',lambda prompt:'unit-test-placeholder')
    monkeypatch.setenv('WANDB_API_KEY','')
    monkeypatch.setenv('WANDB_BASE_URL','')
    class Api:
        default_entity='my-user'
        def __init__(self,api_key,timeout):assert api_key=='unit-test-placeholder'
        def create_project(self,name,entity):assert (name,entity)==('lte-wam','my-team')
    monkeypatch.setitem(sys.modules,'wandb',types.SimpleNamespace(Api=Api))
    commands=[]
    monkeypatch.setattr(setup.subprocess,'run',lambda cmd,check:commands.append(cmd))
    setup.main()
    secret=tmp_path/'.config/energywam/wandb_api_key'
    assert secret.stat().st_mode & 0o077==0
    assert netrc.read_text()=='shared-account-sentinel'
    assert json.loads(config.read_text())['mode']=='online'
    assert 'unit-test-placeholder' not in config.read_text()
    assert commands==[['supervisorctl','restart','energywam-wandb']]


def test_closed_configured_proxy_is_reported_without_using_shared_credentials(monkeypatch):
    import socket
    monkeypatch.setenv('HTTPS_PROXY','http://127.0.0.1:7901')
    def closed(*args,**kwargs):raise OSError('connection refused')
    monkeypatch.setattr(socket,'create_connection',closed)
    assert bridge.configured_proxy_reachable() is False
    monkeypatch.delenv('HTTPS_PROXY');monkeypatch.delenv('https_proxy',raising=False)
    assert bridge.configured_proxy_reachable() is True


def test_open_proxy_with_broken_upstream_does_not_start_sdk(monkeypatch):
    import requests
    monkeypatch.setattr(bridge,'configured_proxy_reachable',lambda:True)
    monkeypatch.setenv('WANDB_API_KEY','unit-test-placeholder')
    def broken(*args,**kwargs):raise requests.exceptions.ProxyError('upstream unavailable')
    monkeypatch.setattr(requests,'post',broken)
    assert bridge.upload_route_ready({'base_url':'https://api.wandb.ai'})==(False,'api_route_unavailable')


def test_batch_comparison_axis_counts_samples_not_only_updates():
    cfg={'trainer':{'learning_rate':1e-5,'warmup_steps':4,'per_device_batch_size':64}}
    result=bridge.training_metrics({'bi/action_loss':.8,'bi/future_loss':.1},4,cfg)
    assert result['comparison/sample_visits']==256
    assert result['comparison/train_action_mse']==.8
    assert result['comparison/train_future_mse']==.1

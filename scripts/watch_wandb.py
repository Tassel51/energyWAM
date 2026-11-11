#!/usr/bin/env python
"""Mirror existing JSON logs into W&B without importing or touching the trainer."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import time
import yaml

ROOT=Path(__file__).resolve().parents[1]
STOP=False


def apply_proxy_settings(settings):
    """Apply an explicit per-collector route without changing proxy exclusions."""
    proxy=settings.get('proxy_url')
    if proxy:
        for name in ['HTTP_PROXY','HTTPS_PROXY','http_proxy','https_proxy']:
            os.environ[name]=proxy


def configured_proxy_reachable():
    """Only check an explicitly configured proxy; never alter shared networking."""
    import socket,urllib.parse
    value=os.environ.get('HTTPS_PROXY') or os.environ.get('https_proxy')
    if not value:return True
    url=urllib.parse.urlparse(value)
    if not url.hostname:return False
    try:
        with socket.create_connection((url.hostname,url.port or (443 if url.scheme=='https' else 80)),timeout=1):return True
    except OSError:return False


def upload_route_ready(settings):
    if not configured_proxy_reachable():return False,'configured_proxy_unavailable'
    key=os.environ.get('WANDB_API_KEY')
    if not key:return False,'project_credential_missing'
    import requests
    try:
        response=requests.post(settings.get('base_url','https://api.wandb.ai').rstrip('/')+'/graphql',
            json={'query':'query { viewer { username } }'},auth=('api',key),timeout=(3,7))
        if response.status_code!=200:return False,f'api_http_{response.status_code}'
        if not response.json().get('data',{}).get('viewer'):return False,'api_viewer_unavailable'
        return True,'ready'
    except (requests.RequestException,ValueError):return False,'api_route_unavailable'


def wait_for_proxy(settings):
    while settings['mode']=='online':
        ready,reason=upload_route_ready(settings)
        if ready:break
        status_path=ROOT/settings.get('status_file','logs/wandb_bridge_status.json')
        try:status=json.loads(status_path.read_text())
        except (OSError,ValueError):status={}
        status.update(mode='online',connectivity=reason,
                      updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
        atomic_json(status_path,status)
        if STOP:return False
        time.sleep(15)
    return not STOP


def completed_step(row):
    if 'global_step_completed' in row:return int(row['global_step_completed'])
    if 'global_step' in row:return int(row['global_step'])+1
    return None


def read_rows(path):
    rows={}
    if not path.exists():return rows
    with path.open() as stream:
        for line in stream:
            try:
                row=json.loads(line);step=completed_step(row)
                if step is not None:rows[step]=row
            except (ValueError,TypeError,KeyError):continue
    return rows


MAPPING={
 'solver_gradient_rms_normalized':'solver/gradient_rms_normalized',
 'optimizer_lr/language':'solver/outer_lr_language',
 'optimizer_lr/task':'solver/outer_lr_task',
 'optimizer_lr/energy_head':'solver/outer_lr_energy_head',
 'optimizer_lr/step_sizes':'solver/outer_lr_step_sizes',
 'optimizer_lr/main':'solver/outer_lr_main',
 'diag/alpha_action':'solver/alpha_action', 'diag/alpha_future':'solver/alpha_future',
 'bi/action_loss':'train/action_mse', 'bi/future_loss':'train/future_mse',
 'joint/action_loss':'train/action_mse','joint/future_loss':'train/future_mse',
 'loss':'train/total_loss','step_seconds':'performance/compute_seconds',
 'end_to_end_seconds':'performance/end_to_end_seconds',
 'samples_per_second':'performance/compute_samples_per_second',
 'end_to_end_samples_per_second':'performance/end_to_end_samples_per_second',
 'gradient_sync_seconds':'performance/sync_and_rank_wait_seconds',
 'forward_seconds':'performance/forward_seconds','backward_seconds':'performance/backward_seconds',
 'optimizer_seconds':'performance/optimizer_seconds','data_seconds':'performance/data_seconds',
 'param_grad_norm':'diagnostics/parameter_grad_norm','param_grad_nonfinite':'diagnostics/nonfinite_parameter_grads',
 'pin_peak_owned_gib':'memory/rank0_pinned_pool_peak_gib',
}
for name in ['action_gradient_norm','weighted_future_gradient_norm','direction_cosine','unweighted_norm_ratio_action_future']:
    MAPPING[f'diag/head_{name}']=f'diagnostics/head_{name}'
for direction in ['action','future']:
    MAPPING[f'diag/iteration_{direction}_grad_norm']=f'diagnostics/iteration_{direction}_grad_norm'
    for name in ['raw_grad','bound_fraction']:
        MAPPING[f'diag/step_{direction}_{name}']=f'solver/{direction}_{name}'
    MAPPING[f'diag/alpha_{direction}_after_update']=f'solver/alpha_{direction}_after_update'
    for metric in ['energy_first','energy_last','energy_delta','energy_nonfinite']:
        MAPPING[f'diag/{direction}_{metric}']=f'diagnostics/{direction}_{metric}'
    for metric in ['grad_norm_mean','grad_norm_first','grad_norm_last','update_norm_mean','update_norm_max']:
        MAPPING[f'diag/{direction}_{direction}_{metric}']=f'diagnostics/{direction}_{metric}'


def training_metrics(row,step,cfg):
    out={'train/update':step}
    if cfg['trainer'].get('per_device_batch_size'):
        out['comparison/sample_visits']=step*cfg['trainer']['per_device_batch_size']
        for direction in ['action','future']:
            value=row.get(f'bi/{direction}_loss')
            if isinstance(value,(int,float)) and math.isfinite(value):out[f'comparison/train_{direction}_mse']=value
    for source,destination in MAPPING.items():
        value=row.get(source)
        if isinstance(value,(int,float)) and math.isfinite(value):out[destination]=value
    for source,destination in [('cuda_max_alloc_mb','rank0_gpu_peak_allocated_gib'),
                               ('cuda_0_max_reserved_mb','rank0_gpu_peak_reserved_gib'),
                               ('max_rss_mb','rank0_process_peak_rss_gib')]:
        value=row.get(source)
        if isinstance(value,(int,float)) and math.isfinite(value):out['memory/'+destination]=value/1024
    trainer=cfg['trainer'];warmup=trainer.get('warmup_steps',0)
    out['train/scheduled_lr']=trainer['learning_rate']*min(1,step/warmup) if warmup else trainer['learning_rate']
    return out


def atomic_json(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2));temp.replace(path)


def reconcile_submitted_state(state,remote_step):
    """Local submission is not a server acknowledgement; replay gaps safely."""
    state=dict(state)
    local=int(state.get('last_step',0));remote=max(0,int(remote_step))
    state['last_step']=min(local,remote)
    if remote<local:
        state['validation_files']={};state['probe_files']={}
        state.pop('saved_step',None);state.pop('last_speed_step',None)
    return state


def remote_recovery_reason(item, remote_state, remote_step, now, stall_seconds=180):
    """Detect a dead SDK stream; locally submitted metrics are not ACKs."""
    remote_step=max(0,int(remote_step or 0))
    local=int(item['state'].get('last_step',0))
    item['state']['server_acknowledged_update']=remote_step
    previous=item.get('_remote_observed_update')
    if previous is None or remote_step>previous:
        item['_remote_progress_at']=now
    item['_remote_observed_update']=remote_step
    if remote_state in ('crashed','failed','finished'):
        return 'remote_stream_terminal_while_training_is_live'
    if local<=remote_step:
        item.pop('_remote_gap_since',None)
        return None
    since=item.setdefault('_remote_gap_since',now)
    progress=item.get('_remote_progress_at',now)
    if now-max(since,progress)>=stall_seconds:
        return 'remote_progress_stalled_despite_local_submissions'
    return None


def live_source(item):
    try:
        resource=json.loads((item['folder']/'resource_summary.json').read_text())
        if resource.get('returncode') is not None:return False
        pid=resource.get('pid')
        if not isinstance(pid,int):return False
        os.kill(pid,0)
        return True
    except (OSError,ValueError,TypeError):return False


def remote_stream_needs_reopen(item,settings,wandb):
    if settings['mode']!='online' or item.get('finished') or not live_source(item):return False
    now=time.monotonic()
    if now<item.get('_next_remote_check',now+60):
        item.setdefault('_next_remote_check',now+60)
        return False
    item['_next_remote_check']=now+60
    try:
        run_id=hashlib.sha256(str(item['folder'].resolve()).encode()).hexdigest()[:12]
        remote=wandb.Api(timeout=10).run(f'{settings["entity"]}/{settings["project"]}/{run_id}')
        reason=remote_recovery_reason(item,remote.state,remote.summary.get('last_uploaded_update',0),now)
    except Exception as error:
        item['state']['remote_health_error']=type(error).__name__
        return False
    item['state'].pop('remote_health_error',None)
    atomic_json(item['state_path'],item['state'])
    if not reason:return False
    print(json.dumps({'event':'reopen_stalled_stream','config':item['source_config'],
                      'reason':reason,'local_submitted_update':item['state'].get('last_step',0),
                      'server_acknowledged_update':item['state']['server_acknowledged_update']}),flush=True)
    # A fatal SDK sender may block finish forever. Bound sidecar shutdown;
    # supervisor restarts the collector and reconciles each source with server ACKs.
    import threading
    thread=threading.Thread(target=lambda:item['run'].finish(exit_code=1),daemon=True)
    thread.start();thread.join(timeout=8)
    if thread.is_alive():
        print(json.dumps({'event':'collector_restart_for_blocked_sdk_finish'}),flush=True)
        os._exit(75)
    return True


def source_training_complete(folder,cfg,state):
    """A sidecar SDK being alive is not proof that training is still running."""
    try:
        resource=json.loads((folder/'resource_summary.json').read_text())
        target=int(cfg['trainer']['max_train_steps'])
        return (target>0 and resource.get('returncode')==0
                and resource.get('exit_classification')=='completed'
                and int(resource.get('last_sample',{}).get('completed_updates',0))>=target
                and int(state.get('last_step',0))>=target
                and int(state.get('saved_step',-1))>=target)
    except (OSError,ValueError,TypeError,KeyError):return False


def source_terminal_status(folder,cfg,state):
    """Close SDK runs only after authoritative termination and log submission."""
    try:
        resource=json.loads((folder/'resource_summary.json').read_text())
        code=resource.get('returncode')
        if source_training_complete(folder,cfg,state):
            return dict(exit_code=0,status='completed',classification='completed',
                        reason=None,started_unix=resource.get('started_unix'))
        completed=int(resource.get('last_sample',{}).get('completed_updates',0))
        if (code is not None and code!=0 and resource.get('ended_unix')
                and int(state.get('last_step',0))>=completed):
            return dict(exit_code=1,status='failed',classification=resource.get('exit_classification'),
                        reason=resource.get('guard_reason'),started_unix=resource.get('started_unix'))
    except (OSError,ValueError,TypeError,KeyError):pass
    return None


def source_session_changed(item):
    if not item.get('finished'):return False
    try:
        resource=json.loads((item['folder']/'resource_summary.json').read_text())
        previous=item['state'].get('collector_terminal_started_unix')
        return previous is not None and resource.get('started_unix') not in [None,previous]
    except (OSError,ValueError,TypeError):return False


def reconcile_resume_marker(folder,state):
    """Replay rows after an archived rollback rather than skipping to an old tail."""
    try:
        marker=json.loads((folder/'resume_history/latest.json').read_text())
        if marker['generation']==state.get('last_resume_generation'):return state,None
        updated=dict(state)
        restored=int(marker['restored_step'])
        if int(updated.get('last_step',0))>restored:
            updated['last_step']=restored
            updated['probe_files']={};updated['validation_files']={}
            updated.pop('last_speed_step',None)
        updated['last_resume_generation']=marker['generation']
        return updated,marker
    except (OSError,ValueError,TypeError,KeyError):return state,None


def start_source(source,settings,wandb):
    config_path=ROOT/source['config'];cfg=yaml.safe_load(config_path.read_text())
    folder=ROOT/cfg['trainer']['output_dir']
    if not read_rows(folder/'train_log.jsonl'):return None
    mode=settings['mode']
    run_id=hashlib.sha256(str(folder.resolve()).encode()).hexdigest()[:12]
    state_path=ROOT/'logs/wandb_bridge'/f'{run_id}_{mode}.json'
    # Offline SDK runs cannot resume their internal history; rebuild a complete
    # snapshot from authoritative JSON on each offline collector start.
    state=json.loads(state_path.read_text()) if state_path.exists() and mode=='online' else {}
    # Reconcile before creating an SDK run. A failing API lookup must not leave
    # an untracked live SDK run behind on every historical-source retry.
    if mode=='online' and state_path.exists():
        remote=wandb.Api(timeout=10).run(f'{settings["entity"]}/{settings["project"]}/{run_id}')
        state=reconcile_submitted_state(state,remote.summary.get('last_uploaded_update',0))
        terminal=source_terminal_status(folder,cfg,state)
        expected='finished' if terminal and terminal['exit_code']==0 else 'failed'
        if state.get('collector_finished') and terminal and remote.state==expected:
            state['collector_terminal_started_unix']=terminal['started_unix']
            state['collector_exit_code']=terminal['exit_code']
            atomic_json(state_path,state)
            return dict(run=None,cfg=cfg,folder=folder,state=state,state_path=state_path,
                        url=remote.url,finished=True,source_config=source['config'])
    state.pop('collector_finished',None)
    state.pop('collector_exit_code',None)
    run=wandb.init(project=settings['project'],entity=settings.get('entity'),id=run_id,
                   name=cfg['run_id'],group=source.get('group','robocasa365-target50'),job_type=cfg['stage']['stage'],
                   resume='allow' if mode=='online' and state_path.exists() else None,mode=mode,reinit='create_new',
                   dir=settings['cache_dir'],config={**cfg,'tracking':{
                       'source':'JSON log sidecar','effective_global_batch':cfg['trainer']['per_device_batch_size'],
                       'validation':source.get('validation','joint action MSE, not success rate'),
                       'memory_scope':'rank0 process/GPU peaks; not total host memory',
                       'scheduled_lr':'derived from config, not sampled optimizer state',
                       'missing_metrics':'not retroactively invented'}},
                   settings=wandb.Settings(init_timeout=30,x_disable_stats=True,disable_code=True,save_code=False,console='off'))
    run.name=cfg['run_id']
    run.summary['source_training_status']='running'
    run.summary['source_training_completed']=False
    for axis in ['train/update','validation/update','checkpoint/update','probe/update']:run.define_metric(axis)
    for prefix in ['train/*','performance/*','memory/*','diagnostics/*','solver/*']:
        run.define_metric(prefix,step_metric='train/update')
    run.define_metric('validation/*',step_metric='validation/update')
    run.define_metric('checkpoint/*',step_metric='checkpoint/update')
    run.define_metric('probe/*',step_metric='probe/update')
    run.define_metric('comparison/sample_visits')
    run.define_metric('comparison/*',step_metric='comparison/sample_visits')
    print(json.dumps({'event':'started','mode':mode,'run_id':run_id,'url':run.url if mode=='online' else None}),flush=True)
    return dict(run=run,cfg=cfg,folder=folder,state=state,state_path=state_path,
                url=run.url if mode=='online' else None,source_config=source['config'])


def poll(item):
    if item.get('finished'):return
    run,cfg,folder,state=[item[k] for k in ['run','cfg','folder','state']]
    state,resume_marker=reconcile_resume_marker(folder,state)
    item['state']=state
    if resume_marker:
        restored=resume_marker['restored_step']
        run.log({'checkpoint/update':restored,'checkpoint/resume_from_update':restored})
        run.summary['source_resume_generation']=str(resume_marker['generation'])
        run.summary['source_resume_restored_update']=restored
        run.summary['source_superseded_logs_archived']=True
    rows=read_rows(folder/'train_log.jsonl')
    for step in sorted(rows):
        if step<=state.get('last_step',0):continue
        run.log(training_metrics(rows[step],step,cfg))
        state['last_step']=step
    run.summary['last_uploaded_update']=state.get('last_step',0)
    # The speed report contains extra timings for its most recent 100 updates.
    # Publish only the newest row on the training axis; do not pretend older fields existed.
    try:
        speed=json.loads((folder/'speed_report.json').read_text())
        latest=speed['steps'][-1];step=completed_step(latest)
        if step==state.get('last_step') and step!=state.get('last_speed_step'):
            run.log(training_metrics(latest,step,cfg));state['last_speed_step']=step
    except (OSError,ValueError,IndexError,KeyError,TypeError):pass
    seen=state.setdefault('validation_files',{})
    for file in sorted(folder.glob('val_step*.json')):
        digest=hashlib.sha256(file.read_bytes()).hexdigest()
        if seen.get(file.name)==digest:continue
        try:
            report=json.loads(file.read_text());step=int(file.stem.removeprefix('val_step'))
            metrics={'validation/update':step,'validation/joint_action_mse':report['offline_action_mse'],
                     'validation/num_windows':report['num_samples']}
            for key,value in report.get('per_task_action_mse',{}).items():
                task=key.split('/')[-2] if '/' in key else key
                metrics[f'validation/task/{task}/joint_action_mse']=value
            run.log(metrics);seen[file.name]=digest
        except (ValueError,KeyError,TypeError):continue
    probes=state.setdefault('probe_files',{})
    for file in sorted(list(folder.glob('eval_*.json'))+list((folder/'train_monitor').glob('eval_*.json'))):
        file_key=str(file.relative_to(folder))
        digest=hashlib.sha256(file.read_bytes()).hexdigest()
        if probes.get(file_key)==digest:continue
        try:
            report=json.loads(file.read_text());groups={}
            for row in report['rows']:
                key=(row['split'],row['direction'],row['k'])
                groups.setdefault(key,[]).append(row['prediction_mse'])
            metrics={'probe/update':report['rows'][0]['update']}
            if cfg['trainer'].get('per_device_batch_size'):metrics['comparison/sample_visits']=report['rows'][0]['update']*cfg['trainer']['per_device_batch_size']
            for (split,direction,k),values in groups.items():
                metrics[f'probe/{split}/{direction}_mse_evalK{k}']=sum(values)/len(values)
                if k==cfg['solver'].get('eval_inner_steps',2):metrics[f'comparison/{split}_fixed_{direction}_mse']=sum(values)/len(values)
            try:
                baselines=json.loads((folder/'dataset_baselines.json').read_text())
                for split,directions in baselines.items():
                    for direction,values in directions.items():
                        for key,value in values.items():
                            if key.endswith('_mse'):
                                metrics[f'probe/{split}/baseline_{direction}_{key}']=value
            except (OSError,ValueError,TypeError):pass
            run.log(metrics);probes[file_key]=digest
        except (OSError,ValueError,KeyError,IndexError,TypeError):continue
    try:
        meta_path=folder/'ckpt_latest.pt.meta.json'
        if not meta_path.exists():meta_path=folder/'retired_full_checkpoint.meta.json'
        meta=json.loads(meta_path.read_text());step=meta['meta']['global_step']
        recoverable=(folder/'ckpt_latest.pt').exists()
        # Read-only remote mirrors never copy large checkpoint files.
        inventory_path=folder/'remote_checkpoint_inventory.json'
        if inventory_path.exists():
            inventory=json.loads(inventory_path.read_text())
            recoverable=bool(inventory.get('latest_exists',False))
            run.summary['checkpoint_storage_host']=inventory.get('hostname')
            run.summary['checkpoint_storage_folder']=inventory.get('folder')
        if step!=state.get('saved_step') or recoverable!=state.get('checkpoint_recoverable'):
            run.log({'checkpoint/update':step,'checkpoint/latest_saved_update':step,
                     'checkpoint/exact_resume_available':int(recoverable)})
            state['saved_step']=step;state['checkpoint_recoverable']=recoverable
        run.summary['checkpoint_reference_only']=(folder/'model_reference.pt').exists() and not recoverable
    except (OSError,ValueError,KeyError):pass
    atomic_json(item['state_path'],state)
    terminal=source_terminal_status(folder,cfg,state)
    if terminal:
        run.summary['source_training_completed']=terminal['exit_code']==0
        run.summary['source_training_status']=terminal['status']
        run.summary['source_exit_classification']=terminal['classification']
        run.summary['source_exit_reason']=terminal['reason']
        run.finish(exit_code=terminal['exit_code'])
        state['collector_finished']=True
        state['collector_exit_code']=terminal['exit_code']
        state['collector_terminal_started_unix']=terminal['started_unix']
        item['finished']=True
        atomic_json(item['state_path'],state)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--settings',default='configs/wandb_bridge.json')
    parser.add_argument('--once',action='store_true');args=parser.parse_args()
    settings=json.loads((ROOT/args.settings).read_text())
    apply_proxy_settings(settings)
    while settings['cache_dir'].startswith('/cloud/') and not os.path.ismount('/cloud'):
        if STOP:return
        time.sleep(5)
    Path(settings['cache_dir']).mkdir(parents=True,exist_ok=True)
    if settings.get('api_key_file'):
        secret=Path(settings['api_key_file'])
        if settings['mode']=='online':
            if not secret.is_file():raise RuntimeError('Project W&B key file missing; refusing shared-account fallback')
            if secret.stat().st_mode & 0o077:raise RuntimeError('Project W&B key must have permissions 600')
            key=secret.read_text().strip()
            if not key:raise RuntimeError('Empty project W&B key')
            os.environ['WANDB_API_KEY']=key
    os.environ['WANDB_BASE_URL']=settings.get('base_url','https://api.wandb.ai')
    if not wait_for_proxy(settings):return
    import wandb
    items=[];active=set();retry_at={}
    try:
        while not STOP:
            if not wait_for_proxy(settings):break
            # Reload only experiment sources; account/project credentials stay
            # fixed for this collector session.
            try:sources=json.loads((ROOT/args.settings).read_text())['sources']
            except (OSError,ValueError,KeyError):sources=settings['sources']
            # A backlog of unreachable historical runs must not block current
            # training logs behind dozens of sequential SDK/API timeouts.
            retained=[]
            for item in items:
                if source_session_changed(item):
                    active.discard(item['source_config'])
                elif remote_stream_needs_reopen(item,settings,wandb):
                    active.discard(item['source_config'])
                    retry_at.pop(item['source_config'],None)
                else:retained.append(item)
            items=retained
            for item in items:poll(item)
            attempts=0
            for source in sorted(sources,key=lambda s:s.get('priority',0),reverse=True):
                if attempts>=settings.get('max_source_starts_per_cycle',1):break
                if source['config'] not in active and time.monotonic()>=retry_at.get(source['config'],0):
                    try:
                        item=start_source(source,settings,wandb)
                        if item:
                            items.append(item);active.add(source['config']);attempts+=1
                    except Exception as error:
                        attempts+=1
                        retry_at[source['config']]=time.monotonic()+120
                        print(json.dumps({'event':'source_retry','config':source['config'],
                                          'error_type':type(error).__name__}),flush=True)
            for item in items:poll(item)
            atomic_json(ROOT/settings.get('status_file','logs/wandb_bridge_status.json'),{
                'mode':settings['mode'],'updated_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                'runs':[{'name':i['cfg']['run_id'],'url':i['url'],**i['state']} for i in items]})
            if args.once:break
            time.sleep(settings.get('poll_seconds',15))
    finally:
        for item in items:
            if not item.get('finished'):item['run'].finish()


def stop(*_):
    global STOP
    STOP=True

if __name__=='__main__':
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    main()

"""One-off completed-run repair using an independently verified server ACK."""
from pathlib import Path
import hashlib
import json
import os
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import watch_wandb as bridge


def main():
    settings = json.loads((ROOT / 'configs/lab_wandb_bridge.json').read_text())
    folder = ROOT / 'logs/lab_wandb_mirror/steps/episode3346_action_half_seed42'
    config = folder / 'config.yaml'
    cfg = yaml.safe_load(config.read_text())
    rid = hashlib.sha256(str(folder.resolve()).encode()).hexdigest()[:12]
    server = json.loads((ROOT / f'logs/lab_remote_setup/wandb_server_history_{rid}.json').read_text())
    assert time.time() - server['checked_unix'] < 1800, 'Refresh server evidence first'
    remote = server['run']
    summary = json.loads(remote['summaryMetrics'])
    assert remote['name'] == rid and remote['state'] == 'crashed'
    ack = int(summary['last_uploaded_update'])
    assert ack == 111
    resource = json.loads((folder / 'resource_summary.json').read_text())
    assert resource['returncode'] == 0, 'Source experiment must have completed'
    state_path = ROOT / f'logs/wandb_bridge/{rid}_online.json'
    original = json.loads(state_path.read_text())
    state = bridge.reconcile_submitted_state(original, ack)
    state.pop('collector_finished', None)
    state.pop('collector_exit_code', None)
    receipt = ROOT / 'logs/lab_remote_setup/action_half_wandb_isolated_repair.json'
    bridge.atomic_json(receipt, dict(started_unix=time.time(), pid=os.getpid(), run_id=rid,
                                    verified_server_ack=ack, original_local_state=original,
                                    training_restarted=False, phase='starting'))
    os.environ['WANDB_API_KEY'] = Path(settings['api_key_file']).read_text().strip()
    os.environ['WANDB_BASE_URL'] = settings.get('base_url', 'https://api.wandb.ai')
    for name in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']:
        os.environ[name] = 'http://127.0.0.1:17902'
    import wandb
    run = wandb.init(entity=settings['entity'], project=settings['project'], id=rid,
                     resume='must', name=cfg['run_id'], dir=settings['cache_dir'],
                     settings=wandb.Settings(init_timeout=60, x_disable_stats=True,
                                             disable_code=True, save_code=False, console='off'))
    for axis in ['train/update', 'validation/update', 'checkpoint/update', 'probe/update']:
        run.define_metric(axis)
    for prefix in ['train/*', 'performance/*', 'memory/*', 'diagnostics/*', 'solver/*']:
        run.define_metric(prefix, step_metric='train/update')
    run.define_metric('probe/*', step_metric='probe/update')
    run.define_metric('validation/*', step_metric='validation/update')
    run.define_metric('checkpoint/*', step_metric='checkpoint/update')
    run.define_metric('comparison/sample_visits')
    run.define_metric('comparison/*', step_metric='comparison/sample_visits')
    item = dict(run=run, cfg=cfg, folder=folder, state=state, state_path=state_path,
                url=run.url, source_config=str(config))
    bridge.poll(item)
    bridge.atomic_json(receipt, dict(ended_unix=time.time(), run_id=rid,
                                    verified_server_ack_before=ack, submitted_through=state.get('last_step'),
                                    phase='sdk_finished', needs_server_verification=True,
                                    training_restarted=False))


if __name__ == '__main__':
    main()

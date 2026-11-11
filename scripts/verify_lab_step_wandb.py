"""Read-only server history verification against the mirrored raw experiment logs."""
import hashlib
import json
import os
from pathlib import Path
import time

import requests

ROOT = Path(__file__).resolve().parents[1]


def main():
    settings = json.loads((ROOT / 'configs/lab_wandb_bridge.json').read_text())
    key = Path(settings['api_key_file']).read_text().strip()
    session = requests.Session()
    session.trust_env = False
    session.proxies = {'https': os.environ.get('ENERGYWAM_WANDB_PROXY', 'http://127.0.0.1:17902')}
    results = []
    for study, name in [('steps', 'episode3346_action_half_seed42'),
                        ('future67', 'episode3346_future_half_gpu67_original_seed42')]:
        folder = ROOT / 'logs/lab_wandb_mirror' / study / name
        rid = hashlib.sha256(str(folder.resolve()).encode()).hexdigest()[:12]
        query = '''query($entity:String!,$project:String!,$run:String!){
          project(name:$project,entityName:$entity){run(name:$run){
            name state summaryMetrics history(minStep:0,maxStep:2000,samples:2000)
          }}}'''
        for attempt in range(4):
            try:
                response = session.post(settings.get('base_url', 'https://api.wandb.ai') + '/graphql',
                                        json={'query': query, 'variables': {
                                            'entity': settings['entity'], 'project': settings['project'], 'run': rid}},
                                        auth=('api', key), timeout=(5, 25))
                response.raise_for_status()
                break
            except requests.RequestException as error:
                print(f'{rid}: network attempt {attempt + 1}: {type(error).__name__}', flush=True)
                if attempt == 3:
                    raise
                time.sleep(2)
        response.raise_for_status()
        data = response.json()
        if data.get('errors'):
            raise RuntimeError('GraphQL history query failed')
        run = data['data']['project']['run']
        if run is None:
            raise RuntimeError(f'Run {rid} not present')
        history = [json.loads(row) if isinstance(row, str) else row for row in run['history']]
        summary = json.loads(run['summaryMetrics'])
        assert int(summary.get('_step', 0)) < 2000, 'Increase history pagination bound'
        (ROOT / f'logs/lab_remote_setup/wandb_server_history_{rid}.json').write_text(
            json.dumps({'checked_unix': time.time(), 'run': run}, indent=2))
        actual = {}
        for row in history:
            if row.get('train/update') is not None:
                for direction in ['action', 'future']:
                    metric = f'train/{direction}_mse'
                    if row.get(metric) is not None:
                        actual[('train', int(row['train/update']), metric)] = row[metric]
            if row.get('probe/update') is not None:
                for metric, value in row.items():
                    if metric.startswith(('probe/train/', 'probe/val/')) and '_mse_evalK' in metric and value is not None:
                        actual[('probe', int(row['probe/update']), metric)] = value
        expected = {}
        for line in (folder / 'train_log.jsonl').read_text().splitlines():
            row = json.loads(line)
            step = int(row['global_step_completed']) if 'global_step_completed' in row else int(row['global_step']) + 1
            for direction in ['action', 'future']:
                expected[('train', step, f'train/{direction}_mse')] = row[f'bi/{direction}_loss']
        files = list(folder.glob('eval_*.json')) + list((folder / 'train_monitor').glob('eval_*.json'))
        for file in files:
            groups = {}
            for row in json.loads(file.read_text())['rows']:
                key2 = ('probe', int(row['update']), f"probe/{row['split']}/{row['direction']}_mse_evalK{row['k']}")
                groups.setdefault(key2, []).append(row['prediction_mse'])
            expected.update({k: sum(values) / len(values) for k, values in groups.items()})
        missing = [k for k in expected if k not in actual]
        mismatches = [k for k, v in expected.items() if k in actual and abs(actual[k] - v) > 1e-8 * max(1., abs(v))]
        result = dict(name=name, run_id=rid, state=run['state'], checked_unix=time.time(),
                      url=f"https://wandb.ai/{settings['entity']}/{settings['project']}/runs/{rid}",
                      server_last_uploaded_update=summary.get('last_uploaded_update'),
                      server_completed=summary.get('source_training_completed'), history_rows=len(history),
                      train_updates=len({k[1] for k in expected if k[0] == 'train'}),
                      expected_metric_points=len(expected), missing=missing, mismatches=mismatches,
                      passed=not missing and not mismatches and run['state'] == 'finished'
                      and summary.get('last_uploaded_update') == 210)
        results.append(result)
        print(json.dumps(result), flush=True)
    (ROOT / 'logs/lab_remote_setup/steps_wandb_history_verified_20261111.json').write_text(
        json.dumps(results, indent=2))


if __name__ == '__main__':
    main()

"""Read-only verification of W&B history against one authoritative JSON source."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]


def expected_metrics(folder):
    expected = {}
    for line in (folder / 'train_log.jsonl').read_text().splitlines():
        row = json.loads(line)
        step = int(row.get('global_step_completed', int(row['global_step']) + 1))
        for source, metric in [('bi/action_loss', 'train/action_mse'),
                               ('bi/future_loss', 'train/future_mse'),
                               ('param_grad_norm', 'diagnostics/parameter_grad_norm')]:
            expected[('train', step, metric)] = row[source]
    for path in list(folder.glob('eval_*.json')) + list((folder / 'train_monitor').glob('eval_*.json')):
        groups = {}
        for row in json.loads(path.read_text())['rows']:
            key = ('probe', int(row['update']),
                   f"probe/{row['split']}/{row['direction']}_mse_evalK{row['k']}")
            groups.setdefault(key, []).append(row['prediction_mse'])
        expected.update({key: sum(values) / len(values) for key, values in groups.items()})
    return expected


def verify(settings, folder, expected):
    rid = hashlib.sha256(str(folder.resolve()).encode()).hexdigest()[:12]
    session = requests.Session()
    session.trust_env = False
    if settings.get('proxy_url'):
        session.proxies = {'https': settings['proxy_url']}
    query = '''query($entity:String!,$project:String!,$run:String!){
      project(name:$project,entityName:$entity){run(name:$run){
        name state summaryMetrics history(minStep:0,maxStep:2000,samples:2000)
      }}}'''
    response = session.post(settings.get('base_url', 'https://api.wandb.ai') + '/graphql',
                            json={'query': query, 'variables': {'entity': settings['entity'],
                                  'project': settings['project'], 'run': rid}},
                            auth=('api', Path(settings['api_key_file']).read_text().strip()),
                            timeout=(4, 20))
    response.raise_for_status()
    data = response.json()
    if data.get('errors'):
        raise RuntimeError('GraphQL query failed')
    run = data['data']['project']['run']
    if run is None:
        raise RuntimeError('Existing run not found')
    summary = json.loads(run['summaryMetrics'])
    if int(summary.get('_step', 0)) >= 2000:
        raise RuntimeError('History exceeds verification bound')
    actual = {}
    for raw in run['history']:
        row = json.loads(raw) if isinstance(raw, str) else raw
        for kind, axis in [('train', 'train/update'), ('probe', 'probe/update')]:
            if row.get(axis) is None:
                continue
            for metric, value in row.items():
                key = (kind, int(row[axis]), metric)
                if key in expected and value is not None:
                    actual[key] = value
    missing = [key for key in expected if key not in actual]
    mismatch = [key for key, value in expected.items() if key in actual
                and abs(actual[key] - value) > 1e-8 * max(1.0, abs(value))]
    target = max(key[1] for key in expected if key[0] == 'train')
    return {'checked_unix': time.time(), 'run_id': rid,
            'url': f"https://wandb.ai/{settings['entity']}/{settings['project']}/runs/{rid}",
            'state': run['state'], 'source_folder': str(folder), 'expected_updates': target,
            'expected_metric_points': len(expected), 'missing': missing, 'mismatches': mismatch,
            'server_last_uploaded_update': summary.get('last_uploaded_update'),
            'server_completed': summary.get('source_training_completed'),
            'passed': not missing and not mismatch and run['state'] == 'finished'
                      and summary.get('last_uploaded_update') == target
                      and summary.get('source_training_completed') is True,
            'training_restarted': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settings', default='configs/wandb_bridge.json')
    parser.add_argument('--source-config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--wait-seconds', type=int, default=0)
    parser.add_argument('--local-only', action='store_true')
    args = parser.parse_args()
    settings = json.loads((ROOT / args.settings).read_text())
    cfg = yaml.safe_load((ROOT / args.source_config).read_text())
    folder = ROOT / cfg['trainer']['output_dir']
    expected = expected_metrics(folder)
    if args.local_only:
        print(json.dumps({'expected_metric_points': len(expected),
                          'train_updates': len({key[1] for key in expected if key[0] == 'train'})}))
        return
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + args.wait_seconds
    while True:
        try:
            result = verify(settings, folder, expected)
            # Preserve a successful server response even if the next network
            # attempt fails; a timeout must not erase the last observed ACK.
            server_path = output.with_name(output.stem + '_last_server.json')
            server_tmp = server_path.with_suffix('.tmp')
            server_tmp.write_text(json.dumps(result, indent=2) + '\n')
            server_tmp.replace(server_path)
        except Exception as error:
            # SDK/network exception messages can contain sensitive URLs.
            result = {'checked_unix': time.time(), 'passed': False,
                      'error_type': type(error).__name__, 'phase': 'waiting_for_server_verification'}
        tmp = output.with_suffix('.tmp')
        tmp.write_text(json.dumps(result, indent=2) + '\n')
        tmp.replace(output)
        print(json.dumps({key: value for key, value in result.items()
                          if key not in ['missing', 'mismatches']}), flush=True)
        if result['passed']:
            return
        if time.monotonic() >= deadline:
            raise SystemExit(1)
        time.sleep(min(30, max(0, deadline - time.monotonic())))


if __name__ == '__main__':
    main()

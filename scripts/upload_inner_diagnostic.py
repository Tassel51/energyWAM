#!/usr/bin/env python
"""Upload only small diagnostic metrics/reports to an isolated W&B run."""
import argparse
import hashlib
import json
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    out=Path(args.output);settings=json.load(open(ROOT/'configs/wandb_bridge.json'))
    os.environ['WANDB_API_KEY']=Path(settings['api_key_file']).read_text().strip()
    os.environ['WANDB_BASE_URL']=settings.get('base_url','https://api.wandb.ai')
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import wandb
    report=json.load(open(out/'report.json'));experiment=json.load(open(out/'experiment_config.json'))
    reference=json.load(open(out/'reference.json'))
    run_id=hashlib.sha256(str(out.resolve()).encode()).hexdigest()[:12]
    run=wandb.init(project=settings['project'],entity=settings['entity'],id=run_id,
                   name=f"{experiment['experiment']}_bi_step{reference['checkpoint_step']}",group='inner-diagnostics',
                   job_type='diagnostic-no-training',dir=settings['cache_dir'],
                   config={'experiment':experiment,'reference':reference,'weight_updates':0,
                           'trained_k':2,'diagnostic_max_k':experiment['max_k'],'not_task_success_rate':True},
                   settings=wandb.Settings(x_disable_stats=True,disable_code=True,save_code=False,console='off'))
    run.define_metric('diagnostic/k')
    run.define_metric('diagnostic/*',step_metric='diagnostic/k')
    for k in experiment['record_k']:
        metrics={'diagnostic/k':k}
        for row in report['aggregates']:
            if row['k']!=k:continue
            prefix=f"diagnostic/{row['split']}/{row['direction']}"
            for key,value in row.items():
                if isinstance(value,(int,float)) and key!='k':metrics[f'{prefix}/{key}']=value
        run.log(metrics)
    rows=[]
    for direction in ['action','future']:
        rows.extend(json.loads(line) for line in (out/f'trace_{direction}.jsonl').read_text().splitlines())
    columns=list(rows[0]);run.log({'diagnostic/traces':wandb.Table(columns=columns,data=[[r[c] for c in columns] for r in rows])})
    artifact=wandb.Artifact(f'inner-diagnostic-{run_id}',type='evaluation')
    for name in ['report.json','sample_manifest.json','experiment_config.json','reference.json','traces.csv']:
        artifact.add_file(str(out/name))
    if (out/'comparison.json').exists():artifact.add_file(str(out/'comparison.json'))
    run.log_artifact(artifact)
    run.summary.update({'weight_updates':0,'completed_windows':40,'conditional_trajectories':160,
                        'checkpoint_step':reference['checkpoint_step'],'not_success_rate':True})
    url=run.url;run.finish()
    (out/'wandb.json').write_text(json.dumps({'url':url,'id':run_id},indent=2))
    print(url,flush=True)


if __name__=='__main__':main()

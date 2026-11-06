"""Upload only compact K comparison reports, never weights or raw dataset."""
import hashlib
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
IS_SHARDED=os.environ.get('ENERGYWAM_K_SHARDED')=='1'
OUT=Path('/cloud/zzc/energyWAM/experiments/train_k248_sharded_20261103' if IS_SHARDED else '/cloud/zzc/energyWAM/experiments/train_k248_20261102')

def main():
    settings=json.loads((ROOT/'configs/wandb_bridge.json').read_text())
    secret=Path(settings['api_key_file'])
    if secret.stat().st_mode & 0o077:raise RuntimeError('W&B key file must be private')
    os.environ.update(WANDB_API_KEY=secret.read_text().strip(),WANDB_BASE_URL=settings['base_url'],CUDA_VISIBLE_DEVICES='')
    for name in ['HTTP_PROXY','HTTPS_PROXY','http_proxy','https_proxy']:os.environ[name]='http://127.0.0.1:7901'
    import wandb
    report=json.loads((OUT/'comparison.json').read_text())
    run=wandb.init(entity=settings['entity'],project=settings['project'],
        id=hashlib.sha256((str(OUT)+'-report').encode()).hexdigest()[:12],resume='allow',
        name=OUT.name+'-comparison',group=OUT.name,job_type='comparison',
        dir=settings['cache_dir'],config={'training_windows':20,'validation_windows':20,'effective_batch':2,'test_used':False},
        settings=wandb.Settings(x_disable_stats=True,disable_code=True,save_code=False,console='off'))
    table=wandb.Table(columns=['train_K','updates','eval_K','direction','validation_MSE','median_training_seconds'])
    for k,g in report['groups'].items():
        for ek in [2,4,8]:
            for direction in ['action','future']:
                values=[r for r in g['evaluations'] if r['split']=='val' and r['eval_k']==ek and r['direction']==direction]
                if values:
                    row=max(values,key=lambda r:r['update'])
                    table.add_data(int(k),row['update'],ek,direction,row['mse'],g['median_step_seconds'])
        run.summary[f'K{k}/status']=g['status'].get('phase','unknown')
        run.summary[f'K{k}/updates']=g['updates']
    run.log({'comparison/final_validation':table})
    artifact=wandb.Artifact(OUT.name,type='experiment-report')
    for name in ['comparison.json','RESULTS_CN.md','status.json','sample_manifest.json','complete.json']:
        artifact.add_file(str(OUT/name))
    run.log_artifact(artifact)
    url=run.url;run.finish()
    (OUT/'wandb_comparison.json').write_text(json.dumps({'url':url},indent=2))

if __name__=='__main__':main()

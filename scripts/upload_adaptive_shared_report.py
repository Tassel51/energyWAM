"""Publish compact adaptive experiment evidence to the user's W&B project."""
import os,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=Path('/cloud/zzc/energyWAM/experiments/adaptive_shared_20261103')

def main():
    settings=json.loads((ROOT/'configs/wandb_bridge.json').read_text())
    secret=Path(settings['api_key_file'])
    if secret.stat().st_mode&0o077:raise RuntimeError('key file must be private')
    os.environ.update(WANDB_API_KEY=secret.read_text().strip(),WANDB_BASE_URL=settings['base_url'],CUDA_VISIBLE_DEVICES='')
    for n in ['HTTP_PROXY','HTTPS_PROXY','http_proxy','https_proxy']:os.environ[n]='http://127.0.0.1:7901'
    import wandb
    data=json.loads((OUT/'report.json').read_text())
    run=wandb.init(entity=settings['entity'],project=settings['project'],name='adaptive-one-model-20261103-report',
        id=hashlib.sha256(str(OUT).encode()).hexdigest()[:12],resume='allow',group='adaptive-one-model-20261103',job_type='comparison',
        dir=settings['cache_dir'],config={'scope':data['scope'],'test_used':False,'samples_per_candidate':80},
        settings=wandb.Settings(x_disable_stats=True,disable_code=True,save_code=False,console='off'))
    table=wandb.Table(columns=['trial','train_K','batch','alpha_future','windows','validation_action_MSE','validation_future_MSE','training_samples_per_second'])
    for r in data['results']:
        table.add_data(r['name'],r['train_K'],r['batch'],r['alpha_future'],r['window_visits'],
            r['metrics']['final/val/action/K2'],r['metrics']['final/val/future/K2'],r['samples_per_second'])
    run.log({'comparison/trials':table});run.summary['conclusion']=data['reason'];run.summary['recommendation']=data['recommendation']
    artifact=wandb.Artifact('adaptive-one-model-20261103',type='experiment-report')
    for name in ['report.json','RESULTS_CN.md','profiles.json','future_scale.json','sample_manifest.json','expanded_comparison.json','expanded_val_manifest.json','future_only_diagnostic.json']:
        if (OUT/name).exists():artifact.add_file(str(OUT/name))
    run.log_artifact(artifact);url=run.url;run.finish()
    (OUT/'wandb_report.json').write_text(json.dumps({'url':url},indent=2))

if __name__=='__main__':main()

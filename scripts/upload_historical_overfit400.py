"""One-shot backfill of the completed400-update run; never imports the trainer."""
import argparse,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import watch_wandb as bridge

def main():
 p=argparse.ArgumentParser();p.add_argument('--settings',default='configs/wandb_bridge.json');a=p.parse_args();settings=json.loads((ROOT/a.settings).read_text())
 if settings['mode']!='online':raise ValueError('This backfill expects an explicitly configured online account')
 secret=Path(settings['api_key_file'])
 if secret.stat().st_mode&0o077:raise ValueError('Project key must have permissions600')
 os.environ['WANDB_API_KEY']=secret.read_text().strip();os.environ['WANDB_BASE_URL']=settings.get('base_url','https://api.wandb.ai')
 ready,reason=bridge.upload_route_ready(settings)
 if not ready:raise RuntimeError('Upload route unavailable: '+reason)
 source={'config':'configs/experiments/ablation_prior1_currentinit1_seed42.yaml','group':'historical-tiny16-overfit400','validation':'Historical completed400-update run.16TRAIN/16VAL windows; original logs replayed, no training rerun. Raw unweighted batch MSE each update; fixed assessments every20updates.'}
 import yaml
 cfg=yaml.safe_load((ROOT/source['config']).read_text());folder=Path(cfg['trainer']['output_dir'])
 rows=bridge.read_rows(folder/'train_log.jsonl')
 if set(rows)!=set(range(1,401)) or json.loads((folder/'final_report.json').read_text())['global_step']!=400:raise ValueError('Historical completion evidence does not match400updates')
 import wandb
 item=bridge.start_source(source,settings,wandb)
 run=item['run'];run.name='historical_overfit400_currentinit_dualprior_seed42'
 bridge.poll(item)
 run.summary['historical_log_backfill']=True
 run.summary['source_training_completed']=True
 run.summary['source_training_status']='completed_historical_backfill'
 run.summary['source_weight_updates_during_upload']=0
 run.summary['checkpoint_files_uploaded']=False
 run.summary['historical_run_directory']=str(folder)
 run.finish()
 item['state']['collector_finished']=True;item['state']['collector_exit_code']=0
 bridge.atomic_json(item['state_path'],item['state'])
 report={'uploaded_unix':time.time(),'url':item['url'],'name':'historical_overfit400_currentinit_dualprior_seed42','submitted_updates':item['state']['last_step'],'weight_updates_during_upload':0,'checkpoint_files_uploaded':False,'fixed_eval_cadence':20}
 bridge.atomic_json(folder/'wandb_historical_backfill.json',report)
 print(json.dumps(report),flush=True)
if __name__=='__main__':main()

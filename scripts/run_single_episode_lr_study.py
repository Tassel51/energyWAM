"""Two-wave, four-arm single-episode overfit LR study with durable records."""
import argparse,copy,hashlib,json,os,shutil,signal,subprocess,sys,time,zipfile
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
OUT=Path('/cloud/zzc/energyWAM/experiments/single_episode_lr_20261108')
OLD=Path('/cloud/zzc/energyWAM/experiments/lr256_comparison_20261107')
SNAPSHOT=24569214417
STOP=False
CHILDREN=[]

def read(p):return json.loads(Path(p).read_text())
def write(p,data):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.pending.json');tmp.write_text(json.dumps(data,indent=2));tmp.replace(p)
def stop(*_):
 global STOP
 STOP=True
 for c in CHILDREN:
  if c.poll() is None:c.send_signal(signal.SIGTERM)

def prepare_manifest():
 from energywam.config import load_config
 from energywam.data.energy_dataset import resolve_dataset_root,LeRobotEpisodeReader
 cfg=load_config(ROOT/'configs/experiments/resume256_lr03.yaml');reader=LeRobotEpisodeReader(resolve_dataset_root(cfg))
 train_ids=set(read(ROOT/'manifests/train_episodes.json')['episode_indices']);val_ids=set(read(ROOT/'manifests/val_episodes.json')['episode_indices'])
 ids=[3346,3423,3483];assert ids[0] in train_ids and all(i in val_ids for i in ids[1:])
 rows=[];episodes=[]
 for ep in ids:
  e=next(x for x in reader.episodes if x['episode_index']==ep);assert e['source_dataset'].split('/')[-2]=='OpenDrawer';episodes.append(e)
  end=e['length']-17;starts=sorted(set([*range(0,end+1,4),end])) if ep==3346 else [round(end*j/7) for j in range(8)]
  assert len(starts)==len(set(starts))
  for frame in starts:rows.append(dict(window_id=len(rows),episode_index=ep,frame_index=frame,split='train' if ep==3346 else 'val',task='OpenDrawer',source_dataset=e['source_dataset']))
 manifest=dict(dataset_identity=reader.dataset_identity,stream_training=False,windows=rows)
 path=OUT/'train_val_manifest.json'
 if path.exists():assert read(path)==manifest
 else:write(path,manifest)
 write(OUT/'train_monitor_manifest.json',dict(dataset_identity=reader.dataset_identity,windows=[x for x in rows if x['split']=='train']))
 return len([x for x in rows if x['split']=='train']),episodes

def prepare_arm(scale,pair,steps=210):
 pair='2,3'  # User restriction2026-11-08: GPUs0/1 belong to others.
 tag=format(scale,'.4g').replace('.','p');name=f'episode3346_lrscale{tag}_b16_micro4_seed42';folder=OUT/name;folder.mkdir(exist_ok=True)
 cfg=yaml.safe_load((ROOT/'configs/experiments/three_task_b16_micro4_seed42.yaml').read_text())
 cfg['run_id']=name;cfg['notes']='Single OpenDrawer TRAIN episode3346,67 stride4 windows;16 auxiliary VAL windows from3423/3483. Fresh identical seed42/pretrained Qwen, no three-task checkpoint. Four outer LR groups jointly scaled; K2/B16/native4/current-DINO init/dual priors/lambdaZ9 unchanged. Primary purpose TRAIN fitting; not generalization selection.'
 cfg['trainer'].update(output_dir=str(folder),max_train_steps=steps,save_interval=20,checkpoint_interval_seconds=1800)
 for key in ['learning_rate','task_learning_rate','energy_head_learning_rate','step_size_learning_rate']:cfg['trainer'][key]*=scale
 config=ROOT/f'configs/experiments/{name}.yaml'
 if config.exists():assert yaml.safe_load(config.read_text())==cfg
 else:config.write_text(yaml.safe_dump(cfg,sort_keys=False))
 job=dict(name=name,config=str(config.relative_to(ROOT)),physical_gpus=pair,steps=steps,manifest=str(OUT/'train_val_manifest.json'),train_monitor_manifest=str(OUT/'train_monitor_manifest.json'),sample_eval_interval=20*16,train_monitor_sample_interval=20*16,alternate_final_eval_seed=2027,keep_best=True,monitor_resources=True,effective_batch=16,guard_min_sample_visits=20*16,max_fixed_mse_ratio=10,max_process_uss_gib=140,max_wall_seconds=12*3600,snapshot_bytes_estimate=SNAPSHOT)
 with zipfile.ZipFile(folder/'study_source.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
  for p in list((ROOT/'energywam').rglob('*.py'))+[ROOT/'scripts'/n for n in ['train_shared_sharded.py','k_training_probe.py','run_iteration_search.py','run_single_episode_lr_study.py']]:z.write(p,str(p.relative_to(ROOT)))
  z.write(config,'experiment_config.yaml')
 write(folder/'arm_definition.json',dict(scale=scale,job=job,hyperparameters=cfg,manifest_sha256=hashlib.sha256((OUT/'train_val_manifest.json').read_bytes()).hexdigest(),initialization='Fresh pretrained Qwen and new modules, seed42; no prior robot-finetuned checkpoint'))
 bridge=ROOT/'configs/wandb_bridge.json';settings=read(bridge)
 if not any(s['config']==job['config'] for s in settings['sources']):
  settings['sources'].append(dict(config=job['config'],group='single-episode3346-lr-20261108',priority=200))
  temp=bridge.with_suffix('.pending.json');temp.write_text(json.dumps(settings,indent=2));temp.chmod(0o600);temp.replace(bridge)
 return job

def safe_stop_old():
 """Stop owned controllers immediately after their next complete rolling save."""
 import torch
 baseline={l:read(OLD/f'resume256_lr{l}/ckpt_latest.pt.meta.json')['meta']['global_step'] for l in ['03','01']}
 record=read(OUT/'old_training_pause.json') if (OUT/'old_training_pause.json').exists() else {}
 while len(record)<2 and not STOP:
  for l in ['03','01']:
   if l in record:continue
   folder=OLD/('resume256_lr'+l);meta=read(folder/'ckpt_latest.pt.meta.json');step=meta['meta']['global_step']
   status=read(OLD/f'status_{l}.json');phase=status.get('resume256_lr'+l+'_to2048',{}).get('phase')
   completed=read(folder/'speed_report.json')['completed_steps']
   # New metadata commits only after verified payload write. Pause on the
   # first polling tick before any subsequent completed update is lost.
   if (step>baseline[l] and completed<=step) or phase in ['complete','failed','interrupted']:
    command=['supervisorctl','stop','energywam-lr256-'+l]
    result=subprocess.run(command,text=True,capture_output=True)
    path=folder/'ckpt_latest.pt';payload=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
    assert all(k in payload for k in ['model','optimizer','scheduler','rng','meta'])
    saved=payload['meta']['global_step'];final_completed=read(folder/'speed_report.json')['completed_steps']
    if saved!=final_completed:raise RuntimeError(f'Pause did not retain every completed update: {l} {saved} vs {final_completed}')
    record[l]=dict(saved_step=saved,completed_step=final_completed,checkpoint=str(path),best=read(folder/'best_selection.json'),supervisor_returncode=result.returncode,stopped_at_unix=time.time())
    write(OUT/'old_training_pause.json',record);del payload
  write(OUT/'controller_status.json',dict(phase='waiting_safe_old_checkpoint',paused=record,baseline_steps=baseline))
  time.sleep(.5)
 if STOP:raise SystemExit('Stopped while waiting for safe checkpoints')


def retire_weights(folder,reason,manifest_path):
 """Only completed, explicitly rejected short trials; all nonweights retained."""
 folder=Path(folder);manifest_path=Path(manifest_path)
 if manifest_path.exists():return
 protected={};records=[];free=shutil.disk_usage(folder).free
 for p in folder.rglob('*'):
  if p.is_file() and not p.is_symlink():
   if p.suffix=='.pt' and p.stat().st_size>1024**3:records.append(dict(path=str(p),bytes=p.stat().st_size,inode=p.stat().st_ino))
   else:protected[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
 write(manifest_path,dict(reason=reason,authorization='Persistent user exception: clearly rejected short trial weights may be retired, keep configs/logs/reports/metadata',weights=records,nonweight_sha256=protected,free_before=free))
 for row in records:Path(row['path']).unlink()
 alias=folder/'ckpt_best.pt'
 if alias.is_symlink():alias.unlink()
 for path,h in protected.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h
 x=read(manifest_path);x['free_after_immediate']=shutil.disk_usage(folder).free;write(manifest_path,x)

def cleanup_rejected_history():
 # Both old200-update head-LR arms failed the declared criterion: both TRAIN
 # improve>1%, neither VAL direction worse>5%. Their reports remain intact.
 for path in [Path('/cloud/zzc/energyWAM/experiments/head_lr_tiny16_20261106/head_lr0001_tiny16_b16_micro4_seed42'),Path('/cloud/zzc/energyWAM/experiments/tiny_future_weight_20261106/tiny16_futureweight9_b16_micro4_seed42')]:
  assert read(path/'final_report.json')['global_step']==200
  retire_weights(path,'Completed old tiny16 head-LR candidate rejected by predeclared validation criterion; no eligible winner. Not the historical400 experiment or current three-task checkpoints.',OUT/(path.name+'_retired_weights.json'))

def run_wave(jobs,wave):
 global CHILDREN
 statuses=[];CHILDREN=[]
 for i,job in enumerate(jobs):
  queue=OUT/f'wave{wave}_jobs_{i}.json';status=OUT/f'wave{wave}_status_{i}.json';write(queue,[job]);statuses.append(status)
  CHILDREN.append(subprocess.Popen([sys.executable,'-u',str(ROOT/'scripts/run_iteration_search.py'),'--jobs',str(queue),'--status',str(status)],cwd=ROOT))
 while any(p.poll() is None for p in CHILDREN) and not STOP:
  summary=summarize(jobs)
  hashes=[a.get('initial_sample_hash') for a in summary['arms']]
  if len(hashes)==2 and None not in hashes and len(set(hashes))!=1:
   raise RuntimeError('Paired initialization hashes differ; refusing unfair LR comparison')
  write(OUT/'controller_status.json',dict(phase='training',wave=wave,arms=summary['arms'],updated_unix=time.time()))
  time.sleep(20)
 if STOP:raise SystemExit('Study controller stopped')
 if any(p.returncode for p in CHILDREN):raise RuntimeError('A queue controller failed')
 CHILDREN=[]
 return summarize(jobs)

def summarize(jobs):
 from scripts.run_iteration_search import collect
 arms=[]
 for job in jobs:
  definition=read(Path(yaml.safe_load((ROOT/job['config']).read_text())['trainer']['output_dir'])/'arm_definition.json');folder=Path(definition['hyperparameters']['trainer']['output_dir']);points=collect(folder)
  speed=read(folder/'speed_report.json') if (folder/'speed_report.json').exists() else {}
  lookup={(p['update'],p['split'],p['direction']):p['mse'] for p in points if p['k']==2}
  step=speed.get('completed_steps',0);final=job['steps'];row=dict(name=job['name'],scale=definition['scale'],folder=str(folder),hyperparameters=definition['hyperparameters']['trainer'],completed_updates=step,equivalent_epochs=step*16/67,points=points)
  if (folder/'initialization_step0.json').exists():row['initial_sample_hash']=read(folder/'initialization_step0.json')['sampled_parameter_sha256']
  if (folder/'final_report.json').exists():row['terminal_success']=read(folder/'final_report.json')['global_step']>=final
  if all((u,'train',d) in lookup for u in [0,final] for d in ['action','future']):
   ratios={d:lookup[final,'train',d]/lookup[0,'train',d] for d in ['action','future']};row['train_ratios']=ratios;row['train_mse']={d:lookup[final,'train',d] for d in ['action','future']};row['fit_score']=sum(ratios.values())/2
   row['both_train_improve_1pct']=all(v<.99 for v in ratios.values())
   row['val_mse']={d:lookup.get((final,'val',d)) for d in ['action','future']}
  if (folder/'timing_history.jsonl').exists():
   times=[json.loads(s) for s in (folder/'timing_history.jsonl').read_text().splitlines()];row['total_update_seconds']=sum(x['end_to_end_seconds'] for x in times);row['median_update_seconds']=__import__('statistics').median(x['end_to_end_seconds'] for x in times)
  arms.append(row)
 result=dict(arms=arms,purpose='Single-episode TRAIN fit; auxiliary VAL/generalization is reported separately; not global optimum',generated_unix=time.time())
 write(OUT/'comparison_report.json',result)
 return result

def choose_next(arms):
 for a in arms:
  if not a.get('terminal_success'):
   retire_weights(a['folder'],'Numerically/resource rejected completed short trial; preserve every config/log/report.',OUT/(a['name']+'_retired_weights.json'))
 valid=[a for a in arms if a.get('terminal_success') and 'fit_score' in a]
 if not valid:return [.1,.03],'Both first-wave arms failed; move lower for stability, without declaring a winner'
 # Avoid allowing a strong direction to hide gross degradation of the other.
 eligible=[a for a in valid if max(a['train_ratios'].values())<=1.05]
 best=min(eligible or valid,key=lambda a:a['fit_score'])
 dominated=[]
 for a in valid:
  if a['name']==best['name']:continue
  if all(best['train_mse'][d]<=.99*a['train_mse'][d] for d in ['action','future']):dominated.append(a)
 for a in arms:
  if a in dominated:
   retire_weights(a['folder'],'Completed short trial clearly rejected: numerical failure or both fixed TRAIN MSEs at least1% worse than same-budget candidate. All metrics and records retained.',OUT/(a['name']+'_retired_weights.json'))
 scales=[.1,.6] if best['scale']==.3 else [.6,2.]
 return scales,f"Best initial TRAIN fit_score: {best['name']}; compare neighboring scales {scales}. Tradeoffs retained in report, not a validated global winner."

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args()
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
 OUT.mkdir(exist_ok=True);count,episodes=prepare_manifest();assert count==67
 first=[prepare_arm(.3,'0,1'),prepare_arm(1.,'2,3')]
 plan=dict(train_task='OpenDrawer',train_episode=3346,train_frames=281,train_windows=67,train_stride=4,auxiliary_val_episodes=[3423,3483],auxiliary_val_windows=16,seed=42,effective_batch=16,native_microbatch=4,K=2,first_wave_scales=[.3,1.],steps_per_arm=210,equivalent_epochs=210*16/67,maximum_arms=4,adaptive_second_wave='If0.3wins:0.1and0.6; if1wins:0.6and2; bothfail:0.1and0.03. Same fresh initialization every arm.',selection='Primary all67 fixed TRAIN Action/Future relative to own untrained baseline at matched210updates and time-to-improvement. No averaging away a worsened direction. Auxiliary16VAL reported separately.',best_checkpoint='Min auxiliary16VAL Action+9Future, complete latest/best; this is distinct from primary overfit ranking.',fixed_eval_every_updates=20,caching='Frozen future and current DINO cached per model; no stochastic augmentation; candidate TRAIN seeds vary by visit, evaluation fixed and final alternate-seed check',retention='Full latest/best per retained arm. Clearly rejected short trials may retire weights with manifest; otherwise wait if space insufficient. Old three-task LR and source256/680 retained.',initialization='Fresh official Qwen3-VL-4B-Instruct plus identical new modules seed42; no robot checkpoint',estimated_total_hours=[8,12],episodes=episodes)
 write(OUT/'plan.json',plan);write(OUT/'arms_wave1.json',first)
 if args.prepare_only:return
 safe_stop_old();cleanup_rejected_history()
 result=run_wave(first,1);scales,reason=choose_next(result['arms']);write(OUT/'adaptive_decision.json',dict(scales=scales,reason=reason,evidence=result))
 second=[prepare_arm(scales[0],'0,1'),prepare_arm(scales[1],'2,3')];write(OUT/'arms_wave2.json',second)
 run_wave(second,2);final=summarize(first+second)
 hashes=[a.get('initial_sample_hash') for a in final['arms']];final['initial_hashes_match']=len(hashes)==4 and None not in hashes and len(set(hashes))==1
 final['outcome']='Four-arm matched-budget review; see separate TRAIN/VAL metrics and stability; no global optimum guarantee'
 write(OUT/'comparison_report.json',final);write(OUT/'controller_status.json',dict(phase='complete',result=str(OUT/'comparison_report.json')))

if __name__=='__main__':
 try:main()
 except Exception as error:
  stop()
  for child in CHILDREN:
   try:child.wait(timeout=100)
   except subprocess.TimeoutExpired:pass
  write(OUT/'controller_status.json',dict(phase='failed',reason=repr(error)))
  raise

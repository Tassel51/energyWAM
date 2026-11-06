"""Prepare a single-variable Future loss-weight comparison on historical16 windows."""
import json,hashlib,copy
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1]
R=Path('/cloud/zzc/energyWAM/experiments/tiny_future_weight_20261106');R.mkdir(exist_ok=True)
old=Path('/cloud/zzc/energyWAM/experiments/iteration_aware_v1_20261104/ablation_prior1_currentinit1_seed42/sample_manifest.json')
m=json.loads(old.read_text());m['note']='Historical16 TRAIN windows from2episodes ofOpenDrawer/OpenCabinet,16VAL windows from4differentepisodes. Weight1vs9 only. Fresh initialization; no TEST.'
tr=[w for w in m['windows'] if w['split']=='train'];va=[w for w in m['windows'] if w['split']=='val'];assert len(tr)==len(va)==16
assert {w['episode_index'] for w in tr}.isdisjoint({w['episode_index'] for w in va})
for name,data in [('train_val_manifest.json',m),('train_monitor_manifest.json',dict(dataset_identity=m['dataset_identity'],windows=tr))]:
 p=R/name
 if p.exists():assert json.loads(p.read_text())==data
 else:p.write_text(json.dumps(data,indent=2))
base=yaml.safe_load((ROOT/'configs/experiments/three_task_b16_micro4_seed42.yaml').read_text());jobs=[];configs=[]
for weight in [1,9]:
 c=copy.deepcopy(base);name=f'tiny16_futureweight{weight}_b16_micro4_seed42';c['run_id']=name;c['notes']='Fresh identical seed42 initialization; fixed historical16TRAIN/16VAL; native4/effective16/K2;50updates=50passes=800samplevisits. Only Future loss weight differs between arms. Compare separate MSE at matched updates, never raw total losses across weights.'
 c['loss']['lambda_future']=float(weight);c['trainer'].update(max_train_steps=50,output_dir=str(R/name),checkpoint_interval_seconds=1800,save_interval=10)
 p=ROOT/f'configs/experiments/{name}.yaml';p.write_text(yaml.safe_dump(c,sort_keys=False));configs.append(c)
 jobs.append(dict(name=name,config=str(p.relative_to(ROOT)),physical_gpus='2,3',steps=50,manifest=str(R/'train_val_manifest.json'),train_monitor_manifest=str(R/'train_monitor_manifest.json'),sample_eval_interval=160,train_monitor_sample_interval=160,keep_best=True,monitor_resources=True,effective_batch=16,guard_min_sample_visits=160,max_fixed_mse_ratio=10,max_process_uss_gib=140,snapshot_bytes_estimate=24568519249))
x,y=copy.deepcopy(configs)
for c in [x,y]:c.pop('run_id');c['trainer'].pop('output_dir');c['loss'].pop('lambda_future')
assert x==y
(R/'jobs.json').write_text(json.dumps(jobs,indent=2))
plan=dict(status='prepared_waiting_for_seed43_safe_pause',seed=42,training_tasks=sorted({w['task'] for w in tr}),train_episodes=2,train_windows=16,val_episodes=4,val_windows=16,effective_batch=16,native_microbatch=4,K=2,updates_each=50,passes_each=50,visits_each=800,weights=[1,9],order=[1,9],warmup_updates=16,only_changed_training_parameter='loss.lambda_future',initialization='Fresh identical pretrained weights and trainable module RNG. No warmstart.',primary_comparison='Unweighted conditional TRAIN and VAL Action/Future MSE at50updates; entire fixed-point trajectories also reported.',best_policy='Each arm retains full latest and own predeclared weighted VAL best. Cross-arm primary comparison uses matched final updates, not differing best scores.',scope='Overfit diagnostic on the historical small pool; not3task full-data validation, not enough to establish final hyperparameters.',expected_total_hours=[2,3],manifest_sha256=hashlib.sha256((R/'train_val_manifest.json').read_bytes()).hexdigest())
(R/'plan.json').write_text(json.dumps(plan,indent=2));print(json.dumps(plan,indent=2))

"""Prepare matched absolute-LR resume experiments; never starts training."""
import copy,hashlib,json,shutil,zipfile
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1]
OUT=Path('/cloud/zzc/energyWAM/experiments/lr256_comparison_20261107')
SOURCE=Path('/cloud/zzc/energyWAM/experiments/three_task_full_20261106/three_task_b16_micro4_seed42')

def main():
 OUT.mkdir(exist_ok=True)
 checkpoint=SOURCE/'ckpt_best.pt'
 assert json.loads(Path(str(checkpoint)+'.meta.json').read_text())['meta']['global_step']==256
 for src,dst in [(SOURCE/'sample_manifest.json',OUT/'sample_manifest.json'),(SOURCE.parent/'train_monitor_manifest.json',OUT/'train_monitor_manifest.json')]:
  shutil.copy2(src,dst)
 base=yaml.safe_load((ROOT/'configs/experiments/three_task_b16_micro4_seed42.yaml').read_text())
 jobs=[]
 for label,scale,pair in [('03',.3,'0,1'),('01',.1,'2,3')]:
  name='resume256_lr'+label;folder=OUT/name
  assert not folder.exists();folder.mkdir()
  cfg=copy.deepcopy(base);cfg['run_id']=name
  cfg['notes']='Resume original best256 model+Adam+RNG; all outer LR groups scaled together. Full original TRAIN pool, original312VAL, fixed12TRAIN. First stage ends512; continuation to768 requires review.'
  cfg['trainer'].update(output_dir=str(folder),max_train_steps=512,checkpoint_interval_seconds=1800,save_interval=64)
  for key in ['learning_rate','task_learning_rate','energy_head_learning_rate','step_size_learning_rate']:cfg['trainer'][key]*=scale
  path=ROOT/f'configs/experiments/{name}.yaml';path.write_text(yaml.safe_dump(cfg,sort_keys=False))
  job=dict(name=name,config=str(path.relative_to(ROOT)),physical_gpus=pair,steps=512,manifest=str(OUT/'sample_manifest.json'),train_monitor_manifest=str(OUT/'train_monitor_manifest.json'),initial_checkpoint=str(checkpoint),resume_config_lrs=True,sample_eval_interval=128*16,train_monitor_sample_interval=64*16,keep_best=True,monitor_resources=True,effective_batch=16,snapshot_bytes_estimate=checkpoint.stat().st_size,max_process_uss_gib=140,max_wall_seconds=24*3600,resume_guard_ratio=10)
  jobs.append(job);(OUT/f'jobs_{label}.json').write_text(json.dumps([job],indent=2))
  (folder/'branch_provenance.json').write_text(json.dumps(dict(source_checkpoint=str(checkpoint),resolved_source=str(checkpoint.resolve()),source_inode=checkpoint.stat().st_ino,source_step=256,lr_scale=scale,preserve_optimizer_moments=True,source_unchanged=True),indent=2))
  with zipfile.ZipFile(folder/'source_branch.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
   for p in (ROOT/'energywam').rglob('*.py'):z.write(p,str(p.relative_to(ROOT)))
   for n in ['train_shared_sharded.py','k_training_probe.py','run_iteration_search.py','prepare_lr256_comparison.py']:z.write(ROOT/'scripts'/n,'scripts/'+n)
   z.write(path,'experiment_config.yaml')
  conf=f'''[program:energywam-lr256-{label}]
command=/usr/local/miniconda3/envs/py312/bin/python -u scripts/run_iteration_search.py --jobs {OUT}/jobs_{label}.json --status {OUT}/status_{label}.json
directory={ROOT}
autostart=false
autorestart=false
startsecs=1
stopasgroup=false
killasgroup=false
stopwaitsecs=100
redirect_stderr=true
stdout_logfile={ROOT}/logs/lr256_{label}.log
environment=OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",OPENBLAS_NUM_THREADS="1",PYTHONPATH="{ROOT}",PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
'''
  Path(f'/etc/supervisor/conf.d/energywam-lr256-{label}.conf').write_text(conf)
 plan=dict(start_step=256,target_step=512,new_updates_per_arm=256,new_sample_visits_per_arm=4096,stage2=False,jobs=jobs,best_criterion='Fixed original312VAL Action MSE + 9*Future MSE; report both separately',selection='Compare each direction against start256 and original512; stability plus improvement, not merely flat loss. Review at512 before extending to768.',retention='Full resumable latest and best; shared global checkpoint lock serializes pending writes',required_peak_additional_gib=(5*checkpoint.stat().st_size)/2**30+8,manifest_sha256=hashlib.sha256((OUT/'sample_manifest.json').read_bytes()).hexdigest())
 (OUT/'plan.json').write_text(json.dumps(plan,indent=2))
 bridge=ROOT/'configs/wandb_bridge.json';b=json.loads(bridge.read_text())
 for j in jobs:b['sources'].append(dict(config=j['config'],group='lr256-comparison-20261107',priority=180))
 temp=bridge.with_suffix('.pending.json');temp.write_text(json.dumps(b,indent=2));temp.chmod(0o600);temp.replace(bridge)
 print(json.dumps({k:v for k,v in plan.items() if k!='jobs'},indent=2))
if __name__=='__main__':main()

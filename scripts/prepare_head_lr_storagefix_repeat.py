"""Prepare a fresh authorized 1e-3 repeat, preserving all training math."""
import copy
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import socket
import subprocess
import yaml
ROOT=Path(__file__).resolve().parents[1]


def main():
 stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
 previous=Path('/cloud/zzc/energyWAM/experiments/single_episode_head_lr_refine_20261109T122513Z')
 oldplan=json.loads((previous/'plan.json').read_text());oldcode=Path(oldplan['code'])
 root=previous.parent/('single_episode_head_lr_storagefix_'+stamp);root.mkdir();(root/'configs').mkdir()
 code=ROOT/'logs/frozen_sources'/('head_repeat_storagefix_'+stamp)/'code'
 shutil.copytree(oldcode,code,symlinks=True,ignore=shutil.ignore_patterns('logs','__pycache__','*.pyc'))
 (code/'logs').symlink_to(ROOT/'logs',target_is_directory=True)
 for name in ['energywam/training/checkpoint.py','energywam/training/run_state.py','energywam/training/checkpoint_metadata.py','scripts/check_checkpoint_metadata_storage.py','scripts/run_head_lr_refinement.py']:
  shutil.copy2(ROOT/name,code/name)
 changes=[];manifest=[]
 for f in code.rglob('*.py'):
  if '__pycache__' in f.parts or 'logs' in f.relative_to(code).parts:continue
  rel=str(f.relative_to(code));original=oldcode/rel
  if not original.exists() or original.read_bytes()!=f.read_bytes():changes.append(rel)
  manifest.append(dict(path=rel,sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
 assert set(changes)=={'energywam/training/checkpoint.py','energywam/training/run_state.py','energywam/training/checkpoint_metadata.py','scripts/check_checkpoint_metadata_storage.py','scripts/run_head_lr_refinement.py'}
 name='episode3346_headscale1_repeat_storagefix_b16_micro4_seed42'
 oldjob=json.loads((previous/'jobs.json').read_text())[0];oldcfg=yaml.safe_load(Path(oldjob['config']).read_text());cfg=copy.deepcopy(oldcfg)
 cfg['run_id']=name;cfg['trainer']['output_dir']=str(root/name)
 config=root/'configs'/(name+'.yaml');config.write_text(yaml.safe_dump(cfg,sort_keys=False))
 for n in ['train_val_manifest.json','train_monitor_manifest.json','environment_manifest.json']:shutil.copy2(previous/n,root/n)
 job=copy.deepcopy(oldjob);job.update(name=name,config=str(config),manifest=str(root/'train_val_manifest.json'),train_monitor_manifest=str(root/'train_monitor_manifest.json'),runtime_environment_manifest=str(root/'environment_manifest.json'),external_checkpoint_reserve_bytes=0)
 (root/'jobs.json').write_text(json.dumps([job],indent=2))
 snapshot=job['snapshot_bytes_estimate'];budget=3*snapshot+8*2**30;assert shutil.disk_usage(root).free>=budget
 plan=copy.deepcopy(oldplan);plan['comparison']='Fresh1e-3only repeat after checkpoint persistence repair; unchanged seed42/210update protocol; compare fixed TRAIN/VAL A/F separately';plan.update(name='single-episode-head-lr-storagefix-'+stamp,root=str(root),code=str(code),jobs=[dict(name=name,folder=str(root/name),config=str(config),head_lr=.001,head_scale=1.)],required_free_bytes=budget,authorization='User2026-11-10 explicitly requested rerunning the failed1e-3 control and authorized retirement of unused checkpoints',target_head_lrs=[.001],total_planned_updates=210,environment_manifest=str(root/'environment_manifest.json'),source_patch='Only checkpoint JSON atomic persistence/fsync/bounded visibility verification; model/solver/loss/training math and all original hyperparameters unchanged',tmux_session='energywam_head_repeat_storagefix_'+stamp,previous_failed_run=str(previous/oldjob['name']),rerun_mode='Fresh official4B/seed42 from0; not resumed from the failed60step checkpoint',storage_gate_result=str(root/'metadata_storage_gate/result.json'))
 (root/'plan.json').write_text(json.dumps(plan,indent=2));(root/'source_manifest.json').write_text(json.dumps(manifest,indent=2));(root/'source_changes.json').write_text(json.dumps(changes,indent=2))
 folder=root/name;folder.mkdir();(folder/'arm_definition.json').write_text(json.dumps(dict(hyperparameters=cfg,job=job,initialization=plan['rerun_mode'],source_changes=changes,previous_failed_run=plan['previous_failed_run'],seed=42,training_windows=67,auxiliary_val_windows=16,target_updates=210,sample_visits=3360,equivalent_epochs=210*16/67,best_criterion='Aux16VAL ownK2 Action MSE +9 Future MSE',cleanup_manifest='/cloud/zzc/energyWAM/experiments/checkpoint_retirement_20261110.json'),indent=2))
 report=dict(plan=plan,free_before_start_bytes=shutil.disk_usage(root).free,changes=changes,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),all_non_storage_training_code_matches_failed_run=True)
 (ROOT/'logs/lab_remote_setup/head_lr_storagefix_deployment.json').write_text(json.dumps(report,indent=2));print(json.dumps(dict(root=str(root),code=str(code),plan=str(root/'plan.json'),python=plan['python'],tmux=plan['tmux_session']),indent=2))

if __name__=='__main__':main()

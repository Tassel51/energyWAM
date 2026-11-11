"""Deploy authorized Future-half retry on GPU0/7 without changing microbatch/math."""
import copy
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.lab_pc_relay import remote_python


def main():
 index=json.loads((ROOT/'logs/lab_remote_setup/ablation_deployment.json').read_text());assert not any(g['name']=='future07' for g in index['groups']),'Future07 already deployed; use status/logs, not duplicate submission'
 x=json.loads((ROOT/'logs/lab_remote_setup/future07_preflight.json').read_text());stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
 base='/data3/xyx_zhuzicheng/projects/energyWAM_remote/releases/energywam_code_20261109T081418Z_4e4994d25f67_release/code'
 root='/data8/xyx_zhuzicheng/energyWAM_studies/future_half_gpu07_'+stamp
 code='/data3/xyx_zhuzicheng/projects/energyWAM_remote/releases/energywam_future07_'+stamp+'/code'
 python='/data7/zzc/energyWAM/lab_execution_20261109/environments/energywam_py310_v2/bin/python'
 cfg=yaml.safe_load(x['base_cfg']);cfg['run_id']='episode3346_future_half_gpu07_native4_seed42';cfg['trainer']['output_dir']=root+'/'+cfg['run_id'];cfg['trainer']['saved_tensor_gpu_budgets_gib']=[0.,0.];cfg['trainer']['language_model_layer_counts']=[19,17]
 assert cfg['trainer']['shared_microbatch_size']==4 and cfg['trainer']['per_device_batch_size']==16
 assert cfg['solver']['alpha_future']==1000. and cfg['solver']['train_inner_steps']==2
 job=copy.deepcopy(x['base_job']);job.update(name=cfg['run_id'],config=root+'/configs/future_half.yaml',physical_gpus='7,0',execution_gpu_policy=root+'/gpu_policy.json',manifest=root+'/manifests/train_val_manifest.json',train_monitor_manifest=root+'/manifests/train_monitor_manifest.json')
 profile=copy.deepcopy(job);profile.update(name='capacity_future_half_gpu07',config=root+'/configs/capacity.yaml',steps=2,profile=True,keep_best=False)
 for k in ['sample_eval_interval','train_monitor_sample_interval','train_monitor_manifest','alternate_final_eval_seed']:profile.pop(k,None)
 capacity=copy.deepcopy(cfg);capacity['run_id']=profile['name'];capacity['trainer']['output_dir']=root+'/capacity_profile'
 policy=dict(hostname='user-SYS-420GP-TNR',allowed_physical_pairs=['7,0'],use_uuid_visibility=True,expected_gpu_uuids={'0':'GPU-199fb0a3-1013-d2e1-fece-6e075f7477a6','7':'GPU-4de65a87-f820-da66-782e-0aab5d22e243'},authorized_sharing=dict(indices=['7'],min_free_mib={'0':31000,'7':41000}),reason='User2026-11-10 requested physical GPU0/7 with supplied occupancy including other GPU7 contexts; no other process is signalled')
 plan=dict(name='future07',hostname='user-SYS-420GP-TNR',root=root,code=code,python=python,gpu_pair='7,0',profile_jobs=root+'/profile_jobs.json',full_jobs=root+'/full_jobs.json',gradient_gate_result=root+'/native_gradient_gate/result.json',whole_study_checkpoint_budget_bytes=3*job['snapshot_bytes_estimate']+8*2**30,arms=[dict(name=cfg['run_id'],folder=cfg['trainer']['output_dir'],factors=dict(initial_alpha_action=5.,initial_alpha_future=1000.,lambda_future=9.,train_k=2,effective_prior_future_precision=.0005))],purpose='Finish Future-half study on user-selected GPU0/7; original native4 training math; only memory placement:GPU7 primary19layers/GPU0 17layers,CPU saved activations; original training math',protocol='Freshofficial4B/seed42,67TRAIN/16VAL,B16/native4/FP32/K2,210updates,step0 baseline,20update fixed evaluations; original optimizer/LRs/outerclip and10x guard retained',comparison_caveat='Hardware migration from L40S6/7 to RTX0/L40S7, shared GPU7; not a strict same-hardware hyperparameter-only comparison',retention='Full latest/best, criterion auxiliary16VAL Action+9Future; preserve failed attempt and no checkpoint/model/dataset deletion',tmux_session='energywam_future07_'+stamp,gate_timeout_seconds=3600,source_base=base)
 plan['gradient_gate_command']=[python,'-u',code+'/scripts/check_native_microbatch_gradients.py','--config',root+'/configs/capacity.yaml','--manifest',job['manifest'],'--output',plan['gradient_gate_result'],'--physical-gpus','7,0','--execution-gpu-policy',root+'/gpu_policy.json','--precision','float32','--native-microbatch','4','--sample-indices','0,16,33,66','--math-mode','strict','--reference-gpu-budgets','0','0','--dedup-cpu','--record-all']
 payload=dict(parallel=(ROOT/'energywam/training/parallel.py').read_text(),plan=plan,job=job,profile=profile,policy=policy,config=yaml.safe_dump(cfg,sort_keys=False),capacity=yaml.safe_dump(capacity,sort_keys=False),stamp=stamp,dashboard=(ROOT/'scripts/monitor_lab_studies.py').read_text())
 result=remote_python('lab-via-pc','PAYLOAD='+repr(payload)+'\n'+REMOTE,retry_safe=False)
 (ROOT/'logs/lab_remote_setup/future07_deployment.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
 path=ROOT/'logs/lab_remote_setup/ablation_deployment.json';index=json.loads(path.read_text());assert not any(g['name']=='future07' for g in index['groups']);index['groups'].append(plan);path.write_text(json.dumps(index,ensure_ascii=False,indent=2))
 print(json.dumps(dict(root=root,code=code,controller=result['controller'],tmux=plan['tmux_session']),ensure_ascii=False,indent=2))


REMOTE=r'''
from pathlib import Path
import json,shutil,socket,hashlib,subprocess,shlex,time,os
assert socket.gethostname()=='user-SYS-420GP-TNR'
p=PAYLOAD['plan'];root=Path(p['root']);code=Path(p['code']);base=Path(p['source_base'])
assert not root.exists() and not code.exists()
assert shutil.disk_usage(root.parent).free>p['whole_study_checkpoint_budget_bytes']
root.mkdir();(root/'configs').mkdir();(root/'manifests').mkdir()
code.parent.mkdir(parents=True);shutil.copytree(base,code,symlinks=True,ignore=shutil.ignore_patterns('logs','__pycache__','*.pyc'))
(code/'logs').symlink_to((base/'logs').resolve(),target_is_directory=True)
# Only the queue needs sharing-aware availability checks, not any model/trainer patch.
queue=Path('/data3/xyx_zhuzicheng/projects/energyWAM_remote/releases/energywam_code_20261109T095430Z_b774d22735dc_release/code/scripts/run_iteration_search.py');shutil.copy2(queue,code/'scripts/run_iteration_search.py')
f=code/'energywam/config.py';text=f.read_text();needle='    language_model_devices: List[int] = field(default_factory=list)';assert needle in text;f.write_text(text.replace(needle,needle+'\n    language_model_layer_counts: List[int] = field(default_factory=list)'))
f=code/'energywam/training/trainer.py';text=f.read_text();needle='shard_language_model(model, cfg.trainer.language_model_devices, self.device)';assert needle in text;f.write_text(text.replace(needle,'shard_language_model(model, cfg.trainer.language_model_devices, self.device, layer_counts=cfg.trainer.language_model_layer_counts)'))
(code/'energywam/training/parallel.py').write_text(PAYLOAD['parallel'])
manifest=[]
for f in code.rglob('*.py'):
 if 'logs' in f.relative_to(code).parts or '__pycache__' in f.parts:continue
 rel=str(f.relative_to(code));manifest.append(dict(path=rel,sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
 if rel.startswith('energywam/') and rel not in ['energywam/config.py','energywam/training/trainer.py','energywam/training/parallel.py']:assert f.read_bytes()==(base/rel).read_bytes(),rel
for file,obj in [('plan.json',p),('full_jobs.json',[PAYLOAD['job']]),('profile_jobs.json',[PAYLOAD['profile']]),('gpu_policy.json',PAYLOAD['policy']),('source_manifest.json',manifest)]:
 (root/file).write_text(json.dumps(obj,indent=2))
(root/'configs/future_half.yaml').write_text(PAYLOAD['config']);(root/'configs/capacity.yaml').write_text(PAYLOAD['capacity'])
oldroot=Path('/data6/zzc/energywam_lab_studies/inner_steps_20261109');oldjobs=json.loads((oldroot/'full_jobs.json').read_text())[-1]
for key,name in [('manifest','train_val_manifest.json'),('train_monitor_manifest','train_monitor_manifest.json')]:shutil.copy2(oldjobs[key],root/'manifests'/name)
folder=Path(p['arms'][0]['folder']);folder.mkdir();(folder/'arm_definition.json').write_text(json.dumps(dict(job=PAYLOAD['job'],config_yaml=PAYLOAD['config'],purpose=p['purpose'],initialization='Fresh official4B/seed42, same initialization as failed0-update Futurehalf; not a continuation of another hyperparameter arm',previous_failed_run=str(oldroot/'episode3346_future_half_seed42'),training_changes='Training objective unchanged; physical GPU order,19/17 language placement,CPU activation storage',source_manifest=str(root/'source_manifest.json'),target_updates=210,best_criterion='Aux16VAL Action MSE+9Future MSE',hardware_caveat=p['comparison_caveat']),indent=2))
(root/'full_status.json').write_text(json.dumps({p['arms'][0]['name']:dict(phase='queued',completed_updates=0,target_updates=210,physical_gpus='7,0')},indent=2))
# Package/env and the resolved batch/solver config are checked without allocating CUDA.
check="""import sys,json,importlib.metadata
from energywam.config import load_config
c=load_config(sys.argv[1]);assert c.trainer.shared_microbatch_size==4 and c.trainer.per_device_batch_size==16
assert c.solver.alpha_future==1000. and c.solver.train_inner_steps==2
assert c.trainer.saved_tensor_gpu_budgets_gib==[0.,0.] and c.trainer.language_model_layer_counts==[19,17]
print(json.dumps(dict(python=sys.version,packages={n:importlib.metadata.version(n) for n in ['torch','transformers','bitsandbytes']},native_microbatch=4,effective_batch=16)))"""
r=subprocess.run([p['python'],'-c',check,str(root/'configs/future_half.yaml')],cwd=code,capture_output=True,text=True,timeout=30);assert r.returncode==0,r.stderr
(root/'runtime_configuration_check.json').write_text(r.stdout)
command=shlex.join([p['python'],'-u',str(code/'scripts/run_lab_ablation.py'),'--plan',str(root/'plan.json')])+' >> '+shlex.quote(str(root/'launcher.log'))+' 2>&1'
subprocess.run(['tmux','new-session','-d','-s',p['tmux_session'],'-c',str(code),command],check=True)
for _ in range(30):
 if (root/'controller_status.json').exists():break
 time.sleep(.2)
state=json.loads((root/'controller_status.json').read_text());assert state['phase']=='gradient_gate',state
# Add the new retry to the server-local dashboard; preserve existing study entries.
tools=Path('/data3/xyx_zhuzicheng/projects/energyWAM_remote/tools');index=tools/'study_dashboard.json';data=json.loads(index.read_text());data['groups'].append(dict(name='future07',root=str(root)))
temp=index.with_suffix('.pending.json');temp.write_text(json.dumps(data,indent=2));temp.replace(index);(tools/'monitor_studies.py').write_text(PAYLOAD['dashboard'])
print(json.dumps(dict(plan=p,controller=state,environment=json.loads(r.stdout),all_model_solver_loss_sources_match_original_GPU67=True,free_bytes=shutil.disk_usage(root).free)))
'''

if __name__=='__main__':main()

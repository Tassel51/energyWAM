"""User-selected original GPU67 Futurehalf retry, queued until both GPUs idle."""
import copy
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.lab_pc_relay import remote_python


def main():
 index_path=ROOT/'logs/lab_remote_setup/ablation_deployment.json';index=json.loads(index_path.read_text());assert not any(g['name']=='future67' for g in index['groups']),'Already deployed; use status'
 x=json.loads((ROOT/'logs/lab_remote_setup/future67_retry_preflight.json').read_text());old=x['plan'];stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
 root='/data8/xyx_zhuzicheng/energyWAM_studies/future_half_original_gpu67_'+stamp;code='/data3/xyx_zhuzicheng/projects/energyWAM_remote/releases/energywam_future67_original_'+stamp+'/code'
 cfg=yaml.safe_load(x['config']);name='episode3346_future_half_gpu67_original_seed42';cfg['run_id']=name;cfg['trainer']['output_dir']=root+'/'+name
 assert cfg['trainer']['shared_microbatch_size']==4 and cfg['trainer']['saved_tensor_gpu_budgets_gib']==[10.,14.]
 assert not cfg['trainer'].get('language_model_layer_counts') and not cfg['trainer'].get('optimizer_state_cpu_offload')
 job=copy.deepcopy(x['job']);job.update(name=name,config=root+'/configs/future_half.yaml',physical_gpus='6,7',execution_gpu_policy=root+'/gpu_policy.json',manifest=root+'/manifests/train_val_manifest.json',train_monitor_manifest=root+'/manifests/train_monitor_manifest.json')
 cap=copy.deepcopy(cfg);cap['run_id']='capacity_future67_original';cap['trainer']['output_dir']=root+'/capacity_profile'
 profile=copy.deepcopy(job);profile.update(name=cap['run_id'],config=root+'/configs/capacity.yaml',steps=2,profile=True,keep_best=False)
 for k in ['sample_eval_interval','train_monitor_sample_interval','train_monitor_manifest','alternate_final_eval_seed']:profile.pop(k,None)
 plan=dict(name='future67',hostname=old['hostname'],root=root,code=code,python=old['python'],gpu_pair='6,7',profile_jobs=root+'/profile_jobs.json',full_jobs=root+'/full_jobs.json',gradient_gate_result=old['gradient_gate_result'],whole_study_checkpoint_budget_bytes=3*job['snapshot_bytes_estimate']+8*2**30,arms=[dict(name=name,folder=cfg['trainer']['output_dir'],factors=dict(initial_alpha_action=5.,initial_alpha_future=1000.,lambda_future=9.,train_k=2))],purpose='User explicitly chose waiting for GPU6/7 idle and original configuration; replace failed Futurehalf, no small-GPU adaptations',protocol='Exact originalGPU67 Futurehalf configuration except run_id/output, fresh official4B seed42,210updates,B16/native4/FP32/K2,original18/18 andactivation10/14,GPUoptimizerstates,67TRAIN16VAL,fixed20updates,original10xguard',retention='Full latest/best selected by aux16VAL A+9F; failed trials retained',tmux_session='energywam_future67_original_'+stamp,source_base=old['code'],gradient_gate_scope='Reuse existing passed originalGPU67 native4/storage gate; exact original source/config plus fresh capacity check after GPUs idle')
 future07=next(g for g in index['groups'] if g['name']=='future07');kplan=next(g for g in index['groups'] if g['name']=='k')
 payload=dict(plan=plan,old=old,job=job,profile=profile,config=yaml.safe_dump(cfg,sort_keys=False),capacity=yaml.safe_dump(cap,sort_keys=False),policy=x['policy'],future07=future07,kplan=kplan,stamp=stamp,dashboard=(ROOT/'scripts/monitor_lab_studies.py').read_text())
 result=remote_python('lab-via-pc','PAYLOAD='+repr(payload)+'\n'+REMOTE,retry_safe=False)
 (ROOT/'logs/lab_remote_setup/future67_original_retry_deployment.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
 index.setdefault('archived_groups',[]).append(future07);index['groups']=[result['k_plan'] if g['name']=='k' else g for g in index['groups'] if g['name']!='future07']+[plan];index_path.write_text(json.dumps(index,ensure_ascii=False,indent=2))
 prop=ROOT/'logs/lab_remote_setup/future07_resource_choice_proposal.json';d=json.loads(prop.read_text());d.update(status='CPUoffload_rejected_by_user_wait_GPU67_selected',selected='WaitGPU6/7 originalconfiguration',replacement_root=root);prop.write_text(json.dumps(d,ensure_ascii=False,indent=2))
 print(json.dumps(dict(root=root,tmux=plan['tmux_session'],controller=result['controller'],queue=result['queue'],k=result['k_controller'],future07_cancelled=True),ensure_ascii=False,indent=2))


REMOTE=r'''
from pathlib import Path
import json,shutil,socket,hashlib,time,subprocess,shlex
assert socket.gethostname()=='user-SYS-420GP-TNR'
p=PAYLOAD['plan'];root=Path(p['root']);code=Path(p['code']);base=Path(p['source_base'])
assert not root.exists() and not code.exists()
assert shutil.disk_usage(root.parent).free>p['whole_study_checkpoint_budget_bytes']
# Retire only the stopped GPU07 controller's pending work; keep all records.
f07=Path(PAYLOAD['future07']['root']);prior=json.loads((f07/'controller_status.json').read_text());assert prior['phase'] in ['blocked_capacity','failed']
for arm in PAYLOAD['future07']['arms']:assert not (Path(arm['folder'])/'train_log.jsonl').exists()
for pid in [prior.get('pid'),prior.get('child_pid')]:
 if pid:
  proc=Path('/proc',str(pid),'cmdline')
  assert not (proc.exists() and str(f07).encode() in proc.read_bytes()),'Old controller still active'
history=f07/'user_decision_wait_GPU67';history.mkdir()
for name in ['controller_status.json','full_jobs.json','full_status.json']:
 f=f07/name
 if f.exists():shutil.copy2(f,history/name)
(f07/'full_jobs.json').write_text('[]\n');prior.update(phase='cancelled',reason='User selected waiting for GPU6/7 original configuration',replacement_root=str(root),ended_at=time.time());(f07/'controller_status.json').write_text(json.dumps(prior,indent=2));(f07/'full_status.json').write_text(json.dumps({a['name']:dict(phase='cancelled',completed_updates=0,target_updates=210,reason=prior['reason']) for a in PAYLOAD['future07']['arms']},indent=2))
root.mkdir();(root/'configs').mkdir();(root/'manifests').mkdir();code.parent.mkdir(parents=True)
shutil.copytree(base,code,symlinks=True,ignore=shutil.ignore_patterns('logs','__pycache__','*.pyc'));(code/'logs').symlink_to((base/'logs').resolve(),target_is_directory=True)
manifest=[]
for f in code.rglob('*.py'):
 if 'logs' in f.relative_to(code).parts or '__pycache__' in f.parts:continue
 rel=str(f.relative_to(code));assert f.read_bytes()==(base/rel).read_bytes();manifest.append(dict(path=rel,sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
for file,obj in [('plan.json',p),('full_jobs.json',[PAYLOAD['job']]),('profile_jobs.json',[PAYLOAD['profile']]),('gpu_policy.json',PAYLOAD['policy']),('source_manifest.json',manifest)]:
 (root/file).write_text(json.dumps(obj,indent=2))
assert not PAYLOAD['policy'].get('authorized_sharing')
(root/'configs/future_half.yaml').write_text(PAYLOAD['config']);(root/'configs/capacity.yaml').write_text(PAYLOAD['capacity'])
basejob=json.loads(Path(PAYLOAD['old']['full_jobs']).read_text())[-1]
for key,name in [('manifest','train_val_manifest.json'),('train_monitor_manifest','train_monitor_manifest.json')]:shutil.copy2(basejob[key],root/'manifests'/name)
folder=Path(p['arms'][0]['folder']);folder.mkdir();(folder/'arm_definition.json').write_text(json.dumps(dict(job=PAYLOAD['job'],resolved_config_yaml=PAYLOAD['config'],initialization='Fresh official4B seed42 at0, not a resume from a different memory adaptation',purpose=p['purpose'],previous_failed_run='/data6/zzc/energywam_lab_studies/inner_steps_20261109/episode3346_future_half_seed42',source_manifest=str(root/'source_manifest.json'),source_and_nonidentity_configuration='byte-identical code/originalcfg except run_id/output; no custom layer counts or CPU optimizer offload',best_criterion='Aux16VAL Action MSE+9Future MSE'),indent=2))
(root/'full_status.json').write_text(json.dumps({p['arms'][0]['name']:dict(phase='queued',completed_updates=0,target_updates=210,physical_gpus='6,7')},indent=2))
check="""import sys,json,copy,yaml
from pathlib import Path
from energywam.config import load_config
r=Path(sys.argv[1]);old=yaml.safe_load(Path('/data6/zzc/energywam_lab_studies/inner_steps_20261109/configs/episode3346_future_half_seed42.yaml').read_text());new=yaml.safe_load((r/'configs/future_half.yaml').read_text())
for c in [old,new]:c.pop('run_id',None);c['trainer'].pop('output_dir',None)
assert old==new
c=load_config(r/'configs/future_half.yaml');assert c.trainer.shared_microbatch_size==4 and c.trainer.saved_tensor_gpu_budgets_gib==[10.,14.] and c.solver.alpha_future==1000.
print(json.dumps(dict(passed=True,all_original_hyperparameters_unchanged=True,original_18x18_default=True,optimizer_state_storage='GPU',native_microbatch=4)))"""
r=subprocess.run([p['python'],'-c',check,str(root)],cwd=code,capture_output=True,text=True,timeout=30);assert r.returncode==0,r.stderr
(root/'configuration_verification.json').write_text(r.stdout)
command=shlex.join([p['python'],'-u',p['code']+'/scripts/run_lab_ablation.py','--plan',str(root/'plan.json')])+' >> '+shlex.quote(str(root/'launcher.log'))+' 2>&1'
subprocess.run(['tmux','new-session','-d','-s',p['tmux_session'],'-c',p['code'],command],check=True)
for _ in range(35):
 if (root/'controller_status.json').exists() and (root/'profile_status.json').exists():break
 time.sleep(.2)
state=json.loads((root/'controller_status.json').read_text());queue=json.loads((root/'profile_status.json').read_text())
# Restore the original K67 continuation, dependent only on this replacement arm.
kr=Path(PAYLOAD['kplan']['root']);ks=json.loads((kr/'controller_status.json').read_text());assert ks['phase']=='blocked_dependency'
assert not any(Path(a['folder'],'train_log.jsonl').exists() for a in PAYLOAD['kplan']['arms'])
kh=kr/'dependency_history'/('wait_GPU67_'+PAYLOAD['stamp']);kh.mkdir(parents=True)
for n in ['plan.json','controller_status.json','after_study_status.json']:
 f=kr/n
 if f.exists():shutil.copy2(f,kh/n)
kp=json.loads((kr/'plan.json').read_text());kp['after_study']=dict(root=str(root),arm=p['arms'][0]['name']);kp['after_study_change_reason']='User selected waitingGPU67 originalFuturehalf instead of GPU07 adaptation';(kr/'plan.json').write_text(json.dumps(kp,indent=2))
assert subprocess.run(['tmux','has-session','-t',kp['tmux_session']],capture_output=True).returncode!=0
kc=shlex.join([kp['python'],'-u',kp['code']+'/scripts/run_lab_after_study.py','--plan',str(kr/'plan.json')])+' >> '+shlex.quote(str(kr/'launcher.log'))+' 2>&1'
subprocess.run(['tmux','new-session','-d','-s',kp['tmux_session'],'-c',kp['code'],kc],check=True)
for _ in range(35):
 ks=json.loads((kr/'controller_status.json').read_text())
 if ks['phase']=='waiting_for_preceding_study':break
 time.sleep(.2)
assert ks['phase']=='waiting_for_preceding_study'
# Update dashboard only; no model or training process on GPU07 is started.
tools=Path('/data3/xyx_zhuzicheng/projects/energyWAM_remote/tools');f=tools/'study_dashboard.json';d=json.loads(f.read_text());d['groups']=[g for g in d['groups'] if g['name']!='future07']+[dict(name='future67',root=str(root))];temp=f.with_suffix('.pending.json');temp.write_text(json.dumps(d,indent=2));temp.replace(f);(tools/'monitor_studies.py').write_text(PAYLOAD['dashboard'])
print(json.dumps(dict(plan=p,controller=state,queue=queue,k_plan=kp,k_controller=ks,verification=json.loads(r.stdout),all_source_files_match_original_GPU67=True,old07_cancelled=True)))
'''

if __name__=='__main__':main()

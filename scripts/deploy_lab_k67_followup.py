#!/usr/bin/env python3
"""Deploy the user-authorized K234 native4 study after the existing GPU67 queue."""
import copy
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.lab_pc_relay import remote_python


def main():
    pre=json.loads((ROOT/'logs/lab_remote_setup/k67_preflight.json').read_text())
    groups={g['name']:g for g in pre['groups']};base=groups['steps'];old=groups['k']
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    root='/data8/xyx_zhuzicheng/energyWAM_studies/k234_gpu67_'+stamp
    release='/data3/xyx_zhuzicheng/projects/energyWAM_remote/releases/energywam_k67_native4_'+stamp
    code=release+'/code';configs={};jobs=[];arms=[]
    base_cfg=yaml.safe_load(base['base_config_yaml'])
    assert base_cfg['trainer']['shared_microbatch_size']==4 and base_cfg['trainer']['per_device_batch_size']==16
    assert not base_cfg['trainer'].get('optimizer_state_cpu_offload',False)
    for k in [2,3,4]:
        name=f'episode3346_k{k}_gpu67_native4_seed42';cfg=copy.deepcopy(base_cfg)
        cfg['run_id']=name;cfg['trainer']['output_dir']=root+'/'+name
        cfg['solver']['train_inner_steps']=cfg['solver']['eval_inner_steps']=k
        cfg['notes']=f'User-authorized K={k} repeat on GPU6/7 after Future-half; original GPU67 native4 configuration. Only K varies; fresh official4B seed42.'
        configs[name]=yaml.safe_dump(cfg,sort_keys=False)
        job=copy.deepcopy(base['jobs'][0]);job.update(name=name,config=root+'/configs/'+name+'.yaml',physical_gpus='6,7',execution_gpu_policy=root+'/gpu_policy.json',manifest=root+'/manifests/train_val_manifest.json',train_monitor_manifest=root+'/manifests/train_monitor_manifest.json',evaluation_k=k,probe_ks=sorted({2,k}))
        jobs.append(job);arms.append(dict(name=name,folder=root+'/'+name,factors=dict(train_k=k,own_evaluation_k=k,common_evaluation_k=2,lambda_future=9.0)))
    cap=copy.deepcopy(yaml.safe_load(configs[arms[-1]['name']]))
    cap['run_id']='capacity_k4_gpu67_native4';cap['trainer']['output_dir']=root+'/capacity_profile'
    configs['capacity']=yaml.safe_dump(cap,sort_keys=False)
    profile=copy.deepcopy(jobs[-1]);profile.update(name='capacity_k4_gpu67_native4',config=root+'/configs/capacity.yaml',steps=2,profile=True,keep_best=False)
    for k in ['sample_eval_interval','train_monitor_sample_interval','train_monitor_manifest','alternate_final_eval_seed']:profile.pop(k,None)
    plan=dict(hostname='user-SYS-420GP-TNR',name='k',root=root,code=code,python=base['plan']['python'],gpu_pair='6,7',profile_jobs=root+'/profile_jobs.json',full_jobs=root+'/full_jobs.json',gradient_gate_result=base['plan']['gradient_gate_result'],whole_study_checkpoint_budget_bytes=base['plan']['whole_study_checkpoint_budget_bytes'],arms=arms,purpose='K2/3/4 matched GPU67/native4 control; no small-GPU memory adaptations',protocol='Fresh official4B/seed42,67TRAIN/16VAL,B16/native4,FP32,210updates each; only train/eval K varies. Fixed evaluation every20updates; ownK and commonK2 separately; best ownK VAL A+9F. Existing10x guard unchanged.',baseline='Exact GPU67 original-step control configuration and all energywam Python sources; no Action/Future half step carries over.',retention='Full verified latest/best; no automatic deletions',execution_caveat='Native4, original default18/18 language layers, GPU optimizer states, saved GPU activations10/14GiB, original CPU activation offload. K4 profile after dependency; block rather than modify if capacity fails.',after_study=dict(root=base['plan']['root'],arm='episode3346_future_half_seed42'),tmux_session='energywam_lab_k67_after_steps_'+stamp,gate_timeout_seconds=7200,gradient_gate_scope='Reuse passed original GPU67 native4 storage/gradient gate; all model/trainer/solver sources byte-identical. New K4 two-update capacity profile is engineering only, not formal progress.',source_base=base['plan']['code'])
    payload=dict(plan=plan,configs=configs,jobs=jobs,profile=[profile],base_plan=base['plan'],old_plan=old['plan'],policy=base['policy'],wrapper=(ROOT/'scripts/run_lab_after_study.py').read_text(),dashboard=(ROOT/'scripts/monitor_lab_studies.py').read_text(),stamp=stamp)
    remote='PAYLOAD='+repr(payload)+'\n'+REMOTE
    result=remote_python('lab-via-pc',remote,retry_safe=False)
    path=ROOT/'logs/lab_remote_setup/k67_followup_deployment.json';path.write_text(json.dumps(result,ensure_ascii=False,indent=2))
    index=ROOT/'logs/lab_remote_setup/ablation_deployment.json';d=json.loads(index.read_text())
    d.setdefault('archived_groups',[]).extend(g for g in d['groups'] if g['name']=='k')
    d['groups']=[g for g in d['groups'] if g['name']!='k']+[plan]
    index.write_text(json.dumps(d,ensure_ascii=False,indent=2))
    print(json.dumps(dict(root=root,code=code,tmux=plan['tmux_session'],status=result['status'],old_queue=result['old_queue'],source_changes=result['source_changes']),ensure_ascii=False,indent=2))


REMOTE=r'''
from pathlib import Path
import copy,difflib,fcntl,hashlib,json,os,shlex,shutil,socket,subprocess,time
assert socket.gethostname()=='user-SYS-420GP-TNR'
p=PAYLOAD['plan'];root=Path(p['root']);code=Path(p['code']);base=Path(PAYLOAD['base_plan']['code']);old=PAYLOAD['old_plan'];oldroot=Path(old['root'])
assert not root.exists() and not code.exists(),'Unique deployment paths already exist; do not repeat'
assert shutil.disk_usage(root.parent).free>p['whole_study_checkpoint_budget_bytes']

def atomic(path,obj):
 path=Path(path);temp=path.with_suffix('.pending.json');temp.write_text(json.dumps(obj,indent=2));temp.replace(path)

# Withdraw only unstarted old arms while holding that queue's own GPU-pair lock.
lock_path=Path(old['code'])/'logs/gpu_pair_1_8.lock';lock_path.parent.mkdir(exist_ok=True)
lock=lock_path.open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
status=json.loads((oldroot/'full_status.json').read_text())
assert not any(v.get('phase')=='running' for v in status.values()),'Old queue has an active trial; refusing to interrupt it'
for a in old['arms']:
 if a['factors']['train_k'] in [3,4]:
  assert not (Path(a['folder'])/'train_log.jsonl').exists(),'Old K3/K4 already started'
original_jobs=Path(old['full_jobs']).read_text()
archive=oldroot/('superseded_gpu67_'+PAYLOAD['stamp']);archive.mkdir()
(archive/'full_jobs.json').write_text(original_jobs);(archive/'full_status.json').write_text(json.dumps(status,indent=2))
atomic(Path(old['full_jobs']),[])
receipt=dict(authorized='User requested K2/3/4 after GPU67 Futurehalf using original GPU67 config',at=time.time(),new_root=str(root),previous_jobs_sha256=hashlib.sha256(original_jobs.encode()).hexdigest(),process_signal_sent=False,previous_failed_K2_preserved=True)
atomic(archive/'withdrawal.json',receipt)
# Existing queue observes its now-empty pending list and exits naturally.
fcntl.flock(lock,fcntl.LOCK_UN);lock.close()

code.parent.mkdir(parents=True);shutil.copytree(base,code,symlinks=True,ignore=shutil.ignore_patterns('logs','__pycache__','*.pyc'))
assert not (code/'.git').exists()
(code/'logs').symlink_to((base/'logs').resolve(),target_is_directory=True)
# Keep every model/trainer/solver byte-identical. Add only K-aware evaluation plumbing.
probe_source=Path(old['code'])/'scripts/k_training_probe.py'
shutil.copy2(probe_source,code/'scripts/k_training_probe.py')
q=code/'scripts/run_iteration_search.py';s=q.read_text()
changes=[("                if row['k'] not in [0,2]:continue\n",''),
 ('min_sample_visits=None):','min_sample_visits=None, evaluation_k=2):'),
 ("p['direction']==direction and p['k']==2]","p['direction']==direction and p['k']==evaluation_k]"),
 ("min_sample_visits=job.get('guard_min_sample_visits'))","min_sample_visits=job.get('guard_min_sample_visits'),evaluation_k=job.get('evaluation_k',2))"),
 ("        if job.get('execution_gpu_policy'):\n            from energywam.training.gpu_policy import authorized_visibility", "        if job.get('probe_ks'):\n            env['ENERGYWAM_PROBE_KS']=','.join(map(str,job['probe_ks']))\n            env['ENERGYWAM_PROBE_MAX_K']=str(max(job['probe_ks']))\n        if job.get('execution_gpu_policy'):\n            from energywam.training.gpu_policy import authorized_visibility")]
for oldtext,newtext in changes:
 assert oldtext in s,oldtext
 s=s.replace(oldtext,newtext)
q.write_text(s);(code/'scripts/run_lab_after_study.py').write_text(PAYLOAD['wrapper'])
source_changes=[];manifest=[]
for f in code.rglob('*.py'):
 if '__pycache__' in f.parts or 'logs' in f.relative_to(code).parts:continue
 rel=str(f.relative_to(code));sha=hashlib.sha256(f.read_bytes()).hexdigest();manifest.append(dict(path=rel,sha256=sha))
 original=base/rel
 if original.is_file() and original.read_bytes()!=f.read_bytes():source_changes.append(rel)
 if rel.startswith('energywam/'):
  assert original.read_bytes()==f.read_bytes(),rel
assert set(source_changes)=={'scripts/run_iteration_search.py','scripts/k_training_probe.py'}
root.mkdir();(root/'configs').mkdir();(root/'manifests').mkdir()
for name,content in PAYLOAD['configs'].items():(root/'configs'/(name+'.yaml')).write_text(content)
basejob=json.loads(Path(PAYLOAD['base_plan']['full_jobs']).read_text())[0]
for field,name in [('manifest','train_val_manifest.json'),('train_monitor_manifest','train_monitor_manifest.json')]:shutil.copy2(basejob[field],root/'manifests'/name)
atomic(root/'gpu_policy.json',PAYLOAD['policy']);atomic(root/'full_jobs.json',PAYLOAD['jobs']);atomic(root/'profile_jobs.json',PAYLOAD['profile']);atomic(root/'plan.json',p)
atomic(root/'full_status.json',{a['name']:dict(phase='queued',completed_updates=0,target_updates=210,physical_gpus='6,7') for a in p['arms']})
atomic(root/'source_manifest.json',manifest)
for name in source_changes:
 (root/(Path(name).name+'.diff')).write_text(''.join(difflib.unified_diff((base/name).read_text().splitlines(True),(code/name).read_text().splitlines(True),fromfile='GPU67_original/'+name,tofile='GPU67_K234/'+name)))
for a,job in zip(p['arms'],PAYLOAD['jobs']):
 folder=Path(a['folder']);folder.mkdir();atomic(folder/'arm_definition.json',dict(job=job,factors=a['factors'],initialization='fresh official4B/seed42, not a resume of failed K2',source_base=str(base),source_manifest=str(root/'source_manifest.json'),purpose=p['purpose'],planned_updates=210,planned_sample_visits=3360,planned_equivalent_epochs=210*16/67,best_criterion='OwnK auxiliary16VAL Action MSE +9 Future MSE',data_manifests={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in (root/'manifests').glob('*.json')}))
proof_code="""import importlib.metadata,json,sys
from energywam.config import load_config
from pathlib import Path
root=Path(sys.argv[1]);rows=[]
for f in sorted((root/'configs').glob('episode*.yaml')):
 c=load_config(f);assert c.trainer.shared_microbatch_size==4 and c.trainer.per_device_batch_size==16
 assert c.trainer.saved_tensor_gpu_budgets_gib==[10.,14.]
 assert c.solver.train_inner_steps==c.solver.eval_inner_steps in [2,3,4]
 rows.append(dict(config=str(f),k=c.solver.train_inner_steps,native_microbatch=c.trainer.shared_microbatch_size))
print(json.dumps(dict(python=sys.version,packages={n:importlib.metadata.version(n) for n in ['torch','transformers','bitsandbytes','numpy']},resolved=rows)))"""
r=subprocess.run([p['python'],'-c',proof_code,str(root)],cwd=code,capture_output=True,text=True,timeout=90)
assert r.returncode==0,r.stderr[-3000:]
proof=json.loads(r.stdout);atomic(root/'runtime_configuration_check.json',proof)
atomic(root/'deployment.json',dict(at=time.time(),hostname=socket.gethostname(),source_changes=source_changes,all_energywam_files_identical_to_GPU67_control=True,old_queue_withdrawal=receipt,dependency=p['after_study'],gpu_allocation='Only6/7; no new GPU process before dependency completes',whole_study_checkpoint_budget_bytes=p['whole_study_checkpoint_budget_bytes'],free_bytes=shutil.disk_usage(root).free))
# Update the user-facing local-server dashboard without touching trainer releases.
tools=Path('/data3/xyx_zhuzicheng/projects/energyWAM_remote/tools');index=tools/'study_dashboard.json';d=json.loads(index.read_text());shutil.copy2(index,root/'prior_study_dashboard.json')
d['groups']=[g for g in d['groups'] if g['name']!='k']+[dict(name='k',root=str(root))];atomic(index,d)
(tools/'monitor_studies.py').write_text(PAYLOAD['dashboard'])
command=shlex.join([p['python'],'-u',str(code/'scripts/run_lab_after_study.py'),'--plan',str(root/'plan.json')])+' >> '+shlex.quote(str(root/'launcher.log'))+' 2>&1'
subprocess.run(['tmux','new-session','-d','-s',p['tmux_session'],'-c',str(code),command],check=True)
for _ in range(20):
 if (root/'controller_status.json').exists():break
 time.sleep(.2)
state=json.loads((root/'controller_status.json').read_text());assert state['phase']=='waiting_for_preceding_study',state
pid=state['pid'];assert b'run_lab_after_study.py' in Path('/proc',str(pid),'cmdline').read_bytes()
print(json.dumps(dict(plan=p,status=state,old_queue=receipt,source_changes=source_changes,environment=proof,source_manifest_sha256=hashlib.sha256((root/'source_manifest.json').read_bytes()).hexdigest())))
'''

if __name__=='__main__':main()

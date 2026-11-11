"""Prepare and queue the authorized joint step-size halving on idle lab GPUs6/7."""
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import yaml

ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / 'logs/lab_remote_setup/both_half67_deployment.json'


def main():
    assert not RECEIPT.exists(), 'Inspect the existing receipt; do not duplicate deployment'
    x = json.loads((ROOT / 'logs/lab_remote_setup/both_half67_preflight_20261111.json').read_text())
    assert x['hostname'] == 'user-SYS-420GP-TNR'
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    root = '/data8/xyx_zhuzicheng/energyWAM_studies/both_half_gpu67_' + stamp
    name = 'episode3346_both_half_gpu67_seed42'
    cfg = yaml.safe_load(x['future_config'])
    cfg['run_id'] = name
    cfg['trainer']['output_dir'] = root + '/' + name
    cfg['solver']['alpha_action'] = 2.5
    cfg['framework']['quadratic_action_weight'] = .5
    assert cfg['solver']['alpha_future'] == 1000
    assert cfg['framework']['quadratic_future_weight'] == .5
    assert cfg['framework']['quadratic_action_weight'] / cfg['solver']['alpha_action'] == .2
    assert cfg['framework']['quadratic_future_weight'] / cfg['solver']['alpha_future'] == .0005
    check = copy.deepcopy(cfg)
    reference = yaml.safe_load(x['future_config'])
    check['run_id'] = reference['run_id']
    check['trainer']['output_dir'] = reference['trainer']['output_dir']
    check['solver']['alpha_action'] = reference['solver']['alpha_action']
    check['framework']['quadratic_action_weight'] = reference['framework']['quadratic_action_weight']
    assert check == reference
    job = copy.deepcopy(x['job'])
    job.update(name=name, config=root+'/configs/'+name+'.yaml',
               execution_gpu_policy=root+'/gpu_policy.json',
               manifest=root+'/manifests/train_val_manifest.json',
               train_monitor_manifest=root+'/manifests/train_monitor_manifest.json')
    cap = copy.deepcopy(cfg)
    cap['run_id'] = 'capacity_both_half67'
    cap['trainer']['output_dir'] = root+'/capacity_profile'
    profile = copy.deepcopy(job)
    profile.update(name=cap['run_id'], config=root+'/configs/capacity.yaml', steps=2,
                   profile=True, keep_best=False)
    for k in ['sample_eval_interval','train_monitor_sample_interval','train_monitor_manifest','alternate_final_eval_seed']:
        profile.pop(k, None)
    plan = dict(name='bothhalf67', hostname=x['hostname'], root=root, code=x['plan']['code'],
                python=x['plan']['python'], gpu_pair='6,7',
                profile_jobs=root+'/profile_jobs.json', full_jobs=root+'/full_jobs.json',
                gradient_gate_result=x['plan']['gradient_gate_result'],
                whole_study_checkpoint_budget_bytes=3*job['snapshot_bytes_estimate']+8*2**30,
                arms=[dict(name=name,folder=cfg['trainer']['output_dir'],factors=dict(
                    initial_alpha_action=2.5,initial_alpha_future=1000.,lambda_future=9.,train_k=2,
                    effective_prior_action_precision=.2,effective_prior_future_precision=.0005,
                    learnable_range_action=[1.25,3.75],learnable_range_future=[500,1500]))],
                purpose='Complete 2x2 step-size comparison; both Action and Future initial steps halved',
                protocol='Fresh official4B seed42; B16/native4/FP32/K2; 210updates; A+9F; original LR/clip1/18x18/GPUoptimizer/activation10-14; fixed67TRAIN16VAL every20updates; fixed eval2026/final2027; original10x stop',
                retention='Full latest/best chosen by auxiliary16VAL Action+9Future; pending write reserved',
                tmux_session='energywam_both_half67_'+stamp,
                reference_study=x['plan']['root'],
                launch_condition='Both physicalGPU6 and7 have zero compute processes; never share or stop another process',
                gradient_gate_scope='Reuse passed original GPU67 native4/storage gate and byte-identical frozen code; fresh two-update capacity check before formal fresh210',
                authorization='User2026-11-11 explicitly requested waiting for both lab GPUs6/7 to be idle, then launching both-half study')
    payload = dict(plan=plan, job=job, profile=profile, config=yaml.safe_dump(cfg,sort_keys=False),
                   capacity=yaml.safe_dump(cap,sort_keys=False), policy=x['policy'],
                   source_job=x['job'], reference_action_config=x['action_config'])
    RECEIPT.write_text(json.dumps(dict(phase='prepared_not_confirmed_deployed',plan=plan),indent=2))
    result = subprocess.run(['ssh','-o','ConnectTimeout=10','-o','ServerAliveInterval=10',
                             '-o','ServerAliveCountMax=2','lab-via-pc','python3 -'],
                            input='PAYLOAD='+repr(payload)+'\n'+REMOTE,
                            text=True,capture_output=True,timeout=150)
    if result.returncode:
        (ROOT/'logs/lab_remote_setup/both_half67_deployment_error.txt').write_text(result.stderr)
        raise RuntimeError('Remote deployment did not confirm success; inspect receipt/root before retrying')
    remote = json.loads(result.stdout)
    RECEIPT.write_text(json.dumps(remote,indent=2))
    index_path = ROOT/'logs/lab_remote_setup/ablation_deployment.json'
    index = json.loads(index_path.read_text())
    assert not any(g['name']=='bothhalf67' for g in index['groups'])
    index['groups'].append(plan)
    index_path.write_text(json.dumps(index,indent=2))
    print(json.dumps(remote,indent=2))


REMOTE = r'''
import json,pathlib,shutil,socket,subprocess,hashlib,zipfile,shlex,time
from datetime import datetime,timezone
p=PAYLOAD['plan'];root=pathlib.Path(p['root']);code=pathlib.Path(p['code'])
assert socket.gethostname()==p['hostname']
assert not root.exists()
assert shutil.disk_usage(root.parent).free>=p['whole_study_checkpoint_budget_bytes']
assert not PAYLOAD['policy'].get('authorized_sharing')
assert json.loads(pathlib.Path(p['gradient_gate_result']).read_text())['passed']
processes=subprocess.check_output(['ps','-u',str(__import__('os').getuid()),'-o','args='],text=True)
assert 'deploy_both_half67' not in processes
assert not any('scripts/run_lab_ablation.py --plan ' in line for line in processes.splitlines()),'Existing lab controller must be inspected before another launch'
root.mkdir();(root/'configs').mkdir();(root/'manifests').mkdir()
for name,obj in [('plan.json',p),('full_jobs.json',[PAYLOAD['job']]),('profile_jobs.json',[PAYLOAD['profile']]),('gpu_policy.json',PAYLOAD['policy'])]:
 (root/name).write_text(json.dumps(obj,indent=2))
pathlib.Path(PAYLOAD['job']['config']).write_text(PAYLOAD['config'])
(root/'configs/capacity.yaml').write_text(PAYLOAD['capacity'])
hashes={}
for key,name in [('manifest','train_val_manifest.json'),('train_monitor_manifest','train_monitor_manifest.json')]:
 src=pathlib.Path(PAYLOAD['source_job'][key]);dest=root/'manifests'/name;shutil.copy2(src,dest)
 hashes[name]=hashlib.sha256(dest.read_bytes()).hexdigest();assert src.read_bytes()==dest.read_bytes()
manifest=[]
with zipfile.ZipFile(root/'study_source.zip','w',zipfile.ZIP_DEFLATED) as archive:
 for f in code.rglob('*.py'):
  rel=f.relative_to(code)
  if 'logs' in rel.parts or '__pycache__' in rel.parts:continue
  manifest.append(dict(path=str(rel),sha256=hashlib.sha256(f.read_bytes()).hexdigest()));archive.write(f,str(rel))
(root/'source_manifest.json').write_text(json.dumps(manifest,indent=2))
folder=pathlib.Path(p['arms'][0]['folder']);folder.mkdir()
(folder/'arm_definition.json').write_text(json.dumps(dict(job=PAYLOAD['job'],resolved_config_yaml=PAYLOAD['config'],purpose=p['purpose'],initialization='Fresh official4B/DINO seed42; no robot checkpoint resume',source_code=str(code),best_criterion='Aux16VAL Action MSE+9Future MSE',data_manifest_sha256=hashes,sample_visits=3360,equivalent_epochs=3360/67),indent=2))
check="""import sys,json,yaml,copy,importlib.metadata
from pathlib import Path
from energywam.config import load_config
r=Path(sys.argv[1]);p=json.loads((r/'plan.json').read_text());job=json.loads((r/'full_jobs.json').read_text())[0]
c=load_config(job['config']);new=yaml.safe_load(Path(job['config']).read_text());old=yaml.safe_load((Path(p['reference_study'])/'configs/future_half.yaml').read_text())
adjusted=copy.deepcopy(new)
for k in ['run_id']:adjusted[k]=old[k]
adjusted['trainer']['output_dir']=old['trainer']['output_dir'];adjusted['solver']['alpha_action']=old['solver']['alpha_action'];adjusted['framework']['quadratic_action_weight']=old['framework']['quadratic_action_weight']
assert adjusted==old
assert c.solver.alpha_action==2.5 and c.solver.alpha_future==1000
assert c.framework.quadratic_action_weight/c.solver.alpha_action==.2 and c.framework.quadratic_future_weight/c.solver.alpha_future==.0005
assert c.trainer.seed==42 and c.trainer.shared_microbatch_size==4 and c.solver.train_inner_steps==2
print(json.dumps(dict(passed=True,only_declared_step_change_vs_futurehalf=True,effective_prior_preserved=True,critical_versions={k:importlib.metadata.version(k) for k in ['torch','transformers','bitsandbytes','numpy']},python=sys.version)))"""
env=dict(__import__('os').environ,PYTHONPATH=str(code),CUDA_VISIBLE_DEVICES='')
v=subprocess.run([p['python'],'-c',check,str(root)],cwd=code,env=env,capture_output=True,text=True,timeout=45)
(root/'configuration_check_console.log').write_text(v.stdout+v.stderr);assert v.returncode==0,v.stderr
(root/'configuration_verification.json').write_text(v.stdout)
(root/'full_status.json').write_text(json.dumps({PAYLOAD['job']['name']:dict(phase='queued',completed_updates=0,target_updates=210,physical_gpus='6,7')},indent=2))
tools=pathlib.Path('/data3/xyx_zhuzicheng/projects/energyWAM_remote/tools');df=tools/'study_dashboard.json';d=json.loads(df.read_text())
assert not any(g['name']=='bothhalf67' for g in d['groups']);d['groups'].append(dict(name='bothhalf67',root=str(root)));tmp=df.with_suffix('.pending.json');tmp.write_text(json.dumps(d,indent=2));tmp.replace(df)
command=shlex.join([p['python'],'-u',str(code/'scripts/run_lab_ablation.py'),'--plan',str(root/'plan.json')])+' >> '+shlex.quote(str(root/'launcher.log'))+' 2>&1'
(root/'launch_receipt.json').write_text(json.dumps(dict(command=command,tmux=p['tmux_session'],started_at=datetime.now(timezone.utc).isoformat(),free_bytes=shutil.disk_usage(root).free),indent=2))
subprocess.run(['tmux','new-session','-d','-s',p['tmux_session'],'-c',str(code),command],check=True)
for _ in range(40):
 if (root/'controller_status.json').exists() and (root/'profile_status.json').exists():break
 time.sleep(.25)
def read(name):
 f=root/name;return json.loads(f.read_text()) if f.exists() else None
print(json.dumps(dict(plan=p,controller=read('controller_status.json'),queue=read('profile_status.json'),verification=json.loads(v.stdout),phase='deployed',new_formal_updates=0)))
'''


if __name__ == '__main__':
    main()

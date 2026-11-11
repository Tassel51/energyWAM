"""Rebalance the measured GPU7 bottleneck while preserving native4 math."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.lab_pc_relay import remote_python


def main():
 path=ROOT/'logs/lab_remote_setup/future07_deployment.json';d=json.loads(path.read_text());p=d['plan'];p['gradient_gate_after_capacity']=True;p['purpose']='Future-half retry on user GPU0/7, GPU7 primary19layers/GPU0 17layers, B16/native4/FP32/K2, GPU optimizer states, CPU activation storage'
 r=remote_python('lab-via-pc','PLAN='+repr(p)+'\n'+REMOTE,retry_safe=False)
 (ROOT/'logs/lab_remote_setup/future07_19x17_relayout.json').write_text(json.dumps(r,ensure_ascii=False,indent=2));d['plan']=p;path.write_text(json.dumps(d,ensure_ascii=False,indent=2));index=ROOT/'logs/lab_remote_setup/ablation_deployment.json';x=json.loads(index.read_text());x['groups']=[p if g['name']=='future07' else g for g in x['groups']];index.write_text(json.dumps(x,ensure_ascii=False,indent=2));print(json.dumps(r,indent=2))

REMOTE=r'''
from pathlib import Path
import json,shutil,time,subprocess,shlex,hashlib,difflib
root=Path(PLAN['root']);code=Path(PLAN['code']);base=Path(PLAN['source_base']);state=json.loads((root/'controller_status.json').read_text());assert state['phase']=='blocked_capacity'
assert not any(Path(a['folder'],'train_log.jsonl').exists() for a in PLAN['arms'])
archive=root/'failed_setup_attempts'/('primary7_20x16_oom_'+str(int(time.time())));archive.mkdir(parents=True)
for name in ['controller_status.json','gradient_gate.log','gradient_gate_resources','native_gradient_gate','capacity_profile','profile_status.json','profile_controller.log','launcher.log','configs','full_jobs.json','profile_jobs.json','plan.json','actual_configuration_comparison.json','runtime_configuration_check.json','source_manifest.json','source_changes.json']:
 f=root/name
 if f.exists():
  if f.is_dir():shutil.copytree(f,archive/name)
  else:shutil.copy2(f,archive/name)
for name in ['native_gradient_gate','capacity_profile','gradient_gate_resources']:
 f=root/name
 if f.exists():shutil.rmtree(f)
for name in ['gradient_gate.log','profile_status.json','profile_controller.log','launcher.log']:
 f=root/name
 if f.exists():f.unlink()
patch="""import sys,yaml,json
from pathlib import Path
from energywam.config import load_config
r=Path(sys.argv[1]);out=[]
for n in ['future_half.yaml','capacity.yaml']:
 f=r/'configs'/n;c=yaml.safe_load(f.read_text());c['trainer']['language_model_layer_counts']=[19,17];f.write_text(yaml.safe_dump(c,sort_keys=False));x=load_config(f);assert x.trainer.shared_microbatch_size==4 and x.trainer.language_model_layer_counts==[19,17]
 out.append(dict(file=str(f),layers=x.trainer.language_model_layer_counts,gpu_cache=x.trainer.saved_tensor_gpu_budgets_gib))
print(json.dumps(out))"""
r=subprocess.run([PLAN['python'],'-c',patch,str(root)],cwd=code,capture_output=True,text=True,timeout=30);assert r.returncode==0,r.stderr
(root/'runtime_configuration_check.json').write_text(r.stdout)
f=code/'scripts/run_lab_ablation.py';s=f.read_text();needle="if plan.get('gradient_gate_command'):";assert needle in s;s=s.replace(needle,"if plan.get('gradient_gate_command') and not plan.get('gradient_gate_after_capacity',False):")
needle="        gate=Path(plan['gradient_gate_result'])\n        while not STOP"
insert="""        if plan.get('gradient_gate_after_capacity',False):
            state['phase']='gradient_gate'
            result=execute(plan['gradient_gate_command'],root/'gradient_gate.log',state_file,state,gpu_pair=plan['gpu_pair'])
            if STOP:raise KeyboardInterrupt
            if result or not read(plan['gradient_gate_result'],{}).get('passed'):raise RuntimeError('Gradient/storage equivalence gate did not pass')
        gate=Path(plan['gradient_gate_result'])
        while not STOP"""
assert needle in s;s=s.replace(needle,insert);compile(s,str(f),'exec');f.write_text(s)
(root/'plan.json').write_text(json.dumps(PLAN,indent=2));changes=[];manifest=[]
for f in code.rglob('*.py'):
 if '__pycache__' in f.parts or 'logs' in f.relative_to(code).parts:continue
 rel=str(f.relative_to(code));manifest.append(dict(path=rel,sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
 if (base/rel).is_file() and f.read_bytes()!=(base/rel).read_bytes():changes.append(rel);(root/(f.name+'.diff')).write_text(''.join(difflib.unified_diff((base/rel).read_text().splitlines(True),f.read_text().splitlines(True))))
(root/'source_manifest.json').write_text(json.dumps(manifest,indent=2));(root/'source_changes.json').write_text(json.dumps(changes,indent=2))
f=Path(PLAN['arms'][0]['folder'])/'arm_definition.json';a=json.loads(f.read_text());a['config_yaml']=(root/'configs/future_half.yaml').read_text();a['source_changes']=changes;a['execution_changes']=dict(physical_gpus=[7,0],language_layer_counts=[19,17],gpu_activation_cache=[0,0],optimizer_state_cpu_offload=False);a['memory_attempts'].append(dict(layers=[20,16],phase='failed_full_capacity_gpu7_oom',archive=str(archive)));a['memory_attempts'].append(dict(layers=[19,17],status='pending_capacity_then_gradient_verification'));f.write_text(json.dumps(a,indent=2))
command=shlex.join([PLAN['python'],'-u',PLAN['code']+'/scripts/run_lab_ablation.py','--plan',str(root/'plan.json')])+' >> '+shlex.quote(str(root/'launcher.log'))+' 2>&1'
assert subprocess.run(['tmux','has-session','-t',PLAN['tmux_session']],capture_output=True).returncode!=0
subprocess.run(['tmux','new-session','-d','-s',PLAN['tmux_session'],'-c',PLAN['code'],command],check=True)
print(json.dumps(dict(archive=str(archive),configuration=json.loads(r.stdout),source_changes=changes,formal_updates=0)))
'''

if __name__=='__main__':main()

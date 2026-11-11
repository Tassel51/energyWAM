"""Apply the validated explicit placement option after 18/18 capacity failed."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.lab_pc_relay import remote_python


def main():
 path=ROOT/'logs/lab_remote_setup/future07_deployment.json';d=json.loads(path.read_text());p=d['plan'];p['purpose']='Finish Future-half on user GPU0/7; primary7 language20layers/secondary0 language16; native4 with GPU Adam states and CPU saved activations'
 payload=dict(plan=p,parallel=(ROOT/'energywam/training/parallel.py').read_text())
 r=remote_python('lab-via-pc','PAYLOAD='+repr(payload)+'\n'+REMOTE,retry_safe=False)
 (ROOT/'logs/lab_remote_setup/future07_20x16_relayout.json').write_text(json.dumps(r,ensure_ascii=False,indent=2));d['plan']=p;d['all_energywam_sources_match_original_GPU67']=False;path.write_text(json.dumps(d,ensure_ascii=False,indent=2));xp=ROOT/'logs/lab_remote_setup/ablation_deployment.json';x=json.loads(xp.read_text());x['groups']=[p if g['name']=='future07' else g for g in x['groups']];xp.write_text(json.dumps(x,ensure_ascii=False,indent=2));print(json.dumps(r,ensure_ascii=False,indent=2))

REMOTE=r'''
from pathlib import Path
import json,shutil,time,subprocess,shlex,hashlib,difflib
p=PAYLOAD['plan'];root=Path(p['root']);code=Path(p['code']);base=Path(p['source_base'])
s=json.loads((root/'controller_status.json').read_text());assert s['phase']=='blocked_capacity'
assert not any(Path(a['folder'],'train_log.jsonl').exists() for a in p['arms'])
archive=root/'failed_setup_attempts'/('primary7_18x18_oom_'+str(int(time.time())));archive.mkdir(parents=True)
for name in ['controller_status.json','gradient_gate.log','gradient_gate_resources','native_gradient_gate','capacity_profile','profile_status.json','profile_controller.log','launcher.log','configs','gpu_policy.json','full_jobs.json','profile_jobs.json','plan.json']:
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
f=code/'energywam/config.py';text=f.read_text();needle='    language_model_devices: List[int] = field(default_factory=list)';assert needle in text
text=text.replace(needle,needle+'\n    language_model_layer_counts: List[int] = field(default_factory=list)');f.write_text(text)
f=code/'energywam/training/trainer.py';text=f.read_text();needle='shard_language_model(model, cfg.trainer.language_model_devices, self.device)';assert needle in text
text=text.replace(needle,'shard_language_model(model, cfg.trainer.language_model_devices, self.device, layer_counts=cfg.trainer.language_model_layer_counts)');f.write_text(text)
(code/'energywam/training/parallel.py').write_text(PAYLOAD['parallel'])
patch="""from pathlib import Path
import yaml,sys,json
r=Path(sys.argv[1]);out=[]
for file in ['future_half.yaml','capacity.yaml']:
 p=r/'configs'/file;c=yaml.safe_load(p.read_text());c['trainer']['language_model_layer_counts']=[20,16];c['trainer']['saved_tensor_gpu_budgets_gib']=[0.,0.];p.write_text(yaml.safe_dump(c,sort_keys=False))
from energywam.config import load_config
for p in (r/'configs').glob('*.yaml'):
 c=load_config(p);assert c.trainer.language_model_layer_counts==[20,16] and c.trainer.shared_microbatch_size==4
 out.append(dict(file=str(p),layers=c.trainer.language_model_layer_counts,gpu_budgets=c.trainer.saved_tensor_gpu_budgets_gib,batch=c.trainer.per_device_batch_size,native=c.trainer.shared_microbatch_size))
print(json.dumps(out))"""
r=subprocess.run([p['python'],'-c',patch,str(root)],cwd=code,capture_output=True,text=True,timeout=30);assert r.returncode==0,r.stderr
(root/'runtime_configuration_check.json').write_text(r.stdout)
(root/'plan.json').write_text(json.dumps(p,indent=2));changes=[];manifest=[]
for f in code.rglob('*.py'):
 if '__pycache__' in f.parts or 'logs' in f.relative_to(code).parts:continue
 rel=str(f.relative_to(code));manifest.append(dict(path=rel,sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
 if (base/rel).is_file() and f.read_bytes()!=(base/rel).read_bytes():
  changes.append(rel);(root/(f.name+'.diff')).write_text(''.join(difflib.unified_diff((base/rel).read_text().splitlines(True),f.read_text().splitlines(True))))
assert set(changes)=={'energywam/config.py','energywam/training/trainer.py','energywam/training/parallel.py','scripts/run_iteration_search.py'}
(root/'source_manifest.json').write_text(json.dumps(manifest,indent=2));(root/'source_changes.json').write_text(json.dumps(changes,indent=2))
afile=Path(p['arms'][0]['folder'])/'arm_definition.json';a=json.loads(afile.read_text());a['config_yaml']=(root/'configs/future_half.yaml').read_text();a['purpose']=p['purpose'];a['source_changes']=changes;a['memory_attempts'].append(dict(primary_physical_gpu=7,layers=[18,18],phase='failed_full_capacity_oom',archive=str(archive)));a['memory_attempts'].append(dict(primary_physical_gpu=7,layers=[20,16],gpu_activation_cache=[0,0],status='pending_validation'));afile.write_text(json.dumps(a,indent=2))
command=shlex.join([p['python'],'-u',p['code']+'/scripts/run_lab_ablation.py','--plan',str(root/'plan.json')])+' >> '+shlex.quote(str(root/'launcher.log'))+' 2>&1'
assert subprocess.run(['tmux','has-session','-t',p['tmux_session']],capture_output=True).returncode!=0
subprocess.run(['tmux','new-session','-d','-s',p['tmux_session'],'-c',p['code'],command],check=True)
print(json.dumps(dict(archive=str(archive),configuration=json.loads(r.stdout),changes=changes,formal_updates=0)))
'''

if __name__=='__main__':main()

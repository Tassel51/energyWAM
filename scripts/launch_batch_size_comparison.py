"""Preserve completed references, provision rolling space, then launch owned jobs."""
from pathlib import Path
import shutil,hashlib,json,os,subprocess,time
R=Path('/cloud/zzc/energyWAM/experiments/iteration_aware_v1_20261104')
A=Path('/root/zzc/energyWAM/checkpoint_archive/iteration_aware_v1_20261104')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as stream:
  for part in iter(lambda:stream.read(16*2**20),b''):h.update(part)
 return h.hexdigest()
for name in ['ablation_prior1_currentinit1_seed42','ablation_prior1_currentinit0_seed42','no_prior_currentinit_fixed_headlr0001_diag_seed42','no_prior_currentinit_learned_headlr0001_diag_seed42']:
 source=R/name/'model_reference.pt'
 if source.is_symlink():continue
 target=A/name/'model_reference.pt';target.parent.mkdir(parents=True,exist_ok=True)
 assert shutil.disk_usage(A).free>source.stat().st_size+2*2**30
 pending=target.with_suffix('.pending.pt');shutil.copyfile(source,pending)
 digest=sha(source);assert sha(pending)==digest
 with pending.open('rb') as stream:os.fsync(stream.fileno())
 os.replace(pending,target)
 fd=os.open(target.parent,os.O_DIRECTORY);os.fsync(fd);os.close(fd)
 link=source.with_suffix('.pending.link');link.symlink_to(target);os.replace(link,source)
 print('Archived byte-identical completed model reference:',name,flush=True)
 audit=R/'checkpoint_retention.json';rows=json.loads(audit.read_text());rows.append(dict(label=name,action='relocate_model_reference',destination=str(target),file_sha256=digest,reason='Provision batch comparison rolling checkpoint space; no model values removed'));audit.write_text(json.dumps(rows,indent=2))
for _ in range(30):
 if shutil.disk_usage(R).free>94*2**30:break
 time.sleep(2)
else:raise RuntimeError('checkpoint headroom insufficient; no training launched')
p=R/'batch_size_comparison_plan.json';plan=json.loads(p.read_text())
for pair,job in zip(['01','23'],plan['jobs']):
 q=R/f'search_jobs_{pair}.json';rows=json.loads(q.read_text())
 if not any(x['name']==job['name'] for x in rows):rows.append(job)
 q.write_text(json.dumps(rows,indent=2))
plan['status']='queued';p.write_text(json.dumps(plan,indent=2))
subprocess.run(['supervisorctl','start','energywam-iteration-search01','energywam-iteration-search23'],check=True)

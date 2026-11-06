import sys,os,json,gc
from pathlib import Path
sys.path.insert(0,'scripts')
from retire_iteration_controls import fingerprint
import torch
r=Path('/cloud/zzc/energyWAM/experiments/iteration_aware_v1_20261104')
for name,step in [('iter_learned',20)]:
 f=r/name;p=f/'ckpt_latest.pt';dst=f/'model_reference.pt'
 if not p.exists() and dst.exists():continue
 assert not p.is_symlink()
 assert json.loads((f/'final_report.json').read_text())['global_step']==step
 x=torch.load(p,map_location='cpu',mmap=True,weights_only=False);assert x['meta']['global_step']==step
 h=fingerprint(x['model']);a=dict(model=x['model'],config=x['config'],meta=x['meta'],rng=x.get('rng'),optimizer=None,scheduler=None,reference_only=True,format_version=x.get('format_version',1))
 tmp=f/'model_reference.pending.pt';torch.save(a,tmp)
 with tmp.open('rb') as stream:os.fsync(stream.fileno())
 y=torch.load(tmp,map_location='cpu',mmap=True,weights_only=False);assert fingerprint(y['model'])==h;del y
 os.replace(tmp,dst)
 record=dict(label=name,step=step,model_values_sha256=h,optimizer_states_retired=True,reason='Completed20-update pilot, supervisor confirmed EXITED; preserve exact model and all logs; reclaim optimizer for user-authorized batch4 comparison',reference_only=True)
 (f/'model_reference.pt.meta.json').write_text(json.dumps(record,indent=2))
 m=Path(str(p)+'.meta.json')
 if m.exists():os.replace(m,f/'retired_full_checkpoint.meta.json')
 p.unlink();audit=r/'checkpoint_retention.json';j=json.loads(audit.read_text());j.append(record);audit.write_text(json.dumps(j,indent=2));del x,a;gc.collect();print(name,'preserved',flush=True)

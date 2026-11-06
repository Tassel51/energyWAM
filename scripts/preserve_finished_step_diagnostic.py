import sys,os,json,gc
from pathlib import Path
sys.path.insert(0,'scripts')
from retire_iteration_controls import fingerprint
import torch
r=Path('/cloud/zzc/energyWAM/experiments/iteration_aware_v1_20261104')
import argparse
p=argparse.ArgumentParser();p.add_argument('--name',required=True,choices=['no_prior_currentinit_fixed_headlr0001_diag_seed42','no_prior_currentinit_learned_headlr0001_diag_seed42']);args=p.parse_args()
for name,step in [(args.name,80)]:
 f=r/name;p=f/'ckpt_latest.pt';dst=f/'model_reference.pt'
 if not p.exists() and dst.exists():continue
 assert not p.is_symlink()
 state=json.loads((r/'search_status_23.json').read_text())[name]
 assert state['phase']=='complete'
 try:os.kill(state['pid'],0)
 except ProcessLookupError:pass
 else:raise RuntimeError('diagnostic process still live')
 assert json.loads((f/'final_report.json').read_text())['global_step']==step
 x=torch.load(p,map_location='cpu',mmap=True,weights_only=False);assert x['meta']['global_step']==step
 h=fingerprint(x['model']);a=dict(model=x['model'],config=x['config'],meta=x['meta'],rng=x.get('rng'),optimizer=None,scheduler=None,reference_only=True,format_version=x.get('format_version',1))
 tmp=f/'model_reference.pending.pt';torch.save(a,tmp)
 with tmp.open('rb') as stream:os.fsync(stream.fileno())
 y=torch.load(tmp,map_location='cpu',mmap=True,weights_only=False);assert fingerprint(y['model'])==h;del y
 os.replace(tmp,dst)
 record=dict(label=name,step=step,model_values_sha256=h,optimizer_states_retired=True,reason='Completed diagnostic; preserve exact model and all evaluation/configuration logs; reclaim optimizer for paired ablation',reference_only=True)
 (f/'model_reference.pt.meta.json').write_text(json.dumps(record,indent=2))
 m=Path(str(p)+'.meta.json')
 if m.exists():os.replace(m,f/'retired_full_checkpoint.meta.json')
 p.unlink();audit=r/'checkpoint_retention.json';j=json.loads(audit.read_text());j.append(record);audit.write_text(json.dumps(j,indent=2));del x,a;gc.collect();print(name,'preserved',flush=True)

"""Preserve a completed bounded control while reserving cloud rolling capacity."""
from pathlib import Path
import json,time,os,sys,torch,gc,shutil
sys.path.insert(0,'scripts')
from retire_iteration_controls import fingerprint
R=Path('/cloud/zzc/energyWAM/experiments/iteration_aware_v1_20261104');name='batch2_tiny16_headlr0003_samples320_seed42';folder=R/name
for _ in range(360):
 s=json.loads((R/'search_status_23.json').read_text()).get(name,{})
 if s.get('phase')=='failed':print('Control failed: logs preserved, no optimizer retirement needed.',flush=True);raise SystemExit(0)
 if s.get('phase')=='complete':
  try:os.kill(s['pid'],0)
  except ProcessLookupError:break
 time.sleep(15)
else:raise RuntimeError('control not terminal; do not alter active checkpoint')
p=folder/'ckpt_latest.pt';dst=Path('/root/zzc/energyWAM/checkpoint_archive/iteration_aware_v1_20261104')/name/'model_reference.pt';dst.parent.mkdir(parents=True,exist_ok=True)
assert json.loads((folder/'final_report.json').read_text())['global_step']==160
assert shutil.disk_usage(dst.parent).free>p.stat().st_size*.26+2*2**30
x=torch.load(p,map_location='cpu',mmap=True,weights_only=False);assert x['meta']['global_step']==160
h=fingerprint(x['model']);tmp=dst.with_suffix('.pending.pt');torch.save(dict(model=x['model'],config=x['config'],meta=x['meta'],rng=x.get('rng'),optimizer=None,scheduler=None,reference_only=True,format_version=x.get('format_version',1)),tmp)
with tmp.open('rb') as stream:os.fsync(stream.fileno())
y=torch.load(tmp,map_location='cpu',mmap=True,weights_only=False);assert fingerprint(y['model'])==h;del y
os.replace(tmp,dst);public=folder/'model_reference.pt';public.symlink_to(dst)
record=dict(label=name,step=160,model_reference=str(public),model_values_sha256=h,optimizer_states_retired=True,reason='Completed bounded B2 failure recheck; preserve exact model and all logs while keeping mainB2 and candidateB8 full optimizer checkpoints within disk capacity')
Path(str(public)+'.meta.json').write_text(json.dumps(record,indent=2));os.replace(Path(str(p)+'.meta.json'),folder/'retired_full_checkpoint.meta.json');p.unlink()
audit=R/'checkpoint_retention.json';rows=json.loads(audit.read_text());rows.append(record);audit.write_text(json.dumps(rows,indent=2));del x;gc.collect();print(json.dumps(record),flush=True)

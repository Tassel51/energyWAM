"""Retire only the completed, baseline-equivalent 40-update prior control."""
from pathlib import Path
import json,os,torch,sys
from retire_iteration_controls import fingerprint
R=Path('/cloud/zzc/energyWAM/experiments/iteration_aware_v1_20261104')
name='tinyfit_master_prior_residual_r2';folder=R/name;source=folder/'ckpt_latest.pt'
if not source.exists():
 assert (folder/'model_reference.pt').exists();raise SystemExit('Already preserved as reference')
state=json.loads((R/'search_status_01.json').read_text())[name];assert state['phase']=='complete'
try:os.kill(state['pid'],0)
except ProcessLookupError:pass
else:raise RuntimeError('live process')
assert json.loads((folder/'final_report.json').read_text())['global_step']==40
for direction in ['action','future']:
 def m(step):
  rows=json.loads((folder/f'eval_{step:04d}_{direction}.json').read_text())['rows'];v=[x['prediction_mse'] for x in rows if x['split']=='train' and x['k']==2];return sum(v)/len(v)
 assert abs(m(40)/m(0)-1)<.001
payload=torch.load(source,map_location='cpu',mmap=True,weights_only=False);assert payload['meta']['global_step']==40
before=fingerprint(payload['model']);meta=dict(payload['meta']);meta.update(optimizer_policy='reset',scheduler_policy='reset',notes='Read-only reference: completed flat prior baseline; preserve every model tensor, retire optimizer/master residual states; exact resume unavailable')
dest=Path('/root/zzc/energyWAM/checkpoint_archive/iteration_aware_v1_20261104')/name/'model_reference.pt';dest.parent.mkdir(parents=True,exist_ok=True);tmp=dest.with_suffix('.pending.pt')
artifact=dict(model=payload['model'],config=payload['config'],meta=meta,rng=payload.get('rng'),optimizer=None,scheduler=None,format_version=payload.get('format_version',1),reference_only=True)
torch.save(artifact,tmp)
with tmp.open('rb') as f:os.fsync(f.fileno())
check=torch.load(tmp,map_location='cpu',mmap=True,weights_only=False);assert fingerprint(check['model'])==before
os.replace(tmp,dest);public=folder/'model_reference.pt';public.symlink_to(dest)
record=dict(label=name,step=40,model_reference=str(public),model_values_sha256=before,optimizer_states_retired=True,full_checkpoint_bytes=source.stat().st_size,reference_bytes=dest.stat().st_size,reason='Both fixed train MSE changed <0.1%; completed baseline-equivalent nonfinalist; all model tensor values preserved exactly; active finalists and formal/old K references untouched')
Path(str(public)+'.meta.json').write_text(json.dumps(record,indent=2));os.replace(Path(str(source)+'.meta.json'),folder/'retired_full_checkpoint.meta.json');source.unlink()
p=R/'checkpoint_retention.json';audit=json.loads(p.read_text());audit.append(record);p.write_text(json.dumps(audit,indent=2));print(json.dumps(record),flush=True)

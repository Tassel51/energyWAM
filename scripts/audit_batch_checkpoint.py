"""CPU mmap structural recoverability audit; no training or CUDA allocation."""
import argparse,json,sys
from pathlib import Path
import torch
R=Path('/cloud/zzc/energyWAM/experiments/iteration_aware_v1_20261104')
def audit(name,expected):
 folder=R/name;path=folder/'ckpt_latest.pt';payload=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
 assert payload['meta']['stage']=='bi' and payload['meta']['global_step']==expected
 sidecar=json.loads(Path(str(path)+'.meta.json').read_text());assert sidecar['meta']==payload['meta']
 assert not payload.get('reference_only',False)
 optimizer=payload['optimizer'];assert optimizer and optimizer['state'] and optimizer['param_groups'] and optimizer['fp32_master_values']
 assert payload['scheduler'] is not None and {'python','numpy','torch','cuda'}<=payload['rng'].keys()
 assert len(payload['rng']['cuda'])==2
 for t in payload['model'].values():
  if torch.is_tensor(t) and t.is_floating_point():assert torch.isfinite(t.reshape(-1)[:64]).all()
 for t in optimizer['fp32_master_values']:
  assert t.dtype==torch.float32 and torch.isfinite(t.reshape(-1)[:64]).all()
 result=dict(checkpoint=str(path),bytes=path.stat().st_size,global_step=expected,stage='bi',model_tensors=len(payload['model']),optimizer_states=len(optimizer['state']),fp32_master_tensors=len(optimizer['fp32_master_values']),scheduler_present=True,rng_components=list(payload['rng']),cuda_rng_devices=2,metadata_matches_sidecar=True,pending_files=[str(p) for p in folder.glob('ckpt_pending*')],scope='CPU mmap container/state structure and first64 floating values per model/master tensor checked. Saved model restored for actual GPU validation separately. This does not claim a live resume-update replay.')
 assert not result['pending_files']
 proof=R/f'{name}_resume_update_verification/result.json'
 if proof.exists():
  live=json.loads(proof.read_text())
  if live.get('passed') and live.get('restored_step')==expected and live.get('source_mtime_ns')==path.stat().st_mtime_ns and live.get('source_size')==path.stat().st_size:
   result['live_resume_update_verification']=str(proof);result['live_resume_update_passed']=True
 (folder/'checkpoint_recoverability_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('--step',type=int,required=True);a=p.parse_args();audit(a.name,a.step)

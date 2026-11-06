#!/usr/bin/env python
"""Explicitly retire optimizer states of completed dominated goal controls.

Preserves byte-identical model tensors, config, train-only normalization, RNG,
all logs and reports. Never touches formal BI2300 or old K2/K4 references.
Default is read-only; --execute performs the listed retirements.
"""
import argparse,gc,hashlib,json,os,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
OUT=Path('/cloud/zzc/energyWAM/experiments/iteration_aware_v1_20261104')
LABELS=['tinyfit_master_tasklr_r2','tinyfit_master_tasklr_f2000_r2','tinyfit_master_rms05_r2']


def fingerprint(model):
    digest=hashlib.sha256();seen=set()
    for name,value in sorted(model.items()):
        digest.update(name.encode());digest.update(str((tuple(value.shape),str(value.dtype))).encode())
        identity=(value.untyped_storage().data_ptr(),value.storage_offset(),value.numel(),str(value.dtype))
        if identity in seen:continue
        seen.add(identity)
        data=value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
        view=memoryview(data)
        for offset in range(0,len(view),64*2**20):digest.update(view[offset:offset+64*2**20])
    return digest.hexdigest()


def validation(folder,direction,step=40):
    report=json.loads((folder/f'eval_{step:04d}_{direction}.json').read_text())
    assert report['weight_updates_during_eval']==0
    rows=[r for r in report['rows'] if r['split']=='val' and r['k']==2]
    assert len(rows)==16
    return sum(r['prediction_mse'] for r in rows)/len(rows),{r['window_id'] for r in rows}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    statuses={}
    for pair in ['01','23']:statuses.update(json.loads((OUT/f'search_status_{pair}.json').read_text()))
    reference=OUT/'tinyfit_master_prior_tokenscale_r2'
    audit_path=OUT/'checkpoint_retention.json'
    audit=json.loads(audit_path.read_text()) if audit_path.exists() else []
    for label in LABELS:
        folder=OUT/label;source=folder/'ckpt_latest.pt';public_ref=folder/'model_reference.pt'
        if not source.exists() and public_ref.exists():continue
        status=statuses[label]
        if status['phase']!='complete':raise RuntimeError(f'{label} is not complete')
        try:os.kill(status['pid'],0)
        except ProcessLookupError:pass
        else:raise RuntimeError('refusing to retire a live process checkpoint')
        step=json.loads((folder/'final_report.json').read_text())['global_step']
        if step!=40:raise RuntimeError('only completed forty-update controls are eligible')
        for d in ['action','future']:
            value,ids=validation(folder,d);better,better_ids=validation(reference,d,160)
            if ids!=better_ids or not better<value:raise RuntimeError('not dominated on identical validation windows')
        original=source.resolve();size=original.stat().st_size
        print(json.dumps(dict(control=label,source=str(original),bytes=size,execute=args.execute)),flush=True)
        if not args.execute:continue
        payload=torch.load(original,map_location='cpu',mmap=True,weights_only=False)
        if payload['meta']['global_step']!=step:raise RuntimeError('checkpoint/report step mismatch')
        before=fingerprint(payload['model'])
        dest=original.parent/'model_reference.pt';temp=dest.with_suffix('.pending.pt')
        meta=dict(payload['meta']);meta.update(optimizer_policy='reset',scheduler_policy='reset',
            notes=meta.get('notes','')+'; READ-ONLY REFERENCE: optimizer, scheduler and FP32 master residuals retired after dominated-control screening')
        artifact=dict(model=payload['model'],config=payload['config'],meta=meta,rng=payload.get('rng'),
            optimizer=None,scheduler=None,format_version=payload.get('format_version',1),reference_only=True)
        torch.save(artifact,temp)
        with temp.open('rb') as stream:os.fsync(stream.fileno())
        restored=torch.load(temp,map_location='cpu',mmap=True,weights_only=False)
        if fingerprint(restored['model'])!=before:raise RuntimeError('model reference is not byte-identical')
        del restored
        os.replace(temp,dest)
        if public_ref!=dest:
            pending_link=public_ref.with_suffix('.pending.link');pending_link.unlink(missing_ok=True)
            pending_link.symlink_to(dest);os.replace(pending_link,public_ref)
        reference_meta=dict(meta=meta,reference_only=True,model_values_sha256=before,original_full_bytes=size)
        Path(str(public_ref)+'.meta.json').write_text(json.dumps(reference_meta,indent=2))
        old_meta=Path(str(source)+'.meta.json')
        if old_meta.exists():os.replace(old_meta,folder/'retired_full_checkpoint.meta.json')
        was_link=source.is_symlink();source.unlink()
        if was_link:original.unlink()
        del artifact,payload;gc.collect()
        record=dict(label=label,step=step,model_reference=str(public_ref),model_values_sha256=before,
                    optimizer_states_retired=True,full_checkpoint_bytes=size,reference_bytes=dest.stat().st_size,
                    reason='completed diagnostic control; both held-out MSE exceed stored prior160 reference; retention only, not causal evidence for prior; all model tensors verified identical')
        audit.append(record);audit_path.write_text(json.dumps(audit,indent=2))
        print(json.dumps(record),flush=True)
if __name__=='__main__':main()

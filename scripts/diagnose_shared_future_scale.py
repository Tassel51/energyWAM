"""Read-only step-scale diagnosis on a frozen, sharded checkpoint."""
import argparse,json,sys,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))

def main():
    p=argparse.ArgumentParser();p.add_argument('--config');p.add_argument('--checkpoint');p.add_argument('--manifest');p.add_argument('--output');args=p.parse_args()
    import os
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='2,3':raise RuntimeError('only GPUs2/3')
    import torch
    from energywam.config import load_config
    from energywam.models.energy_wam import QwenEnergyWAM
    from energywam.training.parallel import shard_language_model
    from energywam.training.checkpoint import load_checkpoint,bundle_from_payload,restore
    from energywam.energy.inner_diagnostic import trace_candidate
    from energywam.data.energy_dataset import collate_energy_batch
    from k_training_probe import fixed_dataset
    cfg=load_config(args.config);payload=load_checkpoint(args.checkpoint);bundle=bundle_from_payload(payload)
    model=QwenEnergyWAM(cfg,stats_bundle=bundle).to('cuda:0')
    shard_language_model(model,[0,1],torch.device('cuda:0'))
    restore(payload,model=model,mode='resume',restore_rng=False);del payload
    model.eval().requires_grad_(False);versions=[p._version for p in model.parameters()]
    source=json.loads(Path(args.manifest).read_text());seen=set();windows=[]
    for w in source['windows']:
        key=(w['split'],w['task'])
        if key not in seen:seen.add(key);windows.append(w)
    manifest=dict(source,windows=windows);rows=[]
    for split in ['train','val']:
        dataset,selected=fixed_dataset(cfg,model,manifest,split)
        for i,w in enumerate(selected):
            batch=collate_energy_batch([dataset[i]])
            with torch.no_grad():prepared=model._prepare_batch(batch);cond=model._condition(batch,prepared)
            generator=torch.Generator(device='cuda:0').manual_seed(2026+w['window_id']*1009)
            a0,z0=model.stage_runner._init(prepared,torch.device('cuda:0'),generator=generator,cond=cond)
            def energy(z):return model.core.energy(cond,z,prepared['action_target'],future_mask=prepared['future_mask'],action_mask=prepared['action_mask'],offsets=prepared['future_offsets'])
            for alpha in [.05,5.,50.,500.]:
                trace,_=trace_candidate(energy,z0,prepared['future_target'],alpha,2)
                rows.extend(dict(r,split=split,task=w['task'],alpha=alpha) for r in trace)
    assert versions==[p._version for p in model.parameters()]
    summaries=[]
    for alpha in [.05,5.,50.,500.]:
        last=[r for r in rows if r['alpha']==alpha and r['k']==2]
        initial=[r for r in rows if r['alpha']==alpha and r['k']==0]
        first=[r for r in rows if r['alpha']==alpha and r['k']==1]
        summaries.append({'alpha':alpha,'initial_mse':statistics.mean(r['prediction_mse'] for r in initial),
            'final_mse':statistics.mean(r['prediction_mse'] for r in last),'first_update_rms':statistics.median(r['applied_update_rms'] for r in first)})
    stable=[r for r in summaries if r['alpha']>.05 and r['final_mse']<1.2*r['initial_mse'] and 1e-5<r['first_update_rms']<.3]
    chosen=min(stable,key=lambda r:abs(r['first_update_rms']-.02))['alpha'] if stable else None
    Path(args.output).write_text(json.dumps({'rows':rows,'summaries':summaries,'candidate_alpha':chosen,'selection':'stable step magnitude near .02, candidate only; not a trained optimum','weight_updates':0,'scope':'one fixed window per task per split'},indent=2))

if __name__=='__main__':main()

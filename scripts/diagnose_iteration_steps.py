#!/usr/bin/env python
"""Read-only 4B candidate step-scale check on one explicitly selected idle GPU."""
import argparse,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--manifest',required=True)
    p.add_argument('--output',required=True);p.add_argument('--physical-gpu',default='0');p.add_argument('--limit',type=int,default=10)
    a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=a.physical_gpu:raise RuntimeError('explicit GPU isolation required')
    import torch
    from energywam.config import load_config
    from energywam.models.energy_wam import QwenEnergyWAM
    from energywam.data.energy_dataset import collate_energy_batch
    from k_training_probe import fixed_dataset,iteration_trace
    import train as T
    cfg=load_config(a.config);torch.manual_seed(cfg.trainer.seed)
    cfg.trainer.gradient_checkpointing=False
    bundle=T.resolve_training_stats(cfg)
    model=QwenEnergyWAM(cfg,stats_bundle=bundle).to('cuda:0').eval().requires_grad_(False)
    manifest=json.loads(Path(a.manifest).read_text());ds,windows=fixed_dataset(cfg,model,manifest,'train')
    selected=torch.linspace(0,len(windows)-1,min(a.limit,len(windows))).long().tolist()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    records=[];started=time.monotonic()
    for i in selected:
        batch=collate_energy_batch([ds[i]])
        with torch.no_grad():
            prepared=model._prepare_batch(batch);cond=model._condition(batch,prepared)
        gen=torch.Generator(device='cuda:0').manual_seed(2026+windows[i]['window_id']*1009)
        a0,z0=model.stage_runner._init(prepared,torch.device('cuda:0'),generator=gen,cond=cond)
        for direction,alphas in [('action',[.05,.5,5.,50.,200.]),('future',[20.,50.,100.,200.,500.])]:
            for alpha in alphas:
                def energy(x,iteration_progress):
                    action,future=(x,prepared['future_target']) if direction=='action' else (prepared['action_target'],x)
                    return model.core.energy(cond,future,action,future_mask=prepared['future_mask'],
                        action_mask=prepared['action_mask'],offsets=prepared['future_offsets'],iteration_progress=iteration_progress)
                trace=iteration_trace(energy,a0 if direction=='action' else z0,
                    prepared['action_target'] if direction=='action' else prepared['future_target'],alpha,[2])
                records.append(dict(**windows[i],direction=direction,alpha=alpha,initial_mse=trace[0]['prediction_mse'],**trace[1]))
        report=dict(weight_updates=0,model='fresh Qwen4B + iteration modules',seconds=time.monotonic()-started,records=records)
        (out/'report.json').write_text(json.dumps(report,indent=2))
        print('finished sample',i,flush=True)
    groups={}
    for r in records:groups.setdefault((r['direction'],r['alpha']),[]).append(r)
    summary=[]
    for (d,alpha),rs in groups.items():
        summary.append(dict(direction=d,alpha=alpha,samples=len(rs),initial_mse=sum(r['initial_mse'] for r in rs)/len(rs),
            final_mse=sum(r['prediction_mse'] for r in rs)/len(rs),update_rms=sum(r['applied_update_rms'] for r in rs)/len(rs),
            gradient_rms=sum(r['gradient_rms'] for r in rs)/len(rs)))
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
if __name__=='__main__':main()

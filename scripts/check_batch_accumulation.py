"""Real4B, K2: accumulated B8/16/32/64 gradients versus independent micro4 sums.

No optimizer updates. Reuse one CPU reference sum per direction to avoid four
full reference models or overlapping higher-order graphs.
"""
import gc, hashlib, json, math, os, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]

def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']=='0,1'
    os.environ['ENERGYWAM_SAMPLE_ALIGNED_RNG']='1'
    import torch
    from energywam.config import load_config
    from energywam.models.energy_wam import QwenEnergyWAM
    from energywam.training.shared_microbatch import SharedMicrobatchTrainer
    from energywam.data.energy_dataset import collate_energy_batch
    from k_training_probe import fixed_dataset
    import train as T
    root=Path('/cloud/zzc/energyWAM/experiments/batch_grid_micro4_20261105')
    output=root/'accumulation_gate';output.mkdir(exist_ok=True)
    cfg=load_config('configs/experiments/fp32_micro4_dedup_asymmetric_profile.yaml')
    cfg.trainer.fp32_master_weights=False;cfg.trainer.preallocate_optimizer_state=False
    cfg.trainer.max_grad_norm=0.;cfg.trainer.directional_head_diagnostics=False
    cfg.trainer.output_dir=str(output/'runtime')
    torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(42);model=QwenEnergyWAM(cfg,stats_bundle=T.resolve_training_stats(cfg))
    trainer=SharedMicrobatchTrainer(cfg,model)
    trainer.optimizer=torch.optim.SGD(model.parameters(),lr=0.);trainer.scheduler=None
    with torch.no_grad():
        head=model.core.energy_head.net[-1].weight
        head.copy_(torch.randn(head.shape,device=head.device,dtype=head.dtype,generator=torch.Generator(device=head.device).manual_seed(919))*.001)
    dataset,windows=fixed_dataset(cfg,model,json.loads((root/'train_quick_val_manifest.json').read_text()),'train')
    indices=[i*(len(windows)-1)//63 for i in range(64)]
    samples=[dataset[i] for i in indices]
    def digest():
        h=hashlib.sha256()
        for n,p in model.named_parameters():h.update(n.encode());h.update(p.detach().flatten()[:64].cpu().numpy().tobytes())
        return h.hexdigest()
    initial=digest();report=dict(passed=False,weight_updates=0,native_microbatch=4,K=2,precision='highest_fp32',sample_indices=indices,comparisons=[])
    def write():
        (output/'gradient_check.json').write_text(json.dumps(report,indent=2))
    for direction in ['action','future']:
        os.environ['ENERGYWAM_ONLY_DIRECTION']=direction
        reference={};previous=0;loss_sum=0.
        for size in [8,16,32,64]:
            for start in range(previous,size,4):
                trainer.global_step=start//4
                row=trainer.train_step(collate_energy_batch(samples[start:start+4]))
                loss_sum+=row['loss']
                for n,p in model.named_parameters():
                    if p.grad is None:continue
                    value=p.grad.detach().cpu()
                    if n not in reference:reference[n]=value.clone()
                    else:reference[n].add_(value)
                print(f'{direction} independent reference: {start+4}/64 samples',flush=True)
            previous=size;trainer.global_step=0
            row=trainer.train_step(collate_energy_batch(samples[:size]))
            error=norm=dot=norm_new=0.;groups={};finite=True
            actual={n:p.grad for n,p in model.named_parameters() if p.grad is not None}
            assert actual.keys()==reference.keys()
            for n,g in actual.items():
                finite=finite and bool(torch.isfinite(g).all())
                ref=reference[n].to(g.device)*(4/size);delta=g-ref
                d=float(delta.square().sum());v=float(ref.square().sum())
                error+=d;norm+=v;dot+=float((g*ref).sum());norm_new+=float(g.square().sum())
                key='language' if 'language_model' in n else n.split('.')[1]
                pair=groups.setdefault(key,[0.,0.]);pair[0]+=d;pair[1]+=v
            relative=math.sqrt(error/max(norm,1e-30));expected_loss=loss_sum*4/size
            grouped={k:math.sqrt(v[0]/max(v[1],1e-30)) for k,v in groups.items()}
            passed=finite and relative<.02 and abs(row['loss']-expected_loss)<max(1e-6,.01*abs(expected_loss)) and all(grouped[k]<.05 for k,v in groups.items() if math.sqrt(v[1])>1e-8)
            assert digest()==initial,'no-op comparison changed weights'
            item=dict(direction=direction,effective_batch=size,native4_calls_per_direction=size//4,loss=row['loss'],reference_loss=expected_loss,gradient_cosine=dot/max(math.sqrt(norm*norm_new),1e-30),relative_gradient_l2=relative,group_relative_l2=grouped,finite=finite,passed=passed)
            report['comparisons'].append(item);write();print(json.dumps(item),flush=True)
            del actual;model.zero_grad(set_to_none=True);gc.collect()
        del reference;model.zero_grad(set_to_none=True);gc.collect()
    report['passed']=all(x['passed'] for x in report['comparisons']);write()
    if not report['passed']:raise RuntimeError('gradient accumulation gate failed')

if __name__=='__main__':main()

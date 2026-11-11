"""Fixed-window evaluation hook; preserves training RNG and requires_grad flags."""
import json
from pathlib import Path
import time

_FIXED_DATA_CACHE = {}


def fixed_dataset(cfg, model, manifest, split):
    from energywam.data.energy_dataset import EnergyWAMEpisodeDataset,LeRobotEpisodeReader,resolve_dataset_root
    import os,hashlib
    cache_enabled = os.environ.get('ENERGYWAM_FIXED_DATA_CACHE') == '1'
    # Full trajectory manifests stream TRAIN; only small fixed assessments cache.
    if split == 'train' and manifest.get('stream_training', False):
        cache_enabled = False
    cache_key = (id(model), split, hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest())
    if cache_enabled and cache_key in _FIXED_DATA_CACHE:
        return _FIXED_DATA_CACHE[cache_key]
    windows=[w for w in manifest['windows'] if w['split']==split]
    bundle=model.stats_bundle
    reader=LeRobotEpisodeReader(resolve_dataset_root(cfg))
    if reader.dataset_identity != manifest['dataset_identity']:
        raise RuntimeError('fixed-window dataset identity mismatch')
    ds=EnergyWAMEpisodeDataset(cfg,reader=reader,action_stats=bundle.action.to_norm_stats(),
        state_stats=None,dino=model.backbone.dino,require_future=True,future_standardizer=bundle.future,
        episode_indices=list(dict.fromkeys(w['episode_index'] for w in windows)),max_samples=1,
        expected_dataset_identity=bundle.provenance['dataset_identity'])
    ds._index=[(w['episode_index'],w['frame_index']) for w in windows]
    if cache_enabled:
        if not cfg.backbone.freeze_dino:
            raise ValueError('fixed feature caching requires a frozen DINO encoder')
        import torch
        cached = []
        for i in range(len(windows)):
            item = dict(ds[i])
            with torch.no_grad():
                current = model.backbone.dino.encode_images(item['images'][0], device='cpu')
            item['current_dino'] = current.detach().float().cpu().numpy()
            cached.append(item)
        ds = cached
        _FIXED_DATA_CACHE[cache_key] = (ds, windows)
        print(f'[fixed-cache] {split}: {len(windows)} samples, frozen DINO only', flush=True)
    return ds,windows


def iteration_trace(energy, initial, target, alpha, budgets, gradient_normalization="none", gradient_rms_epsilon=1e-6,
                    clip_grad=None, candidate_clamp=(None,None), iteration_aware=True):
    """Each K has its own progress schedule, not a prefix of the longest K."""
    import torch
    import time
    from energywam.energy.solver_utils import precondition_candidate_grad,clip_candidate_grad,apply_candidate_constraints
    x0, target = initial.detach().float(), target.detach().float()
    rows = [dict(k=0, prediction_mse=float((x0-target).square().mean()))]
    for budget in budgets:
        x = x0.clone()
        steps = []
        start = time.monotonic()
        for k in range(budget):
            with torch.enable_grad():
                x = x.detach().requires_grad_(True)
                e = energy(x, iteration_progress=k/budget if iteration_aware else None)
                g = torch.autograd.grad(e.sum(), x)[0]
                if not torch.isfinite(e).all() or not torch.isfinite(g).all():
                    raise FloatingPointError("nonfinite iteration-aware diagnostic")
                before_mse = float((x.detach()-target).square().mean())
                raw_rms = float(g.square().mean().sqrt())
                clipped_fraction = float((g.abs()>clip_grad).float().mean()) if clip_grad is not None else 0.
                g = clip_candidate_grad(g,clip_grad)
                delta = alpha * precondition_candidate_grad(g, mode=gradient_normalization, epsilon=gradient_rms_epsilon)
                x = apply_candidate_constraints(x-delta,None,candidate_clamp).detach()
                steps.append(dict(iteration=k,progress=k/budget if iteration_aware else None,
                    energy_before=float(e.detach().mean()),mse_before=before_mse,
                    mse_after=float((x-target).square().mean()),raw_gradient_rms=raw_rms,
                    clipped_fraction=clipped_fraction,proposed_update_rms=float(delta.detach().square().mean().sqrt())))
        if not torch.isfinite(x).all():
            raise FloatingPointError("nonfinite final candidate")
        rows.append(dict(k=budget, prediction_mse=float((x-target).square().mean()),
                         energy_before_last_update=float(e.detach().mean()),
                         gradient_rms=float(g.square().mean().sqrt()),
                         applied_update_rms=float(delta.detach().square().mean().sqrt()),
                         total_displacement_rms=float((x-x0).square().mean().sqrt()),
                         inner_steps=steps,
                         refinement_seconds=time.monotonic()-start))
    return rows


def dataset_baselines(cfg, model, manifest_path, output):
    """Constant predictors are fitted on train targets only, never validation."""
    import numpy as np
    manifest=json.loads(Path(manifest_path).read_text())
    has_train=any(w['split']=='train' for w in manifest['windows'])
    train,_=fixed_dataset(cfg,model,manifest,'train') if has_train else ([],[])
    means={key:np.stack([x[key] for x in train]).mean(axis=0,dtype=np.float64)
           for key in ['action_target','future_target']} if has_train else {}
    result={}
    for split in ['train','val']:
        if not any(w['split']==split for w in manifest['windows']):continue
        ds,windows=fixed_dataset(cfg,model,manifest,split)
        result[split]={}
        for direction in ['action','future']:
            key=direction+'_target'
            targets=np.stack([x[key] for x in ds]).astype(np.float64)
            result[split][direction]=dict(zero_mse=float(np.square(targets).mean()),
                samples=len(windows))
            if key in means:result[split][direction]['training_mean_mse']=float(np.square(targets-means[key]).mean())
            if direction=='future' and all('current_dino' in x for x in ds):
                current=np.stack([x['current_dino'] for x in ds])
                normalized=model.stats_bundle.future.normalize(current)
                persistence=np.broadcast_to(normalized[:,None],targets.shape)
                result[split][direction]['current_visual_persistence_mse']=float(np.square(targets-persistence).mean())
    Path(output,'dataset_baselines.json').write_text(json.dumps(result,indent=2))
    return result


def evaluate(trainer, manifest_path, output, rank, evaluation_seed=2026):
    import torch
    import os
    from energywam.training.checkpoint import _rng_state,_set_rng_state
    from energywam.data.energy_dataset import collate_energy_batch
    from energywam.energy.inner_diagnostic import trace_candidate
    model=trainer.model
    rng=_rng_state(); flags=[p.requires_grad for p in model.parameters()]; was_training=model.training
    trainer.optimizer.zero_grad(set_to_none=True)
    direction='action' if rank==0 else 'future'
    manifest=json.loads(Path(manifest_path).read_text())
    max_k=int(os.environ.get('ENERGYWAM_PROBE_MAX_K','8'))
    splits=os.environ.get('ENERGYWAM_PROBE_SPLITS','train,val').split(',')
    if max_k<2 or any(x not in ['train','val'] for x in splits):raise ValueError('invalid probe protocol')
    rows=[];start=time.monotonic()
    try:
        model.requires_grad_(False);model.eval()
        for split in splits:
            if 'windows' in manifest and not any(w['split']==split for w in manifest['windows']):continue
            ds,windows=fixed_dataset(trainer.cfg,model,manifest,split)
            for i,w in enumerate(windows):
                batch=collate_energy_batch([ds[i]])
                with torch.no_grad():
                    prepared=model._prepare_batch(batch);cond=model._condition(batch,prepared)
                assert prepared['action_mask'].all() and prepared['future_mask'].all()
                if trainer.cfg.trainer.fixed_candidate_initialization:
                    from energywam.training.candidate_seeds import fixed_candidate_seed
                    candidate_seed=fixed_candidate_seed(trainer.cfg.trainer.seed,rank,w['episode_index'],w['frame_index'])
                else:candidate_seed=int(evaluation_seed)+w['window_id']*1009
                generator=torch.Generator(device='cuda:0').manual_seed(candidate_seed)
                a0,z0=model.stage_runner._init(prepared,torch.device('cuda:0'),generator=generator,cond=cond)
                astar,zstar=prepared['action_target'],prepared['future_target']
                def energy(candidate, iteration_progress=None):
                    z,a=(zstar,candidate) if rank==0 else (candidate,astar)
                    return model.core.energy(cond,z,a,future_mask=prepared['future_mask'],action_mask=prepared['action_mask'],offsets=prepared['future_offsets'],iteration_progress=iteration_progress)
                alpha = model.core.effective_step_sizes()[rank]
                alpha = float(torch.as_tensor(alpha).detach())
                solver=trainer.cfg.solver
                clip=solver.clip_action_grad if rank==0 else solver.clip_future_grad
                clamp=(solver.action_clamp_min,solver.action_clamp_max) if rank==0 else (solver.future_clamp_min,solver.future_clamp_max)
                if trainer.cfg.framework.iteration_conditioning or solver.candidate_gradient_normalization != "none" or clip is not None or any(v is not None for v in clamp):
                    trace = iteration_trace(energy, a0 if rank==0 else z0,
                        astar if rank==0 else zstar, alpha, [k for k in [2,4,8] if k<=max_k],
                        gradient_normalization=trainer.cfg.solver.candidate_gradient_normalization,
                        gradient_rms_epsilon=trainer.cfg.solver.gradient_rms_epsilon,
                        clip_grad=clip,candidate_clamp=clamp,iteration_aware=trainer.cfg.framework.iteration_conditioning)
                else:
                    trace,_=trace_candidate(energy,a0 if rank==0 else z0,astar if rank==0 else zstar,alpha,max_k)
                rows.extend(dict(r,**w,direction=direction,update=trainer.global_step,train_k=trainer.cfg.solver.train_inner_steps)
                            for r in trace if r['k'] in [0,2,4,8])
                del batch,prepared,cond,a0,z0,astar,zstar,trace
        path=Path(output)/f'eval_{trainer.global_step:04d}_{direction}.json'
        temporary=path.with_suffix('.tmp')
        temporary.write_text(json.dumps({'rows':rows,'seconds':time.monotonic()-start,'weight_updates_during_eval':0,'max_k':max_k,'splits':splits,
            'candidate_seed_protocol':'fixed_episode_frame_per_direction' if trainer.cfg.trainer.fixed_candidate_initialization else f'independent_fixed_evaluation_seed{evaluation_seed}',
            'evaluation_seed':int(evaluation_seed)},indent=2))
        temporary.replace(path)
        print(f'[k-probe] evaluated update={trainer.global_step} direction={direction}',flush=True)
    finally:
        for p,flag in zip(model.parameters(),flags):p.requires_grad_(flag)
        model.train(was_training);_set_rng_state(rng)

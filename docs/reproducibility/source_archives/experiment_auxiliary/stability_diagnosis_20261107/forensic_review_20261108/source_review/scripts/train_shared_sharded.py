#!/usr/bin/env python
"""One model over an explicitly selected GPU pair; shared BI optimizer step."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--steps',type=int,required=True)
    p.add_argument('--resume');p.add_argument('--fixed-window-manifest',required=True);p.add_argument('--probe-eval-interval',type=int,default=20)
    p.add_argument('--physical-gpus',default='2,3',choices=['2,3','0,1'],help='Explicit authorized physical GPU pair')
    p.add_argument('--profile',action='store_true',help='Skip evaluation/checkpoint writes for bounded throughput probes')
    p.add_argument('--evaluate-only',action='store_true',help='Read-only fixed-window evaluation, no checkpoint rewrite')
    p.add_argument('--keep-best',action='store_true',help='Keep full latest and best by predeclared weighted conditional validation MSE')
    p.add_argument('--sample-eval-interval',type=int,help='Validation cadence in sample visits, independent of effective batch')
    p.add_argument('--train-monitor-manifest',help='Small fixed TRAIN monitor, evaluated separately from VAL')
    p.add_argument('--train-monitor-sample-interval',type=int,default=256)
    args=p.parse_args()
    if args.keep_best and (args.profile or args.evaluate_only):raise ValueError('keep-best is only for real training, never read-only/profile mode')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=args.physical_gpus:raise RuntimeError('visible GPUs must match explicitly selected physical pair')
    physical_gpus=[int(x) for x in args.physical_gpus.split(',')]
    import torch
    from torch.utils.data import DataLoader
    from energywam.config import load_config
    from energywam.models.energy_wam import QwenEnergyWAM
    from energywam.training.shared_microbatch import SharedMicrobatchTrainer
    from energywam.training.run_state import FixedSampleBudgetSampler,save_latest,checkpoint_due
    from energywam.data.energy_dataset import collate_energy_batch
    from k_training_probe import fixed_dataset,evaluate,dataset_baselines
    import train as T
    cfg=load_config(args.config)
    if cfg.backbone.dtype in ['float32','fp32']:
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32=False
    if cfg.trainer.language_model_devices!=[0,1]:raise RuntimeError('expected two-card layer placement')
    cfg.trainer.max_train_steps=args.steps
    output=Path(cfg.trainer.output_dir);output.mkdir(parents=True,exist_ok=True)
    if args.sample_eval_interval and (args.sample_eval_interval<=0 or args.sample_eval_interval%cfg.trainer.per_device_batch_size):raise ValueError('validation sample cadence must be a positive multiple of batch')
    if args.train_monitor_manifest and (args.train_monitor_sample_interval<=0 or args.train_monitor_sample_interval%cfg.trainer.per_device_batch_size):raise ValueError('train monitor cadence must be a positive multiple of batch')
    (output/'execution_options.json').write_text(json.dumps(vars(args),indent=2))
    if not args.resume:
        import zipfile
        with zipfile.ZipFile(output/'source_initial.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
            files=list((ROOT/'energywam').rglob('*.py'))
            files += [ROOT/'scripts'/name for name in ['train_shared_sharded.py','k_training_probe.py','train.py']]
            for source in files:archive.write(source,str(source.relative_to(ROOT)))
            archive.write(args.config,'experiment_config.yaml')
            archive.write(args.fixed_window_manifest,'sample_manifest.json')
    torch.manual_seed(cfg.trainer.seed)
    bundle=T.resolve_training_stats(cfg,checkpoint_path=args.resume)
    model=QwenEnergyWAM(cfg,stats_bundle=bundle)
    trainer=SharedMicrobatchTrainer(cfg,model)
    if args.resume:
        trainer.load(args.resume)
        if not args.evaluate_only:
            from energywam.training.run_state import reconcile_resume_outputs
            reconciled=reconcile_resume_outputs(output,trainer.global_step)
            if reconciled['changed']:
                print(f'[shared-sharded] archived uncheckpointed records: {reconciled["archive_dir"]}',flush=True)
    trainer.export_config()
    digest=hashlib.sha256();common_digest=hashlib.sha256();placement={}
    packing_inputs={'core.future_projector.0.weight','core.current_dino_projector.0.weight'}
    for name,parameter in model.named_parameters():
        sample=parameter.detach().reshape(-1)[:64].float().cpu().numpy().tobytes()
        digest.update(name.encode());digest.update(sample)
        if name not in packing_inputs:common_digest.update(name.encode());common_digest.update(sample)
        placement[str(parameter.device)]=placement.get(str(parameter.device),0)+parameter.numel()
    if set(placement)!={'cuda:0','cuda:1'}:raise RuntimeError('model not distributed over exactly two visible GPUs')
    (output/f'initialization_step{trainer.global_step}.json').write_text(json.dumps({'sampled_parameter_sha256':digest.hexdigest(),
        'sample_per_parameter':64,'full_parameter_hash':False,'common_parameter_sampled_sha256':common_digest.hexdigest(),
        'common_hash_excluded_parameters':sorted(packing_inputs),'global_step':trainer.global_step,'stats_fingerprint':bundle.fingerprint,
        'model_instances':1,'parameter_numel_by_device':placement,'physical_gpus':physical_gpus},indent=2))
    manifest=json.loads(Path(args.fixed_window_manifest).read_text())
    train_ids={w['episode_index'] for w in manifest['windows'] if w['split']=='train'}
    val_ids={w['episode_index'] for w in manifest['windows'] if w['split']=='val'}
    if not train_ids.isdisjoint(val_ids):raise ValueError('train/val episode leakage')
    dataset,_=fixed_dataset(cfg,model,manifest,'train')
    sampler=FixedSampleBudgetSampler(len(dataset),cfg.trainer.seed,consumed=trainer.global_step*cfg.trainer.per_device_batch_size,total=max(args.steps,trainer.global_step)*cfg.trainer.per_device_batch_size)
    loader=DataLoader(dataset,batch_size=cfg.trainer.per_device_batch_size,sampler=sampler,drop_last=True,num_workers=0,
        generator=torch.Generator().manual_seed(cfg.trainer.seed),collate_fn=collate_energy_batch)
    iterator=iter(loader)
    records=json.loads((output/'speed_report.json').read_text())['steps'] if args.resume and (output/'speed_report.json').exists() else []
    last_saved=None;last_checkpoint_time=time.monotonic()
    if args.resume and (output/'ckpt_latest.pt.meta.json').exists():
        saved=json.loads((output/'ckpt_latest.pt.meta.json').read_text())
        if saved['meta']['global_step']==trainer.global_step:last_saved=trainer.global_step
    def checkpoint():
        nonlocal last_saved,last_checkpoint_time
        started=time.monotonic();path=save_latest(trainer,output)
        event=dict(global_step=trainer.global_step,save_seconds=time.monotonic()-started,bytes=path.stat().st_size)
        with (output/'checkpoint_history.jsonl').open('a') as stream:stream.write(json.dumps(event)+'\n')
        last_saved=trainer.global_step;last_checkpoint_time=time.monotonic()
    def fixed_assessment(manifest,target,splits):
        target=Path(target);target.mkdir(parents=True,exist_ok=True)
        old=os.environ.get('ENERGYWAM_PROBE_SPLITS')
        os.environ['ENERGYWAM_PROBE_SPLITS']=splits
        try:
            for rank,direction in enumerate(['action','future']):
                if not (target/f'eval_{trainer.global_step:04d}_{direction}.json').exists():evaluate(trainer,manifest,target,rank)
        finally:
            if old is None:os.environ.pop('ENERGYWAM_PROBE_SPLITS',None)
            else:os.environ['ENERGYWAM_PROBE_SPLITS']=old
    def assess():
        splits='val' if args.train_monitor_manifest else os.environ.get('ENERGYWAM_PROBE_SPLITS','train,val')
        fixed_assessment(args.fixed_window_manifest,output,splits)
        if args.keep_best:
            from energywam.training.best_checkpoint import validation_score_from_reports,promote_latest_to_best
            score,components,criterion=validation_score_from_reports(output,trainer.global_step,lambda_action=cfg.loss.lambda_action,lambda_future=cfg.loss.lambda_future,evaluation_k=cfg.solver.eval_inner_steps)
            selection=output/'best_selection.json';old=json.loads(selection.read_text()) if selection.exists() else None
            if old and old['criterion']!=criterion:raise ValueError('best criterion changed on resume')
            if old is None or score<old['score']:
                if last_saved!=trainer.global_step:checkpoint()
                promote_latest_to_best(output,global_step=trainer.global_step,score=score,metrics=components,criterion=criterion)
    def monitor_train():
        if args.train_monitor_manifest:fixed_assessment(args.train_monitor_manifest,output/'train_monitor','train')
    if not args.profile and os.environ.get('ENERGYWAM_FIXED_DATA_CACHE')=='1' and not manifest.get('stream_training',False):dataset_baselines(cfg,model,args.fixed_window_manifest,output)
    if not args.profile:
        monitor_train();assess()
    if args.evaluate_only:return
    while trainer.global_step<args.steps:
        begin=time.monotonic()
        try:batch=next(iterator)
        except StopIteration:iterator=iter(loader);batch=next(iterator)
        data=time.monotonic()-begin
        print(f'[shared-sharded] begin update={trainer.global_step}',flush=True)
        row=trainer.train_step(batch)
        for dev in range(2):torch.cuda.synchronize(dev)
        elapsed=time.monotonic()-begin
        row.update(end_to_end_seconds=elapsed,data_seconds=data,end_to_end_samples_per_second=cfg.trainer.per_device_batch_size/elapsed,global_step_completed=trainer.global_step)
        import resource
        row['process_rss_peak_gib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20
        for dev in range(2):row[f'gpu{dev}_peak_allocated_gib']=torch.cuda.max_memory_allocated(dev)/2**30
        row['trim_between_directions']=int(os.environ.get('ENERGYWAM_TRIM_BETWEEN_DIRECTIONS','1')=='1')
        row['sample_aligned_rng']=int(os.environ.get('ENERGYWAM_SAMPLE_ALIGNED_RNG','0')=='1')
        with (output/'timing_history.jsonl').open('a') as timing_stream:timing_stream.write(json.dumps(row)+'\n')
        records.append(row);records=records[-100:]
        (output/'speed_report.json').write_text(json.dumps({'mode':'one_model_layer_sharded','model_instances':1,'physical_gpus':physical_gpus,
            'global_batch':cfg.trainer.per_device_batch_size,'completed_steps':trainer.global_step,'steps':records},indent=2))
        print(f'[shared-sharded] completed={trainer.global_step} seconds={elapsed:.3f}',flush=True)
        if not args.profile and checkpoint_due(trainer.global_step,args.steps,time.monotonic()-last_checkpoint_time,
                cfg.trainer.save_interval,cfg.trainer.checkpoint_interval_seconds):
            checkpoint();last_saved=trainer.global_step;last_checkpoint_time=time.monotonic()
        visits=trainer.global_step*cfg.trainer.per_device_batch_size
        if not args.profile and args.train_monitor_manifest and (visits%args.train_monitor_sample_interval==0 or trainer.global_step==args.steps):monitor_train()
        eval_due=(visits%args.sample_eval_interval==0) if args.sample_eval_interval else (trainer.global_step%args.probe_eval_interval==0)
        if not args.profile and (eval_due or trainer.global_step==args.steps):assess()
    if not args.profile and last_saved!=trainer.global_step:checkpoint()
    if args.profile:
        torch.save({n:p.detach().reshape(-1)[:64].float().cpu().clone() for n,p in model.named_parameters()},output/'final_parameter_samples.pt')
    (output/'final_report.json').write_text(json.dumps({'global_step':trainer.global_step,'model_instances':1,'global_batch':cfg.trainer.per_device_batch_size,'profile_only':args.profile,'checkpoint_policy':'none' if args.profile else ('latest_and_best' if args.keep_best else 'rolling_latest'),'last_loss':records[-1] if records else None},indent=2))

if __name__=='__main__':main()

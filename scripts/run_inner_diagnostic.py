#!/usr/bin/env python
"""Two conditional read-only traces on physical GPUs 2/3; never trains weights."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)


def select_windows(cfg,experiment):
    from energywam.data.energy_dataset import LeRobotEpisodeReader,resolve_dataset_root
    reader=LeRobotEpisodeReader(resolve_dataset_root(cfg))
    rng=random.Random(experiment['sample_selection_seed']);windows=[]
    split_indices={s:set(json.load(open(ROOT/f'manifests/{s}_episodes.json'))['episode_indices']) for s in ['train','val','test']}
    assert not(split_indices['train']&split_indices['val'])
    for task in experiment['tasks']:
        for split in ['train','val']:
            pool=[ep for ep in reader.episodes if ep['episode_index'] in split_indices[split]
                  and ep.get('source_dataset','').split('/')[-2]==task and ep['length']>=17]
            selected=rng.sample(pool,2)
            for ep in selected:
                windows.append({'window_id':len(windows),'split':split,'task':task,
                                'episode_index':ep['episode_index'],'frame_index':rng.randint(0,ep['length']-17),
                                'source_dataset':ep['source_dataset']})
    assert len(windows)==40 and len({w['episode_index'] for w in windows})==40
    assert not({w['episode_index'] for w in windows}&split_indices['test'])
    return {'dataset_identity':reader.dataset_identity,'windows':windows,'test_used':False,
            'selection_seed':experiment['sample_selection_seed']}


def worker(args):
    import torch
    from energywam.config import load_config
    from energywam.models.energy_wam import QwenEnergyWAM
    from energywam.training.checkpoint import load_checkpoint,bundle_from_payload,restore
    from energywam.data.energy_dataset import EnergyWAMEpisodeDataset,LeRobotEpisodeReader,resolve_dataset_root,collate_energy_batch
    from energywam.energy.inner_diagnostic import trace_candidate
    exp=json.load(open(args.experiment));out=Path(args.output)
    cfg=load_config(ROOT/exp['base_config']);cfg.trainer.device='cuda:0'
    cfg.trainer.language_model_devices=[]
    payload=load_checkpoint(args.checkpoint);bundle=bundle_from_payload(payload)
    model=QwenEnergyWAM(cfg,stats_bundle=bundle).to('cuda:0')
    restore(payload,model=model,mode='resume',restore_rng=False)
    step=payload['meta']['global_step'];del payload
    model.requires_grad_(False);model.eval()
    versions={name:p._version for name,p in model.named_parameters()}
    manifest=json.load(open(out/'sample_manifest.json'));windows=manifest['windows']
    reader=LeRobotEpisodeReader(resolve_dataset_root(cfg))
    dataset=EnergyWAMEpisodeDataset(cfg,reader=reader,action_stats=bundle.action.to_norm_stats(),
                state_stats=None,dino=model.backbone.dino,require_future=True,future_standardizer=bundle.future,
                episode_indices=[w['episode_index'] for w in windows],max_samples=1,
                expected_dataset_identity=bundle.provenance['dataset_identity'])
    dataset._index=[(w['episode_index'],w['frame_index']) for w in windows]
    path=out/f'trace_{args.direction}.jsonl';completed=set()
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                r=json.loads(line)
                if r['k']==exp['max_k']:completed.add((r['window_id'],r['seed']))
            except (ValueError,KeyError):pass
    begin=time.monotonic();pilot_time=None
    pilot_ids=[0,4,20,24]
    ordered=[windows[i] for i in pilot_ids]+[w for w in windows if w['window_id'] not in pilot_ids]
    for index,w in enumerate(ordered):
        if all((w['window_id'],s) in completed for s in exp['candidate_seeds']):continue
        if time.monotonic()-begin>exp['time_budget_seconds']:raise TimeoutError('Diagnostic exceeded one-hour compute budget')
        torch.cuda.reset_peak_memory_stats()
        window_begin=time.monotonic()
        batch=collate_energy_batch([dataset[w['window_id']]])
        with torch.no_grad():
            prepared=model._prepare_batch(batch);cond=model._condition(batch,prepared)
        assert bool(prepared['action_mask'].all()) and bool(prepared['future_mask'].all())
        astar,zstar=prepared['action_target'],prepared['future_target']
        for seed in exp['candidate_seeds']:
            if (w['window_id'],seed) in completed:continue
            actual_seed=seed+w['window_id']*1009
            generator=torch.Generator(device='cuda:0').manual_seed(actual_seed)
            a0,z0=model.stage_runner._init(prepared,torch.device('cuda:0'),generator=generator,cond=cond)
            if args.direction=='action':
                fn=lambda candidate:model.core.energy(cond,zstar,candidate,future_mask=prepared['future_mask'],action_mask=prepared['action_mask'],offsets=prepared['future_offsets'])
                initial,target,alpha=a0,astar,exp['alpha_action']
            else:
                fn=lambda candidate:model.core.energy(cond,candidate,astar,future_mask=prepared['future_mask'],action_mask=prepared['action_mask'],offsets=prepared['future_offsets'])
                initial,target,alpha=z0,zstar,exp['alpha_future']
            rows,_=trace_candidate(fn,initial,target,alpha,exp['max_k'])
            with path.open('a') as stream:
                for row in rows:stream.write(json.dumps({**w,'seed':seed,'actual_seed':actual_seed,'direction':args.direction,**row})+'\n')
            del a0,z0,rows
        torch.cuda.synchronize()
        elapsed=time.monotonic()-begin
        progress={'direction':args.direction,'completed_windows':index+1,'total_windows':len(windows),
                  'elapsed_seconds':elapsed,'last_window_seconds':time.monotonic()-window_begin,
                  'gpu_peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30}
        if index+1==exp['pilot_windows']:
            pilot_time=elapsed;progress['projected_total_seconds']=elapsed*len(windows)/(index+1)
        write_json(out/f'progress_{args.direction}.json',progress)
        print(json.dumps(progress),flush=True)
        del prepared,cond,batch,astar,zstar
    assert all(not p.requires_grad and p.grad is None and p._version==versions[n] for n,p in model.named_parameters())
    write_json(out/f'complete_{args.direction}.json',{'checkpoint_step':step,'stats_fingerprint':bundle.fingerprint,
               'parameters_unchanged':True,'parameter_gradients_accumulated':False,'weight_updates':0,
               'windows':len(windows),'candidate_seeds':exp['candidate_seeds'],'elapsed_seconds':time.monotonic()-begin})


def summarize(out,experiment):
    rows={}
    for direction in ['action','future']:
        for line in (out/f'trace_{direction}.jsonl').read_text().splitlines():
            r=json.loads(line);rows[(r['direction'],r['window_id'],r['seed'],r['k'])]=r
    rows=list(rows.values())
    expected=len(json.load(open(out/'sample_manifest.json'))['windows'])*len(experiment['candidate_seeds'])*2*(experiment['max_k']+1)
    if len(rows)!=expected:raise RuntimeError(f'incomplete traces: {len(rows)} / {expected}')
    aggregates=[]
    for split in ['train','val']:
        for direction in ['action','future']:
            base={(r['window_id'],r['seed']):r for r in rows if r['split']==split and r['direction']==direction and r['k']==0}
            for k in experiment['record_k']:
                subset=[r for r in rows if r['split']==split and r['direction']==direction and r['k']==k]
                deltas=[r['prediction_mse']-base[(r['window_id'],r['seed'])]['prediction_mse'] for r in subset]
                aggregates.append({'split':split,'direction':direction,'k':k,'count':len(subset),
                    'mean_mse':statistics.mean(r['prediction_mse'] for r in subset),
                    'mean_paired_mse_delta':statistics.mean(deltas),'median_paired_mse_delta':statistics.median(deltas),
                    'improved_fraction':sum(d<0 for d in deltas)/len(deltas),
                    **{'mean_'+key:statistics.mean(r[key] for r in subset) for key in ['energy','gradient_rms','applied_update_rms','relative_total_displacement','refinement_seconds']}})
    completions=[json.load(open(out/f'complete_{d}.json')) for d in ['action','future']]
    per_task=[]
    for split in ['train','val']:
        for direction in ['action','future']:
            for task in sorted({r['task'] for r in rows}):
                subset=[r for r in rows if r['split']==split and r['direction']==direction and r['task']==task]
                initial={(r['window_id'],r['seed']):r['prediction_mse'] for r in subset if r['k']==0}
                for k in experiment['record_k']:
                    group=[r for r in subset if r['k']==k]
                    changes=[r['prediction_mse']-initial[(r['window_id'],r['seed'])] for r in group]
                    per_task.append({'split':split,'direction':direction,'task':task,'k':k,
                                     'mean_mse':statistics.mean(r['prediction_mse'] for r in group),
                                     'mean_paired_delta':statistics.mean(changes),
                                     'improved_fraction':sum(v<0 for v in changes)/len(changes),'count':len(group)})
    report={'per_task':per_task,'experiment':'inner-refinement diagnosis, not training or task success evaluation',
            'weight_updates':0,'aggregates':aggregates,'completions':completions,'row_count':len(rows)}
    write_json(out/'report.json',report)
    with (out/'traces.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main(args):
    from energywam.config import load_config
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    exp=json.load(open(args.experiment));cfg=load_config(ROOT/exp['base_config'])
    if not(out/'sample_manifest.json').exists():write_json(out/'sample_manifest.json',select_windows(cfg,exp))
    write_json(out/'experiment_config.json',exp)
    if args.prepare_only:return
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='2,3':raise RuntimeError('Only physical GPUs 2,3 may be used')
    children=[];streams=[]
    try:
        for direction,gpu in [('action','2'),('future','3')]:
            stream=(out/f'worker_{direction}.log').open('a');streams.append(stream)
            command=[sys.executable,'-u',__file__,'--worker','--direction',direction,'--experiment',args.experiment,
                     '--output',str(out),'--checkpoint',args.checkpoint]
            children.append(subprocess.Popen(command,env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu),stdout=stream,stderr=subprocess.STDOUT))
        while any(p.poll() is None for p in children):
            if any(p.poll() not in (None,0) for p in children):raise RuntimeError('Diagnostic worker failed; see worker logs')
            time.sleep(2)
        if any(p.returncode for p in children):raise RuntimeError('Diagnostic worker failed')
        summarize(out,exp)
    finally:
        for p in children:
            if p.poll() is None:p.terminate()
        for p in children:
            try:p.wait(timeout=15)
            except subprocess.TimeoutExpired:p.kill();p.wait()
        for stream in streams:stream.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--experiment',default='configs/experiments/inner_diag_20261102.json')
    parser.add_argument('--output',default='/cloud/zzc/energyWAM/experiments/inner_diag_20261102')
    parser.add_argument('--checkpoint');parser.add_argument('--worker',action='store_true')
    parser.add_argument('--direction',choices=['action','future']);parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    worker(args) if args.worker else main(args)

#!/usr/bin/env python3
"""Prepare a portable, monitored Stage1 job without loading models or using a GPU."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import sys
import yaml
ROOT=Path(__file__).resolve().parents[1]
EXPERIMENTS={
 'three-task':('three_task_b16_micro4_seed42.yaml','three_task_full_20261106',2798,4096,1024),
 'tiny-weight1':('tiny16_futureweight1_b16_micro4_seed42.yaml','tiny_future_weight_20261106',50,160,160),
 'tiny-weight9':('tiny16_futureweight9_b16_micro4_seed42.yaml','tiny_future_weight_20261106',50,160,160),
}
def validate_dataset(dataset,manifest,fingerprints):
    """Validate numbering and every selected window before any GPU allocation."""
    for name,digest in fingerprints['source_meta_sha256'].items():
        path=dataset/'meta'/name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise ValueError(f'Dataset metadata differs from the frozen view: {path}. Rebuild the same50-task view; do not reuse global episode IDs from another ordering.')
    info=json.loads((dataset/'meta/info.json').read_text())
    if dataset.name+'/'+info['robot_type']!=manifest['dataset_identity']:raise ValueError('dataset identity mismatch')
    episodes={int(e['episode_index']):e for e in map(json.loads,(dataset/'meta/episodes.jsonl').read_text().splitlines())}
    split_ids={s:set() for s in ['train','val']};seen=set()
    for w in manifest['windows']:
        ep=episodes.get(w['episode_index']);key=(w['split'],w['episode_index'],w['frame_index'])
        if key in seen:raise ValueError('duplicate selected window')
        seen.add(key)
        if ep is None or ep.get('source_dataset')!=w['source_dataset'] or not 0<=w['frame_index']<=ep['length']-17:raise ValueError(f'Window invalid for episode{w["episode_index"]}')
        split_ids[w['split']].add(w['episode_index'])
    if split_ids['train']&split_ids['val']:raise ValueError('TRAIN/VAL episode leakage')
    # Check every selected trajectory's parquet and actual current camera video.
    for ep in split_ids['train']|split_ids['val']:
        for path in [dataset/f'data/chunk-000/episode_{ep:06d}.parquet',dataset/f'videos/chunk-000/observation.images.robot0_agentview_left/episode_{ep:06d}.mp4']:
            if not path.is_file():raise FileNotFoundError(f'Missing/broken dataset link: {path}')
    return {s:len(v) for s,v in split_ids.items()}
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment',choices=EXPERIMENTS,required=True)
    p.add_argument('--data-root',type=Path,required=True,help='Parent ofrobocasa365_target_50tasks, not the rawtarget directory')
    p.add_argument('--qwen',type=Path,required=True,help='Complete local Qwen3-VL-4B-Instruct directory')
    p.add_argument('--dino',type=Path,required=True,help='Complete local DINOv3 ViT-S/16 directory')
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--physical-gpus',choices=['0,1','2,3'],default='0,1')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--profile-steps',type=int,help='Bounded no-full-checkpoint GPU smoke; use a separate output directory')
    p.add_argument('--skip-path-check',action='store_true',help='Prepare only; does NOT certify local assets. Prefer normal checks before launch.')
    args=p.parse_args();template,name,steps,eval_interval,monitor_interval=EXPERIMENTS[args.experiment]
    cfg=yaml.safe_load((ROOT/'configs/experiments'/template).read_text());manifest=ROOT/'manifests/experiments'/name/'train_val_manifest.json';monitor=manifest.with_name('train_monitor_manifest.json')
    if args.profile_steps is not None and args.profile_steps<1:raise ValueError('positive profile steps required')
    if not args.skip_path_check:
        for directory in [args.qwen,args.dino]:
            if not (directory/'config.json').is_file() or not any(x.is_file() for x in directory.glob('*.safetensors')):raise FileNotFoundError(f'Missing modelconfig or safetensor weights: {directory}')
        counts=validate_dataset(args.data_root.resolve()/cfg['data']['data_mix'],json.loads(manifest.read_text()),json.loads((ROOT/'docs/reproducibility/dataset_layout_fingerprints.json').read_text()))
    else:counts=None
    out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    cfg['run_id']=out.name;cfg['backbone'].update(base_vlm=str(args.qwen.resolve()),dino_path=str(args.dino.resolve()))
    cfg['data'].update(data_root_dir=str(args.data_root.resolve()),norm_stats_bundle_path=str(ROOT/'stats/trainmix_150k_paper_standardized_v1.json'))
    cfg['trainer'].update(seed=args.seed,output_dir=str(out),max_train_steps=args.profile_steps or steps)
    target=out/'local_config.yaml';text=yaml.safe_dump(cfg,sort_keys=False)
    if target.exists() and target.read_text()!=text:raise ValueError('Refusing to overwrite a different local configuration; use a fresh output directory')
    target.write_text(text)
    job=dict(name=cfg['run_id'],config=str(target),physical_gpus=args.physical_gpus,steps=args.profile_steps or steps,manifest=str(manifest),train_monitor_manifest=str(monitor),sample_eval_interval=eval_interval,train_monitor_sample_interval=monitor_interval,keep_best=args.profile_steps is None,profile=args.profile_steps is not None,monitor_resources=True,effective_batch=16,guard_min_sample_visits=160 if name.startswith('tiny') else 256,max_fixed_mse_ratio=10,max_process_uss_gib=140,snapshot_bytes_estimate=24568519249)
    jobs=out/'jobs.json';jobs.write_text(json.dumps([job],indent=2))
    command=shlex.join([sys.executable,'-u',str(ROOT/'scripts/run_iteration_search.py'),'--jobs',str(jobs),'--status',str(out/'queue_status.json')])
    report={'prepared_only':True,'path_checks_passed':not args.skip_path_check,'checked_episode_counts':counts,'experiment':args.experiment,'steps':job['steps'],'command':command,'notes':['Run from repository root.','The queue chooses only an idle explicitly selected GPU pair.','Whenlatest exists, this queue resumes it. Numerical failures remain stopped for inspection.','Profile skips validation and full checkpoint writes; it cannot validate saving/resume or generalization.']}
    (out/'launch_plan.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2));print('\nLaunch:\n'+command)
if __name__=='__main__':main()

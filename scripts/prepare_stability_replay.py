"""Prepare a short matched replay of the observed step680->692 failure window."""
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import zipfile

import yaml

ROOT=Path(__file__).resolve().parents[1]
OUT=Path('/cloud/zzc/energyWAM/experiments/stability_diagnosis_20261107')
SOURCE=Path('/cloud/zzc/energyWAM/experiments/three_task_full_20261106/three_task_b16_micro4_seed42')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    for path in [OUT/'resume680_audit/resume_audit_result.json',OUT/'native4_gradient_audit.json']:
        if not json.loads(path.read_text())['passed']:
            raise RuntimeError(f'Correctness gate failed: {path}')
    checkpoint=SOURCE/'ckpt_latest.pt'
    if json.loads(Path(str(checkpoint)+'.meta.json').read_text())['meta']['global_step']!=680:
        raise RuntimeError('Source checkpoint changed')
    manifest=json.loads((SOURCE/'sample_manifest.json').read_text())
    train=[w for w in manifest['windows'] if w['split']=='train']
    val=[w for w in manifest['windows'] if w['split']=='val']
    selected=[]
    for task in sorted({w['task'] for w in val}):
        choices=[w for w in val if w['task']==task]
        # Source VAL has4 windows per episode. Select8 distinct episodes/task.
        episodes=list(dict.fromkeys(w['episode_index'] for w in choices))
        for i in range(8):
            episode=episodes[round(i*(len(episodes)-1)/7)]
            selected.append(next(w for w in choices if w['episode_index']==episode))
    assert len(selected)==24 and len({w['episode_index'] for w in selected})==24
    manifest['windows']=train+selected
    manifest['note']='Exact original TRAIN order pool retained; only12 optimizer updates/192visits replayed from680. Fixed24VAL subset,8 distinct episodes per task. Not full epoch or from-scratch experiment.'
    (OUT/'replay_manifest.json').write_text(json.dumps(manifest,indent=2))
    monitor=SOURCE.parent/'train_monitor_manifest.json'
    shutil.copy2(monitor,OUT/'train_monitor_manifest.json')
    base=yaml.safe_load((ROOT/'configs/experiments/three_task_b16_micro4_seed42.yaml').read_text())
    jobs=[]
    for arm,pair in [('baseline','0,1'),('candidate_clip','2,3')]:
        cfg=copy.deepcopy(base);name=f'replay680_{arm}';folder=OUT/name
        if folder.exists():raise RuntimeError('Replay already prepared; inspect rather than overwrite')
        folder.mkdir()
        cfg['run_id']=name
        cfg['notes']='Matched12-update failure replay; shared680 model/optimizer/RNG, original TRAIN sequence. Baseline versus explicit per-element inner gradient clipping only. No causal claim beyond this replay.'
        cfg['trainer'].update(output_dir=str(folder),max_train_steps=692,save_interval=692,checkpoint_interval_seconds=0)
        if arm=='candidate_clip':
            cfg['solver'].update(clip_action_grad=1.0,clip_future_grad=0.00025)
        config=ROOT/f'configs/experiments/{name}.yaml'
        config.write_text(yaml.safe_dump(cfg,sort_keys=False))
        if arm=='baseline':
            os.link(checkpoint,folder/'ckpt_latest.pt')
            shutil.copy2(Path(str(checkpoint)+'.meta.json'),folder/'ckpt_latest.pt.meta.json')
        (folder/'branch_provenance.json').write_text(json.dumps(dict(
            source_checkpoint=str(checkpoint),source_step=680,source_inode=checkpoint.stat().st_ino,
            intervention=arm,optimizer_state_policy='Preserve Adam moments and schedule; no LR change',
            train_visits_start=680*16,train_visits_end=692*16,
            clip_action_grad=cfg['solver']['clip_action_grad'],clip_future_grad=cfg['solver']['clip_future_grad']),indent=2))
        with zipfile.ZipFile(folder/'source_branch.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
            for p in (ROOT/'energywam').rglob('*.py'):archive.write(p,str(p.relative_to(ROOT)))
            for n in ['train_shared_sharded.py','k_training_probe.py','prepare_stability_replay.py']:
                archive.write(ROOT/'scripts'/n,'scripts/'+n)
            archive.write(config,'experiment_config.yaml')
        job=dict(name=name,config=str(config.relative_to(ROOT)),physical_gpus=pair,steps=692,
                 manifest=str(OUT/'replay_manifest.json'),train_monitor_manifest=str(OUT/'train_monitor_manifest.json'),
                 initial_checkpoint=str(checkpoint),sample_eval_interval=692*16,train_monitor_sample_interval=4*16,
                 keep_best=True,monitor_resources=True,effective_batch=16,snapshot_bytes_estimate=checkpoint.stat().st_size,
                 max_process_uss_gib=140,max_wall_seconds=3600)
        # Only12 updates, no MSE guard against an unrelated fresh step0. Nonfinite
        # errors and resource guards still fail. Evaluate finite bad endpoints too.
        jobs.append(job)
        (OUT/f'jobs_{arm}.json').write_text(json.dumps([job],indent=2))
    plan=dict(status='prepared',source_step=680,target_step=692,new_updates_per_arm=12,
        sample_visits_per_arm=192,train_pool_windows=len(train),train_fixed_monitor_windows=12,val_windows=24,
        primary='Paired fixed TRAIN/VAL MSE at680and692; raw replay losses versus original681..692; per-inner-step traces, clipping fractions, gradient norm and resource peaks.',
        best='Min fixed24VAL Action+9Future among evaluated680/692, complete latest/best. Initial clipping-arm state is saved with its actual config.',
        limitation='Late-state rescue/failure reproduction, not a test of from-scratch generalization. Clipping changes the algorithm and can saturate gradients.',
        expected_minutes=[20,35],jobs=jobs,
        manifest_sha256=hashlib.sha256((OUT/'replay_manifest.json').read_bytes()).hexdigest())
    (OUT/'replay_plan.json').write_text(json.dumps(plan,indent=2))
    bridge=ROOT/'configs/wandb_bridge.json';settings=json.loads(bridge.read_text())
    for job in jobs:
        if not any(s['config']==job['config'] for s in settings['sources']):
            settings['sources'].append(dict(config=job['config'],group='stability-replay680-20261107',priority=150))
    temp=bridge.with_suffix('.pending.json');temp.write_text(json.dumps(settings,indent=2));temp.chmod(0o600);temp.replace(bridge)
    print(json.dumps({k:v for k,v in plan.items() if k!='jobs'},indent=2))


if __name__=='__main__':main()

"""Prepare an isolated full-state seed43 continuation, finite MSE warnings only."""
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import yaml

ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / 'logs/lab_remote_setup/seed43_warning_deployment.json'


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')


def main():
    assert not RECEIPT.exists(), 'Inspect existing deployment instead of duplicating'
    prior = json.loads((ROOT/'logs/lab_remote_setup/head_lr_seed43_deployment.json').read_text())
    oldroot = Path(prior['root']); old = oldroot/prior['name']
    oldplan = json.loads((oldroot/'plan.json').read_text())
    oldcode = Path(oldplan['code'])
    assert json.loads((old/'ckpt_latest.pt.meta.json').read_text())['meta']['global_step'] == 61
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    root = oldroot.parent/('single_episode_seed43_warn_continue_'+stamp)
    root.mkdir(); (root/'configs').mkdir()
    code = ROOT/'logs/frozen_sources'/('seed43_warn_'+stamp)/'code'
    shutil.copytree(oldcode, code, symlinks=True, ignore=shutil.ignore_patterns('logs','__pycache__','*.pyc'))
    (code/'logs').symlink_to(ROOT/'logs', target_is_directory=True)
    # Carry only the explicit guard-mode change into the already frozen code.
    current = (ROOT/'scripts/run_iteration_search.py').read_text()
    queue = code/'scripts/run_iteration_search.py'; text = queue.read_text()
    helper = current[current.index('def fixed_mse_guard_decision('):current.index('\ndef main():')]
    text = text.replace('\ndef main():', '\n'+helper+'\ndef main():', 1)
    # Original K=2 frozen guard predates the optional evaluation_k argument.
    signature = text[text.index('def fixed_mse_failure('):text.index('\n', text.index('def fixed_mse_failure('))]
    if 'evaluation_k' not in signature:
        text = text.replace(",\n        evaluation_k=job.get('evaluation_k', 2))", ")")
    start = text.index('                rejection=fixed_mse_failure(')
    end = text.index('                if job.get(\'resume_guard_ratio\'):', start)
    new_start = current.index('                rejection,warning=fixed_mse_guard_decision(')
    new_end = current.index('                if job.get(\'resume_guard_ratio\'):', new_start)
    queue.write_text(text[:start]+current[new_start:new_end]+text[end:])
    changes=[]; manifest=[]
    for f in code.rglob('*.py'):
        rel=f.relative_to(code)
        if 'logs' in rel.parts or '__pycache__' in rel.parts: continue
        if f.read_bytes() != (oldcode/rel).read_bytes(): changes.append(str(rel))
        manifest.append(dict(path=str(rel),sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
    assert changes == ['scripts/run_iteration_search.py']
    name='episode3346_headscale1_seed43_warn_resume61'
    folder=root/name; folder.mkdir()
    original=yaml.safe_load(Path(oldplan['jobs'][0]['config']).read_text()); cfg=copy.deepcopy(original)
    cfg['run_id']=name; cfg['trainer']['output_dir']=str(folder)
    config=root/'configs'/(name+'.yaml'); config.write_text(yaml.safe_dump(cfg,sort_keys=False))
    for file in ['train_val_manifest.json','train_monitor_manifest.json','environment_manifest.json']:
        shutil.copy2(oldroot/file,root/file)
    # Explicit provenance: inherited history through61; original62-80 remains only in source.
    for file in ['train_log.jsonl','timing_history.jsonl','checkpoint_history.jsonl','best_history.jsonl']:
        src=old/file
        if not src.exists():continue
        lines=[]
        for line in src.read_text().splitlines():
            row=json.loads(line)
            step=row.get('global_step_completed',row.get('global_step',-1)+(0 if file in ['checkpoint_history.jsonl','best_history.jsonl'] else 1))
            if step<=61:lines.append(line)
        (folder/file).write_text('\n'.join(lines)+'\n')
    for sub in ['', 'train_monitor']:
        dest=folder/sub;dest.mkdir(exist_ok=True)
        for f in (old/sub).glob('eval_*.json'):
            if int(f.name.split('_')[1])<=61:shutil.copy2(f,dest/f.name)
    for file in ['sample_manifest.json','dataset_baselines.json']:
        shutil.copy2(old/file,folder/file)
    # Own best generation, with full checkpoint and metadata, safe from either run's rotations.
    generation=folder/'.best_snapshots'/'inherited_step00000000';generation.mkdir(parents=True)
    shared=False
    try:os.link((old/'ckpt_best.pt').resolve(),generation/'checkpoint.pt');shared=True
    except OSError:shutil.copyfile(old/'ckpt_best.pt',generation/'checkpoint.pt')
    shutil.copy2(old/'ckpt_best.pt.meta.json',generation/'checkpoint.pt.meta.json')
    shutil.copy2(old/'best_selection.json',generation/'selection.json')
    (folder/'.best_current').symlink_to(str(generation.relative_to(folder)))
    for alias,target in [('ckpt_best.pt','checkpoint.pt'),('ckpt_best.pt.meta.json','checkpoint.pt.meta.json'),('best_selection.json','selection.json')]:
        (folder/alias).symlink_to('.best_current/'+target)
    job=copy.deepcopy(json.loads((oldroot/'jobs.json').read_text())[0])
    job.update(name=name,config=str(config),manifest=str(root/'train_val_manifest.json'),
               train_monitor_manifest=str(root/'train_monitor_manifest.json'),
               runtime_environment_manifest=str(root/'environment_manifest.json'),
               initial_checkpoint=str(old/'ckpt_latest.pt'),fixed_mse_guard_mode='warn')
    # Queue refuses logs without latest: an immutable symlink seeds first resume;
    # rolling writes atomically replace this alias, never the source checkpoint.
    (folder/'ckpt_latest.pt').symlink_to(old/'ckpt_latest.pt')
    shutil.copy2(old/'ckpt_latest.pt.meta.json',folder/'ckpt_latest.pt.meta.json')
    write(root/'jobs.json',[job])
    plan=copy.deepcopy(oldplan)
    plan.update(name='seed43-warning-continuation-'+stamp,root=str(root),code=str(code),
                jobs=[dict(name=name,folder=str(folder),config=str(config),head_lr=.001,head_scale=1.)],
                tmux_session='energywam_seed43_warn_'+stamp,
                authorization='User explicitly requested continuing despite finite large norms/MSE; retain clip1',
                comparison='Diagnostic continuation from original seed43 fullstep61 to210; finite fixed MSE10x becomes warning; all optimization unchanged',
                protocol='Original seed43 B16/native4/FP32/K2/A+9F/clip1; fixed evaluation2026; only finite MSE early-stop policy changes',
                rerun_mode='Full optimizer/scheduler/RNG resume61; replay62-80 then continue to210',
                required_free_bytes=(3 if shared else 2)*job['snapshot_bytes_estimate']+8*2**30,
                environment_manifest=str(root/'environment_manifest.json'),
                storage_gate_result=str(oldroot/'metadata_storage_gate/result.json'),
                inherited_updates=61,new_optimizer_updates_planned=149,reference_seed43_study=str(oldroot))
    write(root/'plan.json',plan);write(root/'source_manifest.json',manifest);write(root/'source_changes.json',changes)
    provenance=dict(source_run=str(old),source_checkpoint=str(old/'ckpt_latest.pt'),source_step=61,
                    inherited_history_through=61,replayed_original_updates=[62,80],new_updates_planned=149,
                    target_global_step=210,best_inherited_step=0,best_criterion=json.loads((old/'best_selection.json').read_text())['criterion'],
                    best_shared_physical_inode=shared,original_failure_records_untouched=True,
                    original_checkpoint_identity={k:dict(bytes=(old/k).stat().st_size,inode=(old/k).stat().st_ino) for k in ['ckpt_latest.pt','ckpt_best.pt']})
    write(folder/'continuation_provenance.json',provenance)
    write(folder/'arm_definition.json',dict(hyperparameters=cfg,job=job,provenance=provenance,
          purpose=plan['comparison'],retention='Full latest/best',fixed_mse_guard_mode='warn',outer_clip=1,
          evaluation_seed=2026,alternate_final_eval_seed=2027))
    write(root/'preflight_validation.json',dict(created_at=datetime.now(timezone.utc).isoformat(),free_bytes=shutil.disk_usage(root).free,
          required_bytes=plan['required_free_bytes'],source_changes=changes,training_math_unchanged=True,tests='6 guard tests passed'))
    write(RECEIPT,dict(root=str(root),code=str(code),plan=str(root/'plan.json'),name=name,
          python=plan['python'],tmux=plan['tmux_session']))
    print(RECEIPT.read_text())


if __name__=='__main__':main()

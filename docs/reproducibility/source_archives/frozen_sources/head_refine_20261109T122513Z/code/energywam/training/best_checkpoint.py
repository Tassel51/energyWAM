"""Preserve one complete best snapshot independently of rolling latest.

The checkpoint, sidecar and selection metadata live in one immutable generation.
Atomic pointer replacement commits a generation; hard links avoid copying full
states when best is also latest. The payload remains an ordinary full checkpoint.
"""
import datetime
import fcntl
import json
import math
import os
import shutil
import uuid
from pathlib import Path


def _sync_dir(path):
    fd=os.open(path,os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def _pointer(path,target):
    pending=path.with_name(path.name+'.link_pending')
    pending.unlink(missing_ok=True)
    pending.symlink_to(target)
    os.replace(pending,path)


def _discard_unreachable_generations(output):
    pointer=output/'.best_current'
    store=output/'.best_snapshots'
    if not pointer.is_symlink():return
    current=pointer.resolve(strict=True)
    if not current.is_relative_to(store.resolve()):raise ValueError('best pointer outside owned snapshot directory')
    for previous in store.iterdir():
        if previous.resolve()!=current and previous.is_dir() and not previous.is_symlink():shutil.rmtree(previous)
    _sync_dir(store)


def promote_latest_to_best(output,*,global_step,score,metrics,criterion):
    """Minimize a predeclared finite validation score; never modify latest."""
    output=Path(output)
    score=float(score)
    if not math.isfinite(score):raise ValueError('best score must be finite')
    if not criterion or not metrics:raise ValueError('explicit validation criterion and component metrics required')
    if any(not math.isfinite(float(v)) for v in metrics.values()):raise ValueError('best component metric must be finite')
    lock=Path(__file__).resolve().parents[2]/'logs/.checkpoint_write.lock'
    lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open('a+b') as stream:
        fcntl.flock(stream.fileno(),fcntl.LOCK_EX)
        _discard_unreachable_generations(output)
        selection=output/'best_selection.json'
        old=json.loads(selection.read_text()) if selection.exists() else None
        if old and old['criterion']!=criterion:raise ValueError('best validation criterion cannot change within a run')
        if old and score>=old['score']:return False
        latest=output/'ckpt_latest.pt';meta=Path(str(latest)+'.meta.json')
        if not latest.exists() or not meta.exists():raise ValueError('complete latest checkpoint and metadata required')
        saved=json.loads(meta.read_text())
        if saved['meta']['global_step']!=global_step:raise ValueError('latest does not represent evaluated weights')
        store=output/'.best_snapshots';store.mkdir(exist_ok=True)
        generation=store/f'step{global_step:08d}_{uuid.uuid4().hex}'
        generation.mkdir();committed=False
        try:
            for source,name in [(latest,'checkpoint.pt'),(meta,'checkpoint.pt.meta.json')]:
                dest=generation/name
                try:os.link(source,dest)
                except OSError:shutil.copyfile(source,dest)
                with dest.open('rb') as f:os.fsync(f.fileno())
            info=dict(global_step=int(global_step),score=score,metrics={k:float(v) for k,v in metrics.items()},criterion=criterion,selected_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
            (generation/'selection.json').write_text(json.dumps(info,indent=2))
            with (generation/'selection.json').open('rb') as f:os.fsync(f.fileno())
            _sync_dir(generation);_sync_dir(store)
            # Public aliases resolve through one atomic generation pointer.
            for alias,target in [('ckpt_best.pt','.best_current/checkpoint.pt'),('ckpt_best.pt.meta.json','.best_current/checkpoint.pt.meta.json'),('best_selection.json','.best_current/selection.json')]:
                path=output/alias
                if path.exists() and not path.is_symlink():raise ValueError(f'legacy best file requires explicit migration: {path}')
                if not path.is_symlink():_pointer(path,target)
            _pointer(output/'.best_current',str(generation.relative_to(output)))
            committed=True;_sync_dir(output)
            with (output/'best_history.jsonl').open('a') as f:f.write(json.dumps(info)+'\n')
            for previous in store.iterdir():
                if previous!=generation and previous.is_dir() and not previous.is_symlink():shutil.rmtree(previous)
            _sync_dir(store)
            return True
        finally:
            # An interruption may occur after pointer replacement but before
            # the Python commit flag is set; never delete its target.
            selected=output/'.best_current'
            if not committed and selected.is_symlink():committed=selected.resolve()==generation.resolve()
            if not committed:shutil.rmtree(generation,ignore_errors=True)


def validation_score_from_reports(output,global_step,*,lambda_action,lambda_future,evaluation_k=2):
    """Declared training-objective weights on fixed conditional VAL MSE only."""
    components={};identity=None;seed=None
    for direction in ['action','future']:
        path=Path(output)/f'eval_{global_step:04d}_{direction}.json'
        report=json.loads(path.read_text())
        if report['weight_updates_during_eval']!=0:raise ValueError('validation must not update weights')
        rows=[x for x in report['rows'] if x['split']=='val' and x['k']==evaluation_k]
        if not rows or any(x['update']!=global_step for x in rows):raise ValueError('validation step/rows mismatch')
        ids=[(x['window_id'],x['episode_index'],x['frame_index']) for x in rows]
        if len(set(ids))!=len(ids):raise ValueError('duplicate validation windows')
        if identity is not None and (ids!=identity or report.get('evaluation_seed')!=seed):raise ValueError('conditional validation windows/seeds differ')
        identity=ids;seed=report.get('evaluation_seed')
        values=[float(x['prediction_mse']) for x in rows]
        if any(not math.isfinite(x) for x in values):raise ValueError('nonfinite validation MSE')
        components[f'val_{direction}_mse']=sum(values)/len(values)
    weights=[float(lambda_action),float(lambda_future)]
    if any(not math.isfinite(v) or v<0 for v in weights) or sum(weights)==0:raise ValueError('invalid validation weights')
    score=weights[0]*components['val_action_mse']+weights[1]*components['val_future_mse']
    criterion=f'weighted_conditional_validation_mse:K={evaluation_k};lambda_action={weights[0]:.17g};lambda_future={weights[1]:.17g}'
    return score,components,criterion

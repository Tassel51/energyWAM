"""Bounded checkpoint storage and deterministic, seekable training order."""
import os
from pathlib import Path
import torch
from torch.utils.data import Sampler


def checkpoint_due(step, final_step, elapsed_seconds, step_interval, seconds_interval=0.):
    if step == final_step:
        return True
    if seconds_interval > 0:
        return elapsed_seconds >= seconds_interval
    return step % step_interval == 0


def reconcile_resume_outputs(output, restored_step):
    """Archive uncheckpointed records before replaying from a saved model.

    Weight files are never modified. Original records remain in resume_history;
    current evaluation paths must not silently reuse a later model's outputs.
    """
    import hashlib
    import json
    import re
    import shutil
    import time
    output = Path(output)
    if restored_step < 0:
        raise ValueError('negative restored step')
    selection = output / 'best_selection.json'
    if selection.exists() and json.loads(selection.read_text())['global_step'] > restored_step:
        raise ValueError('resume checkpoint is older than retained best; use a separate output directory')

    def update(row, completed=False):
        if row.get('event'):
            return None
        if 'global_step_completed' in row:
            return int(row['global_step_completed'])
        if 'global_step' in row:
            return int(round(row['global_step'])) + (0 if completed else 1)
        return None

    planned = {}
    for folder in [output, output / 'train_monitor']:
        for path in folder.glob('eval_*.json'):
            match = re.fullmatch(r'eval_(\d+)_(action|future)\.json', path.name)
            if not match:
                continue
            try:
                report = json.loads(path.read_text())
                rows = report.get('rows')
                valid = (isinstance(rows, list) and bool(rows)
                         and all(row.get('update') == int(match[1]) for row in rows))
            except (OSError, ValueError, AttributeError):
                valid = False
            if int(match[1]) > restored_step or not valid:
                planned[path] = None
    for name in ['train_log.jsonl', 'timing_history.jsonl',
                 'checkpoint_history.jsonl', 'best_history.jsonl']:
        path = output / name
        if not path.exists():
            continue
        original = path.read_bytes()
        kept = []
        for line in original.splitlines(keepends=True):
            try:
                row = json.loads(line)
                step = update(row, completed=name in ['checkpoint_history.jsonl', 'best_history.jsonl'])
            except (ValueError, TypeError, AttributeError):
                continue
            if step is None or step <= restored_step:
                kept.append(line if line.endswith(b'\n') else line + b'\n')
        replacement = b''.join(kept)
        if replacement != original:
            planned[path] = replacement
    speed = output / 'speed_report.json'
    if speed.exists():
        try:
            report = json.loads(speed.read_text())
            rows = [row for row in report['steps'] if update(row) <= restored_step]
            if rows != report['steps'] or report['completed_steps'] != restored_step:
                report.update(completed_steps=restored_step, steps=rows)
                planned[speed] = (json.dumps(report, indent=2) + '\n').encode()
        except (OSError, ValueError, TypeError, KeyError):
            planned[speed] = None
    final = output / 'final_report.json'
    if final.exists():
        planned[final] = None
    if not planned:
        return dict(changed=False, restored_step=restored_step)

    generation = time.time_ns()
    archive = output / 'resume_history' / f'{generation}_from{restored_step}'
    archive.mkdir(parents=True)
    records = []
    # Verify all backups before changing any active log/evaluation path.
    for path, replacement in planned.items():
        target = archive / path.relative_to(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise RuntimeError('resume-output backup verification failed')
        records.append(dict(path=str(path.relative_to(output)), sha256=digest,
                            action='archive' if replacement is None else 'trim_to_checkpoint'))
    manifest = dict(changed=True, generation=generation, restored_step=restored_step,
                    archive_dir=str(archive), records=records,
                    scope='Only log/evaluation records reconciled; checkpoints and datasets untouched.')
    (archive / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    for path, replacement in planned.items():
        if replacement is None:
            path.unlink()
        else:
            pending = path.with_name(path.name + '.resume_pending')
            with pending.open('wb') as stream:
                stream.write(replacement)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(pending, path)
    latest = output / 'resume_history' / 'latest.json'
    pending = latest.with_suffix('.pending.json')
    pending.write_text(json.dumps(manifest, indent=2) + '\n')
    pending.replace(latest)
    return manifest


class ResumableWindowSampler(Sampler):
    def __init__(self, size, seed, consumed=0, shuffle=True, batch_size=2):
        self.size, self.seed, self.consumed = size, seed, consumed
        self.shuffle = shuffle
        self.usable = size // batch_size * batch_size
        if not self.usable:
            raise ValueError('dataset has fewer than one batch')

    def __iter__(self):
        epoch, offset = divmod(self.consumed, self.usable)
        order = torch.randperm(self.size, generator=torch.Generator().manual_seed(self.seed + epoch)) if self.shuffle else torch.arange(self.size)
        for i in range(offset, self.usable):
            yield int(order[i])
        self.consumed = (epoch + 1) * self.usable

    def __len__(self):
        return self.usable - self.consumed % self.usable


def save_latest(trainer, output):
    """Serialize this project's large rolling writes; only one pending copy."""
    import fcntl
    lock_path = Path(__file__).resolve().parents[2]/'logs'/'.checkpoint_write.lock'
    lock_path.parent.mkdir(parents=True,exist_ok=True)
    with lock_path.open('a+b') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
        return _save_latest_locked(trainer,output)


def _save_latest_locked(trainer, output):
    """Keep old checkpoint until the replacement has been saved and fsynced."""
    output = Path(output)
    try:
        temporary = trainer.save('ckpt_pending.pt')
    except Exception:
        # A failed serialization is never a recoverable checkpoint. Keep the
        # previous latest intact and release partial disk usage before retry.
        (output / 'ckpt_pending.pt').unlink(missing_ok=True)
        (output / 'ckpt_pending.pt.meta.json').unlink(missing_ok=True)
        raise
    try:
        # Verify serialization and the resume contract before replacing latest.
        # mmap keeps this check from allocating a second full model in RAM.
        payload=torch.load(temporary,map_location='cpu',weights_only=False,mmap=True)
        if 'model' in payload or hasattr(trainer,'model'):
            required={'model','optimizer','scheduler','meta','config','rng','format_version'}
            if not required.issubset(payload):raise ValueError('incomplete full-resume checkpoint')
            import json
            from .checkpoint_metadata import read_checkpoint_metadata
            sidecar=read_checkpoint_metadata(Path(str(temporary)+'.meta.json'),expected_meta=payload['meta'])
            if payload['meta']!=sidecar['meta']:raise ValueError('checkpoint/sidecar metadata mismatch')
        del payload
        with temporary.open('rb') as stream:os.fsync(stream.fileno())
    except Exception:
        temporary.unlink(missing_ok=True)
        Path(str(temporary)+'.meta.json').unlink(missing_ok=True)
        raise
    latest = output / 'ckpt_latest.pt'
    os.replace(temporary, latest)
    meta = Path(str(temporary) + '.meta.json')
    if meta.exists():
        os.replace(meta, Path(str(latest) + '.meta.json'))
    fd = os.open(output, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return latest


class FixedSampleBudgetSampler(Sampler):
    """Batch-independent order; batches may cross permutation boundaries.

    Unlike epoch drop_last sampling, batch8 cannot silently skip the final
    four windows of a 20-window pool. Positions are the shared experiment axis.
    """
    def __init__(self,size,seed,consumed,total):
        if size<1 or consumed<0 or total<consumed:
            raise ValueError('invalid sample budget')
        self.size,self.seed,self.consumed,self.total=size,seed,consumed,total

    def __iter__(self):
        cached_epoch=None;order=None
        for position in range(self.consumed,self.total):
            epoch,offset=divmod(position,self.size)
            if epoch!=cached_epoch:
                order=torch.randperm(self.size,generator=torch.Generator().manual_seed(self.seed+epoch)).tolist()
                cached_epoch=epoch
            yield order[offset]

    def __len__(self):return self.total-self.consumed

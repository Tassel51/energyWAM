"""Explicit absolute LR override after full resume; Adam moments are untouched."""
import math
import torch


def override_resumed_lrs(optimizer, scheduler, target_base_lrs):
    if not isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR):
        raise ValueError('Only the project LambdaLR schedule is supported')
    groups = optimizer.param_groups
    targets = [float(x) for x in target_base_lrs]
    if len(targets) != len(groups) or any(not math.isfinite(x) or x <= 0 for x in targets):
        raise ValueError('Invalid target learning rates')
    before = [float(g['lr']) for g in groups]
    current = [base * fn(scheduler.last_epoch) for base, fn in zip(targets, scheduler.lr_lambdas)]
    if len(current) != len(groups):
        raise ValueError('Scheduler group mismatch')
    scheduler.base_lrs = targets
    scheduler._last_lr = current
    for group, base, lr in zip(groups, targets, current):
        group['initial_lr'] = base
        group['lr'] = lr
    return dict(before_lrs=before, base_lrs=targets, effective_lrs=current,
                scheduler_last_epoch=scheduler.last_epoch,
                policy='absolute config LRs; preserve Adam moments, RNG and scheduler progress')

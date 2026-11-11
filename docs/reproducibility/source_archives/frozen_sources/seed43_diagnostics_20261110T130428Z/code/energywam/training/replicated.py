"""Exact loss decomposition over replicated models; manual outer-gradient SUM.

No DDP autograd hooks are installed: inner autograd.grad and full unroll stay
local. BI ranks own opposite directions on the SAME batch. Joint ranks own
separate samples. Each rank applies identical summed outer gradients once.
"""
import time
import os
import torch
import torch.distributed as dist
import numpy as np
from .trainer import EnergyWAMTrainer


def slice_batch(batch, start, end):
    size = int(batch['action_target'].shape[0])
    result = {}
    for key, value in batch.items():
        if isinstance(value, (torch.Tensor, np.ndarray)) and value.ndim and value.shape[0] == size:
            result[key] = value[start:end]
        elif isinstance(value, (list, tuple)) and len(value) == size:
            result[key] = value[start:end]
        else:
            result[key] = value
    return result


def sum_outer_gradients(model, bucket_bytes=256 * 2**20):
    parameters = [p for p in model.parameters() if p.requires_grad]
    presence = torch.tensor([p.grad is not None for p in parameters], dtype=torch.int32, device=parameters[0].device)
    dist.all_reduce(presence, op=dist.ReduceOp.MAX)
    active = presence.cpu().tolist()
    gradients = {}
    for p, used in zip(parameters, active):
        if not used:
            continue
        if p.grad is None:
            p.grad = torch.zeros_like(p)
        if not p.grad.is_contiguous():
            p.grad = p.grad.contiguous()
        gradients.setdefault(p.grad.dtype, []).append(p.grad)
    count = 0
    pending = []
    def finish_one():
        work, flat, tensors = pending.pop(0)
        work.wait()
        if len(tensors) > 1:
            offset = 0
            for g in tensors:
                g.copy_(flat[offset:offset+g.numel()].view_as(g))
                offset += g.numel()

    def flush(tensors):
        nonlocal count
        if not tensors:
            return
        flat = tensors[0] if len(tensors) == 1 else torch.cat([g.view(-1) for g in tensors])
        work = dist.all_reduce(flat, op=dist.ReduceOp.SUM, async_op=True)
        pending.append((work, flat, tensors))
        # Bound temporary storage while overlapping packing and communication.
        limit = 2 if os.environ.get('ENERGYWAM_PIPELINE_GRADIENTS', '1') == '1' else 1
        if len(pending) >= limit:
            finish_one()
        count += 1
    for group in gradients.values():
        bucket, size = [], 0
        for grad in group:
            nbytes = grad.numel() * grad.element_size()
            if bucket and size + nbytes > bucket_bytes:
                flush(bucket)
                bucket, size = [], 0
            if nbytes >= bucket_bytes:
                flush([grad])
            else:
                bucket.append(grad)
                size += nbytes
        flush(bucket)
    while pending:
        finish_one()
    return count


def accumulate_log(target, key, value, weight=1.0):
    if key.endswith(('_valid_elems', '_bytes', '_count')):
        target[key] = target.get(key, 0.0) + value
    elif key.endswith('_min'):
        target[key] = min(target.get(key, value), value)
    elif key.endswith('_max'):
        target[key] = max(target.get(key, value), value)
    else:
        target[key] = target.get(key, 0.0) + value * weight


class ReplicatedTrainer(EnergyWAMTrainer):
    def _stash_gradients(self):
        if not hasattr(self, '_gradient_stash'):
            self._gradient_stash = {}
        self._active_stash = []
        for p in self.model.parameters():
            if p.grad is None:
                continue
            if p not in self._gradient_stash:
                self._gradient_stash[p] = torch.empty_like(p.grad, device='cpu', pin_memory=self.device.type=='cuda')
            self._gradient_stash[p].copy_(p.grad, non_blocking=self.device.type=='cuda')
            if self.device.type=='cuda':
                p.grad.record_stream(torch.cuda.current_stream(self.device))
            self._active_stash.append(p)
            p.grad = None
        if self.device.type=='cuda':
            torch.cuda.current_stream(self.device).synchronize()

    def _merge_stashed_gradients(self):
        for p in self._active_stash:
            first = self._gradient_stash[p].to(p.device, non_blocking=self.device.type=='cuda')
            if p.grad is None:
                p.grad = first
            else:
                p.grad.add_(first)
        self._active_stash = []

    def train_step(self, batch):
        rank, world = dist.get_rank(), dist.get_world_size()
        if world != 2 or self.cfg.stage.stage not in ('bi', 'joint'):
            raise ValueError('replicated training requires exactly two ranks and bi/joint stage')
        if not self.cfg.data.require_full_horizon:
            raise ValueError('microbatch loss decomposition requires full, uniformly valid horizons')
        for name in ('action_mask', 'future_mask'):
            if name in batch and not bool(torch.as_tensor(batch[name]).all()):
                raise ValueError('replicated loss decomposition requires fully valid masks')
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats(self.device) if self.device.type == 'cuda' else None
        size = int(batch['action_target'].shape[0])
        if size != 2:
            raise ValueError('current replicated runner preserves effective global batch=2')
        t0 = time.time()
        forward_seconds = backward_seconds = 0.0
        logs = {}
        local_loss = 0.0
        indices = range(size) if self.cfg.stage.stage == 'bi' else (rank,)
        def local_backward():
            nonlocal forward_seconds, backward_seconds, local_loss
            for index in indices:
                micro = slice_batch(batch, index, index + 1)
                micro['_global_step'] = self.global_step
                if self.cfg.stage.stage == 'bi':
                    micro['_bi_only'] = 'action' if rank == 0 else 'future'
                begin = time.time()
                out = self.model(micro)
                forward_seconds += time.time() - begin
                loss = out['action_loss']
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError(f'non-finite loss on rank {rank} at step {self.global_step}')
                # BI: each direction averages its two samples, then rank gradients
                # SUM. Joint: each rank contributes half of the two-sample mean.
                weighted = loss / size
                begin = time.time()
                weighted.backward()
                backward_seconds += time.time() - begin
                local_loss += float(loss.detach()) / (size if self.cfg.stage.stage == 'bi' else 1)
                for key, value in out.items():
                    if key != 'action_loss' and isinstance(value, torch.Tensor) and value.ndim == 0:
                        accumulate_log(logs, key, float(value.detach()), 1 / len(indices))
                del loss, weighted, out
                if self.cfg.stage.stage == 'bi' and index == 0:
                    self._stash_gradients()
                if os.environ.get("ENERGYWAM_SERIAL_BI")=="1" and self.device.type=="cuda":
                    torch.cuda.synchronize(self.device)
                    if hasattr(torch._C,"_host_emptyCache"):torch._C._host_emptyCache()
            if self.cfg.stage.stage == 'bi':
                self._merge_stashed_gradients()
        serial = self.cfg.stage.stage == 'bi' and os.environ.get('ENERGYWAM_SERIAL_BI') == '1'
        if serial:
            for turn in range(world):
                if rank == turn:
                    local_backward()
                    if self.device.type == 'cuda':
                        torch.cuda.synchronize(self.device)
                        if hasattr(torch._C, '_host_emptyCache'):torch._C._host_emptyCache()
                dist.barrier()
        else:
            local_backward()
        begin_sync = time.time()
        buckets = sum_outer_gradients(self.model, self.cfg.trainer.gradient_sync_bucket_mib * 2**20)
        gathered = [None] * world
        dist.all_gather_object(gathered, {'loss': local_loss, 'logs': logs})
        combined = {}
        keys = set().union(*(item['logs'].keys() for item in gathered))
        for key in keys:
            values = [item['logs'][key] for item in gathered if key in item['logs']]
            for value in values:
                accumulate_log(combined, key, value, 1 / len(values))
        total_loss = sum(item['loss'] for item in gathered)
        if self.cfg.stage.stage == 'joint':
            total_loss /= world
        out = {key: torch.tensor(value) for key, value in combined.items()}
        out['action_loss'] = torch.tensor(total_loss)
        out['gradient_sync_seconds'] = torch.tensor(time.time() - begin_sync)
        out['replica_world_size'] = torch.tensor(float(world))
        out['gradient_sync_buckets'] = torch.tensor(float(buckets))
        return self._finish_step(batch, out, t0, forward_seconds, backward_seconds)

"""Budgeted GPU/CPU activation storage with resident parameter views.

Repeated energy forwards reuse the same language parameters. Plain save_on_cpu
copies those parameters again at every use. Parameters already live on their
owning GPU until the outer optimizer step, so saved views can share that same
storage. A per-GPU budget retains activation storage; excess activations are
copied to CPU independently. This changes storage,
not differentiation. Optimizer updates must happen after all backwards.
"""
from contextlib import contextmanager
from numbers import Real
from collections import deque

import torch
import weakref


@contextmanager
def parameter_aware_cpu_offload(model, gpu_budget_gib=0.0, *, pin_budget_gib=0.0, asynchronous=False, max_inflight_gib=1.0, host_cache_limit_gib=0.0, deduplicate_cpu=False):
    storages = {(p.device, p.untyped_storage().data_ptr()) for p in model.parameters()}
    stats = dict(parameter_copies=0, parameter_cache_hits=0, avoided_copy_bytes=0,cpu_dedup_hits=0,cpu_dedup_avoided_bytes=0)
    per_device = not isinstance(gpu_budget_gib, Real)
    budgets = [int(float(v) * 2**30) for v in gpu_budget_gib] if per_device else [int(float(gpu_budget_gib) * 2**30)]
    if per_device and len(budgets) != torch.cuda.device_count():
        raise ValueError("provide one saved-tensor budget per visible CUDA device")
    if any(b < 0 for b in budgets):
        raise ValueError("GPU saved-tensor budget must be nonnegative")
    resident = set()
    used = {}
    pin_limit = int(float(pin_budget_gib) * 2**30)
    pinned_bytes = 0
    copy_streams = {}
    pending = {}
    inflight_limit = int(float(max_inflight_gib) * 2**30)
    host_cache_limit = int(float(host_cache_limit_gib) * 2**30)
    pin_requests = 0
    # Weak references prevent cache entries from retaining dead GPU activations
    # or CPU buffers, and object identity/version avoids recycled-pointer aliasing.
    cpu_cache = {}

    def remember(tensor, packed):
        if deduplicate_cpu:
            try:version=tensor._version
            except RuntimeError:return packed
            cpu_cache[id(tensor)]=(weakref.ref(tensor),version,weakref.ref(packed[1]),packed[2:])
        return packed

    def pack(tensor):
        nonlocal pinned_bytes, pin_requests
        identity = (tensor.device, tensor.untyped_storage().data_ptr())
        if identity in storages:
            stats['parameter_cache_hits'] += 1
            stats['avoided_copy_bytes'] += tensor.numel() * tensor.element_size()
            return tensor.device, tensor.detach()
        budget = budgets[tensor.device.index] if tensor.is_cuda and per_device else budgets[0]
        if tensor.is_cuda and budget:
            size = tensor.untyped_storage().nbytes()
            if identity in resident or used.get(tensor.device, 0) + size <= budget:
                if identity not in resident:
                    resident.add(identity)
                    used[tensor.device] = used.get(tensor.device, 0) + size
                    stats[f'gpu_resident_bytes_{tensor.device.index}'] = used[tensor.device]
                return tensor.device, tensor.detach()
        if deduplicate_cpu:
            entry=cpu_cache.get(id(tensor))
            if entry is not None:
                source,version,buffer,tail=entry
                cpu=buffer()
                try:unchanged=tensor._version==version
                except RuntimeError:unchanged=False
                if source() is tensor and cpu is not None and unchanged:
                    stats['cpu_dedup_hits']+=1
                    if tensor.is_cuda:stats['cpu_dedup_avoided_bytes']+=tensor.numel()*tensor.element_size()
                    return (tensor.device,cpu)+tail
        # Pageable copies block the host for every saved tensor. Bound pinned
        # storage and move D2H work onto a separate stream with explicit events.
        rounded = 1 << max(0, (tensor.numel() * tensor.element_size() - 1).bit_length())
        if tensor.is_cuda and pin_limit and pinned_bytes + rounded <= pin_limit:
            pin_requests += 1
            if host_cache_limit and (pin_requests == 1 or pin_requests % 64 == 0):
                allocated = torch.cuda.memory.host_memory_stats().get('allocated_bytes.current', 0)
                if allocated > host_cache_limit and hasattr(torch._C, '_host_emptyCache'):
                    # Release only inactive blocks; live saved tensors and
                    # stream-dependent copies remain owned by the allocator.
                    torch._C._host_emptyCache()
                    stats['host_cache_trim_count'] = stats.get('host_cache_trim_count', 0) + 1
            cpu = torch.empty_like(tensor, device='cpu', pin_memory=True)
            pinned_bytes += rounded
            stats['pinned_bytes'] = pinned_bytes
            if asynchronous:
                queue = pending.setdefault(tensor.device, deque())
                size = tensor.untyped_storage().nbytes()
                while queue and queue[0][0].query():
                    queue.popleft()
                while queue and sum(n for _, n in queue) + size > inflight_limit:
                    queue.popleft()[0].synchronize()
                stream = copy_streams.get(tensor.device)
                if stream is None:
                    stream = torch.cuda.Stream(device=tensor.device)
                    copy_streams[tensor.device] = stream
                stream.wait_stream(torch.cuda.current_stream(tensor.device))
                event = torch.cuda.Event()
                with torch.cuda.stream(stream):
                    cpu.copy_(tensor.detach(), non_blocking=True)
                    tensor.record_stream(stream)
                    event.record(stream)
                queue.append((event, size))
                if size > inflight_limit:
                    event.synchronize()
                return remember(tensor,(tensor.device, cpu, event))
            cpu.copy_(tensor.detach())
            return remember(tensor,(tensor.device, cpu))
        cpu = tensor.detach().to('cpu')
        if tensor.is_cuda:
            stats['pageable_bytes'] = stats.get('pageable_bytes', 0) + tensor.numel() * tensor.element_size()
        return remember(tensor,(tensor.device, cpu))

    def unpack(packed):
        device, cpu = packed[:2]
        if len(packed) == 3:
            torch.cuda.current_stream(device).wait_event(packed[2])
        return cpu.to(device, non_blocking=cpu.is_pinned())

    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        yield stats

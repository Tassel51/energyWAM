"""Move existing bitsandbytes state without changing its representation or kernels.

State lives on CPU while the model runs. All state returns to its parameter's
owning device before the original optimizer step. Checkpoints serialize the
ordinary optimizer state; no external state or custom checkpoint format exists.
"""
import torch


class OptimizerStateOffloader:
    def __init__(self, optimizer):
        self.optimizer = getattr(optimizer, 'base_optimizer', optimizer)
        self.last_offloaded_bytes = 0

    def _move(self, to_cpu):
        memo = {}
        moved = 0

        def visit(value, target):
            nonlocal moved
            if isinstance(value, torch.Tensor):
                if getattr(value, 'is_paged', False):
                    raise ValueError('Paged optimizer state is not supported by explicit CPU offload')
                key = (id(value), str(target))
                if key not in memo:
                    if value.device != target:
                        moved += value.numel() * value.element_size()
                    # Synchronous copies establish completion before releasing
                    # GPU state or calling the original update kernel.
                    memo[key] = value.to(target, non_blocking=False)
                return memo[key]
            if isinstance(value, dict):
                return {k: visit(v, target) for k, v in value.items()}
            if isinstance(value, list):
                return [visit(v, target) for v in value]
            if isinstance(value, tuple):
                return tuple(visit(v, target) for v in value)
            return value

        for parameter, state in self.optimizer.state.items():
            target = torch.device('cpu') if to_cpu else parameter.device
            self.optimizer.state[parameter] = visit(state, target)
        return moved

    def offload(self):
        self.last_offloaded_bytes = self._move(True)
        return self.last_offloaded_bytes

    def restore(self):
        return self._move(False)

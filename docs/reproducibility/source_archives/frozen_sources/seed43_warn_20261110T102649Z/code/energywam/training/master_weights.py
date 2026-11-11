"""Keep FP32 optimizer weights while forward/backward uses BF16 model weights.

The persistent FP32 values accumulate updates smaller than one BF16 ULP.
Only reduced-precision parameters receive a master copy. FP32 task parameters
and learned step sizes remain the original parameters.
"""
import torch


class FP32MasterOptimizer(torch.optim.Optimizer):
    def __init__(self, base_optimizer):
        self.base_optimizer = base_optimizer
        self.pairs = []
        for group in base_optimizer.param_groups:
            masters = []
            for parameter in group['params']:
                if parameter.dtype in (torch.bfloat16, torch.float16):
                    master = torch.nn.Parameter(parameter.detach().float().clone())
                    self.pairs.append((parameter, master))
                else:
                    master = parameter
                masters.append(master)
            group['params'] = masters
        if base_optimizer.state:
            raise ValueError('master weights must be installed before optimizer state allocation')
        super().__init__(base_optimizer.param_groups, base_optimizer.defaults)
        self.param_groups = base_optimizer.param_groups
        self.state = base_optimizer.state

    def zero_grad(self, set_to_none=True):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)
        for parameter, _ in self.pairs:
            if set_to_none:
                parameter.grad = None
            elif parameter.grad is not None:
                parameter.grad.detach_(); parameter.grad.zero_()

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError('closures are not supported by FP32MasterOptimizer')
        for parameter, master in self.pairs:
            master.grad = None if parameter.grad is None else parameter.grad.detach().float()
        result = self.base_optimizer.step()
        for parameter, master in self.pairs:
            parameter.copy_(master.to(parameter.dtype))
            master.grad = None
        return result

    @torch.no_grad()
    def sync_masters_from_model(self):
        """Required after a weight-only load with explicit optimizer reset.

        Exact resume instead loads saved master values via load_state_dict.
        """
        for parameter, master in self.pairs:
            master.copy_(parameter.detach().float())
            master.grad = None

    def state_dict(self):
        result = self.base_optimizer.state_dict()
        result['fp32_master_values'] = [m.detach() for _, m in self.pairs]
        return result

    def load_state_dict(self, state_dict):
        values = state_dict.get('fp32_master_values')
        if values is None or len(values) != len(self.pairs):
            raise ValueError('checkpoint lacks matching persistent FP32 master weights')
        state = {k:v for k,v in state_dict.items() if k != 'fp32_master_values'}
        self.base_optimizer.load_state_dict(state)
        self.param_groups = self.base_optimizer.param_groups
        self.state = self.base_optimizer.state
        with torch.no_grad():
            for (parameter, master), saved in zip(self.pairs, values):
                if master.shape != saved.shape:raise ValueError('master weight geometry mismatch')
                master.copy_(saved.to(master));parameter.copy_(master.to(parameter.dtype))

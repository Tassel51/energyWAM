"""Continuous iteration conditioning and constrained, differentiable step sizes."""
import math
import torch
from torch import nn


class IterationConditioning(nn.Module):
    def __init__(self, hidden_size, embedding_dim=256, mlp_hidden=256, scale=1000.0):
        super().__init__()
        if embedding_dim < 2 or embedding_dim % 2:
            raise ValueError("embedding_dim must be even and >= 2")
        self.scale = float(scale)
        self.register_buffer("frequencies", torch.exp(
            -math.log(10000.0) * torch.arange(embedding_dim // 2).float() / (embedding_dim // 2)
        ))
        def mlp():
            net = nn.Sequential(nn.Linear(embedding_dim, mlp_hidden), nn.SiLU(),
                                nn.Linear(mlp_hidden, hidden_size))
            nn.init.normal_(net[-1].weight, std=1e-3)
            nn.init.zeros_(net[-1].bias)
            return net
        self.action_mlp, self.future_mlp = mlp(), mlp()

    def encode(self, progress):
        s = torch.as_tensor(progress, device=self.frequencies.device, dtype=torch.float32)
        if s.numel() != 1:
            raise ValueError("one shared scalar progress k/K is required per solver iteration")
        phase = s.reshape(1, 1) * self.scale * self.frequencies[None, :]
        return torch.stack((phase.sin(), phase.cos()), dim=-1).flatten(-2)

    def forward(self, progress, direction):
        if progress is None:
            raise ValueError("iteration-aware energy requires explicit iteration_progress=k/K")
        pe = self.encode(progress)
        mlp = {"action": self.action_mlp, "future": self.future_mlp}[direction]
        return mlp(pe.to(next(mlp.parameters()).dtype)).unsqueeze(1)


class BoundedStepSizes(nn.Module):
    """Two scalar steps shared across inner iterations; positive bounded log ratios."""
    def __init__(self, action, future, min_ratio=0.1, max_ratio=10.0):
        super().__init__()
        if not (0 < min_ratio < 1 < max_ratio and action > 0 and future > 0):
            raise ValueError("positive anchors and bounds straddling ratio=1 required")
        self.register_buffer("initial", torch.tensor([action, future], dtype=torch.float32))
        self.register_buffer("log_bounds", torch.tensor([math.log(min_ratio), math.log(max_ratio)]))
        q = -math.log(min_ratio) / math.log(max_ratio/min_ratio)
        self.raw = nn.Parameter(torch.full((2,), math.log(q/(1-q)), dtype=torch.float32))

    def forward(self):
        lo, hi = self.log_bounds.float().unbind()
        alpha = self.initial.float() * torch.exp(lo + (hi-lo)*self.raw.float().sigmoid())
        return alpha[0], alpha[1]

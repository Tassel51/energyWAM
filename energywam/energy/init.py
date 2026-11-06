"""Candidate initialisation for the energy solver.

The engineering requirements forbid using a noised ground truth as the joint
deployment initialisation.  The supported schemes are:

``normal``               A0 ~ N(init_action_mean, init_action_std^2) in the
                         **model action space**; Z0 ~ N(init_future_mean,
                         init_future_std^2) in the model future space.  Under
                         ``normalization_profile='paper_standardized_v1'`` that
                         space is the standardized coordinate and the approved
                         baseline prior is N(0, I) (paper §4.8 Eq. 36), which
                         ``validate_config`` enforces unless an explicit prior
                         ablation switch is set.
``zeros``                A0 = 0 (action only).
``current_dino_noise``   Z0 = current-frame DINO tokens + N(0, init_future_noise_std^2).
                         Requires the current DINO features; the mean is the
                         *current* frame's DINO patch tokens, the scale is the
                         configured noise std.  This is a training/inference
                         option, not the default.

Under the paper profile the ``current_dino_noise`` base is first mapped from
raw DINO features into the standardized space with the **future** statistics
(``future_stats=``), so the noise lives in the same coordinate the solver
optimises in.  It never uses the future ground truth.  Note this is NOT the
full paper Eq. (40) variant: no residual statistics ``sigma_{dZ}`` are
estimated — the configured ``init_future_noise_std`` is used as the scale.

All schemes are differentiable-free (they only create the starting point) but
return float32 tensors with ``requires_grad`` set appropriately by the caller.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import torch

from ..config import InitConfig

if TYPE_CHECKING:  # pragma: no cover
    from ..data.standardization import ZScoreStats


def init_action(
    batch: int,
    horizon: int,
    action_dim: int,
    cfg: InitConfig,
    *,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    shape = (batch, horizon, action_dim)
    if cfg.initialization_action == "normal":
        return (
            cfg.init_action_mean
            + cfg.init_action_std
            * torch.randn(shape, device=device, dtype=torch.float32, generator=generator)
        )
    if cfg.initialization_action == "zeros":
        return torch.zeros(shape, device=device, dtype=torch.float32)
    raise ValueError(f"unknown action init {cfg.initialization_action!r}")


def init_future(
    batch: int,
    num_frames: int,
    num_cameras: int,
    num_patches: int,
    future_dim: int,
    cfg: InitConfig,
    *,
    device: torch.device,
    current_dino: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
    future_stats: Optional["ZScoreStats"] = None,
) -> torch.Tensor:
    """Return Z0 with shape ``[B, F, V, P, D]``.

    ``current_dino`` is ``[B, V, P, D]`` (or ``[B, P, D]``) and is only used by
    ``initialization_future='current_dino_noise'``.  Under the paper profile
    pass ``future_stats`` so the raw current DINO base is standardized into the
    same coordinate the solver optimises in before the noise is added.
    """
    shape = (batch, num_frames, num_cameras, num_patches, future_dim)
    if cfg.initialization_future == "normal":
        return (
            cfg.init_future_mean
            + cfg.init_future_std
            * torch.randn(shape, device=device, dtype=torch.float32, generator=generator)
        )
    if cfg.initialization_future == "current_dino_noise":
        if current_dino is None:
            raise ValueError("initialization_future='current_dino_noise' requires current_dino features")
        cd = current_dino
        if cd.dim() == 3:  # [B, P, D] -> [B, 1, P, D]
            cd = cd.unsqueeze(1)
        if cd.shape[0] != batch:
            raise ValueError(f"current_dino batch {cd.shape[0]} != {batch}")
        if cd.shape[-1] != future_dim:
            raise ValueError(f"current_dino dim {cd.shape[-1]} != future_dim {future_dim}")
        # broadcast the current frame across the F future time slots
        base = cd.unsqueeze(1).expand(batch, num_frames, cd.shape[1], cd.shape[2], cd.shape[3])
        if base.shape[2] != num_cameras or base.shape[3] != num_patches:
            raise ValueError(
                f"current_dino geometry {tuple(base.shape[2:4])} != ({num_cameras}, {num_patches})"
            )
        if future_stats is not None:
            # paper profile: raw current DINO -> standardized space via the
            # FUTURE statistics, then expand + configured noise.  The future
            # ground truth is never used here.
            base = future_stats.normalize(base.to(torch.float32))
        noise = cfg.init_future_noise_std * torch.randn(
            shape, device=device, dtype=torch.float32, generator=generator
        )
        return base.to(torch.float32) + noise
    raise ValueError(f"unknown future init {cfg.initialization_future!r}")


def init_candidates(
    cfg: InitConfig,
    *,
    batch: int,
    action_horizon: int,
    action_dim: int,
    num_frames: int,
    num_cameras: int,
    num_patches: int,
    future_dim: int,
    device: torch.device,
    current_dino: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
    future_stats: Optional["ZScoreStats"] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    A0 = init_action(
        batch, action_horizon, action_dim, cfg, device=device, generator=generator
    )
    Z0 = init_future(
        batch,
        num_frames,
        num_cameras,
        num_patches,
        future_dim,
        cfg,
        device=device,
        current_dino=current_dino,
        generator=generator,
        future_stats=future_stats,
    )
    return A0, Z0

"""Action/state normalisation — reuses the StarVLA / GR00T LeRobot contract.

The formulas below are copied verbatim (in meaning) from
``starVLA/dataloader/gr00t_lerobot/transform/state_action.py`` so EnergyWAM
operates in exactly the same normalised space as the StarVLA frameworks it is
meant to sit beside:

``min_max``   -> ``2 * (x - min) / (max - min) - 1``, range [-1, 1];
                where ``min == max`` the output is 0.
``mean_std``  -> ``(x - mean) / std``; where ``std == 0`` the value is kept.
``q99``       -> 2 * (x - q01) / (q99 - q01) - 1
``binary``    -> ``(x > threshold)``

Inverse transforms follow the same source.  Nothing here re-derives statistics;
they are read from the dataset's ``meta/stats.json`` (or a supplied dict).

``zscore``    -> ``(x - mean) / (std + epsilon)`` — the paper §4.8 standardized
                coordinate (``normalization_profile='paper_standardized_v1'``).
                The inverse is the exact ``x = normalized * (std + epsilon) +
                mean``.  ``epsilon`` is explicit (default 1e-6) and is what
                makes zero-variance dimensions finite and invertible.  This
                mode is only produced from a fitted
                :class:`energywam.data.standardization.NormalizationBundle`
                (training-split statistics); it is never fitted here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import torch


MODES = ("min_max", "mean_std", "q99", "binary", "zscore", "none")


@dataclass
class NormStats:
    mode: str
    min: Optional[np.ndarray] = None
    max: Optional[np.ndarray] = None
    mean: Optional[np.ndarray] = None
    std: Optional[np.ndarray] = None
    q01: Optional[np.ndarray] = None
    q99: Optional[np.ndarray] = None
    binary_threshold: float = 0.0
    version: str = "starvla_lerobot_minmax_v1"
    #: only used by ``mode='zscore'``: the paper §4.8 ``(std + epsilon)`` divisor
    epsilon: float = 0.0

    def to(self, x: Union[torch.Tensor, np.ndarray]):
        """Return a copy of the stats on the target's container/device."""
        def conv(a):
            if a is None:
                return None
            if isinstance(x, torch.Tensor):
                return torch.as_tensor(a, device=x.device, dtype=x.dtype)
            return np.asarray(a, dtype=np.asarray(x).dtype)

        return NormStats(
            mode=self.mode,
            min=conv(self.min),
            max=conv(self.max),
            mean=conv(self.mean),
            std=conv(self.std),
            q01=conv(self.q01),
            q99=conv(self.q99),
            binary_threshold=self.binary_threshold,
            version=self.version,
            epsilon=self.epsilon,
        )


def _zscore_params(stats: NormStats):
    """Validate and return ``(mean64, denom64)`` for the zscore coordinate."""
    if stats.mean is None or stats.std is None:
        raise ValueError("zscore normalisation requires mean and std statistics")
    if not (float(stats.epsilon) > 0):
        raise ValueError(
            f"zscore normalisation requires an explicit epsilon > 0, got {stats.epsilon}"
        )
    mean64 = np.asarray(stats.mean, dtype=np.float64)
    denom64 = np.asarray(stats.std, dtype=np.float64) + float(stats.epsilon)
    return mean64, denom64


def normalize(x, stats: NormStats):
    """Apply the StarVLA normalisation contract.

    Works on ``torch.Tensor`` or ``np.ndarray``; preserves the input container.
    """
    is_np = isinstance(x, np.ndarray)
    xt = torch.as_tensor(x) if is_np else x
    s = stats.to(xt)
    if s.mode == "none":
        out = xt
    elif s.mode == "min_max":
        mn, mx = s.min, s.max
        mask = mn != mx
        out = torch.zeros_like(xt)
        out[..., mask] = (xt[..., mask] - mn[..., mask]) / (mx[..., mask] - mn[..., mask])
        out[..., mask] = 2 * out[..., mask] - 1
        out[..., ~mask] = 0
    elif s.mode == "mean_std":
        mean, std = s.mean, s.std
        mask = std != 0
        out = torch.zeros_like(xt)
        out[..., mask] = (xt[..., mask] - mean[..., mask]) / std[..., mask]
        out[..., ~mask] = xt[..., ~mask].to(xt.dtype)
    elif s.mode == "zscore":
        # paper §4.8 Eq. (37)/(39).  No special-casing of zero-variance dims:
        # std + epsilon is always > 0, so the map is finite and exactly
        # invertible (a constant column maps to a constant, not to "raw").
        # float64 core, cast back to the input dtype — identical to
        # ``ZScoreStats.normalize`` so both paths share one coordinate.
        mean64, denom64 = _zscore_params(stats)
        mean64 = torch.as_tensor(mean64, device=xt.device, dtype=torch.float64)
        denom64 = torch.as_tensor(denom64, device=xt.device, dtype=torch.float64)
        out = ((xt.double() - mean64) / denom64).to(xt.dtype)
    elif s.mode == "q99":
        q01, q99 = s.q01, s.q99
        mask = q99 != q01
        out = torch.zeros_like(xt)
        out[..., mask] = 2 * (xt[..., mask] - q01[..., mask]) / (q99[..., mask] - q01[..., mask]) - 1
        out[..., ~mask] = 0
    elif s.mode == "binary":
        out = (xt > s.binary_threshold).to(xt.dtype)
    else:
        raise ValueError(f"unknown normalisation mode {s.mode!r}")
    return out.numpy() if is_np else out


def unnormalize(x, stats: NormStats):
    """Inverse of :func:`normalize`, matching the StarVLA inverse transform."""
    is_np = isinstance(x, np.ndarray)
    xt = torch.as_tensor(x) if is_np else x
    s = stats.to(xt)
    if s.mode == "none":
        out = xt
    elif s.mode == "min_max":
        mn, mx = s.min, s.max
        out = (xt + 1) / 2 * (mx - mn) + mn
    elif s.mode == "mean_std":
        out = xt * s.std + s.mean
    elif s.mode == "zscore":
        # exact inverse of the zscore forward map (epsilon included)
        mean64, denom64 = _zscore_params(stats)
        mean64 = torch.as_tensor(mean64, device=xt.device, dtype=torch.float64)
        denom64 = torch.as_tensor(denom64, device=xt.device, dtype=torch.float64)
        out = (xt.double() * denom64 + mean64).to(xt.dtype)
    elif s.mode == "q99":
        out = (xt + 1) / 2 * (s.q99 - s.q01) + s.q01
    elif s.mode == "binary":
        out = (xt > s.binary_threshold).to(xt.dtype)
    else:
        raise ValueError(f"unknown normalisation mode {s.mode!r}")
    return out.numpy() if is_np else out


def load_stats_from_lerobot_meta(
    stats_json: Union[str, Path],
    key: str,
    mode: str = "min_max",
) -> NormStats:
    """Read ``meta/stats.json`` from a LeRobot dataset and build a :class:`NormStats`."""
    data = json.loads(Path(stats_json).read_text())
    if key not in data:
        raise KeyError(f"key {key!r} not in stats file; available: {sorted(data)[:20]}")
    block = data[key]
    return NormStats(
        mode=mode,
        min=np.asarray(block["min"], dtype="float32") if "min" in block else None,
        max=np.asarray(block["max"], dtype="float32") if "max" in block else None,
        mean=np.asarray(block["mean"], dtype="float32") if "mean" in block else None,
        std=np.asarray(block["std"], dtype="float32") if "std" in block else None,
        q01=np.asarray(block.get("q01", block.get("min")), dtype="float32") if ("q01" in block or "min" in block) else None,
        q99=np.asarray(block.get("q99", block.get("max")), dtype="float32") if ("q99" in block or "max" in block) else None,
    )

"""Deterministic synthetic data for the explicit tiny test mode.

Nothing here is used in real mode.  All tensors are generated from a seeded
``torch.Generator`` so runs are reproducible.

The paper-profile (``standardization_profile='paper_standardized_v1'``) tiny
path goes through a **raw synthetic distribution** that is then standardized
with a fitted statistics bundle (see :func:`synthetic_train_bundle`), so the
standardization math is exercised exactly as on a real dataset.  Everything in
this module is explicitly synthetic — a bundle built here is marked
``coverage.kind='synthetic'`` and never presented as real statistics.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch

from ..config import EnergyWAMConfig
from .standardization import (
    DEFAULT_EPSILON,
    BundleFitter,
    NormalizationBundle,
    ZScoreStats,
)

# ---------------------------------------------------------------------------
# synthetic RAW distribution (the "physical units" of the tiny world)
# ---------------------------------------------------------------------------
#: per-dimension offsets/scales of the synthetic raw action / future features.
#: Fixed numbers so a hand-computed test can state the expected mean/std.
TINY_RAW_ACTION_MEAN = 0.7
TINY_RAW_ACTION_SCALE = 2.5
TINY_RAW_FUTURE_MEAN = -1.3
TINY_RAW_FUTURE_SCALE = 3.0


def _raw_scale(base: float, dim: int) -> torch.Tensor:
    """Deterministic per-dimension raw scale: ``base * (1 + 0.1*d)``."""
    return base * (1.0 + 0.1 * torch.arange(dim, dtype=torch.float64))


def make_tiny_raw_batch(
    cfg: EnergyWAMConfig,
    batch_size: int = 2,
    *,
    seed: int = 0,
    include_padding: bool = False,
) -> Dict[str, torch.Tensor]:
    """Raw-space synthetic batch: action/future in synthetic physical units.

    The action raw mean is ``TINY_RAW_ACTION_MEAN`` per dim with a per-dim raw
    scale ``TINY_RAW_ACTION_SCALE * (1 + 0.1 d)``; the future features use the
    ``TINY_RAW_FUTURE_*`` numbers.  Masks / offsets follow :func:`make_tiny_batch`.
    """
    b = cfg.backbone
    g = torch.Generator(device="cpu").manual_seed(int(seed))

    H, Da = cfg.data.action_horizon, b.tiny_action_dim
    F, V, P, D = (
        cfg.data.num_future_frames,
        b.tiny_num_cameras,
        b.tiny_future_patches,
        b.tiny_future_dim,
    )
    a_scale = _raw_scale(TINY_RAW_ACTION_SCALE, Da)
    z_scale = _raw_scale(TINY_RAW_FUTURE_SCALE, D)
    A = TINY_RAW_ACTION_MEAN + a_scale * torch.randn(batch_size, H, Da, generator=g)
    Z = TINY_RAW_FUTURE_MEAN + z_scale * torch.randn(batch_size, F, V, P, D, generator=g)

    action_mask = torch.ones(batch_size, H, Da, dtype=torch.bool)
    future_mask = torch.ones(batch_size, F, V, P, dtype=torch.bool)
    if include_padding:
        action_mask[0, -2:, :] = False
        future_mask[1, -1, :, :] = False

    cond_dim = b.tiny_state_dim + b.tiny_action_dim
    cond_vec = torch.randn(batch_size, cond_dim, generator=g)
    offsets = torch.tensor([resolve_offsets(cfg)] * batch_size, dtype=torch.long)
    return {
        "cond_vec": cond_vec,
        "raw_action_target": A,
        "raw_future_target": Z,
        "action_mask": action_mask,
        "future_mask": future_mask,
        "future_offsets": offsets,
    }


def synthetic_train_bundle(
    cfg: EnergyWAMConfig,
    *,
    n_windows: int = 256,
    seed: int = 1234,
    epsilon: float = DEFAULT_EPSILON,
    include_future: bool = True,
    extra_provenance: Optional[Dict] = None,
) -> NormalizationBundle:
    """Fit a statistics bundle on a synthetic **train split** of raw windows.

    Marked ``provenance.synthetic = True`` and
    ``coverage.kind = 'synthetic_train_split'`` — this is test/smoke material,
    never a real dataset statistic.  The fitting code is the same
    :class:`BundleFitter` the offline CLI uses, including mask handling.
    ``extra_provenance`` updates the provenance dict (e.g. to align the recorded
    DINO identity with a real checkpoint in the real-path tests).
    """
    from ..config import geometry_from_cfg

    geom = geometry_from_cfg(cfg)
    Da, D = geom["action_dim"], geom["future_dim"]
    from .standardization import select_cameras

    cams_used = select_cameras(
        cfg.data.camera_order, cfg.data.future_camera_index, cfg.data.num_cameras
    ) or []
    prov = {
        "synthetic": True,
        "dataset_identity": "synthetic_tiny/train",
        "split": "train",
        "train_episodes": {
            "episode_indices": [0, 1],
            "manifest_sha256": "synthetic",
            "manifest_path": "synthetic",
            "num_episodes": 2,
            "synthetic": True,
        },
        "camera_order": list(cfg.data.camera_order) if cfg.data.camera_order else [],
        "cameras_used": cams_used,
        "camera_index": int(cfg.data.future_camera_index),
        "num_cameras": int(cfg.data.num_cameras),
        "action_horizon": cfg.data.action_horizon,
        "future_offsets": [
            int(x) for x in (cfg.data.future_offsets or resolve_offsets(cfg))
        ],
        "num_future_frames": cfg.data.num_future_frames,
        "dino_preprocess_version": cfg.data.dino_preprocess_version,
        "dino": {
            "model_path": "synthetic_tiny_future_features",
            "config_hash": "synthetic",
            "hidden_dim": D,
        },
        "dims": {"action_dim": Da, "future_dim": D},
    }
    prov.update(extra_provenance or {})
    fitter = BundleFitter(
        action_dim=Da,
        future_dim=D if include_future else None,
        epsilon=epsilon,
        provenance=prov,
    )
    per = max(1, min(8, n_windows))
    done = 0
    while done < n_windows:
        bsz = min(per, n_windows - done)
        raw = make_tiny_raw_batch(cfg, batch_size=bsz, seed=seed + done, include_padding=True)
        fitter.update_action(
            raw["raw_action_target"].numpy(), mask=raw["action_mask"].numpy()
        )
        if include_future:
            fitter.update_future(
                raw["raw_future_target"].numpy(), mask=raw["future_mask"].numpy()
            )
        done += bsz
    return fitter.finalize(
        coverage={
            "kind": "synthetic_train_split",
            "num_windows": int(n_windows),
            "note": "synthetic tiny data; NOT a real dataset statistic",
        }
    )


def make_tiny_batch(
    cfg: EnergyWAMConfig,
    batch_size: int = 2,
    *,
    seed: int = 0,
    device: str | torch.device = "cpu",
    variable_cond_len: bool = False,
    include_padding: bool = False,
    standardizer: Optional[NormalizationBundle] = None,
) -> Dict[str, torch.Tensor]:
    """Build a synthetic batch with the tiny geometry.

    Returns a dict with ``cond_vec``, ``action_target``, ``future_target``,
    ``action_mask``, ``future_mask``, ``future_offsets`` and the raw fields the
    tiny condition encoder needs.

    Without ``standardizer`` the targets are the historical synthetic values
    (standard normal in the model space — the legacy tiny condition).  With a
    :class:`NormalizationBundle` the targets are drawn from
    :func:`make_tiny_raw_batch` and then standardized into the paper §4.8
    coordinate, with ``action_target_space`` / ``future_target_space`` markers.
    """
    b = cfg.backbone
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    device = torch.device(device)

    if standardizer is not None:
        raw = make_tiny_raw_batch(cfg, batch_size=batch_size, seed=seed, include_padding=include_padding)
        A = standardizer.action.normalize(raw["raw_action_target"].to(torch.float32))
        Z = standardizer.future.normalize(raw["raw_future_target"].to(torch.float32))
        A = torch.where(raw["action_mask"], A, torch.zeros_like(A))
        Z = torch.where(raw["future_mask"].unsqueeze(-1), Z, torch.zeros_like(Z))
        out = {
            "cond_vec": raw["cond_vec"].to(device),
            "action_target": A.to(device),
            "future_target": Z.to(device),
            "action_mask": raw["action_mask"].to(device),
            "future_mask": raw["future_mask"].to(device),
            "future_offsets": raw["future_offsets"].to(device),
            "action_target_space": "standardized",
            "future_target_space": "standardized",
        }
        return out

    cond_dim = b.tiny_state_dim + b.tiny_action_dim
    cond_vec = torch.randn(batch_size, cond_dim, generator=g)

    A = torch.randn(batch_size, cfg.data.action_horizon, b.tiny_action_dim, generator=g)
    Z = torch.randn(
        batch_size,
        cfg.data.num_future_frames,
        b.tiny_num_cameras,
        b.tiny_future_patches,
        b.tiny_future_dim,
        generator=g,
    )

    action_mask = torch.ones(batch_size, cfg.data.action_horizon, b.tiny_action_dim, dtype=torch.bool)
    future_mask = torch.ones(
        batch_size,
        cfg.data.num_future_frames,
        b.tiny_num_cameras,
        b.tiny_future_patches,
        dtype=torch.bool,
    )
    if include_padding:
        # mask out the tail of the action chunk for sample 0 and the last future
        # frame of sample 1, to exercise the masking paths
        action_mask[0, -2:, :] = False
        future_mask[1, -1, :, :] = False

    offsets = torch.tensor(
        [resolve_offsets(cfg)] * batch_size, dtype=torch.long
    )

    return {
        "cond_vec": cond_vec.to(device),
        "action_target": A.to(device),
        "future_target": Z.to(device),
        "action_mask": action_mask.to(device),
        "future_mask": future_mask.to(device),
        "future_offsets": offsets.to(device),
    }


def resolve_offsets(cfg: EnergyWAMConfig):
    from ..config import resolve_future_offsets

    return resolve_future_offsets(cfg.data)


def tiny_model_inputs(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Strip helper keys, keep only what the model consumes."""
    keep = ("cond_vec", "action_target", "future_target", "action_mask", "future_mask", "future_offsets")
    return {k: v for k, v in batch.items() if k in keep}

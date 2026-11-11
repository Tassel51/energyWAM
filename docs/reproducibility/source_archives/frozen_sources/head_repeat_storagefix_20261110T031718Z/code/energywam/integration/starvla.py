"""Explicit StarVLA integration for ``QwenEnergyWAM``.

StarVLA registers frameworks in ``starVLA.model.tools.FRAMEWORK_REGISTRY`` and
builds them through ``starVLA.model.framework.base_framework.build_framework``.
Because EnergyWAM must not modify ``../starVLA``, this module registers the
framework from the EnergyWAM side and adapts the two interfaces the trainer and
the model server actually use:

``compute_loss('vla', batch)`` -> dict containing ``action_loss``
``predict_action(examples)``   -> ``{"normalized_actions": np.ndarray[B, H, Da]}``

Nothing here is imported unless you ask for it (``import
energywam.integration.starvla``), so existing StarVLA frameworks are unaffected.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _starvla_root() -> Optional[Path]:
    """Locate the read-only starVLA snapshot (never written to)."""
    cand = _ROOT.parent / "starVLA"
    return cand if cand.exists() else None


def ensure_starvla_on_path() -> Optional[Path]:
    root = _starvla_root()
    if root is not None and str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


class EnergyWAMFrameworkAdapter(nn.Module):
    """A real ``nn.Module`` that duck-types the StarVLA ``baseframework`` surface.

    It subclasses ``nn.Module`` (not a plain object) so the StarVLA trainer's
    ``named_parameters`` / ``modules`` / ``to`` / ``state_dict`` / ``train``
    work exactly as they do for the in-tree frameworks.  The EnergyWAM model is
    a registered submodule named ``model``.
    """

    def __init__(self, model, cfg):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.norm_stats: Dict[str, Any] = {}

    # --- StarVLA trainer interface ------------------------------------
    def supports_training_tag(self, tag: str) -> bool:
        return tag == "vla"

    def compute_loss(self, tag: str, batch, loss_scale: Optional[dict] = None):
        """``batch`` is the ``List[dict]`` produced by StarVLA's DataLoader.

        It is converted to an EnergyWAM batch here — the underlying model must
        never be handed a raw ``list[dict]``.  Future supervision is accepted
        either as pre-computed ``future_latent`` / ``future_target`` features or
        as raw ``future_images`` (the model extracts frozen DINO features).

        Coordinate contract: under ``normalization_profile='paper_standardized_v1'``
        every hand-built ``future_latent`` / ``future_target`` must carry an
        explicit ``future_target_space`` marker ('raw_dino' or 'standardized'),
        and ``action`` must be in the model space (the standardised action
        coordinate).  Without a marker the model refuses to guess (that is the
        guard against double normalisation).
        """
        if tag != "vla":
            return None
        if isinstance(batch, (list, tuple)):
            batch = self.model._examples_to_batch(list(batch), require_targets=True)
        out = self.model(batch)
        scale = (loss_scale or {}).get(tag, 1.0)
        return {k: v * scale for k, v in out.items() if isinstance(v, torch.Tensor)}

    # --- StarVLA model-server interface -------------------------------
    def predict_action(self, examples=None, unnorm_key: Optional[str] = None, **kwargs):
        if examples is None:
            raise ValueError("predict_action requires `examples`")
        if isinstance(examples, dict):
            examples = [examples]
        out = self.model.predict_action(list(examples), **kwargs)
        return out

    def unnormalize_actions(self, norm_actions, unnorm_key: Optional[str] = None):
        from ..data.normalization import unnormalize

        key = unnorm_key or next(iter(self.norm_stats), None)
        if key is None:
            raise ValueError("no norm_stats attached; cannot un-normalize")
        return unnormalize(norm_actions, self.norm_stats[key])

def build_energywam(cfg) -> EnergyWAMFrameworkAdapter:
    """Build an EnergyWAM model from either an EnergyWAMConfig or a StarVLA cfg."""
    from ..config import EnergyWAMConfig, config_from_dict
    from ..models.energy_wam import QwenEnergyWAM

    if isinstance(cfg, EnergyWAMConfig):
        ew = cfg
    else:
        # a StarVLA / OmegaConf-style config: try to read our subtree
        try:
            from omegaconf import OmegaConf

            d = OmegaConf.to_container(cfg, resolve=False) if OmegaConf.is_config(cfg) else dict(cfg)
        except Exception:
            d = dict(cfg)
        sub = d.get("energywam") or d.get("framework") or d
        ew = config_from_dict(sub)
    return EnergyWAMFrameworkAdapter(QwenEnergyWAM(ew), ew)


def register_with_starvla(name: str = "QwenEnergyWAM") -> bool:
    """Register the EnergyWAM builder in StarVLA's ``FRAMEWORK_REGISTRY``.

    Returns True when the registration happened, False when the StarVLA
    snapshot is not available.  Registration is explicit and side-effect-free
    for other frameworks (they keep their own entries).
    """
    root = ensure_starvla_on_path()
    if root is None:
        return False
    try:
        from starVLA.model.tools import FRAMEWORK_REGISTRY
    except Exception:
        return False

    def _builder(cfg):
        return build_energywam(cfg)

    FRAMEWORK_REGISTRY.register(name)(_builder)
    return True


def attach_norm_stats(adapter: EnergyWAMFrameworkAdapter, stats: Dict[str, Any]) -> None:
    """Attach dataset normalisation statistics for action un-normalisation.

    The keys must match the StarVLA ``unnorm_key`` convention so the eval bridge
    can un-normalise predictions exactly the way the baselines do.
    """
    adapter.norm_stats = dict(stats)

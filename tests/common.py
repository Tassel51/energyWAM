"""Shared fixtures for the EnergyWAM test suite (tiny mode only)."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict

import torch

ROOT = Path(__file__).resolve().parents[1]


def tiny_cfg(**overrides):
    from energywam.config import load_config

    cfg = load_config(ROOT / "configs" / "tiny_cpu.yaml")
    for k, v in overrides.items():
        node = cfg
        parts = k.split(".")
        for p in parts[:-1]:
            node = getattr(node, p)
        setattr(node, parts[-1], v)
    return cfg


def tiny_model(cfg=None, seed: int = 0):
    from energywam.models.energy_wam import QwenEnergyWAM

    if cfg is None:
        cfg = tiny_cfg()
    torch.manual_seed(seed)
    m = QwenEnergyWAM(cfg)
    return m, cfg


def tiny_batch(cfg, batch_size=2, seed=0, include_padding=False, **kw):
    from energywam.data.tiny_data import make_tiny_batch

    return make_tiny_batch(cfg, batch_size=batch_size, seed=seed, include_padding=include_padding, **kw)


def clone_state(m):
    return copy.deepcopy(m.state_dict())


def params_with_grad(model):
    out = {}
    for n, p in model.named_parameters():
        if p.grad is not None and torch.isfinite(p.grad).all() and float(p.grad.abs().sum()) > 0:
            out[n] = p.grad.detach().clone()
    return out

#!/usr/bin/env python
"""EnergyWAM websocket policy server — real StarVLA protocol.

Uses ``deployment/model_server/tools/websocket_policy_server.py`` from the
read-only ``../starVLA`` snapshot.  The handshake metadata advertises
``action_chunk_size`` / ``available_unnorm_keys`` / ``action_keys`` /
``state_keys``; ``{"type": "infer", "payload": {"examples": [...]}}`` returns
``{"actions": B,T,D, "normalized_actions": ...}``.

Action statistics are rebuilt from the checkpoint's ``meta.norm_stats`` (the
block ``EnergyWAMTrainer.save`` writes) and handed to
``EnergyWAMPredictor(action_stats=...)``, so served ``actions`` are truly
un-normalized raw actions.  With the **real** backend and no statistics this
entry REFUSES to serve (rather than silently returning normalized outputs and
claiming they are raw); the tiny backend may serve without statistics but the
metadata then explicitly marks ``action_units="normalized"`` /
``physical_eval_valid=false``.

Default bind is **127.0.0.1** — nothing is exposed externally unless you pass
``--host`` explicitly.

Usage::

    python scripts/serve.py --config configs/tiny_overfit.yaml \\
        --checkpoint logs/runs/tiny_overfit/tiny_joint.pt --port 10095
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def resolve_serve_action_stats(
    cfg,
    *,
    payload: Optional[Dict[str, Any]] = None,
    unnorm_key: Optional[str] = None,
) -> Tuple[Dict[str, Any], Optional[str], Any]:
    """Rebuild the action statistics a served policy must un-normalize with.

    Reads ONLY the checkpoint payload: for a paper-standardized checkpoint the
    embedded ``meta.normalization_bundle`` is **authoritative** (a served raw
    action is exactly ``normalized * (std + epsilon) + mean`` of that bundle);
    ``meta.norm_stats`` (see ``energywam.training.checkpoint.norm_stats_from_meta``
    — the exact inverse of what ``EnergyWAMTrainer.save`` serialises) is used
    for legacy checkpoints and cross-checked against the bundle when both are
    present.  An externally supplied statistics file is never consulted here.
    Returns ``(stats_by_key, selected_key, action_stats)``.

    * ``unnorm_key`` explicitly selects one entry; a single entry is selected
      automatically; several entries without ``--unnorm-key`` are an error.
    * With the **real** backend, no usable statistics is a hard error — the
      server must never claim raw actions it cannot produce.  The tiny backend
      may proceed without statistics (the caller then marks the output as
      normalized / physically meaningless).
    """
    from energywam.training.checkpoint import bundle_from_payload, norm_stats_from_meta

    meta = (payload or {}).get("meta", {}) or {}
    bundle = bundle_from_payload(payload)
    stats_by_key = norm_stats_from_meta(meta.get("norm_stats", {}))
    if bundle is not None:
        bundled = bundle.action.to_norm_stats()
        for k, v in stats_by_key.items():
            if v.mode != "zscore":
                raise RuntimeError(
                    f"checkpoint meta.norm_stats[{k!r}] is {v.mode!r} but the embedded "
                    "bundle is paper-standardized; the checkpoint is inconsistent"
                )
            import numpy as _np

            if not _np.allclose(_np.asarray(v.mean, "float64"), bundle.action.mean, atol=1e-6) or not (
                _np.allclose(_np.asarray(v.std, "float64"), bundle.action.std, atol=1e-6)
                and abs(float(v.epsilon) - bundle.epsilon) == 0
            ):
                raise RuntimeError(
                    f"checkpoint meta.norm_stats[{k!r}] disagrees with the embedded "
                    "normalization bundle; refusing to serve with ambiguous statistics"
                )
        if not stats_by_key:
            key_hint = str(cfg.data.data_mix or "action")
            stats_by_key = {key_hint: bundled}

    if unnorm_key is not None:
        if unnorm_key not in stats_by_key:
            raise KeyError(
                f"unnorm_key {unnorm_key!r} not in checkpoint meta.norm_stats; "
                f"available: {sorted(stats_by_key)}"
            )
        key: Optional[str] = unnorm_key
    elif len(stats_by_key) == 1:
        key = next(iter(stats_by_key))
    elif not stats_by_key:
        key = None
    else:
        raise ValueError(
            f"checkpoint carries several norm_stats keys {sorted(stats_by_key)}; "
            "pass --unnorm-key to select the one to serve."
        )

    action_stats = stats_by_key.get(key) if key else None
    if action_stats is None and cfg.backbone.backend != "tiny":
        raise RuntimeError(
            "refusing to serve the real backend without action statistics: the "
            "policy would return normalized actions while clients expect raw "
            "dataset units.  Train with EnergyWAMTrainer.save (which writes "
            "meta.norm_stats) and pass that checkpoint, or use the tiny backend "
            "explicitly for normalized-only smoke serving."
        )
    return stats_by_key, key, action_stats


def build_serve_predictor(
    cfg,
    *,
    checkpoint_path: Optional[str] = None,
    unnorm_key: Optional[str] = None,
    model=None,
):
    """Entry assembly: model + checkpoint statistics -> ``EnergyWAMPredictor``.

    The action statistics used for un-normalisation come from the checkpoint
    (``meta.normalization_bundle`` for the paper profile, ``meta.norm_stats``
    for legacy), never from an external file.  For a paper-standardized
    checkpoint the model is rebuilt from the **embedded** bundle, so serving
    keeps working even if the original statistics file is gone; an external
    bundle that disagrees with the checkpoint is rejected.  The model is placed
    on ``cfg.trainer.device``.
    """
    import torch

    from energywam.inference.predictor import EnergyWAMPredictor
    from energywam.models.energy_wam import QwenEnergyWAM
    from energywam.training.checkpoint import bundle_from_payload, load_checkpoint, restore

    payload = load_checkpoint(checkpoint_path) if checkpoint_path else None
    stats_by_key, key, action_stats = resolve_serve_action_stats(
        cfg, payload=payload, unnorm_key=unnorm_key
    )
    bundle = bundle_from_payload(payload) if payload is not None else None
    if model is None:
        model = QwenEnergyWAM(cfg, stats_bundle=bundle)
    model = model.to(torch.device(cfg.trainer.device))
    if payload is not None:
        restore(payload, model=model, mode="resume", restore_rng=False)
    # keep the full key -> NormStats map on the model (same attribute
    # EnergyWAMTrainer.save reads) so server metadata can list unnorm keys
    model.norm_stats = stats_by_key
    return EnergyWAMPredictor(model, action_stats=action_stats, unnorm_key=key or "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--host", default="127.0.0.1", help="default binds loopback only")
    ap.add_argument("--port", type=int, default=10095)
    ap.add_argument("--unnorm-key", default=None)
    ap.add_argument("--action-keys", nargs="*", default=None)
    ap.add_argument("--state-keys", nargs="*", default=None)
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()

    from energywam.config import load_config
    from energywam.inference.predictor import serve

    cfg = load_config(args.config, overrides=args.set)
    predictor = build_serve_predictor(
        cfg,
        checkpoint_path=args.checkpoint,
        unnorm_key=args.unnorm_key,
    )
    if args.checkpoint:
        print(f"[EnergyWAM serve] loaded {args.checkpoint}")
    if predictor.action_stats is not None:
        print(
            f"[EnergyWAM serve] action stats from checkpoint meta.norm_stats "
            f"(unnorm_key={predictor.unnorm_key!r}) -> served `actions` are RAW dataset units"
        )
    else:
        print(
            "[EnergyWAM serve] NO action statistics: served `actions` are NORMALIZED "
            "model outputs with no physical units (tiny/smoke only)."
        )

    server = serve(
        predictor,
        host=args.host,
        port=args.port,
        action_keys=args.action_keys,
        state_keys=args.state_keys,
        unnorm_key=args.unnorm_key,
        obs_image_size=tuple(cfg.data.obs_image_size),
    )
    print(f"[EnergyWAM serve] listening on ws://{args.host}:{args.port}")
    print(f"[EnergyWAM serve] metadata: {server._metadata}")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

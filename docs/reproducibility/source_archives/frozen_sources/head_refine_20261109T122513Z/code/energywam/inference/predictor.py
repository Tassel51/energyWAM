"""Inference / service / evaluation entry points.

What these do
-------------
* :class:`EnergyWAMPredictor` — a thin wrapper that turns raw observations into
  a normalised action chunk through joint (or configured) energy refinement.
* :func:`offline_eval` — runs the predictor over a dataset and reports
  **offline action MSE against the demonstration**.  This is explicitly NOT a
  closed-loop success rate; the two are never conflated.
* :func:`serve` — optional websocket bridge that speaks the StarVLA
  ``deployment.model_server`` protocol.  It is only constructed when the user
  asks for it, so nothing here is required for training or offline evaluation.

The engineering requirements forbid describing offline MSE as a closed-loop SR;
that rule is enforced by the naming and by the printed banner.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from ..data.normalization import NormStats, normalize, unnormalize
from ..models.energy_wam import QwenEnergyWAM


@dataclass
class PredictorOutput:
    normalized_actions: np.ndarray
    action: np.ndarray
    future_latent: Optional[np.ndarray] = None
    #: space label of ``future_latent`` (always ``raw_dino``: the raw DINO
    #: feature coordinate, unchanged from the legacy profile)
    future_latent_space: str = "raw_dino"
    #: the candidate future in the space the solver optimised in
    normalized_future_latent: Optional[np.ndarray] = None
    normalized_future_latent_space: str = ""
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    latency_s: float = 0.0


class EnergyWAMPredictor:
    """Joint energy inference over raw observations."""

    def __init__(
        self,
        model: QwenEnergyWAM,
        *,
        action_stats: Optional[NormStats] = None,
        unnorm_key: str = "",
    ):
        self.model = model
        self.model.eval()
        self.action_stats = action_stats
        self.unnorm_key = unnorm_key

    @property
    def action_space(self) -> str:
        """Human-readable coordinate of ``normalized_actions`` / ``action``."""
        if self.action_stats is None:
            return "normalized_without_stats"
        if self.action_stats.mode == "zscore":
            return f"zscore(mean/std+{self.action_stats.epsilon:g})"
        return str(self.action_stats.mode)

    @torch.inference_mode(False)
    def __call__(self, examples: List[Dict[str, Any]], *, mode: str = "joint",
                 return_diagnostics: bool = False) -> PredictorOutput:
        t0 = time.time()
        out = self.model.predict_action(examples, mode=mode, return_diagnostics=return_diagnostics)
        norm = out["normalized_actions"]
        if self.action_stats is not None:
            act = unnormalize(norm, self.action_stats)
        else:
            act = norm
        return PredictorOutput(
            normalized_actions=norm,
            action=act,
            future_latent=out.get("future_latent"),
            future_latent_space=str(out.get("future_latent_space", "raw_dino")),
            normalized_future_latent=out.get("normalized_future_latent"),
            normalized_future_latent_space=str(out.get("normalized_future_latent_space", "")),
            diagnostics=out.get("diagnostics", {}),
            latency_s=time.time() - t0,
        )


# ---------------------------------------------------------------------------
# Offline evaluation
# ---------------------------------------------------------------------------
def offline_eval(
    predictor: EnergyWAMPredictor,
    samples: List[Dict[str, Any]],
    *,
    mode: str = "joint",
    action_mask_key: str = "action_mask",
    log_path: Optional[str | Path] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Offline action MSE against the demonstration.

    Reports
    -------
    ``offline_action_mse`` : mean squared error in the model action space,
    averaged over valid elements (``min_max`` under the legacy profile,
    ``zscore`` standardized coordinates under ``paper_standardized_v1`` — the
    space is reported in ``action_space``).  **This is not a success rate** and
    must not be reported as one.  ``extra`` is merged into the report verbatim
    (e.g. the subset provenance written by ``scripts/evaluate.py``).
    """
    print("[EnergyWAM eval] OFFLINE action MSE against the demonstration.")
    print("[EnergyWAM eval] This is NOT a closed-loop RoboCasa365 success rate.")
    sq, n, per_sample = 0.0, 0, []
    for ex in samples:
        out = predictor([ex], mode=mode)
        pred = out.normalized_actions[0]
        tgt = np.asarray(ex["action"], dtype="float32")[-pred.shape[0]:]
        mask = np.asarray(ex.get(action_mask_key, np.ones_like(tgt)), dtype="float32")
        d2 = ((pred - tgt) ** 2) * mask
        valid = float(mask.sum())
        mse = float(d2.sum() / max(valid, 1.0))
        per_sample.append(mse)
        sq += float(d2.sum())
        n += int(valid)
    report = {
        "metric": "offline_action_mse",
        "is_closed_loop_success_rate": False,
        "offline_action_mse": sq / max(n, 1),
        "action_space": predictor.action_space,
        "mode": mode,
        "num_samples": len(samples),
        "per_sample_mse": per_sample,
        "note": "Offline demonstration regression error. Not a task success rate.",
    }
    if extra:
        report.update(extra)
    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(log_path).write_text(json.dumps(report, indent=2))
    return report


# ---------------------------------------------------------------------------
# Service bridge — real StarVLA websocket protocol
# ---------------------------------------------------------------------------
def build_policy_server_adapter(
    predictor: "EnergyWAMPredictor",
    *,
    action_keys: Optional[List[str]] = None,
    state_keys: Optional[List[str]] = None,
    unnorm_key: Optional[str] = None,
    obs_image_size=(224, 224),
):
    """Build the object handed to ``WebsocketPolicyServer``.

    Protocol (matches ``deployment/model_server/tools/websocket_policy_server.py``):

    * handshake sends a metadata dict (see :meth:`metadata`);
    * ``{"type": "infer", "payload": {...}}`` is dispatched to
      ``predict_action(**payload)``;
    * the reply is ``{"actions": np.ndarray[B, T, D], "normalized_actions": ...}``.
      ``actions`` is **un-normalized** iff the predictor was assembled with
      ``action_stats`` (e.g. from the checkpoint's ``meta.norm_stats``); with no
      statistics loaded it is the normalized output and the metadata
      (``action_units`` / ``physical_eval_valid``) says so explicitly.

    The server implementation is the read-only StarVLA one; EnergyWAM only
    supplies the policy object and the metadata.
    """
    geometry = predictor.model.geometry
    cfg = predictor.model.cfg

    class EnergyWAMServerPolicy:
        """Duck-types ``PolicyServerWrapper`` for ``WebsocketPolicyServer``."""

        def __init__(self):
            self._unnorm_key = unnorm_key

        @property
        def metadata(self) -> Dict[str, Any]:
            stats_keys = list(predictor.model.norm_stats.keys()) if getattr(
                predictor.model, "norm_stats", None
            ) else ([predictor.unnorm_key] if predictor.unnorm_key else [])
            has_stats = predictor.action_stats is not None
            return {
                "env": "energywam_policy_server",
                "framework": cfg.framework.name,
                "stage": cfg.stage.stage,
                "backend": cfg.backbone.backend,
                "action_chunk_size": int(geometry["action_horizon"]),
                "action_dim": int(geometry["action_dim"]),
                "available_unnorm_keys": stats_keys,
                "default_unnorm_key": self._unnorm_key or (stats_keys[0] if len(stats_keys) == 1 else None),
                "action_keys": list(action_keys or []),
                "state_keys": list(state_keys or []),
                "training_obs_image_size": list(obs_image_size),
                "future_offsets": list(cfg.data.future_offsets or []),
                # honesty about what `actions` actually contains
                "action_units": "raw_dataset_units" if has_stats else "normalized",
                "physical_eval_valid": bool(has_stats),
                "normalized_actions_also_returned": True,
                "action_units_note": (
                    "actions are un-normalized with the checkpoint's action statistics."
                    if has_stats
                    else "no action statistics loaded: `actions` are NORMALIZED "
                    "model outputs (min_max space). They have no physical units and "
                    "must not be used for physical evaluation."
                ),
                "eval_image_contract": (
                    "Eval clients must explicitly choose image count and order. "
                    "The server does not infer or reorder camera views."
                ),
            }

        def predict_action(self, examples=None, unnorm_key: Optional[str] = None, **kwargs):
            if examples is None:
                raise ValueError("payload must contain `examples`")
            if isinstance(examples, dict):
                examples = [examples]
            key = unnorm_key or self._unnorm_key
            if predictor.action_stats is not None and key and predictor.unnorm_key and key != predictor.unnorm_key:
                raise ValueError(
                    f"requested unnorm_key={key!r} but this server was assembled "
                    f"with unnorm_key={predictor.unnorm_key!r}"
                )
            out = predictor(list(examples), mode=kwargs.get("mode", "joint"))
            norm = np.asarray(out.normalized_actions, dtype="float32")  # [B, H, Da]
            # `out.action` is un-normalized iff the predictor holds action stats
            # (see EnergyWAMPredictor.__call__); without stats it is the
            # normalized output and metadata says so.
            act = np.asarray(out.action, dtype="float32")
            return {
                "actions": act,
                "normalized_actions": norm,
                "action_units": "raw_dataset_units" if predictor.action_stats is not None else "normalized",
            }

    return EnergyWAMServerPolicy()


def serve(
    predictor: "EnergyWAMPredictor",
    *,
    host: str = "127.0.0.1",
    port: int = 10095,
    action_keys: Optional[List[str]] = None,
    state_keys: Optional[List[str]] = None,
    unnorm_key: Optional[str] = None,
    obs_image_size=(224, 224),
    metadata_extra: Optional[Dict[str, Any]] = None,
):
    """Start the real StarVLA ``WebsocketPolicyServer`` with an EnergyWAM policy.

    The server class comes from the read-only ``../starVLA`` snapshot
    (``deployment/model_server/tools/websocket_policy_server.py``).  If that
    snapshot is absent this raises a clear error rather than serving a stub.

    Default host is **127.0.0.1** so nothing is exposed externally unless the
    caller explicitly asks for it.
    """
    root = Path(__file__).resolve().parents[2].parent / "starVLA"
    if root.exists() and str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from deployment.model_server.tools.websocket_policy_server import (
            WebsocketPolicyServer,
        )
    except Exception as exc:
        raise RuntimeError(
            "StarVLA WebsocketPolicyServer is not importable from "
            f"{root}.  Training and offline evaluation do not need it. "
            f"Original error: {exc}"
        ) from exc

    policy = build_policy_server_adapter(
        predictor,
        action_keys=action_keys,
        state_keys=state_keys,
        unnorm_key=unnorm_key,
        obs_image_size=obs_image_size,
    )
    metadata = dict(policy.metadata)
    if metadata_extra:
        metadata.update(metadata_extra)
    server = WebsocketPolicyServer(policy, host=host, port=port, metadata=metadata)
    return server

"""Checkpoint save / load / stage-switch policy.

A checkpoint contains everything needed to reproduce a run:

* model state dict
* outer optimiser state
* LR scheduler state
* training stage
* global step
* RNG states (python / numpy / torch / cuda)
* normalisation statistics (or an explicit version reference)
* the full resolved config

Two distinct restore modes are supported and must not be confused:

``resume``   same stage: restore model + optimiser + scheduler + step + RNG.
``switch``   bi -> joint (or any other stage change): restore the **model**,
             then apply an explicit optimiser/scheduler policy
             (``reset`` or ``continue``) which is logged.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from ..config import EnergyWAMConfig, to_plain_dict
from ..data.normalization import NormStats


@dataclass
class CheckpointMeta:
    stage: str
    global_step: int
    optimizer_policy: str = "continue"
    scheduler_policy: str = "continue"
    norm_stats_version: str = ""
    norm_stats: Dict[str, Any] = field(default_factory=dict)
    #: ``legacy`` | ``paper_standardized_v1`` — which coordinate system the
    #: candidates live in.  Empty means legacy (old checkpoints).
    normalization_profile: str = ""
    #: full paper §4.8 statistics bundle payload (provenance + fingerprint),
    #: making the checkpoint self-contained when the external bundle file is gone.
    normalization_bundle: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""


# ---------------------------------------------------------------------------
# norm-stats serialisation (the checkpoint's ``meta.norm_stats`` contract)
# ---------------------------------------------------------------------------
# ``EnergyWAMTrainer.save`` writes ``{unnorm_key: {mode, version,
# binary_threshold, min/max/mean/std/q01/q99 as lists}}``.  The inverse below is
# the ONLY sanctioned way to turn that back into :class:`NormStats` objects
# (e.g. for the policy server's action un-normalisation).  Keeping both
# directions here means save and load cannot drift apart.
_STATS_FIELDS = ("min", "max", "mean", "std", "q01", "q99")
_META_SCALAR_FIELDS = ("mode", "version", "binary_threshold", "epsilon")


def norm_stats_to_meta(raw_stats: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Serialise ``{key: NormStats}`` into the ``meta.norm_stats`` block."""
    serialisable: Dict[str, Any] = {}
    for k, v in (raw_stats or {}).items():
        block: Dict[str, Any] = {}
        for field_name in _META_SCALAR_FIELDS:
            if hasattr(v, field_name):
                block[field_name] = getattr(v, field_name)
        for field_name in _STATS_FIELDS:
            arr = getattr(v, field_name, None)
            if arr is not None:
                block[field_name] = np.asarray(arr).tolist()
        serialisable[k] = block
    return serialisable


def norm_stats_from_meta(serialised: Optional[Dict[str, Any]]) -> Dict[str, NormStats]:
    """Rebuild ``{key: NormStats}`` from a checkpoint's ``meta.norm_stats``."""
    out: Dict[str, NormStats] = {}
    for k, block in (serialised or {}).items():
        if not isinstance(block, dict):
            raise ValueError(f"meta.norm_stats[{k!r}] is not a dict: {type(block)}")
        out[k] = NormStats(
            mode=str(block.get("mode", "min_max")),
            min=np.asarray(block["min"], dtype="float32") if "min" in block else None,
            max=np.asarray(block["max"], dtype="float32") if "max" in block else None,
            mean=np.asarray(block["mean"], dtype="float32") if "mean" in block else None,
            std=np.asarray(block["std"], dtype="float32") if "std" in block else None,
            q01=np.asarray(block["q01"], dtype="float32") if "q01" in block else None,
            q99=np.asarray(block["q99"], dtype="float32") if "q99" in block else None,
            binary_threshold=float(block.get("binary_threshold", 0.0)),
            version=str(block.get("version", "")),
            epsilon=float(block.get("epsilon", 0.0)),
        )
    return out


# ---------------------------------------------------------------------------
# scheduler restore / reset helpers
# ---------------------------------------------------------------------------
def _apply_scheduler_lrs(scheduler, values: List[float]) -> None:
    """Write ``values`` into the optimiser's param groups (and ``_last_lr``).

    ``LRScheduler.load_state_dict`` only updates the scheduler's own
    attributes — the optimiser keeps whatever ``lr`` it happened to hold — so
    a resume that does not re-apply the schedule here silently trains the next
    step at the wrong learning rate.
    """
    for pg, lr in zip(scheduler.optimizer.param_groups, values):
        if isinstance(pg.get("lr"), torch.Tensor):
            pg["lr"].fill_(float(lr))
        else:
            pg["lr"] = float(lr)
    scheduler._last_lr = [float(pg["lr"]) for pg in scheduler.optimizer.param_groups]


def _schedule_lrs(scheduler) -> Optional[List[float]]:
    """Learning rates the scheduler's CURRENT state prescribes (no stepping)."""
    # LambdaLR family: closed form, no warning-prone get_lr() call needed.
    if hasattr(scheduler, "base_lrs") and hasattr(scheduler, "lr_lambdas"):
        return [
            float(b * l(scheduler.last_epoch))
            for b, l in zip(scheduler.base_lrs, scheduler.lr_lambdas)
        ]
    if getattr(scheduler, "_last_lr", None):
        return [float(v) for v in scheduler._last_lr]
    if hasattr(scheduler, "get_lr"):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return [float(v) for v in scheduler.get_lr()]
    return None


def _reset_scheduler(scheduler) -> None:
    """Return the scheduler to the initial schedule position (as after ``__init__``).

    Not just the counters: the optimiser's learning rate must actually go back
    to ``schedule(0)`` so the next step restarts the configured schedule
    (warmup included).  ``_initial_step`` is exactly what the constructor ran.
    """
    if hasattr(scheduler, "last_epoch"):
        scheduler.last_epoch = -1
    if hasattr(scheduler, "_step_count"):
        scheduler._step_count = 0
    if hasattr(scheduler, "_initial_step"):
        # reproduces __init__: last_epoch -> 0 and applies schedule(0) to the optimiser
        scheduler._initial_step()
        return
    if hasattr(scheduler, "base_lrs") and hasattr(scheduler, "optimizer"):
        for g, base in zip(scheduler.optimizer.param_groups, scheduler.base_lrs):
            g["lr"] = float(base)
    if hasattr(scheduler, "last_epoch"):
        scheduler.last_epoch = 0
    vals = _schedule_lrs(scheduler)
    if vals is not None:
        _apply_scheduler_lrs(scheduler, vals)


def _rng_state() -> Dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _cpu_byte(t):
    """RNG states are CPU ByteTensors; ``map_location`` may have moved them."""
    if isinstance(t, torch.Tensor):
        return t.to(device="cpu", dtype=torch.uint8)
    return torch.as_tensor(t, dtype=torch.uint8)


def _set_rng_state(state: Dict[str, Any]) -> None:
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(_cpu_byte(state["torch"]))
    if "cuda" in state and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all([_cpu_byte(t) for t in state["cuda"]])
        except Exception:
            pass


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    cfg: EnergyWAMConfig,
    meta: CheckpointMeta,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None and hasattr(scheduler, "state_dict") else None,
        "meta": asdict(meta),
        "config": to_plain_dict(cfg),
        "rng": _rng_state(),
        "format_version": 1,
    }
    if hasattr(model, "distributed_rng_states"):
        payload["distributed_rng_states"] = model.distributed_rng_states
    torch.save(payload, path)
    import os
    from .checkpoint_metadata import write_checkpoint_metadata
    with path.open('rb') as stream:
        os.fsync(stream.fileno())
    write_checkpoint_metadata(
        path.parent / (path.name + ".meta.json"),
        {"meta": asdict(meta), "config": to_plain_dict(cfg)},
    )
    return path


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if "model" not in payload:
        raise ValueError(f"{path} is not an EnergyWAM checkpoint (no 'model' key)")
    return payload


# ---------------------------------------------------------------------------
# normalization coordinate contract (paper §4.8)
# ---------------------------------------------------------------------------
def bundle_from_payload(payload: Dict[str, Any]) -> Optional[Any]:
    """The paper-standardized statistics bundle embedded in a checkpoint."""
    from ..data.standardization import NormalizationBundle

    meta = (payload or {}).get("meta", {}) or {}
    raw = meta.get("normalization_bundle") or {}
    if not raw:
        return None
    return NormalizationBundle.from_payload(raw)


def checkpoint_normalization_profile(payload: Dict[str, Any]) -> str:
    from ..data.standardization import PROFILE_LEGACY, PROFILE_PAPER

    meta = (payload or {}).get("meta", {}) or {}
    profile = str(meta.get("normalization_profile") or "")
    if profile:
        return profile
    # old checkpoints: no field at all -> legacy semantics
    return PROFILE_PAPER if (meta.get("normalization_bundle") or {}) else PROFILE_LEGACY


def _contract_owner(model: torch.nn.Module):
    """The module carrying the normalization contract (unwraps one wrapper level).

    ``EnergyWAMFrameworkAdapter`` (StarVLA) nests the EnergyWAM model under
    ``.model``; the contract attributes live there while the buffers are found
    by suffix in the outer ``state_dict()``.
    """
    inner = getattr(model, "model", None)
    if (
        inner is not None
        and isinstance(inner, torch.nn.Module)
        and getattr(inner, "normalization_profile", None)
    ):
        return inner
    return model


def _expected_norm_buffers(bundle) -> Dict[str, Any]:
    """The buffer tensors a paper-standardized state dict must contain."""
    out = {
        "norm_action_mean": (bundle.action.mean, torch.float32),
        "norm_action_denom": (bundle.action.denom, torch.float32),
        "norm_action_count": (bundle.action.count, torch.float64),
        "norm_epsilon": (float(bundle.epsilon), torch.float64),
    }
    if bundle.future is not None:
        out.update({
            "norm_future_mean": (bundle.future.mean, torch.float32),
            "norm_future_denom": (bundle.future.denom, torch.float32),
            "norm_future_count": (bundle.future.count, torch.float64),
        })
    return out


def _find_buffer(state: Dict[str, torch.Tensor], name: str) -> Optional[torch.Tensor]:
    """Locate a norm buffer in a possibly wrapper-prefixed state dict."""
    hits = [v for k, v in state.items() if k == name or k.endswith("." + name)]
    if len(hits) > 1:
        raise ValueError(f"multiple buffers match {name!r} in the state dict")
    return hits[0] if hits else None


def _check_buffer_values(state: Dict[str, torch.Tensor], bundle, *, where: str) -> None:
    for name, (want, dtype) in _expected_norm_buffers(bundle).items():
        got = _find_buffer(state, name)
        if got is None:
            raise ValueError(
                f"{where} is missing the {name!r} normalization buffer; a "
                "paper-standardized checkpoint/model must carry the statistics "
                "as buffers (self-contained)."
            )
        expect = torch.as_tensor(want, dtype=dtype).cpu()
        got_cpu = got.detach().cpu().to(dtype)
        if got_cpu.shape != expect.shape or not torch.equal(got_cpu, expect):
            raise ValueError(
                f"{where} {name!r} buffer does not match the embedded statistics "
                "bundle (mean/denom/count/epsilon mismatch); the checkpoint was "
                "modified or is inconsistent."
            )


def verify_normalization_contract(payload: Dict[str, Any], model: torch.nn.Module) -> None:
    """Reject any coordinate/statistics mismatch BEFORE the state dict loads.

    * legacy checkpoint  <-> paper model (and vice versa) is a hard error: the
      two coordinate systems are silently incomparable and no automatic
      migration is attempted;
    * a paper checkpoint carries the full bundle; the model's embedded bundle
      (from its config file or its own checkpoint) must match it exactly
      (fingerprint over mean/std/count/epsilon/provenance);
    * the ``norm_*`` buffers inside the checkpoint's ``model`` state dict must
      match the embedded bundle (wrapper prefixes supported) — a tampered
      buffer cannot ride along on an untouched fingerprint;
    * the model's own registered buffers must agree with that bundle.
    """
    from ..data.standardization import PROFILE_LEGACY, PROFILE_PAPER, StatsError

    ckpt_profile = checkpoint_normalization_profile(payload)
    owner = _contract_owner(model)
    model_profile = str(getattr(owner, "normalization_profile", "") or PROFILE_LEGACY)
    model_bundle = getattr(owner, "stats_bundle", None)
    if ckpt_profile != model_profile:
        raise ValueError(
            f"normalization coordinate mismatch: checkpoint is {ckpt_profile!r} but the "
            f"model is {model_profile!r}.  A legacy checkpoint cannot be loaded as "
            "paper-standardized (or the reverse); retrain or rebuild the model with the "
            "matching data.normalization_profile.  No automatic migration is performed."
        )
    if ckpt_profile == PROFILE_LEGACY:
        if model_bundle is not None:
            raise ValueError("legacy model unexpectedly carries a statistics bundle")
        return
    ckpt_bundle = bundle_from_payload(payload)
    if ckpt_bundle is None:
        raise ValueError(
            "paper-standardized checkpoint has no embedded statistics bundle; it cannot "
            "be restored self-containedly.  Re-train with this build."
        )
    if model_bundle is None:
        raise StatsError(
            "restoring a paper-standardized checkpoint into a model without statistics; "
            "build the model from the checkpoint's embedded bundle "
            "(QwenEnergyWAM(cfg, stats_bundle=bundle_from_payload(payload)))"
        )
    if ckpt_bundle.fingerprint != model_bundle.fingerprint:
        raise ValueError(
            "statistics bundle mismatch between checkpoint and model: "
            f"checkpoint fingerprint {ckpt_bundle.fingerprint[:12]}... vs model "
            f"{model_bundle.fingerprint[:12]}....  Resume / stage switches require the "
            "same coordinate and the same training statistics; refusing to continue."
        )
    # the BUFFERS INSIDE THE PAYLOAD must match the embedded bundle (P1): the
    # payload is what load_state_dict will actually write into the model.
    _check_buffer_values(payload.get("model", {}), ckpt_bundle, where="checkpoint")
    # and the model's registered buffers must match too (self-consistency).
    _check_buffer_values(dict(model.state_dict()), model_bundle, where="model")


def restore(
    payload: Dict[str, Any],
    *,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    mode: str = "resume",
    optimizer_policy: Optional[str] = None,
    scheduler_policy: Optional[str] = None,
    restore_rng: bool = True,
) -> CheckpointMeta:
    """Restore a checkpoint into ``model`` (and optionally optimiser/scheduler).

    ``mode='resume'``  — same stage: optimiser/scheduler/RNG/step restored.
    ``mode='switch'``  — stage change: model restored; optimiser/scheduler are
                         reset or kept according to the *explicit* policy, which
                         is recorded in the returned :class:`CheckpointMeta`.
    """
    if payload.get("reference_only") and optimizer is not None and mode == "resume":
        raise ValueError("reference-only model has no optimizer/master state: exact resume is unavailable; use an explicit reset warm-start policy")
    meta_d = payload.get("meta", {})
    meta = CheckpointMeta(**{k: v for k, v in meta_d.items() if k in CheckpointMeta.__dataclass_fields__})

    # coordinate/statistics contract first: a legacy <-> standardized mismatch
    # must fail with a clear error, never be silently absorbed by the load.
    # Non-tensor architecture settings (notably PE scale) must also match.
    from ..config import EnergyWAMConfig
    current_cfg = getattr(model, "cfg", None)
    saved_cfg = payload.get("config", {})
    defaults = EnergyWAMConfig()
    if current_cfg is not None:
        if mode == "resume" and optimizer is not None:
            # This runner seeks data/candidate RNG by global_step * batch.
            # Changing the batch or seed would silently skip/repeat samples.
            for name in ("per_device_batch_size", "seed", "warmup_steps", "fixed_candidate_initialization", "shared_microbatch_size"):
                old = saved_cfg.get("trainer", {}).get(name, getattr(defaults.trainer, name))
                new = getattr(current_cfg.trainer, name)
                if old != new:
                    raise ValueError(f"exact training resume mismatch: trainer.{name}: {old!r} != {new!r}; use a fresh, explicit warm-start experiment")
        fields = {"backbone": ["condition_mode"],
                  "framework": ["dino_patch_pack_factor", "quadratic_candidate_prior", "quadratic_action_weight", "quadratic_future_weight", "energy_head_zero_init", "projector_output_scale", "positional_encoding_scale", "type_embedding_init_std",
                                "iteration_conditioning", "iteration_embedding_dim",
                                "iteration_mlp_hidden", "iteration_progress_scale"],
                  "solver": ["learnable_step_sizes", "candidate_gradient_normalization", "gradient_rms_epsilon"]}
        if current_cfg.solver.learnable_step_sizes:
            fields['solver'] += ['step_size_min_ratio', 'step_size_max_ratio']
        if current_cfg.solver.learnable_step_sizes or current_cfg.framework.quadratic_candidate_prior:
            fields['solver'] += ['alpha_action', 'alpha_future']
        for section, names in fields.items():
            for name in names:
                old = saved_cfg.get(section, {}).get(name, getattr(getattr(defaults, section), name))
                new = getattr(getattr(current_cfg, section), name)
                if old != new:
                    raise ValueError(f"checkpoint architecture mismatch: {section}.{name}: {old!r} != {new!r}")
    verify_normalization_contract(payload, model)

    # strict=True: a checkpoint that does not match the network exactly is an
    # error, not a warning.  A stage switch on the SAME network (bi -> joint)
    # needs no laxness because the parameter set is identical.
    model.load_state_dict(payload["model"], strict=True)

    if mode == "resume":
        opt_pol = optimizer_policy or meta.optimizer_policy or "continue"
        sch_pol = scheduler_policy or meta.scheduler_policy or "continue"
    elif mode == "switch":
        opt_pol = optimizer_policy or "reset"
        sch_pol = scheduler_policy or "reset"
    else:
        raise ValueError(f"unknown restore mode {mode!r}")

    if optimizer is not None:
        if opt_pol == "continue" and payload.get("optimizer") is not None:
            optimizer.load_state_dict(payload["optimizer"])
        elif opt_pol == "reset":
            # Genuinely clear any state already present (AdamW moments etc.).
            # A reused optimiser would otherwise silently carry moments over
            # from a previous stage.  ``clear()`` keeps the container type —
            # AdamW indexes it as a defaultdict and a plain dict breaks that.
            optimizer.state.clear()
            if hasattr(optimizer, "sync_masters_from_model"):
                optimizer.sync_masters_from_model()
        else:
            raise ValueError(f"unknown optimizer_policy {opt_pol!r}")

    if scheduler is not None and hasattr(scheduler, "state_dict"):
        if sch_pol == "continue":
            if payload.get("scheduler") is None:
                raise ValueError(
                    "scheduler_policy='continue' but this checkpoint has no scheduler "
                    "state to continue from.  Use scheduler_policy='reset' to restart "
                    "the configured schedule explicitly."
                )
            scheduler.load_state_dict(payload["scheduler"])
            # load_state_dict restores counters/state but NOT the optimiser's
            # param-group lr; re-apply the schedule so the next optimizer.step()
            # uses exactly the learning rate the saved run had in force.
            vals = [float(v) for v in getattr(scheduler, "_last_lr", [])] or _schedule_lrs(scheduler)
            if vals is not None:
                _apply_scheduler_lrs(scheduler, vals)
        elif sch_pol == "reset":
            _reset_scheduler(scheduler)
        else:
            raise ValueError(f"unknown scheduler_policy {sch_pol!r}")

    if restore_rng and mode == "resume":
        _set_rng_state(payload.get("rng", {}))


    meta.optimizer_policy = opt_pol
    meta.scheduler_policy = sch_pol
    return meta

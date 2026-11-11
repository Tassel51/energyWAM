"""``QwenEnergyWAM`` — the Energy World Action Model.

Holds the backbone (tiny or real Qwen3-VL + frozen DINOv3), the EnergyWAM core
(projectors / encodings / energy head) and the stage runner, and exposes a
training forward and a joint inference entry point.

Inference contract
------------------
``predict`` receives **only** the current RGB, the language instruction, the
robot state and model/config information.  It never requires a ground-truth
future image, ``A*`` or ``Z*``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..config import EnergyWAMConfig, resolve_future_offsets
from ..data.standardization import (
    FUTURE_SPACE_RAW,
    FUTURE_SPACE_STANDARDIZED,
    PROFILE_LEGACY,
    PROFILE_PAPER,
    NormalizationBundle,
    StatsError,
    load_bundle,
)
from ..energy.init import init_candidates
from ..energy.solver import refine
from ..energy.types import ConditionBundle
from ..training.stages import StageRunner
from .energy_core import EnergyWAMCore


def _escape_inference(obj):
    """Clone tensors that were created inside ``inference_mode``.

    Inference tensors cannot participate in autograd even after the context
    exits, so a second-order unroll that touches them raises.  Cloning outside
    the context turns them into ordinary tensors.  Recurses through lists,
    tuples and dicts; leaves other objects untouched.
    """
    if isinstance(obj, torch.Tensor):
        if getattr(obj, "is_inference", False) or obj.is_inference():
            return obj.clone()
        return obj
    if isinstance(obj, dict):
        return {k: _escape_inference(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        t = type(obj)
        return t(_escape_inference(v) for v in obj)
    return obj


def build_backbone(cfg: EnergyWAMConfig) -> nn.Module:
    if cfg.backbone.backend == "tiny":
        from ..backbones.tiny_adapter import TinyBackboneAdapter

        return TinyBackboneAdapter(cfg, cfg.backbone.tiny_hidden_dim)
    if cfg.backbone.backend == "qwen3vl":
        from ..backbones.qwen3vl_adapter import Qwen3VLEnergyBackbone

        return Qwen3VLEnergyBackbone(cfg)
    raise ValueError(f"unknown backbone backend {cfg.backbone.backend!r}")


def geometry_from_config(cfg: EnergyWAMConfig, backbone: nn.Module) -> Dict[str, int]:
    """Candidate geometry from the config (``backbone`` kept for API stability)."""
    from ..config import geometry_from_cfg

    return geometry_from_cfg(cfg)


class QwenEnergyWAM(nn.Module):
    """Scalar Energy World Action Model.

    ``forward`` computes a training loss for the configured stage;
    ``predict`` runs joint (or configured) energy refinement and returns
    normalised actions.
    """

    def __init__(
        self,
        cfg: EnergyWAMConfig,
        backbone: Optional[nn.Module] = None,
        *,
        stats_bundle: Optional[NormalizationBundle] = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.normalization_profile = str(cfg.data.normalization_profile or PROFILE_LEGACY)

        # --- statistics first: fail fast BEFORE any heavy backbone is built ---
        # A missing/inconsistent statistics bundle must never reach the point of
        # loading the 4B VLM (or anything else expensive) before erroring.
        self.geometry = geometry_from_config(cfg, None)
        self.stats_bundle = self._resolve_stats_bundle(stats_bundle)
        self._register_stats_buffers(self.stats_bundle)

        # `backbone` lets a test inject a pre-built adapter (small random Qwen +
        # real frozen DINO) while still exercising THIS class end to end.
        self.backbone = backbone if backbone is not None else build_backbone(cfg)
        self.geometry = geometry_from_config(cfg, self.backbone)
        self.core = EnergyWAMCore(cfg, self.backbone, self.geometry)
        # NOTE: the backbone deliberately does NOT hold a reference to core.
        # Registering it would create a module cycle (core.backbone -> backbone
        # -> core) and make .to()/.train()/state_dict() recurse.
        self.stage_runner = StageRunner(cfg, self.core, stats_bundle=self.stats_bundle)
        self.future_offsets = resolve_future_offsets(cfg.data)
        # Candidate initialisation draws from the ordinary global torch RNG.
        # That keeps the training condition unchanged (a fresh random start each
        # step) while remaining fully reproducible: the checkpoint saves and
        # restores the global RNG state, so a resumed run continues the exact
        # stream an uninterrupted run would have seen.

    # ------------------------------------------------------------------
    def _needs_future_stats(self) -> bool:
        """Whether this configuration consumes standardised future latents."""
        if self.normalization_profile != PROFILE_PAPER:
            return False
        if self.cfg.stage.stage != "action_only":
            return True
        # action_only needs no future GT, but its optional future initialisation
        # still consumes future statistics.
        return self.cfg.init.initialization_future == "current_dino_noise"

    def _resolve_stats_bundle(
        self, stats_bundle: Optional[NormalizationBundle]
    ) -> Optional[NormalizationBundle]:
        """Resolve + validate the standardized statistics (paper §4.8).

        Resolution order for the paper profile: an explicitly supplied bundle
        (e.g. the one embedded in a checkpoint being restored) wins over
        ``data.norm_stats_bundle_path``.  The legacy profile never carries a
        bundle and keeps its exact old semantics.  Nothing ever falls back to
        raw coordinates / random statistics / a val/test estimate.
        """
        if self.normalization_profile == PROFILE_LEGACY:
            if stats_bundle is not None:
                raise StatsError(
                    "a paper-standardized statistics bundle was supplied but "
                    "data.normalization_profile='legacy'; set "
                    "data.normalization_profile='paper_standardized_v1' or drop the bundle"
                )
            return None
        if self.normalization_profile != PROFILE_PAPER:
            raise StatsError(f"unknown normalization_profile {self.normalization_profile!r}")

        if stats_bundle is not None:
            bundle = stats_bundle
            # an external bundle file that DISAGREES with the authoritative one
            # (e.g. embedded in a checkpoint) is rejected, never silently mixed
            ext = self.cfg.data.norm_stats_bundle_path
            if ext and Path(ext).exists():
                other = load_bundle(ext)
                if other.fingerprint != bundle.fingerprint:
                    raise StatsError(
                        f"external statistics bundle {ext} disagrees with the "
                        "statistics being used (fingerprint mismatch); refusing to "
                        "mix two different fits"
                    )
        elif self.cfg.data.norm_stats_bundle_path:
            bundle = load_bundle(self.cfg.data.norm_stats_bundle_path)
        else:
            raise StatsError(
                "data.normalization_profile='paper_standardized_v1' requires fitted "
                "training-split statistics.  Set data.norm_stats_bundle_path to a "
                "bundle from scripts/fit_normalization_stats.py, or restore a "
                "checkpoint that embeds its bundle.  Refusing to fall back to raw "
                "coordinates, random statistics or a val/test estimate."
            )
        if not isinstance(bundle, NormalizationBundle):
            raise StatsError(f"stats_bundle must be a NormalizationBundle, got {type(bundle)}")
        from ..data.energy_dataset import run_geometry_validation_kwargs

        bundle.validate_geometry(
            action_dim=self.geometry["action_dim"],
            future_dim=self.geometry["future_dim"],
            require_future=self._needs_future_stats(),
            **run_geometry_validation_kwargs(self.cfg),
        )
        return bundle

    def _register_stats_buffers(self, bundle: Optional[NormalizationBundle]) -> None:
        """Persist the statistics as stable buffers (paper profile only).

        Only the paper profile registers them, so a **legacy** model keeps its
        exact historical ``state_dict`` keys and old checkpoints keep loading
        under ``strict=True``.  The full bundle (provenance + fingerprint) is
        additionally written to the checkpoint payload so a checkpoint is
        self-contained even if the external bundle file disappears.
        """
        if bundle is None:
            return
        a = bundle.action
        self.register_buffer(
            "norm_action_mean", torch.tensor(a.mean, dtype=torch.float32), persistent=True
        )
        self.register_buffer(
            "norm_action_denom", torch.tensor(a.denom, dtype=torch.float32), persistent=True
        )
        self.register_buffer(
            "norm_action_count", torch.tensor(a.count, dtype=torch.float64), persistent=True
        )
        if bundle.future is not None:
            f = bundle.future
            self.register_buffer(
                "norm_future_mean", torch.tensor(f.mean, dtype=torch.float32), persistent=True
            )
            self.register_buffer(
                "norm_future_denom", torch.tensor(f.denom, dtype=torch.float32), persistent=True
            )
            self.register_buffer(
                "norm_future_count", torch.tensor(f.count, dtype=torch.float64), persistent=True
            )
        self.register_buffer(
            "norm_epsilon", torch.tensor(float(bundle.epsilon), dtype=torch.float64), persistent=True
        )

    # ------------------------------------------------------------------
    @property
    def device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # batch preparation
    # ------------------------------------------------------------------
    def _candidate_dtype(self):
        return getattr(torch, str(self.cfg.trainer.candidate_dtype), torch.float32)

    @staticmethod
    def _as_tensor(x, dtype, device):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)
        return x.to(device=device, dtype=dtype)

    # ------------------------------------------------------------------
    # input coordinate contract (avoiding double normalisation)
    # ------------------------------------------------------------------
    def _check_action_target_space(self, batch: Dict[str, Any]) -> None:
        """Optional ``action_target_space`` marker on hand-built batches.

        Absent marker = the documented example contract: ``action`` is already
        in the model's action space (dataset-normalised).  A marker is only
        checked, never inferred.
        """
        space = batch.get("action_target_space")
        if space is None:
            return
        if space == "model_space":
            return
        if space == "standardized" and self.stats_bundle is not None:
            return
        raise StatsError(
            f"action_target_space={space!r} is not valid for "
            f"normalization_profile={self.normalization_profile!r}.  Hand-built "
            "action targets must already be in the model space ('model_space' / "
            "'standardized' under the paper profile); raw actions are standardised "
            "by the data pipeline, never by the model."
        )

    def _resolve_future_target_space(self, Z: torch.Tensor, batch: Dict[str, Any]) -> torch.Tensor:
        """Turn a hand-built ``future_target`` into the model's coordinate.

        The marker ``future_target_space`` is **required** under the paper
        profile: ``'raw_dino'`` inputs are standardised with the embedded
        bundle, ``'standardized'`` inputs are used as-is.  Under the legacy
        profile the historical meaning (raw DINO features) is kept and an
        explicit ``'standardized'`` marker is rejected.
        """
        if Z.numel() == 0:
            return Z  # no future supervision at all (e.g. action_only data path)
        space = batch.get("future_target_space")
        if self.stats_bundle is None:
            if space == FUTURE_SPACE_STANDARDIZED:
                raise StatsError(
                    "future_target_space='standardized' but the model runs in the "
                    "legacy profile (raw DINO future coordinate); use "
                    "data.normalization_profile='paper_standardized_v1'"
                )
            return Z  # legacy: raw DINO (unchanged historical behaviour)
        if space is None:
            raise StatsError(
                "normalization_profile='paper_standardized_v1' requires an explicit "
                "future_target_space marker on hand-built future targets "
                "('raw_dino' or 'standardized'); refusing to guess the coordinate "
                "and risk double normalisation."
            )
        if space == FUTURE_SPACE_STANDARDIZED:
            return Z
        if space == FUTURE_SPACE_RAW:
            std = self.stats_bundle.require_future()
            return std.normalize(Z).to(Z.dtype)
        raise StatsError(f"unknown future_target_space {space!r}")

    def _prepare_batch(self, batch: Dict[str, Any], *, for_inference: bool = False) -> Dict[str, torch.Tensor]:
        """Normalise a caller batch into tensors on the model device.

        ``for_inference=True`` prepares **only** the condition and optional
        padding masks.  It never touches ``action_target`` / ``future_target`` /
        ``future_mask`` — joint inference must not read supervision.
        """
        dev = self.device
        dt = self._candidate_dtype()
        out: Dict[str, torch.Tensor] = {}

        if not for_inference:
            if "action_target" in batch and batch["action_target"] is not None:
                A = self._as_tensor(batch["action_target"], dt, dev)
                if A.dim() == 2:
                    A = A.unsqueeze(0)
                self._check_action_target_space(batch)
                out["action_target"] = A
            if "future_target" in batch and batch["future_target"] is not None:
                Z = self._as_tensor(batch["future_target"], dt, dev)
                if Z.dim() == 3:
                    Z = Z.unsqueeze(1).unsqueeze(1)   # [B,P,D] -> [B,1,1,P,D]
                elif Z.dim() == 4:
                    Z = Z.unsqueeze(2)                 # [B,F,P,D] -> [B,F,1,P,D]
                out["future_target"] = self._resolve_future_target_space(Z, batch)
            if "future_mask" in batch and batch["future_mask"] is not None:
                m = batch["future_mask"]
                m = torch.as_tensor(m) if not isinstance(m, torch.Tensor) else m
                out["future_mask"] = m.to(device=dev)
            if "future_offsets" in batch and batch["future_offsets"] is not None:
                out["future_offsets"] = self._as_tensor(batch["future_offsets"], torch.long, dev)

        # padding masks are geometry, not supervision — always allowed
        if "action_mask" in batch and batch["action_mask"] is not None:
            m = batch["action_mask"]
            m = torch.as_tensor(m) if not isinstance(m, torch.Tensor) else m
            out["action_mask"] = m.to(device=dev)

        if "current_dino" in batch and batch["current_dino"] is not None:
            out["current_dino"] = self._as_tensor(batch["current_dino"], dt, dev)
        if "_global_step" in batch:
            out["_global_step"] = int(batch["_global_step"])
        if "_candidate_seeds" in batch:
            out["_candidate_seeds"] = [int(seed) for seed in batch["_candidate_seeds"]]
        if "_bi_only" in batch:
            out["_bi_only"] = batch["_bi_only"]
        out["_raw"] = batch
        return out

    def _condition(self, batch_raw: Dict[str, Any], prepared: Dict[str, torch.Tensor]) -> ConditionBundle:
        """Build the condition bundle C for the active backend."""
        if self.cfg.backbone.backend == "tiny":
            cond_vec = batch_raw.get("cond_vec")
            if cond_vec is None:
                raise ValueError("tiny mode requires batch['cond_vec']")
            cond_vec = self._as_tensor(cond_vec, self._candidate_dtype(), self.device)
            if cond_vec.dim() == 1:
                cond_vec = cond_vec.unsqueeze(0)
            return self.backbone.encode_condition({"cond_vec": cond_vec})
        cond = self.backbone.encode_condition(batch_raw)
        if self.cfg.framework.quadratic_candidate_prior:
            raw = cond.native.get("extra_raw") or {}
            current = raw.get("dino")
            if current is None:
                raise ValueError("current DINO is required for the visual quadratic prior")
            if current.dim()==3:current=current.unsqueeze(1)
            if self.stats_bundle is not None:
                current = self.stats_bundle.future.normalize(current)
            raw["future_prior_center"] = current.detach()
            cond.native["extra_raw"] = raw
        return cond

    def _infer_batch_size(self, batch: Dict[str, Any], cond: ConditionBundle) -> int:
        """Batch size from the CONDITION only — never from a supervision tensor."""
        if self.cfg.backbone.backend == "tiny":
            cv = batch.get("cond_vec")
            if cv is not None:
                cv = np.asarray(cv) if not isinstance(cv, torch.Tensor) else cv
                return int(cv.shape[0]) if cv.ndim > 1 else 1
        images = batch.get("images")
        if images is not None:
            return len(images)
        langs = batch.get("langs")
        if langs is not None:
            return len(langs)
        if cond.native_attention_mask is not None:
            return int(cond.native_attention_mask.shape[0])
        if cond.group is not None:
            return int(cond.group.batch)
        raise ValueError("cannot determine batch size from the condition")

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Training forward.  Returns a dict containing ``action_loss`` (the
        StarVLA trainer's required key) plus per-stage diagnostics.

        ``action_loss`` is the *total* outer loss for the configured stage so a
        generic trainer minimising ``action_loss`` is doing the right thing.
        """
        if (
            self.cfg.stage.stage == "bi"
            and self.cfg.stage.bi_both_directions_each_batch
            and batch.get("_bi_only") is None
        ):
            return self._forward_bi_split(batch)
        prepared = self._prepare_batch(batch)
        cond = self._condition(batch, prepared)
        # Training candidates use the global torch RNG. Checkpoints restore its
        # state so resumed refinement follows the uninterrupted random stream.
        offload_stats = None
        if self.cfg.trainer.offload_saved_tensors:
            from ..training.offload import parameter_aware_cpu_offload

            budgets = self.cfg.trainer.saved_tensor_gpu_budgets_gib or self.cfg.trainer.saved_tensor_gpu_budget_gib
            with parameter_aware_cpu_offload(
                self, budgets,
                pin_budget_gib=self.cfg.trainer.saved_tensor_cpu_pin_budget_gib,
                asynchronous=self.cfg.trainer.saved_tensor_async_offload,
                max_inflight_gib=self.cfg.trainer.async_offload_max_inflight_gib,
                host_cache_limit_gib=self.cfg.trainer.saved_tensor_host_cache_limit_gib,
                deduplicate_cpu=self.cfg.trainer.saved_tensor_deduplicate_cpu,
            ) as offload_stats:
                out = self.stage_runner.run(cond, prepared, training=True)
        else:
            out = self.stage_runner.run(cond, prepared, training=True)
        result: Dict[str, torch.Tensor] = {"action_loss": out.loss}
        if offload_stats is not None:
            for key in ('pinned_bytes', 'pageable_bytes', 'host_cache_trim_count','cpu_dedup_hits','cpu_dedup_avoided_bytes'):
                result[f'offload/{key}'] = torch.tensor(float(offload_stats.get(key, 0)))
        if getattr(out, "loss_terms", None):
            # per-term graph breakdown: the trainer may backward each term
            # separately and accumulate (identical gradients, lower peak).
            result["loss_terms"] = list(out.loss_terms)
        for k, v in out.logs.items():
            if isinstance(v, torch.Tensor):
                result[k] = v
        return result

    def _forward_bi_split(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Run the two bi directions as independent forwards + backwards.

        Gradients accumulate to exactly ``grad(lambda_A*L_A + lambda_Z*L_Z)``
        but neither unroll graph needs ``retain_graph`` (a single combined
        forward shares the condition-encode graph between the two losses, which
        forces retain_graph on the first backward and keeps its unpacked
        saved tensors alive into the second).  Cost: the condition encode runs
        twice and each direction draws its own (unused for the other direction)
        N(0, I) candidate init — the optimised distributions are unchanged.
        """
        merged: Dict[str, torch.Tensor] = {}
        terms: list = []
        total = None
        for direction in ("action", "future"):
            sub = dict(batch)
            sub["_bi_only"] = direction
            r = self.forward(sub)
            terms.extend(r.get("loss_terms") or [r["action_loss"]])
            total = r["action_loss"] if total is None else total + r["action_loss"]
            for k, v in r.items():
                if k in ("action_loss", "loss_terms", "loss_terms_share_graph"):
                    continue
                merged.setdefault(k, v)
        result: Dict[str, torch.Tensor] = {"action_loss": total}
        result["loss_terms"] = terms
        result["loss_terms_share_graph"] = False
        result.update(merged)
        return result

    # ------------------------------------------------------------------
    def predict(
        self,
        batch: Dict[str, Any],
        *,
        mode: str = "joint",
        return_diagnostics: bool = False,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Joint (or configured) inference.  Inputs: current RGB, language,
        robot state and model/config information only.

        Context handling
        ----------------
        The whole refinement runs inside ``torch.inference_mode(False)`` +
        ``torch.enable_grad()`` so the candidate tensors are ordinary
        grad-requiring tensors even when the caller is under ``inference_mode``
        (the model server / eval loop may be).  Candidate gradients are obtained
        with ``autograd.grad`` (never ``backward``), so **no parameter ``.grad``
        is ever created**; the returned arrays are detached.
        """
        return self._predict_impl(batch, mode=mode, return_diagnostics=return_diagnostics, seed=seed)

    def _predict_impl(
        self, batch: Dict[str, Any], *, mode: str = "joint", return_diagnostics: bool = False,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Everything — condition construction AND refinement — runs inside
        ``inference_mode(False) + enable_grad``.

        If the caller (a model server / eval loop) already entered
        ``inference_mode``, tensors built earlier are *inference tensors* and
        cannot carry gradients; we clone them out of that state before use.
        The previous train/eval mode is restored on exit.
        """
        was_training = self.training
        self.eval()
        try:
            with torch.inference_mode(False), torch.enable_grad():
                raw = _escape_inference(batch)
                prepared = self._prepare_batch(raw, for_inference=True)
                cond = self._condition(raw, prepared)
                B = self._infer_batch_size(raw, cond)
                dev = self.device
                g = self.geometry

                a_mask = prepared.get("action_mask")
                # inference never reads a supervision future mask
                z_mask = None
                if "future_offsets" in raw and raw["future_offsets"] is not None:
                    offsets = self._as_tensor(raw["future_offsets"], torch.long, dev)
                else:
                    offsets = torch.tensor(
                        [self.future_offsets], dtype=torch.long, device=dev
                    ).expand(B, -1).contiguous()

                use_action_only = self.cfg.stage.stage == "action_only" or mode == "action_only"

                # Deterministic inference: a LOCAL generator seeded from the
                # config (or an explicit seed).  The no-leak contract requires
                # "same C and same random seed" to give the same output, and we
                # must NOT consume the global training RNG stream here.
                # The generator must live on the sampling device: a CPU
                # generator passed to torch.randn(device=<cuda>) raises.
                gen = torch.Generator(device=dev)
                gen.manual_seed(int(self.cfg.trainer.seed if seed is None else seed))
                # Current-frame DINO for ``initialization_future='current_dino_noise'``:
                # an explicit batch value wins; otherwise reuse the features the
                # condition encoder already computed for C.  Predict never
                # receives ``current_dino`` in a normal example batch, so without
                # this fallback the init option was unreachable at inference.
                cur_dino = prepared.get("current_dino")
                if cur_dino is None and cond.native:
                    cur_dino = (cond.native.get("extra_raw") or {}).get("dino")
                A0, Z0 = init_candidates(
                    self.cfg.init,
                    batch=B,
                    action_horizon=g["action_horizon"],
                    action_dim=g["action_dim"],
                    num_frames=g["num_future_frames"],
                    num_cameras=g["num_cameras"],
                    num_patches=g["future_patches"],
                    future_dim=g["future_dim"],
                    device=dev,
                    current_dino=cur_dino,
                    generator=gen,
                    future_stats=(
                        self.stats_bundle.future if self.stats_bundle is not None else None
                    ),
                )
                if use_action_only:
                    def energy_fn(a, z, iteration_progress=None):
                        return self.core.energy_action_only(cond, a, action_mask=a_mask, offsets=offsets, iteration_progress=iteration_progress)

                    Z_fixed = torch.zeros_like(Z0)
                    refine_mode = "action_given_future"
                else:
                    def energy_fn(a, z, iteration_progress=None):
                        return self.core.energy(
                            cond, z, a, future_mask=z_mask, action_mask=a_mask, offsets=offsets, iteration_progress=iteration_progress
                        )

                    Z_fixed = Z0
                    refine_mode = mode

                A_K, Z_K, state = refine(
                    energy_fn,
                    action=A0,
                    future=Z_fixed,
                    action_mask=a_mask,
                    future_mask=z_mask,
                    mode=refine_mode,
                    training=False,
                    **{
                        k: v
                        for k, v in self.stage_runner._solver_kwargs(False).items()
                        if k != "training"
                    },
                )
                A_out = A_K.detach()
                Z_out = Z_K.detach()
        finally:
            self.train(was_training)

        result: Dict[str, Any] = {"normalized_actions": A_out.cpu().numpy()}
        if return_diagnostics:
            Z_norm = Z_out.cpu().numpy()
            if self.stats_bundle is not None and self.stats_bundle.future is not None:
                # default contract: ``future_latent`` stays RAW DINO features
                # (same meaning as the legacy profile); the standardised value
                # is returned separately, with its space labelled.
                Z_raw = self.stats_bundle.future.denormalize(Z_norm)
                result["future_latent"] = Z_raw
                result["future_latent_space"] = "raw_dino"
                result["normalized_future_latent"] = Z_norm
                result["normalized_future_latent_space"] = (
                    f"{PROFILE_PAPER}: (Z - mu) / (sigma + eps)"
                )
            else:
                # legacy profile: the historical output schema, unchanged
                # (``future_latent`` IS raw DINO — there is no second space).
                result["future_latent"] = Z_norm
            result["diagnostics"] = {
                k: v.cpu().numpy() for k, v in state.as_diagnostics().items()
            }
            result["mode"] = refine_mode
        return result

    def predict_action(self, examples: List[Dict[str, Any]], **kwargs) -> Dict[str, np.ndarray]:
        """StarVLA-compatible inference entry.

        Returns ``{"normalized_actions": np.ndarray[B, H, Da]}``.
        """
        if isinstance(examples, dict):
            examples = [examples]
        batch = self._examples_to_batch(examples, require_targets=False)
        out = self.predict(batch, mode=kwargs.get("mode", "joint"),
                           return_diagnostics=bool(kwargs.get("return_diagnostics", False)),
                           seed=kwargs.get("seed"))
        return out

    def _examples_to_batch(self, examples: List[Dict[str, Any]], require_targets: bool) -> Dict[str, Any]:
        """Convert StarVLA-style example dicts into an EnergyWAM batch.

        StarVLA example keys: ``image`` (List[PIL]), ``lang`` (str),
        ``action`` ([T, Da]), optional ``state`` ([1, Dq]) and optional
        ``future_images`` / ``future_latent``.  Tiny mode additionally uses
        ``cond_vec``.

        Every value is treated as **one example**.  A leading singleton batch
        dimension left in by a batched helper is squeezed rather than stacked
        twice.
        """
        import numpy as np

        def _one(v):
            if v is None:
                return None
            a = v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v)
            if a.ndim >= 2 and a.shape[0] == 1:
                return a[0]
            return a

        def _first(*keys):
            """Return the first non-None value among ``keys`` (no `or` — numpy
            and torch arrays raise on truth-value ambiguity)."""
            for k in keys:
                for ex in examples:
                    if k in ex and ex[k] is not None:
                        return ex[k]
            return None

        images = [ex.get("image") for ex in examples]
        langs = [ex.get("lang", "") for ex in examples]
        states = [ex.get("state") for ex in examples]
        batch: Dict[str, Any] = {"images": images, "langs": langs, "states": states}

        # optional future RGB (raw) — the caller/core will run frozen DINO on it
        fut_imgs = [ex.get("future_images") for ex in examples]
        if any(f is not None for f in fut_imgs):
            batch["future_images"] = fut_imgs

        if require_targets:
            acts = [ex.get("action") for ex in examples]
            if any(a is None for a in acts):
                raise ValueError("training batch requires 'action' in every example")
            batch["action_target"] = np.stack(
                [np.asarray(_one(a), dtype=np.float32)[-self.geometry["action_horizon"]:] for a in acts]
            )
            fut = [None] * len(examples)
            for i, ex in enumerate(examples):
                v = ex.get("future_latent")
                if v is None:
                    v = ex.get("future_target")
                fut[i] = v
            if any(f is None for f in fut):
                raise ValueError("training batch requires 'future_target' or 'future_latent'")
            batch["future_target"] = np.stack([_one(f).astype("float32") for f in fut])
            # verifiable coordinate contract for hand-built targets: the example
            # may declare ``future_target_space`` / ``action_target_space``; when
            # it does not, the model-side check enforces the configured profile.
            for key in ("future_target_space", "action_target_space"):
                for ex in examples:
                    if ex.get(key) is not None:
                        batch[key] = ex[key]
                        break

        for key in ("action_mask", "future_mask"):
            vals = [ex.get(key) for ex in examples]
            if any(v is not None for v in vals):
                if any(v is None for v in vals):
                    raise ValueError(f"mixed presence of {key!r} in the batch")
                batch[key] = np.stack([_one(v).astype("float32") for v in vals])

        offs = [ex.get("future_offsets") for ex in examples]
        if any(o is not None for o in offs):
            batch["future_offsets"] = np.stack(
                [
                    np.asarray(
                        _one(o) if o is not None else self.future_offsets, dtype="int64"
                    )
                    for o in offs
                ]
            )

        if self.cfg.backbone.backend == "tiny":
            cvs = [ex.get("cond_vec") for ex in examples]
            if any(c is None for c in cvs):
                raise ValueError("tiny mode requires 'cond_vec' in every example")
            batch["cond_vec"] = np.stack([_one(c).astype("float32") for c in cvs])
        return batch

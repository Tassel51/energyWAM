"""Outer trainer for EnergyWAM.

Honesty constraints this module enforces
----------------------------------------
* The **inner** solver never steps an optimiser.  Only one outer
  ``optimizer.step()`` happens per training step.
* Bidirectional losses may use independent graphs with sequential backward;
  the optimizer steps once after accumulating both directions' gradients.
* Parameter gradients are zeroed at the start of every step.
* ``inference_mode`` is not used in training; ``predict`` handles its own
  context boundary.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from ..config import EnergyWAMConfig, to_plain_dict
from .checkpoint import (
    CheckpointMeta,
    load_checkpoint,
    norm_stats_to_meta,
    norm_stats_from_meta,
    restore,
    save_checkpoint,
)


def build_default_scheduler(cfg: EnergyWAMConfig, optimizer: torch.optim.Optimizer):
    """Config-driven default LR schedule: **linear warmup, then constant**.

    ``trainer.warmup_steps`` (previously dead config) drives the ramp: at outer
    step ``t`` the LR multiplier is ``min(1, (t + 1) / warmup_steps)``, so the
    first step already uses ``lr / warmup_steps`` (no wasted zero-LR step) and
    the LR is exactly ``trainer.learning_rate`` from step ``warmup_steps`` on.

    Engineering choice: warmup-then-constant with **no decay**.  The outer
    optimiser is a stable baseline for the inner energy refinement; a decaying
    LR would couple the two time scales for no documented gain.  A custom
    scheduler passed to :class:`EnergyWAMTrainer` always takes priority over
    this default.  ``warmup_steps=0`` gives a constant schedule — numerically
    identical to the previous ``scheduler=None`` behaviour (tiny runs unchanged).
    """
    warmup = int(cfg.trainer.warmup_steps)
    if warmup <= 0:
        lr_lambda = lambda step: 1.0  # noqa: E731 — constant schedule
    else:
        lr_lambda = lambda step: min(1.0, (step + 1) / float(warmup))  # noqa: E731
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class EnergyWAMTrainer:
    def __init__(
        self,
        cfg: EnergyWAMConfig,
        model: torch.nn.Module,
        *,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
    ):
        self.cfg = cfg
        self.model = model
        self.device = torch.device(cfg.trainer.device)
        self.model.to(self.device)
        if cfg.trainer.language_model_devices:
            from .parallel import shard_language_model

            shard_language_model(model, cfg.trainer.language_model_devices, self.device,
                                 layer_counts=cfg.trainer.language_model_layer_counts)
        params = [p for p in model.parameters() if p.requires_grad]
        step_module = getattr(getattr(model, "core", None), "step_sizes", None)
        step_params = list(step_module.parameters()) if step_module is not None else []
        step_ids = {id(p) for p in step_params}
        main_params = [p for p in params if id(p) not in step_ids]
        groups = []
        head_params = []
        if cfg.trainer.energy_head_learning_rate is not None:
            head_ids = {id(p) for p in model.core.energy_head.parameters()}
            head_params = [p for p in main_params if id(p) in head_ids]
            main_params = [p for p in main_params if id(p) not in head_ids]
        if cfg.trainer.task_learning_rate is not None:
            backbone = getattr(model, "backbone", None)
            mm = getattr(backbone, "_mm", None)
            language = getattr(mm, "language_model", None)
            language_ids = {id(p) for p in language.parameters()} if language is not None else set()
            trunk_params = [p for p in main_params if id(p) in language_ids]
            task_params = [p for p in main_params if id(p) not in language_ids]
            if trunk_params:groups.append(dict(params=trunk_params, group_name="language"))
            if task_params:groups.append(dict(params=task_params, lr=cfg.trainer.task_learning_rate, group_name="task"))
        else:
            groups.append(dict(params=main_params))
        if head_params:
            groups.append(dict(params=head_params, lr=cfg.trainer.energy_head_learning_rate, group_name="energy_head"))
        if step_params:
            groups.append(dict(params=step_params, lr=cfg.trainer.step_size_learning_rate,
                               weight_decay=0.0, optim_bits=32, group_name="step_sizes"))
        params = groups
        if optimizer is not None:
            self.optimizer = optimizer
        elif str(cfg.trainer.optimizer).lower() in ("adamw8bit", "adamw_8bit"):
            # Memory engineering: bitsandbytes AdamW8bit keeps the AdamW update
            # rule with 8-bit quantized moments (~16GB -> ~8.5GB for the 4B
            # trunk) so the K>=2 second-order unroll fits a 48G card.  Model
            # math is untouched; only optimizer-state storage is quantized.
            try:
                import bitsandbytes as bnb
            except ImportError as exc:
                raise ImportError(
                    "trainer.optimizer='adamw8bit' requires the bitsandbytes package"
                ) from exc
            self.optimizer = bnb.optim.AdamW8bit(
                params,
                lr=cfg.trainer.learning_rate,
                weight_decay=cfg.trainer.weight_decay,
            )
        else:
            self.optimizer = torch.optim.AdamW(
                params,
                lr=cfg.trainer.learning_rate,
                weight_decay=cfg.trainer.weight_decay,
            )
        if cfg.trainer.fp32_master_weights:
            from .master_weights import FP32MasterOptimizer
            self.optimizer = FP32MasterOptimizer(self.optimizer)
        self.optimizer_state_offloader = None
        if cfg.trainer.optimizer_state_cpu_offload:
            if str(cfg.trainer.optimizer).lower() not in ("adamw8bit", "adamw_8bit") or not cfg.trainer.preallocate_optimizer_state:
                raise ValueError("CPU state offload requires preallocated AdamW8bit state")
            from .optimizer_offload import OptimizerStateOffloader
            self.optimizer_state_offloader = OptimizerStateOffloader(self.optimizer)
        self._preallocate_optimizer_state()
        self.scheduler = scheduler if scheduler is not None else build_default_scheduler(cfg, self.optimizer)
        self.global_step = 0
        self.stage = cfg.stage.stage
        self.log_dir = Path(cfg.trainer.output_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = self.log_dir / "train_log.jsonl"
        self._loss_history = []

    def _preallocate_optimizer_state(self):
        if not self.cfg.trainer.preallocate_optimizer_state:
            return
        if str(self.cfg.trainer.optimizer).lower() not in ("adamw8bit", "adamw_8bit"):
            raise ValueError("optimizer-state preallocation currently requires AdamW8bit")
        opt = getattr(self.optimizer, "base_optimizer", self.optimizer)
        opt.check_overrides()
        opt.to_gpu()
        opt.initialized = True
        for gi, group in enumerate(opt.param_groups):
            for pi, parameter in enumerate(group["params"]):
                if not opt.state.get(parameter):
                    opt.init_state(group, parameter, gi, pi)
        offloader = getattr(self, "optimizer_state_offloader", None)
        if offloader is not None:
            offloader.offload()

    # ------------------------------------------------------------------
    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        if self.device.type == "cuda":
            for dev in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(dev)

        batch = dict(batch)
        batch["_global_step"] = self.global_step
        t0 = time.time()
        streamed = (
            self.cfg.stage.stage == "bi"
            and self.cfg.stage.bi_both_directions_each_batch
            and hasattr(self.model, "_forward_bi_split")
        )
        streamed_bwd = 0.0
        if streamed:
            # Complete one independent direction before allocating the next
            # graph. Parameters stay unchanged until both gradients accumulate.
            out = {}
            total = None
            for direction in ("action", "future"):
                result = self.model(dict(batch, _bi_only=direction))
                term = result["action_loss"]
                if not torch.isfinite(term):
                    raise RuntimeError(f"non-finite {direction} loss at step {self.global_step}")
                print(f"[EnergyWAM] step={self.global_step} {direction} backward", flush=True)
                start_bwd = time.time()
                term.backward()
                streamed_bwd += time.time() - start_bwd
                total = term.detach() if total is None else total + term.detach()
                for k, v in result.items():
                    if k != "action_loss" and isinstance(v, torch.Tensor) and v.dim() == 0:
                        out.setdefault(k, v.detach())
                del result, term
            out["action_loss"] = total
        else:
            out = self.model(batch)
        loss = out["action_loss"]
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {self.global_step}: {loss}")
        t_fwd = time.time() - t0 - streamed_bwd

        t1 = time.time()
        terms = out.get("loss_terms")
        if streamed:
            pass  # both independent backwards already completed
        elif terms:
            # Sequential per-term backward with gradient accumulation:
            # grad(sum of terms) == sum of grads exactly, but only one term's
            # unroll graph is live at a time, halving backward peak memory.
            # retain_graph is only needed when the terms share a graph
            # (combined forward); independent per-direction forwards must not
            # retain or the first graph's unpacked tensors stay alive.
            share = bool(out.get("loss_terms_share_graph", True))
            mem = loss.is_cuda
            for i, term in enumerate(terms):
                if mem:
                    torch.cuda.reset_peak_memory_stats(loss.device)
                    print(
                        f"[mem] term{i} backward start: "
                        f"alloc={torch.cuda.memory_allocated(loss.device) / 2**30:.2f}GB",
                        flush=True,
                    )
                term.backward(retain_graph=share and (i < len(terms) - 1))
                if mem:
                    print(
                        f"[mem] term{i} backward done: "
                        f"peak={torch.cuda.max_memory_allocated(loss.device) / 2**30:.2f}GB",
                        flush=True,
                    )
        else:
            loss.backward()
        t_bwd = time.time() - t1 + streamed_bwd

        return self._finish_step(batch, out, t0, t_fwd, t_bwd)

    def _finish_step(self, batch, out, t0, t_fwd, t_bwd):
        """Validate accumulated gradients, update once and record a step."""
        loss = out["action_loss"]
        # --- parameter gradient diagnostics (before clipping) ---
        grad_nonfinite = 0
        grad_norm_sq = 0.0
        n_params_with_grad = 0
        for p_ in self.model.parameters():
            if p_.grad is None:
                continue
            n_params_with_grad += 1
            if not torch.isfinite(p_.grad).all():
                grad_nonfinite += 1
            else:
                grad_norm_sq += float(p_.grad.detach().float().pow(2).sum())
        if grad_nonfinite:
            raise RuntimeError(
                f"non-finite parameter gradients in {grad_nonfinite} tensors at step "
                f"{self.global_step}"
            )

        extra_diagnostics = {}
        conditioner = getattr(getattr(self.model, "core", None), "iteration_conditioner", None)
        if conditioner is not None:
            for direction in ("action", "future"):
                mlp = getattr(conditioner, direction + "_mlp")
                norm_sq = sum(float(p.grad.detach().float().square().sum())
                              for p in mlp.parameters() if p.grad is not None)
                extra_diagnostics[f"diag/iteration_{direction}_grad_norm"] = norm_sq ** 0.5
        steps = getattr(getattr(self.model, "core", None), "step_sizes", None)
        if steps is not None:
            if steps.raw.grad is not None:
                for i, direction in enumerate(("action", "future")):
                    extra_diagnostics[f"diag/step_{direction}_raw_grad"] = float(steps.raw.grad[i].detach())
            for i, direction in enumerate(("action", "future")):
                extra_diagnostics[f"diag/step_{direction}_bound_fraction"] = float(steps.raw[i].detach().sigmoid())

        clip = self.cfg.trainer.max_grad_norm
        if clip and clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad], clip
            )
        used_lrs = {"optimizer_lr/"+group.get("group_name", "main"):float(group["lr"])
                    for group in self.optimizer.param_groups}
        t2 = time.time()
        offloader = getattr(self, "optimizer_state_offloader", None)
        if offloader is not None:
            extra_diagnostics["offload/optimizer_state_restore_bytes"] = offloader.restore()
        try:
            self.optimizer.step()
        finally:
            if offloader is not None:
                extra_diagnostics["offload/optimizer_state_cpu_bytes"] = offloader.offload()
        if self.scheduler is not None and hasattr(self.scheduler, "step"):
            self.scheduler.step()
        t_opt = time.time() - t2

        scalars = {k: float(v.detach()) for k, v in out.items() if isinstance(v, torch.Tensor) and v.dim() == 0}
        scalars.update(extra_diagnostics)
        scalars.update(used_lrs)
        scalars["solver_gradient_rms_normalized"] = float(self.cfg.solver.candidate_gradient_normalization == "rms")
        if steps is not None:
            for direction, value in zip(("action", "future"), steps()):
                scalars[f"diag/alpha_{direction}_after_update"] = float(value.detach())
        scalars["global_step"] = float(self.global_step)
        scalars["loss"] = float(loss.detach())
        scalars["step_seconds"] = round(time.time() - t0, 6)
        scalars["forward_seconds"] = round(t_fwd, 6)
        scalars["backward_seconds"] = round(t_bwd, 6)
        scalars["optimizer_seconds"] = round(t_opt, 6)
        scalars["param_grad_norm"] = round(grad_norm_sq ** 0.5, 6)
        scalars["param_grad_nonfinite"] = float(grad_nonfinite)
        scalars["params_with_grad"] = float(n_params_with_grad)
        scalars["trainable_params"] = float(sum(p.numel() for p in self.model.parameters() if p.requires_grad))
        scalars["batch_size"] = float(batch["action_target"].shape[0])
        scalars["samples_per_second"] = scalars["batch_size"] / max(scalars["step_seconds"], 1e-9)
        try:
            import resource

            scalars["max_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 2)
        except Exception:
            pass
        if torch.cuda.is_available() and str(self.device).startswith("cuda"):
            try:
                scalars["cuda_max_alloc_mb"] = round(torch.cuda.max_memory_allocated(self.device) / 1e6, 2)
                for dev in range(torch.cuda.device_count()):
                    scalars[f"cuda_{dev}_max_alloc_mb"] = round(torch.cuda.max_memory_allocated(dev) / 1e6, 2)
                    scalars[f"cuda_{dev}_max_reserved_mb"] = round(torch.cuda.max_memory_reserved(dev) / 1e6, 2)
            except Exception:
                pass

        self._loss_history.append(scalars)
        if self.global_step % max(1, self.cfg.trainer.logging_frequency) == 0:
            self._append_log(scalars)
        self.global_step += 1
        return scalars

    # ------------------------------------------------------------------
    def fit(self, batches, max_steps: Optional[int] = None) -> Dict[str, Any]:
        """Run ``max_steps`` optimiser steps over an iterable of batches."""
        max_steps = int(self.cfg.trainer.max_train_steps if max_steps is None else max_steps)
        save_every = int(getattr(self.cfg.trainer, "save_interval", 0) or 0)
        t0 = time.time()
        n = 0
        it = iter(batches)
        while n < max_steps:
            try:
                batch = next(it)
            except StopIteration:
                it = iter(batches)
                batch = next(it)
            self.train_step(batch)
            n += 1
            if save_every > 0 and (self.global_step == 1 or self.global_step % save_every == 0):
                # rolling resume anchor, written atomically (large file)
                tmp = self.save("ckpt_latest.tmp.pt")
                tmp.replace(self.log_dir / "ckpt_latest.pt")
        hist = self._loss_history
        return {
            "steps": n,
            "global_step": self.global_step,
            "seconds": round(time.time() - t0, 3),
            "first_loss": hist[0] if hist else None,
            "last_loss": hist[-1] if hist else None,
            # a bounded tail for the log; first/last are reported separately so
            # a long run cannot hide its starting loss in the tail window
            "loss_history": hist[-min(len(hist), 50):],
        }

    # ------------------------------------------------------------------
    def save(self, name: str = "ckpt.pt", **meta_kwargs) -> Path:
        raw_stats = getattr(self.model, "norm_stats", None) or {}
        bundle = getattr(self.model, "stats_bundle", None)
        if bundle is not None and not raw_stats:
            # self-contained paper checkpoint: the serve/eval action
            # un-normalisation is derivable from the embedded bundle alone
            key = str(self.cfg.data.data_mix or "action")
            raw_stats = {key: bundle.action.to_norm_stats()}
        meta = CheckpointMeta(
            stage=self.stage,
            global_step=self.global_step,
            optimizer_policy="continue",
            scheduler_policy="continue",
            norm_stats_version=str(getattr(self.model, "norm_stats_version", "")) or (
                bundle.profile if bundle is not None else ""
            ),
            norm_stats=norm_stats_to_meta(raw_stats),
            normalization_profile=str(getattr(self.model, "normalization_profile", "") or ""),
            normalization_bundle=bundle.to_payload() if bundle is not None else {},
            notes=meta_kwargs.pop("notes", ""),
        )
        for k, v in meta_kwargs.items():
            if hasattr(meta, k):
                setattr(meta, k, v)
        return save_checkpoint(
            self.log_dir / name,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            cfg=self.cfg,
            meta=meta,
        )

    def load(self, path: str | Path, mode: str = "resume", **policy) -> CheckpointMeta:
        # Loading a sharded model's entire checkpoint onto GPU 0 can OOM.
        # load_state_dict/optimizer.load_state_dict place tensors with owners.
        payload = load_checkpoint(path, map_location="cpu")
        saved_stage = payload.get("meta", {}).get("stage")
        if mode == "resume" and saved_stage is not None and saved_stage != self.cfg.stage.stage:
            raise ValueError(
                f"stage mismatch on resume: checkpoint was saved at stage "
                f"{saved_stage!r} but the current config asks for "
                f"{self.cfg.stage.stage!r}.  Use mode='switch' (trainer.switch_stage) "
                f"to change stages explicitly."
            )
        meta = restore(
            payload,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            mode=mode,
            **policy,
        )
        # Stage-switch reset clears moments; recreate zero states before the
        # first forward so its memory footprint matches the capacity probe.
        self._preallocate_optimizer_state()
        self.model.norm_stats = norm_stats_from_meta(meta.norm_stats)
        self.model.norm_stats_version = meta.norm_stats_version
        self.global_step = int(meta.global_step) if mode == "resume" else 0
        self.stage = self.cfg.stage.stage
        self._append_log(
            {
                "event": "restore",
                "mode": mode,
                "optimizer_policy": meta.optimizer_policy,
                "scheduler_policy": meta.scheduler_policy,
                "restored_step": meta.global_step,
                "new_stage": self.stage,
            }
        )
        return meta

    # ------------------------------------------------------------------
    def switch_stage(
        self,
        new_stage: str,
        ckpt_path: str | Path,
        *,
        optimizer_policy: str = "reset",
        scheduler_policy: str = "reset",
    ):
        """bi -> joint (or any stage change) with **explicit** optimiser/scheduler
        policies.  Both are written to the log so a reader can tell exactly what
        was kept and what was rebuilt."""
        self.cfg.stage.stage = new_stage
        meta = self.load(
            ckpt_path,
            mode="switch",
            optimizer_policy=optimizer_policy,
            scheduler_policy=scheduler_policy,
        )
        self._append_log(
            {
                "event": "stage_switch",
                "from": meta.stage,
                "to": new_stage,
                "optimizer_policy": meta.optimizer_policy,
                "scheduler_policy": meta.scheduler_policy,
            }
        )
        return meta

    # ------------------------------------------------------------------
    def _append_log(self, rec: Dict[str, Any]) -> None:
        with self._log_path.open("a") as f:
            f.write(json.dumps(rec, default=str) + "\n")

    def export_config(self) -> Path:
        p = self.log_dir / "resolved_config.json"
        p.write_text(json.dumps(to_plain_dict(self.cfg), indent=2, default=str))
        return p

"""Training-stage organisation: bi / single-direction / joint / direct_joint / action_only.

Every stage runs the **same** solver and the same scalar energy network.  The
only differences are which candidates are free, what is pinned to ground truth,
and how the outer loss is weighted.  Nothing here does a parameter update — the
outer optimiser is stepped by the trainer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch

from ..config import EnergyWAMConfig
from ..energy.init import init_candidates
from ..energy.losses import action_endpoint_loss, future_endpoint_loss, valid_element_count
from ..energy.solver import SolverState, refine
from ..energy.types import ConditionBundle


@dataclass
class StageOutput:
    loss: torch.Tensor
    logs: Dict[str, torch.Tensor] = field(default_factory=dict)
    action_pred: Optional[torch.Tensor] = None
    future_pred: Optional[torch.Tensor] = None
    state_action: Optional[SolverState] = None
    state_future: Optional[SolverState] = None
    state_joint: Optional[SolverState] = None
    #: Optional per-term breakdown of ``loss`` (each WITH its autograd graph).
    #: When present the trainer may backward the terms sequentially and
    #: accumulate:  grad(sum of terms) == sum of grads exactly, but only one
    #: term's graph is live at a time, halving backward peak memory.
    loss_terms: Optional[list] = None


def _require_grad(x: torch.Tensor) -> torch.Tensor:
    return x.detach().clone().requires_grad_(True)


class StageRunner:
    """Runs one training step for a configured stage."""

    def __init__(self, cfg: EnergyWAMConfig, core, stats_bundle=None):
        self.cfg = cfg
        self.core = core
        #: paper §4.8 bundle (None under the legacy profile).  Only consumed by
        #: the optional ``current_dino_noise`` initialisation, which maps the raw
        #: current DINO base into the standardized coordinate first.
        self.stats_bundle = stats_bundle

    # ------------------------------------------------------------------
    def _solver_kwargs(self, training: bool) -> Dict:
        s = self.cfg.solver
        alpha_a, alpha_z = self.core.effective_step_sizes()
        if not training:
            alpha_a = torch.as_tensor(alpha_a).detach()
            alpha_z = torch.as_tensor(alpha_z).detach()
        return dict(
            iteration_aware=self.cfg.framework.iteration_conditioning,
            candidate_gradient_normalization=s.candidate_gradient_normalization,
            gradient_rms_epsilon=s.gradient_rms_epsilon,
            num_steps=s.train_inner_steps if training else s.eval_inner_steps,
            alpha_action=alpha_a,
            alpha_future=alpha_z,
            training=training,
            clip_action_grad=s.clip_action_grad,
            clip_future_grad=s.clip_future_grad,
            action_clamp=(s.action_clamp_min, s.action_clamp_max),
            future_clamp=(s.future_clamp_min, s.future_clamp_max),
            unroll_mode=s.unroll_mode,
            truncation_length=s.truncation_length,
            joint_update_mode=s.joint_update_mode,
            langevin_noise_std=s.langevin_noise_std,
        )

    def _current_dino(self, batch, cond):
        """Current-frame DINO features for candidate initialisation.

        Explicit ``batch["current_dino"]`` wins.  Otherwise the features the
        condition encoder already computed for ``C`` (``extra_raw['dino']``) are
        reused: without this fallback ``initialization_future='current_dino_noise'``
        could only run when a caller hand-built a raw batch dict, never through
        the real training forward or ``predict`` (which never populate the key).
        """
        cd = batch.get("current_dino")
        if cd is not None:
            return cd
        raw = (cond.native or {}).get("extra_raw") if cond.native else None
        return raw.get("dino") if raw else None

    def _init(self, batch, device, generator=None, cond=None):
        g = self.core.geometry
        B = batch["action_target"].shape[0]
        current_dino = self._current_dino(batch, cond) if cond is not None else batch.get("current_dino")
        if "_candidate_seeds" in batch:
            seeds=batch['_candidate_seeds']
            if len(seeds)!=B:raise ValueError('one candidate seed per sample is required')
            pairs=[init_candidates(self.cfg.init,batch=1,
                action_horizon=g['action_horizon'],action_dim=g['action_dim'],
                num_frames=g['num_future_frames'],num_cameras=g['num_cameras'],
                num_patches=g['future_patches'],future_dim=g['future_dim'],device=device,
                current_dino=None if current_dino is None else current_dino[i:i+1],
                generator=torch.Generator(device=device).manual_seed(int(seed)),
                future_stats=self.stats_bundle.future if self.stats_bundle is not None else None)
                for i,seed in enumerate(seeds)]
            return torch.cat([p[0] for p in pairs]),torch.cat([p[1] for p in pairs])
        A0, Z0 = init_candidates(
            self.cfg.init,
            batch=B,
            action_horizon=g["action_horizon"],
            action_dim=g["action_dim"],
            num_frames=g["num_future_frames"],
            num_cameras=g["num_cameras"],
            num_patches=g["future_patches"],
            future_dim=g["future_dim"],
            device=device,
            current_dino=current_dino,
            generator=generator,
            future_stats=self.stats_bundle.future if self.stats_bundle is not None else None,
        )
        return A0, Z0

    # ------------------------------------------------------------------
    def run(self, cond: ConditionBundle, batch: Dict[str, torch.Tensor], training: bool = True) -> StageOutput:
        stage = self.cfg.stage.stage
        if stage == "bi":
            return self._run_bi(cond, batch, training)
        if stage == "action_given_future":
            return self._run_single(cond, batch, training, direction="A")
        if stage == "future_given_action":
            return self._run_single(cond, batch, training, direction="Z")
        if stage in ("joint", "direct_joint"):
            return self._run_joint(cond, batch, training)
        if stage == "action_only":
            return self._run_action_only(cond, batch, training)
        raise ValueError(f"unknown stage {stage!r}")

    # ------------------------------------------------------------------
    def _loss_terms(self, A_K, Z_K, batch):
        A_star = batch["action_target"]
        Z_star = batch["future_target"]
        a_mask = batch.get("action_mask")
        z_mask = batch.get("future_mask")
        L_A = action_endpoint_loss(A_K, A_star, a_mask, self.cfg.loss.action_loss)
        L_Z = future_endpoint_loss(
            Z_K, Z_star, z_mask, self.cfg.loss.future_loss, self.cfg.loss.future_cosine_weight
        )
        logs = {
            "action_endpoint_loss": L_A.detach(),
            "future_endpoint_loss": L_Z.detach(),
            "action_valid_elems": torch.tensor(
                valid_element_count(a_mask, A_star), dtype=torch.float32
            ),
            "future_valid_elems": torch.tensor(
                valid_element_count(z_mask, Z_star), dtype=torch.float32
            ),
            "action_endpoint_mse": L_A.detach(),
            "future_endpoint_mse": L_Z.detach(),
        }
        return L_A, L_Z, logs

    # ------------------------------------------------------------------
    def _run_bi(self, cond, batch, training):
        cfg = self.cfg
        device = batch["action_target"].device
        A0, Z0 = self._init(batch, device, cond=cond)
        A_star = batch["action_target"]
        Z_star = batch["future_target"]
        a_mask = batch.get("action_mask")
        z_mask = batch.get("future_mask")
        offsets = batch.get("future_offsets")

        energy_fn = lambda a, z, iteration_progress=None: self.core.energy(
            cond, z, a, future_mask=z_mask, action_mask=a_mask, offsets=offsets, iteration_progress=iteration_progress
        )

        both = cfg.stage.bi_both_directions_each_batch
        L_total = torch.zeros((), device=device, dtype=torch.float64 if self.cfg.trainer.candidate_dtype == "float64" else torch.float32)
        logs: Dict[str, torch.Tensor] = {}
        out = StageOutput(loss=L_total)
        terms: list = []

        # Direction scheduling: explicit single-direction override (the model
        # may run the two directions as separate forwards so each backward can
        # free its graph immediately), else every batch (default) or
        # deterministic alternation driven by the trainer's global step.
        only = batch.get("_bi_only")
        if only == "action":
            do_A, do_Z = True, False
        elif only == "future":
            do_A, do_Z = False, True
        elif both:
            do_A, do_Z = True, True
        else:
            step = int(batch.get("_global_step", 0))
            do_A, do_Z = (step % 2 == 0), (step % 2 == 1)

        # --- Future -> Action (Z* pinned, A refined) ---
        if do_A:
            A0r = _require_grad(A0)
            Z_star_det = Z_star.detach().float()
            A_K, _, st = refine(
                energy_fn, action=A0r, future=Z_star_det,
                action_mask=a_mask, future_mask=z_mask,
                mode="action_given_future", **self._solver_kwargs(training),
            )
            L_A, _, logs_A = self._loss_terms(A_K, Z_star_det, batch)
            term_A = cfg.loss.lambda_action * L_A
            terms.append(term_A)
            L_total = L_total + term_A
            logs.update({f"bi/{'action'}_{k}": v for k, v in logs_A.items()})
            logs["bi/action_loss"] = L_A.detach()
            out.state_action = st
            out.action_pred = A_K

        # --- Action -> Future (A* pinned, Z refined) ---
        if do_Z:
            Z0r = _require_grad(Z0)
            A_star_det = A_star.detach().float()
            _, Z_K, st = refine(
                energy_fn, action=A_star_det, future=Z0r,
                action_mask=a_mask, future_mask=z_mask,
                mode="future_given_action", **self._solver_kwargs(training),
            )
            _, L_Z, logs_Z = self._loss_terms(A_star_det, Z_K, batch)
            term_Z = cfg.loss.lambda_future * L_Z
            terms.append(term_Z)
            L_total = L_total + term_Z
            logs.update({f"bi/future_{k}": v for k, v in logs_Z.items()})
            logs["bi/future_loss"] = L_Z.detach()
            out.state_future = st
            out.future_pred = Z_K

        if out.state_action is not None:
            logs.update({f"diag/action_{k}": v for k, v in out.state_action.as_diagnostics().items()})
        if out.state_future is not None:
            logs.update({f"diag/future_{k}": v for k, v in out.state_future.as_diagnostics().items()})
        logs["diag/steps"] = torch.tensor(float(cfg.solver.train_inner_steps if training else cfg.solver.eval_inner_steps))
        logs["diag/alpha_action"] = torch.as_tensor(self.core.effective_step_sizes()[0]).detach()
        logs["diag/alpha_future"] = torch.as_tensor(self.core.effective_step_sizes()[1]).detach()
        out.loss = L_total
        out.loss_terms = terms or None
        out.logs = logs
        return out

    # ------------------------------------------------------------------
    def _run_single(self, cond, batch, training, direction: str):
        cfg = self.cfg
        device = batch["action_target"].device
        A0, Z0 = self._init(batch, device, cond=cond)
        A_star = batch["action_target"]
        Z_star = batch["future_target"]
        a_mask = batch.get("action_mask")
        z_mask = batch.get("future_mask")
        offsets = batch.get("future_offsets")
        energy_fn = lambda a, z, iteration_progress=None: self.core.energy(
            cond, z, a, future_mask=z_mask, action_mask=a_mask, offsets=offsets, iteration_progress=iteration_progress
        )
        out = StageOutput(loss=torch.zeros((), device=device, dtype=torch.float64 if self.cfg.trainer.candidate_dtype == "float64" else torch.float32))
        if direction == "A":
            A0r = _require_grad(A0)
            Zd = Z_star.detach().float()
            A_K, _, st = refine(
                energy_fn, action=A0r, future=Zd,
                action_mask=a_mask, future_mask=z_mask,
                mode="action_given_future", **self._solver_kwargs(training),
            )
            L_A, _, logs = self._loss_terms(A_K, Zd, batch)
            out.loss = cfg.loss.lambda_action * L_A
            out.logs = {f"action_given_future/{k}": v for k, v in logs.items()}
            out.state_action = st
            out.action_pred = A_K
        else:
            Z0r = _require_grad(Z0)
            Ad = A_star.detach().float()
            _, Z_K, st = refine(
                energy_fn, action=Ad, future=Z0r,
                action_mask=a_mask, future_mask=z_mask,
                mode="future_given_action", **self._solver_kwargs(training),
            )
            _, L_Z, logs = self._loss_terms(Ad, Z_K, batch)
            out.loss = cfg.loss.lambda_future * L_Z
            out.logs = {f"future_given_action/{k}": v for k, v in logs.items()}
            out.state_future = st
            out.future_pred = Z_K
        st_out = out.state_action or out.state_future
        if st_out is not None:
            out.logs.update({f"diag/{k}": v for k, v in st_out.as_diagnostics().items()})
        out.logs["diag/steps"] = torch.tensor(
            float(cfg.solver.train_inner_steps if training else cfg.solver.eval_inner_steps)
        )
        return out

    # ------------------------------------------------------------------
    def _run_joint(self, cond, batch, training):
        cfg = self.cfg
        device = batch["action_target"].device
        A0, Z0 = self._init(batch, device, cond=cond)
        a_mask = batch.get("action_mask")
        z_mask = batch.get("future_mask")
        offsets = batch.get("future_offsets")
        energy_fn = lambda a, z, iteration_progress=None: self.core.energy(
            cond, z, a, future_mask=z_mask, action_mask=a_mask, offsets=offsets, iteration_progress=iteration_progress
        )
        A0r = _require_grad(A0)
        Z0r = _require_grad(Z0)
        A_K, Z_K, st = refine(
            energy_fn, action=A0r, future=Z0r,
            action_mask=a_mask, future_mask=z_mask,
            mode="joint", **self._solver_kwargs(training),
        )
        L_A, L_Z, logs = self._loss_terms(A_K, Z_K, batch)
        loss = cfg.loss.beta_action * L_A + cfg.loss.beta_future * L_Z

        pending_aux = None
        if cfg.stage.joint_use_bi_aux and cfg.stage.joint_bi_aux_weight > 0:
            # A REAL conditional bidirectional auxiliary loss: two extra solves
            # in which the ground truth on one side is pinned and only the other
            # candidate is refined.  This is NOT the joint endpoint loss re-used
            # as a stand-in — each term is produced by its own solver run.
            Z_star = batch["future_target"].detach().float()
            A_star = batch["action_target"].detach().float()
            aux_state = {}

            # Future -> Action (Z* pinned)
            A0a = _require_grad(A0)
            A_Ka, _, st_a = refine(
                energy_fn, action=A0a, future=Z_star,
                action_mask=a_mask, future_mask=z_mask,
                mode="action_given_future", **self._solver_kwargs(training),
            )
            aux_A = action_endpoint_loss(A_Ka, A_star, a_mask, cfg.loss.action_loss)
            aux_state["action"] = st_a

            # Action -> Future (A* pinned)
            Z0a = _require_grad(Z0)
            _, Z_Ka, st_z = refine(
                energy_fn, action=A_star, future=Z0a,
                action_mask=a_mask, future_mask=z_mask,
                mode="future_given_action", **self._solver_kwargs(training),
            )
            aux_Z = future_endpoint_loss(
                Z_Ka, Z_star, z_mask, cfg.loss.future_loss, cfg.loss.future_cosine_weight
            )
            aux_state["future"] = st_z

            aux = cfg.loss.lambda_action * aux_A + cfg.loss.lambda_future * aux_Z
            loss = loss + float(cfg.stage.joint_bi_aux_weight) * aux
            aux_logs = {
                "bi_aux_loss": aux.detach(),
                "bi_aux_action_loss": aux_A.detach(),
                "bi_aux_future_loss": aux_Z.detach(),
            }
            for tag, st in aux_state.items():
                for k, v in st.as_diagnostics().items():
                    aux_logs[f"bi_aux_{tag}/{k}"] = v
            pending_aux = aux_logs

        logs = {f"joint/{k}": v for k, v in logs.items()}
        logs["joint/action_loss"] = L_A.detach()
        logs["joint/future_loss"] = L_Z.detach()
        for k, v in (pending_aux or {}).items():
            logs[f"joint/{k}"] = v
        logs.update({f"diag/joint_{k}": v for k, v in st.as_diagnostics().items()})
        logs["diag/steps"] = torch.tensor(
            float(cfg.solver.train_inner_steps if training else cfg.solver.eval_inner_steps)
        )
        logs["diag/alpha_action"] = torch.as_tensor(self.core.effective_step_sizes()[0]).detach()
        logs["diag/alpha_future"] = torch.as_tensor(self.core.effective_step_sizes()[1]).detach()
        return StageOutput(
            loss=loss, logs=logs, action_pred=A_K, future_pred=Z_K, state_joint=st
        )

    # ------------------------------------------------------------------
    def _run_action_only(self, cond, batch, training):
        cfg = self.cfg
        device = batch["action_target"].device
        A0, _ = self._init(batch, device, cond=cond)
        A_star = batch["action_target"]
        a_mask = batch.get("action_mask")
        offsets = batch.get("future_offsets")

        # Independent E(C, A) path — no future ground truth is read.
        def energy_fn(a, z, iteration_progress=None):
            return self.core.energy_action_only(cond, a, action_mask=a_mask, offsets=offsets, iteration_progress=iteration_progress)

        A0r = _require_grad(A0)
        Z_dummy = torch.zeros(
            A0.shape[0],
            self.core.num_future_frames,
            self.core.num_cameras,
            self.core.future_patches,
            self.core.future_dim,
            device=device,
            dtype=torch.float32,
        )
        A_K, _, st = refine(
            energy_fn, action=A0r, future=Z_dummy,
            action_mask=a_mask, future_mask=torch.zeros_like(Z_dummy[..., 0], dtype=torch.bool),
            mode="action_given_future", **self._solver_kwargs(training),
        )
        L_A = action_endpoint_loss(A_K, A_star, a_mask, cfg.loss.action_loss)
        logs = {
            "action_only/action_endpoint_loss": L_A.detach(),
            "action_only/action_valid_elems": torch.tensor(
                valid_element_count(a_mask, A_star), dtype=torch.float32
            ),
        }
        logs.update({f"diag/action_{k}": v for k, v in st.as_diagnostics().items()})
        logs["diag/steps"] = torch.tensor(
            float(cfg.solver.train_inner_steps if training else cfg.solver.eval_inner_steps)
        )
        return StageOutput(loss=cfg.loss.lambda_action * L_A, logs=logs, action_pred=A_K, state_action=st)

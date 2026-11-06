"""EnergyWAM configuration schema.

The engineering requirements sketch a YAML concept list; this module turns that
list into a real, validated, parseable schema.  Every field used anywhere in the
package is declared here so that missing keys are loud rather than silent.

Design notes
------------
* Plain dataclasses + OmegaConf.  YAML keys override dataclass defaults.
* Dimensions (``action_dim`` / ``state_dim`` / camera count / horizon) are read
  from the task config, never hard-coded to a particular robot.
* ``backend`` selects the backbone: ``tiny`` (explicit mock, CPU tests only) or
  ``qwen3vl`` (real pretrained Qwen3-VL + frozen DINOv3).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from omegaconf import OmegaConf


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------
@dataclass
class BackboneConfig:
    condition_mode: str = "qwen_vl"  # legacy; new experiments use text_dino_state
    #: ``tiny`` = deterministic mock (explicit tiny mode only);
    #: ``qwen3vl`` = real Qwen3-VL + frozen DINOv3.
    backend: str = "qwen3vl"

    # --- real Qwen3-VL ---
    #: Local path or HF id.  Defaults to the weight location on this server.
    base_vlm: str = "/data6/zzc/Qwen3-VL-4B-Instruct"
    #: Must be a backend that supports double backward.  ``eager`` is the
    #: reference; sdpa/flash may be tried but are not assumed to work.
    attn_implementation: str = "eager"
    #: dtype used for the backbone forward.  Candidates/energy stay float32.
    dtype: str = "float32"
    freeze_qwen_vision: bool = True
    tune_language_model: bool = True
    tune_qwen_vision: bool = False
    #: LoRA is an explicitly-labelled optional extra, not the main method.
    lora_enabled: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    #: Activation recomputation (use_reentrant=False) on the language trunk.
    #: Storage-only change: same math, lower peak memory.  Required for
    #: K>=2 second-order unrolls of the 4B backbone on 48GB cards.  Validated
    #: against the un-checkpointed gradients by scripts/probe_checkpoint_grads.py.
    gradient_checkpointing: bool = False

    # --- frozen DINO ---
    dino_path: str = "/data6/zzc/dinov3-vits16-pretrain-lvd1689m"
    freeze_dino: bool = True
    dino_image_size: int = 224
    dino_patch_size: int = 16
    dino_hidden_dim: int = 384
    #: tokens to drop from the front of the DINO sequence: 1 CLS + 4 registers.
    dino_num_special_tokens: int = 5
    #: asserted at load time; 224/16 = 14 -> 196 patches
    dino_num_patches: int = 196

    # --- tiny (explicit test mode only) ---
    tiny_hidden_dim: int = 64
    tiny_num_layers: int = 2
    tiny_num_heads: int = 4
    tiny_num_cond_tokens: int = 6
    tiny_future_patches: int = 4
    tiny_future_dim: int = 8
    tiny_action_dim: int = 3
    tiny_state_dim: int = 4
    tiny_num_cameras: int = 1


@dataclass
class DataConfig:
    # --- geometry (read from the task, overridable) ---
    action_dim: int = 12
    state_dim: int = 32
    action_horizon: int = 16
    num_future_frames: int = 3
    #: explicit offsets; when None they are derived from the rule below.
    future_offsets: Optional[List[int]] = None
    future_offset_rule: str = "uniform_within_action_horizon"
    num_cameras: int = 1
    #: camera index used for DINO future supervision (first version: left view)
    future_camera_index: int = 0
    camera_order: Tuple[str, ...] = ("robot0_agentview_left",)

    # --- normalisation contract ---
    #: ``min_max`` follows the RoboCasa365 PandaOmron LeRobot conversion.
    action_normalization: str = "min_max"
    state_normalization: str = "sincos"
    norm_stats_path: Optional[str] = None
    #: Which coordinate system the candidates / targets live in.
    #: ``legacy``                — the historical StarVLA contract (min_max
    #:   actions, raw DINO futures).  A missing key is ALWAYS ``legacy``, so an
    #:   old YAML / old checkpoint keeps its exact old meaning.
    #: ``paper_standardized_v1`` — paper §4.8: actions and future latents are
    #:   standardized with training-split statistics, ``init`` draws N(0, I) in
    #:   that space.  Requires a fitted statistics bundle (see
    #:   ``norm_stats_bundle_path`` / ``scripts/fit_normalization_stats.py``).
    normalization_profile: str = "legacy"
    #: epsilon of the standardized coordinate ``(x - mean) / (std + epsilon)``.
    standardization_epsilon: float = 1e-6
    #: versioned statistics bundle produced by ``scripts/fit_normalization_stats.py``
    #: (required by ``paper_standardized_v1``).
    norm_stats_bundle_path: Optional[str] = None
    #: Synthetic statistics bundles (``provenance.synthetic``) are REFUSED on the
    #: real (``qwen3vl``) backend by default — a real run must never silently
    #: train/serve on test statistics.  The tiny backend always allows them
    #: (explicit test mode).  Set this only for explicit verification runs with
    #: an injected non-production backbone; it is never on for the real 4B path.
    allow_synthetic_stats: bool = False

    # --- episode safety ---
    #: only sample starts whose full action horizon *and* future offsets lie
    #: inside the same episode.
    require_full_horizon: bool = True

    # --- DINO preprocessing version tag (cache key component) ---
    dino_preprocess_version: str = "dino_v3_vits16_224_center_norm_v1"

    # --- real dataset roots (documented; not fetched by this project) ---
    data_root_dir: str = "playground/Datasets/robocasa365"
    data_mix: str = "robocasa365_open_drawer_target_human"
    dataset_py: str = "lerobot_datasets"
    obs_image_size: Tuple[int, int] = (224, 224)
    include_state: bool = True
    #: Set when probing a non-RoboCasa365 corpus (e.g. GR1) for pipeline tests.
    dataset_identity_note: str = ""


@dataclass
class SolverConfig:
    candidate_gradient_normalization: str = "none"
    gradient_rms_epsilon: float = 1e-6
    learnable_step_sizes: bool = False
    step_size_min_ratio: float = 0.1
    step_size_max_ratio: float = 10.0
    """Inner energy-gradient refinement (the "solver").

    The reference implementation is a **full unroll** with simultaneous joint
    updates at the same point ``(A_k, Z_k)``.
    """

    #: ``full`` is the reference.  ``truncated`` exists but is OFF by default
    #: and is explicitly labelled as a deviation from the full method.
    unroll_mode: str = "full"
    train_inner_steps: int = 4
    eval_inner_steps: int = 4
    alpha_action: float = 0.05
    alpha_future: float = 0.05
    #: clip candidate gradients (per-element abs clip) before the update
    clip_action_grad: Optional[float] = None
    clip_future_grad: Optional[float] = None
    #: clamp candidates after each update (normalised space)
    action_clamp_min: Optional[float] = None
    action_clamp_max: Optional[float] = None
    future_clamp_min: Optional[float] = None
    future_clamp_max: Optional[float] = None
    #: ``simultaneous`` is the reference.  ``alternating`` is a separate,
    #: explicitly-named option and is never silently substituted.
    joint_update_mode: str = "simultaneous"
    #: maximum truncated-unroll length when unroll_mode == "truncated"
    truncation_length: int = 2
    #: optional extra per-step Langevin noise (default OFF, not part of method)
    langevin_noise_std: float = 0.0


@dataclass
class InitConfig:
    #: action init: ``normal`` | ``zeros``
    initialization_action: str = "normal"
    #: future init: ``normal`` | ``current_dino_noise``
    initialization_future: str = "normal"
    init_action_std: float = 0.1
    init_future_std: float = 0.1
    #: noise scale for ``current_dino_noise``
    init_future_noise_std: float = 0.1
    #: "normal" draws from N(0, std) in normalised action space; stats source
    #: for any mean/scale is recorded in docs/ARCHITECTURE.md.
    init_action_mean: float = 0.0
    init_future_mean: float = 0.0
    #: paper §4.8 baseline prior is N(0, I) in the standardized space.  Under
    #: ``normalization_profile='paper_standardized_v1'`` a non-(0, 1) prior is
    #: rejected unless this explicit ablation switch is set (paper §8: a valid
    #: prior ablation changes the scale/covariance *within the same coordinate*).
    allow_prior_ablation: bool = False


@dataclass
class LossConfig:
    action_loss: str = "masked_mse"
    future_loss: str = "masked_mse"
    lambda_action: float = 1.0
    lambda_future: float = 1.0
    beta_action: float = 1.0
    beta_future: float = 1.0
    #: optional cosine term on future latents; default OFF
    future_cosine_weight: float = 0.0


@dataclass
class StageConfig:
    #: ``bi`` | ``action_given_future`` | ``future_given_action`` |
    #: ``joint`` | ``direct_joint`` | ``action_only``
    stage: str = "bi"
    #: compute both directions every batch (True) or alternate per batch (False)
    bi_both_directions_each_batch: bool = True
    #: keep the bidirectional auxiliary loss on during the joint stage
    joint_use_bi_aux: bool = False
    joint_bi_aux_weight: float = 0.0


@dataclass
class TrainConfig:
    #: Simultaneously unrolled samples; distinct from effective accumulated batch.
    shared_microbatch_size: int = 1
    #: Diagnostic only: same episode/frame/direction gets the same candidate.
    fixed_candidate_initialization: bool = False
    #: Optional wall-time checkpoint cadence; zero retains save_interval steps.
    checkpoint_interval_seconds: float = 0.0
    #: Read-only per-direction energy-head gradient diagnostics before clipping.
    directional_head_diagnostics: bool = False
    fp32_master_weights: bool = False
    task_learning_rate: Optional[float] = None
    energy_head_learning_rate: Optional[float] = None
    execution_mode: str = "standard"
    gradient_sync_bucket_mib: int = 256
    seed: int = 42
    max_train_steps: int = 100
    per_device_batch_size: int = 2
    step_size_learning_rate: float = 1e-3
    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    warmup_steps: int = 0
    logging_frequency: int = 5
    save_interval: int = 100
    output_dir: str = "logs/runs"
    #: outer optimiser step only; the inner solver never touches parameters
    optimizer: str = "AdamW"
    #: Allocate zero AdamW8bit moments before the first forward for predictable peaks.
    preallocate_optimizer_state: bool = False
    #: explicit policy when switching bi -> joint
    #: ``reset`` (fresh optimiser/scheduler) | ``continue`` (keep both)
    stage_switch_optimizer_policy: str = "reset"
    stage_switch_scheduler_policy: str = "reset"
    num_workers: int = 0
    device: str = "cpu"
    #: candidates / energy accumulation / loss dtype
    candidate_dtype: str = "float32"
    #: save unroll activations to CPU RAM (parameter-aware, deduplicated)
    #: during the training forward; storage only, never differentiation.
    #: Needed for K>=2 second-order unrolls of the 4B backbone on <64GB cards.
    offload_saved_tensors: bool = False
    #: Per-GPU resident activation-storage budget within CPU saved-tensor offload.
    saved_tensor_gpu_budget_gib: float = 0.0
    #: Optional separate budgets for visible CUDA devices, overriding scalar.
    saved_tensor_gpu_budgets_gib: List[float] = field(default_factory=list)
    saved_tensor_cpu_pin_budget_gib: float = 0.0
    saved_tensor_async_offload: bool = False
    async_offload_max_inflight_gib: float = 1.0
    saved_tensor_host_cache_limit_gib: float = 0.0
    saved_tensor_deduplicate_cpu: bool = False
    #: Split language Transformer layers across these visible CUDA indices.
    #: One model / optimizer; full autograd across device copies (no DDP).
    language_model_devices: List[int] = field(default_factory=list)


@dataclass
class FrameworkConfig:
    #: Optional space-to-depth tokenization; raw candidates/targets stay intact.
    dino_patch_pack_factor: int = 1
    quadratic_candidate_prior: bool = False
    quadratic_action_weight: float = 1.0
    quadratic_future_weight: float = 1.0
    energy_head_zero_init: bool = False
    projector_output_scale: float = 1.0
    positional_encoding_scale: float = 1.0
    type_embedding_init_std: float = 1.0
    iteration_conditioning: bool = False
    iteration_embedding_dim: int = 256
    iteration_mlp_hidden: int = 256
    iteration_progress_scale: float = 1000.0
    name: str = "QwenEnergyWAM"
    #: energy head MLP
    energy_head_hidden: int = 256
    energy_head_dropout: float = 0.0
    #: number of state tokens projected from the robot state vector
    num_state_tokens: int = 1
    #: learned placeholder used when a modality is absent (action_only mode)
    use_learned_future_placeholder: bool = True
    #: token type ids
    token_type_vocab: int = 8


@dataclass
class EnergyWAMConfig:
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    data: DataConfig = field(default_factory=DataConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    init: InitConfig = field(default_factory=InitConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    stage: StageConfig = field(default_factory=StageConfig)
    trainer: TrainConfig = field(default_factory=TrainConfig)
    framework: FrameworkConfig = field(default_factory=FrameworkConfig)
    #: free-form provenance
    run_id: str = "energywam"
    notes: str = ""


# ---------------------------------------------------------------------------
# Loading / validation
# ---------------------------------------------------------------------------
def _apply_yaml(dc, node) -> None:
    for f in dataclasses.fields(dc):
        if not OmegaConf.is_config(node) or f.name not in node:
            continue
        raw = OmegaConf.select(node, f.name)
        current = getattr(dc, f.name)
        if dataclasses.is_dataclass(current) and not isinstance(current, type):
            if raw is None:
                continue
            _apply_yaml(current, raw)
        else:
            setattr(dc, f.name, OmegaConf.to_container(raw, resolve=False) if OmegaConf.is_config(raw) else raw)


def config_from_dict(d: Dict[str, Any]) -> EnergyWAMConfig:
    cfg = EnergyWAMConfig()
    node = OmegaConf.create(d) if isinstance(d, dict) else d
    _apply_yaml(cfg, node)
    validate_config(cfg)
    return cfg


def load_config(path: str | Path, overrides: Sequence[str] | None = None) -> EnergyWAMConfig:
    node = OmegaConf.load(str(path))
    if overrides:
        node = OmegaConf.merge(node, OmegaConf.from_dotlist(list(overrides)))
    return config_from_dict(OmegaConf.to_container(node, resolve=False))


def validate_config(cfg: EnergyWAMConfig) -> None:
    if cfg.trainer.task_learning_rate is not None:
        import math
        if not math.isfinite(cfg.trainer.task_learning_rate) or cfg.trainer.task_learning_rate <= 0:
            raise ValueError("task_learning_rate must be finite and positive")
    if cfg.trainer.energy_head_learning_rate is not None:
        import math
        if not math.isfinite(cfg.trainer.energy_head_learning_rate) or cfg.trainer.energy_head_learning_rate <= 0:
            raise ValueError("energy_head_learning_rate must be finite and positive")
    if cfg.backbone.backend not in ("tiny", "qwen3vl"):
        raise ValueError(f"backbone.backend must be 'tiny' or 'qwen3vl', got {cfg.backbone.backend!r}")
    if cfg.stage.stage not in (
        "bi",
        "action_given_future",
        "future_given_action",
        "joint",
        "direct_joint",
        "action_only",
    ):
        raise ValueError(f"unknown stage {cfg.stage.stage!r}")
    if cfg.backbone.condition_mode not in ("qwen_vl", "text_dino_state"):
        raise ValueError("unknown backbone.condition_mode")
    if cfg.backbone.condition_mode == "text_dino_state" and cfg.backbone.tune_qwen_vision:
        raise ValueError("text_dino_state removes Qwen vision; tune_qwen_vision must be false")
    import math
    if not math.isfinite(cfg.trainer.checkpoint_interval_seconds) or cfg.trainer.checkpoint_interval_seconds < 0:
        raise ValueError("checkpoint_interval_seconds must be finite and nonnegative")
    if type(cfg.framework.dino_patch_pack_factor) is not int or cfg.framework.dino_patch_pack_factor not in (1, 2):
        raise ValueError("dino_patch_pack_factor must be 1 or 2")
    for value in [cfg.framework.projector_output_scale, cfg.framework.positional_encoding_scale,
                  cfg.framework.type_embedding_init_std]:
        if not math.isfinite(value) or value <= 0:
            raise ValueError("token scales must be finite and positive")
    if cfg.framework.iteration_embedding_dim < 2 or cfg.framework.iteration_embedding_dim % 2:
        raise ValueError("iteration_embedding_dim must be positive and even")
    if cfg.framework.iteration_mlp_hidden < 1 or cfg.framework.iteration_progress_scale <= 0:
        raise ValueError("iteration MLP width and progress scale must be positive")
    if cfg.solver.learnable_step_sizes:
        import math
        vals = [cfg.solver.alpha_action, cfg.solver.alpha_future,
                cfg.solver.step_size_min_ratio, cfg.solver.step_size_max_ratio,
                cfg.trainer.step_size_learning_rate]
        if not all(math.isfinite(v) and v > 0 for v in vals):
            raise ValueError("learnable step sizes require finite positive settings")
        if not cfg.solver.step_size_min_ratio < 1 < cfg.solver.step_size_max_ratio:
            raise ValueError("step-size ratio bounds must straddle 1")
    if cfg.solver.candidate_gradient_normalization not in ("none", "rms"):
        raise ValueError("unknown candidate gradient normalization")
    if cfg.solver.gradient_rms_epsilon <= 0:
        raise ValueError("gradient_rms_epsilon must be positive")
    if cfg.framework.quadratic_candidate_prior:
        import math
        if not all(math.isfinite(x) and x >= 0 for x in [cfg.framework.quadratic_action_weight, cfg.framework.quadratic_future_weight]):
            raise ValueError("quadratic prior weights must be finite and nonnegative")
        if cfg.backbone.backend != "qwen3vl":
            raise ValueError("current-visual quadratic prior requires the real DINO condition path")
        if cfg.solver.candidate_gradient_normalization != "none":
            raise ValueError("quadratic-prior pilot uses raw GD; RMS is a separate ablation")
        if not all(math.isfinite(x) and x > 0 for x in [cfg.solver.alpha_action,cfg.solver.alpha_future]):
            raise ValueError("quadratic-prior precision needs positive step anchors")
    if cfg.solver.joint_update_mode not in ("simultaneous", "alternating"):
        raise ValueError("solver.joint_update_mode must be 'simultaneous' or 'alternating'")
    if cfg.solver.unroll_mode not in ("full", "truncated"):
        raise ValueError("solver.unroll_mode must be 'full' or 'truncated'")
    if cfg.solver.unroll_mode == "truncated":
        # explicit, named deviation — never silently enabled
        if cfg.solver.truncation_length < 1:
            raise ValueError("solver.truncation_length must be >= 1")
    if cfg.data.num_future_frames != 3:
        # offsets are defined for the 3-point scheme of the research plan
        raise ValueError("data.num_future_frames must be 3 in this version")
    offs = resolve_future_offsets(cfg.data)
    if any(o <= 0 for o in offs):
        raise ValueError(f"future offsets must be positive, got {offs}")
    if sorted(offs) != list(offs) or len(set(offs)) != len(offs):
        raise ValueError(f"future offsets must be strictly increasing, got {offs}")
    if max(offs) > cfg.data.action_horizon:
        raise ValueError(f"future offsets {offs} exceed action_horizon={cfg.data.action_horizon}")
    if cfg.init.initialization_action not in ("normal", "zeros"):
        raise ValueError(f"unknown action init {cfg.init.initialization_action!r}")
    if cfg.init.initialization_future not in ("normal", "current_dino_noise"):
        raise ValueError(f"unknown future init {cfg.init.initialization_future!r}")
    if cfg.backbone.backend == "tiny" and cfg.backbone.attn_implementation not in ("eager", "math"):
        # tiny backbone implements attention itself; keep the flag meaningful
        pass

    # --- normalization profile (paper §4.8 standardized coordinate) ---
    # Imported lazily: config.py must stay importable with no data machinery.
    from .data.standardization import PROFILE_PAPER, PROFILES

    if cfg.data.normalization_profile not in PROFILES:
        raise ValueError(
            f"data.normalization_profile must be one of {PROFILES}, "
            f"got {cfg.data.normalization_profile!r}"
        )
    if not (float(cfg.data.standardization_epsilon) > 0.0):
        raise ValueError(
            f"data.standardization_epsilon must be > 0, got {cfg.data.standardization_epsilon}"
        )
    if cfg.data.normalization_profile == PROFILE_PAPER:
        if cfg.data.action_normalization != "zscore":
            raise ValueError(
                "data.normalization_profile='paper_standardized_v1' requires "
                f"data.action_normalization='zscore', got {cfg.data.action_normalization!r}"
            )
        if not cfg.init.allow_prior_ablation:
            # Approved baseline: N(0, I) in the standardized space (paper §4.8).
            if cfg.init.initialization_action == "normal" and (
                cfg.init.init_action_mean != 0.0 or cfg.init.init_action_std != 1.0
            ):
                raise ValueError(
                    "paper_standardized_v1 uses the N(0, I) action prior: got "
                    f"init_action_mean={cfg.init.init_action_mean}, "
                    f"init_action_std={cfg.init.init_action_std}.  Set "
                    "init.allow_prior_ablation=true only for an explicit prior "
                    "ablation (paper §8) in the SAME coordinate."
                )
            if cfg.init.initialization_future == "normal" and (
                cfg.init.init_future_mean != 0.0 or cfg.init.init_future_std != 1.0
            ):
                raise ValueError(
                    "paper_standardized_v1 uses the N(0, I) future prior: got "
                    f"init_future_mean={cfg.init.init_future_mean}, "
                    f"init_future_std={cfg.init.init_future_std}.  Set "
                    "init.allow_prior_ablation=true only for an explicit prior "
                    "ablation (paper §8) in the SAME coordinate."
                )
    else:
        if cfg.data.action_normalization == "zscore":
            raise ValueError(
                "data.action_normalization='zscore' requires "
                "data.normalization_profile='paper_standardized_v1'"
            )
        if cfg.data.norm_stats_bundle_path is not None:
            raise ValueError(
                "data.norm_stats_bundle_path is only used by "
                "data.normalization_profile='paper_standardized_v1'; the legacy "
                "profile reads data.norm_stats_path / meta/stats.json"
            )


def geometry_from_cfg(cfg: EnergyWAMConfig) -> Dict[str, int]:
    """Candidate geometry derived from the config alone (no backbone needed).

    Kept here (not in the model module) so data-side statistics validation can
    check dimensions against exactly the same numbers the model will build.
    """
    if cfg.backbone.backend == "tiny":
        b = cfg.backbone
        return {
            "action_dim": b.tiny_action_dim,
            "state_dim": b.tiny_state_dim,
            "future_dim": b.tiny_future_dim,
            "future_patches": b.tiny_future_patches,
            "num_cameras": b.tiny_num_cameras,
            "action_horizon": cfg.data.action_horizon,
            "num_future_frames": cfg.data.num_future_frames,
            "dino_dim": b.tiny_future_dim,
        }
    return {
        "action_dim": cfg.data.action_dim,
        "state_dim": cfg.data.state_dim,
        "future_dim": cfg.backbone.dino_hidden_dim,
        "future_patches": cfg.backbone.dino_num_patches,
        "num_cameras": cfg.data.num_cameras,
        "action_horizon": cfg.data.action_horizon,
        "num_future_frames": cfg.data.num_future_frames,
        "dino_dim": cfg.backbone.dino_hidden_dim,
    }


def resolve_future_offsets(data: DataConfig) -> List[int]:
    """Return the 3 future offsets for the configured action horizon.

    Rule ``uniform_within_action_horizon`` -> ``round(i*H/3), i=1,2,3``.
    For H=16 this is [5, 11, 16].
    """
    if data.future_offsets is not None:
        return [int(x) for x in data.future_offsets]
    if data.future_offset_rule != "uniform_within_action_horizon":
        raise ValueError(f"unknown future_offset_rule {data.future_offset_rule!r}")
    H = int(data.action_horizon)
    F = int(data.num_future_frames)
    return [max(1, round(i * H / F)) for i in range(1, F + 1)]


def to_plain_dict(cfg: EnergyWAMConfig) -> Dict[str, Any]:
    return dataclasses.asdict(cfg)

"""Torch ``Dataset`` / ``DataLoader`` for real EnergyWAM training.

Fixes the defects in the first cut of ``scripts/train.py``:

* the frozen DINO future target ``Z*`` is actually extracted from the future RGB
  under ``no_grad`` and **detached** (previously missing entirely, which made
  every real non-``action_only`` run crash);
* action normalisation is **mandatory** — a missing statistics file raises
  instead of silently training on raw actions;
* robot state uses the configured ``state_normalization`` (``sincos`` expands
  the raw 16-d PandaOmron state to 32, matching the StarVLA training contract);
* camera order and future offsets come from the config / ``resolve_future_offsets``,
  never hard-coded;
* the loader is a streaming ``DataLoader`` over the whole dataset.  Only the
  *smoke* entry point limits the sample count.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..config import EnergyWAMConfig, resolve_future_offsets
from .episode_dataset import DatasetSpec, LeRobotEpisodeReader, build_sample
from .future_offsets import plan_future_frames
from .normalization import NormStats, load_stats_from_lerobot_meta, normalize
from .standardization import (
    FUTURE_SPACE_RAW,
    FUTURE_SPACE_STANDARDIZED,
    PROFILE_PAPER,
    NormalizationBundle,
    StatsError,
    ZScoreStats,
    load_bundle,
)


def sincos_state(q: np.ndarray, out_dim: int) -> np.ndarray:
    """Expand a raw state vector to ``out_dim`` with sin/cos features.

    The layout is **``[sin(q), cos(q)]``** — first the sine block over the whole
    raw vector, then the cosine block (verified numerically in
    ``tests/test_review_regressions.py`` and
    ``tests/test_entrypoint_regressions.py``).  Mirrors the StarVLA
    ``StateActionSinCosTransform`` contract used by the RoboCasa365 PandaOmron
    config: raw 16-d -> 32-d.
    """
    q = np.asarray(q, dtype="float32").reshape(-1)
    expected = 2 * q.shape[0]
    if out_dim != expected:
        raise ValueError(
            f"sin/cos state of raw dim {q.shape[0]} requires out_dim={expected}, got {out_dim}"
        )
    return np.concatenate([np.sin(q), np.cos(q)]).astype("float32", copy=False)


class _CurrentFrameOnlyReader:
    """Reader facade used when ``require_future=False`` (offline evaluation).

    ``build_sample`` unconditionally decodes the future RGB frames of the
    planned offsets.  Offline evaluation must not read real future data (and
    needs no DINO future target at all), so this facade delegates every
    current-frame access and answers future-frame reads with a zero placeholder
    **without decoding any video**.  Those placeholder pixels never leave the
    sample builder — eval examples only carry ``image`` / ``lang`` / ``state`` /
    ``action``.
    """

    def __init__(self, reader: LeRobotEpisodeReader, frame_index: int):
        self._reader = reader
        self._frame_index = int(frame_index)
        self._current_frame = None

    def __getattr__(self, name):
        # everything except read_frame is the real reader
        return getattr(self._reader, name)

    def read_frame(self, episode_index, camera_key, frame_index):
        if int(frame_index) == self._frame_index:
            frame = self._reader.read_frame(episode_index, camera_key, frame_index)
            self._current_frame = frame  # shape/dtype template for the placeholders
            return frame
        if self._current_frame is None:
            raise RuntimeError("future frame requested before any current frame")
        return np.zeros_like(self._current_frame)


class EnergyWAMEpisodeDataset(Dataset):
    """Streaming dataset of episode-safe EnergyWAM samples.

    Each item is a dict ready for :meth:`QwenEnergyWAM.forward`.

    ``require_future=False`` is the **offline-eval** contract: no future RGB is
    decoded, no DINO extractor is needed, and the item carries no future
    supervision (``future_target`` / ``future_mask`` are empty).  Training
    always uses ``require_future=True``.

    ``future_standardizer`` (a :class:`ZScoreStats` from the training-split
    bundle) switches the output coordinate to the paper §4.8 standardized
    space: ``future_target`` becomes ``tilde Z*`` and the item carries a
    ``future_target_space`` marker so downstream consumers can verify the
    coordinate instead of guessing.  Without it the output stays raw DINO
    features (the legacy contract).

    ``episode_indices`` restricts the sample index to an explicit episode list
    (the paper-profile TRAIN episodes recorded in the statistics bundle; or an
    explicit held-out selection for evaluation).  It is validated: empty lists,
    duplicates, non-integer / missing episodes all raise — there is NO fallback
    to "all episodes".  ``expected_dataset_identity`` (the bundle's recorded
    dataset identity) must match ``reader.dataset_identity`` exactly.
    """

    def __init__(
        self,
        cfg: EnergyWAMConfig,
        *,
        reader: LeRobotEpisodeReader,
        action_stats: NormStats,
        state_stats: Optional[NormStats],
        dino=None,
        max_samples: Optional[int] = None,
        require_future: bool = True,
        future_standardizer: Optional[ZScoreStats] = None,
        episode_indices: Optional[Sequence[int]] = None,
        expected_dataset_identity: Optional[str] = None,
    ):
        self.cfg = cfg
        self.reader = reader
        self.action_stats = action_stats
        self.state_stats = state_stats
        self.dino = dino
        self.require_future = require_future
        self.future_standardizer = future_standardizer
        if expected_dataset_identity is not None and str(expected_dataset_identity) != str(
            reader.dataset_identity
        ):
            raise StatsError(
                f"dataset identity mismatch: statistics were fitted on "
                f"{expected_dataset_identity!r} but this reader is "
                f"{reader.dataset_identity!r}; refusing to train/evaluate across datasets"
            )
        self.spec = DatasetSpec(
            action_horizon=cfg.data.action_horizon,
            future_offsets=tuple(resolve_future_offsets(cfg.data)),
            num_cameras=cfg.data.num_cameras,
            camera_index=int(cfg.data.future_camera_index),
            camera_keys=tuple(cfg.data.camera_order) if cfg.data.camera_order else None,
            require_full_horizon=cfg.data.require_full_horizon,
        )
        self._index: List[Tuple[int, int]] = []
        self._build_index(max_samples, episode_indices)

    # ------------------------------------------------------------------
    def _build_index(self, max_samples: Optional[int], episode_indices=None) -> None:
        offs = list(self.spec.future_offsets)
        H = self.spec.action_horizon

        if episode_indices is not None:
            raw = list(episode_indices)
            if not raw:
                raise StatsError(
                    "empty episode list: refusing to fall back to all episodes"
                )
            eps = []
            for e in raw:
                if isinstance(e, bool) or not isinstance(e, (int, np.integer)):
                    raise StatsError(f"invalid episode index {e!r} (must be an integer)")
                eps.append(int(e))
            if any(e < 0 for e in eps):
                raise StatsError(f"invalid episode index in {eps}: negative")
            if len(set(eps)) != len(eps):
                raise StatsError(f"duplicate episode indices in {eps}")
            by_idx = {int(ep["episode_index"]): ep for ep in self.reader.episodes}
            missing = [e for e in eps if e not in by_idx]
            if missing:
                raise StatsError(
                    f"episode(s) {missing} not present in the dataset "
                    f"({self.reader.dataset_identity})"
                )
            episodes = [by_idx[e] for e in eps]
        else:
            episodes = list(self.reader.episodes)

        for ep in episodes:
            ep_idx = int(ep["episode_index"])
            L = int(ep["length"])
            for t in range(L):
                plan = plan_future_frames(
                    t, L, H, offs, require_full_horizon=self.cfg.data.require_full_horizon
                )
                if plan.valid:
                    self._index.append((ep_idx, t))
            if max_samples is not None and len(self._index) >= max_samples:
                self._index = self._index[:max_samples]
                return
        if not self._index:
            raise RuntimeError(
                f"no valid (episode, start) pairs in {self.reader.root} for "
                f"H={H}, offsets={offs}, require_full_horizon="
                f"{self.cfg.data.require_full_horizon}"
            )

    def __len__(self) -> int:
        return len(self._index)

    # ------------------------------------------------------------------
    def _future_dino_target(self, future_images: List[List[Any]], valid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """``[F, V, P, D]`` frozen-DINO target + ``[F, V, P]`` mask.

        Extracted under ``no_grad`` and detached: the DINO target is a constant
        for the energy optimisation, never a differentiable branch.
        """
        if self.dino is None:
            raise RuntimeError(
                "EnergyWAMEpisodeDataset requires a frozen DINO extractor to build "
                "the future target Z*.  Pass `dino=` when constructing the dataset."
            )
        F = len(future_images)
        V = len(future_images[0])
        flat = [img for frame in future_images for img in frame]
        with torch.no_grad():
            feats = self.dino.encode_images(flat, device="cpu")  # [F*V, P, D]
        feats = feats.detach().float()
        P, D = feats.shape[1], feats.shape[2]
        Z = feats.view(F, V, P, D).numpy()
        mask = np.repeat(np.asarray(valid, dtype="float32").reshape(F, 1, 1), V, axis=1)
        mask = np.repeat(mask, P, axis=2)  # [F, V, P]
        if self.future_standardizer is not None:
            # paper §4.8: raw DINO -> standardized coordinate, then drop the
            # masked (invalid) positions entirely so they can never participate.
            Z = self.future_standardizer.normalize(Z).astype("float32", copy=False)
            Z[mask == 0] = 0.0
        return Z, mask

    def __getitem__(self, i: int) -> Dict[str, Any]:
        ep_idx, start = self._index[i]
        reader = self.reader
        if not self.require_future:
            reader = _CurrentFrameOnlyReader(self.reader, start)
        sample = build_sample(
            reader,
            ep_idx,
            start,
            self.spec,
            action_stats=self.action_stats,
            state_stats=None,  # state is handled below (sin/cos, not minmax)
        )
        if sample is None:
            raise RuntimeError(f"index {i} ({ep_idx},{start}) unexpectedly invalid")

        cfg = self.cfg
        q = sample.state
        if cfg.data.state_normalization == "sincos":
            q = sincos_state(q, cfg.data.state_dim)
        elif cfg.data.state_normalization in ("none", None):
            if q.shape[0] != cfg.data.state_dim:
                raise ValueError(
                    f"state dim {q.shape[0]} != configured {cfg.data.state_dim}; "
                    f"set data.state_normalization='sincos' or fix data.state_dim"
                )
        else:
            if self.state_stats is None:
                raise ValueError(
                    f"state_normalization={cfg.data.state_normalization!r} needs state statistics"
                )
            q = normalize(q, self.state_stats)

        if self.require_future:
            Z, z_mask = self._future_dino_target(sample.future_images, sample.future_valid)
        else:
            # offline eval: no future GT / DINO supervision at all
            Z = np.zeros((0,), dtype="float32")
            z_mask = np.zeros((0,), dtype="float32")

        a = sample.action.astype("float32")
        standardized = getattr(self.action_stats, "mode", "") == "zscore"
        if standardized:
            # paper profile: drop masked action elements after the transform so
            # padding/invalid slots carry no value at all (mask-first hygiene).
            a = a.copy()
            a[sample.action_mask == 0] = 0.0

        out = {
            "images": [sample.images],
            "langs": [sample.language],
            "states": [np.asarray(q, dtype="float32")],
            "action_target": a,
            "action_mask": sample.action_mask.astype("float32"),
            "future_target": Z.astype("float32"),
            "future_mask": z_mask.astype("float32"),
            "future_offsets": np.asarray(sample.future_offsets, dtype="int64"),
            "dataset_identity": sample.dataset_identity,
            "episode_index": sample.episode_index,
            "frame_index": sample.frame_index,
            "action_target_space": ("standardized" if standardized else "model_space"),
        }
        if self.require_future:
            out["future_target_space"] = (
                FUTURE_SPACE_STANDARDIZED
                if self.future_standardizer is not None
                else FUTURE_SPACE_RAW
            )
        return out


def collate_energy_batch(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate per-sample dicts into one EnergyWAM batch."""
    out: Dict[str, Any] = {
        "images": [it["images"][0] for it in items],
        "langs": [it["langs"][0] for it in items],
        "states": [it["states"][0] for it in items],
        "action_target": torch.as_tensor(np.stack([it["action_target"] for it in items])),
        "action_mask": torch.as_tensor(np.stack([it["action_mask"] for it in items])),
        "future_target": torch.as_tensor(np.stack([it["future_target"] for it in items])),
        "future_mask": torch.as_tensor(np.stack([it["future_mask"] for it in items])),
        "future_offsets": torch.as_tensor(np.stack([it["future_offsets"] for it in items])),
        "dataset_identity": items[0]["dataset_identity"],
        "episode_index": [it["episode_index"] for it in items],
        "frame_index": [it["frame_index"] for it in items],
        # coordinate markers: what space the targets above live in
        "action_target_space": items[0].get("action_target_space"),
        "future_target_space": items[0].get("future_target_space"),
    }
    if any("current_dino" in item for item in items):
        if not all("current_dino" in item for item in items):
            raise ValueError("partial current-DINO cache in one batch")
        out["current_dino"] = torch.as_tensor(np.stack([it["current_dino"] for it in items]))
    return out


# ---------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------
def require_action_stats(reader: LeRobotEpisodeReader, cfg: EnergyWAMConfig) -> NormStats:
    """Load action statistics or **fail loudly**.

    Training on raw actions while the model is documented to operate in the
    min-max normalised space would silently invalidate every result, so a
    missing statistics file is an error, not a warning.
    """
    if cfg.data.norm_stats_path:
        stats_file = Path(cfg.data.norm_stats_path)
    else:
        stats_file = reader.root / "meta" / "stats.json"
    if not stats_file.exists():
        raise FileNotFoundError(
            f"normalisation statistics not found at {stats_file}.  EnergyWAM refuses "
            "to train on un-normalised actions; set data.norm_stats_path to a "
            "stats.json with an 'action' block."
        )
    try:
        return load_stats_from_lerobot_meta(stats_file, "action", mode=cfg.data.action_normalization)
    except KeyError as exc:
        raise KeyError(
            f"{stats_file} has no 'action' statistics block, which is required for "
            f"data.action_normalization={cfg.data.action_normalization!r}.  {exc}"
        ) from exc


def resolve_dataset_root(cfg: EnergyWAMConfig) -> Path:
    """Resolve the LeRobot dataset directory, honouring ``data.data_mix``.

    Tries ``data_root_dir/data_mix`` first (the StarVLA data-mix convention),
    then ``data_root_dir`` itself.  Everything that reads a real dataset goes
    through here so a config's ``data_mix`` can never be silently ignored.
    """
    root = Path(cfg.data.data_root_dir)
    candidates = [root / cfg.data.data_mix, root] if cfg.data.data_mix else [root]
    for c in candidates:
        if (c / "meta" / "info.json").exists():
            return c
    raise FileNotFoundError(
        f"no LeRobot dataset under {cfg.data.data_root_dir} "
        f"(tried {candidates}).  RoboCasa365 is not present on this server; "
        "see docs/ARCHITECTURE.md 'data availability'."
    )


def _load_state_stats(reader: LeRobotEpisodeReader, cfg: EnergyWAMConfig) -> Optional[NormStats]:
    if cfg.data.state_normalization in ("none", None, "sincos"):
        return None
    stats_file = Path(cfg.data.norm_stats_path) if cfg.data.norm_stats_path else reader.root / "meta" / "stats.json"
    return load_stats_from_lerobot_meta(stats_file, "observation.state", mode=cfg.data.state_normalization)


def require_future_for_stage(stage: str) -> bool:
    """Whether a training stage reads the future ground truth at all.

    ``action_only`` trains the independent ``E(C, A)`` path, which cannot see
    any future input (:meth:`EnergyWAMCore.energy_action_only` masks the whole
    future block).  Decoding future RGB and running frozen DINO on it for that
    stage is pure waste and needlessly reads real future data, so the default
    for it is ``False``.  Every other stage supervises ``Z_K`` against ``Z*``
    and keeps ``True``.
    """
    return stage != "action_only"


def run_geometry_validation_kwargs(
    cfg: EnergyWAMConfig, *, cameras_available: Optional[Sequence[str]] = None
) -> Dict[str, Any]:
    """Run-side provenance checks: camera selection, DINO identity, synthetic policy.

    The camera selection is resolved as ``camera_order[future_camera_index : +num_cameras]``
    (the same slice the dataset and the statistics fitter use) — validating only
    the full ``camera_order`` would not catch a different selection index.
    """
    from .standardization import select_cameras

    cams_used = select_cameras(
        cfg.data.camera_order,
        cfg.data.future_camera_index,
        cfg.data.num_cameras,
        available=cameras_available,
    )
    dino_model_path = None
    dino_config = None
    if cfg.backbone.backend == "qwen3vl":
        b = cfg.backbone
        dino_model_path = str(b.dino_path)
        dino_config = {
            "hidden_dim": int(b.dino_hidden_dim),
            "num_patches": int(b.dino_num_patches),
            "num_special_tokens": int(b.dino_num_special_tokens),
            "image_size": int(b.dino_image_size),
            "patch_size": int(b.dino_patch_size),
        }
    return {
        "camera_order": list(cfg.data.camera_order) if cfg.data.camera_order else None,
        "cameras_used": cams_used,
        "camera_index": int(cfg.data.future_camera_index),
        "num_cameras": int(cfg.data.num_cameras),
        "action_horizon": int(cfg.data.action_horizon),
        "future_offsets": resolve_future_offsets(cfg.data),
        "dino_preprocess_version": cfg.data.dino_preprocess_version,
        "dino_model_path": dino_model_path,
        "dino_config": dino_config,
        "epsilon": float(cfg.data.standardization_epsilon),
        "allow_synthetic": bool(
            cfg.backbone.backend == "tiny" or cfg.data.allow_synthetic_stats
        ),
    }


def resolve_stats_bundle(
    cfg: EnergyWAMConfig,
    *,
    stats_bundle: Optional[NormalizationBundle] = None,
    require_future: bool = True,
    cameras_available: Optional[Sequence[str]] = None,
) -> NormalizationBundle:
    """Resolve + validate the paper-profile statistics bundle for a data source.

    Resolution order: an explicitly supplied bundle (e.g. the one embedded in
    the checkpoint being restored) is authoritative; ``data.norm_stats_bundle_path``
    is used otherwise.  If a bundle file exists but **differs** from the
    authoritative one the call is refused (two fits must never be mixed); a
    stale/missing path is ignored so restoring a checkpoint never depends on
    it.  Geometry/provenance is validated against the run configuration
    (including the resolved camera selection and the DINO identity/config);
    a mismatch raises :class:`StatsError` — there is no fallback to raw
    coordinates, random statistics or a val/test estimate.  Synthetic bundles
    are refused on the real backend unless ``data.allow_synthetic_stats``.
    """
    if cfg.data.normalization_profile != PROFILE_PAPER:
        raise StatsError(
            "resolve_stats_bundle is only used by "
            "data.normalization_profile='paper_standardized_v1'"
        )
    if stats_bundle is not None:
        if not isinstance(stats_bundle, NormalizationBundle):
            raise StatsError(f"stats_bundle must be a NormalizationBundle, got {type(stats_bundle)}")
        path = cfg.data.norm_stats_bundle_path
        if path and Path(path).exists():
            other = load_bundle(path)
            if other.fingerprint != stats_bundle.fingerprint:
                raise StatsError(
                    f"the configured statistics bundle {path} differs from the statistics "
                    "in use (e.g. embedded in the checkpoint being restored); refusing to "
                    "mix two different fits"
                )
        bundle = stats_bundle
    elif cfg.data.norm_stats_bundle_path:
        bundle = load_bundle(cfg.data.norm_stats_bundle_path)
    else:
        raise StatsError(
            "data.normalization_profile='paper_standardized_v1' requires fitted "
            "training-split statistics: set data.norm_stats_bundle_path "
            "(scripts/fit_normalization_stats.py) or restore a checkpoint that "
            "embeds its bundle.  Refusing to fall back to raw coordinates, random "
            "statistics or a val/test estimate."
        )
    from ..config import geometry_from_cfg

    geom = geometry_from_cfg(cfg)
    bundle.validate_geometry(
        action_dim=geom["action_dim"],
        future_dim=geom["future_dim"],
        require_future=require_future,
        **run_geometry_validation_kwargs(cfg, cameras_available=cameras_available),
    )
    return bundle


def train_episodes_from_bundle(bundle: NormalizationBundle) -> List[int]:
    """The bundle's recorded TRAIN episodes — required for real training.

    Missing/empty/duplicate/invalid lists raise; there is NO fallback to "all
    episodes" (that would silently train on val/test data).
    """
    prov = bundle.provenance or {}
    raw = (prov.get("train_episodes") or {}).get("episode_indices")
    if raw is None:
        raise StatsError(
            "statistics bundle records no train_episodes.episode_indices; refusing "
            "to train on unspecified episodes (no fallback to all episodes)"
        )
    eps = []
    for e in raw:
        if isinstance(e, bool) or not isinstance(e, (int, np.integer)):
            raise StatsError(f"invalid train episode index {e!r} in the statistics bundle")
        eps.append(int(e))
    if not eps:
        raise StatsError(
            "statistics bundle records an EMPTY train episode list; refusing to "
            "train on unspecified episodes (no fallback to all episodes)"
        )
    if len(set(eps)) != len(eps):
        raise StatsError(f"duplicate train episode indices in the statistics bundle: {eps}")
    return eps


def build_dataloader(
    cfg: EnergyWAMConfig,
    *,
    dino=None,
    max_samples: Optional[int] = None,
    shuffle: bool = True,
    num_workers: Optional[int] = None,
    require_future: Optional[bool] = None,
    stats_bundle: Optional[NormalizationBundle] = None,
) -> DataLoader:
    """Streaming training DataLoader over the resolved dataset root.

    ``require_future=False`` is the offline-eval contract: no future RGB is
    decoded, no DINO extractor is needed and the items carry no future
    supervision.  ``require_future=None`` (default) infers the flag from the
    configured stage via :func:`require_future_for_stage`; an explicit True/False
    always wins.

    Under ``normalization_profile='paper_standardized_v1'`` the targets come out
    in the standardized coordinate (``stats_bundle`` / ``data.norm_stats_bundle_path``);
    pass the bundle embedded in a restored checkpoint as ``stats_bundle`` so the
    loader can never overwrite it with a different on-disk fit.
    """
    if require_future is None:
        require_future = require_future_for_stage(cfg.stage.stage)
    reader = LeRobotEpisodeReader(resolve_dataset_root(cfg))

    future_standardizer = None
    episode_indices = None
    expected_identity = None
    if cfg.data.normalization_profile == PROFILE_PAPER:
        bundle = resolve_stats_bundle(
            cfg, stats_bundle=stats_bundle, require_future=require_future,
            cameras_available=reader.camera_keys,
        )
        action_stats = bundle.action.to_norm_stats()
        future_standardizer = bundle.future if (require_future and bundle.future is not None) else None
        # training is restricted to the episodes the statistics were fitted on
        episode_indices = train_episodes_from_bundle(bundle)
        expected_identity = (bundle.provenance or {}).get("dataset_identity")
    else:
        if stats_bundle is not None:
            raise StatsError(
                "stats_bundle was passed but data.normalization_profile='legacy'"
            )
        action_stats = require_action_stats(reader, cfg)
    state_stats = _load_state_stats(reader, cfg)

    ds = EnergyWAMEpisodeDataset(
        cfg,
        reader=reader,
        action_stats=action_stats,
        state_stats=state_stats,
        dino=dino,
        max_samples=max_samples,
        require_future=require_future,
        future_standardizer=future_standardizer,
        episode_indices=episode_indices,
        expected_dataset_identity=expected_identity,
    )
    return DataLoader(
        ds,
        batch_size=cfg.trainer.per_device_batch_size,
        shuffle=shuffle,
        num_workers=int(num_workers if num_workers is not None else cfg.trainer.num_workers),
        collate_fn=collate_energy_batch,
        drop_last=False,
    )


def build_eval_examples(
    cfg: EnergyWAMConfig,
    *,
    reader: Optional[LeRobotEpisodeReader] = None,
    max_samples: Optional[int] = 32,
    stats_bundle: Optional[NormalizationBundle] = None,
    episode_indices: Optional[Sequence[int]] = None,
):
    """Build offline-eval examples in the ``predict_action`` example contract.

    One dict per sample:

    * ``image``  — list of per-camera RGB arrays at the current frame, in the
      configured ``data.camera_order``;
    * ``lang``   — the episode instruction;
    * ``state``  — the state under the configured ``state_normalization``
      (``sincos`` -> ``[sin(q), cos(q)]``);
    * ``action`` / ``action_mask`` — the demonstration chunk normalised with
      **exactly** the training statistics (legacy: :func:`require_action_stats`;
      paper profile: the training-split bundle), i.e. the space the model
      predicts in and the space ``offline_eval`` compares in.

    Future frames are never decoded and no DINO extractor is required
    (``require_future=False``): offline evaluation must not read real future
    data.  ``reader`` is injectable so tests can drive this with a fake reader.

    Evaluation is **not** restricted to the training episodes: pass
    ``episode_indices`` explicitly to evaluate a held-out selection; without it
    the caller must not claim held-out results (``scripts/evaluate.py`` reports
    this explicitly).
    """
    if reader is None:
        reader = LeRobotEpisodeReader(resolve_dataset_root(cfg))
    future_standardizer = None
    expected_identity = None
    if cfg.data.normalization_profile == PROFILE_PAPER:
        bundle = resolve_stats_bundle(
            cfg, stats_bundle=stats_bundle, require_future=False,
            cameras_available=reader.camera_keys,
        )
        action_stats = bundle.action.to_norm_stats()
        expected_identity = (bundle.provenance or {}).get("dataset_identity")
    else:
        if stats_bundle is not None:
            raise StatsError(
                "stats_bundle was passed but data.normalization_profile='legacy'"
            )
        action_stats = require_action_stats(reader, cfg)
    state_stats = _load_state_stats(reader, cfg)
    ds = EnergyWAMEpisodeDataset(
        cfg,
        reader=reader,
        action_stats=action_stats,
        state_stats=state_stats,
        dino=None,
        max_samples=max_samples,
        require_future=False,
        future_standardizer=future_standardizer,
        episode_indices=episode_indices,
        expected_dataset_identity=expected_identity,
    )
    examples = []
    for i in range(len(ds)):
        it = ds[i]
        examples.append(
            {
                "image": it["images"][0],
                "lang": it["langs"][0],
                "state": it["states"][0],
                "action": it["action_target"],
                "action_mask": it["action_mask"],
            }
        )
    return examples, action_stats

"""Offline fitting of the paper §4.8 statistics bundle from a training split.

Consumed by ``scripts/fit_normalization_stats.py``.  The rules enforced here:

* statistics are fitted on **training episodes only** (an explicit episode
  manifest); val/test splits are rejected and episodes of another split are
  refused;
* accumulation is streaming Welford/Chan in float64 with valid masks
  (:class:`energywam.data.standardization.BundleFitter`);
* the reader is opened **read-only**; no dataset file is ever written;
* the DINO features come from the same :class:`FrozenDINOv3` used by training
  (same preprocess / patch-token selection); no Qwen is loaded.

Element counting (recorded in the bundle provenance): the action statistics
pool the valid elements of every valid training **window** — the same
episode-safe plan :class:`EnergyWAMEpisodeDataset` samples — and the future
statistics pool the valid (F, V, P) positions of those windows per DINO
channel.  A raw frame covered by several windows contributes several times,
matching the element distribution the training loss sees.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config import EnergyWAMConfig, geometry_from_cfg, resolve_future_offsets
from .episode_dataset import DatasetSpec, LeRobotEpisodeReader
from .future_offsets import plan_future_frames
from .standardization import (
    MANIFEST_SCHEMA_VERSION,
    BundleFitter,
    NormalizationBundle,
    StatsError,
    manifest_fingerprint,
)


# ---------------------------------------------------------------------------
# episode manifests
# ---------------------------------------------------------------------------
def load_episode_manifest(
    path: str | Path,
    *,
    reader: Optional[LeRobotEpisodeReader] = None,
    expect_split: Optional[str] = "train",
    other_manifest_paths: Sequence[str | Path] = (),
) -> Tuple[Dict[str, Any], List[int]]:
    """Load and validate an episode manifest.

    ``expect_split='train'`` (the statistics-fitting default) refuses val/test
    manifests; ``expect_split=None`` accepts any declared split and records it
    (used when a caller selects evaluation episodes explicitly).  Checks:
    schema version, a known split, a non-empty episode index list without
    duplicates, every episode exists in ``reader`` (when given), dataset
    identity matches the reader (when recorded), and no episode appears in any
    other supplied split manifest (cross-split leakage).
    """
    path = Path(path)
    if not path.exists():
        raise StatsError(f"episode manifest not found: {path}")
    import json

    try:
        man = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise StatsError(f"episode manifest {path} is not valid JSON: {exc}") from exc
    if not isinstance(man, dict):
        raise StatsError(f"episode manifest {path} must be a JSON object")
    if man.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise StatsError(
            f"episode manifest {path} has schema_version={man.get('schema_version')!r}, "
            f"expected {MANIFEST_SCHEMA_VERSION!r}"
        )
    split = str(man.get("split", "")).lower()
    if split not in ("train", "val", "test"):
        raise StatsError(
            f"episode manifest {path} declares unknown split={man.get('split')!r}; "
            "expected train / val / test"
        )
    if expect_split is not None and split != str(expect_split).lower():
        raise StatsError(
            f"statistics are fitted on the TRAIN split only; manifest {path} declares "
            f"split={man.get('split')!r}.  Evaluation/deployment never fits statistics "
            "(paper §6.3 / §4.8)."
        )
    raw_eps = man.get("episode_indices")
    if not isinstance(raw_eps, list) or not raw_eps:
        raise StatsError(f"episode manifest {path} has no episode_indices")
    episodes = [int(e) for e in raw_eps]
    if len(set(episodes)) != len(episodes):
        raise StatsError(f"episode manifest {path} lists duplicate episode indices")

    if reader is not None:
        known = {int(e["episode_index"]) for e in reader.episodes}
        missing = [e for e in episodes if e not in known]
        if missing:
            raise StatsError(
                f"episode manifest {path} references episodes not in the dataset: "
                f"{missing[:10]}{'...' if len(missing) > 10 else ''}"
            )
        want_id = man.get("dataset_identity")
        if want_id and str(want_id) != str(reader.dataset_identity):
            raise StatsError(
                f"episode manifest {path} was made for dataset {want_id!r} but the "
                f"reader sees {reader.dataset_identity!r}"
            )

    ours = set(episodes)
    for other in other_manifest_paths:
        other = Path(other)
        if not other.exists():
            raise StatsError(f"cross-split manifest not found: {other}")
        import json as _json

        om = _json.loads(other.read_text())
        o_eps = {int(e) for e in (om.get("episode_indices") or [])}
        overlap = sorted(ours & o_eps)
        if overlap:
            raise StatsError(
                f"episode(s) {overlap[:10]}{'...' if len(overlap) > 10 else ''} appear in "
                f"both {path} and {other}; train/val/test must not overlap"
            )
    man["manifest_sha256"] = manifest_fingerprint(man)
    man["manifest_path"] = str(path)
    return man, episodes


def load_train_manifest(
    path: str | Path,
    *,
    reader: Optional[LeRobotEpisodeReader] = None,
    other_manifest_paths: Sequence[str | Path] = (),
) -> Tuple[Dict[str, Any], List[int]]:
    """Training-split manifest loader (``split='train'`` only)."""
    return load_episode_manifest(
        path, reader=reader, expect_split="train",
        other_manifest_paths=other_manifest_paths,
    )


# ---------------------------------------------------------------------------
# streaming fit over training windows
# ---------------------------------------------------------------------------
def fit_bundle_from_reader(
    reader: LeRobotEpisodeReader,
    cfg: EnergyWAMConfig,
    *,
    episodes: Sequence[int],
    dino=None,
    include_future: bool = True,
    max_samples: Optional[int] = None,
    epsilon: Optional[float] = None,
    manifest: Optional[Dict[str, Any]] = None,
    dino_batch_size: int = 1,
) -> NormalizationBundle:
    """Fit the statistics bundle from the given TRAIN episodes (read-only).

    ``dino`` is the same :class:`FrozenDINOv3` training uses; required unless
    ``include_future=False`` (action_only / ``--no-future``).
    ``max_samples`` limits the number of windows and marks the bundle
    ``coverage.kind='subset'`` so it can never masquerade as a full-split fit.
    """
    if dino_batch_size < 1:
        raise ValueError("dino_batch_size must be positive")
    geom = geometry_from_cfg(cfg)
    Da, D = geom["action_dim"], geom["future_dim"]
    eps = float(epsilon if epsilon is not None else cfg.data.standardization_epsilon)
    H = int(cfg.data.action_horizon)
    offs = resolve_future_offsets(cfg.data)
    spec = DatasetSpec(
        action_horizon=H,
        future_offsets=tuple(offs),
        num_cameras=cfg.data.num_cameras,
        camera_index=int(cfg.data.future_camera_index),
        camera_keys=tuple(cfg.data.camera_order) if cfg.data.camera_order else None,
        require_full_horizon=bool(cfg.data.require_full_horizon),
    )
    cams = list(spec.camera_keys) if spec.camera_keys else list(reader.camera_keys)
    sel = cams[spec.camera_index : spec.camera_index + spec.num_cameras]

    if include_future and dino is None:
        raise StatsError("future statistics need the frozen DINO extractor")

    man = manifest or {}
    provenance = {
        "dataset_identity": reader.dataset_identity,
        "dataset_root": str(reader.root),
        "split": "train",
        "train_episodes": {
            "episode_indices": [int(e) for e in episodes],
            "manifest_sha256": man.get("manifest_sha256", ""),
            "manifest_path": man.get("manifest_path", ""),
            "num_episodes": len(list(episodes)),
        },
        "camera_order": [str(c) for c in (cfg.data.camera_order or [])],
        "cameras_used": [str(c) for c in sel],
        "camera_index": int(cfg.data.future_camera_index),
        "num_cameras": int(cfg.data.num_cameras),
        "action_horizon": H,
        "future_offsets": [int(o) for o in offs],
        "num_future_frames": int(cfg.data.num_future_frames),
        "dino_preprocess_version": cfg.data.dino_preprocess_version,
        "dino": {
            "model_path": str(getattr(dino, "model_path", "") or ""),
            "config_hash": dino_config_hash(dino),
            "hidden_dim": int(getattr(dino, "hidden_dim", D)),
            "num_patches": int(getattr(dino, "num_patches", geom["future_patches"])),
            "num_special_tokens": int(getattr(dino, "num_special_tokens", 5)),
            "image_size": int(getattr(dino, "image_size", 224)),
            "patch_size": int(getattr(dino, "patch_size", 16)),
        },
        "dims": {"action_dim": Da, "future_dim": D},
        "require_full_horizon": bool(cfg.data.require_full_horizon),
    }
    fitter = BundleFitter(
        action_dim=Da,
        future_dim=D if include_future else None,
        epsilon=eps,
        provenance=provenance,
    )

    n_windows = 0
    n_action_elems = 0
    n_future_elems = 0
    _t0 = time.time()
    for ep_idx, ep in enumerate(episodes):
        if max_samples is not None and n_windows >= max_samples:
            break
        ep = int(ep)
        L = int(reader.episode_length(ep))
        actions = np.asarray(reader.read_actions(ep), dtype=np.float64)  # [L, Da]
        if actions.shape[1] != Da:
            raise StatsError(
                f"episode {ep} has action dim {actions.shape[1]}, configured {Da}"
            )
        feat_cache: Dict[Tuple[str, int], np.ndarray] = {}
        if include_future and dino_batch_size > 1:
            # Same frames and masks as the streaming path, encoded in batches.
            # Bound prefetch to the windows remaining under --max-samples.
            needed = set()
            remaining = None if max_samples is None else max_samples - n_windows
            planned = 0
            for start in range(L):
                if remaining is not None and planned >= remaining:
                    break
                p = plan_future_frames(start, L, H, offs, require_full_horizon=spec.require_full_horizon)
                if p.valid:
                    needed.update(int(f) for f, valid in zip(p.abs_frames, p.future_valid) if valid and f < L)
                    planned += 1
            frames = sorted(needed)
            for cam in sel:
                for begin in range(0, len(frames), dino_batch_size):
                    indices = frames[begin:begin + dino_batch_size]
                    images = [reader.read_frame(ep, cam, f) for f in indices]
                    features = dino.encode_images(images, device="cpu").detach().float().numpy()
                    for f, features_f in zip(indices, features):
                        feat_cache[(cam, f)] = features_f
        for t in range(L):
            if max_samples is not None and n_windows >= max_samples:
                break
            plan = plan_future_frames(
                t, L, H, offs, require_full_horizon=spec.require_full_horizon
            )
            if not plan.valid:
                continue
            # --- action window: exactly the elements the training loss sees ---
            a = actions[t : t + H]
            if a.shape[0] < H:
                pad = np.zeros((H - a.shape[0], Da), dtype=np.float64)
                a = np.concatenate([a, pad], axis=0)
            a_mask = np.zeros((H, Da), dtype=bool)
            a_mask[: len(plan.action_valid), :] = np.asarray(plan.action_valid, bool)[:, None]
            fitter.update_action(a, mask=a_mask)
            n_action_elems += int(a_mask.sum())

            if include_future:
                for j, abs_f in enumerate(plan.abs_frames):
                    # invalid future positions are excluded entirely (mask-first)
                    if not plan.future_valid[j] or abs_f >= L:
                        continue
                    Zv = np.zeros((spec.num_cameras, geom["future_patches"], D), dtype=np.float64)
                    for v in range(spec.num_cameras):
                        key = (sel[v], int(abs_f))
                        feats = feat_cache.get(key)
                        if feats is None:
                            frame = reader.read_frame(ep, sel[v], int(abs_f))
                            feats = (
                                dino.encode_images([frame], device="cpu")
                                .detach()
                                .float()
                                .numpy()[0]
                            )  # [P, D]
                            feat_cache[key] = feats
                        if feats.shape != (geom["future_patches"], D):
                            raise StatsError(
                                f"DINO feature shape {feats.shape} != "
                                f"({geom['future_patches']}, {D})"
                            )
                        Zv[v] = feats
                    # one future time slot: [F=1, V, P, D], pooled over V, P
                    fitter.update_future(Zv[None, ...])
                    n_future_elems += int(spec.num_cameras * geom["future_patches"])
            n_windows += 1

        if (ep_idx + 1) % 5 == 0:
            el = time.time() - _t0
            rate = n_windows / el if el > 0 else 0.0
            eta = ((max_samples - n_windows) / rate) if (max_samples and rate > 0) else float("nan")
            print(
                f"[fit-stats] progress: {n_windows} windows, "
                f"{ep_idx + 1} episodes, {el:.0f}s, {rate:.2f} win/s, eta {eta:.0f}s",
                flush=True,
            )

    if n_windows == 0:
        raise StatsError("no valid training window found in the given episodes")
    print(f"[fit-stats] accumulated {n_windows} windows in {time.time() - _t0:.0f}s", flush=True)
    coverage = {
        "kind": "subset" if max_samples is not None else "full_train_split",
        "num_windows": int(n_windows),
        "max_samples": int(max_samples) if max_samples is not None else None,
        "action_valid_elements": int(n_action_elems),
        "future_valid_elements": int(n_future_elems),
        "note": (
            "subset limited by --max-samples; NOT a full training-split statistic"
            if max_samples is not None
            else "all valid windows of the manifest's training episodes"
        ),
    }
    return fitter.finalize(coverage=coverage)


def dino_config_hash(dino) -> str:
    """Short stable identity of the DINO feature configuration."""
    import hashlib

    fields = {
        "model_path": str(getattr(dino, "model_path", "") or ""),
        "hidden_dim": int(getattr(dino, "hidden_dim", 0)),
        "num_patches": int(getattr(dino, "num_patches", 0)),
        "num_special_tokens": int(getattr(dino, "num_special_tokens", 0)),
        "image_size": int(getattr(dino, "image_size", 0)),
        "patch_size": int(getattr(dino, "patch_size", 0)),
    }
    import json

    return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()[:16]


def build_dino_extractor(cfg: EnergyWAMConfig):
    """The exact :class:`FrozenDINOv3` training uses (never a Qwen model)."""
    from ..backbones.dino_v3 import FrozenDINOv3

    b = cfg.backbone
    return FrozenDINOv3(
        b.dino_path,
        image_size=b.dino_image_size,
        patch_size=b.dino_patch_size,
        hidden_dim=b.dino_hidden_dim,
        num_special_tokens=b.dino_num_special_tokens,
        num_patches=b.dino_num_patches,
        freeze=b.freeze_dino,
    )

"""Episode-safe LeRobot dataset producing EnergyWAM training samples.

Scope
-----
Reads a LeRobot v2-style dataset directory (``meta/`` + ``data/`` + ``videos/``)
and yields ``EnergyWAMSample`` records with

* the current observation (images per camera, robot state, language),
* the normalised action chunk ``A*`` of length ``H``,
* the **episode-safe** future RGB frames at the configured offsets,
* the action / future masks and the actual offsets used.

It deliberately does **not** read past the end of an episode: see
:func:`energywam.data.future_offsets.plan_future_frames`.

Camera / state / action dimensions are read from ``meta/info.json`` +
``meta/modality.json``, never hard-coded.

This dataset is written against the LeRobot schema shared by the RoboCasa365
conversion and the GR1 corpus on this server, so the same code can be used to
probe whichever corpus is present.  Which corpus was actually read is recorded
in ``EnergyWAMSample.dataset_identity`` and MUST be reported in experiment logs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .future_offsets import plan_future_frames
from .normalization import NormStats, normalize


@dataclass
class EnergyWAMSample:
    """One training sample (per-camera images are lists of arrays/PIL)."""

    episode_index: int
    frame_index: int
    language: str
    images: List[Any]                 # one per camera at the current frame
    state: np.ndarray                 # [Dq]
    action: np.ndarray                # [H, Da], normalised
    action_mask: np.ndarray           # [H, Da] float32 0/1
    future_images: List[List[Any]]    # [F][V] raw RGB of the future frames
    #: RELATIVE time offsets from the chunk start (e.g. [5, 11, 16]).
    #: These are what the encodings / ``resolve_future_offsets`` use.
    future_offsets: List[int]
    #: ABSOLUTE frame indices inside the episode (start + offset).  Recorded
    #: separately for debugging / cache keys; never used as a time encoding.
    future_frames_abs: List[int]
    future_valid: np.ndarray          # [F]
    dataset_identity: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)


class LeRobotEpisodeReader:
    """Minimal reader over a LeRobot v2 dataset directory."""

    def __init__(
        self,
        root: str | Path,
        *,
        camera_keys: Optional[Sequence[str]] = None,
        state_key: str = "observation.state",
        action_key: str = "action",
        language_key: str = "annotation.human.task_description",
    ):
        self.root = Path(root)
        if not (self.root / "meta" / "info.json").exists():
            raise FileNotFoundError(
                f"{self.root} is not a LeRobot dataset (missing meta/info.json). "
                "If you expected RoboCasa365 here, note that it is NOT present on "
                "this server; see docs/ARCHITECTURE.md 'data availability'."
            )
        self.info = json.loads((self.root / "meta" / "info.json").read_text())
        self.modality = json.loads((self.root / "meta" / "modality.json").read_text())
        self.stats = json.loads((self.root / "meta" / "stats.json").read_text())
        self.state_key = state_key
        self.action_key = action_key
        self.language_key = language_key
        self.camera_keys = list(camera_keys) if camera_keys else self._default_cameras()
        self.episodes = self._read_episodes()

    # ------------------------------------------------------------------
    def _default_cameras(self) -> List[str]:
        vids = self.modality.get("video", {})
        # keep a stable order: the order in modality.json is the recording order
        return [vids[k]["original_key"] for k in vids]

    def _read_episodes(self) -> List[Dict[str, Any]]:
        out = []
        p = self.root / "meta" / "episodes.jsonl"
        if p.exists():
            with p.open() as f:
                for line in f:
                    line = line.strip()
                    if line:
                        out.append(json.loads(line))
            return out
        # LeRobot v3 layout: meta/episodes/**/*.parquet — fall back to the index
        import glob

        files = sorted(glob.glob(str(self.root / "meta" / "episodes" / "**" / "*.parquet"), recursive=True))
        if not files:
            raise FileNotFoundError(f"no episode index found under {self.root}/meta")
        import pandas as pd

        for fp in files:
            df = pd.read_parquet(fp)
            for _, row in df.iterrows():
                out.append(row.to_dict())
        return out

    # ------------------------------------------------------------------
    @property
    def dataset_identity(self) -> str:
        robot = self.info.get("robot_type", "unknown")
        name = self.root.name
        return f"{name}/{robot}"

    def episode_length(self, episode_index: int) -> int:
        for ep in self.episodes:
            if int(ep["episode_index"]) == int(episode_index):
                return int(ep["length"])
        raise KeyError(f"episode {episode_index} not found")

    # ------------------------------------------------------------------
    def read_actions(self, episode_index: int) -> np.ndarray:
        """``[L, Da]`` raw action rows for one episode."""
        import pandas as pd

        path = self.root / self.info["data_path"].format(episode_chunk=0, episode_index=episode_index)
        if not Path(path).exists():
            # v3 layout uses chunk dirs
            alt = list(self.root.glob(f"data/**/episode_{int(episode_index):06d}.parquet"))
            if not alt:
                raise FileNotFoundError(f"parquet for episode {episode_index} not found under {self.root}/data")
            path = alt[0]
        df = pd.read_parquet(path)
        if self.action_key not in df.columns:
            raise KeyError(f"{self.action_key} not in parquet columns: {list(df.columns)[:20]}")
        return np.stack([np.asarray(v, dtype="float32") for v in df[self.action_key].tolist()])

    def read_states(self, episode_index: int) -> np.ndarray:
        import pandas as pd

        path = self.root / self.info["data_path"].format(episode_chunk=0, episode_index=episode_index)
        if not Path(path).exists():
            alt = list(self.root.glob(f"data/**/episode_{int(episode_index):06d}.parquet"))
            path = alt[0]
        df = pd.read_parquet(path)
        return np.stack([np.asarray(v, dtype="float32") for v in df[self.state_key].tolist()])

    def read_language(self, episode_index: int) -> str:
        for ep in self.episodes:
            if int(ep["episode_index"]) == int(episode_index):
                tasks = ep.get("tasks") or ep.get("task") or [""]
                return tasks[0] if isinstance(tasks, list) else str(tasks)
        return ""

    # ------------------------------------------------------------------
    def video_path(self, episode_index: int, camera_key: str) -> Path:
        # Official RoboCasa metadata uses short modality names (robot0_*) in
        # configs and full feature names (observation.images.robot0_*) on disk.
        # Resolve only the explicit mapping; never guess another camera.
        camera_meta = self.modality.get("video", {}).get(camera_key)
        if camera_meta is not None:
            camera_key = camera_meta["original_key"]
        tmpl = self.info["video_path"]
        # the template is like "videos/chunk-000/{video_key}/episode_000000.mp4";
        # substitute only the keys we know, then fill the rest.
        rel = tmpl.replace("{video_key}", camera_key)
        rel = rel.format(episode_chunk=0, episode_index=episode_index)
        p = self.root / rel
        if p.exists():
            return p
        alt = list(self.root.glob(f"videos/**/{camera_key}/episode_{int(episode_index):06d}.*"))
        if not alt:
            raise FileNotFoundError(f"video for episode {episode_index}, camera {camera_key} not found")
        return alt[0]

    def read_frame(self, episode_index: int, camera_key: str, frame_index: int):
        """Return one RGB frame as a numpy array [H, W, 3] uint8."""
        path = self.video_path(episode_index, camera_key)
        return _read_video_frame(path, frame_index)


_VIDEO_CACHE: Dict[str, np.ndarray] = {}


def _read_video_frame(path: Path, frame_index: int) -> np.ndarray:
    """Decode a single frame.  Tries torchvision.io first, then imageio/av.

    A per-path decoded-clip cache keeps repeated frame reads cheap (an episode
    sample needs ~H+F frames from the same clip).
    """
    key = str(path)
    frames = _VIDEO_CACHE.get(key)
    if frames is None:
        frames = _decode_video(path)
        # keep the cache small: only a handful of clips are touched per run
        if len(_VIDEO_CACHE) > 8:
            _VIDEO_CACHE.clear()
        _VIDEO_CACHE[key] = frames
    if frame_index < 0 or frame_index >= frames.shape[0]:
        raise IndexError(f"frame {frame_index} out of range for {path} ({frames.shape[0]} frames)")
    return frames[frame_index]


def _decode_video(path: Path) -> np.ndarray:
    """Decode deterministically with a bounded decoder thread count.

    Automatic FFmpeg thread counts scale with the host CPU count and can
    exhaust threads on a shared server. Keep this per-reader and explicit.
    """
    try:
        import av
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            stream.codec_context.thread_count = 1
            frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(stream)]
        if not frames:
            raise ValueError("video has no decodable frames")
        return np.stack(frames).astype("uint8", copy=False)
    except Exception as exc:
        raise RuntimeError(f"cannot decode video {path}: {exc}") from exc


# ---------------------------------------------------------------------------
# Sample construction
# ---------------------------------------------------------------------------
@dataclass
class DatasetSpec:
    action_horizon: int = 16
    future_offsets: Tuple[int, ...] = (5, 11, 16)
    num_cameras: int = 1
    #: index of the primary camera in ``camera_keys``; the first
    #: ``num_cameras`` keys starting there are used, in this order.
    camera_index: int = 0
    #: explicit camera-key order (overrides the reader's default order)
    camera_keys: Optional[Tuple[str, ...]] = None
    action_mode: str = "min_max"
    state_mode: str = "none"
    require_full_horizon: bool = True
    image_size: int = 224


def build_sample(
    reader: LeRobotEpisodeReader,
    episode_index: int,
    frame_index: int,
    spec: DatasetSpec,
    action_stats: Optional[NormStats] = None,
    state_stats: Optional[NormStats] = None,
) -> Optional[EnergyWAMSample]:
    """Build one EnergyWAM sample, or ``None`` if the plan is invalid.

    The action chunk covers ``[frame_index, frame_index + H - 1]`` and the
    future offsets are relative to ``frame_index``.  No frame outside the
    episode is ever read.
    """
    L = reader.episode_length(episode_index)
    plan = plan_future_frames(
        frame_index,
        L,
        spec.action_horizon,
        list(spec.future_offsets),
        require_full_horizon=spec.require_full_horizon,
    )
    if not plan.valid:
        return None

    actions = reader.read_actions(episode_index)  # [L, Da]
    states = reader.read_states(episode_index)  # [L, Dq]
    Da = actions.shape[1]

    a = actions[frame_index : frame_index + spec.action_horizon]
    if a.shape[0] < spec.action_horizon:
        pad = np.zeros((spec.action_horizon - a.shape[0], Da), dtype="float32")
        a = np.concatenate([a, pad], axis=0)
    action_mask = np.zeros((spec.action_horizon, Da), dtype="float32")
    action_mask[: len(plan.action_valid), :] = np.asarray(plan.action_valid, dtype="float32")[:, None]

    if action_stats is not None:
        a = normalize(a, action_stats)

    q = states[min(frame_index, states.shape[0] - 1)]
    if state_stats is not None:
        q = normalize(q, state_stats)

    cams = list(spec.camera_keys) if spec.camera_keys else list(reader.camera_keys)
    if spec.camera_index < 0 or spec.camera_index + spec.num_cameras > len(cams):
        raise ValueError(
            f"camera_index={spec.camera_index} + num_cameras={spec.num_cameras} "
            f"exceeds the {len(cams)} available cameras {cams}"
        )
    sel = cams[spec.camera_index : spec.camera_index + spec.num_cameras]
    images = [reader.read_frame(episode_index, sel[v], frame_index) for v in range(spec.num_cameras)]
    future_images = []
    future_valid = []
    for j, abs_f in enumerate(plan.abs_frames):
        if plan.future_valid[j] and abs_f < L:
            # same selected cameras as the current frame (sel[v], NOT cams[v]):
            # current and future supervision must come from one and the same view
            future_images.append([reader.read_frame(episode_index, sel[v], abs_f) for v in range(spec.num_cameras)])
            future_valid.append(1.0)
        else:
            future_images.append([np.zeros_like(images[v]) for v in range(spec.num_cameras)])
            future_valid.append(0.0)

    return EnergyWAMSample(
        episode_index=int(episode_index),
        frame_index=int(frame_index),
        language=reader.read_language(episode_index),
        images=images,
        state=np.asarray(q, dtype="float32"),
        action=np.asarray(a, dtype="float32"),
        action_mask=action_mask,
        future_images=future_images,
        future_offsets=list(spec.future_offsets),
        future_frames_abs=list(plan.abs_frames),
        future_valid=np.asarray(future_valid, dtype="float32"),
        dataset_identity=reader.dataset_identity,
        meta={
            "action_dim": int(Da),
            "state_dim": int(states.shape[1]),
            "cameras": list(sel),
            "episode_length": int(L),
            "plan": plan.as_dict(),
        },
    )


def probe_one_sample(root: str | Path, spec: Optional[DatasetSpec] = None) -> Dict[str, Any]:
    """Read one real sample and return a de-identified shape/索引 report.

    This is what the engineering requirements ask for in stage A: print shapes,
    episode/frame indices, future offsets, masks and the normalisation contract.
    """
    reader = LeRobotEpisodeReader(root)
    spec = spec or DatasetSpec()
    ep = int(reader.episodes[0]["episode_index"])
    L = reader.episode_length(ep)
    offsets = list(spec.future_offsets)
    start = 0
    for t in range(L):
        if plan_future_frames(t, L, spec.action_horizon, offsets, require_full_horizon=True).valid:
            start = t
            break
    sample = build_sample(reader, ep, start, spec)
    if sample is None:
        raise RuntimeError(f"could not build a valid sample from {root}")
    return {
        "dataset_identity": reader.dataset_identity,
        "episode_index": sample.episode_index,
        "frame_index": sample.frame_index,
        "episode_length": L,
        "language_preview": sample.language[:80],
        "num_cameras": len(sample.images),
        "image_shape": list(np.asarray(sample.images[0]).shape),
        "state_shape": list(sample.state.shape),
        "action_shape": list(sample.action.shape),
        "action_mask_shape": list(sample.action_mask.shape),
        "action_mask_valid_counts": sample.action_mask.sum(axis=1).astype(int).tolist(),
        "future_offsets_rel": sample.future_offsets,
        "future_frames_abs": sample.future_frames_abs,
        "frame_index": sample.frame_index,
        "future_valid": sample.future_valid.tolist(),
        "future_image_shape": list(np.asarray(sample.future_images[0][0]).shape),
        "action_min_max": [float(sample.action.min()), float(sample.action.max())],
        "meta": {k: v for k, v in sample.meta.items() if k != "plan"},
    }

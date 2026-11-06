"""A tiny on-disk LeRobot dataset with real videos, for CLI/pipeline tests.

Explicitly synthetic: deterministic actions/states/frames so a test can compute
the expected statistics by hand.  Layout matches ``LeRobotEpisodeReader``:

    meta/info.json, meta/modality.json, meta/stats.json, meta/episodes.jsonl
    data/chunk-000/episode_XXXXXX.parquet
    videos/chunk-000/<camera>/episode_XXXXXX.mp4
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Sequence

import numpy as np


DEFAULT_CAMERA = "robot0_agentview_left"


def synthetic_action(episode: int, t: int, dim: int) -> np.ndarray:
    """Deterministic raw action row (the 'physical units' of this corpus)."""
    return np.asarray(
        [10.0 * episode + 0.5 * t + 0.25 * d for d in range(dim)], dtype="float32"
    )


def synthetic_state(episode: int, t: int, dim: int) -> np.ndarray:
    return np.asarray([0.01 * t + 0.1 * d for d in range(dim)], dtype="float32")


def synthetic_frame(episode: int, t: int, size: int = 32, cam_seed: int = 0) -> np.ndarray:
    """Deterministic per-(camera, episode, t) frame — cameras differ numerically."""
    rng = np.random.RandomState(1000 * episode + t + 7919 * cam_seed)
    return rng.randint(0, 256, size=(size, size, 3), dtype=np.uint8)


def write_synthetic_lerobot(
    root: str | Path,
    *,
    num_episodes: int = 3,
    episode_length: int = 24,
    action_dim: int = 12,
    state_dim: int = 16,
    camera: str = DEFAULT_CAMERA,
    cameras: Sequence[str] | None = None,
    dataset_name: str = "synthetic_mix",
) -> Path:
    """Write the dataset under ``root/dataset_name`` and return that path.

    ``cameras`` writes one video track per camera (frames differ per camera so a
    test can detect which camera was actually read).
    """
    import pandas as pd

    cam_list = list(cameras) if cameras is not None else [camera]
    base = Path(root) / dataset_name
    (base / "meta").mkdir(parents=True, exist_ok=True)
    (base / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    for cam in cam_list:
        (base / "videos" / "chunk-000" / cam).mkdir(parents=True, exist_ok=True)

    (base / "meta" / "info.json").write_text(json.dumps({
        "robot_type": "SyntheticRobot",
        "data_path": "data/chunk-{episode_chunk}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk}/{video_key}/episode_{episode_index:06d}.mp4",
    }))
    (base / "meta" / "modality.json").write_text(json.dumps({
        "video": {cam: {"original_key": cam} for cam in cam_list},
    }))
    (base / "meta" / "stats.json").write_text(json.dumps({
        "action": {"min": [0.0] * action_dim, "max": [1.0] * action_dim},
        "note": "synthetic corpus; this file is NOT used for the paper profile",
    }))
    with (base / "meta" / "episodes.jsonl").open("w") as f:
        for ep in range(num_episodes):
            f.write(json.dumps({
                "episode_index": ep,
                "length": episode_length,
                "tasks": [f"synthetic task {ep}"],
            }) + "\n")

    for ep in range(num_episodes):
        rows = {
            "action": [synthetic_action(ep, t, action_dim).tolist() for t in range(episode_length)],
            "observation.state": [synthetic_state(ep, t, state_dim).tolist() for t in range(episode_length)],
        }
        pd.DataFrame(rows).to_parquet(base / "data" / "chunk-000" / f"episode_{ep:06d}.parquet")

        for ci, cam in enumerate(cam_list):
            frames = np.stack([synthetic_frame(ep, t, cam_seed=ci) for t in range(episode_length)])
            _write_video(base / "videos" / "chunk-000" / cam / f"episode_{ep:06d}.mp4", frames)
    return base


def _write_video(path: Path, frames: np.ndarray, fps: int = 10) -> None:
    try:
        import imageio.v3 as iio

        iio.imwrite(str(path), frames, plugin="pyav", codec="libx264", fps=fps)
        return
    except Exception:
        pass
    from torchvision.io import write_video
    import torch

    write_video(str(path), torch.from_numpy(frames), fps=fps)


def write_manifest(path: str | Path, *, split: str, episodes: Sequence[int],
                   dataset_identity: str, dataset_root: str = "") -> Path:
    path = Path(path)
    path.write_text(json.dumps({
        "schema_version": "energywam_episode_manifest_v1",
        "split": split,
        "dataset_identity": dataset_identity,
        "dataset_root": dataset_root,
        "episode_indices": [int(e) for e in episodes],
        "notes": "synthetic corpus manifest written by tests/synthetic_lerobot.py",
    }, indent=2))
    return path

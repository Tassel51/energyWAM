"""Pre-computed DINO feature cache — interface contract.

The engineering requirements allow the cache to be an *extensible contract*
without a full implementation, but they require that a real cache key includes
``episode/frame/camera, DINO checkpoint, preprocessing and normalisation
version`` so features can never be silently mismatched.

This module defines that contract and a reference in-memory implementation.
A disk-backed implementation can be added later by subclassing
:class:`FeatureCache` and honouring :meth:`FeatureCache.key`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class CacheKey:
    """Exactly the components the requirements demand in a cache key."""

    dataset_id: str
    episode_index: int
    frame_index: int
    camera: str
    dino_checkpoint: str
    preprocess_version: str
    normalization_version: str

    def to_str(self) -> str:
        payload = {
            "dataset_id": self.dataset_id,
            "episode_index": int(self.episode_index),
            "frame_index": int(self.frame_index),
            "camera": str(self.camera),
            "dino_checkpoint": self.dino_checkpoint,
            "preprocess_version": self.preprocess_version,
            "normalization_version": self.normalization_version,
        }
        return json.dumps(payload, sort_keys=True)

    def digest(self) -> str:
        return hashlib.sha256(self.to_str().encode("utf-8")).hexdigest()


class FeatureCache:
    """Abstract cache over ``(CacheKey -> np.ndarray)``.

    ``get`` returns ``None`` on a miss.  ``put`` stores a detached numpy array.
    Implementations must treat the key as opaque — never partial-match.
    """

    def get(self, key: CacheKey) -> Optional[np.ndarray]:
        raise NotImplementedError

    def put(self, key: CacheKey, value: np.ndarray) -> None:
        raise NotImplementedError

    def __contains__(self, key: CacheKey) -> bool:
        return self.get(key) is not None


class InMemoryFeatureCache(FeatureCache):
    """Reference in-memory implementation (useful for tests / single-process runs)."""

    def __init__(self) -> None:
        self._store: Dict[str, np.ndarray] = {}
        self.hits = 0
        self.misses = 0

    def get(self, key: CacheKey) -> Optional[np.ndarray]:
        k = key.digest()
        if k in self._store:
            self.hits += 1
            return self._store[k]
        self.misses += 1
        return None

    def put(self, key: CacheKey, value: np.ndarray) -> None:
        self._store[key.digest()] = np.asarray(value)

    def __len__(self) -> int:
        return len(self._store)


def make_key(
    *,
    dataset_id: str,
    episode_index: int,
    frame_index: int,
    camera: str,
    dino_checkpoint: str,
    preprocess_version: str,
    normalization_version: str,
) -> CacheKey:
    return CacheKey(
        dataset_id=str(dataset_id),
        episode_index=int(episode_index),
        frame_index=int(frame_index),
        camera=str(camera),
        dino_checkpoint=str(dino_checkpoint),
        preprocess_version=str(preprocess_version),
        normalization_version=str(normalization_version),
    )

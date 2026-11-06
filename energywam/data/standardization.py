"""Paper §4.8 standardized coordinate — the canonical stats-bundle / normalizer API.

This module is the **single** place that knows the standardized coordinate

    tilde_x = (x - mean) / (std + epsilon)          (paper Eq. 37 / 39)
    x       = tilde_x * (std + epsilon) + mean      (exact inverse)

with

* ``mean`` / ``std`` fitted on the **training split only** (never val/test,
  never at evaluation or deployment time),
* ``std`` the **population** standard deviation (divide by the count, not
  count-1),
* ``epsilon`` explicit and recorded in every bundle (default ``1e-6``),
* all statistics and accumulators in **float64**; model candidates / losses
  stay float32 (they simply consume the transformed values).

Element counting rules (also recorded in the bundle provenance):

* action statistics are per action dimension ``Da`` over the **valid elements
  of every training window** the training dataset would sample (same
  ``require_full_horizon`` plan as :mod:`energywam.data.energy_dataset`), with
  the sample's ``action_mask`` applied first — masked / non-finite elements do
  not participate.  A raw frame covered by ``k`` valid windows contributes ``k``
  times; this is exactly the element distribution the training loss sees.
* future statistics are per DINO channel (384 for DINOv3-ViT-S) pooled over the
  **valid future time / view / patch** positions (``F, V, P`` pooling) of every
  training window — not per patch, not per trajectory.

Zero-variance dimensions are **not** silently treated as raw: they get
``std = 0`` and the transform becomes ``(x - mean) / epsilon``, which is finite
and exactly invertible.  A column with zero valid count is an error.

Nothing here reads or writes the legacy StarVLA ``meta/stats.json`` contract;
:mod:`energywam.data.normalization` keeps that for the ``legacy`` profile.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np

try:  # torch is a hard dep of the project, but keep the module importable standalone
    import torch
except Exception:  # pragma: no cover
    torch = None


# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
SCHEMA_VERSION = "energywam_norm_bundle_v1"
MANIFEST_SCHEMA_VERSION = "energywam_episode_manifest_v1"

PROFILE_LEGACY = "legacy"
PROFILE_PAPER = "paper_standardized_v1"
PROFILES = (PROFILE_LEGACY, PROFILE_PAPER)

#: paper §4.8 Eq. (37)/(39): (x - mean) / (std + epsilon)
COORDINATE = "(x - mean) / (std + epsilon)"
INVERSE_COORDINATE = "x = normalized * (std + epsilon) + mean"
STD_KIND = "population"
DEFAULT_EPSILON = 1e-6

#: markers accepted for hand-built ``future_target`` inputs
FUTURE_SPACE_RAW = "raw_dino"
FUTURE_SPACE_STANDARDIZED = "standardized"
FUTURE_SPACES = (FUTURE_SPACE_RAW, FUTURE_SPACE_STANDARDIZED)


class StatsError(ValueError):
    """Raised when statistics are missing, inconsistent or mis-provenanced.

    The policy is fail-loud: a run never falls back to raw coordinates, random
    statistics or a val/test estimate.
    """


def _require_torch():
    if torch is None:  # pragma: no cover
        raise ImportError("torch is required for the tensor transforms")
    return torch


def _is_torch(x) -> bool:
    return torch is not None and isinstance(x, torch.Tensor)


# ---------------------------------------------------------------------------
# streaming moments (Welford / Chan parallel variance, float64)
# ---------------------------------------------------------------------------
class StreamingMoments:
    """Per-channel mean / population variance accumulator.

    ``update(x, mask)`` accepts an array of shape ``[..., D]`` (any leading
    shape) plus an optional mask broadcastable to it.  Masked-out or
    non-finite elements never enter the statistics, and each channel keeps its
    own valid count, so partial masks are handled exactly.
    """

    def __init__(self, dim: int):
        self.dim = int(dim)
        self.count = np.zeros(self.dim, dtype=np.float64)
        self.mean = np.zeros(self.dim, dtype=np.float64)
        self.m2 = np.zeros(self.dim, dtype=np.float64)  # sum of squared deviations

    # ------------------------------------------------------------------
    def update(self, x, mask=None) -> None:
        a = np.asarray(x, dtype=np.float64)
        if a.shape[-1] != self.dim:
            raise StatsError(f"last dim {a.shape[-1]} != expected {self.dim}")
        if a.size == 0:
            return
        flat = a.reshape(-1, self.dim)
        valid = np.isfinite(flat)
        if mask is not None:
            valid &= self._broadcast_mask(mask, a.shape).reshape(-1, self.dim)
        self._merge_block(flat, valid)

    @staticmethod
    def _broadcast_mask(mask, shape) -> np.ndarray:
        """Bool mask broadcast to ``shape`` (reduced masks are left-aligned)."""
        m = np.asarray(mask)
        if m.dtype != bool:
            m = m.astype(bool)
        if m.shape == shape:
            return m
        candidates = [m]
        if m.ndim < len(shape):  # left-align: trailing channel dim is implicit
            left = m.reshape(m.shape + (1,) * (len(shape) - m.ndim))
            candidates.append(left)
        candidates.append(np.expand_dims(m, 0))  # batch-shared
        for cand in candidates:
            try:
                return np.broadcast_to(cand, shape)
            except ValueError:
                continue
        raise StatsError(f"mask shape {np.shape(mask)} not broadcastable to {shape}")

    def _merge_block(self, flat: np.ndarray, valid: np.ndarray) -> None:
        """Chan's parallel merge of one block of observations."""
        # Vectorized Chan merge; counts remain independent per channel.
        # Removing the Python loop over 384 DINO channels makes offline fitting
        # practical without changing masks, population variance or precision.
        n_b = valid.sum(axis=0, dtype=np.float64)
        nz = n_b > 0
        divisor = np.maximum(n_b, 1.0)
        mean_b = np.where(valid, flat, 0.0).sum(axis=0, dtype=np.float64) / divisor
        centered = np.where(valid, flat - mean_b, 0.0)
        m2_b = (centered * centered).sum(axis=0, dtype=np.float64)
        n_a = self.count[nz]
        n = n_a + n_b[nz]
        delta = mean_b[nz] - self.mean[nz]
        self.mean[nz] += delta * n_b[nz] / n
        self.m2[nz] += m2_b[nz] + delta * delta * n_a * n_b[nz] / n
        self.count[nz] = n

    # ------------------------------------------------------------------
    @property
    def variance(self) -> np.ndarray:
        """Population variance (``M2 / n``); zero-count channels -> 0."""
        out = np.zeros(self.dim, dtype=np.float64)
        nz = self.count > 0
        out[nz] = self.m2[nz] / self.count[nz]
        return out

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self.variance)

    def finalize(self, epsilon: float = DEFAULT_EPSILON) -> "ZScoreStats":
        if not bool((self.count > 0).all()):
            missing = [int(i) for i in np.where(self.count <= 0)[0][:8]]
            raise StatsError(
                f"channels/dims {missing} (and possibly more) received no valid "
                "element; refusing to fit statistics with empty columns"
            )
        return ZScoreStats(
            mean=self.mean.copy(),
            std=self.std,
            count=self.count.copy(),
            epsilon=float(epsilon),
        )


# ---------------------------------------------------------------------------
# per-variable z-score statistics
# ---------------------------------------------------------------------------
@dataclass
class ZScoreStats:
    """``mean``/``std``/``count`` for one variable (action or future channel)."""

    mean: np.ndarray
    std: np.ndarray
    count: np.ndarray
    epsilon: float = DEFAULT_EPSILON

    def __post_init__(self):
        self.mean = np.asarray(self.mean, dtype=np.float64).reshape(-1)
        self.std = np.asarray(self.std, dtype=np.float64).reshape(-1)
        self.count = np.asarray(self.count, dtype=np.float64).reshape(-1)
        d = self.mean.shape[0]
        if self.std.shape[0] != d or self.count.shape[0] != d:
            raise StatsError(
                f"inconsistent stats lengths: mean {self.mean.shape[0]} std "
                f"{self.std.shape[0]} count {self.count.shape[0]}"
            )
        if not np.isfinite(self.mean).all() or not np.isfinite(self.std).all():
            raise StatsError("non-finite mean/std in statistics")
        if (self.std < 0).any():
            raise StatsError("negative std in statistics")
        if not np.isfinite(self.count).all() or (self.count <= 0).any():
            raise StatsError("non-finite or non-positive count in statistics")
        self.epsilon = float(self.epsilon)
        if not np.isfinite(self.epsilon) or not (self.epsilon > 0):
            raise StatsError(f"epsilon must be finite and > 0, got {self.epsilon}")

    # ------------------------------------------------------------------
    @property
    def dim(self) -> int:
        return int(self.mean.shape[0])

    @property
    def denom(self) -> np.ndarray:
        """``std + epsilon`` — the exact divisor used by the forward transform."""
        return self.std + self.epsilon

    # ------------------------------------------------------------------
    def _check_last_dim(self, v):
        if v.shape[-1] != self.dim:
            raise StatsError(f"last dim {v.shape[-1]} != stats dim {self.dim}")

    def normalize(self, x):
        """``(x - mean) / (std + epsilon)``.  Preserves container/dtype/shape.

        The arithmetic runs in float64 (same precision the statistics were
        fitted in) and is cast back to the input dtype at the end, so the torch
        and numpy paths — and the :class:`NormStats` ``zscore`` adapter used for
        serving — all see exactly the same coordinate.
        """
        self._check_last_dim(x)
        if _is_torch(x):
            t = _require_torch()
            mean = t.as_tensor(self.mean, device=x.device, dtype=t.float64)
            denom = t.as_tensor(self.denom, device=x.device, dtype=t.float64)
            return ((x.double() - mean) / denom).to(x.dtype)
        a = np.asarray(x)
        return ((a.astype(np.float64) - self.mean) / self.denom).astype(a.dtype, copy=False)

    def denormalize(self, y):
        """Exact inverse: ``y * (std + epsilon) + mean`` (same float64 core)."""
        self._check_last_dim(y)
        if _is_torch(y):
            t = _require_torch()
            mean = t.as_tensor(self.mean, device=y.device, dtype=t.float64)
            denom = t.as_tensor(self.denom, device=y.device, dtype=t.float64)
            return (y.double() * denom + mean).to(y.dtype)
        b = np.asarray(y)
        return (b.astype(np.float64) * self.denom + self.mean).astype(b.dtype, copy=False)

    # ------------------------------------------------------------------
    def to_norm_stats(self):
        """Adapter for the legacy serve / predictor un-normalisation path."""
        from .normalization import NormStats

        return NormStats(
            mode="zscore",
            mean=self.mean.astype(np.float32),
            std=self.std.astype(np.float32),
            binary_threshold=0.0,
            version=f"{PROFILE_PAPER}:{SCHEMA_VERSION}",
            epsilon=self.epsilon,
        )

    def to_payload(self) -> Dict[str, Any]:
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "count": self.count.tolist(),
            "epsilon": self.epsilon,
            "std_kind": STD_KIND,
            "coordinate": COORDINATE,
        }

    @classmethod
    def from_payload(cls, p: Dict[str, Any]) -> "ZScoreStats":
        try:
            return cls(
                mean=np.asarray(p["mean"], dtype=np.float64),
                std=np.asarray(p["std"], dtype=np.float64),
                count=np.asarray(p["count"], dtype=np.float64),
                epsilon=float(p.get("epsilon", DEFAULT_EPSILON)),
            )
        except KeyError as exc:
            raise StatsError(f"stats payload missing key {exc}") from exc


# ---------------------------------------------------------------------------
# the versioned bundle
# ---------------------------------------------------------------------------
@dataclass
class NormalizationBundle:
    """Self-describing statistics bundle (paper §4.8 implementation contract)."""

    profile: str
    epsilon: float
    action: ZScoreStats
    future: Optional[ZScoreStats]
    provenance: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION
    coordinate: str = COORDINATE

    def __post_init__(self):
        if self.profile != PROFILE_PAPER:
            raise StatsError(f"bundle profile must be {PROFILE_PAPER!r}, got {self.profile!r}")
        if self.schema_version != SCHEMA_VERSION:
            raise StatsError(
                f"unknown bundle schema_version {self.schema_version!r} (expected {SCHEMA_VERSION!r})"
            )
        if self.coordinate != COORDINATE:
            raise StatsError(f"unknown coordinate spec {self.coordinate!r}")
        self.epsilon = float(self.epsilon)
        if self.action is None:
            raise StatsError("bundle must carry action statistics")
        if abs(self.action.epsilon - self.epsilon) > 0:
            raise StatsError("action stats epsilon differs from bundle epsilon")
        if self.future is not None and abs(self.future.epsilon - self.epsilon) > 0:
            raise StatsError("future stats epsilon differs from bundle epsilon")

    # ------------------------------------------------------------------
    @property
    def fingerprint(self) -> str:
        return bundle_fingerprint(self._body())

    @property
    def dims(self) -> Dict[str, int]:
        d = {"action_dim": self.action.dim}
        if self.future is not None:
            d["future_dim"] = self.future.dim
        return d

    def _body(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "profile": self.profile,
            "coordinate": self.coordinate,
            "inverse_coordinate": INVERSE_COORDINATE,
            "std_kind": STD_KIND,
            "epsilon": self.epsilon,
            "action": self.action.to_payload(),
            "future": self.future.to_payload() if self.future is not None else None,
            "provenance": self.provenance,
        }

    def to_payload(self) -> Dict[str, Any]:
        out = self._body()
        out["fingerprint"] = self.fingerprint
        return out

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "NormalizationBundle":
        if not isinstance(payload, dict):
            raise StatsError(f"bundle payload must be a dict, got {type(payload)}")
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise StatsError(
                f"unsupported bundle schema_version {payload.get('schema_version')!r}; "
                f"this build reads {SCHEMA_VERSION!r}"
            )
        if "fingerprint" not in payload or not payload["fingerprint"]:
            # production statistics files and checkpoint-embedded payloads must
            # carry a content fingerprint; hand-built objects (the test API)
            # never go through here.
            raise StatsError(
                "statistics payload has no 'fingerprint'; refusing to trust an "
                "unversioned/unsigned statistics payload"
            )
        fut = payload.get("future")
        bundle = cls(
            profile=str(payload.get("profile")),
            epsilon=float(payload.get("epsilon", DEFAULT_EPSILON)),
            action=ZScoreStats.from_payload(payload["action"]),
            future=ZScoreStats.from_payload(fut) if fut is not None else None,
            provenance=dict(payload.get("provenance") or {}),
            schema_version=str(payload.get("schema_version")),
            coordinate=str(payload.get("coordinate")),
        )
        fp = payload.get("fingerprint")
        if fp != bundle.fingerprint:
            raise StatsError(
                f"bundle fingerprint mismatch: payload says {fp}, recomputed "
                f"{bundle.fingerprint}.  The file/payload was modified or corrupted."
            )
        return bundle

    # ------------------------------------------------------------------
    def require_future(self) -> ZScoreStats:
        if self.future is None:
            raise StatsError(
                "this bundle carries no future statistics (fitted with "
                "--no-future / action_only); a future standardization needs them"
            )
        return self.future

    def validate_geometry(
        self,
        *,
        action_dim: Optional[int] = None,
        future_dim: Optional[int] = None,
        camera_order: Optional[Sequence[str]] = None,
        cameras_used: Optional[Sequence[str]] = None,
        camera_index: Optional[int] = None,
        num_cameras: Optional[int] = None,
        action_horizon: Optional[int] = None,
        future_offsets: Optional[Sequence[int]] = None,
        dino_preprocess_version: Optional[str] = None,
        dino_model_path: Optional[str] = None,
        dino_config: Optional[Dict[str, Any]] = None,
        epsilon: Optional[float] = None,
        require_future: bool = True,
        allow_synthetic: bool = False,
    ) -> None:
        """Reject any provenance/dimension mismatch against the run configuration.

        ``allow_synthetic=False`` (the production default) refuses bundles marked
        ``provenance.synthetic`` — synthetic statistics must never silently reach
        a real (qwen3vl) run.  The future-side DINO identity checks are skipped
        ONLY when the bundle carries no future statistics and the run needs none
        (action_only): the current-frame DINO is loaded normally either way.
        """
        prov = self.provenance
        if prov.get("synthetic") and not allow_synthetic:
            raise StatsError(
                "this statistics bundle is marked synthetic (provenance.synthetic); "
                "refusing to use synthetic statistics for a real run.  Use "
                "scripts/fit_normalization_stats.py on the real training split, or "
                "set data.allow_synthetic_stats=true only for explicit verification."
            )
        if epsilon is not None and abs(float(epsilon) - self.epsilon) > 0:
            raise StatsError(
                f"epsilon mismatch: config {epsilon} vs bundle {self.epsilon}"
            )
        if action_dim is not None and self.action.dim != int(action_dim):
            raise StatsError(
                f"action_dim mismatch: bundle {self.action.dim} vs configured {action_dim}"
            )
        if future_dim is not None:
            if require_future:
                self.require_future()
            if self.future is not None and self.future.dim != int(future_dim):
                raise StatsError(
                    f"future_dim mismatch: bundle {self.future.dim} vs configured {future_dim}"
                )
        if require_future and self.future is None:
            raise StatsError(
                "bundle has no future statistics but this run needs future "
                "standardization (stage/initialization requires Z)"
            )
        if camera_order is not None:
            got = list(prov.get("camera_order") or [])
            want = [str(c) for c in camera_order]
            if got != want:
                raise StatsError(
                    f"camera order mismatch: bundle {got} vs configured {want}"
                )
        if camera_index is not None:
            got = prov.get("camera_index")
            if got is None:
                raise StatsError(
                    "bundle provenance has no camera_index; refusing to guess which "
                    "camera the statistics were fitted on"
                )
            if int(got) != int(camera_index):
                raise StatsError(
                    f"camera selection mismatch: bundle camera_index={got} vs "
                    f"configured {camera_index}"
                )
        if num_cameras is not None:
            got = prov.get("num_cameras")
            if got is None:
                raise StatsError(
                    "bundle provenance has no num_cameras; refusing to guess the "
                    "camera count the statistics were fitted with"
                )
            if int(got) != int(num_cameras):
                raise StatsError(
                    f"camera selection mismatch: bundle num_cameras={got} vs "
                    f"configured {num_cameras}"
                )
        if cameras_used is not None:
            got = list(prov.get("cameras_used") or [])
            if not got:
                raise StatsError(
                    "bundle provenance has no cameras_used; refusing to guess which "
                    "camera(s) the statistics were fitted on"
                )
            want = [str(c) for c in cameras_used]
            if got != want:
                raise StatsError(
                    f"camera selection mismatch: bundle cameras_used={got} vs "
                    f"run selection {want}"
                )
        if action_horizon is not None:
            got = prov.get("action_horizon")
            if int(got) != int(action_horizon):
                raise StatsError(
                    f"action_horizon mismatch: bundle {got} vs configured {action_horizon}"
                )
        if future_offsets is not None:
            got = [int(x) for x in (prov.get("future_offsets") or [])]
            want = [int(x) for x in future_offsets]
            if got != want:
                raise StatsError(
                    f"future offsets mismatch: bundle {got} vs configured {want}"
                )
        if self.future is not None or require_future:
            # future-side identity: preprocess version + DINO checkpoint AND its
            # geometry config (a changed config at the same path must be refused).
            if dino_preprocess_version is not None:
                got = str(prov.get("dino_preprocess_version") or "")
                if got != str(dino_preprocess_version):
                    raise StatsError(
                        f"DINO preprocess version mismatch: bundle {got!r} vs configured "
                        f"{dino_preprocess_version!r}"
                    )
            dino_prov = prov.get("dino") or {}
            if dino_model_path is not None:
                got = dino_prov.get("model_path")
                if got is None:
                    raise StatsError("bundle provenance has no dino.model_path")
                if str(got) != str(dino_model_path):
                    raise StatsError(
                        f"DINO checkpoint mismatch: bundle {got!r} vs configured {dino_model_path!r}"
                    )
            for key, want in (dino_config or {}).items():
                if key not in dino_prov:
                    raise StatsError(
                        f"bundle provenance has no dino.{key}; refusing to accept "
                        "incomplete DINO identity"
                    )
                got = dino_prov[key]
                if isinstance(want, (int, float)) and isinstance(got, (int, float)):
                    if int(got) != int(want):
                        raise StatsError(
                            f"DINO config mismatch at the same checkpoint path: "
                            f"dino.{key} bundle={got} vs configured {want}"
                        )
                elif str(got) != str(want):
                    raise StatsError(
                        f"DINO config mismatch at the same checkpoint path: "
                        f"dino.{key} bundle={got!r} vs configured {want!r}"
                    )


def select_cameras(
    camera_order: Optional[Sequence[str]],
    camera_index: int,
    num_cameras: int,
    *,
    available: Optional[Sequence[str]] = None,
) -> Optional[List[str]]:
    """Resolve the camera selection ``camera_order[camera_index : +num_cameras]``.

    ``available`` (a reader's camera list) is used when ``camera_order`` is
    empty.  Returns ``None`` only when neither source can resolve the list.
    Out-of-range ``camera_index`` / counts raise :class:`StatsError` — the
    selection is never silently widened.
    """
    cams = [str(c) for c in (camera_order or available or [])]
    if not cams:
        return None  # cannot resolve without a camera list; callers skip the check
    idx, n = int(camera_index), int(num_cameras)
    if idx < 0 or n < 1:
        raise StatsError(f"invalid camera selection: camera_index={idx}, num_cameras={n}")
    if idx + n > len(cams):
        raise StatsError(
            f"camera selection out of range: camera_index={idx} + num_cameras={n} "
            f"exceeds the {len(cams)} available cameras {cams}"
        )
    return cams[idx: idx + n]


def bundle_fingerprint(payload: Dict[str, Any]) -> str:
    """Content fingerprint over the canonical JSON of the payload (minus itself)."""
    body = {k: v for k, v in payload.items() if k != "fingerprint"}
    blob = json.dumps(body, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def manifest_fingerprint(manifest: Dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(manifest, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


# ---------------------------------------------------------------------------
# fitting
# ---------------------------------------------------------------------------
class BundleFitter:
    """Accumulate action / future moments from training windows, then finalize.

    The fitter never sees val/test data by construction: the caller passes the
    training episode manifest and the CLI rejects anything but ``split=train``.
    """

    def __init__(
        self,
        *,
        action_dim: int,
        future_dim: Optional[int] = None,
        epsilon: float = DEFAULT_EPSILON,
        provenance: Optional[Dict[str, Any]] = None,
    ):
        self.epsilon = float(epsilon)
        self.action = StreamingMoments(int(action_dim))
        self.future = StreamingMoments(int(future_dim)) if future_dim is not None else None
        self.provenance: Dict[str, Any] = dict(provenance or {})
        self.action_updates = 0
        self.future_updates = 0

    # ------------------------------------------------------------------
    def update_action(self, a, mask=None) -> None:
        self.action.update(a, mask)
        self.action_updates += 1

    def update_future(self, z, mask=None) -> None:
        if self.future is None:
            raise StatsError("fitter was created without future statistics")
        self.future.update(z, mask)
        self.future_updates += 1

    # ------------------------------------------------------------------
    def finalize(self, *, coverage: Optional[Dict[str, Any]] = None) -> NormalizationBundle:
        prov = dict(self.provenance)
        if coverage is not None:
            prov["coverage"] = dict(coverage)
        prov.setdefault("coverage", {"kind": "unknown"})
        prov.setdefault("fitted_at", datetime.now(timezone.utc).isoformat())
        prov.setdefault("element_counting", {
            "action": (
                "valid elements of every training window (episode-safe plan, "
                "action_mask applied first, non-finite excluded); a raw frame in "
                "k valid windows contributes k times"
            ),
            "future": (
                "valid (F, V, P) future positions of every training window pooled "
                "per DINO channel; future_mask applied first, non-finite excluded"
            ),
        })
        return NormalizationBundle(
            profile=PROFILE_PAPER,
            epsilon=self.epsilon,
            action=self.action.finalize(self.epsilon),
            future=self.future.finalize(self.epsilon) if self.future is not None else None,
            provenance=prov,
        )


# ---------------------------------------------------------------------------
# (de)serialisation
# ---------------------------------------------------------------------------
def save_bundle(bundle: NormalizationBundle, path: Union[str, Path], *, overwrite: bool = False) -> Path:
    path = Path(path)
    if path.exists() and not overwrite:
        raise StatsError(
            f"refusing to overwrite existing statistics file {path}; pass "
            "overwrite=True / --overwrite explicitly to replace it"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bundle.to_payload(), indent=2, sort_keys=True))
    return path


def load_bundle(path: Union[str, Path]) -> NormalizationBundle:
    path = Path(path)
    if not path.exists():
        raise StatsError(
            f"normalization statistics bundle not found: {path}.  The paper "
            f"{PROFILE_PAPER!r} profile requires fitted training statistics "
            "(scripts/fit_normalization_stats.py); refusing to fall back to raw "
            "coordinates, random statistics or a val/test estimate."
        )
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise StatsError(f"statistics bundle {path} is not valid JSON: {exc}") from exc
    return NormalizationBundle.from_payload(payload)
